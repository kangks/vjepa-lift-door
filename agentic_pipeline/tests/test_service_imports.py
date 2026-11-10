from __future__ import annotations

import ast
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_IMPORTS = {"numpy", "PIL", "torch", "jsonschema"}
PACKAGE_ALIASES = {
    "PIL": "pillow",
    "yaml": "pyyaml",
    "amazon": "aws-opentelemetry-distro",
    "opentelemetry": "aws-opentelemetry-distro",
}
LOCAL_IMPORTS = {"lda_core", "services", "vendor", "app"}
STANDARD_IMPORTS = set(getattr(sys, "stdlib_module_names", ()))


def _module_path(module: str) -> Path:
    return ROOT / Path(*module.split(".")).with_suffix(".py")


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0], alias.name
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            yield node.module.split(".", 1)[0], node.module


def _service_external_imports(service: str) -> set[str]:
    service_dir = ROOT / "services" / service
    queued = [path for path in service_dir.glob("*.py")]
    core_modules: set[str] = set()
    external: set[str] = set()
    visited: set[Path] = set()

    while queued:
        path = queued.pop()
        path = path.resolve()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        if path.is_relative_to(ROOT / "lda_core"):
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[:-len(".__init__")]
            current_package = module.rsplit(".", 1)[0] if path.name != "__init__.py" else module
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        top = imported.split(".", 1)[0]
                        if top in LOCAL_IMPORTS:
                            if top == "lda_core":
                                core_modules.add(imported)
                        elif top not in STANDARD_IMPORTS and top != "__future__":
                            external.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package_parts = current_package.split(".")
                        base_parts = package_parts[:max(1, len(package_parts) - node.level + 1)]
                        base = ".".join(base_parts)
                        if node.module:
                            target = f"{base}.{node.module}"
                            core_modules.add(target)
                        else:
                            for alias in node.names:
                                core_modules.add(f"{base}.{alias.name}")
                    elif node.module:
                        imported = node.module
                        top = imported.split(".", 1)[0]
                        if top in LOCAL_IMPORTS:
                            if top == "lda_core":
                                core_modules.add(imported)
                        elif top not in STANDARD_IMPORTS and top != "__future__":
                            external.add(top)
        else:
            for top, imported in _imports(path):
                if top in LOCAL_IMPORTS:
                    if top == "lda_core":
                        core_modules.add(imported)
                elif top not in STANDARD_IMPORTS and top != "__future__":
                    external.add(top)

        for module in list(core_modules):
            module_path = _module_path(module)
            package_init = ROOT / Path(*module.split(".")) / "__init__.py"
            if module_path.exists() and module_path not in visited:
                queued.append(module_path)
            elif package_init.exists() and package_init not in visited:
                queued.append(package_init)
    return external


def _dockerfile_packages(service: str) -> set[str]:
    text = (ROOT / "services" / service / "Dockerfile").read_text().replace("\\\n", " ")
    commands = re.findall(r"\bpip\s+install\b(.*?)(?=\s&&\s|\n[A-Z]|$)", text, flags=re.DOTALL)
    packages = set()
    for command in commands:
        tokens = shlex.split(command)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("-"):
                if token in {"--index-url", "--extra-index-url", "--find-links"}:
                    index += 2
                else:
                    index += 1
                continue
            if "://" in token:
                index += 1
                continue
            package = re.split(r"[<>=!~]", token, maxsplit=1)[0]
            if package:
                packages.add(package.lower().replace("_", "-"))
            index += 1
    return packages


def test_ticket_and_reaper_import_without_large_or_optional_packages():
    blocker = f"""
import importlib.abc, sys
blocked = {sorted(BLOCKED_IMPORTS)!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise ImportError('blocked in ticket image: ' + fullname)
sys.meta_path.insert(0, Blocker())
"""
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    for module in ("services.ticket.handler", "services.reaper.handler"):
        result = subprocess.run(
            [sys.executable, "-c", blocker + f"\nimport {module}\n"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"


def test_embed_knn_and_coordinator_top_level_imports_are_installed():
    for service in ("embed", "knn", "coordinator"):
        required = {
            PACKAGE_ALIASES.get(name, name).lower().replace("_", "-")
            for name in _service_external_imports(service)
        }
        installed = _dockerfile_packages(service)
        missing = sorted(required - installed)
        assert not missing, (
            f"{service} imports {sorted(required)}, missing from Dockerfile pip install: {missing}"
        )
