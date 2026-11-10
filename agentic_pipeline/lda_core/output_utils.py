from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_stack_outputs(path: str | Path, stack_name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"CDK outputs must be a JSON object: {path}")
    nested = value.get(stack_name, value)
    if not isinstance(nested, dict):
        raise ValueError(f"outputs for {stack_name} must be a JSON object")
    return nested


def get_output(path: str | Path, stack_name: str, output_key: str) -> Any:
    outputs = read_stack_outputs(path, stack_name)
    if output_key not in outputs:
        raise KeyError(f"missing {stack_name}.{output_key} in {path}")
    return outputs[output_key]


def _text(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return "" if value is None else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read a value from nested CDK outputs.")
    parser.add_argument("path", type=Path)
    parser.add_argument("stack_name")
    parser.add_argument("keys", nargs="+")
    args = parser.parse_args(argv)
    values = read_stack_outputs(args.path, args.stack_name)
    missing = [key for key in args.keys if key not in values]
    if missing:
        raise SystemExit(f"missing output(s) for {args.stack_name}: {', '.join(missing)}")
    print("\t".join(_text(values[key]) for key in args.keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
