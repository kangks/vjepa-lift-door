"""lda per-invocation cost report.

  python3 -I workshop/lift-door/lda_cost_joiner/cli.py collect [--cost-explorer] [--no-attribute]
  python3 -I workshop/lift-door/lda_cost_joiner/cli.py report
  python3 -I workshop/lift-door/lda_cost_joiner/cli.py all --cost-explorer

collect: read-only AWS + Attribute calls, writes out/raw/*.json (account-filtered, no secrets).
report:  offline; reads out/raw and writes out/per_invocation.{csv,md}, out/reconciliation.md,
         out/SUMMARY.md, out/rates_used.md.
Use the venv that has boto3, e.g.
  <LOCAL_PATH>/Documents/workplace/sydney_physical_ai_october2026/workshop/lift-lobby-water-leak/.venv/bin/python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # python -I does not add the script dir

DEFAULT_RUNS_DIR = Path("<LOCAL_PATH>/.claude/jobs/0964db1f/tmp/lda_cost/runs")
DEFAULT_DAYS = ["2026-11-07", "2026-11-08", "2026-11-09"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["collect", "report", "all"])
    p.add_argument("--out", type=Path, default=HERE / "out")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--profile", default="attrb_sandbox")
    p.add_argument("--days", nargs="+", default=DEFAULT_DAYS)
    p.add_argument("--cost-explorer", action="store_true",
                   help="also query ce:GetCostAndUsage (2 requests, ~US$0.02) as a billing cross-check")
    p.add_argument("--no-attribute", action="store_true")
    a = p.parse_args(argv)
    if a.command in ("collect", "all"):
        from lda_cost_joiner.collect import collect_all

        collect_all(a.out, a.runs_dir, a.profile, a.days, cost_explorer=a.cost_explorer,
                    attribute=not a.no_attribute)
        print(f"collected -> {a.out / 'raw'}")
    if a.command in ("report", "all"):
        from lda_cost_joiner.report import build_report

        for path in build_report(a.out):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
