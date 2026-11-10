from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.trace.read_text())
    if isinstance(source, dict) and "Traces" in source:
        segments = [segment for trace in source["Traces"] for segment in trace.get("Segments", [])]
        spans = [json.loads(segment["Document"]) for segment in segments]
    else:
        spans = source
    rows = []
    for span in spans:
        start = float(span.get("startTime", span.get("start_time", 0)))
        end = float(span.get("endTime", span.get("end_time", start)))
        name = span.get("name", "(unnamed)")
        service = span.get("service", {}).get("name") or span.get("resource", {}).get("attributes", {}).get("service.name", "")
        rows.append((service, name, start, max(0.001, end - start)))
    if not rows:
        raise ValueError("trace has no spans")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install matplotlib to render trajectory diagrams") from exc
    services = list(dict.fromkeys(row[0] for row in rows))
    y_by_service = {service: index for index, service in enumerate(services)}
    origin = min(row[2] for row in rows)
    fig, ax = plt.subplots(figsize=(12, max(3, len(services) * 0.7)))
    for service, name, start, duration in rows:
        y = y_by_service[service]
        ax.barh(y, duration, left=start - origin, height=0.55)
        ax.text(start - origin, y, " " + name, va="center", fontsize=8)
    ax.set_yticks(range(len(services)), services)
    ax.set_xlabel("seconds from trace start")
    ax.set_title("Lift-door end-to-end trace trajectory")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
