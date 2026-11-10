from __future__ import annotations

import argparse
import json
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", default="anthropic.claude-opus-5-5")
    args = parser.parse_args(argv)
    from anthropic import AnthropicBedrockMantle

    start = time.perf_counter()
    client = AnthropicBedrockMantle(
        aws_region="us-east-1", aws_profile=args.profile, max_retries=0,
    )
    response = client.messages.create(
        model=args.model,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply OK"}],
    )
    value = response.model_dump(mode="json")
    print(json.dumps({
        "model": args.model,
        "status": "ok",
        "usage": value.get("usage"),
        "latency_s": round(time.perf_counter() - start, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
