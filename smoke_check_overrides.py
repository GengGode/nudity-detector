"""Smoke-test request-scoped overrides on the patched /check endpoint.

Run this from the service repo's virtualenv after starting server.py:

    python smoke_check_overrides.py --base http://127.0.0.1:8000 --image sample.jpg

It checks:
  - the old one-field call still works;
  - resolution / threshold overrides are echoed in the response;
  - invalid values are rejected with 422.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def check(client: httpx.Client, image: Path, data: dict[str, str] | None = None) -> dict:
    with image.open("rb") as fh:
        response = client.post(
            "/check",
            files={"file": (image.name, fh.read(), "image/jpeg")},
            data=data or {},
        )
    if response.status_code != 200:
        raise SystemExit(f"/check failed ({response.status_code}): {response.text[:500]}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")

    with httpx.Client(base_url=args.base.rstrip("/"), timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("health:", json.dumps(health.json(), ensure_ascii=False))

        baseline = check(client, args.image)
        print("baseline:", baseline["verdict"], baseline.get("nsfw_score"), baseline.get("elapsed_ms"), "ms")

        overridden = check(
            client,
            args.image,
            {
                "resolution": "1280",
                "nsfw_block_threshold": "0.60",
                "nsfw_review_threshold": "0.25",
                "explicit_block_threshold": "0.45",
                "explicit_review_threshold": "0.18",
                "male_score_boost": "2.7",
            },
        )
        print(
            "overridden:",
            overridden["verdict"],
            overridden.get("inference_resolution"),
            overridden.get("policy"),
            overridden.get("elapsed_ms"),
            "ms",
        )

        assert overridden.get("inference_resolution") == 1280, "resolution was not applied"
        policy = overridden.get("policy") or {}
        assert abs(policy.get("nsfw_block_threshold", 0) - 0.60) < 1e-9, "threshold was not applied"
        assert abs(policy.get("male_score_boost", 0) - 2.7) < 1e-9, "male boost was not applied"

        bad = client.post(
            "/check",
            files={"file": (args.image.name, args.image.read_bytes(), "image/jpeg")},
            data={"resolution": "4096"},
        )
        assert bad.status_code == 422, f"invalid resolution should be 422, got {bad.status_code}"
        print("invalid resolution rejected:", bad.status_code)

    print("\nOK: request-scoped overrides are working.")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        sys.exit(1)
