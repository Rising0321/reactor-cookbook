#!/usr/bin/env python3
"""Download one immutable public Hugging Face snapshot and record completion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """Download the requested snapshot and write its exact completion marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--allow-pattern", action="append", default=[])
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern or None,
    )
    marker = args.local_dir / ".reactor-snapshot.json"
    pending = marker.with_suffix(".tmp")
    pending.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "allow": args.allow_pattern,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(pending, marker)


if __name__ == "__main__":
    main()
