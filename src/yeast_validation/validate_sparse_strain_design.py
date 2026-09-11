#!/usr/bin/env python3
"""Validate a sparse GSM strain design JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import deployment_benchmark as deployment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    deployment.ensure_dirs()
    deployment.build_editable_reaction_universe()
    design = json.loads(args.json_path.read_text(encoding="utf-8"))
    print(json.dumps(deployment.validate_sparse_design(design), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
