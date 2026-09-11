#!/usr/bin/env python3
"""Prepare the public bundle for the first isolated scientist task."""

from __future__ import annotations

import pandas as pd

import deployment_benchmark as deployment


def main() -> None:
    deployment.ensure_dirs()
    deployment.write_benchmark_classification()
    deployment.build_editable_reaction_universe()
    deployment.write_objective_and_scenarios()
    deployment.write_protocols()
    deployment.prepare_public_sandbox()
    deployment.write_sparse_edit_safeguards()
    deployment.write_reaction_universe_audit()
    deployment.export_public_bundle()

    deployment.write_verifier_interface_smoke_audit()
    audit = deployment.refresh_public_bundle_audit()
    print(audit.to_string(index=False))
    print(pd.read_csv(deployment.DATA / "deployment_benchmark_static_solve_accounting.csv").to_string(index=False))


if __name__ == "__main__":
    main()
