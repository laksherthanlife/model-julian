#!/usr/bin/env python3
"""Phase 1 of the v2 powered rerun: for every campaign, run the hybrid arm's
6000-candidate virtual screen (new checkpoint + new surrogate, no LP) once
and PERSIST everything Phase 3 needs to disk -- the top-50 shortlist as
candidate JSONs (for Phase 2's exact-Yeast9 reranking array on Vanda) AND the
full 6000-row virtual ledger (for Phase 3's isolated-campaign audit ledger).

Phase 3 does NOT recompute this screen -- it only LOADS these files. This is
deliberate: an earlier version had Phase 3 regenerate the screen via the same
RNG seed and assumed bit-identical results, which failed in practice --
Phase 1 running locally (this machine) and Phase 3 running on Vanda's Linux
nodes produced tiny floating-point differences in the surrogate's forward
pass, enough to flip candidate ordering right at the rank-50 cutoff boundary
(12/50 shortlist candidates ended up with mismatched hashes on the first
sanity run). Persisting Phase 1's output as the single source of truth
removes any cross-machine determinism dependency.

Cheap (~10s/campaign, pure numpy) -- run locally, not on Vanda.

Idempotent/resumable: skips a campaign if its manifest file already exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HPC_DIR = ROOT / "hpc_bundles" / "repaired_prospective_v2"
CANDIDATES_DIR = HPC_DIR / "candidates"
SHORTLISTS_DIR = HPC_DIR / "shortlists"
VIRTUAL_LEDGERS_DIR = HPC_DIR / "virtual_ledgers"


def main() -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    SHORTLISTS_DIR.mkdir(parents=True, exist_ok=True)
    VIRTUAL_LEDGERS_DIR.mkdir(parents=True, exist_ok=True)

    # Must match Phase 3's patched BENCHMARK_ID exactly -- candidate_hash
    # includes candidate["metadata"]["benchmark_id"] (set by
    # CandidateFactory.make() from the bench module's global at call time),
    # so Phase 1 and Phase 3 must agree on this value or shortlist hashes
    # would silently diverge between phases.
    bench.BENCHMARK_ID = v2.REPAIRED_BENCHMARK_ID_V2

    library = v2.build_common_intervention_library()
    scorer = v2.load_scorer_v2()

    for world_id, seed in v2.all_campaigns():
        campaign_id = v2.campaign_id_for(world_id, seed)
        manifest_path = SHORTLISTS_DIR / f"{campaign_id}_shortlist_manifest.csv"
        virtual_ledger_path = VIRTUAL_LEDGERS_DIR / f"{campaign_id}_virtual.csv"
        if manifest_path.exists() and virtual_ledger_path.exists():
            print(f"already done: {campaign_id}")
            continue

        factory = v2.make_factory(library, campaign_id, seed)
        shortlist, virtual = v2.generate_hybrid_shortlist(
            factory, v2.VIRTUAL_EVALUATIONS, v2.SHORTLIST_SIZE, set(), library, scorer, campaign_id, world_id
        )

        rows = []
        for cand in shortlist:
            cand_path = CANDIDATES_DIR / f"{cand['candidate_id']}.json"
            cand_path.write_text(json.dumps(cand), encoding="utf-8")
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "world_id": world_id,
                    "seed": seed,
                    "candidate_id": cand["candidate_id"],
                    "candidate_hash": cand["candidate_hash"],
                    "surrogate_score": cand["_surrogate_score"],
                }
            )
        import pandas as pd

        manifest = pd.DataFrame(rows).sort_values("surrogate_score", ascending=False).reset_index(drop=True)
        manifest["surrogate_rank"] = manifest.index + 1
        manifest.to_csv(manifest_path, index=False)
        virtual.to_csv(virtual_ledger_path, index=False)
        print(f"{campaign_id}: shortlist={len(shortlist)}, clipped_fraction={(virtual['n_control_values_clipped']>0).mean():.3f}")

    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
