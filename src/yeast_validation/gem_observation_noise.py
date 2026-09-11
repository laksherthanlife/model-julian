#!/usr/bin/env python3
"""Observation-layer measurement noise for exact G3 trajectories.

The deterministic internal trajectory produced by ``gem_backend``/
``design_benchmark_exact`` is the audited ground truth and is never mutated
here. This module adds a seeded, reproducible *post-hoc* observation layer on
top of it: measurement noise on the externally-measurable readouts
(product, biomass, extracellular glucose) plus a small per-culture
multiplicative biological-variability factor, mirroring the lag+noise
reporter layer already implemented in
``generate_gem_state_space_dataset.build_reporters``.

Nothing here touches ``z_*`` hidden states, internal GEM constraint bounds,
or internal LP fluxes -- this is strictly an observation-layer concern, kept
separate from generator dynamics (``GEMCultureConfig``) on purpose.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

DETERMINISTIC_COLUMNS = ("B_total", "X", "S_glc")


def _culture_seed(seed: int, culture_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{culture_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def apply_observation_noise(
    traj_df: pd.DataFrame,
    seed: int,
    sigma_product: float = 0.03,
    sigma_biomass: float = 0.02,
    sigma_glc: float = 0.03,
    replicate_variability_sigma: float = 0.015,
    culture_id_col: str = "culture_id",
) -> pd.DataFrame:
    """Return a copy of ``traj_df`` with noisy ``*_observed`` columns added.

    For each column in :data:`DETERMINISTIC_COLUMNS` present in ``traj_df``:
      * the original deterministic values are preserved unchanged under
        ``{col}_true`` (in addition to keeping the original column name, so
        code that already reads ``B_total``/``X`` continues to see the exact
        deterministic ground truth by default);
      * a new ``{col}_observed`` column is added, equal to
        ``true_value * (1 + biological_variability) + N(0, sigma * scale)``,
        clipped at 0 (concentrations/masses cannot be negative), where
        ``scale`` is the per-culture max of the true trajectory (so sigma is
        a fraction of the trajectory's own dynamic range, not an absolute
        unit choice) and ``biological_variability`` is one multiplicative
        draw per culture (culture-to-culture variability), not one per time
        point.

    Reproducible: same ``seed`` + same ``traj_df`` -> identical noise.
    """
    sigmas = {"B_total": sigma_product, "X": sigma_biomass, "S_glc": sigma_glc}
    out = traj_df.copy()
    present = [c for c in DETERMINISTIC_COLUMNS if c in out.columns]
    for col in present:
        out[f"{col}_true"] = out[col]

    if culture_id_col not in out.columns:
        rng = np.random.default_rng(seed)
        for col in present:
            true_vals = out[f"{col}_true"].to_numpy(dtype=float)
            scale = max(float(np.max(np.abs(true_vals))), 1e-9)
            noise = rng.normal(0.0, sigmas[col] * scale, size=len(true_vals))
            out[f"{col}_observed"] = np.clip(true_vals + noise, 0.0, None)
        return out

    for culture_id, idx in out.groupby(culture_id_col).groups.items():
        rng = np.random.default_rng(_culture_seed(seed, str(culture_id)))
        bio_variability = rng.normal(0.0, replicate_variability_sigma)
        rows = out.loc[idx]
        for col in present:
            true_vals = rows[f"{col}_true"].to_numpy(dtype=float)
            scale = max(float(np.max(np.abs(true_vals))), 1e-9)
            noise = rng.normal(0.0, sigmas[col] * scale, size=len(true_vals))
            observed = np.clip(true_vals * (1.0 + bio_variability) + noise, 0.0, None)
            out.loc[idx, f"{col}_observed"] = observed
        out.loc[idx, "biological_variability_factor"] = bio_variability
    return out
