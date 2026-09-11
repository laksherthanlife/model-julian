#!/usr/bin/env python3
"""Regulation-only complexity screen for the published yeast rxncon model.

This script intentionally does not call Yeast9, COBRApy, pFBA, ML training, or
DBTL code.  It compiles the published S. cerevisiae rxncon cell-cycle model to a
Boolean network, generates a perturbational panel using the model's explicit
Boolean input and KO/OE hooks, and applies the same matched 125 x 49 empirical
complexity protocol used for the current synthetic system audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
RXNCON_DIR = ROOT / "external_models/rxncon_screening/rxncon"
MODELS_DIR = ROOT / "external_models/rxncon_screening/models"
MODEL_XLS = MODELS_DIR / "CDC_S_cerevisiae.xls"
OUT_DIR = ROOT / "data/rxncon_external_complexity_screen"
FIG_DIR = ROOT / "figures"
EPS = 1e-12
RNG_SEED = 20260822
N_MATCHED = 125
N_TIMEPOINTS = 49
MAX_HANKEL_WINDOWS = 500

EXTERNAL_INPUTS = {
    "[Nutrients]": "nutrient availability",
    "[Pheromone]": "mating pheromone / alpha-factor signal",
    "[HU]": "hydroxyurea DNA replication stress",
    "[LatA]": "latrunculin A actin/cytoskeleton perturbation",
    "[Nocodazole]": "microtubule/spindle perturbation",
}

OBSERVABLE_STATES = [
    "[CD]",
    "[ND]",
    "[SEP]",
    "[SEG]",
    "[Cytokinesis]",
    "[DNALicensed]",
    "[DNAReplicated]",
    "[OriginFiring]",
    "[BipolarSpindle]",
    "[StableTension]",
    "[SpindlePositioning]",
    "[ApicalGrowth]",
    "[BudSite]",
    "[Bridge]",
    "[Satellite]",
    "[SeptinPol]",
    "[CriticalError]",
    "[CDerror]",
    "[SEGerror]",
    "[dNTP]",
    "[ssDNA]",
    "[ssdsDNAjunctions]",
]

MAJOR_COMPONENT_HINTS = [
    "Cdc28",
    "Cln1",
    "Cln2",
    "Cln3",
    "Clb1",
    "Clb2",
    "Clb3",
    "Clb4",
    "Clb5",
    "Clb6",
    "Sic1",
    "Cdc20",
    "Cdh1",
    "Cdc5",
    "Cdc14",
    "Pds1",
    "Esp1",
    "APC",
    "Mcm1",
    "Swi4",
    "Swi6",
    "Mbp1",
    "Ndd1",
    "Fkh1",
    "Fkh2",
    "Whi5",
    "SBF",
    "MBF",
    "Cdc6",
    "Orc",
    "Mcm2",
    "Mcm3",
    "Mcm4",
    "Mcm5",
    "Mcm6",
    "Mcm7",
    "Mad2",
    "Bub1",
    "Bub3",
    "Swe1",
    "Mih1",
]


def standardize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    fill = float(np.nanmedian(x[finite])) if finite.any() else 0.0
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (x - mu) / sd


def remove_constant_columns(x: np.ndarray, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x)
    if x.ndim != 2:
        x = x.reshape((x.shape[0], -1))
    sd = np.nanstd(x.astype(float), axis=0)
    keep = sd > 1e-10
    if names is None:
        names = [f"f{i}" for i in range(x.shape[1])]
    return x[:, keep], [n for n, k in zip(names, keep) if k]


def standardized_gram(x: np.ndarray | sparse.spmatrix, standardize: bool = True) -> np.ndarray:
    if sparse.issparse(x):
        x = x.tocsr().astype(float)
    else:
        x = sparse.csr_matrix(np.asarray(x, dtype=float))
    n, d = x.shape
    if n == 0 or d == 0:
        return np.empty((n, n))
    mean = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    var = second - mean**2
    sd = np.sqrt(np.maximum(var * n / max(n - 1, 1), EPS))
    if not standardize:
        sd = np.ones_like(sd)
    inv_var = 1.0 / (sd**2)
    weighted = x.multiply(np.sqrt(inv_var))
    gram = (weighted @ weighted.T).toarray()
    c = mean * inv_var
    v = np.asarray(x @ c).ravel()
    gram -= v[:, None]
    gram -= v[None, :]
    gram += float(np.sum((mean**2) * inv_var))
    return gram


def eigenvalues_from_samples(x: np.ndarray | sparse.spmatrix, standardize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x, _ = remove_constant_columns(x)
    if x.shape[0] < 2 or x.shape[1] == 0:
        return np.array([])
    if x.shape[1] > 5000:
        gram = standardized_gram(x, standardize=standardize) / max(x.shape[0] - 1, 1)
    else:
        x = standardize_matrix(x) if standardize else x - x.mean(axis=0, keepdims=True)
        gram = (x @ x.T) / max(x.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(gram)
    eig = np.sort(eig[eig > EPS])[::-1]
    return eig


def dimension_metrics(eig: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eig, dtype=float)
    eig = eig[eig > EPS]
    if eig.size == 0:
        return {
            "n_components_nonzero": 0,
            "entropy_rank": np.nan,
            "participation_rank": np.nan,
            "pc90": np.nan,
            "pc95": np.nan,
            "pc99": np.nan,
            "pc999": np.nan,
        }
    p = eig / eig.sum()
    c = np.cumsum(p)
    return {
        "n_components_nonzero": int(eig.size),
        "entropy_rank": float(np.exp(-(p * np.log(p + EPS)).sum())),
        "participation_rank": float((eig.sum() ** 2) / np.sum(eig**2)),
        "pc90": int(np.searchsorted(c, 0.90) + 1),
        "pc95": int(np.searchsorted(c, 0.95) + 1),
        "pc99": int(np.searchsorted(c, 0.99) + 1),
        "pc999": int(np.searchsorted(c, 0.999) + 1),
    }


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x, _ = remove_constant_columns(np.asarray(x))
    if x.shape[0] > 1:
        x = np.unique(x, axis=0)
    if x.shape[0] < 3 or x.shape[1] == 0:
        return np.empty((x.shape[0], x.shape[0]))
    if x.shape[1] > 5000:
        gram = standardized_gram(x, standardize=True)
        diag = np.diag(gram)
        d2 = np.maximum(diag[:, None] + diag[None, :] - 2 * gram, 0.0)
    else:
        x = standardize_matrix(np.asarray(x, dtype=float))
        sq = np.sum(x * x, axis=1, keepdims=True)
        d2 = np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    return d


def twonn_id(x: np.ndarray) -> float:
    d = pairwise_distances(x)
    if d.size == 0:
        return np.nan
    nearest = np.sort(d, axis=1)[:, :2]
    good = (nearest[:, 0] > EPS) & np.isfinite(nearest[:, 1])
    mu = nearest[good, 1] / nearest[good, 0]
    if mu.size < 5:
        return np.nan
    return float(1.0 / np.mean(np.log(mu + EPS)))


def levina_bickel_id(x: np.ndarray, ks: Iterable[int] = (5, 8, 12)) -> float:
    d = pairwise_distances(x)
    if d.size == 0:
        return np.nan
    vals = []
    sorted_d = np.sort(d, axis=1)
    for k in ks:
        if sorted_d.shape[1] <= k:
            continue
        r = sorted_d[:, : k + 1]
        rk = r[:, k]
        good = (r[:, 0] > EPS) & (rk > EPS) & np.isfinite(rk)
        if good.sum() < 5:
            continue
        logs = np.log((rk[good, None] + EPS) / (r[good, :k] + EPS))
        inv = np.mean(logs, axis=1)
        local = 1.0 / np.maximum(inv, EPS)
        vals.append(np.median(local[np.isfinite(local)]))
    return float(np.mean(vals)) if vals else np.nan


def summarize_layer(layer: str, x: np.ndarray, compute_nonlinear: bool = True) -> tuple[dict[str, float], pd.DataFrame]:
    x, _ = remove_constant_columns(np.asarray(x))
    eig = eigenvalues_from_samples(x)
    nonlinear_ok = compute_nonlinear and x.shape[1] <= 2000 and x.shape[0] <= 2500
    row = {
        "layer": layer,
        "n_samples": int(x.shape[0]),
        "n_features_varying": int(x.shape[1]),
        **dimension_metrics(eig),
        "twonn_id": twonn_id(x) if nonlinear_ok else np.nan,
        "levina_bickel_id": levina_bickel_id(x) if nonlinear_ok else np.nan,
        "nonlinear_id_status": "computed" if nonlinear_ok else "skipped_high_dimensional_boolean_or_bootstrap",
    }
    p = eig / eig.sum() if eig.size else np.array([])
    spectrum = pd.DataFrame(
        {
            "layer": layer,
            "pc": np.arange(1, len(eig) + 1),
            "eigenvalue": eig,
            "variance_explained": p,
            "cumulative_variance": np.cumsum(p),
        }
    )
    return row, spectrum


def delay_embed(traj: np.ndarray, lag: int = 8, max_windows: int | None = None, seed: int = RNG_SEED) -> np.ndarray:
    # traj: n_trajectories x n_timepoints x n_channels
    windows = []
    for i in range(traj.shape[0]):
        for start in range(traj.shape[1] - lag + 1):
            windows.append(traj[i, start : start + lag, :].reshape(-1))
    if max_windows is not None and len(windows) > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(windows)), size=max_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]
    return np.asarray(windows, dtype=float)


def venn_solutions_compat_patch() -> None:
    """Patch rxncon's pyeda-based satisfying-assignment enumeration.

    The published compiler path still builds the same Boolean rules, but pyeda's
    old C extension segfaults on current macOS/Python while enumerating
    satisfiers.  The replacement returns DNF partial assignments for the same
    rxncon Venn-set objects and avoids brute-force truth-table enumeration.
    """
    if str(RXNCON_DIR) not in sys.path:
        sys.path.insert(0, str(RXNCON_DIR))

    from rxncon.core import rxncon_system as rxncon_system_mod
    from rxncon.venntastic import sets as sets_mod

    def merge_cubes(a: dict, b: dict) -> dict | None:
        merged = dict(a)
        for key, val in b.items():
            if key in merged and merged[key] != val:
                return None
            merged[key] = val
        return merged

    def cube_key(cube: dict) -> tuple:
        return tuple(sorted((str(k), bool(v)) for k, v in cube.items()))

    def dedup(cubes: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for cube in cubes:
            key = cube_key(cube)
            if key not in seen:
                seen.add(key)
                out.append(cube)
        return out

    def solve(expr, polarity: bool = True) -> list[dict]:
        ValueSet = sets_mod.ValueSet
        Complement = sets_mod.Complement
        Intersection = sets_mod.Intersection
        Union = sets_mod.Union
        DisjunctiveUnion = sets_mod.DisjunctiveUnion
        UniversalSet = sets_mod.UniversalSet
        EmptySet = sets_mod.EmptySet

        if isinstance(expr, UniversalSet):
            return [{}] if polarity else []
        if isinstance(expr, EmptySet):
            return [] if polarity else [{}]
        if isinstance(expr, ValueSet):
            return [{expr.value: polarity}]
        if isinstance(expr, Complement):
            return solve(expr.expr, not polarity)
        if isinstance(expr, Intersection):
            if not polarity:
                cubes = []
                for sub in expr.exprs:
                    cubes.extend(solve(sub, False))
                return dedup(cubes)
            cubes = [{}]
            for sub in expr.exprs:
                sub_cubes = solve(sub, True)
                next_cubes = []
                for a in cubes:
                    for b in sub_cubes:
                        merged = merge_cubes(a, b)
                        if merged is not None:
                            next_cubes.append(merged)
                cubes = dedup(next_cubes)
                if not cubes:
                    break
            return cubes
        if isinstance(expr, (Union, DisjunctiveUnion)):
            if polarity:
                cubes = []
                for sub in expr.exprs:
                    cubes.extend(solve(sub, True))
                return dedup(cubes)
            cubes = [{}]
            for sub in expr.exprs:
                sub_cubes = solve(sub, False)
                next_cubes = []
                for a in cubes:
                    for b in sub_cubes:
                        merged = merge_cubes(a, b)
                        if merged is not None:
                            next_cubes.append(merged)
                cubes = dedup(next_cubes)
                if not cubes:
                    break
            return cubes
        raise AssertionError(f"Unsupported rxncon Venn expression: {type(expr)}")

    def calc_solutions(self):
        return solve(self, True)

    sets_mod.Set.calc_solutions = calc_solutions
    rxncon_system_mod.RxnConSystem.validate = lambda self: None


def compile_model(prefix: Path, perturbable: bool, force: bool = False) -> dict[str, float]:
    boolnet = prefix.with_suffix(".boolnet")
    symbols = prefix.parent / f"{prefix.name}_symbols.csv"
    initial = prefix.parent / f"{prefix.name}_initial_vals.csv"
    if boolnet.exists() and symbols.exists() and initial.exists() and not force:
        return {"compiled": 0.0, "compile_seconds": 0.0}

    venn_solutions_compat_patch()
    from rxncon.input.excel_book.excel_book import ExcelBook
    from rxncon.simulation.boolean.boolean_model import (
        KnockoutStrategy,
        OverexpressionStrategy,
        SmoothingStrategy,
    )
    from rxncon.simulation.boolean.boolnet_from_boolean_model import (
        QuantitativeContingencyStrategy,
        boolnet_strs_from_rxncon,
    )

    t0 = time.perf_counter()
    rxncon_sys = ExcelBook(str(MODEL_XLS)).rxncon_system
    k_strategy = KnockoutStrategy.knockout_all_states if perturbable else KnockoutStrategy.no_knockout
    oe_strategy = (
        OverexpressionStrategy.overexpress_neutral_states
        if perturbable
        else OverexpressionStrategy.no_overexpression
    )
    model_str, symbol_str, initial_str = boolnet_strs_from_rxncon(
        rxncon_sys,
        smoothing_strategy=SmoothingStrategy.smooth_production_sources,
        knockout_strategy=k_strategy,
        overexpression_strategy=oe_strategy,
        k_plus_strategy=QuantitativeContingencyStrategy.strict,
        k_minus_strategy=QuantitativeContingencyStrategy.strict,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    boolnet.write_text(model_str)
    symbols.write_text(symbol_str)
    initial.write_text(initial_str)
    return {"compiled": 1.0, "compile_seconds": time.perf_counter() - t0}


def model_inventory() -> dict[str, object]:
    if str(RXNCON_DIR) not in sys.path:
        sys.path.insert(0, str(RXNCON_DIR))
    venn_solutions_compat_patch()
    import xlrd
    from rxncon.input.excel_book.excel_book import ExcelBook

    rxncon_sys = ExcelBook(str(MODEL_XLS)).rxncon_system
    book = xlrd.open_workbook(str(MODEL_XLS))
    sheets = {sheet.name: {"n_rows": sheet.nrows, "n_cols": sheet.ncols} for sheet in book.sheets()}
    git_model = subprocess.check_output(["git", "-C", str(MODELS_DIR), "rev-parse", "HEAD"], text=True).strip()
    git_rxncon = subprocess.check_output(["git", "-C", str(RXNCON_DIR), "rev-parse", "HEAD"], text=True).strip()
    return {
        "publication": "Muenzner, Klipp, Krantz, Nat Commun 2019, DOI 10.1038/s41467-019-08903-w",
        "model_file": str(MODEL_XLS.relative_to(ROOT)),
        "models_repo_commit": git_model,
        "rxncon_repo_commit": git_rxncon,
        "reaction_list_rows": sheets.get("ReactionList", {}).get("n_rows"),
        "contingency_list_rows": sheets.get("ContingencyList", {}).get("n_rows"),
        "rxncon_reactions_loaded": len(rxncon_sys.reactions),
        "rxncon_contingencies_loaded": len(rxncon_sys.contingencies),
        "rxncon_states_loaded": len(rxncon_sys.states),
        "rxncon_components_loaded": len(rxncon_sys.components()),
        "sheets": sheets,
        "compatibility_deviation": (
            "Original rxncon2boolnet.py segfaulted in pyeda satisfy_all on this Python/macOS stack; "
            "this script replaces only Venn-set satisfier enumeration with DNF partial assignments and "
            "skips the pyeda validation pass. Boolean model construction, smoothing, KO/OE strategies, "
            "and BoolNet serialization remain the published rxncon code paths."
        ),
    }


def read_symbols(prefix: Path) -> pd.DataFrame:
    symbols = prefix.parent / f"{prefix.name}_symbols.csv"
    rows = []
    with symbols.open() as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            rows.append({"boolnet_id": row[0].strip(), "rxncon_name": row[1].strip()})
    df = pd.DataFrame(rows)
    df["kind"] = df["boolnet_id"].str.extract(r"^([A-Z]+)")
    df["ordinal"] = df["boolnet_id"].str.extract(r"(\d+)").astype(int)
    return df.sort_values(["kind", "ordinal"]).reset_index(drop=True)


def read_initial(prefix: Path, symbol_order: list[str]) -> np.ndarray:
    path = prefix.parent / f"{prefix.name}_initial_vals.csv"
    vals = {}
    with path.open() as fh:
        for line in fh:
            raw = line.split("#", 1)[0].strip()
            if not raw:
                continue
            name, val = [part.strip() for part in raw.split(",", 1)]
            vals[name] = val.lower().startswith("true") or val == "1"
    return np.asarray([vals[name] for name in symbol_order], dtype=bool)


@dataclass
class BoolNet:
    symbols: pd.DataFrame
    symbol_order: list[str]
    index: dict[str, int]
    names: dict[str, str]
    initial: np.ndarray
    rule_targets: list[int]
    rule_codes: list[object]


def load_boolnet(prefix: Path) -> BoolNet:
    symbols = read_symbols(prefix)
    symbol_order = symbols["boolnet_id"].tolist()
    index = {name: i for i, name in enumerate(symbol_order)}
    names = dict(zip(symbols["boolnet_id"], symbols["rxncon_name"]))
    initial = read_initial(prefix, symbol_order)
    rule_targets = []
    rule_codes = []

    token_re = re.compile(r"\b[RSKO]\d+\b")

    def translate(expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("!", "~")
        return token_re.sub(lambda m: f"s[:, {index[m.group(0)]}]", expr)

    with prefix.with_suffix(".boolnet").open() as fh:
        header = next(fh)
        if "targets" not in header:
            raise ValueError("Unexpected BoolNet header")
        for line in fh:
            if not line.strip():
                continue
            target, expr = line.split(",", 1)
            target = target.strip()
            rule_targets.append(index[target])
            rule_codes.append(compile(translate(expr), f"<rxncon_rule_{target}>", "eval"))
    return BoolNet(symbols, symbol_order, index, names, initial, rule_targets, rule_codes)


def build_panel(model: BoolNet, n_panel: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    sym_to_name = model.names
    name_to_sym = {v: k for k, v in sym_to_name.items()}
    input_syms = [name_to_sym[x] for x in EXTERNAL_INPUTS if x in name_to_sym]
    k_syms = sorted([s for s, n in sym_to_name.items() if s.startswith("K") and n.startswith("Knockout<")])
    o_syms = sorted([s for s, n in sym_to_name.items() if s.startswith("O") and n.startswith("Overexpression<")])

    def component(sym: str) -> str:
        name = sym_to_name[sym]
        return name.split("<", 1)[1].rstrip(">")

    major_k = [s for s in k_syms if component(s) in MAJOR_COMPONENT_HINTS]
    major_o = [s for s in o_syms if component(s) in MAJOR_COMPONENT_HINTS]
    if not major_k:
        major_k = k_syms[: min(64, len(k_syms))]
    if not major_o:
        major_o = o_syms[: min(64, len(o_syms))]

    designs = []

    def add_design(kind: str, env: dict[str, int], forced: dict[str, int], interp: str, support: str) -> None:
        key = (
            tuple(sorted(env.items())),
            tuple(sorted(forced.items())),
            kind,
        )
        if key in seen:
            return
        seen.add(key)
        designs.append(
            {
                "trajectory_id": f"rxncon_{len(designs):04d}",
                "perturbation_type": kind,
                "environment": env,
                "forced": forced,
                "biological_interpretation": interp,
                "support": support,
                "single_or_combinatorial": "single"
                if sum(v != int(model.initial[model.index[k]]) for k, v in forced.items()) <= 1
                else "combinatorial",
            }
        )

    seen = set()
    baseline_env = {sym: int(model.initial[model.index[sym]]) for sym in input_syms}
    if "[Nutrients]" in name_to_sym:
        baseline_env[name_to_sym["[Nutrients]"]] = 1
    for label in ["[Pheromone]", "[HU]", "[LatA]", "[Nocodazole]"]:
        if label in name_to_sym:
            baseline_env[name_to_sym[label]] = 0
    add_design("baseline", baseline_env, {}, "wild-type baseline input state", "direct model input")

    # Distinct constant and pulsed environmental programs.
    n_inputs = len(input_syms)
    for mask in range(2**n_inputs):
        env = {sym: (mask >> i) & 1 for i, sym in enumerate(input_syms)}
        if "[Nutrients]" in name_to_sym:
            env[name_to_sym["[Nutrients]"]] = env.get(name_to_sym["[Nutrients]"], 1)
        add_design("environment_constant", env, {}, "constant external input combination", "direct model input")
    for sym in input_syms:
        env_on = dict(baseline_env)
        env_on[sym] = 1
        add_design("environment_late_on", env_on, {}, f"late-on pulse for {sym_to_name[sym]}", "direct model input schedule")
        env_off = dict(baseline_env)
        env_off[sym] = 0
        add_design("environment_late_off", env_off, {}, f"late-off pulse for {sym_to_name[sym]}", "direct model input schedule")

    for sym in k_syms:
        add_design("single_knockout", dict(baseline_env), {sym: 1}, f"component knockout {component(sym)}", "rxncon compiler KO target")
    for sym in o_syms:
        add_design(
            "single_overexpression",
            dict(baseline_env),
            {sym: 1},
            f"component overexpression {component(sym)}",
            "rxncon compiler OE target",
        )

    env_variants = []
    for _ in range(max(80, n_panel // 6)):
        env = dict(baseline_env)
        for sym in input_syms:
            if rng.random() < 0.35:
                env[sym] = int(not env.get(sym, 0))
        env_variants.append(env)
    genetic_pool = major_k + major_o + rng.choice(k_syms + o_syms, size=min(150, len(k_syms + o_syms)), replace=False).tolist()
    for sym in genetic_pool:
        for env in rng.choice(len(env_variants), size=min(3, len(env_variants)), replace=False):
            add_design(
                "environment_plus_genetic",
                dict(env_variants[int(env)]),
                {sym: 1},
                f"external input program plus {sym_to_name[sym]}",
                "direct input plus rxncon compiler perturbation target",
            )

    for _ in range(max(100, n_panel // 5)):
        choices = rng.choice(genetic_pool, size=2, replace=False)
        add_design(
            "double_genetic",
            dict(baseline_env),
            {str(choices[0]): 1, str(choices[1]): 1},
            f"paired compiler-supported perturbation: {sym_to_name[str(choices[0])]} + {sym_to_name[str(choices[1])]}",
            "rxncon compiler perturbation target combination",
        )

    if len(designs) > n_panel:
        keep = [0]
        remaining = np.arange(1, len(designs))
        keep.extend(rng.choice(remaining, size=n_panel - 1, replace=False).tolist())
        designs = [designs[i] for i in sorted(keep)]
    elif len(designs) < n_panel:
        for _ in range(n_panel - len(designs)):
            env = dict(baseline_env)
            for sym in input_syms:
                if rng.random() < 0.5:
                    env[sym] = int(not env.get(sym, 0))
            forced_syms = rng.choice(genetic_pool, size=int(rng.integers(1, 4)), replace=False)
            add_design(
                "sampled_combinatorial",
                env,
                {str(sym): 1 for sym in forced_syms},
                "space-filling sampled external/genetic combination",
                "direct input plus rxncon compiler perturbation target combination",
            )

    panel = pd.DataFrame(designs).iloc[:n_panel].copy()
    force = np.full((N_TIMEPOINTS, len(panel), len(model.symbol_order)), -1, dtype=np.int8)
    for i, row in panel.iterrows():
        env = row["environment"]
        forced = row["forced"]
        for sym, val in env.items():
            idx = model.index[sym]
            if row["perturbation_type"] == "environment_late_on":
                force[: N_TIMEPOINTS // 3, i, idx] = int(model.initial[idx])
                force[N_TIMEPOINTS // 3 :, i, idx] = val
            elif row["perturbation_type"] == "environment_late_off":
                force[: N_TIMEPOINTS // 3, i, idx] = val
                force[N_TIMEPOINTS // 3 :, i, idx] = int(model.initial[idx])
            else:
                force[:, i, idx] = val
        for sym, val in forced.items():
            force[:, i, model.index[sym]] = val

    panel["environment"] = panel["environment"].apply(lambda d: json.dumps({sym_to_name[k]: v for k, v in d.items()}, sort_keys=True))
    panel["forced"] = panel["forced"].apply(lambda d: json.dumps({sym_to_name[k]: v for k, v in d.items()}, sort_keys=True))
    return panel, force


def simulate_panel(model: BoolNet, force: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    n_steps, n_traj, n_vars = force.shape
    s = np.tile(model.initial, (n_traj, 1))
    traj = np.zeros((n_traj, n_steps, n_vars), dtype=bool)
    t0 = time.perf_counter()
    for t in range(n_steps):
        mask = force[t] >= 0
        if mask.any():
            s[mask] = force[t][mask].astype(bool)
        traj[:, t, :] = s
        new = np.empty_like(s)
        env = {"s": s, "np": np, "True": True, "False": False}
        for target, code in zip(model.rule_targets, model.rule_codes):
            val = eval(code, {"__builtins__": {}}, env)
            new[:, target] = val
        s = new
    elapsed = time.perf_counter() - t0
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = raw_rss / (1024 * 1024) if platform.system() == "Darwin" else raw_rss / 1024
    meta = pd.DataFrame(
        [
            {
                "n_trajectories": n_traj,
                "n_timepoints": n_steps,
                "n_variables": n_vars,
                "wall_seconds": elapsed,
                "seconds_per_trajectory": elapsed / max(n_traj, 1),
                "max_rss_mb": max_rss_mb,
            }
        ]
    )
    return traj, meta


def layer_arrays(model: BoolNet, traj: np.ndarray) -> dict[str, np.ndarray]:
    sym = model.symbols
    obs_syms = [s for s, n in model.names.items() if n in OBSERVABLE_STATES]
    input_syms = [s for s, n in model.names.items() if n in EXTERNAL_INPUTS]
    state_syms = sym.loc[sym["kind"].eq("S"), "boolnet_id"].tolist()
    internal_syms = [s for s in state_syms if s not in set(input_syms + obs_syms)]

    def flatten(symbols: list[str]) -> np.ndarray:
        idx = [model.index[s] for s in symbols if s in model.index]
        if not idx:
            return np.empty((traj.shape[0], 0))
        return traj[:, :, idx].reshape((traj.shape[0], -1)).astype(float)

    def series(symbols: list[str]) -> np.ndarray:
        idx = [model.index[s] for s in symbols if s in model.index]
        return traj[:, :, idx].astype(float)

    return {
        "Observable/phenotypic state": flatten(obs_syms),
        "Internal regulatory state": flatten(internal_syms),
        "Dynamical observable state": delay_embed(series(obs_syms), lag=8, max_windows=None),
        "Dynamical internal state": delay_embed(
            series(internal_syms), lag=8, max_windows=MAX_HANKEL_WINDOWS
        ),
    }


def phenotype_descriptors(model: BoolNet, traj: np.ndarray) -> tuple[np.ndarray, list[str]]:
    obs_syms = [s for s, n in model.names.items() if n in OBSERVABLE_STATES]
    idx = [model.index[s] for s in obs_syms]
    names = [model.names[s] for s in obs_syms]
    if not idx:
        return np.empty((traj.shape[0], 0)), []
    y = traj[:, :, idx].astype(float)
    descs = []
    desc_names = []
    final = y[:, -1, :]
    descs.append(final)
    desc_names.extend([f"final::{n}" for n in names])
    auc = y.mean(axis=1)
    descs.append(auc)
    desc_names.extend([f"active_fraction::{n}" for n in names])
    transitions = np.abs(np.diff(y, axis=1)).sum(axis=1)
    descs.append(transitions)
    desc_names.extend([f"transitions::{n}" for n in names])
    first_on = np.full((y.shape[0], y.shape[2]), N_TIMEPOINTS, dtype=float)
    for j in range(y.shape[2]):
        active = y[:, :, j] > 0.5
        any_on = active.any(axis=1)
        first_on[any_on, j] = active[any_on].argmax(axis=1)
    descs.append(first_on)
    desc_names.extend([f"first_on::{n}" for n in names])
    return np.concatenate(descs, axis=1), desc_names


def intervention_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    records = []
    all_env = sorted({k for s in panel["environment"] for k in json.loads(s).keys()})
    all_forced = sorted({k for s in panel["forced"] for k in json.loads(s).keys()})
    types = sorted(panel["perturbation_type"].unique())
    for _, row in panel.iterrows():
        env = json.loads(row["environment"])
        forced = json.loads(row["forced"])
        rec = {}
        for key in all_env:
            rec[f"env::{key}"] = float(env.get(key, 0))
        for key in all_forced:
            rec[f"force::{key}"] = float(forced.get(key, 0))
        for typ in types:
            rec[f"type::{typ}"] = 1.0 if row["perturbation_type"] == typ else 0.0
        records.append(rec)
    df = pd.DataFrame(records).fillna(0.0)
    return df.to_numpy(float), df.columns.tolist()


def intervention_response(panel: pd.DataFrame, phenotype: np.ndarray, desc_names: list[str]) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    u, u_names = intervention_matrix(panel)
    u, kept_u = remove_constant_columns(u, u_names)
    y, kept_y = remove_constant_columns(phenotype, desc_names)
    uz = standardize_matrix(u)
    yz = standardize_matrix(y)
    ridge = 1e-6
    j = np.linalg.solve(uz.T @ uz + ridge * np.eye(uz.shape[1]), uz.T @ yz)
    s = np.linalg.svd(j, compute_uv=False)
    p = (s**2) / np.sum(s**2) if len(s) else np.array([])
    row = {
        "layer": "Intervention response",
        "n_samples": int(panel.shape[0]),
        "n_features_varying": int(uz.shape[1]),
        "n_outputs_varying": int(yz.shape[1]),
        **dimension_metrics(s**2),
        "condition_number": float(s[0] / s[-1]) if len(s) and s[-1] > EPS else np.nan,
        "twonn_id": twonn_id(np.concatenate([uz, yz], axis=1)),
        "levina_bickel_id": levina_bickel_id(np.concatenate([uz, yz], axis=1)),
    }
    spectrum = pd.DataFrame(
        {
            "layer": "Intervention response",
            "pc": np.arange(1, len(s) + 1),
            "singular_value": s,
            "variance_explained": p,
            "cumulative_variance": np.cumsum(p),
        }
    )
    coeff = pd.DataFrame(j, index=kept_u, columns=kept_y)
    coeff.index.name = "design_feature"
    return row, spectrum, coeff.reset_index()


def matched_complexity(
    model: BoolNet,
    panel: pd.DataFrame,
    traj: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    phenotype, phenotype_names = phenotype_descriptors(model, traj)

    representative_rows = []
    representative_spectra = []
    coeff = pd.DataFrame()
    boot_rows = []
    n = traj.shape[0]
    for b in range(n_boot):
        print(f"matched bootstrap {b + 1}/{n_boot}", flush=True)
        idx = rng.choice(np.arange(n), size=N_MATCHED, replace=False)
        sub_panel = panel.iloc[idx].reset_index(drop=True)
        sub_traj = traj[idx]
        sub_layers = layer_arrays(model, sub_traj)
        for layer, x in sub_layers.items():
            row, spec = summarize_layer(layer, x, compute_nonlinear=(b == 0))
            row["bootstrap_id"] = b
            boot_rows.append(row)
            if b == 0:
                row_rep = dict(row)
                row_rep["scope"] = "representative_matched_subset"
                representative_rows.append(row_rep)
                representative_spectra.append(spec)
        sub_pheno, sub_pheno_names = phenotype_descriptors(model, sub_traj)
        row, spec, coeff_b = intervention_response(sub_panel, sub_pheno, sub_pheno_names)
        row["bootstrap_id"] = b
        boot_rows.append(row)
        if b == 0:
            row_rep = dict(row)
            row_rep["scope"] = "representative_matched_subset"
            representative_rows.append(row_rep)
            representative_spectra.append(spec)
            coeff = coeff_b

    boot = pd.DataFrame(boot_rows)
    summary = []
    for layer, g in boot.groupby("layer"):
        rec = {"layer": layer, "n_bootstrap_matched_subsets": int(g.shape[0])}
        for metric in ["entropy_rank", "participation_rank", "pc95", "pc99", "twonn_id", "levina_bickel_id"]:
            vals = g[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rec[f"{metric}_median"] = float(np.median(vals))
                rec[f"{metric}_lo"] = float(np.percentile(vals, 2.5))
                rec[f"{metric}_hi"] = float(np.percentile(vals, 97.5))
        summary.append(rec)

    return (
        pd.DataFrame(representative_rows),
        pd.concat(representative_spectra, ignore_index=True),
        boot,
        pd.DataFrame(summary),
        coeff,
    )


def save_plots(spectra: pd.DataFrame, boot_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    layers = [
        "Observable/phenotypic state",
        "Internal regulatory state",
        "Dynamical observable state",
        "Dynamical internal state",
        "Intervention response",
    ]
    plt.figure(figsize=(8, 5))
    for layer in layers:
        g = spectra[spectra["layer"] == layer]
        if g.empty:
            continue
        plt.plot(g["pc"], g["cumulative_variance"], marker="o", markersize=2.5, linewidth=1.1, label=layer)
    plt.axhline(0.95, color="#777777", linestyle="--", linewidth=0.8)
    plt.axhline(0.99, color="#999999", linestyle=":", linewidth=0.8)
    plt.xlabel("Principal component / singular direction")
    plt.ylabel("Cumulative variance")
    plt.ylim(0, 1.01)
    plt.title("rxncon yeast cell-cycle model: empirical complexity spectra")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_external_complexity_spectra.svg")
    plt.close()

    plot_df = boot_summary.copy()
    plot_df = plot_df[plot_df["layer"].isin(layers)]
    x = np.arange(len(plot_df))
    plt.figure(figsize=(8.5, 4.8))
    med = plot_df["entropy_rank_median"].to_numpy(float)
    lo = plot_df["entropy_rank_lo"].to_numpy(float)
    hi = plot_df["entropy_rank_hi"].to_numpy(float)
    plt.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", capsize=3)
    plt.xticks(x, plot_df["layer"], rotation=30, ha="right")
    plt.ylabel("Entropy effective rank, matched 125 x 49")
    plt.title("rxncon matched-subset effective dimensionality")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_external_complexity_bootstrap.svg")
    plt.close()


def write_panel_long(model: BoolNet, panel: pd.DataFrame, traj: np.ndarray, out: Path) -> None:
    rows = []
    watch = sorted(
        set(
            [s for s, n in model.names.items() if n in OBSERVABLE_STATES or n in EXTERNAL_INPUTS]
            + [s for s in model.symbol_order if s.startswith(("K", "O"))]
        )
    )
    idx = [model.index[s] for s in watch]
    for i, row in panel.iterrows():
        for t in range(traj.shape[1]):
            rec = {"trajectory_id": row["trajectory_id"], "time_index": t}
            for sym, j in zip(watch, idx):
                val = int(traj[i, t, j])
                if val:
                    rec[model.names[sym]] = val
            rows.append(rec)
    pd.DataFrame(rows).fillna(0).to_csv(out, index=False)


def acceptance_gate(boot_summary: pd.DataFrame) -> pd.DataFrame:
    current = {
        "Internal regulatory state": {"entropy": 2.79, "pr": 2.41, "priority": 1},
        "Dynamical internal state": {"entropy": 2.11, "pr": 1.52, "priority": 2},
        "Observable/phenotypic state": {"entropy": 6.93, "pr": 3.29, "priority": 3},
        "Dynamical observable state": {"entropy": 2.11, "pr": 1.52, "priority": 4},
        "Intervention response": {"entropy": 1.92, "pr": 1.82, "priority": 5},
    }
    rows = []
    for _, row in boot_summary.iterrows():
        layer = row["layer"]
        if layer not in current:
            continue
        ref = current[layer]
        pass_entropy = row["entropy_rank_median"] > 2.0 * ref["entropy"]
        pass_pr = row["participation_rank_median"] > 1.5 * ref["pr"]
        rows.append(
            {
                "layer": layer,
                "priority": ref["priority"],
                "current_entropy_reference": ref["entropy"],
                "current_pr_reference": ref["pr"],
                "rxncon_entropy_median": row["entropy_rank_median"],
                "rxncon_pr_median": row["participation_rank_median"],
                "entropy_ratio": row["entropy_rank_median"] / ref["entropy"],
                "pr_ratio": row["participation_rank_median"] / ref["pr"],
                "passes_entropy_2x": bool(pass_entropy),
                "passes_pr_1_5x": bool(pass_pr),
                "passes_layer_gate": bool(pass_entropy and pass_pr),
            }
        )
    gate = pd.DataFrame(rows).sort_values("priority")
    important_passes = int(gate["passes_layer_gate"].sum()) if not gate.empty else 0
    gate["passes_model_gate_at_least_two_layers"] = important_passes >= 2
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-panel", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=80)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--force-simulate", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = MODELS_DIR / "CDC_S_cerevisiae_perturbable"
    inventory = model_inventory()
    compile_info = compile_model(prefix, perturbable=True, force=args.force_compile)
    inventory.update(compile_info)

    model = load_boolnet(prefix)
    symbol_summary = model.symbols.groupby("kind").size().reset_index(name="n_targets")
    symbol_summary.to_csv(OUT_DIR / "rxncon_symbol_summary.csv", index=False)
    model.symbols.to_csv(OUT_DIR / "rxncon_compiled_symbols.csv", index=False)

    panel_path = OUT_DIR / "rxncon_perturbation_panel.csv"
    traj_path = OUT_DIR / "rxncon_trajectory_panel_bool.npz"
    if panel_path.exists() and traj_path.exists() and not args.force_simulate:
        panel = pd.read_csv(panel_path)
        traj = np.load(traj_path)["traj"].astype(bool)
        runtime = pd.DataFrame()
    else:
        panel, force = build_panel(model, args.n_panel, RNG_SEED)
        traj, runtime = simulate_panel(model, force)
        panel.to_csv(panel_path, index=False)
        np.savez_compressed(traj_path, traj=traj)
        runtime.to_csv(OUT_DIR / "rxncon_runtime_benchmark.csv", index=False)
        write_panel_long(model, panel, traj, OUT_DIR / "rxncon_observable_input_perturbation_trajectories.csv")

    full, spectra, boot, boot_summary, coeff = matched_complexity(model, panel, traj, args.n_bootstrap, RNG_SEED + 1)
    gate = acceptance_gate(boot_summary)
    full.to_csv(OUT_DIR / "rxncon_complexity_representative_matched_subset.csv", index=False)
    spectra.to_csv(OUT_DIR / "rxncon_complexity_spectra.csv", index=False)
    boot.to_csv(OUT_DIR / "rxncon_complexity_matched_bootstrap.csv", index=False)
    boot_summary.to_csv(OUT_DIR / "rxncon_complexity_matched_summary.csv", index=False)
    coeff.to_csv(OUT_DIR / "rxncon_intervention_response_coefficients.csv", index=False)
    gate.to_csv(OUT_DIR / "rxncon_high_complexity_gate.csv", index=False)

    inventory.update(
        {
            "compiled_boolnet_targets": int(len(model.symbols)),
            "compiled_rule_count": int(len(model.rule_targets)),
            "panel_trajectories": int(len(panel)),
            "timepoints": N_TIMEPOINTS,
            "external_inputs": EXTERNAL_INPUTS,
            "observable_states_requested": OBSERVABLE_STATES,
            "observable_states_found": [model.names[s] for s in model.names if model.names[s] in OBSERVABLE_STATES],
        }
    )
    (OUT_DIR / "rxncon_model_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True))
    save_plots(spectra, boot_summary)

    print("rxncon external complexity screen complete")
    print(boot_summary.sort_values("layer").to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
