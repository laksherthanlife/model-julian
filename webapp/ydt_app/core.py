"""Small, dependency-light, observable-data API used by the handover UI.

This module deliberately does not expose simulator truth or run the expensive
rxncon/Yeast9 validation scripts.  It implements a general culture-level
trajectory baseline that can be trained from a wet-lab CSV: standardized
inputs -> PCA curve coefficients -> ridge regression.  Every split is made on
complete cultures, never on timepoint rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    _SRC = str(Path(__file__).resolve().parents[2] / "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from yeast_validation import gem_trainable_state_space as tss
except ImportError:  # The API can still run without the optional research package.
    tss = None

ROLE_NAMES = {
    "culture_id": "Culture ID",
    "time": "Time",
    "environment": "Environment / control",
    "product": "Product",
    "biomass": "Biomass",
    "reporter": "Reporter",
    "genotype": "Genotype / edit",
    "ignore": "Ignore",
}

HIDDEN_PATTERNS = re.compile(r"rxncon|module|gsm|flux|reaction_bound|bottleneck|latent_truth|simulator", re.I)


def load_dataset(text: str, filename: str = "upload.csv") -> tuple[pd.DataFrame, str]:
    """Read CSV/TSV text and return a dataframe plus its content hash."""
    if not text or not text.strip():
        raise ValueError("The uploaded file is empty.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # pragma: no cover - pandas message varies by version
        raise ValueError(f"Could not read {filename}. Upload a valid CSV or TSV file.") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) == 0:
        raise ValueError("The file has no columns.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return df, digest


def _suggest_role(name: str, series: pd.Series) -> str:
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    if any(x in n for x in ("cultureid", "culture", "replicate", "well", "sampleid")):
        return "culture_id"
    if n in {"t", "time", "hour", "hours", "day", "days", "timestamp"} or "time" in n:
        return "time"
    if any(x in n for x in ("product", "betacarotene", "titer", "titre")):
        return "product"
    if any(x in n for x in ("biomass", "od600", "cellmass", "density")) or n == "od" or n.startswith("od") and n[2:].isdigit():
        return "biomass"
    if any(x in n for x in ("strain", "genotype", "knockout", "ko", "oe", "edit")):
        return "genotype"
    if any(x in n for x in ("reporter", "sensor", "gfp", "rfp", "mcherry", "stress", "energy", "resource", "checkpoint", "morphology", "capacity")):
        return "reporter"
    if any(x in n for x in ("temperature", "temp", "ph", "do", "oxygen", "glucose", "carbon", "nutrient", "feed", "kla", "pressure", "salinity")):
        return "environment"
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1:
        return "environment"
    return "ignore"


def infer_column_roles(df: pd.DataFrame) -> dict[str, str]:
    """Suggest a role for every uploaded column; users may override all choices."""
    result = {str(c): _suggest_role(str(c), df[c]) for c in df.columns}
    # Resolve common collisions deterministically: first matching ID/time/product wins.
    for role in ("culture_id", "time", "product"):
        matches = [c for c, r in result.items() if r == role]
        for c in matches[1:]:
            result[c] = "ignore"
    return result


def _mapping_sets(mapping: dict[str, Any]) -> dict[str, list[str]]:
    out = {k: [] for k in ROLE_NAMES}
    for key, value in mapping.items():
        if key in out and isinstance(value, list):
            out[key] = [str(x) for x in value if str(x)]
        elif key in out and value:
            out[key] = [str(value)]
    # UI may send one role per column as {column: role}.
    if not any(out.values()):
        for col, role in mapping.items():
            if role in out:
                out[role].append(str(col))
    return out


def validate_dataset(df: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    roles = _mapping_sets(mapping)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    info: list[dict[str, str]] = []
    required = (("culture_id", "Culture ID"), ("time", "Time"), ("product", "Product"), ("environment", "Environment / control"))
    for role, label in required:
        if not roles[role]:
            errors.append({"message": f"No column has been assigned as {label}."})
    missing_cols = [c for cols in roles.values() for c in cols if c not in df.columns]
    if missing_cols:
        errors.append({"message": f"Mapped columns are not present in the file: {', '.join(missing_cols)}."})
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info, "metrics": {"cultures": 0, "rows": len(df)}}

    cid, time, product = roles["culture_id"][0], roles["time"][0], roles["product"][0]
    if df[cid].isna().any():
        errors.append({"message": "Culture ID contains missing values."})
    numeric_cols = [time, product, *roles["environment"], *roles["biomass"], *roles["reporter"]]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.isna().sum() > 0:
                errors.append({"message": f"Column '{col}' contains non-numeric values where numeric measurements are required."})
    if df.duplicated([cid, time]).any():
        errors.append({"message": "At least one culture contains duplicate timepoints."})
    duplicate_count = int(df.duplicated([cid, time]).sum())
    cultures = df[cid].dropna().astype(str).unique().tolist()
    grouped = df.assign(_culture=df[cid].astype(str)).groupby("_culture", sort=False)
    counts = grouped[time].count().to_numpy()
    time_grids = [tuple(pd.to_numeric(g[time], errors="coerce").dropna().round(10).tolist()) for _, g in grouped]
    if len(set(counts.tolist())) > 1:
        warnings.append({"message": f"Time sampling differs between {int(np.sum(counts != np.median(counts)))} cultures."})
    for col in [*roles["environment"], *roles["biomass"], *roles["reporter"], product]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            warnings.append({"message": f"{col} contains {n_missing} missing value(s)."})
    if not roles["biomass"]:
        warnings.append({"message": "No biomass column is mapped. Product-only candidates remain available."})
    if not roles["reporter"]:
        warnings.append({"message": "No reporter columns are mapped. Biosensor candidates will be unavailable."})
    suspicious = [c for c in df.columns if HIDDEN_PATTERNS.search(str(c)) and c not in roles["ignore"]]
    if suspicious:
        warnings.append({"message": f"Potential simulator-only columns detected: {', '.join(suspicious)}. Map them to Ignore unless they are genuine wet-lab measurements."})
    info.append({"message": "Cultures, not timepoint rows, are the independent statistical units."})
    if duplicate_count == 0:
        info.append({"message": "No duplicate culture/time pairs detected."})
    metrics = {
        "cultures": len(cultures), "rows": len(df), "timepoints_median": int(np.median(counts)) if len(counts) else 0,
        "environment_count": len(roles["environment"]), "reporter_count": len(roles["reporter"]),
        "biomass_count": len(roles["biomass"]), "product_count": len(roles["product"]),
        "duplicate_rows": duplicate_count, "missing_cells": int(df.isna().sum().sum()),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info, "metrics": metrics,
            "culture_ids": cultures, "roles": roles, "preview": df.head(8).fillna("").to_dict(orient="records")}


def build_culture_manifest(df: pd.DataFrame, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    roles = _mapping_sets(mapping)
    cid, time = roles["culture_id"][0], roles["time"][0]
    result = []
    for culture, group in df.assign(_culture=df[cid].astype(str)).groupby("_culture", sort=False):
        result.append({"culture_id": culture, "row_indices": [int(x) for x in group.index], "n_timepoints": int(len(group)),
                       "time_min": float(pd.to_numeric(group[time], errors="coerce").min()), "time_max": float(pd.to_numeric(group[time], errors="coerce").max())})
    return result


def split_cultures(manifest: list[dict[str, Any]], seed: int = 42, fractions=(0.7, 0.15, 0.15)) -> dict[str, list[str]]:
    ids = [m["culture_id"] for m in manifest]
    if len(ids) < 3:
        raise ValueError("At least three complete cultures are required for a culture-level split.")
    rng = np.random.default_rng(seed)
    shuffled = ids.copy(); rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * fractions[0])))
    n_val = max(1, int(round(n * fractions[1])))
    if n_train + n_val >= n: n_val = 1; n_train = n - 2
    return {"train": shuffled[:n_train], "validation": shuffled[n_train:n_train+n_val], "test": shuffled[n_train+n_val:]}


def _culture_arrays(df: pd.DataFrame, mapping: dict[str, Any], splits: dict[str, list[str]], target_points: int | None = None):
    roles = _mapping_sets(mapping); cid, time, product = roles["culture_id"][0], roles["time"][0], roles["product"][0]
    groups = {str(k): g.sort_values(time) for k, g in df.assign(_culture=df[cid].astype(str)).groupby("_culture", sort=False)}
    lengths = [len(g) for g in groups.values()]
    target_points = target_points or max(2, int(np.median(lengths)))
    grid = np.linspace(0, 1, target_points)
    targets = {}; feature_values = {}
    feature_columns = [*roles["environment"], *roles["biomass"], *roles["reporter"], *roles["genotype"]]
    for culture, group in groups.items():
        t = pd.to_numeric(group[time], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[product], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(t) & np.isfinite(y)
        if keep.sum() < 2: continue
        t = t[keep]; y = y[keep]; span = max(float(t.max()-t.min()), 1e-12); tn = (t-t.min())/span
        targets[culture] = np.interp(grid, tn, y).astype(float)
        row = group.iloc[0].copy()
        vals = []
        for col in feature_columns:
            val = row[col]
            if col in roles["genotype"] and not pd.api.types.is_numeric_dtype(df[col]): vals.append(str(val) if pd.notna(val) else "__MISSING__")
            else:
                nums = pd.to_numeric(group[col], errors="coerce")
                vals.append(float(nums.dropna().iloc[0]) if len(nums.dropna()) else 0.0)
        feature_values[culture] = vals
    return targets, feature_values, feature_columns, grid


def _measurement_curves(df: pd.DataFrame, mapping: dict[str, Any], ids: list[str], grid: np.ndarray):
    """Resample observable channels only. Missing optional values remain masked."""
    roles = _mapping_sets(mapping); cid, time = roles["culture_id"][0], roles["time"][0]
    channels = [roles["product"][0], *roles["biomass"], *roles["reporter"]]
    groups = {str(k): g.sort_values(time) for k, g in df.assign(_culture=df[cid].astype(str)).groupby("_culture", sort=False)}
    curves = {c: [] for c in channels}; masks = {c: [] for c in channels}
    for culture in ids:
        group = groups[culture]; t = pd.to_numeric(group[time], errors="coerce").to_numpy(float)
        keep_t = np.isfinite(t); t = t[keep_t]
        if len(t) < 2: continue
        span=max(float(t.max()-t.min()),1e-12); tn=(t-t.min())/span
        for col in channels:
            y=pd.to_numeric(group[col],errors="coerce").to_numpy(float)[keep_t]
            valid=np.isfinite(y)
            if valid.sum() >= 2:
                curves[col].append(np.interp(grid,tn[valid],y[valid]))
                masks[col].append(np.interp(grid,tn[valid],valid.astype(float)) > 0.5)
            elif valid.sum() == 1:
                curves[col].append(np.full(len(grid),float(y[valid][0]))); masks[col].append(np.zeros(len(grid),bool))
            else:
                curves[col].append(np.zeros(len(grid))); masks[col].append(np.zeros(len(grid),bool))
    return {c:np.asarray(v,float) for c,v in curves.items()}, {c:np.asarray(v,bool) for c,v in masks.items()}


def _train_observation_model(df, mapping, X, columns, ids, train_ids, val_ids, test_ids, grid, selected_reporters, use_biomass, seed=42):
    roles=_mapping_sets(mapping); env_active=[i for i,col in enumerate(columns) if col in roles["environment"]]
    curves, masks=_measurement_curves(df,mapping,ids,grid); lookup={c:i for i,c in enumerate(ids)}
    tr=np.array([lookup[c] for c in train_ids]); va=np.array([lookup[c] for c in val_ids]); te=np.array([lookup[c] for c in test_ids])
    product=roles["product"][0]; obs_cols=[product] + (roles["biomass"] if use_biomass else []) + list(selected_reporters)
    # Train-only channel normalization and explicit observed masks.
    stats={}
    for col in obs_cols:
        raw=curves[col][tr]; valid=masks[col][tr]
        vals=raw[valid]; mean=float(vals.mean()) if len(vals) else 0.0; std=float(vals.std()) if len(vals) else 1.0
        stats[col]=[mean, std if std>1e-9 else 1.0]
    env_mean=X[tr][:,env_active].mean(axis=0); env_std=np.where(X[tr][:,env_active].std(axis=0)<1e-9,1.0,X[tr][:,env_active].std(axis=0))
    env_z=(X[:,env_active]-env_mean)/env_std
    cutoffs=[max(1,min(len(grid)-2,int(round(len(grid)*f)))) for f in (0.1,0.2,0.4,0.6)]
    models={}; validation_scores=[]; test_scores=[]
    for cutoff in cutoffs:
        feats=[]
        for b in range(len(ids)):
            parts=[env_z[b]]
            for col in obs_cols:
                z=(curves[col][b]-stats[col][0])/stats[col][1]; observed=(masks[col][b] & (np.arange(len(grid))<=cutoff)).astype(float)
                parts.extend([z*observed,observed])
            feats.append(np.concatenate(parts))
        F=np.asarray(feats); Ftr=F[tr]; meanF=Ftr.mean(axis=0); stdF=np.where(Ftr.std(axis=0)<1e-9,1.0,Ftr.std(axis=0)); Fz=(F-meanF)/stdF
        y=(curves[product]-stats[product][0])/stats[product][1]
        W=_ridge(Fz[tr],y[tr],alpha=10.0); pred= (np.column_stack([np.ones(len(Fz)),Fz])@W)*stats[product][1]+stats[product][0]
        models[str(cutoff)]={"W":W.tolist(),"feature_mean":meanF.tolist(),"feature_std":stdF.tolist(),"cutoff":cutoff}
        def future_score(ix):
            return _metrics(curves[product][ix,cutoff+1:],pred[ix,cutoff+1:])
        validation_scores.append(future_score(va)); test_scores.append(future_score(te))
    return {"name":"Observation-conditioned state-space (observable)","selected_columns":[*roles["environment"],*(roles["biomass"] if use_biomass else []),*selected_reporters],"train":{},"validation":{"trajectory_nrmse":float(np.mean([x["trajectory_nrmse"] for x in validation_scores])),"final_product_error":float(np.mean([x["final_product_error"] for x in validation_scores])),"auc_error":float(np.mean([x["auc_error"] for x in validation_scores]))},"test":{"trajectory_nrmse":float(np.mean([x["trajectory_nrmse"] for x in test_scores])),"final_product_error":float(np.mean([x["final_product_error"] for x in test_scores])),"auc_error":float(np.mean([x["auc_error"] for x in test_scores]))},"future_by_cutoff":{"validation":validation_scores,"test":test_scores,"fractions":[0.1,0.2,0.4,0.6]},"model":{"kind":"observable_observer","input_columns":roles["environment"],"observation_columns":obs_cols,"reporters":selected_reporters,"use_biomass":use_biomass,"grid":grid.tolist(),"stats":stats,"env_mean":env_mean.tolist(),"env_std":env_std.tolist(),"cutoffs":cutoffs,"models":models,"target_grid":grid.tolist(),"seed":seed}}


def _encode_features(feature_values, feature_columns, train_ids, ids):
    categorical = []
    names = []; means=[]; stds=[]; train_rows=[]
    train_lookup = {c: feature_values[c] for c in train_ids if c in feature_values}
    for j, col in enumerate(feature_columns):
        vals = [train_lookup[c][j] for c in train_lookup]
        is_cat = any(isinstance(v, str) for v in vals); categorical.append(is_cat)
        if is_cat:
            cats = sorted(set(str(v) for v in vals));
            for cat in cats: names.append(f"{col}={cat}")
        else:
            names.append(col)
    for cid in ids:
        row=[]
        for j, col in enumerate(feature_columns):
            v=feature_values[cid][j]
            if categorical[j]:
                cats=sorted(set(str(feature_values[c][j]) for c in train_lookup))
                row.extend([1.0 if str(v)==cat else 0.0 for cat in cats])
            else: row.append(float(v) if np.isfinite(float(v)) else 0.0)
        train_rows.append(row)
    X=np.asarray(train_rows,dtype=float)
    train_mask=np.array([cid in set(train_ids) for cid in ids])
    means=X[train_mask].mean(axis=0) if X.shape[1] else np.zeros(0); stds=X[train_mask].std(axis=0) if X.shape[1] else np.zeros(0)
    stds=np.where(stds<1e-9,1.0,stds)
    return (X-means)/stds, names, means, stds


def _ridge(X, Y, alpha=1e-3):
    Xa=np.column_stack([np.ones(len(X)), X]); reg=np.eye(Xa.shape[1]); reg[0,0]=0
    return np.linalg.solve(Xa.T@Xa+alpha*reg, Xa.T@Y)


def _state_space_candidate(X, columns, active, targets, ids, train_ids, val_ids, test_ids, grid, seed=42, epochs=180):
    """Train the repository's actual BPTT state-space model on observable curves.

    This deliberately uses product only as the supervised output. Reporters,
    if present in the upload, are not converted into privileged hidden-state
    targets or GSM controls.
    """
    if tss is None:
        return None
    lookup = {c: i for i, c in enumerate(ids)}
    tr = np.array([lookup[c] for c in train_ids], dtype=int)
    va = np.array([lookup[c] for c in val_ids], dtype=int)
    te = np.array([lookup[c] for c in test_ids], dtype=int)
    env = X[:, active]
    train_mask = np.zeros(len(ids), dtype=bool); train_mask[tr] = True
    env_mean = env[train_mask].mean(axis=0); env_std = np.where(env[train_mask].std(axis=0) < 1e-9, 1.0, env[train_mask].std(axis=0))
    env_z = (env - env_mean) / env_std
    y = np.asarray([targets[c] for c in ids], dtype=float).T
    y_mean = float(y[:, tr].mean()); y_std = float(y[:, tr].std()) or 1.0
    y_z = (y - y_mean) / y_std
    mask = np.tile(train_mask.astype(float), (y.shape[0], 1))
    params = tss.init_params(env.shape[1], latent_dim=8, hidden_dim=16, head_names=["product"], n_controls=0, seed=seed)
    adam = tss.AdamState(params, lr=0.018)
    for _ in range(epochs):
        _, grads, _ = tss.compute_loss_and_grads(params, env_z, y.shape[0], {"product": y_z}, {"product": mask}, lambda_dyn=2e-4)
        adam.step(params, grads)
    cache = tss.forward(params, env_z, y.shape[0])
    pred = tss.head_predictions(params, cache, ["product"])["product"] * y_std + y_mean
    def score(index): return _metrics(y[:, index].T, pred[:, index].T)
    arrays = {k: getattr(params, k).tolist() for k in ("Wg0", "bg0", "Wz", "We", "b1", "Wo", "bo")}
    arrays["heads"] = {k: [w.tolist(), b.tolist()] for k, (w, b) in params.heads.items()}
    return {"name":"Observable state-space (product)","selected_columns":[columns[i] for i in active],"train":score(tr),"validation":score(va),"test":score(te),
            "model":{"kind":"state_space","params":arrays,"env_mean":env_mean.tolist(),"env_std":env_std.tolist(),"product_mean":y_mean,"product_std":y_std,"target_grid":grid.tolist(),"input_columns":[columns[i] for i in active],"latent_dim":8},
            "test_predictions":{"culture_ids":test_ids,"truth":y[:,te].T.tolist(),"predicted":pred[:,te].T.tolist()}}


def _metrics(y, pred):
    err=pred-y; denom=max(float(np.std(y)),1e-3)
    auc = lambda a: np.trapezoid(a, axis=1) if hasattr(np, "trapezoid") else np.trapz(a, axis=1)
    return {"trajectory_nrmse": float(np.sqrt(np.mean(err**2))/denom), "rmse": float(np.sqrt(np.mean(err**2))),
            "final_product_error": float(np.mean(np.abs(err[:,-1]))), "auc_error": float(np.mean(np.abs(auc(pred)-auc(y))))}


def train_candidate_models(df: pd.DataFrame, mapping: dict[str, Any], splits: dict[str, list[str]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config=config or {}; roles=_mapping_sets(mapping); targets, fv, columns, grid=_culture_arrays(df,mapping,splits)
    allowed = set(config.get("allowed_columns", columns))
    ids=[c for c in sum((splits[k] for k in ("train","validation","test")),[]) if c in targets and c in fv]
    train_ids=[c for c in splits["train"] if c in targets]; val_ids=[c for c in splits["validation"] if c in targets]; test_ids=[c for c in splits["test"] if c in targets]
    X,names,means,stds=_encode_features(fv,columns,train_ids,ids); idx={c:i for i,c in enumerate(ids)}
    Y=np.asarray([targets[c] for c in ids]); tr=np.array([idx[c] for c in train_ids]); va=np.array([idx[c] for c in val_ids]); te=np.array([idx[c] for c in test_ids])
    candidates=[("Environment only",roles["environment"],False,False),("Environment + biomass",roles["environment"]+roles["biomass"],True,False),("Environment + reporters",roles["environment"]+roles["reporter"],False,True),("Environment + biomass + reporters",roles["environment"]+roles["biomass"]+roles["reporter"],True,True)]
    if roles["genotype"]: candidates.append(("Environment + genotype",roles["environment"]+roles["genotype"],False,False))
    results=[]
    # Fit each candidate by selecting its feature columns, preserving train-only encoding.
    for name, selected, _, _ in candidates:
        selected = [col for col in selected if col in allowed]
        active=[i for i,name in enumerate(names) if name in selected or any(name.startswith(f"{col}=") for col in selected)]
        if not active: continue
        W=_ridge(X[tr][:,active],Y[tr]); pred=lambda ix: np.column_stack([np.ones(len(ix)),X[ix][:,active]])@W
        pva=pred(va); pte=pred(te); ptr=pred(tr)
        result={"name":name,"selected_columns":selected,"train":_metrics(Y[tr],ptr),"validation":_metrics(Y[va],pva),"test":_metrics(Y[te],pte),
                "model":{"kind":"direct_ridge","W":W.tolist(),"feature_indices":active,"feature_columns":names,"feature_means":means.tolist(),"feature_stds":stds.tolist(),"input_columns":columns,"target_grid":grid.tolist(),"pca_note":"Direct curve ridge baseline; PCA-free for arbitrary uploaded schemas."},
                "test_predictions":{"culture_ids":test_ids,"truth":Y[te].tolist(),"predicted":pte.tolist()}}
        results.append(result)
    env_active = [i for i, col in enumerate(columns) if col in roles["environment"] and col in allowed]
    if env_active and tss is not None:
        state_result = _state_space_candidate(X, columns, env_active, targets, ids, train_ids, val_ids, test_ids, grid)
        if state_result is not None:
            results.append(state_result)
    if roles["environment"]:
        reporter_sets=[([],False),([],True)]
        if roles["reporter"]:
            reporter_sets += [(roles["reporter"],False),(roles["reporter"],True)]
        for reporter_set,use_biomass in reporter_sets:
            if any(c not in allowed for c in reporter_set) or (use_biomass and any(c not in allowed for c in roles["biomass"])): continue
            obs_result=_train_observation_model(df,mapping,X,columns,ids,train_ids,val_ids,test_ids,grid,reporter_set,use_biomass)
            obs_result["name"] += " + biomass" if use_biomass else ""
            obs_result["name"] += " + reporters" if reporter_set else ""
            results.append(obs_result)
    if not results: raise ValueError("No usable numeric environment columns were available for training.")
    observed_ranges={}
    for col in columns:
        vals=[]
        j=columns.index(col)
        for culture in ids:
            value=fv[culture][j]
            if not isinstance(value,str): vals.append(float(value))
        if vals: observed_ranges[col]={"min":float(min(vals)),"max":float(max(vals))}
    for result in results:
        result["model"]["observed_ranges"]={c:observed_ranges[c] for c in result["selected_columns"] if c in observed_ranges}
    results.sort(key=lambda r:(r["validation"]["trajectory_nrmse"],len(r["selected_columns"])))
    best=results[0]
    run_id=f"ydt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    return {"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"candidates":results,"best":best,"splits":splits,"n_train":len(tr),"n_validation":len(va),"n_test":len(te),"target_points":len(grid),"model_family":"observable direct trajectory ridge","scientific_note":"Generic uploaded-data state-space and exact Yeast9 verification are not enabled by this run."}


def select_best_model(training_result: dict[str, Any]) -> dict[str, Any]:
    return training_result["best"]


def _predict_features(model, condition):
    cols=model["input_columns"]; feature_names=model.get("feature_columns",cols); means=np.asarray(model["feature_means"],float); stds=np.asarray(model["feature_stds"],float)
    values=[]
    for name in feature_names:
        if "=" in name:
            col, category=name.split("=",1); values.append(1.0 if str(condition.get(col,""))==category else 0.0)
        else:
            v=condition.get(name,0.0)
            try: values.append(float(v))
            except (ValueError,TypeError): values.append(0.0)
    x=(np.asarray(values)-means)/stds
    active=model["feature_indices"]; W=np.asarray(model["W"],float)
    return (np.concatenate(([1.0], x[active])) @ W)[None, :]


def _predict_state_space(model, condition):
    env_cols = model["input_columns"]
    env = np.asarray([float(condition.get(col, 0.0)) for col in env_cols], dtype=float)
    env = (env - np.asarray(model["env_mean"], float)) / np.asarray(model["env_std"], float)
    p = model["params"]
    params = tss.StateSpaceParams(
        Wg0=np.asarray(p["Wg0"]), bg0=np.asarray(p["bg0"]), Wz=np.asarray(p["Wz"]), We=np.asarray(p["We"]),
        b1=np.asarray(p["b1"]), Wo=np.asarray(p["Wo"]), bo=np.asarray(p["bo"]),
        heads={k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in p["heads"].items()},
    )
    cache=tss.forward(params, env[None, :], len(model["target_grid"]))
    pred=tss.head_predictions(params, cache, ["product"])["product"][:, 0]
    pred=pred*float(model["product_std"])+float(model["product_mean"])
    return pred


def predict_with_observations(model_result: dict[str, Any], condition: dict[str, Any], observations: dict[str, list[Any]]) -> dict[str, Any]:
    model=model_result["model"] if "model" in model_result else model_result
    if model.get("kind") != "observable_observer":
        raise ValueError("The selected model does not support observation-updated prediction.")
    grid=np.asarray(model["target_grid"],float); cutoff_value=float(observations.get("cutoff_fraction",0.4)); cutoff=min(model["cutoffs"],key=lambda c:abs(c/ max(len(grid)-1,1)-cutoff_value))
    parts=[(float(condition.get(c,0.0))-m)/s for c,m,s in zip(model["input_columns"],model["env_mean"],model["env_std"])]
    for col in model["observation_columns"]:
        vals=np.asarray(observations.get(col,[]),dtype=float); vals=np.pad(vals[:cutoff+1],(0,max(0,len(grid)-len(vals))),constant_values=np.nan)[:len(grid)]
        mean,std=model["stats"][col]; valid=np.isfinite(vals).astype(float); vals=np.nan_to_num((vals-mean)/std,nan=0.0); observed=valid*(np.arange(len(grid))<=cutoff)
        parts.extend([vals*observed,observed])
    Fraw=np.concatenate([np.asarray([v]) if np.asarray(v).ndim==0 else np.asarray(v).reshape(-1) for v in parts])
    F=(Fraw-np.asarray(model["models"][str(cutoff)]["feature_mean"]))/np.asarray(model["models"][str(cutoff)]["feature_std"])
    W=np.asarray(model["models"][str(cutoff)]["W"]); y=np.concatenate(([1.0],F))@W
    product_mean,product_std=model["stats"][model["observation_columns"][0]]; y=y*product_std+product_mean
    return {"time_normalized":grid.tolist(),"predicted_product":y.tolist(),"observed_cutoff_fraction":cutoff/(len(grid)-1),"final_product":float(y[-1]),"auc":float(np.trapezoid(y,grid) if hasattr(np,"trapezoid") else np.trapz(y,grid)),"uncertainty":None,"uncertainty_note":"No calibrated uncertainty model is enabled."}


def predict_trajectory(model_result: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    model=model_result["model"] if "model" in model_result else model_result
    if model.get("kind")=="state_space":
        y=_predict_state_space(model, condition)
    elif model.get("kind")=="observable_observer":
        y=np.asarray(predict_with_observations(model_result,condition,{"cutoff_fraction":0.1})["predicted_product"],float)
    else:
        y=_predict_features(model,condition)[0]
    grid=np.asarray(model["target_grid"],float)
    auc = np.trapezoid(y,grid) if hasattr(np, "trapezoid") else np.trapz(y,grid)
    return {"time_normalized":grid.tolist(),"predicted_product":y.tolist(),"final_product":float(y[-1]),"auc":float(auc),"uncertainty":None,"uncertainty_note":"No calibrated uncertainty model is enabled for this workflow."}


def screen_candidates(model_result: dict[str, Any], ranges: dict[str, dict[str, float]], n_candidates: int=100, seed: int=42) -> list[dict[str, Any]]:
    rng=np.random.default_rng(seed); rows=[]; observed=model_result.get("model",model_result).get("observed_ranges",{})
    out_of_domain={col:(float(spec["min"])<observed[col]["min"] or float(spec["max"])>observed[col]["max"]) for col,spec in ranges.items() if col in observed}
    for i in range(max(1,min(int(n_candidates),5000))):
        condition={col: float(rng.uniform(float(spec["min"]),float(spec["max"]))) for col,spec in ranges.items()}
        pred=predict_trajectory(model_result,condition); rows.append({"rank":i+1,"candidate_id":f"V-{i+1:04d}","condition":condition,**pred,"verification":"MODEL PREDICTION ONLY","out_of_domain":any(out_of_domain.values()),"domain_warning":"Candidate range extends beyond observed training support." if any(out_of_domain.values()) else None})
    rows.sort(key=lambda r:r["final_product"],reverse=True)
    for i,row in enumerate(rows,1): row["rank"]=i
    return rows


def verify_with_yeast9(*args, **kwargs) -> dict[str, Any]:
    return {"available":False,"status":"unavailable","adapter":"canonical synthetic interface","reason":"Generic exact Yeast9 verification is not exposed for arbitrary uploaded schemas. No exact solver was run."}


def run_metadata(dataset_hash: str, mapping: dict[str, Any], splits: dict[str, list[str]], training: dict[str, Any]) -> dict[str, Any]:
    return {"run_id":training["run_id"],"timestamp":training["created_at"],"dataset_hash":dataset_hash,"column_mapping":mapping,"culture_ids_by_split":splits,"model_configuration":{"family":training["model_family"],"seed":42,"selected_inputs":training["best"]["selected_columns"]},"metrics":training["best"]["test"],"lockbox":{"raw_rxncon":True,"gsm_interface":True,"fluxes":True}}
