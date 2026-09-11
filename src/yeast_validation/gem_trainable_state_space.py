#!/usr/bin/env python3
"""A genuinely trainable small state-space model, replacing the production
teacher's untrained-random-feature architecture.

Why this exists: the audit of `run_reporter_grounded_hybrid_distillation.py`
found `initialize_state_space` draws `A, Wg, C, bf` once from a fixed seed and
never updates them -- `latent_rollout`'s recurrence is therefore a fixed
random nonlinear feature map of `environment`, not a learned dynamical model.
Only the closed-form ridge output heads are fit, which is also why reporter
supervision could never influence product prediction (each head is an
independent regression on the same untrained features).

This module implements the smallest defensible repair: every weight in both
the transition function and the output heads is trained by gradient descent,
with hand-derived backpropagation-through-time (BPTT) -- there is no autodiff
library in this environment (checked: no torch/jax/tensorflow installed) and
at D_model=16 with ~1200 total parameters, manual BPTT is both fast and, per
the handover spec's explicit preference, more auditable than opaque autodiff.

Architecture (all trainable):

    z[0]    = tanh(env @ Wg0 + bg0)                                  # initial state encoder
    h[t]    = tanh(z[t] @ Wz + env @ We + b1)                        # transition hidden layer
    dz[t]   = tanh(h[t] @ Wo + bo)                                   # bounded state derivative
    z[t+1]  = z[t] + dt_latent * dz[t]                                # Euler step (residual/Neural-ODE-style)

    y_col[t]  = z[t] @ Wh[col] + bh[col]           for col in output_heads (reporters, direct product/biomass in non-hybrid variants)
    c[t]      = (z[t] @ Wc + bc) * control_std + control_mean        for t in 0..T-2 (compact GSM control vector, hybrid variant only)

Every head reads the *same* z[t] -- so gradients from any head's loss flow
back through that head's weights into z[t], and from there through the
transition function's weights (Wz, We, Wo, b1, bo) and the initial encoder
(Wg0, bg0), for every earlier time step, via standard BPTT. This is the
structural fix for the reporter-invariance finding: a reporter loss now has a
real, checkable gradient path into the shared dynamics.

D_model = 16 is fixed per the handover spec (continuity with the previous
selected teacher latent dimension). The hidden transition width (H) and the
control-vector dimension are separate, smaller architecture choices, not
re-tuned against the mismatch experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-9
Z_CLIP = 12.0


@dataclass
class StateSpaceParams:
    Wg0: np.ndarray
    bg0: np.ndarray
    Wz: np.ndarray
    We: np.ndarray
    b1: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray
    heads: dict[str, tuple[np.ndarray, np.ndarray]]  # col -> (Wh (D,1), bh (1,))
    Wc: np.ndarray | None = None
    bc: np.ndarray | None = None

    def copy(self) -> "StateSpaceParams":
        return StateSpaceParams(
            Wg0=self.Wg0.copy(), bg0=self.bg0.copy(), Wz=self.Wz.copy(), We=self.We.copy(),
            b1=self.b1.copy(), Wo=self.Wo.copy(), bo=self.bo.copy(),
            heads={k: (w.copy(), b.copy()) for k, (w, b) in self.heads.items()},
            Wc=None if self.Wc is None else self.Wc.copy(),
            bc=None if self.bc is None else self.bc.copy(),
        )


def init_params(env_dim: int, latent_dim: int, hidden_dim: int, head_names: list[str], n_controls: int, seed: int, scale: float = 0.35) -> StateSpaceParams:
    rng = np.random.default_rng(seed)

    def normal(shape):
        return rng.normal(0.0, scale, size=shape) / np.sqrt(max(shape[0], 1))

    heads = {name: (normal((latent_dim, 1)), np.zeros(1)) for name in head_names}
    Wc = normal((latent_dim, n_controls)) if n_controls > 0 else None
    bc = np.zeros(n_controls) if n_controls > 0 else None
    return StateSpaceParams(
        Wg0=normal((env_dim, latent_dim)),
        bg0=np.zeros(latent_dim),
        Wz=normal((latent_dim, hidden_dim)),
        We=normal((env_dim, hidden_dim)),
        b1=np.zeros(hidden_dim),
        Wo=normal((hidden_dim, latent_dim)),
        bo=np.zeros(latent_dim),
        heads=heads,
        Wc=Wc,
        bc=bc,
    )


def _zero_like(params: StateSpaceParams) -> StateSpaceParams:
    z = params.copy()
    z.Wg0[:] = 0.0
    z.bg0[:] = 0.0
    z.Wz[:] = 0.0
    z.We[:] = 0.0
    z.b1[:] = 0.0
    z.Wo[:] = 0.0
    z.bo[:] = 0.0
    for k in z.heads:
        w, b = z.heads[k]
        w[:] = 0.0
        b[:] = 0.0
    if z.Wc is not None:
        z.Wc[:] = 0.0
        z.bc[:] = 0.0
    return z


@dataclass
class ForwardCache:
    env: np.ndarray  # (B, env_dim)
    z: np.ndarray  # (T, B, D)
    a0: np.ndarray  # (B, D) pre-tanh initial state
    h_pre: np.ndarray  # (T-1, B, H)
    h: np.ndarray  # (T-1, B, H)
    dz_pre: np.ndarray  # (T-1, B, D)
    dz: np.ndarray  # (T-1, B, D)
    dt_latent: float


def forward(params: StateSpaceParams, env: np.ndarray, t_len: int, dt_latent: float | None = None) -> ForwardCache:
    B = env.shape[0]
    D = params.Wg0.shape[1]
    H = params.Wz.shape[1]
    dt_latent = dt_latent if dt_latent is not None else 1.0 / max(t_len - 1, 1)

    a0 = env @ params.Wg0 + params.bg0
    z0 = np.tanh(a0)
    z = np.zeros((t_len, B, D))
    z[0] = z0
    h_pre = np.zeros((t_len - 1, B, H))
    h = np.zeros((t_len - 1, B, H))
    dz_pre = np.zeros((t_len - 1, B, D))
    dz = np.zeros((t_len - 1, B, D))
    for t in range(t_len - 1):
        h_pre[t] = z[t] @ params.Wz + env @ params.We + params.b1
        h[t] = np.tanh(h_pre[t])
        dz_pre[t] = h[t] @ params.Wo + params.bo
        dz[t] = np.tanh(dz_pre[t])
        z[t + 1] = np.clip(z[t] + dt_latent * dz[t], -Z_CLIP, Z_CLIP)
    return ForwardCache(env=env, z=z, a0=a0, h_pre=h_pre, h=h, dz_pre=dz_pre, dz=dz, dt_latent=dt_latent)


def head_predictions(params: StateSpaceParams, cache: ForwardCache, cols: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for col in cols:
        Wh, bh = params.heads[col]
        out[col] = (cache.z @ Wh + bh)[..., 0]  # (T, B)
    return out


def control_predictions(params: StateSpaceParams, cache: ForwardCache, control_mean: np.ndarray, control_std: np.ndarray) -> np.ndarray:
    """c[t] for t in 0..T-2 (one control vector per interval), destandardized to physical control units."""
    assert params.Wc is not None
    c_pre = cache.z[:-1] @ params.Wc + params.bc  # (T-1, B, n_controls)
    return c_pre * control_std + control_mean


def compute_loss_and_grads(
    params: StateSpaceParams,
    env: np.ndarray,
    t_len: int,
    head_targets: dict[str, np.ndarray],
    head_masks: dict[str, np.ndarray],
    surrogate=None,
    control_mean: np.ndarray | None = None,
    control_std: np.ndarray | None = None,
    surrogate_out_index: dict[str, int] | None = None,
    X_true: np.ndarray | None = None,
    B_true: np.ndarray | None = None,
    X_std: float = 1.0,
    B_std: float = 1.0,
    dt_real: float = 0.25,
    beta_deg_base: float = 0.002,
    lambda_dyn: float = 1e-3,
    dt_latent: float | None = None,
) -> tuple[float, StateSpaceParams, dict[str, float]]:
    """Forward pass + hand-derived BPTT. `head_targets`/`head_masks` cover the
    direct output heads (reporters always; product/biomass too for
    non-hybrid variants). If `surrogate` is given, the hybrid mechanistic
    path is also included: c[t] = control head output -> surrogate ->
    predicted biomass/product flux -> one-step teacher-forced mechanistic
    integration against `X_true`/`B_true` -- this is the path that lets
    product/biomass loss shape the *control* head (and, through it, the
    shared latent dynamics) without ever supervising the control head
    directly against privileged simulator values (Section 8 of the
    handover spec)."""
    cache = forward(params, env, t_len, dt_latent=dt_latent)
    T, B, D = cache.z.shape
    grads = _zero_like(params)
    total_loss = 0.0
    diagnostics: dict[str, float] = {}

    dz_grad = np.zeros((T, B, D))

    for col, target in head_targets.items():
        Wh, bh = params.heads[col]
        y_pre = (cache.z @ Wh + bh)[..., 0]
        mask = head_masks[col]
        n = max(float(np.sum(mask)), 1.0)
        diff = (y_pre - target) * mask
        col_loss = float(np.sum(diff**2)) / n
        total_loss += col_loss
        diagnostics[f"loss__{col}"] = col_loss
        dL_dy = (2.0 / n) * diff
        gW, gb = grads.heads[col]
        gW += np.einsum("tbd,tb->d", cache.z, dL_dy).reshape(D, 1)
        gb += np.array([np.sum(dL_dy)])
        dz_grad += dL_dy[..., None] * Wh[:, 0][None, None, :]

    if surrogate is not None:
        assert params.Wc is not None and control_mean is not None and control_std is not None
        assert X_true is not None and B_true is not None and surrogate_out_index is not None
        c_pre = cache.z[:-1] @ params.Wc + params.bc  # (T-1, B, n_controls)
        c = c_pre * control_std + control_mean
        n_hyb = max(float((T - 1) * B), 1.0)
        hyb_loss = 0.0
        bidx = surrogate_out_index["biomass_flux"]
        pidx = surrogate_out_index["beta_carotene_flux"]
        for t in range(T - 1):
            values, jac = surrogate.predict_and_grad(c[t], env)  # values (B,n_out), jac (B,n_out,n_controls)
            biomass_flux_hat = values[:, bidx]
            beta_flux_hat = values[:, pidx]
            X_hat_next = X_true[t] + dt_real * biomass_flux_hat * X_true[t]
            B_hat_next = B_true[t] + dt_real * beta_flux_hat * X_true[t] - dt_real * beta_deg_base * B_true[t]
            dX = (X_hat_next - X_true[t + 1]) / X_std
            dB = (B_hat_next - B_true[t + 1]) / B_std
            hyb_loss += float(np.sum(dX**2) + np.sum(dB**2)) / n_hyb

            dL_dXhat = (2.0 / (n_hyb * X_std**2)) * (X_hat_next - X_true[t + 1])
            dL_dBhat = (2.0 / (n_hyb * B_std**2)) * (B_hat_next - B_true[t + 1])
            dL_dbiomass_flux = dL_dXhat * dt_real * X_true[t]
            dL_dbeta_flux = dL_dBhat * dt_real * X_true[t]

            dL_dvalues = np.zeros_like(values)
            dL_dvalues[:, bidx] = dL_dbiomass_flux
            dL_dvalues[:, pidx] = dL_dbeta_flux
            dL_dc = np.einsum("bk,bkc->bc", dL_dvalues, jac)  # (B, n_controls)
            dL_dc_pre = dL_dc * control_std

            gWc_t = np.einsum("bd,bc->dc", cache.z[t], dL_dc_pre)
            grads.Wc += gWc_t
            grads.bc += np.sum(dL_dc_pre, axis=0)
            dz_grad[t] += dL_dc_pre @ params.Wc.T
        total_loss += hyb_loss
        diagnostics["loss__hybrid_mechanistic"] = hyb_loss

    # Dynamics regularization (encourages bounded, non-exploding derivatives).
    dyn_reg = lambda_dyn * float(np.mean(cache.dz_pre**2))
    total_loss += dyn_reg
    diagnostics["loss__dyn_reg"] = dyn_reg
    diagnostics["loss__total"] = total_loss

    g = dz_grad[T - 1].copy()
    for t in range(T - 2, -1, -1):
        pre_clip = cache.z[t] + cache.dt_latent * cache.dz[t]
        clip_mask = (np.abs(pre_clip) < Z_CLIP).astype(float)
        g_clipped = g * clip_mask

        dL_ddz = g_clipped * cache.dt_latent
        dL_ddz_pre = dL_ddz * (1 - cache.dz[t] ** 2)
        dL_ddz_pre += lambda_dyn * 2.0 * cache.dz_pre[t] / cache.dz_pre.size

        grads.Wo += np.einsum("bh,bd->hd", cache.h[t], dL_ddz_pre)
        grads.bo += np.sum(dL_ddz_pre, axis=0)
        dL_dh = dL_ddz_pre @ params.Wo.T
        dL_dh_pre = dL_dh * (1 - cache.h[t] ** 2)

        grads.Wz += np.einsum("bd,bh->dh", cache.z[t], dL_dh_pre)
        grads.We += np.einsum("be,bh->eh", env, dL_dh_pre)
        grads.b1 += np.sum(dL_dh_pre, axis=0)
        dz_t_from_transition = dL_dh_pre @ params.Wz.T

        g = g_clipped + dz_t_from_transition + dz_grad[t]

    dL_da0 = g * (1 - cache.z[0] ** 2)
    grads.Wg0 += np.einsum("be,bd->ed", env, dL_da0)
    grads.bg0 += np.sum(dL_da0, axis=0)

    return total_loss, grads, diagnostics


def gradient_check(params: StateSpaceParams, env: np.ndarray, t_len: int, loss_and_grad_fn, eps: float = 1e-5, n_params_to_check: int = 12, seed: int = 0) -> dict[str, object]:
    """Finite-difference check of the hand-derived backward pass.
    loss_and_grad_fn(params) -> (loss, grads, ...) -- any extra return values
    (e.g. diagnostics) are ignored."""
    rng = np.random.default_rng(seed)
    base_loss, grads = loss_and_grad_fn(params)[:2]
    checks = []
    flat_specs: list[tuple[str, tuple]] = []
    for name in ("Wg0", "bg0", "Wz", "We", "b1", "Wo", "bo"):
        arr = getattr(params, name)
        for _ in range(max(1, n_params_to_check // 7)):
            idx = tuple(int(rng.integers(0, s)) for s in arr.shape)
            flat_specs.append((name, idx))
    for name, idx in flat_specs:
        arr = getattr(params, name)
        orig = arr[idx]
        arr[idx] = orig + eps
        loss_plus = loss_and_grad_fn(params)[0]
        arr[idx] = orig - eps
        loss_minus = loss_and_grad_fn(params)[0]
        arr[idx] = orig
        numeric_grad = (loss_plus - loss_minus) / (2 * eps)
        analytic_grad = float(getattr(grads, name)[idx])
        checks.append({"param": name, "index": str(idx), "numeric_grad": numeric_grad, "analytic_grad": analytic_grad, "abs_diff": abs(numeric_grad - analytic_grad)})
    max_abs_diff = max(c["abs_diff"] for c in checks)
    denom = max(1e-6, max(abs(c["numeric_grad"]) for c in checks))
    return {"base_loss": base_loss, "checks": checks, "max_abs_diff": max_abs_diff, "relative_max_abs_diff": max_abs_diff / denom, "passed": bool(max_abs_diff / denom < 0.03)}


class AdamState:
    def __init__(self, params: StateSpaceParams, lr: float = 0.02, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = _zero_like(params)
        self.v = _zero_like(params)

    def step(self, params: StateSpaceParams, grads: StateSpaceParams) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        def upd(p_arr, g_arr, m_arr, v_arr):
            m_arr[:] = self.beta1 * m_arr + (1 - self.beta1) * g_arr
            v_arr[:] = self.beta2 * v_arr + (1 - self.beta2) * (g_arr**2)
            p_arr[:] = p_arr - lr_t * m_arr / (np.sqrt(v_arr) + self.eps)

        for name in ("Wg0", "bg0", "Wz", "We", "b1", "Wo", "bo"):
            upd(getattr(params, name), getattr(grads, name), getattr(self.m, name), getattr(self.v, name))
        for col in params.heads:
            pw, pb = params.heads[col]
            gw, gb = grads.heads[col]
            mw, mb = self.m.heads[col]
            vw, vb = self.v.heads[col]
            upd(pw, gw, mw, vw)
            upd(pb, gb, mb, vb)
        if params.Wc is not None:
            upd(params.Wc, grads.Wc, self.m.Wc, self.v.Wc)
            upd(params.bc, grads.bc, self.m.bc, self.v.bc)


def params_fingerprint(params: StateSpaceParams) -> str:
    h = hashlib.sha256()
    for name in ("Wg0", "bg0", "Wz", "We", "b1", "Wo", "bo"):
        h.update(np.ascontiguousarray(getattr(params, name)).tobytes())
    for col in sorted(params.heads):
        w, b = params.heads[col]
        h.update(np.ascontiguousarray(w).tobytes())
        h.update(np.ascontiguousarray(b).tobytes())
    if params.Wc is not None:
        h.update(np.ascontiguousarray(params.Wc).tobytes())
        h.update(np.ascontiguousarray(params.bc).tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, params: StateSpaceParams, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"Wg0": params.Wg0, "bg0": params.bg0, "Wz": params.Wz, "We": params.We, "b1": params.b1, "Wo": params.Wo, "bo": params.bo}
    for col, (w, b) in params.heads.items():
        arrays[f"head__{col}__W"] = w
        arrays[f"head__{col}__b"] = b
    if params.Wc is not None:
        arrays["Wc"] = params.Wc
        arrays["bc"] = params.bc
    meta = dict(meta)
    meta["params_fingerprint"] = params_fingerprint(params)
    arrays["metadata_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: Path) -> tuple[StateSpaceParams, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    heads = {}
    for key in data.files:
        if key.startswith("head__") and key.endswith("__W"):
            col = key[len("head__"):-len("__W")]
            heads[col] = (data[key], data[f"head__{col}__b"])
    params = StateSpaceParams(
        Wg0=data["Wg0"], bg0=data["bg0"], Wz=data["Wz"], We=data["We"], b1=data["b1"], Wo=data["Wo"], bo=data["bo"],
        heads=heads, Wc=data["Wc"] if "Wc" in data.files else None, bc=data["bc"] if "bc" in data.files else None,
    )
    return params, meta
