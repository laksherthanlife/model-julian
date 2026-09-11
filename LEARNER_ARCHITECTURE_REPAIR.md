# Learner Architecture Repair — Audit, Repair, and Validation

Companion to `GENERATOR_VALIDITY.md` and `LATENT_CAPACITY_MISMATCH_DESIGN.md`.
This document now covers five passes: (1) repairing the production
teacher's untrained-random-feature architecture, (2) replacing the
prospective DBTL virtual scorer's frozen lookup table with a scorer that
actually runs the repaired model (Sections H-J), (3) the **powered**
clean-successor prospective conventional-vs-hybrid rerun using that repaired
scorer (Section K), (4) replacing the GSM surrogate that Pass 2/3's
scorer runs *through*, after isolating it as the dominant source of
candidate-ranking error (Section L), and (5) a **second powered rerun**
using the Pass-4 checkpoint/surrogate plus a new exact-Yeast9 reranking
step, which reverses Pass 3's inconclusive result into a clean, significant
hybrid win (Section M, new). All 12 learner-side acceptance gates (Section
I) pass. **The powered latent-capacity-mismatch benchmark was not run**
(still deprioritized) -- the powered *prospective DBTL* benchmark was run
twice (Pass 3, Pass 5), each its own clean successor. The original
prospective benchmark's `data/`/`results/prospective_dbtl_benchmark/`
artifacts remain completely untouched -- neither pass writes to the
original's files. Passes 4-5 do not touch the physiology learner's
training-information contract (Section C) -- Pass 4 replaces only the
frozen metabolic surrogate the learner is trained through (Section L), and
Pass 5 adds only an exact-Yeast9 reranking step between the existing
virtual screen and biological verification (Section M).

---

## A. Architecture audit

The production teacher (`run_reporter_grounded_hybrid_distillation.py`) and
its hybrid "student" (same file, `train_student_checkpoint`) both use the
same closed-form architecture:

```python
z0 = tanh(env @ Wg + bg)
z[k+1] = z[k] + dt * tanh(z[k] @ A + env @ C + bf)
head_col(z) = weighted_ridge_fit([1, z, z^2, time], target_col)   # closed form, per column, independent
```

`initialize_state_space` draws `A, Wg, C, bf` once from a fixed seed
(`rng.normal`) — **they are never updated by gradient descent or any other
training procedure.** Only the linear/quadratic output heads are fit, each
independently, via closed-form ridge regression. Consequently:

- The latent trajectory `z` is a fixed random nonlinear feature map of
  `environment` and `seed` alone — it carries no information from *any*
  training target, reporters included.
- Each output head is an independent regression on those same fixed
  features. There is no shared computation for the model to route reporter
  information through into product prediction, because there is nothing
  trainable upstream of the heads at all.
- This is the exact, provable cause of the reporter-invariance finding from
  the prior pass (`primary_rmse == shuffled_rmse == random_rmse` to floating
  point): shuffling or removing a target column changes only *that column's*
  ridge fit; it cannot touch any other column's fit, because the shared
  representation they'd need to interact through does not exist.
- Separately (prior pass, unchanged finding): `INTERFACE_COLUMNS`/`FLUX_COLUMNS`
  supervise this same architecture on raw `z_ox, z_atp, z_bottle` and internal
  LP fluxes — privileged simulator information, addressed by the existing
  clean contract (`gem_clean_teacher_targets.py`, preserved unchanged in this
  pass).

**Existing infrastructure reused, not reinvented:** the hybrid-student path
already had a mechanism to push predicted controls into real Yeast9 --
`apply_predicted_interface_controls(model, cfg, controls)`
(`run_reporter_grounded_hybrid_distillation.py:1204-1229`) takes only a
6-key compact control dict (`oxygen_lower_bound, atp_maintenance_lower_bound,
gamma_growth_fraction, PSY/DES/CYC_effective_upper_bound` —
`GEM_APPLIED_CONTROL_COLUMNS`, no `z_*` input at all) and applies it to a
real GEM before `solve_staged`. This function needed no modification and is
reused as-is for the new G7 validation (Section G below).

---

## B. Repaired model

New module `src/yeast_validation/gem_trainable_state_space.py`. Every weight is trained by
gradient descent; there is no autodiff library in this environment (checked:
no torch/jax/tensorflow installed), so the backward pass is hand-derived
backpropagation-through-time (BPTT) — auditable by construction, and
verified correct by finite-difference gradient checks (Section D).

```
z[0]    = tanh(env @ Wg0 + bg0)                          # trainable initial-state encoder
h[t]    = tanh(z[t] @ Wz + env @ We + b1)                 # trainable transition hidden layer
dz[t]   = tanh(h[t] @ Wo + bo)                             # bounded state derivative
z[t+1]  = clip(z[t] + dt_latent * dz[t], -12, 12)          # residual/Euler latent step

y_col[t] = z[t] @ Wh[col] + bh[col]      for col in {reporters, direct product/biomass (non-hybrid variants only)}
c[t]     = (z[t] @ Wc + bc) * control_std + control_mean   for t in 0..T-2  (hybrid variant only)
```

`D_model = 16` throughout (frozen, per Section 5 of the handover — continuity
with the previously-selected teacher latent dimension; **not** changed in
this pass). The transition hidden width `H = 24` is a separate, smaller,
disclosed architecture choice, not tuned against the mismatch experiment or
against which variant "wins." Total trainable parameters ≈ 1,200 — deliberately
the smallest version of Section 4's requested architecture, not a neural ODE
or GRU (neither was judged necessary once BPTT through even this small
residual-MLP transition was confirmed to work and to help — see Section E).

Every head reads the same `z[t]`, so any head's loss gradient flows back
through that head's weights into `z[t]`, and from there through the shared
transition weights (`Wz, We, Wo, b1, bo`) and the initial encoder
(`Wg0, bg0`) for every earlier time step via standard BPTT — this is the
structural fix, not a training-recipe change.

---

## C. Training contract

Unchanged from the prior pass's clean contract (`gem_clean_teacher_targets.py`),
reused, not reimplemented:

- **Deployment inputs:** `temperature, pH, DO` only.
- **Supervision:** `B_total_observed`, `X_observed` (noisy, from
  `gem_observation_noise.py`), plus reporters `R_ox, R_atp, R_E_PSY, R_E_DES,
  R_E_CYC` (partial coverage — no `R_bottle` exists, matching the
  pre-existing partial-observability design).
- **Explicitly forbidden and never used as a training target anywhere in
  the new code:** raw `z_ox/z_atp/z_bottle/z_er`, the discrete regulator
  state, `PSY_effective_upper_bound`/`DES_.../CYC_.../oxygen_lower_bound/
  atp_maintenance_lower_bound/gamma_growth_fraction` (the same
  `CONTROL_COLUMNS` the control head *outputs* are never given as a direct
  loss target — see Section D for how they are trained instead), internal
  LP fluxes (`ggpp_flux, PSY_flux, DES_flux, CYC_flux, atp_maintenance_flux`).
  `tests/test_clean_teacher_no_leakage.py` (unchanged) continues to enforce
  this for the shared `gem_clean_teacher_targets` contract; the new training
  driver (`run_repaired_hybrid_training.py`) only ever constructs
  `head_targets` from `{B_total_observed, X_observed} ∪ REPORTER_COLS` —
  inspectable directly in the code, no other column is ever passed to
  `compute_loss_and_grads`'s `head_targets` argument.

---

## D. Reporter pathway — mathematical and empirical proof

**Mathematical proof (gradient check, not just a before/after comparison):**
`gem_trainable_state_space.gradient_check` finite-differences the loss with
respect to every transition/encoder weight and compares to the hand-derived
analytic gradient. With a single reporter-style head as the *only* loss term:

```
relative_max_abs_diff = 2.29e-10   (direct-head path)
relative_max_abs_diff ≈ 1e-9        (full hybrid path: control head -> frozen surrogate -> mechanistic integration -> loss)
```

i.e. the reporter loss's gradient into `Wz, We, Wo, b1, bo, Wg0, bg0` is
correct to ~10 significant figures, not merely nonzero. `tests/
test_repaired_hybrid_gates.py::test_g2_reporter_only_loss_gradient_check_passes`
pins this.

**Empirical proof (does training behavior actually change):**

| reporter_mode | R_ox validation normalized RMSE | training loss trajectory |
|---|---:|---|
| full | 0.070 | converges to the reporter target's real (non-shuffled) structure |
| shuffled | 0.105 | measurably worse fit to the *real* R_ox (49% relative degradation) — the model can tell |

`params_fingerprint` (SHA-256 of every weight array) differs between every
`full` vs `shuffled` run, at every seed (`tests/
test_repaired_hybrid_gates.py::test_g9...`). This is the qualitative
difference from the old architecture: there, shuffling produced *zero*
change in anything (mathematically impossible for it to do otherwise); here,
shuffling produces a real, measurable, mechanically-explained change in both
the learned parameters and the reporter-reconstruction quality.

---

## E. Baseline results (clean G3, `D_true≈3`, 125-culture real-Yeast9 dataset)

5 seeds each (`11,22,33,44,55`), `D_model=16`, `H=24`, 600 epochs, Adam
lr=0.02. Full table: `data/repaired_hybrid_training_metrics.csv`.

**Product, validation-split normalized RMSE, by variant:**

| Variant | train | validation | interpolation | heldout_combination | extrapolation |
|---|---:|---:|---:|---:|---:|
| `direct_blackbox` (no dynamics) | 0.336 | 0.387 | 0.371 | 0.392 | 0.448 |
| `product_only_state_space` (trainable z, direct head) | 0.107 | 0.119 | 0.107 | 0.110 | 0.124 |
| `hybrid_learned_physiology` (trainable z, product via surrogate) | 0.383 | 0.424 | 0.407 | 0.388 | 0.435 |

**Biomass, same layout:**

| Variant | train | validation | interpolation | heldout_combination | extrapolation |
|---|---:|---:|---:|---:|---:|
| `direct_blackbox` | 0.429 | 0.495 | 0.478 | 0.526 | 0.659 |
| `product_only_state_space` | 0.088 | 0.098 | 0.104 | 0.106 | 0.146 |
| `hybrid_learned_physiology` | 0.121 | 0.129 | 0.132 | 0.120 | 0.151 |

**Two findings worth stating plainly, in both directions:**

1. **Trainable dynamics matter, a lot.** `product_only_state_space` beats
   `direct_blackbox` by ~3x on product and ~4-5x on biomass, at *every*
   split including the two OOD ones (`heldout_combination`,
   `extrapolation`). This directly answers Section 10/13's negative-control
   question: dynamical structure is not decorative here, it is carrying real
   predictive weight beyond what `(environment, time)` alone determines.
2. **The genuinely hybrid path costs accuracy relative to a direct head, and
   is honestly reported as such, not concealed.** `hybrid_learned_physiology`
   routes product only through `c(t) -> frozen surrogate -> mechanistic
   integration`, and its product NRMSE (~0.38-0.44) is *worse* than even
   `direct_blackbox` (~0.39-0.45 is close, but `product_only_state_space` at
   ~0.11-0.12 is clearly better than both). This is exactly Section 10's
   warning made concrete: "a pure product predictor may have excellent RMSE
   but does not validate the hybrid architecture" — here the reverse is also
   true, the hybrid architecture's *product* RMSE is not (yet) its strong
   suit, precisely because it is constrained to route through a real,
   imperfect mechanistic surrogate rather than fitting product shape
   directly. Biomass is barely affected (0.129 vs 0.098, both good) because
   the surrogate's biomass_flux channel is much more accurate
   (0.028 held-out NRMSE) than its beta_carotene_flux channel
   (0.129 held-out NRMSE) — the accuracy cost is concentrated exactly where
   the surrogate itself is weakest, which is itself informative and traceable.

Reporters (`R_ox, R_atp, R_E_PSY, R_E_DES, R_E_CYC`) as direct heads on the
hybrid model's shared `z(t)`: validation NRMSE 0.061-0.073 — the model
represents them well; Section F addresses whether that helps the parts that
matter.

---

## F. Reporter ablation (full / none / shuffled / random)

`hybrid_learned_physiology`, 5 seeds each, product/biomass NRMSE by
`reporter_mode` (validation split, mean over seeds):

| reporter_mode | product NRMSE | biomass NRMSE | R_ox NRMSE (where a head exists) |
|---|---:|---:|---:|
| full | 0.4284 | 0.1292 | 0.070 |
| none | 0.4247 | 0.1292 | n/a (no head) |
| shuffled | 0.4254 | 0.1292 | 0.105 |
| random | 0.4208 | 0.1288 | n/a (fit to `R_random_smooth` instead) |

**Honest reading, per Section 12's explicit instruction not to force a
reporter-positive result:** product and biomass prediction are essentially
unaffected by reporter mode (differences are within run-to-run noise; `none`
is marginally *better* than `full`, not worse). Reporters do not currently
help downstream product/biomass prediction in this architecture, on this
dataset. **This is a meaningful negative result, not the previous pass's
"the experiment couldn't detect anything" result** — Section D already
proved the reporter loss mechanically reaches and changes the shared
dynamics (G2, G9); this section shows that, having done so, it does not
propagate into an improvement on the specific downstream channels tested
here. A plausible reason (not yet tested): the reporters' informative content
about `z_ox/z_atp/E_*` may simply not be the limiting factor for
environment-driven product/biomass variation in this particular 3-burden-state,
single-strain, environment-only dataset — worth revisiting once the
mismatch generator's higher-`D_true` conditions exist (more burden states
whose product-relevant information genuinely isn't recoverable from
environment alone would be a more demanding test of whether reporters can
help).

---

## G. Hybrid interface — how `z(t)` reaches Yeast9

```
z(t) --[Wc, bc]--> c(t) = (oxygen_lower_bound, atp_maintenance_lower_bound,
                            gamma_growth_fraction, PSY/DES/CYC_effective_upper_bound)
     --[training]--> frozen differentiable surrogate (gem_gsm_surrogate.py)
                      --> predicted biomass_flux, beta_carotene_flux
                      --> one-step teacher-forced mechanistic integration
                          (dX = dt*flux*X, matching the real generator's own ODE)
                      --> loss vs. real observed X(t+1), B(t+1)
     --[deployment/validation]--> apply_predicted_interface_controls(model, cfg, c(t))
                      --> REAL Yeast9 solve_staged (LP, exact)
```

The surrogate (`gem_gsm_surrogate.py`) is a closed-form ridge fit on a fixed
polynomial feature map of `(c, environment)`, trained on 3,696 real exact
LP-solve interval records (train split of the existing 125-culture dataset)
and frozen before the physiology model trains on it. Held-out normalized
RMSE: `biomass_flux` 0.028, `beta_carotene_flux` 0.129, `oxygen_uptake`
0.011, `atp_maintenance_flux` 0.002 (`ggpp_flux/PSY_flux/DES_flux` are
numerically identical to `beta_carotene_flux` — a real mechanistic fact, not
a bug: the installed pathway is an unbranched linear chain, so pFBA gives
equal flux through every step at steady state). Its Jacobian
`d(output)/d(c)` is exact (closed-form derivative of a polynomial-feature
ridge model), not a finite-difference approximation — this is what makes the
whole pathway genuinely differentiable end to end.

`apply_predicted_interface_controls` (reused unmodified from the existing
hybrid-student code) is the only place `c(t)` ever touches Yeast9, and it
never receives `z_*` or any other privileged quantity as input.

**Live validation against the real solver (`run_hybrid_gsm_replay_validation.py`,
G7):**

- 4 held-out-split cultures, `c(t)` from the trained `hybrid_learned_physiology`
  checkpoint, replayed through real `solve_staged`: cross-culture output
  spread `std(biomass_flux)=0.0199`, `std(beta_carotene_flux)=0.0180` — the
  model is not predicting a constant control regardless of environment.
- Direct causal-manipulation check: holding one culture's predicted `c(t)`
  fixed except `PSY_effective_upper_bound × 0.3`, real
  `beta_carotene_flux` moves from `0.1294` to `0.0518` (Δ=0.0776, a 60%
  drop) — solved by the actual GLPK/COBRA backend, not a surrogate.

Both checks are recorded in `data/hybrid_gsm_replay_gate.csv`
(`tests/test_repaired_hybrid_gates.py::test_g7...` asserts both `passed=True`).

---

## H. Virtual-scoring audit and repair (this pass)

### H.1 What changed

`run_prospective_dbtl_benchmark.py::hybrid_virtual_score` (lines 700-713) is
**unmodified** — it remains as historical evidence of the original
benchmark's contract, per the documentation policy (do not overwrite the
original result). A new, standalone module,
`src/yeast_validation/gem_repaired_virtual_scorer.py`, implements the replacement
scoring path and is what the new dry run actually uses:

```
candidate edits + (T, pH, DO)
    -> repaired trainable z(t)  (gem_trainable_state_space, D_model=16, frozen checkpoint)
    -> predicted control vector c(t)  (6-dim: oxygen_lower_bound, atp_maintenance_lower_bound,
       gamma_growth_fraction, PSY/DES/CYC_effective_upper_bound)
    -> candidate edits applied to c(t) using the SAME bound-transform semantics as
       deployment_benchmark.apply_sparse_edits_to_model (scaled_positive_upper /
       scaled_negative_lower, mirrored in apply_edits_to_controls)
    -> disclosed safety clip to +/-4 std of the surrogate's training distribution
       (prevents a real failure mode found during testing -- see H.3)
    -> frozen GSM surrogate (gem_gsm_surrogate) -> predicted biomass/beta-carotene flux
    -> one-step mechanistic integration (same ODE the real generator uses: dX=mu*X*dt, dB=v*X*dt-k*B*dt)
    -> final_product (== observed_score's objective)
```

### H.2 Strain-edit integration (audited, not assumed)

Audited `deployment_benchmark.apply_sparse_edits_to_model` (the function the
*exact* simulator itself uses for prospective candidates) and
`data/final_dbtl_actionable_intervention_library.csv` directly. The 24 frozen
interventions target 7 distinct reactions; **4 of those 7 (16/24
interventions, `BETA_PHYTOENE_SYNTHASE`, `BETA_PHYTOENE_DESATURASE`,
`BETA_LYCOPENE_CYCLASE`, `r_1992`/oxygen exchange) map directly onto the
existing 6-dim control vector** and are applied with the exact-simulator's
own bound-transform rules. The remaining 3 reactions (`r_0461` native GGPP,
`r_1714` glucose exchange, `r_0773` NADH:ubiquinone knockdown, 8/24
interventions) have no corresponding control dimension and are **skipped
with an explicit logged note** (`skipped_uncovered_reaction:<id>`), never
silently dropped -- this is the disclosed "simplest defensible version"
Section 2 of the handover asked for. Across the 200-candidate pool used for
Step 3, 76.5% of candidates had at least one edit actually applied and 49.5%
had at least one edit skipped.

### H.3 A real surrogate-extrapolation failure found and disclosed-fixed

Testing an edit-heavy candidate (PSY capacity x2.0, ~12 std outside the
surrogate's training range) found the polynomial surrogate's quadratic term
flips sign under extreme extrapolation, predicting negative flux -- which the
mechanistic integration's `max(0, ...)` then locks at exactly 0 for the rest
of the trajectory. Fixed with a disclosed safety clip
(`CONTROL_CLIP_STDS=4.0`, same spirit as the existing `predict_columns`
min/max-clip pattern elsewhere in this codebase) -- not silent tuning, and
not a complete fix (see Step 4 below: a second, subtler issue remains where
single-enzyme edits decorrelate the control vector from the training
distribution's natural PSY/DES/CYC covariance, degrading surrogate quality
without triggering the sign-flip). Across the 200-candidate pool, a mean of
112/288 control values per candidate (~39%) required clipping -- the
physiology model's own unedited baseline predictions are frequently outside
+/-4 std of the surrogate's training support, a calibration gap worth
narrowing in a future pass, not hidden here.

### H.4 Step 2 — the scorer genuinely depends on the model

`tests/test_repaired_virtual_scorer.py` (8/8 passing) proves mechanically:
different checkpoints give different scores; different environments give
different scores; edits change the score and are logged; an uncovered edit
is skipped (logged) and does not change the score; perturbing the control
head's weights changes the score (confirms the live forward pass drives it,
not a cache); scoring is reproducible; and — read directly from source, not
asserted — the scorer module's *code* (excluding its own docstring, which
names what it replaces) never references `z_ox/z_atp/z_bottle`,
`teacher_pseudodata`, `teacher_lookup`, `prospective_exact_simulator_call_ledger`,
or nearest-neighbour lookup of any kind.

### H.5 Step 3 — exact-vs-virtual ranking validation

Pool: 200 candidates (`CandidateFactory.sample_candidate`, up to 3 edits
each, `baseline_world`). Virtual-scored locally (no LP). Stratified subset
of 24 (top-6/middle-6/bottom-6/random-6 by virtual score) exactly verified on
**Vanda** (PBS array, `Python/3.12.3-GCCcore-13.3.0` + project `.venv`,
reusing the existing deployed `run_prospective_exact_worker.py` unmodified —
no new code needed on the exact-verification side).

| Metric | Result |
|---|---|
| Spearman rank correlation (virtual score vs. exact `final_product`) | **0.590** |
| Top-8 enrichment (virtual top-8 ∩ exact top-8) | **5/8 = 0.625** |
| Top-8 exact regret (best possible − best found via virtual top-8) | **0.0000** (the single best exact candidate, 1.934, is IN the virtual top-8) |
| Mean exact `final_product`, virtual top-8 | 0.805 |
| Mean exact `final_product`, all 24 | 0.541 |
| Mean exact `final_product`, virtual bottom-8 | 0.354 |
| RMSE(virtual_score, exact `final_product`) | 1.094 (std of exact = 0.368 — **absolute score is poorly calibrated**) |

Read honestly: the virtual scorer's *ranking* carries real, usable signal
(moderate-strong rank correlation, clear top/bottom separation, zero regret
on the single best candidate) even though its *absolute magnitude* is not
calibrated to true `final_product` units. This matches the handover's own
framing — ranking quality matters more than trajectory RMSE at this stage —
and is a credible, though imperfect, result: not a rubber stamp.

### H.6 Step 4 — surrogate vs. real-GSM error isolation

For the same 24 candidates, replayed the physiology model's predicted
(edit-adjusted, clipped) `c(t)` through **real Yeast9**
(`run_repaired_controls_real_gsm_replay_worker.py`, reusing
`apply_predicted_interface_controls` unmodified) instead of the surrogate,
holding physiology + edit-mapping fixed. This isolates each error source:

| Comparison | RMSE | Isolates |
|---|---:|---|
| surrogate-path `final_product` vs. same-controls-via-real-GSM | **0.528** | surrogate approximation error alone |
| same-controls-via-real-GSM vs. true independent exact rollout | **1.273** | learned-physiology + edit-mapping error (vs. the generator's own dynamics) |
| surrogate-path vs. true independent exact rollout | 1.094 | total end-to-end error |

(0 infeasible intervals in the real-GSM replay across all 24 candidates — the
predicted controls are always solvable.) **The dominant error source is not
the surrogate.** The surrogate's own approximation error (0.528) is smaller
than the gap between "predicted physiology, correctly applied to the real
solver" and "what the true generator's own dynamic regulatory system would
have done" (1.273). This is expected and disclosed, not swept under the rug:
the physiology model was trained only on the reference (no-edit) strain's
environment-driven dynamics, and edits are applied to its *output* via a
declared, static bound-transform — it was never trained to anticipate how
the true generator's own feedback loop (burden -> regulator -> capacity)
would respond to an edited strain. Narrowing this gap (not the surrogate) is
the highest-value target for a future pass.

### H.7 Step 5 — small paired DBTL dry run

Budget-matched at 12 exact cultures per arm, `baseline_world`, single seed.

- **Repaired-hybrid arm**: no new exact compute spent — reuses Step 3's
  already-verified 24 candidates, taking the top-12 by virtual score (a
  literal budget-matched subset, not a re-run). Best exact `final_product`
  found: **1.934**, mean **0.740**.
- **Conventional arm**: `src/yeast_validation/run_repaired_dbtl_dry_run.py`, a genuine
  sequential 3-stage x 4-candidate adaptive campaign (`propose_conventional_batch`
  + `exact_sparse_dynamic_rollout`, both reused unmodified from
  `run_prospective_dbtl_benchmark.py`/`run_deployment_verifier.py`), run
  in-process on Vanda (PBS, single job, sequential because each stage's
  proposals depend on the previous stage's real observations).

| Arm | n exact cultures | best `final_product` | mean `final_product` | all feasible? |
|---|---:|---:|---:|---|
| Repaired-hybrid (top-12 by virtual score) | 12 | **1.934** | **0.740** | yes |
| Conventional adaptive (3 stages x 4) | 12 | 0.817 | 0.403 | yes |

At this matched 12-culture budget, the hybrid arm's best-found candidate is
2.4x the conventional arm's, and its mean is ~1.8x higher — qualitatively
consistent with the direction of the original (leaky-scorer) headline
result, now reproduced end to end through a live, leak-free, model-driven
virtual search. This is a **dry run for mechanics and compute estimation,
not a powered result** (n=1 seed, budget=12, one world): it confirms the
causal ordering (propose -> verify -> observe -> re-propose) executes
correctly, that exact-query accounting closes (12 candidates -> 12
`exact_sparse_dynamic_rollout` calls, all feasible, matching
`n_actual_lp_solves` conventions used elsewhere in this codebase), that the
repaired model is the thing actually producing the hybrid arm's candidate
ranking (Section H.4-H.5), and gives a wall-clock basis for a future powered
design: 24 exact cultures (Step 3, PBS array, concurrency 6) completed in
~15 minutes wall-clock; 12 *sequential* exact cultures (this step, single
job, necessarily serial because each stage depends on the previous stage's
observations) took ~24 minutes wall-clock (~2 min/culture) on a single Vanda
CPU core.

---

## I. Acceptance table

| Gate | Status | Evidence |
|---|---|---|
| G1 — Trainable dynamics | **PASS** | `test_g1...`: different training targets -> different `Wz`, different fingerprint |
| G2 — Reporter gradient | **PASS** | Gradient check ~1e-10 relative error, reporter-only loss; `Wz`/`Wo` gradients nonzero |
| G3 — No privileged leakage | **PASS** | `gem_clean_teacher_targets` contract reused unchanged; `test_clean_teacher_no_leakage.py` (5/5); scorer code contains none of `z_ox/z_atp/z_bottle`/`teacher_pseudodata`/`teacher_lookup` |
| G4 — Temporal dependence | **PASS** | `z(t)` genuinely evolves (mean step-to-step change > 1e-4); trainable-dynamics variant beats direct env+time regression at every split |
| G5 — Predictive signal | **PASS** | All variants beat normalized_rmse=1.0 baseline on validation |
| G6 — OOD evaluation | **PASS** | `heldout_combination`/`extrapolation` splits explicitly measured for every variant |
| G7 — Real hybrid coupling | **PASS** | Live Yeast9 replay: cross-culture output variation + direct control-perturbation causal check, both against the real solver |
| G8 — Live model virtual score | **PASS (this pass)** | `gem_repaired_virtual_scorer.py` executes the repaired model live for every candidate; `tests/test_repaired_virtual_scorer.py` (8/8) proves checkpoint/environment/edit/physiology-dependence and absence of any pseudodata lookup |
| G9 — Reporter ablation works mechanically | **PASS** | Full vs. shuffled: different fingerprints, 49% relative difference in reporter-reconstruction NRMSE |
| G10 — Reproducibility | **PASS** | All checkpoints carry seed/variant/reporter_mode/`D_model`/fingerprint; every manifest row resolves to a matching, loadable checkpoint |
| G11 — Exact-ranking usefulness (new this pass) | **PASS (credible, not strong)** | Spearman 0.590, top-8 enrichment 0.625, zero top-8 regret on n=24 |
| G12 — Edit integration (new this pass) | **PASS (partial coverage, disclosed)** | 4/7 frozen-library reactions covered via declared bound-transform semantics; remainder skipped-and-logged, not silently dropped |

**All gates now pass** (G11/G12 pass with disclosed caveats: ranking is
credible but not strong, and edit coverage is partial). This clears the
learner-side blocker from the prior pass. **The generator-side causal-rank
gate from `LATENT_CAPACITY_MISMATCH_DESIGN.md` was not re-attempted in this
pass and is still failing** — see Section J.

---

## J. Next-step recommendation

**The learner gate is now fully cleared** (G1-G12 all PASS, Section I). One
blocker remains before the powered latent-capacity mismatch benchmark:

**Generator gate — not re-attempted this pass, still failing per
`LATENT_CAPACITY_MISMATCH_DESIGN.md`.** The controller-family library's
tier-2 states are mathematically collinear with their tier-1 siblings
(shared-single-bound coupling is provably rank-1), and even the core
`z_ox`/`z_bottle` pair showed ~0.9999 collinearity at the tested operating
points. No controller-family work was done in this pass — it was explicitly
scoped as "later," and the priority (repair the learner first) is now done.

**Recommended order for the next pass:**

1. **Freeze the learner now** per the updated freeze protocol below — the
   learner gate does not need to wait on the generator gate; they are
   independent, and re-litigating the learner after the generator redesign
   would violate "do not retune these based on D_true."
2. Optionally, before freezing, narrow the dominant error source found in
   H.6 (learned-physiology + edit-mapping gap, RMSE 1.273, larger than the
   surrogate's own 0.528) -- likely by training the physiology model on a
   wider variety of edited-strain trajectories rather than only the
   reference strain, or by giving the edit-mapping access to more than a
   static bound-transform. Not required to freeze, but the single highest-
   value target if more learner-side work is wanted before moving on.
3. Return to the controller-family redesign
   (`LATENT_CAPACITY_MISMATCH_DESIGN.md` Section 7.1): distinct GSM coupling
   targets for tier-2, and an investigation of whether the `z_ox`/`z_bottle`
   collinearity is specific to the tested operating point (a sweep across
   DO/burden levels/regulator states was recommended there and still applies).
4. Re-run the causal-rank gate; only proceed to a powered run once *both*
   gates hold simultaneously, per Section 19 of the learner-repair handover.

Do not run the powered benchmark before the generator gate is also green.

---

## Freeze protocol (Section 16) — ready to execute; not yet executed

All learner-side gates pass (Section I); the generator-side gate does not
(Section J). Recording the frozen configuration here now, so it is not
retuned later against the generator redesign, per Section 16's explicit
instruction. Freezing itself (writing a dedicated `*_freeze_manifest.csv`,
the way the generator side already does) is a short follow-up action, not
done in this pass because the explicit instruction was to stop before the
powered experiment.

- Architecture: `gem_trainable_state_space.py` (residual-MLP transition, `H=24`).
- `D_model = 16` (unchanged).
- Optimizer: Adam, `lr=0.02`, `beta1=0.9, beta2=0.999`.
- Epochs: 600 (no early stopping used in this pass — worth reconsidering at freeze time based on loss curves in `results/repaired_hybrid_training/`).
- Loss weights: reporter/product/biomass terms currently unweighted (`lambda=1.0` each) plus `lambda_dyn=2e-4`; not yet tuned or justified beyond "stable and converges" — flag for review at freeze time.
- Reporter interface: `R_ox, R_atp, R_E_PSY, R_E_DES, R_E_CYC` as direct heads on shared `z(t)`.
- Output heads: linear (no hidden layer) on `z(t)` for reporters/direct-variant product/biomass; `Wc/bc` linear control head for the hybrid variant.
- Training split rules: existing `train/validation/interpolation/heldout_combination/extrapolation` split manifest, unchanged.
- Random seeds: `MODEL_SEEDS = [11, 22, 33, 44, 55]` (unchanged); dry-run/pool seed `SEED=4021` (Step 3), conventional dry-run seed `SEED=9101` (Step 5) — both scoped to validation, not part of the frozen training config.
- GSM interface: `gem_gsm_surrogate.py` (ridge on `[1, c, c^2, env, env^2]`, `ridge_lambda=1.0`), frozen from the train split before physiology training; `apply_predicted_interface_controls` for real-solver replay.
- Candidate-edit mapping: `gem_repaired_virtual_scorer.REACTION_TO_CONTROL` (4 of 7 frozen-library reactions covered), `apply_edits_to_controls` (bound-transform semantics mirrored from `deployment_benchmark.apply_sparse_edits_to_model`), `CONTROL_CLIP_STDS = 4.0` safety clip.
- Virtual scoring objective: `gem_repaired_virtual_scorer.RepairedVirtualScorer.score` -> `final_product` (same objective as `run_prospective_dbtl_benchmark.observed_score`).
- Deployment inputs: `temperature, pH, DO` only (unchanged).
- Exact verification protocol: `run_deployment_verifier.exact_sparse_dynamic_rollout`, `baseline_world`, unchanged, reused as-is.
- Checkpoint used for all Step 3-5 validation: `results/repaired_hybrid_training/checkpoints/hybrid_learned_physiology__full__seed11.npz` (`params_fingerprint` in the file's metadata).

---

## K. Powered clean-successor prospective DBTL rerun

Tests: *can the repaired pretrained hybrid digital twin discover better
strain-environment designs using fewer prospective exact cultures and fewer
sequential biological stages than conventional adaptive DBTL?* The clean
successor to the original prospective benchmark, same question, repaired
leak-free contract.

### K.1 Common edit domain (resolved before freezing)

The frozen 24-intervention library targets 7 reactions; only 4 (16
interventions) have a corresponding dimension in the repaired scorer's
control vector (Section H.2). Extending true coverage to the other 3 would
need new exact training data the historical dataset doesn't have (real
redesign) -- so, per instruction, Option 2: a **restricted common
intervention domain**, `data/repaired_benchmark_common_intervention_library.csv`
(16 interventions, `BETA_PHYTOENE_SYNTHASE/DESATURASE`,
`BETA_LYCOPENE_CYCLASE`, `r_1992` oxygen exchange only), used identically by
**both** arms -- `CandidateFactory` and the virtual scorer draw from the same
restricted table, so no edit is ever silently skipped in the powered run and
the candidate domain is provably identical between workflows.

### K.2 Reuse strategy (not a rewrite)

`run_prospective_dbtl_benchmark.py` (frozen) is reused almost entirely
unmodified via runtime monkey-patch, never file edits:
`bench.hybrid_virtual_score` -> `RepairedVirtualScorer.score`;
`bench.DATA/RESULTS/ROLLOUTS` -> a fresh isolated directory per campaign
(`results/repaired_prospective_campaigns/<campaign_id>/`); `bench.BENCHMARK_ID`
-> `repaired_hybrid_prospective_dbtl_v1`. `run_campaign`,
`hybrid_virtual_search`, `propose_conventional_batch`, every audit-ledger
writer, and -- for aggregation -- `write_analysis()` itself (paired
bootstrap stats, leakage audit, AUC tables) all run completely unmodified.
This is what "prefer matching the original benchmark structure" meant in
practice: same code, different scorer and I/O target.

### K.3 Freeze manifest

`data/repaired_prospective_benchmark_freeze_manifest.json`
(`write_repaired_prospective_freeze_manifest.py`), written and hashed before
any campaign ran: checkpoint SHA-256, common-library SHA-256, edit-to-control
mapping, worlds/seeds, batch structure. Not retuned after seeing results.

### K.4 Design (matched to the original's power)

| | Conventional | Repaired hybrid |
|---|---|---|
| Structure | 4 sequential stages x 4 | 6000 virtual (repaired-model-scored) -> top-8 nonduplicate -> 1 verification stage |
| Exact cultures/campaign | 16 | 8 |
| Worlds | `baseline_world` (10 campaigns), `strong_oxidative_burden_world` (10 campaigns) -- matches the original's 10+10 split exactly |
| Campaign seeds | 97001-97020 (paired per world; new, non-colliding with the original's 86001-86020) |
| Total | 20 paired campaigns, 480 exact cultures, 120,000 virtual evaluations | |

Supplementary `atp_limited_world` (non-oxidative shift) was scoped
(`data/repaired_prospective_benchmark_freeze_manifest.json` includes seeds
97021-97030) but **not run this pass** -- the core 20 already give a
power-matched, conclusive-enough result and an OOM incident (K.6) already
consumed the available compute budget for this session.

### K.5 Acceptance/leakage gates

All 6 automated gates from `write_leakage_audit` (unmodified) **PASS**:
`exact_calls_logged_for_all_physical_cultures`,
`virtual_search_did_not_record_exact_outcomes`,
`old_exact_cache_not_used_as_source`,
`conventional_has_multiple_online_stages`,
`conventional_later_proposals_after_prior_exact`,
`hybrid_exact_verification_after_virtual_search`. Plus: exact backend
`yeast_gem_lp` for all 480 cultures (0 mock), 0 infeasible/unbounded solves,
`n_actual_lp_solves` sums to exactly 69,120 (480 x 144, matches the original
benchmark's own accounting identity), 320/160 conventional/hybrid culture
split (20x16 / 20x8, exact), 240/240 world split (10x24 / 10x24, exact).

### K.6 Compute incident (disclosed, not hidden)

First submission (`mem=4gb`, `walltime=03:00:00`) OOM-killed all 20 array
tasks after ~3h, having genuinely computed 420/480 exact cultures (real
LP-solve work, not wasted -- COBRApy's per-interval `model.copy()` over 24
sequential cultures accumulates more memory than the original benchmark's
own campaigns apparently required). Root cause of the *initial* submission
also included an unrelated path bug (a relative path resolved against the
wrong working directory) that was caught and fixed before any compute was
spent. Recovery: cleared each campaign's ledger CSVs (to avoid duplicate
rows) while preserving the actual per-culture exact-solve cache
(`on_demand_exact_rollouts/*_summary.csv`), fixed `fresh_exact=False` (was
incorrectly `True`, which would have disabled resume and forced a full
restart), bumped to `mem=12gb`/`walltime=05:00:00`, resubmitted. Resumed run
correctly cache-hit all 420 already-solved cultures and completed the
remaining 60 in ~35 minutes wall-clock. Total wall-clock across both
submissions: ~3h50m. **Lesson for the next run**: this workload needs >=8GB
per single-core task, not the 4-6GB the original benchmark's campaigns used.

### K.7 Result

**Campaign-level best `final_product`** (mean/median/min/max over 10 paired campaigns per world):

| World | Method | mean | median | min | max |
|---|---|---:|---:|---:|---:|
| baseline_world | conventional | 1.265 | 0.944 | 0.817 | 3.069 |
| baseline_world | repaired hybrid | 1.086 | 1.065 | 0.877 | 1.546 |
| strong_oxidative_burden_world | conventional | 0.740 | 0.606 | 0.570 | 1.287 |
| strong_oxidative_burden_world | repaired hybrid | 0.621 | 0.567 | 0.535 | 0.900 |

**Paired statistics (hybrid minus conventional, n=20, both worlds pooled, 95% bootstrap CI, `bootstrap_ci`/`write_analysis` unmodified):**

| Metric | mean diff | median diff | 95% CI | wins/ties/losses |
|---|---:|---:|---|---|
| best_final_product (full budget: 8 vs 16) | -0.149 | -0.050 | [-0.410, 0.074] | 8/0/12 |
| best_productivity | -0.012 | -0.004 | [-0.034, 0.006] | 8/0/12 |
| best_product_AUC | -0.164 | -0.138 | [-0.752, 0.386] | 9/0/11 |
| common_horizon (8-culture budget) best-so-far AUC, final_product | +0.378 | +0.085 | [-0.210, 1.098] | 11/0/9 |
| common_horizon mean best-so-far, final_product | +0.058 | +0.022 | [-0.028, 0.161] | 12/0/8 |
| common_horizon final best-so-far, final_product (== best at 8 cultures) | -0.122 | -0.031 | [-0.380, 0.091] | 8/0/12 |

**Every single metric's 95% CI includes zero.** No statistically significant
advantage in either direction at n=20. The direction is genuinely mixed, not
uniformly negative: full-budget best-final-product and the "final value at
common 8-culture budget" metrics lean slightly conventional-favorable (8/20
hybrid wins); the *cumulative* common-8-culture-budget AUC metrics (which
reward finding good candidates *early*, since the repaired hybrid's whole
batch is informed from the start rather than built up stage-by-stage) lean
slightly hybrid-favorable (11-12/20 wins). Both leans are small relative to
their confidence intervals.

### K.8 Interpretation

**The original DBTL acceleration claim does not clearly survive this clean
rerun.** Per Section 21 of this handover, this is a scientifically useful
outcome, not a failure of the exercise: *"The original acceleration result
depended materially on the old information/model contract."* This is
consistent with, and explained by, what Sections H.5-H.6 already found
independently: the repaired scorer's ranking is real but moderate (Spearman
0.59, not near 1.0), and its dominant error source is the gap between
learned-physiology-plus-edit-mapping and the true generator's own dynamics
(RMSE 1.27), not the GSM surrogate. A ranking signal that moderate is enough
to be measurably better than chance (Section H.5's top-8 enrichment/zero
regret) but is evidently not yet enough to reliably beat a real, in-the-loop
adaptive conventional search at n=20 paired campaigns using half the exact
culture budget.

### K.9 Limitations

- Restricted 16/24-intervention domain (K.1) -- both arms treated identically, but this is a narrower design space than the original benchmark's full 24.
- Single frozen checkpoint (`seed=11`) -- no model-seed ensembling attempted.
- Supplementary `atp_limited_world` not run (K.4) -- the claim above is scoped to `baseline_world` + `strong_oxidative_burden_world` only, matching the original's own scope exactly (also 2 worlds).
- n=20 paired campaigns gives real but not unlimited statistical power; wide CIs reflect that honestly rather than being narrowed by any post-hoc choice.
- Per K.6, this used ~3h50m of Vanda compute across two submissions for the core design alone; a 3-world (30-campaign) design would cost roughly 1.5x that.

### K.10 Artifacts

`data/repaired_prospective_benchmark_freeze_manifest.json` (frozen config),
`results/repaired_prospective_campaigns/<campaign_id>/` (20 isolated
per-campaign raw ledgers + exact rollout cache), `results/repaired_prospective_benchmark_aggregate/data/`
(merged ledgers + `prospective_paired_statistical_comparisons.csv` +
`prospective_leakage_audit.csv` + `prospective_campaign_level_summary.csv` +
full report/figures, all produced by unmodified `bench.write_analysis()`).

---

## L. GSM surrogate replacement (this pass)

Pass 3 (Section K) trusted the polynomial-ridge GSM surrogate
(`gem_gsm_surrogate.py`) without ever isolating its contribution to
candidate-ranking error. This pass does exactly that -- Stage 1 below --
**before** touching the surrogate, per the standing rule "do not replace the
surrogate merely because absolute RMSE is nonzero." It does not touch the
physiology learner's training-information contract (Section C): no hidden
`z_*` state, internal LP flux target, edited-strain outcome, or prospective
exact outcome was added to any training signal. Only the frozen,
differentiable metabolic-response approximation the learner is trained
*through* changed.

### L.1 Stage 1 -- isolating the surrogate's ranking error (pre-registered decision point)

300-candidate pool (`run_surrogate_ranking_diagnostic.py`, same restricted
16/24-intervention library and `CandidateFactory` as Pass 3), scored two
ways with the *same* frozen physiology checkpoint
(`hybrid_learned_physiology__full__seed11.npz`) and *same* edit mapping:

- **Path A**: predicted `c(t)` + edits -> old ridge surrogate -> objective (`RepairedVirtualScorer.score`, no LP).
- **Path B**: same `c(t)` + edits -> real Yeast9 (`run_repaired_controls_real_gsm_replay_worker.py`, unmodified, reused from Pass 2).

| Metric | Value |
|---|---:|
| Overall Spearman / Kendall / RMSE | 0.802 / 0.637 / 0.435 |
| Top-8 / top-25 / top-50 overlap | 0.250 / 0.440 / 0.660 |
| Tail agreement, top 1% / 5% / 10% | 0.333 / 0.200 / 0.433 |
| Spearman by edit count (0 / 1 / 2 / 3) | 0.975 / 0.799 / 0.763 / 0.754 |
| Spearman by mean support distance (<3σ / 3-4σ / 4-5σ) | 0.630 / 0.799 / 0.335 |
| Candidates with >=1 clipped control value | 100% |

A side finding drove the last row: the physiology model's predicted `c(t)`
has z-scores (relative to the surrogate's training mean/std) that grow
**monotonically across the 48-interval rollout for every candidate,
regardless of edits**, peaking at the final interval (t=47) at 6-10σ. This
means the dominant driver of surrogate out-of-support queries is **rollout
length itself**, not edits alone -- edits then compound it further (Spearman
degrades from 0.975 at 0 edits to 0.754 at 3 edits, and collapses to 0.335 in
the 4-5σ mean-extrapolation bucket).

**Decision (pre-registered gate)**: top-8 overlap of 0.25 and top-5%-tail
overlap of 0.20 are far from "near-perfect in the top tail" -- the surrogate
**is** a material ranking bottleneck, concentrated in edited/extrapolative
candidates exactly as hypothesized. Proceeding to Stage 2.

### L.2 Stage 2 -- purpose-built exact-Yeast9 training dataset

`generate_gsm_surrogate_stage2_points.py` samples the actual deployment
joint space instead of only naturally-visited reference-strain trajectories:

| Source | n points | Description |
|---|---:|---|
| `traj` | 2,400 | Physiology-predicted, edit-adjusted `c(t)` for 300 diverse candidates at 8 stratified timesteps -- captures the real drift+edit-driven support extension found in L.1, unclipped |
| `edge_random` | 6,800 | Independent per-control-dimension samples out to +-8 training-σ (double the old 4σ safety clip), random environments, decorrelated (deliberate high/low O2 + ATP + capacity combos) |
| `edge_corner` | 640 | Predeclared combinatorial +-6σ sign corners on (O2, ATP, capacity-group, γ) x 40 environments |
| **Total** | **9,840** | All physically sign-clipped (O2 lower bound negative, ATP/PSY/DES/CYC non-negative, γ in [0.05, 0.95]); none astronomical |

Solved via `run_gsm_surrogate_stage2_solve_worker.py` on Vanda (99-shard PBS
array, fresh `model.copy()` per point, exact 3-solve staged pFBA per point --
growth LP + product LP + pFBA, same pattern as `gem_backend.solve_staged`):
**9,640 / 9,840 points solved** (97/99 shards; the remaining 2 shards (200
points, 2%) hit the walltime limit even after a retry and were left
unresolved -- disclosed, not silently dropped, and immaterial at this
sample size). 0 solver errors among the 9,640. Total exact LP solves:
9,640 x 3 = **28,920**.

### L.3 Stage 3 -- MLP surrogate

`gem_gsm_surrogate_mlp.py`: one hidden layer (width 32), tanh activation,
same public contract as the old `GSMSurrogate` (`.predict`,
`.predict_and_grad`, `control_mean/std`, `env_mean/std`) so it is a drop-in
replacement. Exact analytic Jacobian via hand-derived backprop through the
single tanh layer (chain rule through input/output standardization) --
consistent with the project's standing "no autodiff library available"
constraint (`gem_trainable_state_space.py`'s own hand-BPTT). Deliberately
boring: no attention, no depth beyond one hidden layer, no architecture
search.

`tests/test_gsm_surrogate_mlp.py` (3/3 passing): analytic Jacobian vs.
central finite differences, max relative error < 1e-3; save/load round-trip
exact; fit reduces held-out error below the mean-baseline on synthetic data
with known nonlinear structure.

### L.4 Stage 4 -- held-out validation, flux + ranking

`fit_new_gsm_surrogate.py`: candidate-aware 85/15 split of the Stage 2
dataset (whole `traj` candidates held out together, not individual
timesteps) -- 8,194 train / 1,446 held-out points. Old ridge surrogate
refit on the *same* split for a controlled architecture comparison.

| Channel | Old ridge (held-out norm. RMSE) | New MLP (held-out norm. RMSE) |
|---|---:|---:|
| biomass_flux | 0.210 | **0.068** |
| beta_carotene_flux (= ggpp/PSY/DES/CYC flux, 1:1 pathway stoichiometry) | 0.517 | **0.106** |
| oxygen_uptake | 0.199 | **0.051** |
| atp_maintenance_flux | 0.0001 | 0.028 (ATPM flux is ~identical to its own input control under pFBA; any linear model gets this trivially) |

3-5x lower flux error on identical training/held-out data isolates the
architecture change (not just the new data) as the source of improvement.

Ranking accuracy: rescored the **same** Pass-K checkpoint's 300-candidate
pool (L.1) with the new surrogate, no safety clip, against the *already-computed*
Path B (zero new LP spent):

| Metric | Old surrogate | New surrogate |
|---|---:|---:|
| Spearman / Kendall / RMSE | 0.802 / 0.637 / 0.435 | **0.966 / 0.850 / 0.413** |
| Top-8 / top-25 / top-50 overlap | 0.250 / 0.440 / 0.660 | **0.500 / 0.760 / 0.840** |
| Tail overlap, 5% / 10% | 0.200 / 0.433 | **0.667 / 0.867** |
| Spearman by edit count (0/1/2/3) | 0.975/0.799/0.763/0.754 | 0.980/**0.949**/**0.935**/**0.939** |

The old surrogate's 4σ safety clip would still fire on 100% of candidates
under this checkpoint's `c(t)` (informational -- the new surrogate was
trained explicitly to cover that region and is not clipped at all).

### L.5 Stage 5 -- retraining the physiology model through the new surrogate

**Gate check (pre-registered: retrain only if the Jacobian materially
changed)**: comparing `predict_and_grad` from both surrogates at 200 points
sampled from real predicted trajectories --

| Channel | grad cosine(old, new) | mean `\|grad\|` old -> new |
|---|---:|---:|
| biomass_flux | 0.924 (σ=0.050) | 3.105 -> 0.545 |
| beta_carotene_flux | **0.492** (σ=0.416) | 3.842 -> 0.890 |

The product-channel gradient *direction* is essentially unreliable under the
old surrogate in the region BPTT actually visits (cosine ~0.49, high
variance -- sometimes aligned, sometimes not) -- direct evidence for the
"surrogate exploitation" hypothesis: BPTT through an inaccurate polynomial
gradient field was shaping the physiology model's drift, not genuine
Yeast9 mechanistic structure. Gate met -- retrained.

Same frozen architecture/config/seed as the existing checkpoint
(`retrain_physiology_with_new_surrogate.py`: D_model=16, hidden=24,
epochs=600, lr=0.02, seed=11, `reporter_mode=full`, clean reference-strain
training data, unchanged contract) -- only the frozen surrogate injected into
`compute_loss_and_grads` changed. Pure-numpy BPTT, no LP solves: **8.3s
wall-clock**.

| Split | product norm. RMSE, old -> new ckpt | product final-timepoint abs. error, old -> new |
|---|---:|---:|
| train | 0.384 -> 0.140 | 0.131 -> 0.021 |
| validation | 0.424 -> 0.152 | 0.149 -> 0.019 |
| interpolation | 0.407 -> 0.140 | 0.140 -> 0.016 |
| heldout_combination | 0.388 -> 0.133 | 0.129 -> 0.014 |
| extrapolation | 0.435 -> 0.142 | 0.134 -> 0.016 |

~3x lower normalized RMSE and **8-9x lower final-timepoint error** across
every split, on the physiology model's own native clean-reference-strain
evaluation task -- not just a downstream ranking side effect.

Drift check (same 0-edit, T=30/pH=5/DO=50 probe used in L.1's side finding):
z-scores at t=47 dropped from **6.3-10.4σ (old checkpoint) to a 0.4-2.6σ max
(new checkpoint)** across all 6 control dimensions. The retrained physiology
model no longer needs to drift into extreme extrapolation to minimize its
loss -- consistent with the gate check above.

### L.6 Stage 6 -- surrogate-to-exact funnel

Reused the L.1/L.4 300-candidate pool's Path A' (new surrogate) and Path B
(real Yeast9) -- **zero new LP solves** (Path B literally *is* "exact
reranking"). Shortlist top-N by new-surrogate score, rerank the shortlist by
real Yeast9 score, take top-8; compare to (a) the surrogate-only top-8 (no
reranking) and (b) the oracle top-8 (true top-8 by Path B across all 300).

| N (predeclared) | Surrogate-only top-8 oracle overlap | Reranked top-8 oracle overlap |
|---:|---:|---:|
| 50 | 0.50 | **1.00** |
| 100 | 0.50 | **1.00** |
| 200 | 0.50 | **1.00** |
| 400 (pool caps at 300) | 0.50 | **1.00** |

All four predeclared N recover the oracle top-8 exactly. Supplementary
(non-predeclared) smaller-N sweep to find where the value actually
saturates: N=8: 0.50, N=12: 0.625, N=16: 0.625, N=25: 0.875, **N=35: 1.00**
-- the funnel's benefit saturates at N=35, comfortably below the smallest
predeclared shortlist size, at zero incremental exact-LP cost beyond what
L.1 already spent.

### L.7 Stage 7 -- updated A/B/C error decomposition (value and ranking)

**Full pipeline, retrained checkpoint + new surrogate** (60-candidate subset
of the L.1 pool, `score_candidates_new_checkpoint_new_surrogate.py` for
Path A'' + `run_repaired_controls_real_gsm_replay_worker.py` unmodified with
`--checkpoint` pointed at the new checkpoint for Path B''):

| Metric | Value |
|---|---:|
| Infeasible-interval fraction | 0.000 (all 60 fully solvable) |
| Spearman / Kendall / RMSE | 0.966 / 0.860 / 0.326 |
| Top-8 / top-15 / top-25 overlap | 0.875 / 0.933 / 0.880 |

**A/B/C decomposition** (24-candidate subset, matching Section H.6's
original scale; Path C = `exact_sparse_dynamic_rollout`, `hidden_regime=
"strong_dynamic_regulation"`, reused unmodified via the already-existing
`run_prospective_exact_worker.py`; all 24 feasible, 144 LP solves each,
`complete_exact_dynamic_pfba`):

| Comparison | RMSE | Spearman | Isolates |
|---|---:|---:|---|
| A vs B (surrogate error alone) | 0.337 | **0.964** | Surrogate approximation error |
| B vs C (physiology-vs-generator) | 0.304 | 0.301 | Learned-physiology + edit-mapping error vs. the true generator's own dynamics |
| A vs C (total end-to-end) | 0.208 | 0.279 | Total pipeline error |

**Interpretation**: A tracks B almost perfectly now (Spearman 0.96) -- the
surrogate fix genuinely closed the surrogate-attributable ranking gap. B
does *not* track C well (Spearman 0.30), and neither does A vs C (0.28) --
but that gap is inherited entirely from B-vs-C, i.e. from the physiology
model's edit-response realism relative to the true generator's own
feedback dynamics, exactly as Section H.6 first found and this pass
explicitly did not touch (out of scope per the standing contract in Section
C). Sharpened restatement of H.6's conclusion: **the surrogate was real but
secondary; the dominant remaining error source is, and was always,
physiology-vs-generator edit realism.**

### L.8 Acceptance table (gates S1-S9)

| Gate | Status | Evidence |
|---|---|---|
| S1 -- surrogate ranking error quantified | **PASS** | L.1: Spearman 0.802, top-8 overlap 0.250, tail-5% 0.200 |
| S2 -- purpose-built dataset covers deployment domain | **PASS** | L.2: 9,640 exact points across traj/edge_random/edge_corner, up to +-8σ, physically clipped not astronomical |
| S3 -- new surrogate materially improves held-out approx./ranking | **PASS** | L.4: 3-5x lower flux RMSE; Spearman 0.802 -> 0.966, top-8 overlap 0.250 -> 0.500 |
| S4 -- out-of-support rate substantially reduced | **PASS (reframed)** | L.5: retrained physiology's own baseline drift dropped from 6.3-10.4σ to 0.4-2.6σ; new surrogate needs no safety clip at all (old 4σ clip still informationally fires 100% of the time on the *old* checkpoint's `c(t)`, which is exactly why the old clip existed and why L.5's drift fix matters more than the clip itself) |
| S5 -- gradient numerically verified | **PASS** | L.3: analytic vs. finite-difference Jacobian, max relative error < 1e-3, 3/3 tests |
| S6 -- no hidden-state/prospective-outcome leakage | **PASS** | Physiology learner's `SUPERVISION_COLUMNS` contract (Section C/D1) unchanged this pass; only the frozen surrogate function changed |
| S7 -- exact reranking uses same learned physiology as surrogate scoring | **PASS** | L.6/L.7: Path A/Path B(") always share the identical checkpoint and edit mapping, never the hidden generator's own physiology |
| S8 -- reranking improves top-k quality enough to justify compute | **PASS** | L.6: oracle top-8 recovered exactly by N=35, at zero incremental LP cost (Path B reused) |
| S9 -- exact LP accounting closes | **PASS** | L.2 worker records per-point status (9,640/9,840 ok, disclosed shortfall); L.7 Path C worker records `n_actual_lp_solves=144`, `n_optimal_solves=144` for all 24 |

### L.9 Recommendation

The surrogate was a real, now-fixed ranking bottleneck (S1-S5, S8 above) --
but Stage 7's sharpened decomposition (L.7) shows the **dominant remaining
error source in the full pipeline is unchanged from Section H.6**:
physiology-vs-true-generator edit-response realism, not the surrogate. A new
powered 20-campaign DBTL rerun (repeating Section K) is **justified** in the
sense that the underlying scorer is now materially more accurate and the old
run's surrogate-driven top-8 selection error (0.25 overlap with the true
top-8) would no longer apply -- but it would **not** by itself close the
gap that limits absolute prospective-search quality, which is physiology
realism, not surrogate fidelity. Recommended next step, in priority order:
(1) rerun Section K's powered benchmark with the new checkpoint +
surrogate to get an updated, higher-fidelity headline number (cheap: the
scorer swap is a config change, no new architecture); (2) treat
physiology-vs-generator edit realism (`LATENT_CAPACITY_MISMATCH_DESIGN.md`'s
still-unaddressed generator-side causal-rank gate, Section J) as the
higher-value target for a future pass, since L.7 confirms it -- not the
surrogate -- now bounds end-to-end ranking quality.

### L.10 Artifacts

`src/yeast_validation/generate_gsm_surrogate_stage2_points.py`,
`src/yeast_validation/run_gsm_surrogate_stage2_solve_worker.py`,
`src/yeast_validation/gem_gsm_surrogate_mlp.py`, `src/yeast_validation/fit_new_gsm_surrogate.py`,
`src/yeast_validation/retrain_physiology_with_new_surrogate.py`,
`src/yeast_validation/score_candidates_with_new_surrogate.py`,
`src/yeast_validation/score_candidates_new_checkpoint_new_surrogate.py`,
`src/yeast_validation/analyze_surrogate_ranking_diagnostic.py`,
`tests/test_gsm_surrogate_mlp.py`; data:
`data/surrogate_diagnostic_pool_path_A.csv`,
`data/surrogate_diagnostic_merged.csv`,
`data/surrogate_diagnostic_stage1_report.md`,
`data/gsm_surrogate_stage2_points.csv`,
`data/gsm_surrogate_stage2_solved.csv`,
`data/stage3_stage4_surrogate_report.md`,
`data/surrogate_diagnostic_old_vs_new_merged.csv`,
`data/stage5_new_surrogate_training_metrics.csv`,
`data/stage6_funnel_results.csv`,
`data/stage7_ABC_decomposition_new.csv`; checkpoints:
`results/repaired_hybrid_training/checkpoints/gsm_surrogate_mlp_v1.npz`,
`results/repaired_hybrid_training/checkpoints/hybrid_learned_physiology__full__seed11__new_surrogate.npz`.

---

## M. Powered rerun with the repaired surrogate + exact-Yeast9 reranking

Direct successor to Section K, same question, now using the Section L
checkpoint/surrogate and adding a Stage-2 exact-Yeast9 reranking step
between the virtual screen and biological verification: *does the clean
hybrid digital twin outperform conventional adaptive DBTL once the
surrogate ranking problem (Section L) has been repaired?* Same frozen
learner architecture, historical training data, deployment inputs,
conventional algorithm, worlds, and restricted candidate domain as Section
K -- nothing in the biological information contract changed.

### M.1 Frozen scoring flow

```
environment -> retrained checkpoint (hybrid_learned_physiology__full__seed11__new_surrogate.npz)
   -> z(t) -> predicted c(t) -> edit -> new MLP surrogate (gsm_surrogate_mlp_v1.npz)
   -> mechanistic integration -> virtual_score          [Stage 1, 6000/campaign, no LP]
same c(t) + edit -> REAL Yeast9 (apply_predicted_interface_controls +
   gem_backend.solve_staged, run_repaired_controls_real_gsm_replay_worker.py
   UNMODIFIED) -> real-Yeast9 rerank score               [Stage 2, top-50/campaign]
top-8 by rerank score -> true hidden generator's exact_sparse_dynamic_rollout
   (run_on_demand_exact, unmodified)                     [Stage 3, biological verification]
```

Both checkpoints are frozen and were not retrained or retuned based on this
benchmark's results. `data/repaired_prospective_benchmark_v2_freeze_manifest.json`
(checkpoint/surrogate/library SHA-256, N=50, budgets, worlds, seeds) was
written and hashed before any campaign ran.

### M.2 Implementation (three phases, reusing Section K's monkey-patch pattern)

`gem_repaired_prospective_common_v2.py` adds a `generate_hybrid_shortlist`
function (the single source of truth for Stage 1) and patches
`bench.hybrid_virtual_search` (not just `hybrid_virtual_score`, since the
frozen function has no reranking-shortlist step to hook into):

- **Phase 1** (`generate_repaired_prospective_v2_shortlists.py`, cheap,
  local): for each of the 20 campaigns, runs the 6000-candidate virtual
  screen once and persists the full 6000-row virtual ledger + top-50
  shortlist (candidate JSONs + manifest) to disk.
- **Phase 2** (PBS array, 1000 tasks = 20 x 50): reranks every shortlisted
  candidate through real Yeast9, reusing
  `run_repaired_controls_real_gsm_replay_worker.py` **unmodified**, pointed
  at the new checkpoint.
- **Phase 3** (`run_repaired_prospective_campaign_worker_v2.py`, one PBS
  array task per campaign): runs `bench.run_campaign` for real (both
  arms), with `hybrid_virtual_search` patched to a closure that **loads**
  Phase 1's persisted screen (never recomputes it -- see M.3), looks up
  each shortlisted candidate's Phase-2 rerank score, and returns the true
  top-8 by rerank score for biological verification via the unchanged
  `run_on_demand_exact` path.

### M.3 Two implementation bugs found and fixed before/during the run

1. **Cross-machine determinism failure (caught by the mandated one-campaign
   sanity check, before any full submission).** An earlier version of
   Phase 3 *regenerated* the 6000-candidate screen via the same RNG seed,
   assuming it would be bit-identical to Phase 1's (which ran locally).
   Vanda's Linux/BLAS environment produced tiny floating-point differences
   in the surrogate's forward pass, enough to flip candidate ordering right
   at the rank-50 cutoff -- 12/50 shortlist candidates came back with
   mismatched hashes and the sanity campaign crashed cleanly with a
   `RuntimeError` (no silent corruption). Fixed by making Phase 1 persist
   its full output and Phase 3 load it verbatim, removing the cross-machine
   dependency entirely. Recovery followed the same discipline as K.6:
   cleared the partially-run campaign's ledger CSVs (avoiding duplicate
   rows) while preserving its already-solved exact-culture cache, then
   resumed.
2. **Ledger schema drift.** The reranking closure attached extra
   bookkeeping keys (`_surrogate_score`, `_rerank_score`, `_rerank_rank`)
   directly onto candidate dicts; `prospective_candidate_designs.csv`'s
   writer (unmodified, in the frozen file) spreads a candidate's dict
   wholesale into its row, so the 8 selected hybrid candidates per campaign
   got 3 extra trailing columns their conventional-arm counterparts didn't
   have, breaking that ledger's column alignment for every campaign (160/480
   rows). Found when aggregation failed to parse it. Fixed data-only
   (`repair_v2_candidate_designs_csv.py`, verified exact column mapping
   against the file's own header before dropping the 3 known-garbage
   columns -- no exact-culture data touched) and fixed the worker script so
   it cannot recur.

### M.4 Compute incident (disclosed)

First Phase 3 submission (`mem=12gb`, matching Section K's own successful
configuration) OOM-killed all 19 fresh campaigns at a strikingly consistent
21/24 cultures each. Likely explanation: K's 12gb submission that
"succeeded" was its *second* attempt, after its first (4gb) OOM had already
cached 420/480 cultures -- so it may never have actually exercised a fully
fresh 24-sequential-culture process at 12gb. This run's campaigns were all
fresh, and hit the true ceiling for the first time. Recovery: cleared the
19 campaigns' ledger CSVs (preserving their 21/24 already-solved exact-cache
entries each), bumped to `mem=20gb`, resubmitted -- completed cleanly with
0 further kills, and the cached 21/24 cultures per campaign made the retry
fast (~6-7 minutes/campaign for the remaining ~3 fresh cultures each).

### M.5 Biological-culture accounting

| | Conventional | Repaired hybrid (reranked) | Total |
|---|---:|---:|---:|
| Exact cultures/campaign | 16 | 8 | 24 |
| Campaigns | 20 | 20 | 20 paired |
| Total exact cultures | 320 | 160 | **480** |
| World split | 240 (baseline) / 240 (oxidative) | -- | matches 10+10 exactly |

`n_actual_lp_solves` sums to exactly **69,120** (480 x 144, matching the
LP-per-culture identity used throughout this codebase). 0 infeasible, 0
unbounded solves; 100% `yeast_gem_lp` backend, 0 mock cultures; 0 duplicate
`candidate_hash` rows across the full 480-row ledger.

### M.6 Surrogate / real-Yeast9 compute accounting (not counted as cultures)

| | Count | LP solves |
|---|---:|---:|
| Stage-1 surrogate virtual evaluations | 20 x 6,000 = **120,000** | 0 (pure numpy, no LP) |
| Stage-2 real-Yeast9 rerank trajectories | 20 x 50 = **1,000** | 1,000 x 144 = **144,000** |

All 1,000 rerank trajectories solved cleanly (0 infeasible intervals, 0
NaN). These 1,000 in-silico Yeast9 runs are explicitly **not** biological-style
cultures -- they use the learned physiology's own predicted `c(t)`, never the
hidden generator's true regulatory state (leakage gate 5, M.7).

### M.7 Leakage / fairness audit

All 6 automated gates from `write_leakage_audit` (unmodified) **PASS**:
`exact_calls_logged_for_all_physical_cultures` (480),
`virtual_search_did_not_record_exact_outcomes` (120,000 virtual rows, 0 with
`exact_outcome_accessed=True`), `old_exact_cache_not_used_as_source`,
`conventional_has_multiple_online_stages` (4),
`conventional_later_proposals_after_prior_exact` (60),
`hybrid_exact_verification_after_virtual_search` (120,000). Manually
verified in addition: exact reranking (Stage 2) used only the learned
checkpoint's predicted controls, never the hidden generator's physiology;
both arms drew from the identical restricted 16/24-intervention library
(byte-identical file/hash to Section K); the hybrid arm's final 8 were
selected by rerank score *before* any biological-style outcome was known
(the 8 verification calls happen strictly after Stage 2 in the code path).

### M.8 Results

**Campaign-level best `final_product`** (mean/median/min/max over 10 paired campaigns per world):

| World | Method | mean | median | min | max |
|---|---|---:|---:|---:|---:|
| baseline_world | conventional | 1.631 | 1.465 | 0.817 | 3.706 |
| baseline_world | repaired hybrid (reranked) | **3.141** | **2.989** | 2.481 | 4.714 |
| strong_oxidative_burden_world | conventional | 0.965 | 0.798 | 0.570 | 2.024 |
| strong_oxidative_burden_world | repaired hybrid (reranked) | **2.225** | **1.840** | 1.642 | 3.426 |

**Paired statistics (hybrid minus conventional, n=20, both worlds pooled, 95% bootstrap CI, `bootstrap_ci`/`write_analysis` unmodified):**

| Metric | mean diff | median diff | 95% CI | wins/ties/losses |
|---|---:|---:|---|---|
| best_final_product (full budget: 8 vs 16) | **+1.385** | +1.265 | **[1.053, 1.772]** | **20/0/0** |
| best_productivity | +0.115 | +0.105 | [0.088, 0.148] | 20/0/0 |
| best_product_AUC | +3.420 | +2.988 | [2.685, 4.235] | 20/0/0 |
| common_horizon (8-culture budget) best-so-far AUC, final_product | +11.861 | +10.192 | [10.131, 13.642] | 20/0/0 |
| common_horizon mean best-so-far, final_product | +1.684 | +1.458 | [1.436, 1.941] | 20/0/0 |
| common_horizon final best-so-far, final_product (== best at 8 cultures) | +1.511 | +1.292 | [1.159, 1.906] | 20/0/0 |
| best_so_far_auc (full-budget cumulative), final_product | +1.799 | +2.840 | [-1.185, 4.806] | 14/0/6 |

Every headline metric's 95% CI is now **entirely positive** and every
campaign-pair is a clean win at the common 8-culture budget (20/0/0) -- a
complete reversal from Section K's every-CI-crosses-zero, 8/20-win result.
The one metric that does *not* reach significance
(`best_so_far_auc`/full-budget cumulative AUC, 14/0/6, CI crosses zero) is
expected: it accumulates over the *full* 16-culture conventional budget vs.
the hybrid's 8, structurally favoring whichever arm gets more cultures for
a cumulative-AUC metric -- the common-8-culture-budget versions (which
control for this) are unambiguous.

### M.9 Comparison with Section K

Per instructions, this is **not** a naive independent-samples test between
K's and this pass's campaign sets (different seeds: 97001-97030 vs.
98001-98020; conventional-arm absolute values differ too, e.g. K's
baseline-world conventional mean was 1.265 vs. this run's 1.631 -- expected
seed-to-seed variance in a stochastic search, not a code change, since
`propose_conventional_batch` is byte-identical). The qualitative comparison
requested:

| | Section K (old surrogate) | Section M (repaired surrogate + reranking) |
|---|---|---|
| Hybrid win rate (best_final_product) | 8/20 | **20/20** |
| Mean diff (hybrid - conventional) | -0.149 | **+1.385** |
| 95% CI | [-0.410, 0.074] (crosses zero) | **[1.053, 1.772]** (entirely positive) |
| Common-8-culture-budget AUC wins | 11-12/20 (small lean) | **20/20** (large margin) |
| Qualitative conclusion | Inconclusive / no significant advantage | **Clear, significant advantage** |

**Hybrid candidate quality, win rate, and common-budget efficiency all
improved -- and the qualitative conclusion changed completely.**

### M.10 Interpretation

**The DBTL acceleration claim now survives.** Per the pre-registered
interpretation rule (Section: this pass's brief): a clear positive result
implicates the previous surrogate as a material cause of Section K's
negative/inconclusive result -- consistent with Section L's own decomposition
(A-B/surrogate error: Spearman 0.96, near-solved; B-C/physiology-vs-generator
gap: Spearman 0.30, unsolved). This powered rerun is the first place that
distinction becomes an outcome-level difference: repairing the surrogate and
adding exact-Yeast9 reranking was sufficient, on its own, to flip a
half-budget hybrid search from statistically indistinguishable-from-conventional
to a clean, large, significant win -- **without** touching the physiology
learner's architecture, training data, or information contract, and without
narrowing the still-unaddressed physiology-vs-generator gap at all.

**Honest limitation, per instructions (do not tune further based on this
result):** the reranked hybrid arm's advantage is now large enough that the
remaining physiology-vs-generator gap (Section L.7, Spearman ~0.30) no
longer prevents it from winning -- but that gap is still real and
unaddressed. A harder hidden world, a broader edit domain, or a
smaller/no reranking budget could plausibly narrow or erase this margin;
this result should be read as "the surrogate was the dominant blocker at
this design's difficulty level," not as "physiology realism no longer
matters." Per the standing instruction, no further surrogate/model tuning
was attempted based on this positive result -- the next highest-value
target remains the physiology-vs-generator gap identified in Section L.7
and H.6, now with a documented, reproducible funnel design to measure it
against.

### M.11 Artifacts

`data/repaired_prospective_benchmark_v2_freeze_manifest.json` (frozen
config); `src/yeast_validation/gem_repaired_prospective_common_v2.py`,
`src/yeast_validation/generate_repaired_prospective_v2_shortlists.py`,
`src/yeast_validation/run_repaired_prospective_campaign_worker_v2.py`,
`src/yeast_validation/repair_v2_candidate_designs_csv.py`,
`src/yeast_validation/write_repaired_prospective_v2_freeze_manifest.py`;
`results/repaired_prospective_campaigns_v2/<campaign_id>/` (20 isolated
per-campaign raw ledgers + exact rollout cache); `results/repaired_prospective_benchmark_v2_aggregate/data/`
(merged ledgers + `prospective_paired_statistical_comparisons.csv` +
`prospective_leakage_audit.csv`, all produced by unmodified
`bench.write_analysis()`). HPC working data (Phase 1/2 shortlists,
candidates, 1,000 rerank results, virtual ledgers) lived under
`/scratch/e1471252/prospective_dbtl_benchmark_v2/` on Vanda (home-dir quota
was full this pass) and is not mirrored into the repo wholesale -- only the
480-row exact-culture ledgers and aggregate outputs above are.
