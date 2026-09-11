# Latent Capacity Mismatch Benchmark — Design, Audit, and Pilot Record

This document is the companion to `GENERATOR_VALIDITY.md` for the new
experiment: can a fixed, low-dimensional pretrained hybrid digital twin still
accelerate DBTL when the hidden organism's true causal physiological
dimensionality (`D_true`) exceeds the learner's fixed latent dimensionality
(`D_model`)? It records what Stage A's audit found, what Stage B/C/D built and
verified, and the frozen acceptance gates, so the eventual Stage E freeze and
Stage F powered run have a single source of truth to point at.

Status as of this pass: **Stages A-D complete** (audit, hardened generator,
controller-dimensionality family + causal-rank diagnostic, clean-teacher
retrain + small pilot). **Stages E (freeze) and F (powered run) have not been
run.** See Section 7 for the exact next command.

---

## 1. Stage A audit findings

### 1.1 Production architecture and D_model

The production "hybrid digital twin" teacher (`src/yeast_validation/run_reporter_grounded_hybrid_distillation.py`)
is a closed-form, seeded, deterministic architecture -- **not** a
gradient-trained neural network:

```
z0 = tanh(env @ Wg + bg)
z[k+1] = z[k] + dt * tanh(z[k] @ A + env @ C + bf)      # A, Wg, C, bf are fixed, seeded random matrices, never trained
head_col(z) = weighted_ridge_fit([1, z, z^2, time], target_col)   # closed-form, independent per output column
```

`LATENT_DIMS = [4, 8, 16]` were evaluated; `full_reporter_interface_state_space`
(latent dimension **16**) is `PRIMARY_TEACHER_VARIANT`, selected consistently
across all 5 `MODEL_SEEDS`. **`D_model = 16`.**

The headline prospective DBTL loop (`run_prospective_dbtl_benchmark.py::hybrid_virtual_score`)
does not call this teacher live during virtual search -- it scores candidates
from a frozen linear intervention-effect table plus a distance-weighted
nearest-neighbour lookup into `teacher_pseudodata.csv`, which the teacher
generates once, offline.

### 1.2 The information-boundary violation (the reason for this pass)

`run_reporter_grounded_hybrid_distillation.py:56-66` (`INTERFACE_COLUMNS`)
includes raw hidden burden states directly: `z_ox, z_atp, z_bottle`. Lines
75-84 (`FLUX_COLUMNS`) include internal LP fluxes with no wet-lab analogue:
`ggpp_flux, PSY_flux, DES_flux, CYC_flux, atp_maintenance_flux`. The selected
`PRIMARY_TEACHER_VARIANT` trains on both (`interface: True, flux: True`).

This is exactly the violation Section 6 of the handover prohibits. It was not
caught by the existing leakage audit (`GENERATOR_VALIDITY.md` Section 6.5),
which checks *prospective-outcome* leakage (no future exact culture outcomes
reach virtual search) but not *hidden-state* leakage into training.

**Resolution (user-directed):** retrain a clean `D_model=16` teacher, same
architecture and hyperparameters, on a leak-free target contract (Section 3
below). The existing headline result and its leakage caveat are left
untouched; this is a new, separate, clean baseline for the mismatch
experiment. See `src/yeast_validation/gem_clean_teacher_targets.py` and
`tests/test_clean_teacher_no_leakage.py` (which also pins a regression
assertion that the *old* teacher's leak is still present and unmodified, as
an audit trail, not a fix-in-place).

### 1.3 Other confirmed G3 weaknesses (addressed in Stage B)

- No extracellular glucose mass balance -- **fixed**, opt-in (`GENERATOR_VALIDITY.md` Section 2.9).
- No observation noise on product/biomass (reporters already had lag+noise) -- **fixed**, opt-in (`GENERATOR_VALIDITY.md` Section 2.10).
- No oxidative-coupling ablation switch -- **fixed** (`GEMCultureConfig.oxidative_beta_coupling`).
- Only 2 of 9 implemented worlds ever run in the powered benchmark -- `atp_limited_world` (non-oxidative) activated for this pilot; see Section 5.
- Live parameter-provenance bug (`selected_parameter_config()` returns the pre-sweep base point, not the swept winner) -- documented, decision recorded (`GENERATOR_VALIDITY.md` Section 6.6): old result untouched, new freeze manifest will state its parameter set explicitly.

---

## 2. Controller-dimensionality family (Stage C)

`src/yeast_validation/gem_controller_family.py`. Core triad (`z_ox, z_atp, z_bottle`,
mechanistic, unmodified) is always present. `D_true = N` selects the first
`N - 3` entries of a 21-state library (`MAX_D_TRUE = 24`), each with a
mechanism, time constant, real Yeast9 GSM coupling target, and evidence class.

### 2.1 Tier 1 (states 4-10): distinct mechanisms

| State | Mechanism | tau (h) | GSM coupling | Evidence class |
|---|---|---|---|---|
| `z_nadph` | NADPH/redox cofactor availability | 1.5 | PPP NADPH source (`r_0466`) upper bound | literature-shaped |
| `z_proteostasis` | Heterologous protein/proteostasis burden | 3.0 | shared PSY+DES+CYC upper-bound ceiling | literature-shaped |
| `z_membrane` | Membrane/transport burden | 4.0 | ammonium exchange (`r_1654`) lower bound | declared design |
| `z_precursor_dyn` | Dynamic native-isoprenoid competition | 2.5 | native GGPP (`r_0461`) bound | literature-shaped |
| `z_translation` | Ribosome/protein-allocation trade-off | 1.0 | biomass reaction **upper** bound (new target) | literature-shaped |
| `z_recovery` | Slow cumulative-exposure recovery | 8.0 | independent ATPM relief term | declared design |
| `z_osmotic` | General osmotic/environmental stress | 1.0 | glucose exchange lower bound (independent of B1 substrate limit) | declared design |

### 2.2 Tier 1b (states 11-14): slow-pool sister variants

`z_ox_slow, z_atp_slow, z_bottle_slow, z_nadph_slow` -- same thematic family as
their fast sibling, distinct (PSY/CYC/DES/O2) coupling target, tau 7-10h.
Declared design, explicitly not independently calibrated.

### 2.3 Tier 2 (states 15-24): dimensionality-scaling variants

Ten further states (`z_redox2, z_transport2, ..., z_respiration2`) reusing the
ten reaction targets already established above via a shared coupling-function
factory (`_make_scaling_coupling`), each with its own tau and generation
function, added solely to reach `D_true` up to 24. All declared design.
Whether each is causally distinguishable from earlier states on the same
reaction is answered empirically (Section 2.4), not assumed.

### 2.4 Causal-rank sensitivity diagnostic and frozen acceptance gate

`gem_controller_family.sensitivity_matrix(d_true, cfg, base_model)`: finite-difference
Jacobian (`delta=0.15`) of a fixed constraint+flux output vector w.r.t. each
of the `D_true` states, evaluated at one reference point (`D_true + 1` LP
solves, cheap). SVD gives an effective-rank estimator
(`(sum(sv))^2 / sum(sv^2)`) and pairwise column cosine similarities.

**Frozen gate (fixed before any result was inspected):** effective rank >=
`0.6 * D_true`, and at most 1 pairwise `|cosine similarity| > 0.9`.

**Result (`data/latent_capacity_mismatch_sensitivity_gate.csv`, primary frozen
reference point T=30, pH=5, DO=40, all z=0.30, state=balanced):**

| D_true (nominal) | effective rank | rank ratio | high-cosine pairs | gate |
|---:|---:|---:|---:|---|
| 3  | 1.01 | 0.34 | 1  | **FAIL** |
| 8  | 4.71 | 0.59 | 1  | FAIL (closest to passing) |
| 12 | 5.05 | 0.42 | 1  | FAIL |
| 16 | 5.35 | 0.33 | 3  | FAIL |
| 20 | 5.40 | 0.27 | 6  | FAIL |
| 24 | 5.80 | 0.24 | 10 | FAIL |

**The gate fails across the sweep, and this is a real finding, not a bug to
patch away** (per Section 9 of the handover spec: *"A D_true condition that
fails this gate is reported as such... not silently patched"*). Two distinct
causes, both mathematically diagnosed rather than assumed:

1. **`z_ox` / `z_bottle` collinearity (~0.9999) is present at every D_true,
   every reference point tested (default, `z=0.5`+`oxidative_stress` state,
   larger perturbation delta), including the unmodified, pre-existing,
   validated core mechanism.** Diagnosis: at these operating points the
   oxygen-exchange bound relief `z_ox` also produces (`apply_interval_constraints`)
   is not binding -- so `z_ox`'s only *realized* effect on the LP solution is
   through the shared `pathway_capacity_upper_bound` term it and `z_bottle`
   both feed into, making their marginal effect vectors parallel. This means
   the *existing, already-shipped* G3 core burden triad may have only ~2
   truly independent causal channels at typical operating conditions, not 3
   as implicitly assumed by calling it "D_true=3". This is worth flagging to
   the team independently of the new experiment.

2. **Every tier-2 state (15-24) is collinear (cosine = 1.0 to machine
   precision) with the tier-1/1b sibling it shares a reaction-bound target
   with.** Diagnosis: this is a mathematical necessity, not an implementation
   bug -- two hidden states whose *only* causal route is scaling the same
   single scalar reaction bound have a marginal effect on every downstream
   output that is, by the chain rule, always a scalar multiple of
   `d(output)/d(bound)`. No amount of differing generation dynamics, time
   constants, or scaling-function shape changes this: shared-single-bound
   coupling is inherently rank-1. Reusing tier-1's ten reaction targets for
   tier-2 (`gem_controller_family._make_scaling_coupling`) was therefore the
   wrong way to extend the library past `D_true≈14` -- it inflates the
   *nominal* dimension without adding *causal* dimension, exactly the
   confound Section 9 warns against.

**Conclusion for this pass:** the controller-family library, as built, does
not currently support a valid `D_true` sweep above roughly 12-14 against the
frozen gate as specified (`D_true=8` under the oxidative-stress reference
point is the only condition that actually passed: rank ratio 0.65, 1
high-cosine pair). Reaching genuinely-independent `D_true` up to 24 requires
either (a) identifying ~10 more distinct real Yeast9 reaction targets for a
true tier-2, or (b) redesigning tier-2 coupling to act on combinations/ratios
of existing targets rather than the same single bound. **Neither is done in
this pass** -- see Section 7 for the recommended next step. The Stage D3
pilot below still exercises nominal `D_true=3/16/24` for pipeline-mechanics
purposes only (data generation -> teacher retrain -> evaluation runs
end-to-end at every nominal dimension), and every plot/table downstream of
this point must report *effective*, not nominal, dimension alongside it.

---

## 3. Clean training-target contract (Stage D1)

`src/yeast_validation/gem_clean_teacher_targets.py`. Deployment inputs: `temperature, pH, DO`
(unchanged). Supervision: `B_total`/`X`/`S_glc` (preferring the noisy
`*_observed` variant from Stage B2 when present) plus reporter columns for
whichever hidden states declare a `reporter_spec` -- partial coverage by
design (Section 12 of the handover spec): of the 21-state library, only 2
(`z_nadph`, `z_proteostasis`) plus the 3 existing capacity reporters and 2
existing burden reporters (`R_ox`, `R_atp`) have any reporter at all; the
remaining 19 states are never directly observed, even during training.
Explicitly excluded (asserted in `tests/test_clean_teacher_no_leakage.py`):
every raw `z_*` name in the library, every internal GEM constraint-bound
column, every internal LP flux column.

---

## 4. Clean teacher retrain (Stage D2)

`src/yeast_validation/run_clean_teacher_retrain.py` reuses the production architecture's
pure-math functions verbatim (`initialize_state_space`, `latent_rollout`,
`fit_heads`, `weighted_ridge_fit`, `predict_columns`, `StateSpaceCheckpoint`)
-- only the target-column list changes. `D_model = 16`, not retuned.

**D_true=3 (current G3, reusing the existing `gem_state_space_*.csv`
artifacts) result:** 5/5 seeds trained; validation product `normalized_rmse =
0.222` (well under the 1.0 training-mean baseline -- real predictive signal
from environment alone). `clean_contract_no_leakage`: **PASS** by
construction.

**Notable, honestly-reported finding:** the clean teacher's product
prediction is **exactly** invariant to reporter shuffling/removal
(`primary_rmse == shuffled_rmse == random_rmse` to floating-point precision).
This is not a bug introduced by the clean retrain -- it is a structural
property of the architecture (Section 1.1): the latent trajectory `z` is a
function of environment and seed only, never of the training targets, and
each output column (including product) gets an independently-fit linear
readout head. Reporters therefore cannot influence product prediction in this
architecture at all. This independently reconfirms
`GENERATOR_VALIDITY.md` Section 6.3's pre-existing finding ("reporter
supervision was not established as necessary") from a different angle. It is
recorded as an `INFO_` row in the acceptance table, not hidden.

---

## 5. Pilot (Stage D3)

`src/yeast_validation/run_latent_capacity_mismatch_pilot.py`. Mechanics-check scale (not
the powered scale): `grid_size=2` (8 environments) x `n_time=13` (12
intervals), vs. the production 125-environment / 48-interval design.

**Worlds:** the pilot itself uses `selected_parameter_config()` (the
baseline/pre-sweep parameter point, per the Section 1.3 provenance decision)
across an environment grid, not the regime-world sweep -- world-level
replicates (`baseline_world`, `atp_limited_world` non-oxidative shift per
Section 7.5, Protocol B allocation replicate per Section 7.4) are scoped to
Stage E/F, not exercised in this mechanics pilot.

**Result** (`data/latent_capacity_mismatch_pilot/summary_d*.csv`,
`grid_size=2` → 8 environments, `n_time=13` → 12 intervals):

| D_true (nominal) | n_extra_states | LP solves | infeasible/unbounded | generation wall-clock | teacher retrain wall-clock | validation product norm-RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 3  | 0  | 288 | 0 / 0 | 219.96 s | 0.05 s | 1.35 |
| 16 | 13 | 288 | 0 / 0 | 217.89 s | 0.05 s | 1.30 |
| 24 | 21 | 288 | 0 / 0 | 218.08 s | 0.05 s | 1.28 |

**Mechanics: PASS for all three.** LP-solve accounting closes exactly
(8 envs × 12 intervals × 3 LP = 288) with zero infeasible/unbounded solves at
every nominal `D_true`, confirming the generalized rollout, the clean-contract
retrain, and the evaluation path all run end-to-end regardless of how many
hidden states are active. Wall-clock is flat across `D_true` (~218-220s) --
confirms the design intent that additional hidden-state dynamics cost is
negligible next to the fixed 3-LP-per-interval cost; the marginal LP cost
that would matter is only in the eventual dataset's environment-grid size and
interval count, not in `D_true` itself.

**Predictive quality: not meaningful at this scale, by design.** `normalized_rmse
> 1.0` (worse than the training-mean constant baseline) at all three
conditions -- expected and not a defect: 6 training environments is far too
few for this closed-form ridge-on-random-features architecture to generalize
to 2 held-out environments. Stage D explicitly scopes power estimation, not a
powered result, to this pass. The reporter-shuffle-invariance finding from
Section 4 reproduces identically at every `D_true` (architectural, not
dimension-dependent).

**Compute estimate for Stage F** (linear extrapolation from 288 LP solves /
~219s, single process, no parallelism assumed): the production dataset design
is 125 environments x 49 time points (48 intervals) x 3 LP/interval = 18,000
LP solves per `D_true` condition, versus this pilot's 288 -- a 62.5x factor,
i.e. **~3.8 hours per `D_true` condition** for dataset generation alone
(teacher retrain and evaluation remain sub-second, closed-form). A full
6-point sweep (`{3,8,12,16,20,24}`) would cost **~23 hours single-process**
for generator-dataset construction, before any DBTL campaign compute (the
existing headline benchmark's 20-campaign, 2-world run alone was 480 exact
cultures x 144 LP = 69,120 solves). This estimate does **not** yet justify
spending compute on `D_true in {16,20,24}` given Section 2.4's finding that
these conditions do not currently achieve their nominal causal dimension --
see Section 7.

---

## 6. Acceptance gate summary

| Gate | Status |
|---|---|
| Stage B: substrate balance opt-in, regression-safe | **PASS** (`tests/test_gem_substrate_balance.py`, 5/5) |
| Stage B: observation noise seeded/reproducible | **PASS** (`tests/test_gem_observation_noise.py`, 5/5) |
| Stage B: oxidative-coupling ablation switch | **PASS** (`tests/test_gem_oxidative_ablation.py`, 3/3) |
| Stage C: D_true=3 bit-for-bit matches gem_backend | **PASS** (`tests/test_gem_controller_family.py`, 8/8) |
| Stage C: causal-rank acceptance gate, full sweep {3,8,12,16,20,24} | **FAIL** (all 6; see Section 2.4 -- real finding, not a bug) |
| Stage D1: clean contract leak-free | **PASS** (`tests/test_clean_teacher_no_leakage.py`, 5/5) |
| Stage D2: clean D=16 teacher beats training-mean baseline (D_true=3, full 125-env dataset) | **PASS** (normalized RMSE 0.222) |
| Stage D3: pilot mechanics run end-to-end for D_true in {3,16,24} | **PASS** (0 infeasible/unbounded, exact LP accounting, all 3 conditions) |
| Stage D3: pilot predictive quality meaningful | **N/A by design** (8-env pilot is a mechanics check, not a powered result) |

**Overall verdict for this pass: the generator hardening (Stage B) and the
clean, leak-free teacher-retrain pipeline (Stage D1/D2) are validated and
ready to freeze. The controller-dimensionality family (Stage C), as currently
built, is NOT ready to freeze** -- it does not yet deliver a `D_true` sweep
whose nominal labels match causally-verified effective dimension, which is a
precondition Section 9 of the handover spec explicitly requires before any
training against it. Proceeding to Stage E/F on the current library would
produce a mismatch curve that is confounded by unverified nominal dimension,
undermining the experiment's core claim.

---

## 7. Next steps

### 7.1 Before Stage E freeze (must fix)

The controller library needs a second design pass so nominal and effective
`D_true` agree, per the two causes diagnosed in Section 2.4:

1. **Tier-2 states need genuinely distinct GSM coupling targets**, not a
   second multiplicative layer on a tier-1 target. Candidates: resolve ~6-10
   more distinct Yeast9 reactions via `gem_backend.reaction_search` (e.g.
   phosphate exchange, CO2 exchange, trehalose/glycogen storage reactions --
   literature-shaped for stress response) and rebuild tier 2 against those,
   re-running `sensitivity_matrix` before accepting any of them.
2. **Investigate the `z_ox`/`z_bottle` collinearity in the core (unmodified)
   mechanism.** Check whether it is specific to the DO=40/mid-burden
   reference point (i.e. the oxygen bound isn't binding there) by sweeping
   the reference point's DO and burden level explicitly, rather than only the
   two supplementary checks run in this pass (which varied `state`/`reference`/
   `delta` but not `cfg.DO` -- an oversight to correct). If the oxidative
   channel only becomes causally distinct from bottleneck under
   oxygen-limited conditions, that should be stated as a scope condition on
   any claim built from it, and considered a separate finding worth raising
   with the wider team regardless of this experiment (the *existing* headline
   `strong_oxidative_burden_world` result may be similarly confounded).
3. Re-run the full `sensitivity_matrix` sweep against the frozen gate
   (unchanged: effective rank >= 0.6*D_true, <=1 high-cosine pair) after (1)
   and (2). Only pick a final `D_true` sweep from conditions that pass.

### 7.2 Once the library passes the causal-rank gate

Run the exact same commands already built and tested in this pass, just
pointed at the corrected library and full production scale:

```bash
# Stage D3 pilot, full scale, per corrected D_true sweep point (~3.8h each, single process):
python src/yeast_validation/run_latent_capacity_mismatch_pilot.py --d-true <D> --grid-size 5 --n-time 49

# Stage D2-equivalent clean retrain once real (non-pilot) datasets exist per D_true:
python src/yeast_validation/run_clean_teacher_retrain.py --d-true <D>
```

### 7.3 Stage E freeze manifest (not yet written)

Must record explicitly, per Section 24 of the handover spec: the corrected
`D_true` sweep values (nominal = effective, gate-verified), `D_model=16`
(unchanged), `MODEL_SEEDS=[11,22,33,44,55]` (unchanged), the parameter set in
use (`selected_parameter_config()`'s pre-sweep base point -- Section 1.3),
worlds (`baseline_world`, `strong_oxidative_burden_world`,
`atp_limited_world`), noise seeds, DBTL budgets, and the frozen causal-rank
gate thresholds (already fixed: 0.6x rank ratio, <=1 high-cosine pair).

### 7.4 Stage F (powered run) -- not run, do not run yet

Blocked on 7.1-7.3. Once unblocked, the powered run reuses
`run_prospective_dbtl_benchmark.py`'s existing paired-campaign/bootstrap
machinery, pointed at the per-`D_true` clean teacher's pseudodata table in
place of the current (leaky) `teacher_pseudodata.csv` -- that wiring is not
yet built in this pass and is the next implementation task after 7.1-7.3
land.
