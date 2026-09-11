# Physiology-Quest Validation Narrative

> This long-form narrative preserves the development record. Start with the
> numbered notebooks for executable, end-to-end experiment phases; module paths
> below are provenance references for the cached evidence.

This document is the canonical scientific narrative for the fixed-environment
Yeast Digital Twin validation chain. It describes synthetic data only.

## Scientific Question

Given one fixed set of culture conditions `e = [I, O, S]`, can a model predict
the complete production trajectory `P(t)`, including for environmental
combinations that were not present during training?

Simulator input: fixed `e` and the configured time grid. Hidden variables:
curve parameters in Experiment 1 and synthetic physiological states in
Experiment 2. Observable outputs: product, and reporters where generated.
Model input at deployment: fixed `e` only. Model output: a full predicted
product curve `P_hat(t | e)`.

The shared product equation is:

`dP/dt = F(t,e) - k_P P`.

## Experiment 1

Experiment 1 changes only the form of `F`.

1A uses `F_A(t,e)=V_A(e)`, where `V_A` is a softplus-transformed environmental
linear score. 1B uses
`F_B(t,e)=V_B(e)[1-exp(-t/tau_B(e))]`. 1C uses
`F_C(t,e)=V_C(e)[1-exp(-t/tau_on(e))]exp(-t/tau_off(e))` with interaction
terms in `V_C`.

Structured models predict positive curve parameters from `e` and solve the
ODE. The direct black-box baseline receives `[e,t]`; it does not receive an
environment sequence.

Held-out-combination best RMSE values are: 1A `0.0069`, 1B
`0.0116`, and 1C `0.0142`.

## Experiment 2

Experiment 2 introduces hidden states `z_A(t)` and `z_B(t)`. Because `e` is
fixed, the activation levels `s_A(e)` and `s_B(e)` are constant for one
culture, while the hidden states evolve by first-order activation and recovery.
Product is generated from one effective production function:

`F_2(t,e,z_A,z_B)=F_base(t,e) * positive_factor(1-alpha_A z_A-alpha_B z_B-alpha_AB z_A z_B)`.

Reporters are generated as lagged measurements of the synthetic states:
`R_A`, `R_B`, and `R_mean`. Reporter labels are used only in training losses.
At deployment every learned model receives only `e*`.

Experiment 2A is an interpolation control. Experiment 2B is the primary
held-out-combination biosensor hypothesis. Experiment 2C corrupts reporter
labels during training with lag, noise, sparse sampling, missingness, gain
variation, and baseline offset.

Held-out 2B best model: `aggregate_reporter_supervision` with product RMSE `0.0047`.

## Data Splits

Splits are culture-level. No time point from a culture appears in more than one
split. The combinatorial holdout removes high induction with high stress from
training while keeping high induction with low stress and low induction with
high stress available. Numerical extrapolation is reported separately and is
not treated as guaranteed.

Split counts per dataset seed: `{'extrapolation': 36, 'heldout_combination': 30, 'interpolation': 30, 'train': 192, 'validation': 24}`.

## Negative and Bounded Results

Reporter supervision is not assumed to help. The shuffled-reporter condition
remains visible as a negative control, and reporter-quality conditions can fail
the 5% usefulness criterion. Numerical extrapolation remains a stress test, not
a supported deployment guarantee.

Safeguards passed `24/24` in the current run.

## Visual True-Versus-Predicted Validation

The primary visual validation uses held-out environmental combinations only.
Representative cultures in `figures/fixed_06_predicted_vs_true_curves.svg` are
selected algorithmically as low, median, and high trajectory-level RMSE cases
for the selected model. `figures/fixed_08_reporter_supervision_heldout.svg`
selects Experiment 2B cultures by the paired RMSE difference between the
product-only latent model and aggregate reporter-supervised model. The scatter
summary in `figures/fixed_11_heldout_prediction_scatter.svg` uses all held-out
product time points.

The curve overlays show that the saved model predictions generally follow the
true held-out trajectory shape, magnitude, startup, and decline. They are also
a necessary check on low aggregate RMSE: a single aggregate value can hide
culture-specific misses in peak height, late decline, or final product.

| Experiment | Selected model | RMSE | normalized RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| 1A | small_mlp_parameter | 0.0077 | 0.0016 | 1.0000 |
| 1B | small_mlp_parameter | 0.0130 | 0.0028 | 0.9999 |
| 1C | small_mlp_parameter | 0.0159 | 0.0129 | 0.9959 |
| 2B | aggregate_reporter_supervision | 0.0050 | 0.0085 | 0.9982 |

## Experiments 3A-3F: dFBA State-Machine Stage

The next canonical stage keeps the same deployment contract but changes the
biological generator:

`fixed environment e = [T, pH, DO] -> complete beta-carotene titer curve B(t)`.

The model input is never a time series. Reporters `R_ox`, `R_ATP`, and `R_ER`
are training-time auxiliary labels only. At deployment, every learned model
receives only `[T, pH, DO]`.

Canonical commands:

```bash
.venv/bin/python src/yeast_validation/audit_surrogate_complexity.py
.venv/bin/python src/yeast_validation/audit_yeast_gem.py
.venv/bin/python src/yeast_validation/run_dfba_state_machine_validation.py
.venv/bin/python src/yeast_validation/run_dfba_state_machine_validation.py --fast
.venv/bin/python src/yeast_validation/run_dfba_state_machine_validation.py --backend yeast_gem
.venv/bin/python -m pytest tests/test_dfba_state_machine_validation.py tests/test_yeast_gem_backend.py
```

Non-fast runs require a local yeast GEM asset configured through
`YEAST_GEM_PATH` or `models/yeast_gem.xml`; no model is downloaded at runtime.
The current repository still refuses non-fast execution until a real repeated
LP/FBA yeast-GEM backend is wired, so it cannot silently use the surrogate and
call it FBA. Fast mode uses the documented reduced stoichiometric surrogate and
still runs Experiments 3A-3F plus the critical safeguards.

New canonical outputs:

```text
data/dfba_environment_grid.csv
data/dfba_split_manifest.csv
data/dfba_trajectories.csv
data/dfba_fluxes.csv
data/dfba_state_transitions.csv
data/dfba_reporters.csv
data/dfba_model_predictions.csv
data/dfba_canonical_metrics.csv
data/dfba_sample_efficiency.csv
data/dfba_reporter_robustness.csv
data/dfba_safeguards.csv
figures/dfba_*.svg
```

The beta-carotene pathway is represented by declared synthetic reactions from
native acetyl-CoA/isoprenoid precursor supply through GGPP, phytoene/lycopene
lumps, and an intentional non-mass-balanced beta-carotene accumulation sink.
In full mode this pathway must be installed on a configured yeast GEM; in fast
mode the same declared constraints are evaluated by a reduced stoichiometric
surrogate.

Held-out-combination headline RMSE:

| Experiment | Best learned model | RMSE |
| --- | --- | ---: |
| 3A | coordinate_conditioned_mlp | 0.0343 |
| 3B | coordinate_conditioned_mlp | 0.0209 |
| 3C | random_smooth_aux_control | 0.0301 |
| 3D | oxidative_reporter_state_space | 0.0314 |
| 3E | er_only_state_space | 0.0208 |
| 3F | coordinate_conditioned_mlp | 0.0209 |

Safeguards passed `16/16`. ER causal isolation is checked by a
finite-difference test showing that changing `z_ER` alone does not alter
growth, beta-carotene flux, precursor flux, ATP maintenance, or ATP pressure.

## Dynamic metabolic generator audit

The audited fast headline run uses `25` time points, `24` simulation intervals,
and `dt=0.25`. Each culture now performs `24` metabolic updates: one documented
surrogate evaluation per interval. The run uses `0` real LP/FBA solves and
`24` reduced-surrogate evaluations per culture. The backend recorded in the
trajectory, flux, prediction, and metric tables is
`reduced_stoichiometric_surrogate`, not a yeast-GEM LP solve. No flux solutions
are reused or cached; the final time-point flux row is a carried-forward
endpoint, not an additional metabolic solve.

The pre-repair audit showed weak hybrid dynamics in specific ways: no cultures
entered `recovering`, no cultures showed beta-carotene plateau or decline, the
median mean adjacent beta-flux change was only `0.0013`, and the reduced
surrogate rather than a full GEM produced the results. After repair, dynamic
culture transition counts are: `0 transitions: 0`, `1 transition: 65`,
`2 transitions: 0`, and `3 or more transitions: 100`. Across dynamic cultures,
the median number of unique beta-carotene flux values is `24`, median mean
adjacent beta-flux change is `0.0088`, median maximum adjacent beta-flux change
is `0.0412`, and median constant-beta-flux interval fraction is `0.0417`.

Active-constraint audit: beta-carotene capacity is active in `0.705` of
intervals and changes in every culture; ATP maintenance is active in `0.727` of
intervals and changes in every culture; the minimum biomass requirement is
active in all intervals and changes by state. Oxygen uptake, precursor capacity,
and growth capacity can be active in the surrogate, but their bounds are fixed
within one culture because `[T, pH, DO]` is fixed, so they do not create dynamic
within-culture bound changes.

Repairs made after the audit:

- changed the simulator to evaluate metabolism once per interval, then advance
  biomass, product, burdens, and the state machine;
- recorded `metabolic_backend`, metabolic update counts, actual FBA solve
  counts, and surrogate evaluation counts in canonical outputs;
- strengthened bottleneck and oxidative burden effects on beta-carotene
  capacity;
- added state-specific repair/recovery multipliers;
- reduced burden accumulation during stress and recovery;
- added state-specific product-degradation multipliers so severe stress can
  create bounded plateaus or declines.

The revised generator passes all `12/12` proof-of-concept dynamic-diversity
criteria in `data/dfba_generator_acceptance_criteria.csv`: `60.6%` of dynamic
cultures have at least two transitions, `60.6%` have at least three transitions,
`10` distinct state sequences occur, `66.7%` show substantial beta-flux change,
`66.7%` show partial recovery after stress, and `24.2%` show plateau or decline.

Experiments 3B-3F were rerun after generator validation without changing the
neural-network architectures. The coordinate-conditioned MLP remains the best
learned held-out model for 3B and 3F (`0.0209` RMSE). Product-only state-space
improved relative to the pre-repair fast run (`3B: 0.0358 -> 0.0325`), but the
state-space model did not become the winner. This supports the interpretation
that the earlier negative state-space result was not only an artifact of a
static generator.

## Surrogate complexity decision gate

The repaired reduced surrogate was audited before any neural architecture
changes or genome-scale dFBA implementation. The audit outputs are
`data/dfba_surrogate_normalized_errors.csv`,
`data/dfba_model_seed_uncertainty.csv`,
`data/dfba_trajectory_pca_summary.csv`,
`data/dfba_simple_baseline_metrics.csv`, and
`data/dfba_surrogate_decision_gate.csv`.

For canonical 3F held-out combinations, the coordinate-conditioned MLP has raw
RMSE `0.0209`, range-normalized RMSE `0.2073`, standard-deviation-normalized
RMSE `0.9992`, and trajectory-normalized RMSE `0.3537`. That error is `0.3598`
of mean final beta-carotene titer, `0.4413` of median trajectory amplitude,
`0.6555` of median neighboring-environment trajectory difference, and `2.6560`
of the median distinct held-out trajectory difference. The product-only
state-space model has held-out raw RMSE `0.0325` and trajectory-normalized
RMSE `0.5485`; the oxidative-plus-ATP reporter state-space model is worse on
held-out 3F with raw RMSE `0.0384` and trajectory-normalized RMSE `0.6676`.

The model-seed uncertainty audit currently has only one paired
state-space-versus-coordinate-MLP comparison per split, so its uncertainty
estimate is a limitation rather than a stable model-ranking claim. The paired
differences, defined as product-only state-space RMSE minus coordinate MLP
RMSE, are `-0.004959` on extrapolation, `0.001970` on interpolation, and
`0.011622` on held-out combinations.

Training-only PCA shows that the surrogate trajectory family is low
dimensional: cumulative explained variance is `0.9945` with one component,
`0.9993` with two, `0.9998` with three, `1.0000` with five, and `1.0000` with
ten. The minimum components required are `1` for 90%, 95%, and 99% variance,
and `2` for 99.9%. Ten-component reconstruction RMSE is `0.0006` on
validation, `0.0016` on interpolation, `0.0011` on held-out combinations, and
`0.0012` on extrapolation.

Very simple 3F held-out baselines are competitive with the neural models but
not enough to make the task effectively solved: mean trajectory RMSE `0.0223`
with trajectory-normalized RMSE `0.4434`, nearest environment `0.0336` and
`0.4468`, linear environment-to-PCA `0.0316` and `0.4744`, and polynomial
environment-to-PCA `0.0181` and `0.3087`.

The decision gate in `data/dfba_surrogate_decision_gate.csv` is
`stop_surrogate_not_solved`. Only `3/7` criteria passed. The strongest reasons
to stop are that the coordinate MLP held-out trajectory-normalized error is
`0.3537`, not 1-2%; the state-space held-out trajectory-normalized error is
`0.5485`; the visual proxy range-normalized error is `0.2073`; and seed
uncertainty is not estimable from one paired model realization.

## Real GEM pilot implementation

A real Yeast9 XML asset was located and selected from the existing raw data directory. The selected file is:

```text
/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml
```

The SHA-256 checksum is:

```text
9fd2c572cace73c2ea835205617313554d1bf89f4ef077f49defbdfa219a4ad7
```

COBRApy loads the model as `yeastGEM_v9__46__0__46__2`, named `The Consensus Genome-Scale Metabolic Model of Yeast`. The model has `4131` reactions, `2806` metabolites, and `1161` genes. The selected objective is `r_2111`; biomass-related reactions detected by audit are `r_2111`, `r_4041`, and `r_4046`. The glucose exchange is `r_1714` (with related glucose-phosphate exchanges `r_4502` and `r_4504`), the oxygen exchange is `r_1992`, and ATP maintenance is `r_4046`. The solver interface is optlang GLPK using `swiglpk`; dependency versions recorded in the run were COBRApy `0.31.1`, optlang `1.9.1`, NumPy `2.5.1`, pandas `2.3.3`, and Python `3.12.13`.

The beta-carotene pilot copies the loaded GEM and does not modify the raw XML. It reuses native Yeast9 GGPP formation through `r_0461` (`s_0189 + s_0943 -> s_0633 + s_1311`) and adds four pilot reactions: `BETA_PHYTOENE_SYNTHASE`, `BETA_PHYTOENE_DESATURASE`, `BETA_LYCOPENE_CYCLASE`, and `DM_beta_carotene_c`. These additions drain native GGPP into a heterologous beta-carotene accumulation sink without creating ATP, reducing equivalents, or precursor-producing cycles. The desaturase/cyclase steps are documented lumped pilot reactions, and the demand reaction is an intentional non-balanced product sink.

The staged optimization protocol is:

1. maximize growth through `r_2111`;
2. constrain growth to a declared state-dependent fraction of maximum growth;
3. maximize `DM_beta_carotene_c`;
4. preserve the product solution and run deterministic pFBA.

For the central static condition (`T=30`, `pH=5.0`, `DO=40`), the unmodified GEM is feasible with maximum growth `0.085844`. The beta-carotene augmented model is feasible and produces static beta-carotene flux `0.158055` under staged optimization. Disabling `BETA_PHYTOENE_SYNTHASE`, `BETA_PHYTOENE_DESATURASE`, or `BETA_LYCOPENE_CYCLASE` eliminates product flux. Increasing required growth reduces product allocation, increasing ATP maintenance reduces product allocation under the same growth rule, and changing oxygen availability changes oxygen flux.

The one-culture dynamic pilot uses `25` time points, `24` simulation intervals, and `dt=0.25`. Each interval performs three real LP optimizations: growth, product, and pFBA. Therefore the pilot records `72` actual LP solves per culture, with `24` growth optimizations, `24` product optimizations, `24` pFBA optimizations, `72` optimal solves, `0` infeasible solves, `0` unbounded solves, and `0` surrogate evaluations. The recorded backend is `yeast_gem_lp`. Final pilot biomass is `0.297009` and final accumulated beta-carotene is `0.163431`. Dynamic burdens alter oxygen, ATP-maintenance, and pathway-capacity bounds; those altered bounds change beta-carotene, biomass, oxygen, and GGPP fluxes. ER perturbation changes no causal output (`max_abs=0`). The pilot acceptance table passes all `12/12` criteria.

The small diagnostic grid uses `27` environments (`3` temperatures, `3` pH values, `3` DO values) with a short `7`-time-point diagnostic horizon. It completed `486` real LP solves and `0` surrogate evaluations. Feasibility rate was `1.0`; beta-carotene flux ranged from `0.101809` to `0.301183`; growth flux ranged from `0.181619` to `0.363223`; there were `72` rounded beta-flux values and `72` active constraint vectors. At this short horizon, trajectory PCA remains very low-dimensional: PC1 explains `99.9984%` of curve variance. This confirms genuine LP-backed generation and richer flux/constraint accounting than the reduced surrogate, but it does not yet prove that beta-carotene trajectory shapes are high-dimensional enough for the full state-space claim.

## Real GEM temporal-diversity validation

The real-GEM generator was extended from the pilot/small-grid checks to a full-horizon `27`-environment grid with `25` time points and `24` dynamic intervals per environment. The run completed `1944` actual Yeast9 LP optimizations (`27 * 24 * 3`: growth, product, and pFBA), with `0` surrogate evaluations, `0` ignored infeasible solves, and `0` ignored unbounded solves. The generated local artifacts are `data/gem_full_horizon_grid_trajectories.csv`, `data/gem_full_horizon_grid_fluxes.csv`, `data/gem_full_horizon_grid_constraints.csv`, `data/gem_full_horizon_grid_states.csv`, `data/gem_full_horizon_grid_solver_accounting.csv`, `data/gem_growth_quantity_audit.csv`, `data/gem_flux_to_burden_audit.csv`, `data/gem_flux_to_burden_summary.csv`, `data/gem_allocation_protocol_comparison.csv`, the raw/normalized/flux/slope/biomass PCA tables, and the SVG/PNG `figures/gem_*` mechanism plots. These generated files are intentionally ignored by Git.

The growth-number discrepancy is now explicitly audited. The earlier `0.085844` value is the unmodified Yeast9 default-medium maximum growth, where the glucose exchange lower bound is `-1.0`. The full-horizon interval biomass fluxes use the constrained augmented pilot medium with glucose uptake `-10`, DO-dependent oxygen bounds, ATP-maintenance constraints, state-dependent growth preservation, product allocation, and pFBA. Under those interval rules, biomass flux spans `0.174405` to `0.363223`, so it is not the same quantity as the raw default-medium maximum.

Burden updates are now flux-driven and audited at term level. The constraint audit records oxidative generation/repair from oxygen and product flux, ATP demand/supply/repair from maintenance, growth, product, and oxygen flux, GGPP loading versus beta-carotene clearance, and ER generation/repair. Intervention safeguards pass: increasing respiratory flux raises oxidative burden, increasing ATP surplus lowers ATP pressure, increasing GGPP relative to beta-carotene raises bottleneck burden, and ER-only perturbation remains causally isolated from GEM flux outputs.

Three allocation protocols were compared with genuine LP solves: Protocol A is the original growth-preserve/product-maximize/pFBA rule, Protocol B uses a fixed heterologous product floor followed by growth/pFBA, and Protocol C uses a softer product target after preserved growth. The short protocol comparison used `5` time points per environment and found different allocation behavior: mean beta-carotene flux was `0.191981` for A, `0.058595` for B, and `0.191981` for C; mean growth flux was `0.252707` for A, `0.342512` for B, and `0.252707` for C.

The temporal-diversity acceptance result is `gem_flux_rich_but_product_shape_simple`, not `gem_generator_ready_for_ml`. The generator passes real-Yeast9 use, zero-surrogate accounting, growth-unit reconciliation, flux-to-burden feedback, multiple state sequences, multiple beta-flux regimes, allocation protocol difference, infeasible/unbounded solve checks, and ER isolation. It fails the recovery/plateau-or-decline thresholds and remains too low-dimensional in product-shape PCA: raw product PC1 explains `99.9944%`, amplitude-normalized product PC1 explains `99.0696%`, and normalized PC1 does not fall below the `98%` acceptance threshold. Therefore Experiments 3A-3F remain intentionally blocked on this real-GEM generator until the feedback design or horizon is revised enough to create stronger product-shape diversity.

Canonical commands:

```bash
export YEAST_GEM_PATH="/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"

.venv/bin/python src/yeast_validation/audit_yeast_gem.py
.venv/bin/python src/yeast_validation/run_gem_pilot.py
.venv/bin/python src/yeast_validation/run_gem_small_grid.py
.venv/bin/python src/yeast_validation/run_gem_temporal_diversity.py --protocol-grid-n-time 5

.venv/bin/python -m pytest \
  tests/test_yeast_gem_backend.py \
  tests/test_beta_carotene_gem_pathway.py \
  tests/test_gem_dynamic_pilot.py \
  tests/test_gem_temporal_diversity.py
```

The full neural Experiments 3A-3F were not rerun on GEM-generated data. That remains intentionally blocked because the full-horizon real-GEM temporal-diversity validation classified the generator as flux-rich but still product-shape simple.

## Dynamic heterologous pathway-capacity limits

The real-GEM generator now has an explicit `capacity_mode` control. The previous generator is preserved as `no_dynamic_capacity`; the new mechanism ladder adds `constant_pathway_caps`, `environment_initial_capacity`, `dynamic_damage_recovery`, and `dynamic_congestion_feedback`. All modes use the same Yeast9 asset, copied beta-carotene pathway, environment grid, time integration, burden equations, state-machine thresholds, allocation protocol, and LP solver. The only intended difference is the pathway-capacity mechanism.

The dynamic-capacity states are `E_PSY`, `E_DES`, and `E_CYC` for `BETA_PHYTOENE_SYNTHASE`, `BETA_PHYTOENE_DESATURASE`, and `BETA_LYCOPENE_CYCLASE`. They are bounded to `[0.08, 1]`. Initial values use smooth deterministic environment factors:

```text
E_j(0) = clip(E_j,base * f_T,j(T) * f_pH,j(pH) * f_DO,j(DO), 0.08, 1)
```

The imposed GEM bounds are reaction-specific:

```text
0 <= v_PSY <= V_PSY,max E_PSY
0 <= v_DES <= V_DES,max E_DES
0 <= v_CYC <= V_CYC,max E_CYC
```

The selected parameter set came from a six-candidate, five-environment deterministic pilot sweep using generator criteria only, with no neural metrics. The selected values are `V_PSY,max=0.391`, `V_DES,max=0.322`, `V_CYC,max=0.2645`, `E_min=0.08`, synthesis rates `0.084/0.072/0.066`, baseline decay rates `0.018/0.024/0.022`, oxidative sensitivities `0.040/0.125/0.160`, and congestion sensitivities `0.320` for phytoene congestion and `0.280` for lycopene congestion. The selected candidate had no infeasible or unbounded solves, nonzero production, no universal shutdown, and `3` distinct state sequences in the pilot subset.

Capacity updates occur only after the current interval's real GEM solution:

```text
E_j,k+1 = clip(E_j,k + dt * (S_j,k - D_j,k E_j,k), 0.08, 1)
```

Synthesis uses the same environment factors plus state-specific recovery multipliers. Damage uses predeclared enzyme sensitivities: downstream `DES` and `CYC` are more oxidative/ATP sensitive than upstream `PSY`; `PSY` is more bottleneck/congestion sensitive. Congestion feedback uses both flux mismatch and capacity pressure:

```text
C_phytoene = max(0, v_PSY - v_DES) plus PSY-to-DES capacity pressure
C_lycopene = max(0, v_DES - v_CYC) plus DES-to-CYC capacity pressure
```

The full selected dynamic-congestion run used `27` environments, `49` time points, and `48` intervals per environment. It completed `3888` actual Yeast9 LP solves (`1296` growth, `1296` product, `1296` pFBA), with `0` surrogate evaluations, `0` infeasible solves, and `0` unbounded solves. Saved local artifacts include `data/gem_dynamic_capacity_trajectories.csv`, `data/gem_dynamic_capacity_fluxes.csv`, `data/gem_dynamic_capacity_states.csv`, `data/gem_dynamic_capacity_constraints.csv`, `data/gem_dynamic_capacity_audit.csv`, `data/gem_dynamic_capacity_solver_accounting.csv`, `data/gem_dynamic_capacity_pca.csv`, `data/gem_dynamic_capacity_shape_metrics.csv`, `data/gem_dynamic_capacity_acceptance.csv`, `data/gem_dynamic_capacity_control_comparison.csv`, `data/gem_capacity_parameter_sweep.csv`, and `data/gem_capacity_selected_parameters.csv`. Mechanism figures were saved as SVG/PNG under `figures/gem_dynamic_capacity_*` and `figures/gem_capacity_control_comparison.*`. These generated files are intentionally ignored by Git.

The capacity states changed substantially: minimum capacity across cultures ranged from `0.284153` to `0.381309`, mean minimum capacity was `0.329473`, and mean capacity-loss rate was `0.027352` per time unit. Mean final titer was `0.598243`, mean beta-carotene flux was `0.112274`, recovery-like slope behavior was detected in all cultures, and multiple product-slope regimes were detected in all cultures. However, plateau frequency was `0.0`, decline frequency was `0.0`, and limiting-pathway-reaction switch frequency was `0.0`.

The mechanism ladder showed that dynamic mechanisms changed amplitude and improved shape PCA relative to fixed caps, but did not create enough distinct trajectory families. Normalized-shape PC1 values were `0.999983` for `no_dynamic_capacity`, `0.999959` for `constant_pathway_caps`, `0.996724` for `environment_initial_capacity`, `0.996127` for `dynamic_damage_recovery`, and `0.991558` for the short `dynamic_congestion_feedback` ladder control. In the full 49-point grid, raw product PC1 was `99.7633%`, amplitude-normalized product PC1 was `99.8049%`, beta-carotene-flux PC1 was `97.8008%`, product-slope PC1 was `99.7538%`, and active-biomass PC1 was `99.9797%`. The normalized product trajectory family required `2` PCs for `99.9%` variance, but had only `2` distinct normalized trajectory clusters.

The final dynamic-capacity acceptance classification is `gem_capacity_feedback_too_weak`. Mandatory technical safeguards passed: real Yeast9 was used, zero surrogate evaluations occurred, LP accounting was complete, infeasible/unbounded solves were not ignored, dynamic capacity bounds altered GEM optima, capacity states depended on prior solved fluxes and burdens, ER isolation remained accepted, and the deterministic run is reproducible. The generator did not unlock state-space ML because normalized-shape PC1 did not fall below the requested `99%` threshold, did not reach the stronger `98%` target, did not produce at least three normalized trajectory families, and did not produce limiting-reaction switches in at least `30%` of cultures. No neural models were tuned or run.

Canonical dynamic-capacity commands:

```bash
export YEAST_GEM_PATH="/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"

.venv/bin/python src/yeast_validation/run_gem_dynamic_capacity.py --mode parameter-sweep
.venv/bin/python src/yeast_validation/run_gem_dynamic_capacity.py --mode full-grid --n-time 49
.venv/bin/python src/yeast_validation/audit_gem_dynamic_capacity.py

.venv/bin/python -m pytest \
  tests/test_yeast_gem_backend.py \
  tests/test_gem_temporal_diversity.py \
  tests/test_gem_dynamic_capacity.py
```

## Diagnostic classical neural state-space experiment

This diagnostic was run under an explicit override of the previous generator gate. The generator classification remains `gem_capacity_feedback_too_weak`; this section asks only whether a classical neural state-space model can exploit hidden flux/capacity dynamics better than direct env-to-curve baselines when deployment input is restricted to fixed `[T, pH, DO]`. The neural validation script does not import COBRApy or the GEM backend, does not call Yeast9 at deployment, and uses reporters/states only as optional training-time labels.

The expanded deterministic grid used `5` temperatures (`27, 28.5, 30, 31.5, 33`), `5` pH values (`4.5, 4.75, 5.0, 5.25, 5.5`), and `5` DO values (`20, 35, 50, 65, 80`): `125` cultures, `49` time points, and `48` intervals. The real Yeast9 dynamic-capacity generator completed `18000` actual LP solves (`6000` growth, `6000` product, `6000` pFBA), with `0` surrogate evaluations, `0` infeasible solves, and `0` unbounded solves. Split counts were environment-level with no timepoint leakage: `77` train, `15` validation, `15` interpolation, `10` heldout_combination, and `8` extrapolation. Preprocessing statistics and PCA bases were fit on train cultures only.

The primary state-space form was:

```text
z0 = g(e)
z_k+1 = z_k + dt f(z_k, e)
B_hat,k = h(z_k)
```

Latent dimensions `4`, `8`, and `16` were validation-selected for every seed. The mechanistic-rate variant used a positive rate decoder, `B_hat,k+1 = B_hat,k + dt softplus(h(z_k))`. Direct baselines were mean training trajectory, nearest environment, linear env-to-PCA, validation-selected polynomial env-to-PCA, tree env-to-PCA, multi-output MLP, and coordinate-conditioned MLP. State-space variants covered product-only, mechanistic-rate, oxidative reporter, ATP reporter, oxidative+ATP, capacity reporters, combined reporters, ER-only, shuffled reporter control, random smooth auxiliary control, and oracle true state/capacity supervision. Model seeds were `11, 22, 33, 44, 55`.

Heldout-combination trajectory-normalized RMSE means were led by direct models: `polynomial_environment_pca=0.002492`, `multi_output_mlp=0.005622`, `coordinate_conditioned_mlp=0.008639`, then `mechanistic_rate_state_space=0.012561`. Product-only state-space was `0.023712`; ATP and oxidative reporter variants were `0.020308` and `0.020457`; combined reporters were `0.029127`; oracle state/capacity supervision was `0.028356`. The paired best-state-space versus best-direct comparison over seeds had mean delta `+0.008895` state-space-minus-direct, win fraction `0.0`, bootstrap CI `[0.007374, 0.011401]`, so the diagnostic conclusion is `state_space_matches_direct_models`, not state-space improvement. Reporter conclusion is `reporter_supervision_not_supported`.

Sample efficiency on heldout combinations also favored direct env-to-curve structure at full data: polynomial env-to-PCA improved from `0.049894` at `25%` train cultures to `0.002492` at `100%`, while product-only state-space went from `0.064812` to `0.023712` and combined reporter state-space from `0.062439` to `0.029127`. Controls behaved as expected: shuffled reporters (`0.030030`), ER-only (`0.032332`), and random smooth auxiliary (`0.049895`) did not create useful signal; oracle state/capacity labels also failed to beat direct polynomial PCA, consistent with accumulated beta-carotene remaining effectively low-dimensional in this generator.

Generated diagnostic artifacts are intentionally ignored by Git: `data/gem_state_space_environment_grid.csv`, `data/gem_state_space_trajectories.csv`, `data/gem_state_space_fluxes.csv`, `data/gem_state_space_states.csv`, `data/gem_state_space_reporters.csv`, `data/gem_state_space_solver_accounting.csv`, `data/gem_state_space_split_manifest.csv`, `data/gem_state_space_model_predictions.csv`, `data/gem_state_space_metrics.csv`, `data/gem_state_space_seed_summary.csv`, `data/gem_state_space_paired_comparisons.csv`, `data/gem_state_space_sample_efficiency.csv`, `data/gem_state_space_reporter_metrics.csv`, `data/gem_state_space_training_log.csv`, `data/gem_state_space_diagnostic_conclusion.csv`, and SVG/PNG figures `figures/gem_state_space_*`.

Canonical diagnostic commands:

```bash
.venv/bin/python src/yeast_validation/generate_gem_state_space_dataset.py
.venv/bin/python src/yeast_validation/run_gem_state_space_validation.py
.venv/bin/python src/yeast_validation/audit_gem_state_space_validation.py

.venv/bin/python -m pytest \
  tests/test_gem_dynamic_capacity.py \
  tests/test_gem_state_space_validation.py
```

## Reporter-grounded teacher to hybrid pFBA distillation

The next chronological layer reuses the existing expanded real-Yeast9
Experiment 3 state-space dataset rather than regenerating GEM trajectories.
The audited dataset remains `125` fixed-environment cultures, `49` time
points, `48` dynamic intervals, culture-level splits, real Yeast9 LP/pFBA
solutions, reporter trajectories, capacity/burden states, flux summaries, and
`0` surrogate evaluations. The interval constraint table is discovered from
`data/gem_state_space_constraints.csv` when present, otherwise from the
completed audited checkpoint
`data/gem_state_space_generator_constraints.partial.csv`.

The implemented Stage 1 teacher keeps the deployment contract fixed:
`[temperature, pH, DO]` is the only input. Biosensors, internal states, compact
interface controls, and flux summaries are labels only. The teacher architecture
is a classical neural state-space rollout,
`z0 = g(e)`, `z[k+1] = z[k] + dt f(z[k], e)`, with separate heads for product,
oxidative reporter, ATP reporter, pathway-capacity reporters, compact metabolic
interface controls, and selected flux summaries. Latent coordinates are not
claimed to be true yeast regulatory states.

The compact interface uses only recorded audited quantities: oxygen exchange
lower bound, ATP-maintenance lower bound, preserved growth fraction, PSY/DES/CYC
effective pathway bounds, and the burden proxies `z_ox`, `z_atp`, and
`z_bottle`. These controls map to the existing Yeast9 staged optimization and
pFBA protocol; they do not expose thousands of independent reaction bounds.

Stage 1 trained the requested teacher variants and controls across seeds
`11, 22, 33, 44, 55`, with latent dimensions `4, 8, 16` selected by validation.
The selected full reporter+interface ensemble passed its gate in
`data/teacher_model_selection.csv`: product, reporter, and interface validation
errors were usable; shuffled and random-smooth controls were worse on
physiological reporter metrics; rollouts were finite and bounded. This is
evidence that the existing real-GEM data can train a stable reporter-grounded
interpolation teacher, not evidence that the learned controls are the true yeast
regulatory system.

Stage 2 froze that ensemble and generated `5,000` Latin-hypercube conditions
inside the original support (`27-33 C`, `pH 4.5-5.5`, `DO 20-80%`). It saved
teacher means, uncertainties, confidence weights, pseudo-train/pseudo-validation
splits, and plot grids. The Stage 2 audit passed finite-output, bounded-control,
domain, confidence-weight, and split-integrity checks. Dense teacher data are
knowledge distillation/interpolation artifacts, not new biological observations.

Stage 3 now trains an open-loop regulatory hybrid student without
differentiating through GLPK. The student predicts reporters, compact interface
controls, and flux summaries from `[T, pH, DO]`; the primary hybrid has no direct
neural product head. Exact product generation is implemented by applying the
predicted compact interface to a fresh Yeast9 augmented model each interval,
running the existing staged growth/product/pFBA protocol, and accumulating
beta-carotene from solved product flux and biomass.

The exact replay pilot exposed one implementation defect before the full run:
ATP-maintenance lower bounds above the model's old upper bound were assigned
before raising the upper bound. COBRA correctly rejected the transient invalid
state. The replay mapping now follows the original generator order, raising the
ATP-maintenance upper bound before assigning the lower bound. A one-culture
oracle/teacher/hybrid exact replay pilot then completed `432` LP stages
(`3` replay types * `48` intervals * growth/product/pFBA), with `432` optimal
solves, `0` infeasible solves, `0` unbounded solves, `0` solver errors, `0`
skipped intervals, and `0` surrogate evaluations. Oracle compact-control replay
matched the saved original GEM trajectory to numerical precision for that smoke
culture (`B_total RMSE = 2.61e-16`), confirming that the compact interface and
replay protocol are sufficient when the recorded controls are used.

The full `125`-culture exact replay is now being generated from frozen assets,
without retraining. A five-process full-shard attempt was too memory heavy for
the laptop because each worker repeatedly rebuilt Yeast9/COBRA/SymPy model
state and held solver objects during staged growth/product/pFBA intervals. The
safe runbook is now one replay type and one culture per process, with BLAS
thread counts capped and the replay code explicitly clearing interval models,
garbage collection, and the SymPy cache between intervals.

As of the current checkpoint, verified exact replay artifacts cover cultures
`0-21` across oracle, teacher, and hybrid semantics: `22` culture triples and
`66` culture/replay pairs. Current exact replay accounting is `3,168` actual LP
solves per replay type and `9,504` total LP solves across oracle, teacher, and
hybrid, with actual solves equal to optimal solves and zero infeasible,
unbounded, solver-error, skipped-interval, and surrogate-evaluation counters.
Some early checkpoint files were produced as five-culture micro-shards
(`num_shards=25`) before switching to one-culture shards (`num_shards=125`), so
completion is tracked by `culture_id + replay_type` in
`data/hybrid_exact_replay_completion.csv` rather than by shard filenames alone.

Preliminary metrics for completed cultures `0-21` are saved in
`data/hybrid_exact_replay_preliminary_metrics.csv`. Oracle replay remains at
numerical precision against the original GEM generator (`raw RMSE 1.94e-16`,
trajectory-normalized RMSE `3.76e-16`). Teacher-control pFBA versus teacher
direct product has raw RMSE `0.0194`; hybrid-control pFBA versus teacher-control
pFBA has raw RMSE `0.00355`; total hybrid pFBA versus the original GEM has raw
RMSE `0.00449` and trajectory-normalized RMSE `0.0101` on this preliminary
subset. The formal Stage 3 status is still
`stage_3_blocked_full_125_culture_exact_rollout_incomplete`; Stage 4 exact
edited-strain validation and acceptance remain blocked until Stage 3 passes.

Canonical commands:

```bash
.venv/bin/python src/yeast_validation/run_reporter_grounded_hybrid_distillation.py --dense-samples 5000

# Exact replay smoke/pilot without retraining.
.venv/bin/python src/yeast_validation/run_reporter_grounded_hybrid_distillation.py \
  --exact-stage3-validation \
  --pilot-only \
  --pilot-cultures 1

# Optional expensive exact Yeast9 rollout after inspecting Stage 1-2 gates.
.venv/bin/python src/yeast_validation/run_reporter_grounded_hybrid_distillation.py \
  --exact-stage3-validation \
  --pilot-cultures 18 \
  --max-pfba-cultures 125

# Conservative one-culture exact replay shard. Repeat replay-types as
# oracle, teacher, and hybrid, advancing shard-index after all three pass.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python src/yeast_validation/run_reporter_grounded_hybrid_distillation.py \
  --exact-stage3-validation \
  --replay-types oracle \
  --shard-index 22 \
  --num-shards 125 \
  --max-pfba-cultures 125 \
  --resume

.venv/bin/python -m pytest \
  tests/test_gem_state_space_validation.py \
  tests/test_reporter_grounded_hybrid_distillation.py
```

Generated artifacts are intentionally ignored by Git and include
`data/hybrid_distillation_data_inventory.csv`,
`data/hybrid_interface_inventory.csv`, `data/teacher_model_selection.csv`,
`data/teacher_channel_metrics.csv`, `data/teacher_interface_metrics.csv`,
`data/teacher_ensemble_manifest.csv`,
`data/teacher_dense_environment_samples.csv`, `data/teacher_pseudodata.csv`,
`data/teacher_pseudodata_uncertainty.csv`,
`data/teacher_pseudodata_audit.csv`, `data/hybrid_student_manifest.csv`,
`data/hybrid_student_metrics.csv`, `data/hybrid_student_ablation_metrics.csv`,
`data/hybrid_student_acceptance.csv`,
`results/reporter_grounded_hybrid_distillation/*`, and
`figures/teacher_*.svg` / `figures/hybrid_student_*.svg`.

## Observable-only latent-interface validation

The privileged hybrid above is now paired with a narrower observability test:
can a GSM-compatible controller be inferred when the biological encoder sees
only wet-lab-plausible observables? The new script
`src/yeast_validation/run_observable_latent_interface_validation.py` reuses the fixed-strain
`125`-culture Yeast9 environment sweep, writes an explicit lockbox audit, trains
a frozen differentiable ridge surrogate for gradient propagation, and then
trains two observable-only encoders: product/biomass only and
product/biomass plus reporters. Lockbox variables such as
`PSY_effective_upper_bound`, `DES_effective_upper_bound`,
`CYC_effective_upper_bound`, `gamma_growth_fraction`, `z_ox`, `z_atp`, and
`z_bottle` are excluded from observable-only encoder inputs, losses, early
stopping, and model selection.

Initial narrow result, using seeds `11,22,33`, `120` training steps, and a
one-interval real Yeast9/pFBA smoke replay on one culture from each primary
split: observable-only product prediction is `PARTIAL`
(`heldout_combination` surrogate product NRMSE `0.657` without reporters and
`5.846` with reporters); real-GSM transfer is only a `LIMITED_PASS` smoke check
(mean exact-replay NRMSE `0.002` over the limited rows); post-hoc controller
recovery is `PARTIAL`; ambiguity analysis still indicates non-identifiability.
Reporters improved post-hoc controller recovery (`0.276` to `0.237` normalized
MAE) but did not improve held-out product prediction in this run. This does not
justify advancing to DBTL/metabolic-engineering claims without scaling the
exact real-GSM replay and diagnosing the reporter loss.

Canonical command:

```bash
.venv/bin/python src/yeast_validation/run_observable_latent_interface_validation.py \
  --seeds 11 22 33 \
  --steps 120 \
  --ambiguity-envs 3 \
  --ambiguity-samples 120 \
  --exact-cultures-per-split 1 \
  --exact-max-intervals 1

.venv/bin/pytest -q tests/test_observable_latent_interface_validation.py
```

Primary outputs are
`data/observable_latent_interface_decision_table.csv`,
`data/observable_latent_interface_metrics.csv`,
`data/observable_latent_interface_latent_recovery.csv`,
`data/observable_latent_interface_ambiguity.csv`,
`results/observable_latent_interface/observable_latent_interface_report.md`,
and `figures/observable_latent_interface_*.svg`.

## Inverse GSM controller calibration

The next diagnostic strips away the learned encoder entirely. It asks whether a
single culture's compact controller can be inverse-calibrated from plausible
measurements while every candidate controller is evaluated by the real
Yeast9/pFBA backend. The optimizer sees only `[T, pH, DO]`, product, biomass,
and optionally reporter measurements. It does not see true oxygen/ATP/pathway
controller trajectories, `z_ox`, `z_atp`, `z_bottle`, hidden states, or fluxes.

`src/yeast_validation/run_inverse_gsm_calibration.py` freezes an algorithmic five-culture
diagnostic manifest, writes a lockbox table, and defines controller bounds from
the global training split. Stage A deliberately avoids the earlier
knot-interpolation overparameterization: it optimizes one single-interval direct
control vector with `9` free parameters, one per compact controller channel.
`data/inverse_gsm_parameterization_audit.csv` records this setup. The objective
is functional reconstruction only: `GSM(u_inferred) ~= measured phenotype`.
Post-hoc controller-distance metrics are diagnostic and are not used for
fitting.

Canonical controlled Stage A command:

```bash
.venv/bin/python src/yeast_validation/run_inverse_gsm_calibration.py \
  --stage-a \
  --random-samples 100 \
  --max-evaluations 500 \
  --restart-seeds 11 22 33 44 55 \
  --optimizers differential_evolution powell_multistart annealing \
  --stage-a-reporter-weights 0.0 0.1 0.3
```

The bounded real-backend checkpoint used
`--random-samples 10 --max-evaluations 5 --stage-a-reporter-weights 0.0` to
avoid pretending the full exact-GSM budget is cheap. It reached oracle replay
loss `2.32e-14`, random median loss `0.398`, training-mean loss `0.0814`, and
best inverse loss `0.0617` from `powell_multistart`, closing `24.2%` of the
oracle gap. The Stage A gate still stops: the random distribution was not the
required `100` samples, and optimizer success was not reproducible across
restart seeds. Reporter ablations were implemented but not run in this
checkpoint because Stage A product/biomass reproducibility did not pass.

The current decision is therefore `stop_after_stage_A`: the inverse path is
label-clean and no longer a trivial plumbing-only pilot, but it is not yet
evidence for inferring controller labels across the full training set.

Primary outputs are
`data/inverse_gsm_calibration_culture_manifest.csv`,
`data/inverse_gsm_calibration_lockbox.csv`,
`data/inverse_gsm_parameterization_audit.csv`,
`data/inverse_gsm_stageA_reference_losses.csv`,
`data/inverse_gsm_stageA_random_baseline.csv`,
`data/inverse_gsm_stageA_optimizer_comparison.csv`,
`data/inverse_gsm_stageA_convergence.csv`,
`data/inverse_gsm_stageA_1d_landscape.csv`,
`data/inverse_gsm_stageA_gate.csv`,
`data/inverse_gsm_stageA_decision_table.csv`,
`data/inverse_gsm_stageA_controller_solutions.csv`,
`data/inverse_gsm_stageA_loss_breakdown.csv`,
`data/inverse_gsm_stageA_solver_accounting.csv`,
`results/inverse_gsm_calibration/inverse_gsm_stageA_summary.md`, and
`figures/inverse_gsm_stageA_*.svg`.

## Stage 4: probabilistic strain-environment design

Strain engineering is the next synthetic test because the reporter-grounded
hybrid is meant to support design, not only replay. The Stage 4 implementation
therefore freezes the learned regulatory controller and asks whether moderate
metabolic edits plus fixed `[T, pH, DO]` can be ranked by a probabilistic,
trajectory-producing design system. The first transfer assumption is bounded:
the controller is approximately transferable only for moderate metabolic edits,
while the editable metabolic layer handles direct flux consequences.

Large edited FBA grids are avoided. The design system reuses the existing
baseline Yeast9 trajectories, compact controls, reporter/state labels, teacher
pseudo-data, hybrid checkpoints, and exact replay ledgers. Exact Yeast9/pFBA is
reserved for planned calibration, transfer validation, and shortlisted
verification cases. Until Stage 3 passes, those exact edited runs are explicitly
marked `pending_stage3_gate`.

The bounded edit library contains `10` interpretable dimensions covering
competing precursor sinks, precursor supply, PSY/DES/CYC capacity, ATP support,
oxygen support, export capacity, carbon uptake, and product-loss reduction. A
transfer-distance score combines edit count, edit magnitude, cost, and risk,
and the optimiser is constrained to this compact edit space rather than
arbitrary whole-genome reaction-bound manipulation.

The current fast metabolic layer is a baseline-exact, bootstrap ridge residual
surrogate. It predicts interval flux summaries and integrates biomass and
beta-carotene trajectories; it does not directly output final titer alone. It
combines ensemble disagreement, edit-transfer distance, and baseline calibration
residuals into practical uncertainty summaries. Because no exact edited cases
have been run yet, the uncertainty audit is labelled
`baseline_only_uncertainty_not_validated_for_edits`.

The first gated audit searched `20,000` surrogate candidates and produced a
diverse `10`-candidate shortlist in
`data/stage4_ranked_strain_environment_candidates.csv`. These candidates are
useful as planned verification targets only. Single-edit transfer,
combination-transfer, exact verification, and optimisation-regret tables are
present, but they are blocked/pending rather than accepted scientific evidence.

Bounded interpretation: this implementation tests whether a reporter-grounded
regulatory model, combined with an editable metabolic model, can eventually
rank moderate strain edits and environments with calibrated uncertainty. It
does not validate real engineered yeast strains, and it does not claim Stage 4
success before exact edited Yeast9 verification.

Formal Stage 4 status:
`stage_4_blocked_stage3_exact_replay_incomplete`.

## Parallel screening versus sequential optimisation benchmark

The design question is now explicit: under a bounded edit library and fixed
environment controls, should the project spend measurements on one large
parallel screen, on q-batched adaptive optimisation, or on fully sequential
optimisation? The implemented benchmark keeps this as an efficiency comparison
over measurements, strains, and rounds, not as another bulk Yeast9/pFBA data
generation job.

The candidate pool is bounded to the Stage 4 edit library plus `[T, pH, DO]`
limits. Every candidate has a deterministic hash that includes edit values,
environment values, simulator configuration, regulatory-tree version, GEM
checksum placeholder, time grid, integration convention, and solver/backend
configuration. The oracle cache is global, so duplicate candidates are computed
once, while the observation ledger remains method-specific so adaptive methods
cannot see other methods' observations.

Methods now registered are random, space-filling, static GEM target ranking,
reporter heuristic, black-box UCB, reporter-informed/hybrid UCB, direct product
surrogate, hybrid digital twin, and an oracle-upper-bound reference placeholder.
The reference row is registered but is not a competing method in the surrogate
pilot. The direct-surrogate and hybrid digital-twin adaptive methods share the
same UCB outer optimiser; only their model representation and prior score
differ.

The current pilot is deliberately labelled
`surrogate_oracle_for_scheduler_pilot`. It validates scheduler mechanics,
cache reuse, observation privacy, budget accounting, and metric generation, but
it is not exact edited Yeast9 evidence. With `180` candidates it produced
`145` global cache entries, `1,944` method-specific measurements, `360` round
records, and `1,118` strain-construction records across `12` competing methods
and `99` runs. Exact dynamic pFBA rollouts are `0`, total LP solves are `0`,
and regret is reported only as `best_known_regret_incomplete_pool`.

Representative Stage 3 acceptance is now separated from final Stage 3
acceptance. The representative manifest covers `22` completed exact replay
culture triples across train, validation, interpolation, heldout-combination,
and extrapolation splits and passes the preliminary metric gate. That permits
a small scheduler pilot, but final Stage 4 claims still require the full
`125`-culture Stage 3 exact replay and exact edited-strain pFBA validation.

The key interpretation is narrow: the benchmark currently shows that the
experimental design machinery is ready and memory-safe. It does not yet show
that the hybrid digital twin wins under the real hidden Yeast9/pFBA oracle.

## First exact dynamic-pFBA finite-pool benchmark

The scheduler-only oracle has now been replaced, for one bounded benchmark, by
a fully cached exact Yeast9/pFBA reference pool driven by the decision-tree
regulator in `gem_backend`. The hidden oracle is not the learned hybrid model:
it is the exact dynamic growth/product/pFBA loop with `49` time points and
`48` intervals. Each candidate therefore requires `144` LP solves.

The exact pool contains two frozen regimes:
`weak_dynamic_regulation` and `strong_dynamic_regulation`. Each regime has
`48` candidates split into `20` edit-surrogate calibration, `8` uncertainty
calibration, `16` benchmark reference-pool, and `4` final untouched
verification candidates. All `96/96` rollouts completed with `13,824` actual
LP solves, `13,824` optimal solves, `0` infeasible/unbounded/error solves, and
`0` surrogate oracle evaluations. The exact cache has `96` unique entries.

The memory-safe execution pattern was one fresh process per candidate with
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`. A single long-lived exact process was stopped after
`8` completed candidates because memory high-water reached `2821.6` MB. The
process-per-candidate continuation stayed around `1.2` GB peak per candidate
and completed the remaining pool without solver failures.

The pool has real dynamic range. The weak regime objective spans `0.550118` to
`4.022673`, with final product `0.377921` to `3.097389`; the strong regime
objective spans `-0.042043` to `1.461934`, with final product `0.186344` to
`1.198654`. Matched weak/strong edit-vector objective ranks have Spearman
correlation `0.953317`, but the mean absolute objective shift is `1.320466`.
Interpretation: the strong-regulation regime compresses attainable product and
objective values, so the two regimes should remain separately reported.

Six methods were benchmarked on the exact cache:
`random_parallel`, `space_filling_parallel`, `static_gem_parallel`,
`black_box_bo`, `direct_trajectory_ucb`, and
`hybrid_digital_twin_ucb`. Protocols used budgets `8`, `16`, `24`, and `32`
for one-shot parallel, batched `q=4`, batched `q=8`, and sequential `q=1`
runs, with seeds `11`, `22`, and `33`. Setting A used the existing baseline
prior. Setting B used matched edit calibration for direct and hybrid methods.

Final untouched verification candidates are excluded from method selection, so
exact-pool regret is measured against the full finite pool, including held-out
verification optima. Under both setting A and setting B,
`space_filling_parallel` was best by mean exact objective in both regimes. Its
mean exact-pool regret was `0.210637` in the weak regime and `0.319835` in the
strong regime. Matched calibration narrowed direct/hybrid regret, but did not
beat space-filling. The acceptance table therefore reports
`hybrid_no_clear_measurement_advantage`,
`hybrid_no_clear_round_advantage`, and
`hybrid_no_clear_strain_advantage` for both regimes, with overall status
`hybrid_no_clear_advantage_consistent_across_regimes` and
`exact_finite_pool_complete`.

Overall method ranking by mean `exact_pool_regret` across both regimes and
both settings:

| Rank | Method | Mean exact-pool regret | Mean best observed objective |
|---:|---|---:|---:|
| 1 | `space_filling_parallel` | `0.265236` | `2.477068` |
| 2 | `direct_trajectory_ucb` | `0.303118` | `2.439186` |
| 3 | `hybrid_digital_twin_ucb` | `0.303241` | `2.439062` |
| 4 | `black_box_bo` | `0.344624` | `2.397679` |
| 5 | `random_parallel` | `0.421701` | `2.320602` |
| 6 | `static_gem_parallel` | `0.499934` | `2.242369` |

This ranking is the preferred compact comparison for the exact finite-pool
benchmark because it preserves the near-tie between direct and hybrid UCB while
still showing the clear space-filling lead.

Evidence files:
`data/design_benchmark_regime_manifest.csv`,
`data/design_benchmark_exact_pool_manifest.csv`,
`data/design_benchmark_exact_rollouts/`,
`data/design_benchmark_exact_pool_audit.csv`,
`data/design_benchmark_exact_pool_ranking.csv`,
`data/design_benchmark_oracle_cache_index.csv`,
`data/design_benchmark_oracle_solver_accounting.csv`,
`data/design_benchmark_regret.csv`,
`data/design_benchmark_seed_summary.csv`,
`data/design_benchmark_acceptance.csv`, and
`figures/design_benchmark_exact_regret.*`.

Bounded interpretation: this is exact evidence for the declared finite pilot
pool only. It is not a global optimum over all edits/environments, and it does
not overturn the need for later Stage 3 full replay and Stage 4 edited-strain
validation. The next exact step should remain conservative: expand the pool
incrementally or select a wet-lab-facing shortlist from the exact top
candidates, reusing the global exact cache rather than recomputing completed
rollouts.

## Deployment-style scientist versus hybrid framing

The exact finite-pool benchmark remains archived as
`matched_oracle_query_acquisition_benchmark` with valid status
`exact_finite_pool_complete` and
`hybrid_no_clear_advantage_consistent_across_regimes`. Its negative result is
scientifically useful, but it answers an equal-hidden-query acquisition
question rather than the deployed-twin question. A deployed digital twin is
supposed to spend many virtual evaluations and then buy only a small number of
parallel biological verifications.

The new benchmark type is
`digital_twin_deployment_and_verification_benchmark`. The hidden biological
verifier remains the exact decision-tree-regulated Yeast9/pFBA organism, but
the current execution is explicitly
`cached_small_pool_deployment_reanalysis`: no completed exact candidates were
rerun, and every hidden exact result is revealed only through a recorded
verification request. Virtual model evaluations are not counted as cultures.

The isolated LLM metabolic-engineer DBTL agent was not run because the
available execution mechanism does not provide a hard filesystem sandbox, and a
fresh public-only Codex task was not executed in this run. The status is
therefore
`llm_scientist_agent_not_run_isolation_unavailable`. The harness creates a
public bundle containing only the editable universe, heterologous library,
objective, operational scenarios, agent protocol, and handoff prompt, plus a
filesystem-access audit showing hidden files were not copied. The scientist
public prompt is only: improve dynamic beta-carotene production while
maintaining viable growth using sparse GSM edits and fixed `[T, pH, DO]`.

Graph-level strain engineering is represented as sparse editable GSM designs
with reaction IDs, edit types, magnitudes, rationales, environments, and assay
requests. The public editable universe currently contains `240` actual native
Yeast reaction IDs across `39` subsystems and a curated heterologous library.
For the public bundle, baseline fluxes and FVA ranges were computed with public
static Yeast9 analysis: `1` FBA solve, `1` pFBA solve, and `480` FVA LP solves
over the `240` public reactions (`482` LP solves total). These analyses are
available to the scientist as planning tools and are not hidden dynamic
verification. Validity checks reject direct beta-carotene demand edits, biomass
deletion, hidden-regulator edits, uncurated heterologous/source reactions,
edit-count violations, out-of-bound environments, excessive edit magnitudes,
objective-coefficient edits, inconsistent reversible bounds, unrestricted
nutrient-source additions, and ATP/redox source additions outside the curated
library. Invalid proposals consume design effort but no physical culture.

The self-contained public bundle is now prepared at
`results/deployment_benchmark/public_scientist_bundle/`. It exposes the public
Yeast9 copy, reaction/metabolite annotations, editable universe, heterologous
library, objective configuration, environment limits, edit and request schemas,
public operational budget, public static tools, request/response exchange
folders, tool documentation, and `SCIENTIST_AGENT_HANDOFF.md`. It does
not include hidden simulator source, regulator thresholds, exact-cache outcomes,
finite-pool rankings, hybrid checkpoints/predictions, successful strain lists,
or narrative outcome summaries. The bundle leakage scan passes.

The verifier interface smoke test was deliberately invalid and labelled
`evaluator_interface_smoke_test`; it rejected beta-carotene demand-reaction
editing before exact execution and recorded `0` hidden exact simulations. The
fresh scientist then produced `round_001_request.json`, which the hidden
administrator executed as the first open-ended exact graph-edited campaign.
The round contained `8` candidates, all fresh relative to the old exact cache.
Execution used the hidden decision-tree-regulated dynamic Yeast9 growth/product
pFBA loop, one candidate at a time, with `8` hidden exact simulations,
`1152` LP solves, `0` surrogate evaluations, and no growth-failure flags.
The strongest public observation was `round_001_candidate_04`
(`final_product=0.571041`, `product_AUC=2.392381`, `final_biomass=1.802349`).
The next two by final product were `round_001_candidate_05` (`0.559724`) and
`round_001_candidate_07` (`0.559266`). Public results are in
`data/deployment_benchmark_round_001_public_results.csv`; hidden evaluator
accounting is in `data/deployment_benchmark_round_001_exact_accounting.csv`.

The pretrained-hybrid replay generates `10,000` virtual proposals per cached
run, but this is not `10,000` distinct dynamic hybrid evaluations. The audit
shows each hybrid run maps those proposals to `48` cached eligible candidates
per regime, scores `48` cached dynamic values, uses no sparse surrogate layer,
and then requests verification batches of `4`, `8`, `12`, or `8+4` candidates.
The deterministic replay is not an LLM scientist claim; it is a
`scripted_public_information_DBTL_baseline` control that uses static GEM
scores, observed feedback, and simple BO-like scoring across `3` seeds and up
to `4` physical rounds.

In the central operational scenario, the scripted public-information DBTL
baseline reached the best-known verified objectives in both regimes:
`4.022673` weak and `1.461934` strong. It required `32` cultures, about `29`
unique strains, `4` physical rounds, `37.0` calendar days, `40.0`
scientist-hours, and `210.7` resource points. The best hybrid cached one-shot
result used `8` or `12`
cultures in `1` round and reached `2.482096` weak and `0.754442` strong,
with `9.25` to `14.25` calendar days and `51.0` to `74.0` resource points.
Hybrid one-correction used `12` cultures in `2` rounds and did not improve
quality beyond the one-shot `8/12` result.

The convenience comparison is therefore concrete but not a hybrid win: hybrid
has far better virtual-to-physical leverage (`1250:1` for the `8`-culture
batch and `833:1` for `12` cultures), fewer rounds, fewer cultures, fewer
strains, less calendar time, and fewer resource points, but it is not
quality-noninferior under the predeclared `delta=0.05` in this small cached
pool. The formal cached-pilot status is split accordingly:
`hybrid_quality_noninferiority_failed_cached_pilot` for quality, true
operational-advantage statuses for rounds/cultures/time/effort/resource, and
overall `hybrid_operationally_cheaper_but_quality_inferior_cached_pilot`.

Greenfield versus deployed-twin accounting is reported separately. In the
central scenario, an existing deployed-twin campaign averages `58.0` resource
points, while a greenfield digital twin costs about `740.5`, `246.8`, `148.1`,
or `74.1` resource points per campaign when amortised across `1`, `3`, `5`,
or `10` future campaigns. These are configurable operational assumptions, not
monetary claims.

Evidence files:
`data/deployment_benchmark_public_asset_manifest.csv`,
`data/deployment_benchmark_hidden_asset_manifest.csv`,
`data/deployment_benchmark_public_bundle_manifest.csv`,
`data/deployment_benchmark_public_bundle_audit.csv`,
`data/deployment_benchmark_sparse_edit_safeguards.csv`,
`data/deployment_benchmark_verifier_interface_audit.csv`,
`data/deployment_benchmark_static_solve_accounting.csv`,
`data/deployment_benchmark_claim_audit.csv`,
`data/deployment_benchmark_virtual_evaluation_audit.csv`,
`data/deployment_benchmark_dynamic_hybrid_accounting.csv`,
`data/deployment_benchmark_reaction_universe_audit.csv`,
`data/deployment_benchmark_editable_reaction_universe.csv`,
`data/deployment_benchmark_candidate_registry.csv`,
`data/deployment_benchmark_virtual_evaluations.csv`,
`data/deployment_benchmark_verification_requests.csv`,
`data/deployment_benchmark_verification_results.csv`,
`data/deployment_benchmark_resource_accounting.csv`,
`data/deployment_benchmark_quality_metrics.csv`,
`data/deployment_benchmark_acceptance.csv`, and
`figures/deployment_*.svg`/`.png`.

## PCA coefficient and error decomposition

The fixed-environment reduced surrogate was diagnosed with a training-only PCA
basis for `B(t)` and direct coefficient models from `[T, pH, DO]` to PCA
coefficients. PC1 explains `99.4527%` of training variance and PCs 1-2
explain `99.9337%`. The primary basis therefore uses the smallest K reaching
99.9% training variance, and all validation/test reconstructions use that
frozen training basis.

Best held-out PCA-coefficient reconstruction: `gradient_boosted_stumps` at `K=2`,
RMSE `0.0084`, range-normalized RMSE
`0.0833`, trajectory-normalized RMSE
`0.1299`. The decomposition tables separate
amplitude, final titer, AUC, peak time, normalized shape, late decline,
plateau, decline, recovery, and curve-derived stress-transition timing.

## Multi-seed model comparison

The reduced-surrogate multi-seed comparison now evaluates dataset seeds
`101, 202, 303` and model seeds `11, 22, 33`, saving every paired result and
bootstrap paired differences. Product-only state-space win fraction versus the
coordinate-conditioned MLP on held-out combinations is `0.000`
over `9` paired runs.

## ODE Physiological Regime Discovery and Inverse Design

This stage is deliberately not another Yeast9/pFBA run. It is a fast,
biologically structured synthetic ODE testbed used to ask whether a learned
dynamical surrogate can support regime discovery, transition-boundary
localization, mechanistic explanation, and inverse design while retaining the
same deployment contract:

`fixed [T, pH, DO] -> deterministic beta-carotene trajectory P(t)`.

The ODE states are:

```text
X(t): biomass
A(t): active heterologous pathway capacity
R(t): precursor or carbon reserve
E(t): cellular energy or ATP capacity
S(t): oxidative-stress burden
C(t): pathway congestion or bottleneck burden
P(t): accumulated beta-carotene
```

The shared product flux is:

```text
v_product =
  k_product * X
  * pathway_support(A)
  * precursor_support(R)
  * energy_support(E)
  * stress_inhibition(S)
  * congestion_inhibition(C)
```

and product evolves as:

```text
dP/dt = v_product - degradation(S,C) * P
```

Other equations follow the same physiological interpretation: biomass growth is
smoothly controlled by temperature, pH, oxygen, energy, stress, and congestion;
pathway capacity is synthesized and damaged; precursor is shared between growth
and production; energy is generated from oxygen-supported respiration and spent
on growth/product/stress repair; stress accumulates from high temperature,
excess oxygen, product flux, and congestion; congestion accumulates from
overload and clears through energy-dependent maintenance.

Regime labels are transparent post hoc diagnostics, not generation rules. The
current pilot audit contains balanced sustained production, delayed activation,
precursor-limited plateau, ATP-limited slow production, oxidative collapse,
congestion recovery, growth-dominated low-product behavior, and mixed
transition regions. The audit explicitly checks finite nonnegative states,
solver failures, raw and normalized product PCA, physiological-state PCA,
cluster separability, regime counts, representative trajectories, regime maps,
within/between-regime distances, smooth neighboring behavior, and boundary-like
regions.

Prediction models receive only `[T, pH, DO]` at deployment. Direct baselines
include the mean training trajectory, nearest environment, linear and
polynomial environment-to-PCA models, tree environment-to-PCA, multi-output
MLP, coordinate-conditioned MLP, and a direct multi-output physiology MLP.
State-space variants include product-only, physiologically supervised,
shuffled-auxiliary, and random-smooth-auxiliary controls:

```text
z0 = g(e)
z[k+1] = z[k] + dt f(z[k], e)
P_hat[k] = h_P(z[k])
```

Physiology labels are auxiliary training labels only. They are not deployment
inputs and should be judged by regime separation, boundary recovery, and design
verification as well as product RMSE. A direct model may still win product RMSE;
that remains a valid bounded result.

The inverse-design script fits a continuous trajectory surrogate on training
conditions, searches bounded environmental space for endpoint, productivity,
sustained, physiologically constrained, and robust objectives, and verifies
every reported proposal with fresh ODE simulations. It also constructs a
continuous regime atlas, identifies neighboring predicted regime transitions,
and reruns the original simulator on both sides and at candidate boundary
points. Local finite-difference sensitivities compare neural/surrogate
sensitivities against simulator-side sensitivities for product, stress, energy,
precursor, and congestion.

The bounded claim is:

```text
A biologically structured synthetic ODE testbed demonstrates how a learned
neural dynamical surrogate can move beyond trajectory prediction to recover
production regimes, locate unsampled transition boundaries, and perform
physiologically constrained inverse design.
```

This does not validate real yeast kinetics, prove latent coordinates are
unique cellular variables, imply state-space models must beat MLPs, treat
interior held-out blocks as true extrapolation, or claim experimental
validation.

Current full-run interpretation:

1. Fixed environmental conditions can predict the full trajectory in this
   synthetic ODE testbed. On held-out interior holes, the validation-selected
   state-space models beat the best validation-selected direct baseline for all
   three seeds, although the random-smooth auxiliary control was unexpectedly
   competitive and remains a cautionary negative control.
2. The simulator itself generates distinct physiological regimes and internal
   states are richer than product alone by PCA. However, the first learned
   latent representations did not yet recover regime labels better than
   product-PCA clustering, so representation discovery remains the main area
   for improvement rather than a solved claim.
3. Transition-boundary discovery works partially: the continuous surrogate
   found candidate neighboring regime changes and fresh ODE simulations
   confirmed `63/120` checked qualitative transitions. False positives are
   retained in the boundary table.
4. Inverse design can propose unsampled bounded conditions and every proposal
   is ODE-verified. Several robust/sustained proposals improved verified
   product or stress tradeoffs relative to sampled conditions, while some
   objectives did not improve after verification. These failures are part of
   the design-reliability result, not filtered out.

## Open-Ended Yeast9/pFBA Deployment Round 001

The deployment-stage physiological question is now an open-ended strain-design
question under a fixed public design contract: sparse reaction edits plus a
fixed culture environment are proposed, then evaluated by the hidden Yeast9
dynamic pFBA verifier. This is separate from the synthetic ODE regime testbed
above.

The first matched comparison is
`open_ended_matched_round_001_scientist_vs_hybrid`. The scientist side is the
frozen public-bundle Round 001 request and is labelled
`public_bundle_scientist_round_001`; its context isolation remains
`scientist_round_001_context_isolation_unverified`, so no stronger blinding
claim is made. The hybrid side used a staged blinded workspace and denied-asset
manifest; this is manifest/code-path blinding, not hard filesystem isolation.

Hybrid search generated `50,000` unique valid candidates from the `240` public
editable reactions. The final selected `8` were frozen before overlap and exact
lookup, and every selected candidate was novel to both the old exact pool and
scientist Round 001. Hybrid screening labels are deliberately separated:
generation/static filtering used no new exact LP solves, the sparse surrogate
layer had `0` validated scores, and the dynamic layer was a short-horizon
`hybrid_dynamic_yeast9` refinement of `100` candidates using `300` LP solves.
Only the verifier results are counted as exact hidden Yeast9 dynamic pFBA.

Exact verification completed with `8` hidden simulations and `1152` LP solves
for each method. The scientist best verified objective was `0.816502` with best
final product `0.571041`. The hybrid best verified objective was `0.802500`
with best final product `0.559266`. There were no growth failures or solver
errors in either batch. Under the predeclared `0.05` noninferiority margin, the
hybrid status is `hybrid_round_001_quality_noninferior`, not superior.

## Replicated and Adaptive Open-Ended Deployment Benchmark

The replication phase now contains valid Campaign 002 and Campaign 003
scientist batches from fresh public-only DBTL agents. The agents used only
their campaign public bundles, kept hidden verifier and hybrid outcomes
unavailable during planning, ran sequential public digital planning/validation
cycles, and then froze one parallel physical batch of `8` cultures each.

The invalid Campaign 002 shared-path duplicate remains archived as
`campaign_002_scientist_replication_invalid_duplicate_request` and is excluded
from independent inference. The replacement Campaign 002 scientist batch was
registered through the campaign-specific exchange path and verified with `8`
fresh exact simulations (`1152` LP solves). Campaign 002 hybrid was preserved
without rerun: best objective `0.678774`, mean objective `0.678774`.

Campaign 003 completed with the same blinded boundaries. The scientist batch
was frozen before hybrid search. The hybrid then used seed `33003`, generated
`50,000` candidates, dynamically refined `100` candidates using `300`
lightweight LP solves, froze `8` candidates before exact lookup, and verified
them with `8` exact simulations (`1152` LP solves).

Across Campaigns 001-003, scientist wins best verified objective in all three
campaigns, while hybrid wins mean verified objective in all three. Hybrid
best-objective noninferiority under the `0.05` margin holds in `1/3`
campaigns, so the one-shot classification is `one_shot_result_mixed`. The
report conclusion is bounded as
`result_depends_on_campaign_and_accounting_scenario`: the hybrid looks useful
as a batch-quality virtual screening tool, but the best-hit advantage is not
supported by this three-campaign replication.

## Open-Ended Benchmark Readiness Audit

The benchmark is now at a readiness-audit stage. The immediate task is not to
add another campaign, but to decide whether the existing open-ended
strain-design benchmark is technically trustworthy enough to freeze for a
sequential DBTL acceleration experiment. The audit is implemented in
`src/yeast_validation/audit_open_ended_benchmark_readiness.py`; the baseline proposal
selectors are in `src/yeast_validation/open_ended_baselines.py`.

The audit reuses cached exact artifacts only. It compares requested candidate
specifications, deterministic candidate hashes, exact-accounting summaries,
hidden biomass/product/burden trajectories, hidden active-constraint traces,
hidden flux summaries, output-file provenance, and public static GEM
application of requested edits. It does not run new hidden exact simulations.

Repeated hybrid objective values are not benign formatting noise. Campaign 002
has `8` unique requested hybrid candidates but only `1` unique effective
phenotype and `1` unique objective value. The repeated `0.802500` top outcome
in Campaigns 001 and 003 also maps to the same effective phenotype despite
different requested candidate hashes. The repeated-outcome classification is
therefore inactive intervention/effective phenotype collapse rather than cache
collision, hash collision, or duplicate result rows.

Edit effectiveness is the main readiness concern. Many selected sparse edits
formally map to valid public Yeast9 reactions, but the applied bounds do not
change the model or the edited reaction has no verified influence on the
hidden trajectory. This means the current candidate space contains many
formally different interventions that are biologically ineffective under the
current exact-verifier rules.

The reanalysis adds best objective, top-2 mean, top-4 mean, whole-batch mean,
median, standard deviation, worst candidate, fraction above parent, growth
failure rate, unique candidate specifications, unique effective phenotypes,
unique objective values, edit counts, environment diversity, reaction-edit
diversity, and biological objective components. After these metrics, the
previous batch-risk interpretation should be treated cautiously: hybrid
whole-batch means remain higher, but Campaign 002's apparent consistency is
mostly a repeated effective phenotype.

Best-candidate noninferiority was recomputed for `delta=0.02`, `0.05`, and
`0.10`. Campaign 001 is descriptively noninferior at all three margins;
Campaign 002 is not; Campaign 003 misses by just over `0.10`. With only three
one-shot campaigns, this is descriptive sensitivity analysis, not formal
statistical noninferiority.

The minimum missing baselines are now prepared as proposal logic only:
`random_valid`, `space_filling_maximin`, and `static_gem_public_rank`. They use
only public edit-universe, environment, annotation, pFBA/FVA, and
construction-risk information and do not inspect hidden exact outcomes. They
must not be exact-verified until the benchmark spec, target definition,
duplicate handling, stopping rules, and seed policy are frozen.

Readiness outputs are:
`data/open_ended_benchmark_readiness_report.md`,
`data/open_ended_hybrid_candidate_audit.csv`,
`data/open_ended_edit_effectiveness.csv`,
`data/open_ended_repeated_outcome_classification.csv`,
`data/open_ended_campaign_metric_reanalysis.csv`,
`data/open_ended_noninferiority_sensitivity.csv`,
`data/open_ended_information_leakage_matrix.csv`,
`data/open_ended_benchmark_freeze_spec.json`,
`data/open_ended_baseline_random_valid_proposals.csv`,
`data/open_ended_baseline_space_filling_proposals.csv`,
`data/open_ended_baseline_static_gem_proposals.csv`, and
`data/open_ended_baseline_selector_documentation.csv`.

## Final DBTL Stage 1 Freeze-Prep

The next stage was deliberately limited to benchmark repair and freezing, not
to the final sequential `8+4+4` DBTL evaluation. The previous readiness audit
showed that requested hybrid candidates could collapse to a single effective
phenotype. Stage 1 therefore traced the edit path from public candidate specs
through model reconstruction, sparse-edit application, dynamic constraint
application, exact-solver staging, objective scoring, and cache/identity
handling.

The main implementation defect was that many capacity edits were syntactically
valid but numerically equivalent after application. Default Yeast9-style bounds
such as `1000` could be multiplied and then clipped back to `1000`; some dynamic
constraints overwrote globally controlled edits; and many reactions in the
public universe had no flux consequence in the tested exact condition.
`src/yeast_validation/deployment_benchmark.py` now applies finite target/reference bounds,
capacity decreases, reversible knockdown movement toward zero, transport
capacity semantics, and heterologous pathway-enzyme capacity semantics.

`src/yeast_validation/final_dbtl_benchmark_freeze.py` freezes the repaired Stage 1 artifacts.
It writes `data/final_dbtl_actionable_intervention_library.csv`,
`data/final_dbtl_excluded_interventions.csv`,
`data/final_dbtl_edit_application_diagnostics.csv`,
`data/final_dbtl_objective_config.json`,
`data/final_dbtl_target_config.json`,
`data/final_dbtl_benchmark_spec.json`,
`data/final_dbtl_method_registry.csv`,
`data/final_dbtl_campaign_manifest.csv`,
`data/final_dbtl_method_access_matrix.csv`,
`data/final_dbtl_budget_estimate.csv`, and
`data/final_dbtl_readiness_decision.csv`. No new Markdown report is introduced.

The candidate actionability library retains `22` interventions across `7`
mechanistic classes: `6` transport/exchange edits, `6` byproduct or carbon
rerouting edits, `3` capacity decreases, `2` precursor-supply edits, `3`
pathway-enzyme capacity edits, `1` oxygen-exchange edit, and `1` ATP-demand
reduction. The exclusion file has `221` rows, including `202`
not-selected-after-reliability-and-diversity-cap entries and `19`
per-mechanism-cap exclusions. Retained rows include biological actionability
metadata such as mechanism class, subsystem, construction tier, reversibility,
expected phenotype direction, risk level, and public evidence fields.

Identity is now separated into three hashes/fingerprints. The candidate-spec
hash describes the public requested intervention and environment. The
effective-model hash describes the post-edit model/configuration actually sent
to the verifier. The phenotype fingerprint describes rounded output behavior
while excluding candidate IDs and runtime/cache identifiers. This prevents
duplicate requested hashes, equivalent model states, and repeated phenotypes
from being conflated.

Representative staged-solve diagnostics still fail the readiness gate. Of `18`
checked retained interventions, `2` were effective, `15` were inactive or
equivalent, and `1` failed diagnostics, giving an effective rate of `0.111111`.
Because this cheap exact repair-verification gate failed, the optional short
dynamic exact diagnostics were not run and the final sequential campaigns were
not launched.

The frozen utility hash is
`19c635db7d3ae199a92feb34e3642a0b93d067729ecc583370aec6f8d7363753`; the target
hash is `b2b26e2d758e52b2d7629241f62748c072b1264dd9ecf513e8b7522e26edeb66`;
and the benchmark spec hash is
`f10b282dfd5235e71460e331c9cbcf3c180f8dc1e407fcab74ce3be6c7561794`. The frozen
budget is `168` minimum exact simulations with immediate early stopping, `336`
maximum exact simulations, `144` LP solves per full exact candidate, and
`48384` maximum LP solves.

Current status: `NO_GO_FINAL_DBTL_EVALUATION`. The benchmark should not be used
to claim accelerated DBTL until the intervention library is rebuilt around
edits that measurably survive dynamic constraints and the freeze-prep
diagnostics are rerun.

## Final DBTL Intervention Redesign And Re-Gate

The previous `NO_GO_FINAL_DBTL_EVALUATION` result is retained as a negative
finding. The failed Stage 1 intervention set was reconstructed and diagnosed in
`data/final_dbtl_failed_intervention_diagnosis.csv`: `18` representative
interventions, `2` effective, `15` inactive/equivalent, and `1`
diagnostic-failed. Failures traced mostly to non-limiting default-capacity
bounds, transport direction errors, pFBA compensation, edits outside active
beta-carotene leverage, and dynamic interval rules erasing nominal edits.

The dynamic relevance map in
`data/final_dbtl_dynamic_reaction_relevance.csv` covers `244` public or curated
reactions/parameters. It combines public pFBA/FVA evidence, representative
environment flux ranges, active-bound frequency, pathway proximity, GPR
redundancy, compensation risk, dynamic-override risk, and direct burden-state or
interval-constraint involvement. The relevance score is outcome-blind to the
final benchmark.

Edit semantics were redesigned around finite references and dynamic survival:
pathway-enzyme edits alter PSY/DES/CYC dynamic capacity parameters,
oxygen/glucose edits alter interval uptake coefficients, precursor edits use a
finite public reference capacity, and generic multiplication of unconstrained
`1000` bounds is no longer accepted as sufficient evidence. Product-loss and
ATP-maintenance probes were tested but failed the staged gate in this pass.

The diagnostic library has `53` finite-semantics edits. The staged exact screen
tested `30` of them across `8` representative environments (`240` exact
staged cases). Results were `188` effective, `36` inactive/equivalent, and `16`
diagnostic-failed cases; `24/30` interventions passed the staged gate. Short
dynamic exact diagnostics were then run for those `24` interventions only, and
all `24` produced distinct short-horizon phenotypes above tolerance with
complete exact pFBA status.

The final retained library in
`data/final_dbtl_actionable_intervention_library.csv` has `24` interventions:
`12` direct heterologous pathway-capacity edits, `8` transport/environment
coupling edits, and `4` precursor or competing-sink leverage edits. The updated
exclusion file has `6` rows, all excluded because they failed staged exact
screening. Effective-model and phenotype duplicates are removed before
retention.

The effective design-space audit sampled `360` valid candidates and found
`325` unique effective models and `325` unique diagnostic phenotypes, giving a
candidate-to-effective-model compression ratio of `1.107692` and an
effective-model-to-phenotype compression ratio of `1.0`. Mechanism diversity is
`3`, subsystem diversity is `4`, and environment diversity is `8`.

The frozen objective hash remains
`19c635db7d3ae199a92feb34e3642a0b93d067729ecc583370aec6f8d7363753`; the target
hash remains `b2b26e2d758e52b2d7629241f62748c072b1264dd9ecf513e8b7522e26edeb66`;
the updated benchmark spec hash is
`4953ea2955d0de2f1fc4e15041739c715911bcbe3a9872533891e9d67a5f8fd0`. The rerun
readiness decision is `GO_FINAL_DBTL_EVALUATION`, with the explicit boundary
that the final sequential campaigns were not run here.

## Final Sequential DBTL Evaluation

The final sequential `8+4+4` DBTL evaluation has now been executed, with the
computationally expensive exact rollouts offloaded to Vanda. The run used the
`vanda-codex` SSH alias specified in the project handover, unpacked the bundle
under
`/home/svu/e1471252/final_dbtl_benchmark_vanda_20260803_1205`, and submitted
through queue-less PBS routing after explicit queue requests were denied by
site policy. The completed replacement job was `1283942.stdct-mgmt-02` and
loaded `Python/3.12.3-GCCcore-13.3.0`.

The evaluation used the frozen objective, target, method registry, intervention
library, campaign seeds, and `8+4+4` early-stopping protocol. It wrote
`data/final_dbtl_final_report.md`,
`data/final_dbtl_method_summary.csv`,
`data/final_dbtl_statistical_analysis.csv`,
`data/final_dbtl_observations.csv`,
`data/final_dbtl_best_so_far.csv`,
`data/final_dbtl_round_summary.csv`, and
`figures/final_dbtl_*.svg/.png`. The execution manifest status is
`complete_final_dbtl_benchmark`.

The final exact accounting is `35` physical culture observations, `5040`
recorded Yeast9 dynamic pFBA LP solves, `0` solver failures, `0` cache hits,
and `140` cached rollout files. This lower-than-maximum culture count is due to
target-based early stopping, not hidden outcome leakage.

All seven methods reached the target in all three campaign seeds. Hybrid
digital twin reached target with success rate `1.0`, median cultures-to-target
`1`, mean cultures-to-target `1.666667`, and mean best utility `2.071318`.
The best mean utility was `space_filling_maximin` (`2.219857`), followed by
`static_gem_public_rank` (`2.189569`) and the public-bundle scientist agent
(`2.153550`). Space filling, static GEM, and the public-bundle scientist agent
also reached the target in `1.333333` cultures on average, while hybrid needed
`1.666667`.

Scientific interpretation: the redesign solved the earlier inactive-edit
collapse enough to make the final benchmark executable and phenotype-effective,
but the final synthetic evidence does not support a unique hybrid advantage in
physical culture count or best utility. The bounded positive claim is that the
hybrid reliably reaches the predefined target under the exact simulator. The
bounded negative result is that, over three seeds, public/static baselines can
match or exceed hybrid on the primary experiment-count and utility endpoints.

## Harder Biological Worlds And Distribution Shift

The next scientific question is now frozen as a computational validation
extension: does the hybrid digital twin become more useful when the biological
optimisation problem is harder and when the target simulator differs from the
source simulator used for model training or calibration? This stage does not
design a new neural architecture. It tests the existing hybrid-versus-baseline
story under harder exact Yeast9 dynamic-pFBA worlds and source-to-target hidden
physiology shifts.

The exact simulator catalogue in `src/yeast_validation/design_benchmark_exact.py` now
includes seven predeclared hidden worlds:

- `baseline_world`
- `strong_oxidative_burden_world`
- `atp_limited_world`
- `precursor_competition_world`
- `pathway_bottleneck_damage_world`
- `compensatory_metabolism_world`
- `mixed_multi_mechanism_world`

Each world has a biological interpretation, frozen dynamic configuration,
burden coefficients, repair coefficient, pathway-damage scale, degradation
rule, and world hash. The world specifications are written to
`data/harder_shift_world_manifest.csv` and
`data/harder_shift_world_specs.json`. The intended acceptance gate is
simulator-level landscape difficulty, not model performance: target attainment
fraction, objective/final-product/biomass distributions, growth-failure rate,
rank correlation with baseline, ranking changes, phenotype diversity, and
top-candidate mechanism diversity.

Harder target rules are frozen in
`data/harder_shift_target_rules.json`. The target family
`harder_shift_multicriterion_targets_v1` defines `easy`, `moderate`, and
`hard` tiers from parent-relative final titer, biomass preservation, yield
preservation, utility improvement, maximum interventions, and burden/growth
failure limits. The target-rule hash is
`5cafb5141c592707f426cbf1e32360c5d68de916adebb2f4d438051e623efcd8`. These
rules are not tuned to method outcomes.

Distribution-shift source-to-target pairs are frozen in
`data/distribution_shift_pair_manifest.csv`: baseline to baseline
in-distribution, baseline to strong oxidative burden, baseline to ATP-limited,
baseline to precursor competition, baseline to pathway bottleneck/damage,
baseline to compensatory metabolism, and baseline to mixed multi-mechanism
shift. The evaluation plan records zero-shot transfer and equal few-shot target
world budgets of `0`, `2`, `4`, and `8` exact cultures.

The campaign scaffold is `src/yeast_validation/run_harder_shift_validation.py`. It writes
the harder-DBTL exact manifest, the distribution-shift exact manifest, the
distribution-shift decision manifest, the method access matrix, the target
rules, the world specs, and a Vanda bundle. Current manifest sizes are:
`3017` harder-DBTL exact tasks, `2352` unique distribution-shift exact tasks,
and `9408` distribution-shift decision rows. The decision rows preserve the
zero/few-shot adaptation views without duplicating exact simulations.

Vanda was inspected and used for the smoke execution. The cluster reports PBS
with `qsub`/`qstat`, Python module `Python/3.12.3-GCCcore-13.3.0`, CPU queues
including `batch_cpu`, and scratch at `/scratch/e1471252`. Direct queue
selection for `batch_cpu` was denied by policy, but queue-less routing placed
the smoke job on `batch_cpu`. The generated PBS scripts therefore omit an
explicit queue and request `1` CPU, `6gb` memory, `04:00:00` walltime, no GPU,
and single-threaded BLAS/OpenMP settings.

The bundle is `hpc_bundles/harder_shift_validation_vanda.tar.gz` and was
unpacked on Vanda at:

```text
/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda
```

One exact smoke task completed successfully on Vanda:
`13f1a86176e652db284775f0`, status `complete_exact_dynamic_pfba`. The copied
local outputs are in
`results/harder_dbtl/exact_rollouts/baseline_world/`.

The simulator-level landscape gate has now completed. PBS array `1291064[]`
evaluated manifest rows `1-665`, giving `95` shared candidate phenotypes in
each of the seven hidden worlds. The completion audit records `665/665`
gate tasks complete, `0` pending, and completion fraction `1.0`. The one
early task-1 cache-hit traceback was repaired and recovered as a completed
cached exact result, so it is an execution note rather than a missing
phenotype.

Gate metrics in `data/harder_dbtl_difficulty_gate_summary.csv` show that all
modified worlds induce meaningful simulator-level landscape shift. The
baseline mean final product is `0.576823` with no rank change by definition.
Strong oxidative burden lowers mean final product to `0.394546` and changes
`0.884211` of rankings. ATP limitation lowers mean final product to
`0.243197` and changes `0.915789` of rankings. Precursor competition lowers
mean final product to `0.384736` and changes `0.947368` of rankings. Pathway
bottleneck/damage lowers mean final product to `0.257782` and changes
`0.873684` of rankings. Mixed multi-mechanism stress is the hardest by this
simple product summary, with mean final product `0.157969` and `0.926316`
ranking changed. None of the gate worlds had a growth-failure rate above
`0.0`, so the landscapes are difficult by product and ranking structure, not
by wholesale infeasibility.

The compensatory-metabolism world needs a careful label. It changes
`0.768421` of rankings and increases mean biomass, but its mean final product
is `0.586152`, slightly above baseline. This is a valid hidden physiology
shift, but not a lower-product hard world under the difficulty-gate summary.
That negative/caveated result is preserved rather than retuned away.

The matched harder sequential DBTL array `1291479[]` covered manifest rows
`666-3017` and has now finished on Vanda. As of
`2026-08-09T21:29:01+08:00`, no active `harder_seq` scheduler entry was shown,
`3017/3017` harder-DBTL summary files existed, sequential success logs covered
`2352/2352` tasks, and the harder sequential failure-log count was `0`. The
last copied local checkpoint was still partial, so final harder-world method
claims require copying the full Vanda outputs back and rerunning
`analyze-harder`.

The distribution-shift exact array `1292360[]` covered manifest rows `1-2352`
and has also finished on Vanda. As of `2026-08-09T21:29:01+08:00`, no active
`shift_dbtl` scheduler entry was shown, `2352/2352` target-world summary files
existed, shift success logs covered `2352/2352` tasks, the shift cache-row
count was `2352`, and the shift failure-log count was `0`. The analysis
outputs are now defined in `src/yeast_validation/run_harder_shift_validation.py`:
harder-world tables and figures come from `analyze-harder`; zero/few-shot
transfer, source-target utility degradation, ranking quality,
adaptation-budget performance, uncertainty, and robustness tables come from
`analyze-shift`. No final claim about hybrid advantage under hard or shifted
worlds should be made until the completed Vanda outputs are copied back and
the matched analyses are rerun locally.

## Definitive Simulated DBTL Acceleration Benchmark

The next validation question is now the full workflow question: can physical
biosensor measurements plus a modular regulation-metabolism digital twin
replace enough simulated physical experimentation to reach a predefined
engineering target in fewer cultures or fewer DBTL rounds? This reframes RMSE,
finite-pool regret, static-GEM ranking, and hybrid-versus-space-filling
comparisons as supporting diagnostics rather than the primary claim.

The new scaffold is
`src/yeast_validation/run_simulated_dbtl_acceleration_benchmark.py`. It treats the exact
Yeast9 dynamic-pFBA verifier as the hidden wet lab. The wet lab accepts sparse
metabolic edits plus temperature, pH, and dissolved oxygen, then returns product
and biomass outputs together with reporter measurements when the workflow is
allowed to observe them. Exact cache reuse is allowed for compute efficiency,
but physical-culture accounting remains separate: each workflow request is
charged one simulated physical culture.

The hidden wet-lab wrapper freezes culture-to-culture latent variability for
oxidative susceptibility, ATP burden, pathway capacity, expression burden,
pathway damage, recovery, precursor availability, growth lag, and product
degradation susceptibility. Reporter conditions are frozen as no reporters,
single reporters, full reporters, noisy full reporters, and sparse full
reporters. Reporter measurements are lagged/noisy/sparse transforms of latent
physiology, not privileged hidden-state access.

The frozen workflow comparison now includes conventional DBTL, Bayesian
optimisation, black-box digital twin, modular hybrid without biosensors, and
biosensor-informed modular hybrid. The full prepared manifest spans `6` major
worlds, `10` campaign seeds, `15,840` workflow culture requests, and `13,920`
unique exact hidden-wet-lab culture tasks. The protocol hash is
`2ed0c8e083537bf918013619c3ed0081bc930fd52bd725624d23940083dc1612`.

The local mock pilot gate passed scaffold checks, and the real exact
non-mock Yeast9 pilot gate has now passed on Vanda. The preliminary
`12`-task pilot `1297243[].stdct-mgmt-02` completed but was not accepted as the
final gate because it covered only one campaign seed. The accepted pilot is
`1297249[].stdct-mgmt-02`: `18` exact tasks spanning `6` worlds, `3` campaign
seeds, all `5` workflow methods, and all `7` reporter conditions. Its protocol
hash remains
`2ed0c8e083537bf918013619c3ed0081bc930fd52bd725624d23940083dc1612`.

The accepted exact pilot passed the frozen gates recorded in
`data/simulated_dbtl_exact_pilot_acceptance.csv`: all expected artifacts were
present; all runs ended with `complete_exact_dynamic_pfba`; the backend was
non-mock `yeast_gem_lp`; LP accounting was complete at `144` actual LP solves
per task; solver errors, infeasible solves, and unbounded solves were all
zero; latent culture-to-culture variability was expressed; reporter outputs
were present only when visible under the declared reporter condition and
remained biologically consistent with the hidden latents; physical cultures
were charged once per requested culture (`18` total); virtual evaluations were
recorded separately (`900,000` total) and not counted as cultures; no method
was allowed access to hidden simulator internals or future exact outcomes.
Pilot runtime was safe for the PBS plan (`142.143` s mean, `145.599` s max).

The full exact Vanda array has been launched as of
`2026-08-10T02:53:38+0800`. Because Vanda rejected one `1-13920` array under
its `max_array_size = 10000` server limit, the unchanged frozen manifest was
submitted as three non-overlapping chunks: `1297261[].stdct-mgmt-02` for rows
`1-1000`, `1297262[].stdct-mgmt-02` for rows `1001-10920`, and
`1297263[].stdct-mgmt-02` for rows `10921-13920`. Current status:
`EXACT_PILOT_PASSED_FULL_VANDA_ARRAY_SUBMITTED_RUNNING`. No final scientific
conclusion should be drawn until the full exact outputs are copied back and
analysed with the predeclared cultures-to-target, rounds-to-target, quality,
and right-censoring endpoints.

Acceleration migration snapshot, `2026-08-11T18:56:37+0800`: Vanda's
`batch_cpu` user cap limited the original arrays to `6` running tasks at a
time, so `1297262[]` and `1297263[]` were held, their active elements were
allowed to finish, and the queued leftovers were cancelled. Before
resubmission, the synced local audit reported `5065` valid exact tasks,
`8855` missing exact tasks, and `0` failed, invalid, duplicate, or
hash-mismatch outputs. The remaining tasks were written to
`data/simulated_dbtl_exact_missing_manifest.csv`.

The missing-only array was resubmitted through Vanda's `auto_free` routing
queue and routed to `cpu_serial`, increasing concurrency to `32` running tasks.
The accelerated array is `1300458[].stdct-mgmt-02`; at the verification
snapshot it had `64` finished tasks, `32` running tasks, and `8759` queued
tasks. The remote output tree contained `5129` exact summary files, and the
new fast-array logs showed no traceback, solver-error, infeasible, or
unbounded patterns. The final aggregation script,
`src/yeast_validation/analyze_simulated_dbtl_acceleration.py`, remains gated on all `13920`
tasks passing the exact completion audit.

Follow-up status, `2026-08-12T11:38:17+0800`: the accelerated array
`1300458[]` reached `12605/13920` exact output sets but then stopped under a
PBS system hold. Scheduler state was `7540` accelerated-array tasks expired,
`1313` queued, `0` running, and two held subjobs (`7533`, `7535`) with the PBS
comment `job held, too many failed attempts to run`; those held elements had no
task logs. This indicates a cluster launch/retry problem, not a documented
biological or solver failure. A remaining-only cleanup manifest,
`data/simulated_dbtl_exact_remaining_20260812_manifest.csv`, was generated for
the `1315` missing task IDs and submitted through the same exact runner.
Cleanup job `1301555[].stdct-mgmt-02` routed to `cpu_serial`; at the
verification snapshot it had `28` running tasks, `1287` queued tasks, and `0`
expired cleanup tasks.

Second follow-up status, `2026-08-12T13:13:16+0800`: the `cpu_serial` cleanup
path produced more exact outputs but repeatedly triggered Vanda system holds
before all queued subjobs launched. The first cleanup added `28` exact output
sets, and the subsequent `54` small-array split-tail rescue raised the remote
output count to `12722/13920`. No checked cleanup logs showed traceback,
solver-error, infeasible, or unbounded patterns. The remaining split-tail
queued/held work was cancelled after active subjobs completed. A final-tail
manifest, `data/simulated_dbtl_exact_final_tail_20260812_manifest.csv`, was
generated for the remaining `1198` missing task IDs and submitted as
`1301767[].stdct-mgmt-02` through the stable queue-less route. Vanda routed it
to `batch_cpu`, with `6` running tasks, `1192` queued tasks, and `0` expired
tasks at verification.

Third follow-up status, `2026-08-12T15:04:17+0800`: the `batch_cpu` tail was
held after its active tasks completed, and its queued leftovers were cancelled.
A fresh range-tail manifest,
`data/simulated_dbtl_exact_range_tail_20260812_manifest.csv`, was generated
from the `12992/13920` completed-output state, leaving `928` missing exact
tasks. The remaining work is now running through non-array `cpu_parallel` range
workers to avoid the PBS array launch-hold failure mode. Worker
`1301958.stdct-mgmt-02` covers rows `465-928` with `16` CPUs and `20gb`;
worker `1301972.stdct-mgmt-02` covers rows `1-464` with `8` CPUs and `12gb`.
Both were running without holds at the snapshot, with latest exact summary
count `13024/13920` and no checked range-worker traceback, solver-error,
infeasible, or unbounded patterns.

Final status, `2026-08-12T17:08:00+0800`: the final exact benchmark is complete
and has passed the strict local audit. Vanda produced `13920/13920` summaries,
reporters, trajectories, and flux files, with no active `sim_dbtl` jobs left.
After syncing the latent JSON sidecars, the audit reported `13920` expected,
`13920` completed, `13920` valid, and `0` missing, failed, duplicate,
hash-mismatch, or invalid outputs under protocol hash
`2ed0c8e083537bf918013619c3ed0081bc930fd52bd725624d23940083dc1612`. The final
analysis completed. The moderate-tier final claim audit is `NEUTRAL`: exact
data support strong improvement over Bayesian optimisation, but not a robust
physical-culture or round reduction versus conventional DBTL for the canonical
full-reporter biosensor hybrid.
The hard-tier sensitivity analysis also completed and remained `NEUTRAL`:
canonical full-reporter hybrid target attainment `0.15`, median censored
cultures `25.0`, median censored rounds `4.0`, mean cultures saved versus
conventional DBTL `-1.9`, and mean best-utility delta versus conventional DBTL
`0.00343`.

Corrected wall-clock re-analysis, `2026-08-12`: the exact cache was reused and
no new Yeast9 simulations were launched. The new analysis separates biological
wall-clock depth from total physical culture count. Because proposal batches in
the definitive manifest were generated from placeholder observations rather
than exact wet-lab outcomes, the digital-twin verification candidates can be
collapsed into calibration plus one parallel verification batch without
future-outcome leakage. This changes the time-efficiency interpretation but not
the exact biological outcomes. For the cached top-16 one-shot full-reporter
biosensor hybrid, mean sequential biological stages are `1.967` versus `3.2`
for conventional DBTL, with mean scenario calendar time `17.7` versus `31.0`
days excluding compute. However, the hybrid uses more parallel cultures on
average (`23.47` versus `20.67`) and has lower target attainment (`0.4833`
versus `0.5333`). Mean best-utility remains noninferior under the `0.05`
margin. The corrected evidence supports wall-clock acceleration with greater
parallel culture capacity, not fewer cultures or superior target reliability.

## Best-Found Utility And Biosensor Information-Regime Follow-Up

A follow-up analysis reused the completed `13920/13920` exact hidden-wet-lab
tasks and launched no new Yeast9 simulations. The analysis is
`src/yeast_validation/analyze_simulated_dbtl_best_found_followup.py`; the compact report is
`data/simulated_dbtl_best_found_followup.md`.

The new primary question is distinct from the previous threshold endpoint. The
time-to-target analysis asks how quickly a workflow first reaches a predefined
target. The fixed-budget best-found analysis asks which workflow finds the
highest verified utility after spending the same operational budget. Both are
valid, but they answer different DBTL questions.

At a full `24`-culture fixed budget, the highest mean best verified utility
across all workflow variants is the ATP-only biosensor hybrid (`1.8861`).
Among the five canonical headline workflows, the full-reporter biosensor
modular hybrid is highest (`1.8842`), but the separation from conventional DBTL
(`1.8808`) and the black-box twin (`1.8641`) is small. Conventional DBTL has
higher moderate target-attainment probability (`0.5333`) than the full-reporter
hybrid (`0.4833`). This does not support a large hybrid-specific best-design
advantage.

At matched sequential-depth and calendar-time budgets, the digital-twin
workflows are stronger because they can spend calibration plus
post-calibration verification in two biological stages. At two stages,
full-reporter hybrid, black-box twin, and modular no-biosensor reach mean best
utility `1.8842`, `1.8641`, and `1.8531`, respectively, while conventional
DBTL reaches `1.6942`. At `18` scenario days, conventional and BO have only
completed the initial stage (`1.2214` mean best utility), while one-shot twin
workflows have completed calibration plus parallel verification. This
reinforces the bounded claim that the twin improves DBTL primarily through
reduced sequential biological depth.

The follow-up also resolves the biosensor ambiguity. In the final benchmark,
reporters were allowed for the biosensor-informed method, but the
manifest-generation loop used placeholder observations rather than exact
reporter outcomes. Exact reporters were not available during virtual search,
did not update regulatory state online, and did not drive later candidate
selection from previous cultures. The exact reporter artifacts are one-row
culture-level summaries, not early reporter time series. Therefore the weak
final biosensor contribution should be interpreted as a negative result for
training/calibration-style reporter use in this benchmark, not as a decisive
test of live early biosensor assimilation.

A bounded same-culture reporter proxy was run from saved culture-level reporter
summaries. Negative RMSE deltas are better and positive rank/recovery deltas
are better. Mean deltas versus nominal design-only prediction were small: ATP
reporter RMSE `-0.0001`, Spearman `+0.0004`; oxidative RMSE `-0.0004`,
Spearman `+0.0021`; full reporters RMSE `-0.0115`, Spearman `+0.0304`; pathway
reporter RMSE `-0.0194`, Spearman `+0.0487`. This suggests some same-culture
physiological information exists, especially in pathway/full reporters, but it
is not the requested 10-30% live time-series assimilation test.

## Prospective On-Demand DBTL Benchmark

The next validation layer is a prospective simulated DBTL benchmark rather
than a finite-pool acquisition benchmark. It is implemented in
`src/yeast_validation/run_prospective_dbtl_benchmark.py` and compares only two primary
workflows: `conventional_dbtl` and `pretrained_hybrid_digital_twin`.

The mandatory pre-run audit is saved in
`data/prospective_historical_training_audit.csv`. The current audit passes the
fixed-reference-strain framing:

| Audit item | Current finding |
| --- | --- |
| Teacher-training strains | `single_reference_no_edit_strain` |
| Historical cultures | `125` fixed-environment cultures |
| Teacher-training cultures | `77` training-split cultures |
| Varied environment variables | `temperature,pH,DO` over `27-33 C`, `pH 4.5-5.5`, `DO 20-80` |
| Metabolic edit dimensions in training | absent |
| Benchmark edit overlap with training | none detected in teacher-training columns |
| Training world/condition | `dynamic_congestion_feedback` |
| Reporter labels | oxidative, ATP, ER, and pathway-capacity reporters |
| Deployment inputs | `temperature,pH,DO` |
| Historical exact cost | `18,000` LP solves and `0` surrogate evaluations |

The prospective runner enforces the online dependency in code. Conventional
DBTL proposes a batch, calls the on-demand exact wet lab, appends observations,
and only then proposes the next batch. The hybrid uses the frozen teacher and
the frozen intervention library to score many virtual strain-environment
candidates without exact outcomes, then requests a small verification batch.
The old finite-pool exact cache is not an acquisition source; prospective
rollouts are written under `results/prospective_dbtl_benchmark/`.

A small smoke pilot has been run with `--mock` to validate mechanics before
spending exact Yeast9 compute. It covered `3` worlds, `2` campaign seeds per
world, both workflows, `54` physical-culture charges, and `720` hybrid virtual
evaluations. The leakage/access audit passed:
`exact_calls_logged_for_all_physical_cultures`,
`virtual_search_did_not_record_exact_outcomes`,
`old_exact_cache_not_used_as_source`,
`conventional_has_multiple_online_stages`, and
`hybrid_exact_verification_after_virtual_search`.

Because the smoke pilot used `mock_prospective_pilot_backend`, its numerical
method ranking is not scientific evidence.

The first real exact Yeast9 prospective pilot has now completed. It was run as:

```bash
.venv/bin/python src/yeast_validation/run_prospective_dbtl_benchmark.py --pilot --reset --fresh-exact --campaigns 2 --world-ids baseline_world,strong_oxidative_burden_world --conventional-batches 2,2,2 --hybrid-batch 3 --virtual-evaluations 600
```

This exact pilot covered `2` worlds, `2` campaign seeds per world, `3`
conventional online batches per campaign, and one hybrid exact verification
stage per campaign. It produced `36` fresh on-demand exact cultures:
`24` conventional cultures and `12` hybrid verification cultures. Accounting in
`data/prospective_exact_simulator_call_ledger.csv` records `5,184` Yeast9 LP
solves (`144` per culture), `0` surrogate evaluations, `0` mock rows, `0`
infeasible solves, `0` unbounded solves, `0` solver errors, and `0` skipped
intervals. The exact backend is `yeast_gem_lp` for every scientific culture.

The anti-leakage and provenance outputs now include
`data/prospective_old_cache_collision_audit.csv`,
`data/prospective_conventional_adaptivity_audit.csv`,
`data/prospective_transfer_novelty_audit.csv`, and
`data/prospective_leakage_audit.csv`. They show `fresh=36/36`,
`old_result_accessed=0`, `old_cache_collision=0`, no exact outcomes exposed to
the `2,400` hybrid virtual evaluations, and `8` later conventional adaptive
stages whose proposal timestamps follow the prior exact-result timestamps.
All `36` edit vectors and edit-environment combinations are new relative to
teacher training.

The exact pilot was therefore a readiness check, not a full powered method
claim. Its result remains archived in
`results/prospective_dbtl_benchmark/frozen_exact_pilot_20260812/`.

The full powered prospective benchmark has now completed from the frozen
manifest `data/prospective_full_run_frozen_manifest.json`. The full run used
`10` campaign seeds in each of `baseline_world` and
`strong_oxidative_burden_world`, with conventional online batches
`[4,4,4,4]`, one hybrid verification batch of `8`, and `6,000` virtual hybrid
evaluations per campaign. Execution on Vanda used isolated world-seed shard
workers and one exact-culture subprocess at a time to avoid COBRA/SymPy/GLPK
memory growth. Valid prospective exact sidecars from interrupted/pilot
prospective attempts were reused, but old finite-pool exact caches were never
used as acquisition results.

The aggregate accounting is exact:

| Quantity | Value |
| --- | ---: |
| World-seed campaign pairs | `20/20` |
| Exact physical cultures | `480` |
| Conventional cultures | `320` |
| Hybrid verification cultures | `160` |
| Yeast9 LP solves | `69,120` |
| Hybrid virtual evaluations | `120,000` |
| Fresh exact simulations | `268` |
| Valid prospective sidecar cache hits | `212` |
| Solver failures, infeasible, unbounded, skipped intervals | `0` |
| Mock rows | `0` |
| Old finite-pool result access | `0` |

All `11/11` final gates in `data/prospective_pilot_acceptance.csv` pass, and
the status is `FULL_POWERED_PROSPECTIVE_BENCHMARK_VALID`. The leakage audit
passes: hybrid virtual search records no exact outcomes; conventional later
stage proposals start after prior exact results; all exact rows use
`yeast_gem_lp`; and transfer novelty shows no edit-vector overlap with teacher
training.

Across the `20` paired world-seed campaigns, mean best final product is
`1.190680` for conventional DBTL and `1.977756` for the pretrained hybrid
digital twin. The paired hybrid-minus-conventional mean difference is
`+0.787075`, median `+0.117703`, bootstrap CI
`[+0.083642,+1.565145]`, with `13` hybrid wins and `7` losses. Mean best
productivity also favors the hybrid by `+0.065590`, and mean best product AUC
by `+1.525483`. The hybrid uses `8` fewer physical cultures and `3` fewer
biological stages per paired campaign (`8` cultures and `1` stage versus `16`
cultures and `4` stages).

World-separated final-product effects are directionally positive but less
precise: in `baseline_world`, the mean paired difference is `+0.976554`
with bootstrap CI `[-0.276445,+2.397854]`; in
`strong_oxidative_burden_world`, it is `+0.597597` with CI
`[-0.166118,+1.488991]`. This supports a combined prospective advantage but
not a strong per-world-alone superiority claim.

Search-performance accounting is split by horizon. Raw full-horizon
best-so-far AUC over physical cultures favors conventional because
conventional continues through `16` cultures while the hybrid stops at `8`.
The matched common-`8`-culture horizon is therefore the fair physical-culture
comparison: hybrid minus conventional common-horizon final-product best-so-far
AUC is `+3.402982` on average, median `+1.266266`, CI
`[+0.890797,+6.092893]`. At retrospective `90%` of each campaign's observed
best final product, the hybrid reaches threshold in mean `3.71` cultures and
`1.0` stage among campaigns where it reaches the threshold; conventional reaches
it in mean `6.78` cultures and `2.22` stages among its threshold-reaching
campaigns. These thresholds are retrospective fractions of observed campaign
best, not predeclared biological targets.

Canonical full-run outputs include
`data/prospective_dbtl_final_report.md`,
`data/prospective_full_hpc_completion_audit.csv`,
`data/prospective_campaign_summary.csv`,
`data/prospective_paired_statistical_comparisons.csv`,
`data/prospective_world_paired_summary.csv`,
`data/prospective_best_so_far_by_culture.csv`,
`data/prospective_best_so_far_by_stage.csv`,
`data/prospective_best_so_far_auc_common_budget.csv`,
`data/prospective_relative_threshold_efficiency.csv`, and
`figures/prospective_12_best_final_product_vs_cultures_full.svg` through
`figures/prospective_18_virtual_vs_physical_accounting.svg`.

## Final Boundary-Condition Validation

This final validation closes the two remaining scientific questions: data
sufficiency under trajectory-level subsampling, and dependence on one synthetic
physiology controller. The analysis is implemented in
`src/yeast_validation/analyze_final_validation_boundary_conditions.py`; aggregate outputs are
stored under `data/final_validation_prospective/`.

The execution preserved whole-culture accounting. One independent biological
trajectory is one fixed environment/strain culture with `49` time points and
`48` dynamic intervals. Time points were never split independently. The frozen
base dataset has `125` trajectories with trajectory-level splits:
`77` train, `15` validation, `15` interpolation, `10` heldout-combination, and
`8` extrapolation. The final realistic learner uses observable design and assay
quantities: controlled environment, strain/edit specification, biomass/product
trajectory information, and reporter signals. Hidden regulatory states,
effective pathway capacities, exact objective weights, future simulator states,
and hidden bottleneck variables are not given to the production prospective
learner or its virtual search. Legacy oracle-supervised code remains in the
repository as diagnostic code, but it is not mixed into this benchmark.

The Vanda execution completed the planned essential matrix. It produced
`3900/3900` Yeast9/pFBA rerank summaries, selected `624` hybrid candidates for
hidden exact verification, completed `624/624` hidden exact hybrid rollouts, and
completed the `6/6` new conventional campaigns needed for the ATP-limited and
pathway-bottleneck physiology worlds. A local-vs-Vanda fixed-candidate smoke
check matched final products to numerical tolerance and preserved candidate
ordering. The primary prospective endpoint is the common matched budget of
`8` physical cultures per campaign for both methods. The older raw comparison
in which hybrid used `8` cultures and conventional continued to `16` cultures is
retained only as secondary context, not as an equal-budget headline.

One implementation detail matters for reproducibility. The most recent positive
v2 benchmark documentation described the Stage 2 rerank as using the new MLP
surrogate scaling, but the executed v2 rerank path loaded
`RepairedVirtualScorer` with `surrogate=None` and therefore used the legacy v2
control scaling. This final validation preserves that executed v2 behavior
(`legacy_v2` scaling) for strict comparability. Corrected MLP-scale reranking
was not substituted into the primary claim.

### 1. Demonstrated Computational Result

With the same prospective budget of `8` physical cultures per campaign, the
hybrid strategy found better hidden-exact-verified designs than conventional
sequential DBTL in the tested synthetic worlds.

For the four-world physiology-robustness endpoint at full training data
(`77` training cultures), the common-8 result across `12` paired world-seed
campaigns was:

| Endpoint | Hybrid | Conventional DBTL | Hybrid minus conventional |
| --- | ---: | ---: | ---: |
| Mean best final product | `1.731517` | `0.812550` | `+0.918967` |
| Median paired final-product delta | | | `+0.736278` |
| Bootstrap CI for mean delta | | | `[+0.682314,+1.231545]` |
| Win rate | | | `12/12` |
| Mean productivity delta | | | `+0.076581` |
| Mean product-AUC delta | | | `+2.665281` |

World-level common-8 final-product deltas were all positive:

| Physiology world | Paired runs | Mean hybrid | Mean conventional | Mean delta | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_world` | `3` | `2.549989` | `1.639541` | `+0.910448` | `3/3` |
| `strong_oxidative_burden_world` | `3` | `2.178554` | `0.786675` | `+1.391879` | `3/3` |
| `atp_limited_world` | `3` | `1.038896` | `0.355993` | `+0.682904` | `3/3` |
| `pathway_bottleneck_damage_world` | `3` | `1.158629` | `0.467992` | `+0.690637` | `3/3` |

As secondary non-matched context, in the reused v2 baseline/oxidative subset the
hybrid `8`-culture endpoint also exceeded conventional after `16` cultures
across the six reused world seeds: mean hybrid `8`-culture final product
`2.364272`, mean conventional `16`-culture final product `1.368233`, mean
delta `+0.996038`, win rate `6/6`. This is useful context but not the primary
matched-budget result.

### 2. Data Sufficiency

The data-sufficiency stress test retrained the unchanged current architecture
on complete culture trajectories only: `77`, `58`, `39`, `20`, and `10`
training cultures. The `58`, `39`, `20`, and `10` settings used three
independent trajectory subsampling seeds each; the `77` setting used the full
available training split. Evaluation reused the same hidden exact simulator and
the same common `8`-culture prospective campaign budget.

The prospective result did not identify a failure threshold within the tested
range. Even the smallest tested learner trained on `10` independent culture
trajectories retained a positive exact-verified prospective advantage:

| Training cultures | Paired runs | Mean hybrid | Mean conventional | Mean delta | 95% bootstrap CI | Win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `18` | `2.494413` | `1.213108` | `+1.281305` | `[+1.010009,+1.581542]` | `18/18` |
| `20` | `18` | `2.774576` | `1.213108` | `+1.561468` | `[+1.210268,+1.922439]` | `18/18` |
| `39` | `18` | `2.491475` | `1.213108` | `+1.278367` | `[+1.006265,+1.578720]` | `18/18` |
| `58` | `18` | `2.488880` | `1.213108` | `+1.275773` | `[+1.001122,+1.579112]` | `18/18` |
| `77` | `6` | `2.364272` | `1.213108` | `+1.151164` | `[+0.749409,+1.684763]` | `6/6` |

The curve is not monotonic, which argues against interpreting the exact values
as a smooth sample-size law. The scientifically supported boundary statement is
therefore: under this synthetic system and candidate-search protocol, the
hybrid did not stop being useful down to the minimum tested learning dataset of
`10` independent culture trajectories. The experiment has not established the
behavior below `10` cultures.

Predictive overfitting was also checked at the trajectory level. Product
normalized RMSE stayed similar across train, validation, and extrapolation
splits, with small train-validation gaps:

| Training cultures | Train RMSE | Validation RMSE | Extrapolation RMSE | Validation gap |
| ---: | ---: | ---: | ---: | ---: |
| `10` | `0.140161` | `0.151164` | `0.141257` | `+0.011003` |
| `20` | `0.143868` | `0.157255` | `0.147292` | `+0.013387` |
| `39` | `0.139483` | `0.154559` | `0.144698` | `+0.015076` |
| `58` | `0.138383` | `0.148188` | `0.138328` | `+0.009804` |
| `77` | `0.139642` | `0.152332` | `0.141900` | `+0.012690` |

This does not prove that real biological datasets of this size are sufficient.
It does show that, in the audited synthetic world, the final prospective
advantage is not an artifact of timepoint-level leakage or a learner that only
fits the full `77`-culture training split.

### 3. Biological Grounding

The synthetic cell has a mechanistically grounded metabolic layer and a more
synthetic regulatory-physiology layer.

Established systems-biology approximations in the generator include Yeast9
stoichiometric mass balance, exchange-reaction bounds, quasi-steady intracellular
metabolism within slower extracellular dynamic updates, dynamic FBA-style
rollout, and pFBA-like parsimonious flux selection. Each exact culture records
Yeast9 LP accounting; the final runs used the `yeast_gem_lp` backend and did
not substitute learned-model predictions for hidden exact verification.

Biologically plausible abstractions include oxygen availability changing
metabolic feasibility, ATP/energy burden affecting growth and product tradeoffs,
oxidative burden reducing effective capacities, finite engineered pathway
capacity, bottleneck-like damage or recovery states, strain edits acting through
capacity and exchange constraints, and reporters acting as noisy partial
observations of latent stress/physiology. These are sensible causal directions,
but they are not quantitatively calibrated to wet-lab yeast measurements.

Synthetic modeling choices include the exact latent-state dimensionality,
coupling coefficients, Hill or clipping nonlinearities, thresholds, stress time
constants, reporter lag/filtering constants, noise scales, and hand-selected
capacity relationships for PSY/DES/CYC-equivalent engineered pathway controls.
Those details are invented controller parameterizations. The final
physiology-robustness test targeted the highest-risk synthetic choices by
testing baseline physiology, stronger oxidative burden, ATP-limited physiology,
and pathway-bottleneck damage physiology. The hybrid advantage survived all
four tested regimes, which reduces but does not eliminate dependence on the
synthetic controller family.

### 4. Remaining Real-World Uncertainty

The demonstrated result is computational and prospective within hidden
synthetic worlds: with the same `8`-culture physical verification budget, the
hybrid strategy used experiments more effectively than conventional sequential
DBTL in the tested generator family, and this remained true down to `10`
training cultures in the data-sufficiency sweep.

The remaining empirical question is whether real yeast provides sufficiently
informative trajectories, at a feasible number of cultures, for the learned
regulatory component to achieve the same prospective DBTL advantage observed in
the synthetic benchmark.

## Current-System Complexity Audit

The current-system complexity audit was run from cached artifacts only. It did
not generate new Yeast9 trajectories, retrain the learner, rerun DBTL, or launch
new prospective simulations. The primary reference dataset is the canonical
`125`-culture real-Yeast9 generator dataset:
`data/gem_state_space_trajectories.csv`,
`data/gem_state_space_reporters.csv`,
`data/gem_state_space_states.csv`,
`data/gem_state_space_fluxes.csv`,
`data/gem_state_space_generator_constraints.partial.csv`, and
`data/gem_state_space_environment_grid.csv`. The audit script is
`src/yeast_validation/run_current_system_complexity_audit.py`; outputs are stored in
`data/current_system_complexity_audit/` and figures are
`figures/current_system_complexity_product_observable_spectra.svg`,
`figures/current_system_complexity_hidden_metabolic_spectra.svg`, and
`figures/current_system_complexity_dynamic_intervention_spectra.svg`.

The state-space inventory separates variables as follows. Observable state
contains biomass `X`, product `B_total`, and reporter trajectories `R_ox`,
`R_atp`, `R_er`, `R_E_PSY`, `R_E_DES`, and `R_E_CYC`. Hidden physiological
state contains simulator-only `z_ox`, `z_atp`, `z_bottle`, `z_er`, and pathway
capacity states `E_PSY`, `E_DES`, and `E_CYC`. Metabolic state contains cached
selected Yeast9/pFBA fluxes and active dynamic bound/constraint vectors.
Intervention/design state contains `temperature`, `pH`, `DO`, and cached
candidate `candidate_json` strain-edit specifications. Exact candidate outcomes
come from cached prospective exact simulator call ledgers.

The compact fingerprint is:

| Layer | Entropy rank | Participation rank | Nonlinear ID | PCs / directions for 95% | PCs / directions for 99% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product raw | `1.017` | `1.005` | TwoNN `0.523`; LB `0.151` | `1` | `1` |
| Product normalized | `1.018` | `1.005` | TwoNN `0.373`; LB `0.195` | `1` | `1` |
| Observable cell | `6.928` | `3.286` | TwoNN `19.824`; LB `8.697` | `38` | `80` |
| Hidden physiology | `2.789` | `2.406` | TwoNN `0.306`; LB `0.414` | `3` | `4` |
| Metabolic flux state | `2.864` | `2.348` | TwoNN `0.309`; LB `0.162` | `3` | `4` |
| Dynamical observable state | `2.108` | `1.522` | TwoNN `9.225`; LB `6.323` | `3` | `4` |
| Intervention response | `1.923` | `1.823` | TwoNN `2.712`; LB `2.907` | `2` | `2` |

Product trajectories are almost one-dimensional: raw `B(t)` PC1 explains
`99.7626%` of variance, amplitude-normalized PC1 explains `99.7750%`, and
product-slope PC1 explains `95.5271%`. This is strong evidence that the current
output-prediction problem is intrinsically low dimensional.

The observable multichannel state is the main exception. After channel-level
standardization, observable trajectories have entropy rank `6.928`,
participation rank `3.286`, and require `38` PCs for `95%` variance and `80`
PCs for `99%`. This does not mean there are 80 strong biological degrees of
freedom: the first two PCs already explain `77.24%`, while many weak reporter
and noise components fill the tail. Nearest-neighbor estimators also see more
local structure in observables than in hidden/metabolic vectors, with Levina-
Bickel estimates near `8.4-9.0` over `k=5,8,12`.

Hidden and metabolic complexity remain compact. Hidden physiology has entropy
rank `2.789` and reaches `99%` variance in `4` PCs. Adding pathway capacities
or selected fluxes does not increase empirical dimensionality materially:
hidden-plus-capacity reaches `99%` in `4` PCs, hidden-plus-selected-fluxes in
`4` PCs, and selected flux plus active constraint state in `4` PCs. Yeast9 has
`4131` nominal reactions and `2806` metabolites, but the cached empirical
state used here contains `17` selected flux channels, all varying, and `14`
active constraint channels, `13` varying. Reaction count is therefore not the
right complexity measure for the current visited state manifold.

Dynamical block-Hankel complexity is also low. Product delay embeddings are
effectively one-dimensional. Observable delay embeddings have entropy rank
`2.108` and need `3` PCs for `95%` and `4` for `99%`; hidden-state embeddings
have entropy rank `1.850` and also need `3-4` PCs. This indicates low empirical
predictive/dynamical order, not a high-order regulatory system.

The intervention-response audit used `960` cached exact candidate outcomes
from repaired prospective benchmark ledgers. The standardized response
Jacobian from environment/edit features to phenotype descriptors has only two
important singular directions: singular-direction variance is `65.76%`,
`34.06%`, `0.179%`, and `0.001%` for the first four directions; `2` directions
explain `99%`. Thus the current DBTL design landscape is effectively
two-dimensional in phenotype response despite more nominal edit categories.

Subsampling by complete cultures confirms that most low-dimensional estimates
are stable. At `25`, `50`, `75`, `100`, and `125` cultures, product raw and
normalized trajectories always have PC95/PC99 equal to `1`. Hidden physiology
has PC95 `3` and PC99 `4` across all subsample sizes. Metabolic flux/constraint
state has PC95 `3` and PC99 `4` across all subsample sizes. Observable-cell
entropy rank rises from `5.20` at `25` cultures to `6.93` at `125`, and PC95
rises from `14` to `38`, consistent with weak high-dimensional observable tail
structure becoming visible as more cultures are sampled.

The strongest low-complexity evidence is the near-one-dimensional product
manifold, the `3-4` PC hidden/metabolic manifolds, the low block-Hankel order,
and the two-direction intervention-response Jacobian. Hidden complexity is not
substantially larger than observable complexity in the compact simulator states;
instead, the observable table has a broader weak tail from reporters and noise.
The scarce-data result at `10` training cultures is therefore **moderately
impressive but not genuinely surprising**: prospective exact verification still
matters, but the current synthetic optimization problem is empirically much
lower dimensional than its nominal Yeast9 reaction count or reporter-vector
length suggests.

Future external-model comparisons must apply the same preprocessing: generate
or sample complete trajectories, downsample to `125` independent trajectories
and `49` time points, standardize each measurement channel across all
culture-time entries, remove constant features before PCA, compute entropy rank,
participation rank, PCs for `90/95/99/99.9%`, delay-embedded ranks with the
same lag, and at least TwoNN plus Levina-Bickel nearest-neighbor IDs at
`k=5,8,12`. A published yeast model should be called **substantially more
complex** only if, after this matched downsampling, at least two biologically
central layers exceed the current system by both criteria: entropy rank above
`2x` the current value and participation rank above `1.5x` the current value,
while also exceeding the current bootstrap/subsampling upper bound. For the
most relevant baselines, that means roughly observable entropy rank
`>14` with participation rank `>5`, hidden/metabolic entropy rank `>6` with
participation rank `>3.7`, or dynamical observable entropy rank `>4.2` with
participation rank `>2.3`, under identical trajectory-count and timepoint
limits. Nominal node or reaction count alone is not sufficient.

## External High-Complexity Model Screen: Yeast rxncon CDC Model

The first external screen used the published executable rxncon yeast cell-cycle
model from Muenzner, Klipp, and Krantz, *Nature Communications* 2019
(`10.1038/s41467-019-08903-w`). The model file is
`external_models/rxncon_screening/models/CDC_S_cerevisiae.xls` from
`rxncon/models` commit `793c1407e36c64715c1e278fc03b2a3925fc28db`; the rxncon
runtime is commit `2204ac365f8ac79afcae06cd4398efe5e6cea04d`. This stage was
regulation-only: no Yeast9, LP solver, ML training, DBTL campaign, or beta-
carotene objective was run. The audit script is
`src/yeast_validation/run_rxncon_external_complexity_screen.py`; outputs are in
`data/rxncon_external_complexity_screen/` and figures are
`figures/rxncon_external_complexity_spectra.svg` and
`figures/rxncon_external_complexity_bootstrap.svg`.

The original `rxncon2boolnet.py` path did not run unchanged on the current
Python/macOS stack because `pyeda` segfaulted during satisfying-assignment
enumeration. The compatibility repair was intentionally narrow: Venn-set
satisfier enumeration was replaced by a pure-Python DNF partial-assignment
enumerator and the pyeda validation pass was skipped. Boolean model
construction, smoothing, KO/OE perturbation strategies, quantitative
contingency strictness, and BoolNet serialization remained the published rxncon
code paths.

Nominal model size is large. The source workbook has `804` reaction-list rows
and `2729` contingency-list rows; the loader expands these to `2664` rxncon
reactions, `3720` contingencies, `1967` states, and `358` components. The
perturbable compiled BoolNet has `3874` synchronous deterministic Boolean
targets: `1095` reaction targets, `2063` state targets, `358` knockout targets,
and `358` overexpression targets. The explicit external inputs used were
`[Nutrients]`, `[Pheromone]`, `[HU]`, `[LatA]`, and `[Nocodazole]`. The
perturbation panel contained `1000` trajectories and `49` timepoints:
`23` direct input cases, `4` input schedules, `218` single knockouts, `215`
single overexpressions, `413` environment-plus-genetic cases, and `127` paired
genetic perturbations. Simulation of the cached panel took `3.84` seconds total
(`0.00384` seconds per trajectory; peak resident memory about `579 MB`).

Matched `125 x 49` complexity was estimated from `10` complete-trajectory
subsets. High-dimensional internal Hankel embeddings were capped at `500`
windows per subset to keep this screening stage cheap; this affects only the
dynamical-internal estimate and is conservative for the pass/fail decision
because the measured gap is already large.

| rxncon layer | Entropy rank, median [95% range] | Participation rank, median [95% range] | 95% PCs / dirs | 99% PCs / dirs | Gate result |
| --- | ---: | ---: | ---: | ---: | --- |
| Internal regulatory state | `40.06` [`35.92`,`43.86`] | `31.22` [`26.35`,`35.34`] | `46.5` | `51.5` | Pass |
| Dynamical internal state | `40.16` [`35.87`,`44.03`] | `31.04` [`26.18`,`35.13`] | `46.5` | `51.5` | Pass |
| Observable/phenotypic state | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` | Fail |
| Dynamical observable state | `1.073` [`1.061`,`1.089`] | `1.027` [`1.022`,`1.034`] | `1` | `2` | Fail |
| Intervention response | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` | Fail |

Against the current synthetic benchmark, rxncon clearly exceeds the predeclared
high-complexity threshold in two priority layers: internal regulatory state
(`14.36x` entropy-rank ratio, `12.96x` participation-rank ratio) and dynamical
internal state (`19.03x` entropy-rank ratio, `20.42x` participation-rank
ratio). It therefore passes as a substantially higher-dimensional internal
regulatory dynamical system.

The result is not a blanket pass for a harder DBTL optimization benchmark. The
documented observable/phenotypic variables available in this Boolean CDC model
collapse to an almost one-dimensional endpoint under the current perturbation
panel, and the linear intervention-response descriptor is also rank `1`. This
means the model is high-dimensional internally but the readily observable
phenotype surface is not yet a high-dimensional engineering landscape. A later
scarce-data experiment should therefore either define a documented assay-like
objective from richer internal module activity/timing, or screen `WM_S288C` if
the project specifically requires higher observable/intervention-response
complexity before running ML/DBTL.

## rxncon Observable-Complexity Follow-Up

The follow-up question was whether rxncon's high-dimensional internal
regulatory dynamics can be exposed by biologically defensible observables. This
analysis reused the same published rxncon network, BoolNet rules, perturbation
logic, cached `1000`-trajectory panel, and `49`-timepoint simulation cache. It
did not change topology, reaction rules, contingencies, update semantics,
regulatory parameters, or perturbations; it also did not run Yeast9, pFBA, ML,
or DBTL. The executable audit is
`src/yeast_validation/run_rxncon_observable_complexity_audit.py`. Outputs are in
`data/rxncon_observable_complexity_audit/`; figures are
`figures/rxncon_observable_tier_spectra.svg`,
`figures/rxncon_observable_tier_bootstrap.svg`, and
`figures/rxncon_observable_intervention_response_spectra.svg`.

The previous phenotype layer collapsed because it used only the conservative
macroscopic CDC/global states: `[CD]`, `[ND]`, `[SEP]`, `[SEG]`, replication,
spindle, cytokinesis, morphology, stress, and error flags. In the representative
matched subset, the old observable trajectory layer had only `48` varying
flattened features, entropy rank `1.00`, participation rank `1.00`, PC95 `1`,
PC99 `1`, and PC1 explained `100%`. The old dynamical observable layer had
entropy rank `1.074`, participation rank `1.027`, and PC1 explained `98.65%`.
The old intervention-response descriptor was also rank `1`. Quantitatively,
these macroscopic states are mostly off/constant or synchronized into the same
cell-cycle/stress axis under the perturbation panel; they discard module timing
and molecular pathway information.

Three nested observable tiers were audited:

| Tier | Signals | Definition | Experimental interpretation |
| --- | ---: | --- | --- |
| A conservative | `22` | explicit macroscopic CDC/global output states | microscopy/phenotype calls such as DNA replication, spindle, cytokinesis, arrest/stress flags |
| B systems module | `30` | Tier A plus eight derived module activity traces | measurable module summaries from cyclin/CDK, replication, APC/exit, checkpoint, spindle, morphogenesis, and mating markers |
| C rich marker | `725` | Tier A plus curated explicit rxncon state targets involving measurable CDC components; transcription-delay bookkeeping counters excluded | multiplexed reporter/phosphoproteomic/localization/complex-state panel |

Signal redundancy was low in the expanded panels despite many nominal
measurements: median absolute signal correlation was `0.033` in Tier B and
`0.002` in Tier C. Tier C still has many inactive or redundant columns:
`725` nominal signals but `322` varying signals in the full panel. Effective
rank per nominal feature is therefore small, and the claim is based on
effective dimensionality, not column count.

Matched `125 x 49` complexity was bootstrapped over `10` complete-trajectory
subsets:

| Tier / representation | Entropy rank | Participation rank | PC95 | PC99 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| A trajectory | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` | Fail |
| A temporal features | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` | Fail |
| A Hankel lag-8 | `1.079` [`1.064`,`1.091`] | `1.029` [`1.023`,`1.035`] | `1` | `2` | Fail |
| B trajectory | `8.06` [`7.56`,`8.46`] | `7.31` [`6.92`,`7.78`] | `8` | `9` | Acceptable |
| B temporal features | `8.73` [`8.35`,`9.30`] | `7.03` [`6.65`,`7.62`] | `9` | `11.5` | Acceptable |
| B Hankel lag-8 | `8.13` [`7.66`,`8.36`] | `7.32` [`6.95`,`7.67`] | `8` | `9` | Acceptable |
| C trajectory | `26.06` [`18.63`,`28.71`] | `20.63` [`13.40`,`23.71`] | `28.5` | `32.5` | Strong |
| C temporal features | `25.83` [`18.47`,`28.77`] | `20.50` [`13.30`,`23.61`] | `28.5` | `32.5` | Strong |
| C Hankel lag-8 | `26.40` [`18.94`,`29.34`] | `20.70` [`13.46`,`23.84`] | `28.5` | `33.5` | Strong |

The richer phenotype vector also repaired the intervention-response collapse:

| Tier / response representation | Intervention entropy rank | Participation rank | PC95 dirs | PC99 dirs | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| A temporal/combined | `1.00` | `1.00` | `1` | `1` | Fail |
| B temporal features | `9.06` [`8.46`,`9.68`] | `7.45` [`6.80`,`8.06`] | `10` | `11.5` | Acceptable |
| B combined | `8.46` [`7.93`,`8.85`] | `7.72` [`7.21`,`8.16`] | `8` | `9` | Acceptable |
| C temporal features | `26.50` [`18.11`,`28.87`] | `20.93` [`12.41`,`23.55`] | `30.5` | `34.5` | Strong |
| C combined | `26.66` [`18.23`,`28.84`] | `21.25` [`12.52`,`23.69`] | `30.5` | `34.5` | Strong |

Trajectory-family diagnostics show that richer observables expose many more
observable response families, but not mainly as "same endpoint, different
trajectory" hidden-path ambiguity. Tier A produced only `2` endpoint patterns
and `2` trajectory patterns. Tier B produced `145` endpoint patterns and `161`
trajectory/temporal patterns. Tier C produced `242` endpoint patterns and `244`
trajectory patterns. Internal state, for context only, had `339` unique final
and trajectory patterns. Endpoint-vs-trajectory distance correlations were
near `1.0` in all observable tiers, and the fraction of pairs with very similar
endpoints but very different trajectories was `0`. Thus temporal measurement is
still useful, but the present panel's diversity is largely already visible in
rich endpoint/temporal marker descriptors rather than being hidden behind
identical endpoints.

Decision: rxncon **passes** as a genuinely higher-dimensional observable
learning/design problem if the experiment measures at least Tier B module-level
systems-biology observables, and it strongly passes with the Tier C multiplexed
marker panel. It **fails** if restricted to the original conservative
macroscopic phenotype flags. The defensible next scarce-data benchmark should
therefore use Tier B as the practical default and Tier C as a rich-observation
sensitivity or upper-observability condition.

If this benchmark is pursued, the smallest next ML/DBTL test should remain
cheap: training cultures `N = 10, 25, 50`; three learner seeds; one held-out
combination split plus one perturbational extrapolation split; `4-6` exact
verification cultures per method; and a small number of campaign seeds. Inputs
would be explicit environmental inputs plus KO/OE perturbation specifications;
training observations would be Tier B trajectories or temporal features;
hidden state would remain all unmeasured rxncon state targets; candidate
interventions would be the same explicit inputs and compiler-supported KO/OE
perturbations. Candidate scalar objectives should preserve multidimensional
regulatory difficulty, for example: fast completion of a full cell-cycle marker
program under stress, checkpoint-preserving division, or matching a desired
temporal regulatory phenotype while avoiding error/arrest markers. The final
objective should not collapse the task back to one trivial binary endpoint.

## rxncon -> Yeast9 Hidden-Generator Gate, Environment-Only v1

This stage began replacing the old hand-designed hidden physiology with a
published rxncon regulatory generator coupled to the existing Yeast9/pFBA
backend. It did not train ML, run DBTL, generate a canonical `125`-culture
dataset, or expose rxncon states to any learner. The executable gate is
`src/yeast_validation/run_rxncon_gsm_generator_gate.py`; outputs are in
`data/rxncon_gsm_generator_gate/` and figures are
`figures/rxncon_gsm_generator_*.svg`.

The hidden regulatory ground truth remains the reproduced published yeast CDC
rxncon BoolNet from Muenzner, Klipp, and Krantz, with deterministic synchronous
updates and unchanged rules/topology. The primary learner-facing contract for
this generator phase is environment-only: `[Nutrients]`, `[Pheromone]`, `[HU]`,
`[LatA]`, `[Nocodazole]`, temperature, pH, DO, and glucose uptake. The observable
preview table contains only environment, time, and predefined reporters. Raw
rxncon Boolean nodes, regulatory module truth, GSM interface controls, true GSM
bounds, fluxes, event-trigger decisions, and future phenotypes are lockbox
variables.

The v1 rxncon -> GSM interface is deterministic and one-way. It maps rxncon
module summaries and direct environmental effects into ten time-smoothed GSM
control channels:

| Channel | GSM target |
| --- | --- |
| `carbon_uptake_capacity` | glucose exchange lower bound `r_1714` |
| `oxygen_capacity` | oxygen exchange lower bound `r_1992` |
| `atp_maintenance_multiplier` | ATP maintenance lower bound `r_4046` |
| `growth_allocation_gamma` | preserved biomass-growth fraction before product optimization |
| `stress_maintenance_load` | product degradation and maintenance burden |
| `precursor_availability` | native GGPP reaction `r_0461` effective upper bound |
| `resource_translation_capacity` | global heterologous enzyme/resource capacity |
| `PSY_capacity` | `BETA_PHYTOENE_SYNTHASE` upper bound |
| `DES_capacity` | `BETA_PHYTOENE_DESATURASE` upper bound |
| `CYC_capacity` | `BETA_LYCOPENE_CYCLASE` upper bound |

An explicit GPR-name overlap inventory found `127` rxncon/Yeast9 label-match
rows, including cell-wall, nucleotide-reduction, lipid-signalling, and membrane
related reactions. The present v1 interface uses those overlaps only as an
inventory; most active control channels remain higher-level physiological
mappings rather than direct GPR-derived reaction capacities. That is a
scientific limitation to resolve before claiming a strongly GPR-grounded
interface.

Six biosensors were generated as overlapping nonlinear delayed mixtures, not
one-to-one hidden-state readouts: `R_stress`, `R_resource`, `R_checkpoint`,
`R_pathway_capacity`, `R_morphology`, and `R_energy`. Each reporter has at least
four contributors. The largest single reporter/raw hidden-state correlation was
`0.678`, below the direct-exposure safeguard threshold `0.95`; the largest
cross-reporter correlation was `0.633`. The reporters are informative but not
invertible by the current audit: mean top hidden-PC reconstruction `R^2` was
`0.262`.

The environment-only `60 x 49` regulatory/interface pilot produced the following
complexity ladder:

| Layer | Varying features | Entropy rank | Participation rank | PC95 | PC99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rxncon state | `293` | `5.82` | `4.57` | `6` | `12` |
| regulatory modules | `195` | `4.91` | `4.07` | `5` | `10` |
| GSM interface | `392` | `5.36` | `4.27` | `6` | `8` |
| biosensors, noiseless | `294` | `3.31` | `2.50` | `4` | `6` |
| biosensors, noisy | `294` | `16.38` | `6.32` | `42` | `55` |

The environment -> interface response was moderately multidimensional
(entropy rank `5.64`, participation rank `5.04`, PC95 `6`). The environment ->
biosensor response also exceeded the old approximately two-dimensional
intervention landscape (entropy rank `5.07`, participation rank `3.61`, PC95
`7`).

Sparse/event-triggered Yeast9 replay was smoke-tested on three cultures against
a dense every-step reference. Dense replay used `108` LP solves per culture;
sparse replay used `39`, a `63.9%` reduction. Mean dense-vs-sparse discrepancies
were small on this tiny subset: biomass RMSE `5.6e-5`, product RMSE `4.1e-6`,
and final-product absolute difference `5.9e-6`.

Decision: the v1 generator gate is **not accepted for canonical ML/DBTL yet**.
It is an encouraging interface smoke test, but not a sufficient high-complexity
hidden organism. Under the required environment-only contract, the rxncon state
expressed entropy rank only `5.82`, far below the `~40` internal rank observed
when KO/OE perturbations were allowed in the regulatory-only screen. The GSM
interface is more complex than the old hidden physiology (`5.36` versus
approximately `2.8` entropy rank), but complexity is already greatly compressed
before Yeast9. The noisy reporter rank is also inflated by measurement noise:
noisy/noiseless reporter entropy-rank ratio `4.96`, so noisy reporter rank must
not be counted as biological complexity.

Before generating the canonical historical dataset or training any learner, the
next generator iteration should increase biologically legitimate environment
excitation and strengthen the interface's direct GPR/metabolic grounding where
possible. The stop criterion is scientific, not computational: sparse Yeast9
appears viable, but the environment-only rxncon -> interface -> reporter chain
does not yet preserve enough of the published model's high-dimensional
regulatory dynamics.

## rxncon Hidden Process-Variability Experiment

This experiment preserved the existing v1 hidden organism and added only an
upstream hidden process layer. The deployable learner input was restricted to
nominal `T_set`, `pH_set`, and `DO_set`. Realized temperature, pH, DO,
carbon/feed, nitrogen, kLa/mixing, batch/plate/reader/position effects, raw
rxncon state, regulatory modules, GSM interface controls, reaction bounds, and
Yeast9 fluxes remained lockbox variables. No arbitrary hidden-node connections
were introduced. The executable workflow is
`src/yeast_validation/run_rxncon_process_variability_experiment.py`; outputs are in
`data/rxncon_process_variability/`; figures are
`figures/rxncon_process_variability_*.svg`.

Three matched worlds used the same nominal culture design: `perfect_control`,
`random_process`, and `random_systematic_process`. Each world generated `125`
complete independent cultures with `49` timepoints, for `375` total cultures.
The hidden process variables were conservative synthetic process assumptions:
temperature random SD `0.22 C`, pH random SD `0.035`, kLa relative SD `5.5%`,
carbon/feed relative SD `4.5%`, nitrogen relative SD `4.5%`, and DO offset SD
`1%` air saturation. The systematic world added shared batch temperature/feed/
nitrogen biases, plate pH offsets, reader DO scale/offset, and plate-position
kLa effects. These variables entered only through realized environmental
conditions, nutrient sufficiency, exchange capacities, and the existing Yeast9
environment/GSM routes.

Expanded dense-vs-event-triggered validation used `12` full 49-timepoint
cultures across the three worlds. Dense replay used `144` LP solves per culture;
event-triggered replay used `42`, a `70.8%` LP reduction. Median product RMSE
was `7.94e-6`, median final-product difference was `1.22e-5`, maximum
final-product difference was `3.22e-5`, and median biomass RMSE was `6.16e-5`.
The canonical sparse generation used `15,750` exact Yeast9 LP solves
(`5,250` per world).

The central result is negative for the rxncon part of this particular contract:
because the primary learner-facing nominal intervention space was only
`T_set,pH_set,DO_set`, and the hidden process variation never crossed the
nutrient threshold or applied pheromone/HU/LatA/nocodazole treatments, the
published rxncon Boolean state and regulatory-module summaries were constant
across cultures. Thus their effective rank is best interpreted as `0` varying
task dimensions, not as the previous environment-v1 rank of about `5-6`.
Downstream hidden process variables still affected Yeast9 through realized
temperature, pH, DO, carbon/feed, nitrogen, and kLa.

Complexity ladder:

| World | D_regulatory | D_module | D_interface | D_metabolic | D_observable bio | D_product |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| perfect control | `0` | `0` | `1.00` | `1.98` | `1.06` | `1.01` |
| random process | `0` | `0` | `2.00` | `1.99` | `1.07` | `1.01` |
| random + systematic | `0` | `0` | `1.99` | `1.97` | `1.09` | `1.01` |

Noisy observables had entropy rank about `37`, but this is a measurement-noise
tail and is not counted as structured biological complexity. Dynamical/Hankel
observable-biology rank was stable around `2.56`; product Hankel rank remained
about `1.01`. Environment-to-phenotype response rank from nominal
`T_set,pH_set,DO_set` stayed near `1`, showing that the product task remained
mostly one-dimensional even with hidden process variation.

Replicate analysis confirmed that the hidden process layer introduced real
irreducible variability. Product within-nominal replicate RMSE was effectively
zero in `perfect_control`, `4.91e-4` in `random_process`, and `7.58e-4` in
`random_systematic_process`. These correspond to within/between product
variation ratios of `0`, `0.159`, and `0.226`. Biomass within/between ratios
rose similarly from `0` to `0.170` and `0.243`.

Model comparison used identical cultures and deployable nominal inputs only.
For product NRMSE averaged over non-training splits:

| World | Hybrid/state-space + reporters | Hybrid/state-space no reporters | Direct polynomial | Nearest-environment | Full-info oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| perfect control | `0.816` | `0.345` | `0.605` | `0.201` | `0.605` |
| random process | `0.769` | `0.306` | `0.601` | `0.408` | `0.599` |
| random + systematic | `0.758` | `0.354` | `0.539` | `0.214` | `0.519` |

Reporter supervision did not help product prediction in this architecture. The
state-space model without reporter heads was consistently better than
state-space with real reporters, shuffled reporters, or random auxiliary
signals. The best deployable learned state-space result was the no-reporter
variant. The nearest-environment baseline was strongest in two of three worlds,
consistent with the product landscape being low-dimensional and heavily
interpolative. The full-information oracle that saw hidden realized process
variables improved only slightly over the direct polynomial model, indicating
that hidden process variables increased replicate variability but did not
dominate the product prediction error under these magnitudes.

Systematic-bias holdout did not cause a special collapse for the no-reporter
state-space model: in `random_systematic_process`, product NRMSE was `0.203` on
the systematic holdout versus `0.632` on validation and `0.337` on extrapolation
stress. The reporter-supervised state-space model remained worse
(`0.587` on systematic holdout). Post-hoc latent-to-GSM-interface PC regression
gave `R^2` about `0.60` in the random world and `0.66` in the systematic world,
but this is diagnostic only and was not used for training or model selection.

Decision: the compact learned model can approximate the low-dimensional
observable product/biomass response of this rxncon -> Yeast9 process-variability
generator, but the experiment does **not** demonstrate learning of an active
published rxncon regulatory manifold. Under the strict `T_set,pH_set,DO_set`
contract used here, the supported rxncon external inputs were not perturbed, so
regulation collapsed before the GSM interface. The strongest defensible claim is
therefore narrower: modest hidden process variability increases replicate
uncertainty and GSM-interface variation, but product remains nearly
one-dimensional; a compact state-space model without reporters captures this
response better than a simple polynomial direct model, while reporters do not
provide product-prediction benefit in the current training setup.

All machine-readable safeguards passed, including nominal/realized environment
separation, no realized-process or systematic-group leakage into observable
tables, rxncon/GSM/flux lockboxing, reporter many-to-many construction,
culture-level splits, systematic holdout integrity, event-triggered Yeast9
validation, and oracle labeling as non-deployable.


## Active-rxncon Structural Validation

This stage is distinct from the previous `[T,pH,DO]` hidden-process robustness
experiment. Here the benchmark deliberately drives the published yeast CDC
rxncon model through its supported external inputs: `Nutrients`, `Pheromone`,
`HU`, `LatA`, and `Nocodazole`. These controls are learner-facing external
perturbations for structural validation, not claims about the final iGEM
deployment variables.

The generator remains: external rxncon controls -> published synchronous
Boolean rxncon state -> frozen regulatory-module aggregation -> frozen
10-channel GSM interface -> real Yeast9/pFBA beta-carotene dynamics -> biomass,
product, and six mixed biosensors. Raw rxncon nodes, module truth, GSM interface
controls, constraints, and fluxes remain lockbox variables. The executable
workflow is `src/yeast_validation/run_rxncon_active_structural_validation.py`; generated
tables are under `data/rxncon_active_structural_validation/`.

Canonical dataset size was 125 complete cultures at
49 timepoints. Sparse/event-triggered Yeast9 used
5043 LP solves; the dense-vs-sparse
check had median final-product difference
3.49e-06 and
median LP reduction 70.8%.

Entropy effective ranks in the full coupled dataset were:
rxncon raw state 5.45,
regulatory modules 4.55,
GSM interface 5.22,
Yeast9 metabolic flux 2.91,
biological observables 4.10,
noiseless biosensors 3.35,
and product alone 1.27. This ladder is
the primary information-collapse result.

The primary learner comparison uses only external controls as inputs and
observable trajectories as targets; hidden simulator state is used only for
post-hoc diagnostics after model fitting. The main machine-readable summary is
`data/rxncon_active_structural_validation/main_summary.csv`.


### Reporter-Supervision Diagnostic

The active-rxncon benchmark unexpectedly showed worse product prediction when
the state-space model was supervised with all six training-time biosensors. The
diagnostic in `src/yeast_validation/run_rxncon_reporter_supervision_diagnostic.py` reused
the same 125 cultures, splits, reporters, and hidden lockbox tables; it did not
regenerate Yeast9 data or alter the generator.

The original reporter setting was already per-channel standardized, but it
summed six reporter losses alongside product and biomass. The diagnostic
therefore compared that original summed objective with a normalized objective
`L = L_product + L_biomass + lambda_R * mean(L_reporters)`, swept
`lambda_R`, computed shared-gradient norms and product-vs-reporter gradient
cosines, tested reporter ablations, latent dimensions, and culture-count
subsamples, and used hidden simulator quantities only for post-hoc recovery.

The best validation/product setting was `lambda_R=0.0` with
non-train product NRMSE 0.221, compared with
`lambda_R=0` product-only NRMSE 0.221. The original
summed reporter objective had product NRMSE 0.475.
Selected failure classifications were: reporter_overweighting, loss_scaling_problem, insufficient_latent_capacity, reporter_information_not_product_relevant.
The central result table is
`data/rxncon_reporter_supervision_diagnostic/reporter_diagnostic_summary_table.csv`.

The bounded interpretation is that this is a negative result for
training-time auxiliary reporter supervision in the current architecture and
dataset. It does not test live reporter assimilation during a new culture, and
it does not imply that the biosensors lack biological information.


### Genetic-Edit Transfer Biosensor Test

The fixed-strain active-rxncon diagnostic showed that training-time biosensor
supervision hurt beta-carotene prediction because product was highly compressed
and reporter gradients were mostly orthogonal to product gradients. This
follow-up tests a different hypothesis: reporters may be useful when the learner
must transfer across deliberate rxncon genetic perturbations.

The generator remains frozen: external rxncon controls plus rxncon
compiler-supported KO/OE edit targets drive the published CDC Boolean network,
the existing regulatory-module aggregation, the existing 10-channel GSM
interface, and real Yeast9/pFBA beta-carotene dynamics. The learner receives
only external controls plus a machine-readable edit vector. Raw rxncon state,
module truth, GSM-interface controls, constraints, and fluxes remain lockbox
variables. Tables are under `data/rxncon_edit_transfer_benchmark/`; executable
workflow: `src/yeast_validation/run_rxncon_edit_transfer_benchmark.py`.

The rxncon-only edit screen selected 18
compiler-supported edit targets that propagated into the GSM interface. The
canonical edit-transfer dataset contains 200
complete cultures and used 8418 exact Yeast9 LP
solves.

Entropy effective ranks over the coupled edit-transfer dataset were:
rxncon 17.79, modules
7.80, GSM interface
5.69, Yeast9 metabolic flux
2.79, biosensors
3.80, biological observables
4.73, and product
1.60. The critical evaluation splits are
`unseen_single_edit`, `unseen_edit_combination`, and `hard_edit_family`, not IID
fixed-strain interpolation.

The reporter-transfer hypothesis was not supported for the present
architecture. With validation-selected reporter weights, state-space
product-only NRMSE was 2.23 on `unseen_single_edit`, 0.23 on
`unseen_edit_combination`, and 3.06 on `hard_edit_family`. `R_energy`
supervision improved the hard edit-family split to 1.63 but worsened unseen
single edits to 2.57 and unseen combinations to 0.25; all-six reporter
supervision was 2.28, 0.30, and 3.65 on the same splits. The random smooth
auxiliary control was comparable to or better than real reporters on the hard
family split, so the hard-family gain is not evidence of a biosensor-specific
transfer effect.

Direct low-order baselines were much stronger than all state-space variants in
this edit-transfer task: environment/edit-to-PCA ridge reached 0.11 on unseen
single edits and 0.03 on unseen combinations, and nearest-edit/environment was
0.33 on the hard family split. The bounded conclusion is therefore negative for
training-time biosensor supervision: genetic edits increased hidden rxncon
complexity, but the beta-carotene output remained compressed enough that
reporters did not provide a reproducible deployable transfer advantage.


### Live Biosensor Assimilation / Edit-Effect Identifiability Validation

Gate 1 reused the completed 200-culture active-rxncon edit-transfer dataset and
did not regenerate Yeast9 trajectories. Genotype is encoded as 18 binary sparse
OE columns; the reference strain is the all-zero vector and edit combinations
are multi-hot. KO targets were screened, but none propagated through the frozen
v1 GSM interface, so no KO column entered the selected learner-facing library.
No biological descriptors, rxncon states, module activities, GSM-interface
truth, fluxes, or product-response summaries are included in the genotype
vector.

The product target was weakly sensitive to genotype. For final product, the
regression variance decomposition gave environment-only R2
0.563, genotype-only R2
0.127, environment+genotype R2
0.603, and incremental genotype given
environment 0.040. Median edit
trajectory effect was 0.0004198,
or 0.144 times the
reference-strain environment RMS variation.

Genotype-removal controls did not show a robust genotype-specific prediction
advantage. For PCA ridge on the primary edit-transfer splits, mean
environment+genotype product NRMSE was
0.163, environment-only was
0.168, reference-genotype was
0.168, and shuffled-genotype was
0.235. The machine-readable decision is
`EDIT_EFFECT_MEANINGFUL=False` in
`data/rxncon_edit_identifiability_live_biosensor/gate1_decision.csv`.

Because Gate 1 did not establish a strong product-level genotype-transfer task,
Gate 2 live biosensor assimilation was not run. The bounded conclusion is that
the current beta-carotene projection is too insensitive to these rxncon edit
effects to provide a clean test of whether live biosensors improve prediction
after genetic intervention. This does not invalidate live assimilation as a
future idea; it says the present edit/product dataset is not the right substrate
for that claim.

### Hidden-History / Live Biosensor State-Assimilation Validation

This gated experiment tested whether unobserved preconditioning history could
create matched-inoculum cultures with identical production environment and
genotype but different persistent rxncon states. The cheap screen evaluated 50
predeclared histories: baseline; nutrient-withdrawal, pheromone, HU, LatA, and
Nocodazole pulses at 6, 12, and 24 Boolean steps with 0, 6, or 12 recovery
steps; and four sequential-pulse histories. The published rxncon rules, six
reporters, and existing 10-channel interface were reused unchanged.

The history/interface gate failed before Yeast9. Although transfer-state rank
was nontrivial (entropy rank 4.72; participation rank 4.44), common
post-transfer forcing removed the differences immediately: production rxncon,
regulatory-module, and GSM-interface ranks were each approximately 1.0. The
only retained internal difference identified in the extended pulse audit was
`[dNTP]` after HU; the frozen interface has no `[dNTP]` channel, so this did not
reach metabolic controls. Longer 6--200-step pulses did not change that result.

The selected history panel therefore contained only four histories including
baseline, with zero histories meeting the persistent-interface criterion. No
Yeast9 cultures were generated, no LP solves were spent, and the conditional
product-ambiguity gate was not evaluated. The correct conclusion is Negative A
at the interface stage: the current rxncon-to-beta-carotene generator does not
retain enough hidden history after common transfer to justify a live-biosensor
assimilation test. This does not establish that live reporters are generally
unhelpful; it identifies the missing prerequisite as persistent history reaching
the frozen metabolic interface.
### Time-Varying DO Intervention Validation

After the hidden-history gate failed, a separate dynamic-control test asked
whether the frozen generator responds to explicitly time-varying dissolved
oxygen. Phase A reused the canonical DO values 20/40/80, the existing
temperature/pH/glucose axes, active rxncon inputs, initial conditions, staged
pFBA protocol, and frozen interface. It generated 45 exact cultures: 15
schedule types across three backgrounds, all at 49 timepoints. The schedule
panel included static controls, early/middle/late low-to-high and high-to-low
switches, and matched-exposure low/high pulses at early/middle/late positions.

The sparse exact replay used 2,286 LP solves. One representative culture was
replayed densely for the sparse gate: median final-product difference was
6.42e-6 and median LP reduction was 70.8%. Timing effects were reproducible
across the three backgrounds; switch timing reached 0.587, 0.783, and 0.974
times the corresponding static phenotype spread, while matched-pulse timing
effects were 0.057, 0.070, and 0.068. The predeclared gate therefore passed on
the reproducible switch effect, although the pulse-only timing effect was
smaller. Order-sensitive summaries improved phenotype fit over order-insensitive
DO summaries by delta R2 of 0.327 for final product, 0.402 for product AUC,
and 0.228 for final biomass.

Phase B used only observable current controls: temperature, pH, glucose
uptake, current DO, and normalized time. A coordinate polynomial, causal
recurrent model, and dynamic state-space model were trained with culture-level
schedule splits (21 train, 12 validation, 12 late-timing test). Validation
selected the dynamic state-space model. Its test trajectory NRMSE was 0.477,
compared with 0.614 for the coordinate baseline; the causal recurrent model
had seed-dependent test NRMSE values from 0.527 to 0.637. The dynamic
state-space transition explicitly received the current control at each step;
it did not receive biosensors or any hidden rxncon/GSM/flux quantity. No DBTL
benchmark was run. Full outputs and provenance are in
`data/rxncon_dynamic_intervention_validation/`.
