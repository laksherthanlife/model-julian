# Development and Reproducibility Log

> Historical implementation log. The notebook sequence is now the primary
> scientific entry point; the source-module references below document provenance
> for cached artifacts and are not the recommended way to run an experiment.

## Canonical Commands

```bash
.venv/bin/python src/yeast_validation/run_fixed_environment_validation.py
.venv/bin/python src/yeast_validation/run_fixed_environment_validation.py --fast
.venv/bin/python -m pytest tests/test_fixed_environment_validation.py
```

## Experiment Definitions

- 1A: `F_A(t,e)=V_A(e)`.
- 1B: `F_B(t,e)=V_B(e)[1-exp(-t/tau_B(e))]`.
- 1C: `F_C(t,e)=V_C(e)[1-exp(-t/tau_on(e))]exp(-t/tau_off(e))`.
- 2A: reporter-supervision control on interpolation splits.
- 2B: held-out environmental combinations.
- 2C: reporter-quality frontier.

## Seeds and Splits

Dataset seeds: `(101, 202, 303)`.
Model seeds: `(11, 22, 33)`.
Split counts: `{'extrapolation': 36, 'heldout_combination': 30, 'interpolation': 30, 'train': 192, 'validation': 24}`.

All splits are at the culture/environment level. The split manifest is
`data/environment_split_manifest.csv`.

## Outputs

Canonical tables:

- `data/fixed_environment_trajectories.csv`
- `data/fixed_environment_parameters.csv`
- `data/environment_split_manifest.csv`
- `data/canonical_metrics.csv`
- `data/heldout_model_predictions.csv`
- `data/true_vs_predicted_summary.csv`
- `data/reporter_quality_frontier.csv`
- `data/safeguards.csv`

Canonical figures are `figures/fixed_*.svg`.
The true-versus-predicted visual checks are:

- `figures/fixed_06_predicted_vs_true_curves.svg`
- `figures/fixed_08_reporter_supervision_heldout.svg`
- `figures/fixed_11_heldout_prediction_scatter.svg`

## Safeguards

Safeguards passed `24/24`. The canonical run raises an error if a
critical safeguard fails.

## Current Headline Results

| Experiment | Split | Best model | Product RMSE |
| --- | --- | --- | ---: |
| 1A | held-out combination | small_mlp_parameter | 0.0069 |
| 1B | held-out combination | small_mlp_parameter | 0.0116 |
| 1C | held-out combination | small_mlp_parameter | 0.0142 |
| 2B | held-out combination | aggregate_reporter_supervision | 0.0047 |


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


## Archive

The previous formulation, `environment time series -> latent state -> product`
with live reporter correction and mini-FBA/deployment experiments, is preserved
under `archive/time_varying_environment_validation/`.

## Prospective On-Demand DBTL Benchmark

New prospective runner:

```bash
.venv/bin/python src/yeast_validation/run_prospective_dbtl_benchmark.py --pilot --mock --reset
.venv/bin/python src/yeast_validation/run_prospective_dbtl_benchmark.py --pilot --reset --fresh-exact --campaigns 2 --world-ids baseline_world,strong_oxidative_burden_world --conventional-batches 2,2,2 --hybrid-batch 3 --virtual-evaluations 600
.venv/bin/python -m pytest tests/test_prospective_dbtl_benchmark.py
```

Purpose: answer the deployment-style question,
`pretrained fixed-environment regulation/metabolism twin + Yeast9 virtual search`
versus `conventional adaptive DBTL`, with exact simulator calls made only when a
workflow requests a culture.

Current pre-run audit: passed. The historical teacher was trained on one
reference/no-edit strain over fixed environmental variation
`[temperature, pH, DO]`, with `125` historical cultures, `77` train-split
cultures, `18,000` exact LP solves, `0` surrogate evaluations, reporter labels
available only as labels, and no metabolic edit/candidate/strain dimensions in
the teacher training tables.

Execution status: the mechanics-only mock pilot remains available as a cheap
ledger and figure smoke test, but its numerical ranking is not exact Yeast9
evidence. The first real exact prospective pilot has now completed with
`--fresh-exact`, using `2` worlds
(`baseline_world`, `strong_oxidative_burden_world`), `2` campaign seeds per
world, conventional batches `2,2,2`, hybrid verification batch `3`, and `600`
hybrid virtual evaluations per campaign. It wrote `data/prospective_*`,
`figures/prospective_*.svg`, and `results/prospective_dbtl_benchmark/`.

The exact pilot charged `36` fresh on-demand physical cultures and logged
`5,184` Yeast9 LP solves, with `2,400` hybrid virtual evaluations and no mock
or surrogate lab rows. `data/prospective_pilot_acceptance.csv` passes all
`11/11` gates, including exact backend `yeast_gem_lp`, `fresh=36/36`, no solver
failures, no old-cache result access, nonconstant candidate quality, and
conventional adaptive-stage proposals after prior exact outcomes. Current
status: `READY_FOR_FULL_PROSPECTIVE_BENCHMARK`. The full powered prospective
benchmark was not launched in this step.

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

## Stage 4: probabilistic strain-environment design

Files added for the gated Stage 4 implementation are:
`src/yeast_validation/stage4_design.py`, `src/yeast_validation/define_stage4_edit_library.py`,
`src/yeast_validation/select_stage4_exact_calibration.py`,
`src/yeast_validation/run_stage4_exact_edit_rollouts.py`,
`src/yeast_validation/train_edit_aware_metabolic_surrogate.py`,
`src/yeast_validation/audit_stage4_uncertainty.py`,
`src/yeast_validation/run_stage4_single_edit_transfer.py`,
`src/yeast_validation/run_stage4_combination_transfer.py`,
`src/yeast_validation/optimise_strain_environment.py`,
`src/yeast_validation/verify_stage4_candidates.py`,
`src/yeast_validation/audit_stage4_design_system.py`, and
`tests/test_stage4_design.py`.

Reused assets are recorded in `data/stage4_reused_asset_inventory.csv` and
include the baseline real-Yeast9 dataset, split manifest, compact controls,
fluxes, reporters, states, frozen teacher/hybrid manifests, teacher
pseudo-data, and exact replay completion files. Newly generated Stage 4 data
files are `data/strain_edit_library.csv`,
`data/stage4_exact_calibration_manifest.csv`,
`data/stage4_exact_calibration_predictions.csv`,
`data/stage4_exact_calibration_solver_accounting.csv`,
`data/stage4_edit_surrogate_training_manifest.csv`,
`data/stage4_edit_surrogate_metrics.csv`,
`data/stage4_uncertainty_calibration.csv`,
`data/stage4_acquisition_history.csv`,
`data/stage4_single_edit_transfer_metrics.csv`,
`data/stage4_combination_transfer_metrics.csv`,
`data/stage4_search_candidates.csv`,
`data/stage4_exact_verified_candidates.csv`,
`data/stage4_ranked_strain_environment_candidates.csv`,
`data/stage4_optimisation_regret.csv`, and
`data/stage4_acceptance.csv`. A compact score figure is saved as
`figures/stage4_candidate_scores.svg`.

Commands executed:

```bash
.venv/bin/python src/yeast_validation/audit_stage4_design_system.py --n-candidates 20000 --seed 6201
.venv/bin/python -m pytest \
  tests/test_gem_state_space_validation.py \
  tests/test_reporter_grounded_hybrid_distillation.py \
  tests/test_stage4_design.py
```

The edit-aware surrogate is a bootstrap ridge residual model trained on
baseline exact Yeast9 interval data. It predicts flux trajectories and
integrates biomass/product trajectories, with all fast-search rows labelled
`metabolic_backend=edit_aware_surrogate`. Teacher pseudo-data are used only as
regulatory/control coverage metadata, not exact metabolic evidence.

The current Stage 4 calibration-rollout FBA budget consumed is still `0` LP
solves. The selected initial calibration manifest contains `60` planned edited
rollouts, each with `144` expected LP solves, but
`run_stage4_exact_edit_rollouts.py` writes explicit pending solver-accounting
placeholders while Stage 3 is blocked. Separately, the exact design benchmark
below has consumed a bounded `13,824` LP solves for a finite pilot pool. The
first broad Stage 4 search evaluated `20,000` cheap surrogate candidates and
produced a diverse `10`-candidate shortlist. Stage 4 exact verification,
single-edit transfer, combination-transfer, uncertainty calibration for edited
strains, and broad optimisation regret remain blocked/pending.

Formal Stage 4 result:
`stage_4_blocked_stage3_exact_replay_incomplete`. The exact next step remains
to finish Stage 3 exact replay; after Stage 3 passes, run the planned Stage 4
exact calibration batch, then single-edit transfer validation before accepting
any strain-engineering recommendations.

## Parallel screening versus sequential strain-environment optimisation

Files added for the design-efficiency benchmark are:
`src/yeast_validation/design_benchmark.py`, `src/yeast_validation/build_design_benchmark_pool.py`,
`src/yeast_validation/query_design_benchmark_oracle.py`,
`src/yeast_validation/run_parallel_screen_benchmarks.py`,
`src/yeast_validation/run_batch_optimisation_benchmarks.py`,
`src/yeast_validation/run_sequential_optimisation_benchmarks.py`,
`src/yeast_validation/run_static_gem_baseline.py`,
`src/yeast_validation/run_reporter_informed_baseline.py`,
`src/yeast_validation/run_direct_surrogate_baseline.py`,
`src/yeast_validation/run_hybrid_design_benchmark.py`,
`src/yeast_validation/analyse_design_benchmark.py`,
`src/yeast_validation/audit_design_benchmark.py`, and
`tests/test_design_benchmark.py`.

The benchmark reuses the Stage 4 edit library and bounded environment domain.
It writes `data/design_benchmark_candidate_pool.csv`,
`data/design_benchmark_method_registry.csv`,
`data/design_benchmark_protocols.csv`,
`data/design_benchmark_oracle_cache_index.csv`,
`data/design_benchmark_oracle_solver_accounting.csv`,
`data/design_benchmark_observation_ledger.csv`,
`data/design_benchmark_round_history.csv`,
`data/design_benchmark_measurement_history.csv`,
`data/design_benchmark_strain_construction_history.csv`,
`data/design_benchmark_best_found.csv`, `data/design_benchmark_regret.csv`,
`data/design_benchmark_threshold_efficiency.csv`,
`data/design_benchmark_topk_recovery.csv`,
`data/design_benchmark_feasibility_efficiency.csv`,
`data/design_benchmark_compute_accounting.csv`,
`data/design_benchmark_seed_summary.csv`, and
`data/design_benchmark_acceptance.csv`. Compact SVG summaries are saved as
`figures/design_benchmark_*.svg`.

Commands executed:

```bash
.venv/bin/python src/yeast_validation/audit_design_benchmark.py --pool-size 180
.venv/bin/python -m pytest tests/test_design_benchmark.py
```

The pilot generated a `180`-candidate pool and measured `145` unique
candidate cache entries through the labelled
`surrogate_oracle_for_scheduler_pilot` backend. Across `12` methods and `99`
method/seed/protocol runs, the method-specific ledger contains `1,944`
measurements, the round history contains `360` rows, and the strain
construction history contains `1,118` rows. Cache reuse occurred `1,799`
times. The exact edited dynamic pFBA budget consumed is `0` rollouts and `0`
LP solves.

The protocol table separates one-shot parallel screening, batched adaptive
optimisation, and fully sequential optimisation: one-shot budgets are `8`,
`16`, `24`, and `32`; batched adaptive protocols include `q=4`, `q=8`, and
`q=16`; sequential protocols use `q=1`. The metric tables separately track
best found versus measurements, versus rounds, and versus unique constructed
strains, along with threshold efficiency, top-k recovery, feasibility
efficiency, compute accounting, and seed summaries.

Representative Stage 3 validation is recorded in
`data/stage3_representative_validation_manifest.csv` and
`data/stage3_representative_acceptance.csv`. It passes on the completed `22`
culture triples spanning train, validation, interpolation, heldout-combination,
and extrapolation splits. This only unlocks the small scheduler pilot. The
formal Stage 3 and Stage 4 scientific acceptances remain blocked until exact
dynamic pFBA evidence exists for the full replay and edited-strain candidates.

## Exact dynamic-pFBA finite-pool design benchmark

Files added for the exact benchmark are:
`src/yeast_validation/design_benchmark_exact.py`,
`src/yeast_validation/run_design_benchmark_exact_pool.py`, and
`tests/test_design_benchmark_exact.py`.

This benchmark reuses the Stage 4 edit library and existing Yeast9/pFBA
backend, but replaces the scheduler-pilot surrogate oracle with a fully cached
exact decision-tree-regulator + dynamic pFBA pool. It intentionally stays
small: `48` candidates per regime and two frozen regimes,
`weak_dynamic_regulation` and `strong_dynamic_regulation`. Per regime, roles
are `20` edit-surrogate calibration, `8` uncertainty calibration, `16`
benchmark reference-pool, and `4` final untouched verification candidates.

Commands executed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python src/yeast_validation/design_benchmark_exact.py run-one --n-per-regime 48 --candidate-index 0

for i in {1..95}; do \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python src/yeast_validation/design_benchmark_exact.py run-one --n-per-regime 48 --candidate-index "$i" || exit 1; \
done

.venv/bin/python src/yeast_validation/design_benchmark_exact.py audit
.venv/bin/python src/yeast_validation/design_benchmark_exact.py rank
.venv/bin/python src/yeast_validation/design_benchmark_exact.py benchmark
.venv/bin/python -m pytest tests/test_stage4_design.py tests/test_design_benchmark.py tests/test_design_benchmark_exact.py
```

The exact pool completed `96/96` candidates. Each candidate used `48` growth
optimisations, `48` product optimisations, and `48` pFBA optimisations, so the
run consumed `13,824` actual LP solves. Solver accounting reports `13,824`
optimal solves, `0` solver errors, `0` skipped intervals, and `0` surrogate
oracle evaluations. A long-lived process was stopped after `8` candidates when
memory high-water reached `2821.6` MB; the conservative process-per-candidate
continuation stayed around `1.2` GB peak per candidate.

The exact cache contains `96` entries under `metabolic_backend=yeast_gem_lp`.
Weak-regime objective range is `0.550118` to `4.022673`; strong-regime
objective range is `-0.042043` to `1.461934`. Weak-regime final product range
is `0.377921` to `3.097389`; strong-regime final product range is `0.186344`
to `1.198654`. The weak/strong objective-rank Spearman value over matched edit
vectors is `0.953317`, with mean absolute objective difference `1.320466`.

The exact benchmark compares six methods:
`random_parallel`, `space_filling_parallel`, `static_gem_parallel`,
`black_box_bo`, `direct_trajectory_ucb`, and
`hybrid_digital_twin_ucb`. Protocols cover budgets `8`, `16`, `24`, and `32`
for one-shot parallel, batched `q=4`, batched `q=8`, and sequential `q=1`,
with seeds `11`, `22`, and `33`. Two settings are written:
`A_existing_baseline_prior` and `B_matched_edit_calibration`.

Main result: final untouched verification candidates are excluded from method
selection, while exact-pool regret is still measured against the full finite
pool. In both setting A and setting B, `space_filling_parallel` is best by
measurement, round, and strain-construction efficiency in both regimes. Its
mean exact-pool regret is `0.210637` in the weak regime and `0.319835` in the
strong regime. Matched calibration narrows direct/hybrid regret, but does not
beat space-filling. The scientific status is therefore
`hybrid_no_clear_advantage_consistent_across_regimes`, not a hybrid win.

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

This table should be used as the primary method comparison for this benchmark:
it is more informative than pass/fail acceptance because it exposes the small
direct-versus-hybrid gap and the larger gap to space-filling.

Primary outputs:
`data/design_benchmark_regime_manifest.csv`,
`data/design_benchmark_exact_pool_manifest.csv`,
`data/design_benchmark_exact_rollouts/`,
`data/design_benchmark_exact_pool_audit.csv`,
`data/design_benchmark_exact_pool_ranking.csv`,
`data/design_benchmark_oracle_cache_index.csv`,
`data/design_benchmark_oracle_solver_accounting.csv`,
`data/design_benchmark_observation_ledger.csv`,
`data/design_benchmark_measurement_history.csv`,
`data/design_benchmark_round_history.csv`,
`data/design_benchmark_strain_construction_history.csv`,
`data/design_benchmark_best_found.csv`,
`data/design_benchmark_regret.csv`,
`data/design_benchmark_threshold_efficiency.csv`,
`data/design_benchmark_topk_recovery.csv`,
`data/design_benchmark_feasibility_efficiency.csv`,
`data/design_benchmark_diversity.csv`,
`data/design_benchmark_compute_accounting.csv`,
`data/design_benchmark_seed_summary.csv`, and
`data/design_benchmark_acceptance.csv`. Figures are saved as
`figures/design_benchmark_best_vs_measurements.*`,
`figures/design_benchmark_best_vs_rounds.*`,
`figures/design_benchmark_best_vs_strains.*`,
`figures/design_benchmark_exact_regret.*`,
`figures/design_benchmark_parallel_frontier.*`,
`figures/design_benchmark_weak_vs_strong.*`, and
`figures/design_benchmark_top_candidates.*`.

Exact next step: reuse the completed exact cache and either enlarge the finite
pool cautiously in process-per-candidate mode or turn the top exact candidates
into a small verification shortlist. Do not rerun completed exact rollouts.

## Deployment-style scientist versus hybrid benchmark

Files added for the deployment benchmark are:
`src/yeast_validation/deployment_benchmark.py`,
`src/yeast_validation/build_editable_reaction_universe.py`,
`src/yeast_validation/validate_sparse_strain_design.py`,
`src/yeast_validation/reanalyse_cached_deployment.py`,
`src/yeast_validation/analyse_deployment_benchmark.py`,
`src/yeast_validation/audit_deployment_benchmark.py`,
`src/yeast_validation/scientist_agent_tools.py`, and
`tests/test_deployment_benchmark.py`.

The old exact benchmark is preserved and relabelled as
`matched_oracle_query_acquisition_benchmark`. The new benchmark type is
`digital_twin_deployment_and_verification_benchmark`, and the executed status
is `cached_small_pool_deployment_reanalysis`. No new hidden pFBA rollouts were
run. The verifier reused exact cached outcomes only after methods requested a
candidate, and method costs charge `0` benchmark-evaluator LP solves.

Isolation method: a public sandbox is written at
`results/deployment_benchmark/scientist_agent/public_sandbox`, with only public
reaction-universe, objective, operational-scenario, heterologous-library, and
agent-protocol files copied. A hard filesystem sandbox for a separate LLM
scientist process was not available, so the status is
`llm_scientist_agent_not_run_isolation_unavailable`. The harness records the
initial public prompt in
`results/deployment_benchmark/scientist_agent/initial_system_prompt.txt` and a
filesystem audit in `data/deployment_benchmark_agent_isolation_audit.csv`.

The editable reaction universe contains `240` public native Yeast reaction IDs
across `39` subsystems plus a curated heterologous library. Sparse designs are
validated through reaction-ID checks, edit-tier limits, magnitude limits,
environment bounds, curated-addition checks, mass-balance/energy/redox
checks for curated edits, bound consistency, static growth feasibility,
beta-carotene pathway continuity, unrestricted-source rejection, and
direct-objective cheating rejection. The public-bundle preparation stage
computed baseline fluxes and FVA ranges with public static Yeast9 analysis:
`1` FBA solve, `1` pFBA solve, and `480` FVA LP solves over the `240` public
reaction subset (`482` LP solves total). These are not hidden dynamic exact
verifications. The invalid-design audit rejects direct product-demand editing,
biomass deletion, objective-coefficient editing, hidden-regulator edits,
unbounded magnitude edits, and uncurated source-reaction addition; those
proposals consume design effort but not physical cultures.

The public scientist bundle is now self-contained at
`results/deployment_benchmark/public_scientist_bundle/`. Visible files for the
future scientist include the public Yeast9 asset copy, reaction and metabolite
annotation tables, editable reaction universe, curated heterologous library,
objective configuration, environment limits, operational budget, request schema,
candidate schema example, `scientist_agent_tools.py`, tool documentation,
exchange folders, and `SCIENTIST_AGENT_HANDOFF.md`. Hidden
assets withheld are the simulator source, regulator thresholds and state logic,
exact cache, exact rollout files, exact rankings, hybrid checkpoints,
predictions, successful strain lists, and outcome-revealing project narrative.
The bundle manifest has `15` approved entries, and the leakage scan passed.

The verifier interface was smoke-tested only with
`evaluator_interface_smoke_test`, a deliberately invalid demand-reaction edit.
It wrote `round_000_validation_errors.json`, rejected the request before exact
simulation, and recorded `0` hidden exact simulations. The fresh public
scientist then wrote `round_001_request.json`, and the hidden administrator ran
the first open-ended exact verification round. The request had `8` candidates,
all new by full candidate hash. Serial exact execution completed `8` hidden
dynamic Yeast9 growth/product/pFBA simulations and `1152` LP solves (`144` per
candidate), with `0` surrogate evaluations, `0` failed candidates, and `0`
growth-failure flags. Ranking by final product was: `round_001_candidate_04`
at `0.571041`, `round_001_candidate_05` at `0.559724`,
`round_001_candidate_07` at `0.559266`, `round_001_candidate_02` at
`0.512914`, `round_001_candidate_08` at `0.423430`,
`round_001_candidate_03` at `0.399580`, and `round_001_candidate_01` and
`round_001_candidate_06` tied at `0.368007`. The public response is
`results/deployment_benchmark/exchange/public_responses/round_001_results.json`.
The compact public result CSV is
`data/deployment_benchmark_round_001_public_results.csv`, and hidden evaluator
accounting is `data/deployment_benchmark_round_001_exact_accounting.csv`.

Commands executed:

```bash
.venv/bin/python src/yeast_validation/prepare_public_scientist_bundle.py
.venv/bin/python src/yeast_validation/reanalyse_cached_deployment.py
.venv/bin/python src/yeast_validation/audit_deployment_benchmark.py
.venv/bin/python -m pytest \
  tests/test_stage4_design.py \
  tests/test_design_benchmark.py \
  tests/test_design_benchmark_exact.py \
  tests/test_deployment_benchmark.py
```

The exact next command after a fresh scientist writes its first public request
is:

```bash
.venv/bin/python src/yeast_validation/run_deployment_verifier.py \
  --request results/deployment_benchmark/exchange/incoming_requests/round_001_request.json
```

The cached deployment run wrote `408` verification-result rows from cached
exact outcomes and `0` new exact FBA rollouts. The deterministic
public-information DBTL replay has been relabelled as
`scripted_public_information_DBTL_baseline`. It used `3` seeds, `4` physical
rounds, `32` cultures, about `29` unique strains, `37.0` central-scenario
calendar days, `40.0` scientist-hours, and `210.7` resource points, reaching
`4.022673` in the weak regime and `1.461934` in the strong regime. This is not
claimed as an isolated LLM scientist result.

The hybrid cached one-shot deployments generated `10,000` virtual proposals per
run, but the audit corrects the accounting: those proposals collapse to `48`
cached eligible candidate hashes per regime, `48` cached dynamic scores per
regime, `0` sparse-surrogate scores, and `9952` generation-only proposals per
run. Batch `4` used `1` round, `4` cultures, `4` strains, `9.25` calendar days,
`10.0` scientist-hours, and `28.0` resource points, reaching `1.691205` weak
and `0.464525` strong. Batch `8` used `1` round, `8` cultures, `7` strains,
`9.25` days, and `51.0` resource points, reaching `2.482096` weak and
`0.754442` strong. Batch `12` used `1` round, `12` cultures, `10` strains,
`14.25` days, and `74.0` resource points, with the same best objective as
batch `8`. The `8+4` correction mode used `2` rounds, `12` cultures, `11`
strains, `18.50` days, and `79.0` resource points, without improving quality
over the one-shot `8/12` result.

Virtual-to-physical leverage is explicit: the scripted public-information DBTL
baseline has `4.5:1`, hybrid batch `4` has `2500:1`, hybrid batch `8` has
`1250:1`, and hybrid batch `12` or `8+4` has `833:1`. Hit rate against `90%`
of the best-known verified objective is nonzero only for the scripted baseline
in this small old pool (`0.03125` strong, `0.06250` weak); the hybrid cached
batches had `0` hit rate at that threshold.

Final comparison: the hybrid is operationally cheaper and faster but not
quality-noninferior under `delta=0.05`. The formal pilot statuses are therefore
split: `hybrid_quality_noninferiority_failed_cached_pilot` for quality, true
operational advantages for fewer rounds/cultures/strains/calendar days/effort
and lower resources, and overall
`hybrid_operationally_cheaper_but_quality_inferior_cached_pilot`. In central
resource accounting, an existing deployed-twin campaign averages `58.0`
resource points. A greenfield twin averages `740.5`, `246.8`, `148.1`, or
`74.1` resource points per campaign when amortised across `1`, `3`, `5`, or
`10` future campaigns.

Primary outputs are:
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
`data/deployment_benchmark_heterologous_library.csv`,
`data/deployment_benchmark_objective_config.csv`,
`data/deployment_benchmark_operational_scenarios.csv`,
`data/deployment_benchmark_candidate_registry.csv`,
`data/deployment_benchmark_virtual_evaluations.csv`,
`data/deployment_benchmark_verification_requests.csv`,
`data/deployment_benchmark_verification_results.csv`,
`data/deployment_benchmark_physical_rounds.csv`,
`data/deployment_benchmark_strain_constructions.csv`,
`data/deployment_benchmark_assay_usage.csv`,
`data/deployment_benchmark_people_effort.csv`,
`data/deployment_benchmark_resource_accounting.csv`,
`data/deployment_benchmark_calendar_timeline.csv`,
`data/deployment_benchmark_best_known_reference.csv`,
`data/deployment_benchmark_quality_metrics.csv`,
`data/deployment_benchmark_time_to_threshold.csv`,
`data/deployment_benchmark_hit_rate.csv`,
`data/deployment_benchmark_virtual_to_physical_leverage.csv`,
`data/deployment_benchmark_pareto_front.csv`,
`data/deployment_benchmark_seed_summary.csv`, and
`data/deployment_benchmark_acceptance.csv`. Figures are saved as
`figures/deployment_*.svg` and `figures/deployment_*.png`.

Next step: obtain a genuinely hard-isolated scientist-agent environment, then
run real isolated LLM scientist campaigns and open-ended hybrid graph-edit
campaigns with conservative process-per-candidate exact verification. Reuse
the global exact cache and do not recompute completed candidates.

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

## Open-Ended Matched Deployment Round 001

Experiment label:
`open_ended_matched_round_001_scientist_vs_hybrid`.

Conservative commands used for the hybrid round:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python src/yeast_validation/run_blinded_hybrid_round_001.py prepare

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python src/yeast_validation/run_deployment_verifier.py \
  --request results/deployment_benchmark/exchange/incoming_requests/hybrid_round_001_request.json \
  --execute-fresh \
  --hidden-regime strong_dynamic_regulation

.venv/bin/python src/yeast_validation/run_blinded_hybrid_round_001.py compare
.venv/bin/python src/yeast_validation/deployment_benchmark.py audit
.venv/bin/python -m pytest tests/test_deployment_benchmark.py -q
```

The hybrid search generated `50,000` unique valid candidate hashes from the
`240`-reaction public sparse-edit universe. Novelty to the old exact pool was
`100.0%`, exceeding the `90%` requirement. Screening was staged as generation,
static public-feature prefiltering, sparse-surrogate accounting, and short
dynamic refinement. The sparse-surrogate layer had `0` validated scores, so it
is kept as an explicit empty accounting layer rather than overclaimed. The
short `hybrid_dynamic_yeast9` refinement scored `100` candidates with `300` LP
solves before selection. The final selected `8` were frozen before the overlap
audit and exact lookup.

Fresh exact verification produced sanitized public responses for both methods.
Scientist Round 001 and hybrid Round 001 each consumed `8` hidden exact
simulations and `1152` exact LP solves. Both batches had `0` solver errors and
`0` growth failures.

Ranking at equal physical budget:

| Rank | Method | Best objective | Mean objective | Best final product | Exact sims | Exact LP solves | Physical rounds | Cultures | Resource points | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `public_bundle_scientist_round_001` | 0.816502 | 0.677740 | 0.571041 | 8 | 1152 | 1 | 8 | 46.0 | best single verified objective |
| 2 | `blinded_hybrid_round_001` | 0.802500 | 0.684860 | 0.559266 | 8 | 1152 | 1 | 8 | 56.0 | noninferior within delta `0.05` |

The formal acceptance status is
`hybrid_quality_noninferior_at_equal_physical_budget`. This is not a claim of
hybrid superiority: the scientist batch retains the best single verified
objective by `0.0140023`, while the hybrid is inside the configured
noninferiority margin and has a slightly higher mean verified objective.

New Round 001 CSV outputs:
`data/deployment_benchmark_scientist_round_001_provenance.csv`,
`data/deployment_benchmark_round_001_frozen_config.csv`,
`data/deployment_benchmark_hybrid_round_001_input_manifest.csv`,
`data/deployment_benchmark_hybrid_round_001_denied_assets.csv`,
`data/deployment_benchmark_hybrid_round_001_access_audit.csv`,
`data/deployment_benchmark_hybrid_round_001_candidate_generation.csv`,
`data/deployment_benchmark_hybrid_round_001_static_accounting.csv`,
`data/deployment_benchmark_hybrid_round_001_surrogate_accounting.csv`,
`data/deployment_benchmark_hybrid_round_001_dynamic_accounting.csv`,
`data/deployment_benchmark_hybrid_round_001_intervention_audit.csv`,
`data/deployment_benchmark_hybrid_round_001_ranked_candidates.csv`,
`data/deployment_benchmark_hybrid_round_001_selected_batch.csv`,
`data/deployment_benchmark_hybrid_round_001_overlap_audit.csv`,
`data/deployment_benchmark_hybrid_round_001_screening_funnel.csv`,
`data/deployment_benchmark_hybrid_round_001_public_results.csv`,
`data/deployment_benchmark_hybrid_round_001_exact_accounting.csv`,
`data/deployment_benchmark_matched_round_001_comparison.csv`, and
`data/deployment_benchmark_matched_round_001_acceptance.csv`.

New Round 001 figures are saved as both SVG and PNG:
`figures/deployment_round_001_best_verified_objective.*`,
`figures/deployment_round_001_candidate_distributions.*`,
`figures/deployment_round_001_product_trajectories.*`,
`figures/deployment_round_001_biomass_trajectories.*`,
`figures/deployment_round_001_screening_funnel.*`,
`figures/deployment_round_001_virtual_compute.*`,
`figures/deployment_round_001_resource_comparison.*`,
`figures/deployment_round_001_edit_subsystems.*`, and
`figures/deployment_round_001_quality_cost_pareto.*`.

The repository audit currently reports `0` missing required deployment
artifacts after adding the Round 001 files and figures. The next scientific
step is justified Round 002 adaptation based on verified Round 001 outcomes,
but no scientist Round 002 was run in this step.

## Replicated and Adaptive Open-Ended Deployment Benchmark

This phase now contains a completed three-campaign one-shot replication with
fresh public-only DBTL scientist agents for Campaigns 002 and 003. It does not
alter the hidden simulator, objective, reaction universe, noninferiority
margin, operational model, or frozen Campaign 001 results. The original
Campaign 002 duplicate registration remains archived and excluded.

Preparation command:

```bash
.venv/bin/python src/yeast_validation/prepare_deployment_report_checkpoint.py
```

Prepared/frozen outputs:

- `data/deployment_benchmark_report_checkpoint_frozen_config.csv`
- `data/deployment_benchmark_campaign_001_freeze_manifest.csv`
- `data/deployment_benchmark_campaign_registry.csv`
- `data/deployment_benchmark_campaign_002_public_manifest.csv`
- `data/deployment_benchmark_campaign_002_leakage_audit.csv`
- `data/deployment_benchmark_campaign_002_scientist_provenance.csv`
- `data/deployment_benchmark_campaign_002_hybrid_allowed_inputs.csv`
- `data/deployment_benchmark_campaign_002_hybrid_denied_inputs.csv`
- `data/deployment_benchmark_campaign_002_hybrid_access_audit.csv`
- `data/deployment_benchmark_campaign_003_public_manifest.csv`
- `data/deployment_benchmark_campaign_003_leakage_audit.csv`
- `data/deployment_benchmark_campaign_003_scientist_provenance.csv`
- `data/deployment_benchmark_campaign_003_hybrid_allowed_inputs.csv`
- `data/deployment_benchmark_campaign_003_hybrid_denied_inputs.csv`
- `data/deployment_benchmark_campaign_003_hybrid_access_audit.csv`
- `data/deployment_benchmark_report_checkpoint_summary.csv`
- `data/deployment_benchmark_report_checkpoint_acceptance.csv`
- `data/deployment_benchmark_campaign_002_scientist_public_results.csv`
- `data/deployment_benchmark_campaign_002_hybrid_public_results.csv`
- `data/deployment_benchmark_campaign_002_exact_accounting.csv`
- `data/deployment_benchmark_campaign_002_comparison.csv`
- `data/deployment_benchmark_campaign_002_acceptance.csv`
- `data/deployment_benchmark_campaign_002_request_overlap_audit.csv`
- `data/deployment_benchmark_campaign_002_invalid_registration_audit.csv`
- `data/deployment_benchmark_campaign_002_interim_invalid_duplicate_scientist_comparison.csv`
- `data/deployment_benchmark_one_shot_replication_summary.csv`
- `data/deployment_benchmark_one_shot_paired_differences.csv`
- `data/deployment_benchmark_one_shot_bootstrap.csv`
- `data/deployment_benchmark_one_shot_acceptance.csv`
- `data/deployment_benchmark_campaign_001_adaptive_requests.csv`
- `data/deployment_benchmark_campaign_001_adaptive_results.csv`
- `data/deployment_benchmark_campaign_001_adaptive_comparison.csv`
- `data/deployment_benchmark_time_to_threshold.csv`

Scientist-agent execution commands:

```bash
.venv/bin/python src/yeast_validation/run_public_scientist_agent_campaign.py --campaign-id campaign_002
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python src/yeast_validation/run_deployment_replication_campaign.py --campaign-id campaign_002 register-scientist
.venv/bin/python src/yeast_validation/run_deployment_replication_campaign.py --campaign-id campaign_002 compare

.venv/bin/python src/yeast_validation/run_public_scientist_agent_campaign.py --campaign-id campaign_003
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python src/yeast_validation/run_deployment_replication_campaign.py --campaign-id campaign_003 register-scientist
```

Campaign 003 hybrid execution commands:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python src/yeast_validation/run_deployment_replication_campaign.py --campaign-id campaign_003 --seed 33003 --n-dynamic 100 prepare-hybrid
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python src/yeast_validation/run_deployment_verifier.py --request results/deployment_benchmark/exchange/incoming_requests/campaign_003_hybrid_round_001_request.json --execute-fresh --hidden-regime strong_dynamic_regulation
.venv/bin/python src/yeast_validation/run_deployment_replication_campaign.py --campaign-id campaign_003 compare
.venv/bin/python src/yeast_validation/analyse_deployment_replication.py
```

The public scientist agents performed sequential digital DBTL planning with
public validation/static/risk information only, then submitted one parallel
physical batch of `8` cultures. Campaign 002 scientist verification used `8`
fresh exact simulations and `1152` LP solves; Campaign 003 scientist
verification also used `8` fresh exact simulations and `1152` LP solves.
Campaign 002 hybrid was preserved without rerun. Campaign 003 hybrid generated
`50,000` valid candidates, dynamically refined `100` candidates with `300`
lightweight LP solves, froze `8` selected candidates, and verified them with
`8` exact simulations and `1152` LP solves.

The registration script now rejects Campaign 002/003 scientist requests unless
they are submitted to the campaign-specific incoming paths:

```text
results/deployment_benchmark/campaign_002/exchange/incoming_requests/round_001_request.json
results/deployment_benchmark/campaign_003/exchange/incoming_requests/round_001_request.json
```

It also requires matching `campaign_id`, request SHA-256 calculation before
registration, fresh task/session metadata, public-bundle manifest hash, initial
prompt hash, creation timestamp after bundle export, and required provenance
sidecar files before exact/cache verification can proceed. A regression test
confirms that the old shared path is rejected for replication campaigns.

Current status:

| Campaign | Scientist provenance | Hybrid blinding/search | Exact verification |
| --- | --- | --- | --- |
| `campaign_001` | context isolation unverified, results frozen | completed pilot frozen | `16` total one-shot candidates verified across scientist/hybrid |
| `campaign_002` | fresh public-only agent accepted; legacy duplicate archived invalid | frozen blinded hybrid reused without rerun | scientist `8` exact / `1152` LP; hybrid `8` exact / `1152` LP |
| `campaign_003` | fresh public-only agent accepted | blinded search completed with seed `33003`; `50,000` generated; `100` dynamic refinements | scientist `8` exact / `1152` LP; hybrid `8` exact / `1152` LP |

Completed one-shot results:

| Campaign | Scientist best objective | Hybrid best objective | Hybrid-scientist best delta | Scientist mean | Hybrid mean | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `campaign_001` | `0.816502` | `0.802500` | `-0.014002` | `0.677740` | `0.684860` | complete; hybrid noninferior under `0.05` margin |
| `campaign_002` | `0.801518` | `0.678774` | `-0.122744` | `0.631249` | `0.678774` | hybrid inferior by best objective; hybrid higher mean |
| `campaign_003` | `0.905180` | `0.802500` | `-0.102681` | `0.677497` | `0.694240` | hybrid inferior by best objective; hybrid higher mean |

The one-shot classification is `one_shot_result_mixed`: hybrid
best-objective noninferiority holds in `1/3` campaigns, scientist wins best
objective in `3/3`, and hybrid wins mean objective in `3/3`. Bootstrap
intervals are labelled exploratory because `n=3`. The bounded report
conclusion is `result_depends_on_campaign_and_accounting_scenario`.

## Benchmark-readiness audit for the open-ended campaigns

The project is now in an audit-and-freeze-preparation phase. Do not interpret
the three one-shot campaigns as evidence that the hybrid reduces physical DBTL
experiments to reach a target. That claim remains untested because there is no
frozen production target, no sequential stopping rule, and no demonstrated
reduction in DBTL rounds.

New files:

- `src/yeast_validation/audit_open_ended_benchmark_readiness.py`
- `src/yeast_validation/open_ended_baselines.py`
- `tests/test_open_ended_benchmark_readiness.py`
- `data/open_ended_benchmark_readiness_report.md`
- `data/open_ended_hybrid_candidate_audit.csv`
- `data/open_ended_edit_effectiveness.csv`
- `data/open_ended_repeated_outcome_classification.csv`
- `data/open_ended_campaign_metric_reanalysis.csv`
- `data/open_ended_noninferiority_sensitivity.csv`
- `data/open_ended_information_leakage_matrix.csv`
- `data/open_ended_benchmark_freeze_spec.json`
- `data/open_ended_baseline_random_valid_proposals.csv`
- `data/open_ended_baseline_space_filling_proposals.csv`
- `data/open_ended_baseline_static_gem_proposals.csv`
- `data/open_ended_baseline_selector_documentation.csv`

Audit commands:

```bash
.venv/bin/python src/yeast_validation/audit_open_ended_benchmark_readiness.py
.venv/bin/python src/yeast_validation/audit_open_ended_benchmark_readiness.py --skip-baselines
.venv/bin/python -m pytest tests/test_open_ended_benchmark_readiness.py
```

The audit does not launch new hidden exact simulations. It reconstructs the
existing hybrid candidates from frozen requests and compares requested edits,
resolved reaction IDs, public static bound application, exact-accounting
summaries, hidden trajectories, active constraints, flux summaries, and
output-file provenance.

Repeated-outcome classifications:

| Rounded objective | Candidates | Campaigns | Unique effective phenotypes | Classification |
| ---: | ---: | --- | ---: | --- |
| `0.579391` | `2` | `campaign_001` | `1` | inactive intervention |
| `0.678774` | `19` | `campaign_001`; `campaign_002`; `campaign_003` | `1` | inactive intervention |
| `0.802500` | `3` | `campaign_001`; `campaign_003` | `1` | inactive intervention |

The important Campaign 002 result is that the hybrid selected `8` unique
requested candidates but only `1` unique effective phenotype. Its mean,
median, top-2, top-4, best, and worst objectives are all `0.678774`, so the
batch appears consistent largely because the requested edits collapse under
the current verifier.

The richer reanalysis keeps the original mixed conclusion. Scientist has the
best individual design in all three campaigns. Hybrid has the higher
whole-batch mean in all three campaigns. Hybrid has higher top-4 mean in
Campaigns 001 and 002 but not Campaign 003, and top-2/best comparisons do not
support a best-design advantage. No growth failures occurred in any batch.

Descriptive noninferiority sensitivity:

| Campaign | Hybrid-scientist best delta | Delta `0.02` | Delta `0.05` | Delta `0.10` |
| --- | ---: | --- | --- | --- |
| `campaign_001` | `-0.014002` | noninferior | noninferior | noninferior |
| `campaign_002` | `-0.122744` | inferior | inferior | inferior |
| `campaign_003` | `-0.102681` | inferior | inferior | inferior |

Three campaigns are insufficient for formal statistical noninferiority. The
table above is only descriptive sensitivity analysis.

The minimum missing baseline selectors are prepared but not hidden-exact
evaluated:

- `random_valid`: samples valid non-duplicate candidates from the same public
  design space.
- `space_filling_maximin`: greedily maximizes minimum distance over public
  environment, edit magnitude, pathway-distance, and risk features.
- `static_gem_public_rank`: ranks sampled valid candidates using public
  static-GEM features such as pFBA baseline flux, FVA span, subsystem/pathway
  annotation, environment prior, and construction risk.

The freeze spec in `data/open_ended_benchmark_freeze_spec.json` is explicitly
`prepared_not_executed`. Before any sequential DBTL acceleration experiment,
the target definition, duplicate/effective-phenotype accounting, invalid
candidate handling, stopping rules, seed policy, and reporting metrics need to
be frozen.

## Final DBTL Stage 1 Benchmark Freeze-Prep

This stage implements the freeze-prep requested before any final sequential
DBTL campaign. It does not run the final `8+4+4` benchmark. The reason is the
same one identified by the open-ended readiness audit: the benchmark can
produce distinct requested candidate hashes while collapsing to repeated
effective phenotypes.

The repaired edit path is:

```text
public candidate spec
-> sparse edit validation
-> fresh augmented Yeast9 model
-> apply_sparse_edits_to_model
-> dynamic verifier config
-> staged growth/product/pFBA solve
-> utility, target, identity, and cache accounting
```

The implementation defect was not a single hash bug. It was a biological
actionability problem amplified by edit semantics. Capacity edits against
default `1000` bounds often multiplied and clipped back to no change, dynamic
constraints could overwrite intended controls, and many reactions from the
public `240`-reaction universe had no measurable flux consequence in the tested
regime. `src/yeast_validation/deployment_benchmark.py` now supports finite target/reference
bounds, capacity decreases, reversible knockdown movement toward zero,
transport/exchange capacity semantics, and heterologous pathway-enzyme capacity
semantics.

The new freeze utility is `src/yeast_validation/final_dbtl_benchmark_freeze.py`. It produces
these Stage 1 artifacts:

- `data/final_dbtl_actionable_intervention_library.csv`
- `data/final_dbtl_excluded_interventions.csv`
- `data/final_dbtl_edit_application_diagnostics.csv`
- `data/final_dbtl_objective_config.json`
- `data/final_dbtl_target_config.json`
- `data/final_dbtl_benchmark_spec.json`
- `data/final_dbtl_method_registry.csv`
- `data/final_dbtl_campaign_manifest.csv`
- `data/final_dbtl_method_access_matrix.csv`
- `data/final_dbtl_budget_estimate.csv`
- `data/final_dbtl_readiness_decision.csv`

The retained intervention library has `22` rows across `7` mechanistic classes:
transport/exchange `6`, byproduct or carbon rerouting `6`, capacity decrease
`3`, precursor supply `2`, pathway-enzyme capacity `3`, oxygen exchange `1`,
and ATP-demand reduction `1`. The exclusions table has `221` rows: `202`
entries were not selected after reliability/diversity capping and `19` were
excluded by per-mechanism caps after the initial static screen. The retained
library includes biological actionability metadata covering mechanism class,
subsystem, construction tier, reversibility, expected phenotype direction,
risk, public evidence, and manual curation notes.

The frozen identity scheme separates three objects. `candidate_spec_hash`
tracks the requested public intervention and environment. `effective_model_hash`
tracks the edited model and verifier configuration after application.
`phenotype_fingerprint` tracks rounded observed behavior while excluding
candidate IDs, runtime IDs, and cache identifiers. Replacement logic rejects
duplicates and charges physical cultures even when hidden exact cache reuse can
avoid recomputation.

Representative exact staged-solve diagnostics were run as the cheap repair
verification. They checked `18` retained interventions and classified `2` as
effective, `15` as inactive/equivalent, and `1` as diagnostic-failed. The
diagnostic effective rate is `0.111111`, which is below the readiness bar. The
optional short exact dynamic diagnostics were therefore not run, and no final
sequential campaign was launched.

The frozen objective hash is
`19c635db7d3ae199a92feb34e3642a0b93d067729ecc583370aec6f8d7363753`. The target
hash is `b2b26e2d758e52b2d7629241f62748c072b1264dd9ecf513e8b7522e26edeb66`.
The benchmark spec hash is
`f10b282dfd5235e71460e331c9cbcf3c180f8dc1e407fcab74ce3be6c7561794`. The method
registry freezes `7` methods: hybrid digital twin, public-bundle DBTL scientist
agent, random valid, space-filling maximin, static GEM public rank,
black-box/outcome-only BO, and direct trajectory surrogate. The method access
matrix freezes public versus hidden data access boundaries and leakage checks.

Budget estimate: `168` exact simulations minimum with immediate target-stop,
`336` exact simulations maximum, `144` LP solves per full exact candidate, and
`48384` maximum LP solves. The campaign manifest freezes `3` campaign seeds and
parallel batch sizes `8`, `4`, and `4`, but the manifest status remains
freeze-prep only.

Canonical freeze-prep commands:

```bash
.venv/bin/python src/yeast_validation/final_dbtl_benchmark_freeze.py
.venv/bin/python -m pytest tests/test_final_dbtl_benchmark_freeze.py tests/test_open_ended_benchmark_readiness.py
```

Only after staged diagnostics pass should the optional small exact diagnostic be
run:

```bash
.venv/bin/python src/yeast_validation/final_dbtl_benchmark_freeze.py --run-small-exact-diagnostics
```

Current decision: `NO_GO_FINAL_DBTL_EVALUATION`. The next concrete experiment
is not the final benchmark; it is an intervention-library redesign around edits
that survive dynamic constraints and produce measurable phenotype consequences,
followed by rerunning this freeze-prep gate.

## Final DBTL Intervention-Space Redesign

The redesign starts from the previous `NO_GO_FINAL_DBTL_EVALUATION`, not from a
positive benchmark claim. The old `18` representative interventions were
reconstructed and written to
`data/final_dbtl_failed_intervention_diagnosis.csv`. The preserved result is
`2` effective, `15` inactive/equivalent, and `1` diagnostic-failed. Failure
labels include non-limiting bound changes, wrong-direction uptake edits,
alternate-pathway or pFBA compensation, dynamic-rule overwrites, and unresolved
solver failure for the ATP-maintenance probe.

The dynamically relevant subspace is now mapped in
`data/final_dbtl_dynamic_reaction_relevance.csv`. It analyses `244` public or
curated reactions/parameters using public pFBA/FVA fields, environment flux
ranges, active-bound frequency, pathway distance, GPR/gene count,
compensation-risk labels, dynamic-override risk, and a reproducible relevance
score. Hidden final benchmark outcomes are not used.

Edit semantics were rebuilt around finite, direction-aware, dynamic-surviving
operations:

- PSY/DES/CYC edits change the dynamic pathway-capacity parameters used every
  exact interval.
- Oxygen and glucose uptake edits change verifier configuration fields that set
  interval lower bounds.
- Precursor and competing-sink edits derive finite references from public
  baseline/FVA/dynamic flux evidence.
- Generic multiplication of an unconstrained `1000` capacity is not treated as
  a meaningful intervention.
- Product-loss and ATP-maintenance edits are supported by semantics and tests
  but were excluded because they failed this staged screen.

The diagnostic library contains `53` finite-semantics interventions in
`data/final_dbtl_diagnostic_intervention_library.csv`. The staged exact screen
tested the top `30` interventions across `8` representative environments,
writing `240` rows to `data/final_dbtl_staged_screening_diagnostics.csv` and
mirroring the staged result in
`data/final_dbtl_edit_application_diagnostics.csv`. Staged results were `188`
effective cases, `36` inactive/equivalent cases, and `16` diagnostic failures.
At the intervention level, `24/30` passed staged screening.

Short dynamic exact diagnostics were then run only for the `24` staged-pass
interventions, with thread caps set for BLAS/OpenMP libraries. Each case is a
paired parent/edit short-horizon exact dynamic pFBA rollout. All `24/24` passed
the short dynamic phenotype gate and wrote
`data/final_dbtl_short_dynamic_diagnostics.csv` and
`data/final_dbtl_small_exact_diagnostics.csv`. These diagnostics consumed `576`
short-dynamic LP solves (`24` paired cases times `24` LP solves per pair).

The final retained library is
`data/final_dbtl_actionable_intervention_library.csv`: `24` interventions,
`3` mechanism classes, no retained effective-model duplicates, and no retained
phenotype-equivalent duplicates at the frozen tolerance. Retained mechanisms
are direct heterologous pathway capacity `12`, transport/environment coupling
`8`, and precursor/competing-sink leverage `4`. The exclusion table has `6`
rows, all `failed_staged_exact_screen`.

The effective search-space audit is
`data/final_dbtl_design_space_compression.csv`, with a sample table in
`data/final_dbtl_design_space_sample.csv`. It sampled `360` candidate specs,
found `325` unique effective models and `325` unique diagnostic phenotypes,
and measured compression ratios of `1.107692` candidate/effective-model and
`1.0` effective-model/phenotype.

The objective and target were preserved. Objective hash:
`19c635db7d3ae199a92feb34e3642a0b93d067729ecc583370aec6f8d7363753`. Target
hash: `b2b26e2d758e52b2d7629241f62748c072b1264dd9ecf513e8b7522e26edeb66`.
Updated benchmark spec hash:
`4953ea2955d0de2f1fc4e15041739c715911bcbe3a9872533891e9d67a5f8fd0`.

Commands run for the re-gate:

```bash
.venv/bin/python src/yeast_validation/final_dbtl_benchmark_freeze.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python src/yeast_validation/final_dbtl_benchmark_freeze.py --run-small-exact-diagnostics
.venv/bin/python -m pytest tests/test_final_dbtl_benchmark_freeze.py
```

Current decision: `GO_FINAL_DBTL_EVALUATION`. The final sequential `8+4+4`
campaigns were not launched in this task. The next action is a separate,
explicit final-evaluation execution task using the frozen spec and retained
library.

## Final Sequential DBTL Execution On Vanda

The explicit final-evaluation execution task has now been completed with the
expensive exact computations offloaded to Vanda. The bundle was created by
`src/yeast_validation/prepare_final_dbtl_hpc_bundle.py`, copied as
`hpc_bundles/final_dbtl_benchmark_vanda.tar.gz`, and unpacked in
`/home/svu/e1471252/final_dbtl_benchmark_vanda_20260803_1205`. Following
`HANDOVER_VANDA.md`, access used the `vanda-codex` SSH alias and did not touch
the protected aptamer backup paths.

Queue-specific submissions were denied by the cluster, so the successful PBS
path used queue-less routing. The first successful queue-less submission was
replaced after discovering that default `python3` was Python 3.9; the completed
job was `1283942.stdct-mgmt-02` after loading
`Python/3.12.3-GCCcore-13.3.0`. The harness also supports resumable execution:
`data/final_dbtl_observations.csv`,
`data/final_dbtl_exact_cache_index.csv`, and
`results/final_dbtl_benchmark/exact_rollouts` let an interrupted run continue
without losing completed exact simulations while still charging physical
cultures.

Canonical final command sequence:

```bash
.venv/bin/python src/yeast_validation/prepare_final_dbtl_hpc_bundle.py --overwrite
qsub hpc/run_final_dbtl_vanda.pbs
.venv/bin/python src/yeast_validation/run_final_dbtl_benchmark.py --summarize-only
```

The completed local artifacts are:

- `data/final_dbtl_execution_manifest.csv`
- `data/final_dbtl_observations.csv`
- `data/final_dbtl_method_summary.csv`
- `data/final_dbtl_statistical_analysis.csv`
- `data/final_dbtl_best_so_far.csv`
- `data/final_dbtl_round_summary.csv`
- `data/final_dbtl_final_report.md`
- `figures/final_dbtl_*.svg/.png`
- `results/final_dbtl_benchmark/exact_rollouts/`

Final accounting: `35` exact culture observations, `140` rollout files, `5040`
Yeast9 dynamic pFBA LP solves, `0` solver failures, and `0` cache hits. The
execution manifest status is `complete_final_dbtl_benchmark` with `7` methods,
`3` campaigns, and `final_campaigns_executed=True`.

Overall method results:

| Method | Success rate | Median cultures to target | Mean cultures to target | Mean best utility | LP solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `space_filling_maximin` | `1.0` | `1` | `1.333333` | `2.219857` | `576` |
| `static_gem_public_rank` | `1.0` | `1` | `1.333333` | `2.189569` | `576` |
| `public_bundle_dbtL_scientist_agent` | `1.0` | `1` | `1.333333` | `2.153550` | `576` |
| `hybrid_digital_twin` | `1.0` | `1` | `1.666667` | `2.071318` | `720` |
| `black_box_outcome_only_bo` | `1.0` | `1` | `2.000000` | `1.615844` | `864` |
| `random_valid` | `1.0` | `1` | `1.666667` | `1.504449` | `720` |
| `direct_trajectory_surrogate` | `1.0` | `2` | `2.333333` | `1.944488` | `1008` |

Concrete conclusion: the final benchmark ran to completion and the repaired
intervention space avoided the earlier collapse enough for every method to
reach target. The result does not show a unique hybrid reduction in physical
DBTL experiments. Hybrid succeeds reliably, but space filling, static GEM, and
the public-bundle scientist agent reached target with fewer cultures on average
and space filling had the highest mean best utility. The final answer to the
experiment-reduction question is therefore bounded/mixed rather than a positive
hybrid-win claim.

## Harder DBTL And Distribution-Shift Execution Scaffold

The next concrete experiment is now defined around harder hidden biological
worlds and source-to-target generalisation. It starts from the final sequential
DBTL result above: the benchmark is executable, but most strong methods reach
the existing target too early, so the current endpoint is not sufficiently
discriminative. The new question is whether the hybrid digital twin is more
useful when the simulator landscape changes and when the evaluation world is
not the source world.

The implementation does not add another neural architecture. Instead,
`src/yeast_validation/design_benchmark_exact.py` now exposes a frozen world catalogue:
`baseline_world`, `strong_oxidative_burden_world`, `atp_limited_world`,
`precursor_competition_world`, `pathway_bottleneck_damage_world`,
`compensatory_metabolism_world`, and `mixed_multi_mechanism_world`. These are
biologically interpretable variants of the existing exact Yeast9 dynamic-pFBA
organism. The legacy weak/strong dynamic-regulation regimes are retained for
backward compatibility.

The campaign-preparation and exact-task runner is
`src/yeast_validation/run_harder_shift_validation.py`. It creates frozen world specs,
source-target shift pairs, harder target rules, exact task manifests,
distribution-shift decision rows, HPC execution metadata, and a Vanda bundle.

Canonical local preparation:

```bash
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py prepare
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py prepare-hpc-bundle --overwrite
.venv/bin/python -m pytest tests/test_harder_shift_validation.py tests/test_final_dbtl_benchmark.py tests/test_final_dbtl_benchmark_freeze.py
```

Canonical exact-task interface:

```bash
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py run-exact \
  --manifest data/harder_dbtl_exact_task_manifest.csv \
  --array-index 1

.venv/bin/python src/yeast_validation/run_harder_shift_validation.py summarize-gate
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py analyze-harder
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py analyze-shift
```

Generated local manifests:

- `data/harder_shift_world_manifest.csv`
- `data/harder_shift_world_specs.json`
- `data/harder_shift_target_rules.json`
- `data/harder_shift_method_access_matrix.csv`
- `data/harder_shift_hpc_execution_manifest.csv`
- `data/harder_dbtl_exact_task_manifest.csv`
- `data/distribution_shift_pair_manifest.csv`
- `data/distribution_shift_exact_task_manifest.csv`
- `data/distribution_shift_decision_manifest.csv`

Current manifest sizes are `3017` harder-DBTL exact tasks, `2352`
distribution-shift exact tasks, and `9408` distribution-shift decision rows.
The exact-task manifest rows include deterministic task IDs, hidden-world
hashes, candidate hashes, simulator version, LP backend, candidate JSON,
expected LP solves, output location, and scheduler/accounting fields. The
distribution-shift decision manifest separately records `0`, `2`, `4`, and
`8` target-world adaptation budgets without duplicating exact simulations.

The harder target rule file is
`data/harder_shift_target_rules.json`. It freezes `easy`, `moderate`, and
`hard` multi-criterion targets from per-world parent controls. Each target tier
requires final-titer improvement, biomass preservation, yield preservation,
utility improvement, bounded construction complexity, and no severe
burden/growth failure. Target-rule hash:
`5cafb5141c592707f426cbf1e32360c5d68de916adebb2f4d438051e623efcd8`.

HPC environment inspection found Vanda uses PBS/OpenPBS. The configured alias
is `vanda-codex`; the login host was `stdct-login-01`, home is
`/home/svu/e1471252`, and scratch is available at `/scratch/e1471252`.
Relevant queues include `batch_cpu`, `cpu_serial`, `cpu_parallel`, `large_mem`,
and GPU queues, but exact Yeast9 work is CPU/LP-bound and the scripts request
no GPU. The Python module used is `Python/3.12.3-GCCcore-13.3.0`.

The generated Vanda bundle is:

```text
hpc_bundles/harder_shift_validation_vanda.tar.gz
```

It was uploaded and unpacked at:

```text
/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda
```

Bundle PBS scripts:

- `hpc/run_harder_dbtl_smoke.pbs`
- `hpc/run_distribution_shift_smoke.pbs`
- `hpc/run_harder_dbtl_array.pbs`
- `hpc/run_distribution_shift_array.pbs`

Queue-specific `batch_cpu` submission was denied by cluster policy. Queue-less
PBS routing placed the smoke job onto `batch_cpu`, so the regenerated scripts
omit explicit queue selection. Each exact task requests `select=1:ncpus=1:mem=6gb`,
`walltime=04:00:00`, and exports:

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Smoke status: after one failed non-array clone due to an unset
`PBS_ARRAY_INDEX`, a corrected single-task smoke job completed successfully on
Vanda. Exact task `13f1a86176e652db284775f0` finished with status
`complete_exact_dynamic_pfba`. The result was copied back to:

```text
results/harder_dbtl/exact_rollouts/baseline_world/
data/harder_dbtl_exact_cache_index.csv
```

The harder-world difficulty gate was then submitted as PBS array
`1291064[]`, covering manifest rows `1-665`. These are the `95` shared
landscape candidates in each of the `7` frozen worlds. The array completed
and was copied back locally. The aggregate/audit commands are:

```bash
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py aggregate-cache --campaign-family harder_dbtl
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py summarize-gate
```

The completion audit is:

```text
harder_dbtl difficulty_acceptance_gate 665/665 complete, 0 pending, fraction 1.0
harder_dbtl sequential_dbtl 0/2352 complete before sequential submission
```

The local difficulty summary is
`data/harder_dbtl_difficulty_gate_summary.csv`. It now records complete
landscape metrics for all seven worlds:

| world | completed | mean final product | hard-like fraction | rank changed vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline_world` | `95/95` | `0.576823` | `0.063158` | `0.000000` |
| `strong_oxidative_burden_world` | `95/95` | `0.394546` | `0.063158` | `0.884211` |
| `atp_limited_world` | `95/95` | `0.243197` | `0.042105` | `0.915789` |
| `precursor_competition_world` | `95/95` | `0.384736` | `0.042105` | `0.947368` |
| `pathway_bottleneck_damage_world` | `95/95` | `0.257782` | `0.042105` | `0.873684` |
| `compensatory_metabolism_world` | `95/95` | `0.586152` | `0.084211` | `0.768421` |
| `mixed_multi_mechanism_world` | `95/95` | `0.157969` | `0.042105` | `0.926316` |

Gate conclusion: the difficulty landscape is usable for matched method
evaluation. ATP limitation, precursor competition, pathway bottleneck/damage,
mixed multi-mechanism stress, and oxidative burden are lower-product and
ranking-shifted relative to baseline. Compensatory metabolism is shifted but
not lower-product by the same summary; keep it as a distribution-shift world
and do not retune it after observing the result.

The harder sequential DBTL exact array was submitted as PBS array `1291479[]`,
covering rows `666-3017`. As of `2026-08-09T21:29:01+08:00`, Vanda showed no
active `harder_seq` scheduler entry. The remote output directory contained
`3017/3017` harder-DBTL summary files: the completed `665`-task gate plus all
`2352` sequential exact tasks. The matching sequential success-log count was
`2352/2352` and the failure-log count was `0`. The remote per-task cache-row
count was `3010`, but the summary files cover the full manifest.

An earlier partial harder analysis command completed locally before the full
Vanda arrays finished:

```text
status harder_dbtl_analysis_updated
observations 550
method_campaigns 35
complete_method_campaigns 34
```

It writes `data/harder_dbtl_observations.csv`,
`data/harder_dbtl_method_summary.csv`,
`data/harder_dbtl_world_method_summary.csv`,
`data/harder_dbtl_best_so_far.csv`,
`data/harder_dbtl_round_summary.csv`,
`data/harder_dbtl_validation_report.md`, and
`figures/harder_dbtl_*.svg/.png`.

The distribution-shift exact array was submitted as PBS array `1292360[]`,
covering rows `1-2352` of
`data/distribution_shift_exact_task_manifest.csv`. As of
`2026-08-09T21:29:01+08:00`, Vanda showed no active `shift_dbtl` scheduler
entry. The remote output directory contained `2352/2352` distribution-shift
summary files, the success-log count was `2352/2352`, the cache-row count was
`2352`, and the failure-log count was `0`.

Next local execution step: copy the completed Vanda outputs back and rerun the
analysis commands:

```bash
rsync -az --include='*/' --include='*.csv' --exclude='*' \
  vanda-codex:/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda/results/harder_dbtl/exact_rollouts/ \
  results/harder_dbtl/exact_rollouts/
rsync -az \
  vanda-codex:/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda/results/harder_dbtl/exact_cache_rows/ \
  results/harder_dbtl/exact_cache_rows/
rsync -az --include='*/' --include='*.csv' --exclude='*' \
  vanda-codex:/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda/results/distribution_shift/exact_rollouts/ \
  results/distribution_shift/exact_rollouts/
rsync -az \
  vanda-codex:/home/svu/e1471252/harder_shift_validation_20260807/harder_shift_validation_vanda/results/distribution_shift/exact_cache_rows/ \
  results/distribution_shift/exact_cache_rows/

.venv/bin/python src/yeast_validation/run_harder_shift_validation.py aggregate-cache --campaign-family harder_dbtl
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py analyze-harder
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py aggregate-cache --campaign-family distribution_shift
.venv/bin/python src/yeast_validation/run_harder_shift_validation.py analyze-shift
```

## Definitive Simulated DBTL Acceleration Benchmark Scaffold

The next concrete benchmark now targets whole simulated DBTL workflows rather
than one-shot finite-pool ranking or prediction RMSE. The implementation entry
point is:

```bash
.venv/bin/python src/yeast_validation/run_simulated_dbtl_acceleration_benchmark.py prepare --n-campaigns 10
.venv/bin/python src/yeast_validation/run_simulated_dbtl_acceleration_benchmark.py pilot-gate --mock
.venv/bin/python -m pytest tests/test_simulated_dbtl_acceleration_benchmark.py
```

The runner freezes a hidden wet-lab protocol in which exact Yeast9 dynamic pFBA
is the expensive simulator and culture-to-culture latent physiology is sampled
from predeclared distributions. Exact result caching is compute-only; every
workflow-requested strain-environment culture is still charged one simulated
physical culture. Virtual model evaluations are recorded separately and do not
count as cultures.

Frozen workflow methods:

- `conventional_dbtl`
- `bayesian_optimization`
- `black_box_digital_twin`
- `modular_hybrid_no_biosensors`
- `biosensor_informed_modular_hybrid`

Frozen reporter conditions:

- `no_reporters`
- `oxidative_only`
- `atp_only`
- `pathway_only`
- `full_reporters`
- `noisy_full_reporters`
- `sparse_full_reporters`

Generated files:

```text
src/yeast_validation/run_simulated_dbtl_acceleration_benchmark.py
tests/test_simulated_dbtl_acceleration_benchmark.py
data/simulated_dbtl_protocol_config.json
data/simulated_dbtl_world_manifest.csv
data/simulated_dbtl_latent_distributions.json
data/simulated_dbtl_reporter_conditions.csv
data/simulated_dbtl_method_access_matrix.csv
data/simulated_dbtl_target_config.json
data/simulated_dbtl_workflow_manifest.csv
data/simulated_dbtl_exact_culture_manifest.csv
data/simulated_dbtl_hpc_execution_manifest.csv
data/simulated_dbtl_pilot_gate.csv
data/simulated_dbtl_exact_pilot_manifest.csv
data/simulated_dbtl_exact_pilot_acceptance.csv
data/simulated_dbtl_exact_pilot_summary.csv
hpc_bundles/simulated_dbtl_acceleration_vanda/hpc/run_simulated_dbtl_exact_smoke.pbs
hpc_bundles/simulated_dbtl_acceleration_vanda/hpc/run_simulated_dbtl_exact_pilot.pbs
hpc_bundles/simulated_dbtl_acceleration_vanda/hpc/run_simulated_dbtl_exact_array.pbs
```

The prepared full manifest has `6` worlds, `10` campaign seeds, `15,840`
workflow culture-request rows, and `13,920` unique hidden-wet-lab exact culture
tasks. Protocol hash:
`2ed0c8e083537bf918013619c3ed0081bc930fd52bd725624d23940083dc1612`. The PBS
array scaffold is `1-13920`, with `select=1:ncpus=1:mem=6gb`,
`walltime=04:00:00`, no explicit queue, and BLAS/OpenMP thread caps of `1`.

The local mock pilot gate passed:

```text
hidden_wet_lab_reproducibility true
physiological_variability_nontrivial true
reporters_imperfect_but_informative true
no_hidden_information_access true
physical_culture_accounting true
virtual_evaluation_accounting true
right_censoring_logic_declared true
configs_and_hashes_frozen true
```

The real non-mock Yeast9 pilot gate has now run on Vanda. A preliminary
`12`-task smoke/pilot array, `1297243[].stdct-mgmt-02`, completed but exposed a
coverage weakness because it used one campaign seed. The accepted gate is the
expanded `18`-task pilot array `1297249[].stdct-mgmt-02`, which covers `6`
worlds, `3` campaign seeds, all `5` methods, and all `7` reporter conditions.
The accepted pilot passed all checks in
`data/simulated_dbtl_exact_pilot_acceptance.csv`: all `18` summary, trajectory,
flux, constraint, latent, and reporter artifacts exist; every task completed
with `complete_exact_dynamic_pfba`; the backend is non-mock `yeast_gem_lp`;
each task records `144` actual LP solves; solver errors, infeasible solves, and
unbounded solves are all `0`; physical-culture charge is `18`; virtual
evaluations are `900,000` and are not counted as cultures; reporter outputs are
visible only under their declared conditions and remain consistent with hidden
physiology; method access controls expose no hidden simulator internals or
future outcomes; latent culture-to-culture variability is expressed. Mean pilot
runtime was `142.143` s, with max `145.599` s.

The full exact hidden-wet-lab run is submitted on Vanda as of
`2026-08-10T02:53:38+0800`. Vanda's `max_array_size = 10000` rejected a single
`1-13920` submission, so the frozen manifest was submitted as three
non-overlapping PBS chunks without changing the benchmark:
`1297261[].stdct-mgmt-02` covers rows `1-1000`,
`1297262[].stdct-mgmt-02` covers rows `1001-10920`, and
`1297263[].stdct-mgmt-02` covers rows `10921-13920`. Current status:
`EXACT_PILOT_PASSED_FULL_VANDA_ARRAY_SUBMITTED_RUNNING`.

Acceleration migration snapshot, `2026-08-11T18:56:37+0800`: the slow
`batch_cpu` arrays `1297262[]` and `1297263[]` were held, their remaining six
running elements were allowed to finish, and the held queued leftovers were
cancelled. A synced local audit immediately before migration found `5065`
valid exact tasks, `8855` missing exact tasks, and `0` failed, duplicate,
hash-mismatch, or invalid-output tasks. The missing-only manifest is
`data/simulated_dbtl_exact_missing_manifest.csv`; it preserves the original
array index in `original_array_index` while renumbering `array_index` for the
new array.

The missing-only manifest was resubmitted through Vanda's `auto_free` route,
which routed to `cpu_serial` with `1` CPU and `6gb` memory per task. The new
accelerated array is `1300458[].stdct-mgmt-02`; at the migration verification
snapshot it had `64` finished tasks, `32` running tasks, and `8759` queued
tasks. The remote hidden-wet-lab rollout tree contained `5129` exact summary
files, and no traceback, solver-error, infeasible, or unbounded patterns were
found in the new fast-array logs. The analysis pipeline is staged in
`src/yeast_validation/analyze_simulated_dbtl_acceleration.py` and will only run after the
exact completion audit passes for all `13920` tasks.

Follow-up status, `2026-08-12T11:38:17+0800`: `1300458[]` did not finish
cleanly. It produced `12605/13920` exact output sets, then PBS placed the array
under a system hold with `7540` accelerated-array tasks expired, `1313` queued,
`0` running, and two held elements (`7533`, `7535`) marked `job held, too many
failed attempts to run`. Those held elements had no task logs, so the observed
problem is a cluster launch/retry failure rather than a recorded simulator
failure. The latest completed logs still showed normal exact culture summaries.
A remaining-only manifest,
`data/simulated_dbtl_exact_remaining_20260812_manifest.csv`, was generated for
the `1315` missing task IDs and submitted through the same frozen exact runner.
Cleanup array `1301555[].stdct-mgmt-02` routed to `cpu_serial`; at the
verification snapshot it had `28` running tasks, `1287` queued tasks, and `0`
expired cleanup tasks.

Second follow-up status, `2026-08-12T13:13:16+0800`: the `cpu_serial` cleanup
strategy made progress but was not stable enough for the final tail. Cleanup
array `1301555[]` contributed `28` more exact output sets before entering the
same PBS system-hold state. A split-tail rescue submitted `54` smaller
`auto_free` arrays with at most `24` tasks each; these raised the completed
output count to `12722/13920` but also developed partial launch holds. After
all running split-tail subjobs finished, the remaining queued/held split-tail
work was cancelled. A fresh final-tail manifest,
`data/simulated_dbtl_exact_final_tail_20260812_manifest.csv`, was generated for
the `1198` still-missing task IDs and submitted as
`1301767[].stdct-mgmt-02` without an explicit queue. Vanda routed this final
tail to `batch_cpu`; verification showed `6` running tasks, `1192` queued
tasks, and `0` expired tasks.

Third follow-up status, `2026-08-12T15:04:17+0800`: the final `batch_cpu` tail
was used as a stable fallback until its active tasks drained, then the held
queued leftovers were cancelled. A new range-tail manifest,
`data/simulated_dbtl_exact_range_tail_20260812_manifest.csv`, was generated at
`12992/13920` completed exact outputs, leaving `928` missing exact tasks. To
avoid the PBS array launch-hold mode, the remaining work was moved to non-array
`cpu_parallel` range workers. Worker `1301958.stdct-mgmt-02` runs rows
`465-928` with `16` CPUs and `20gb`; worker `1301972.stdct-mgmt-02` runs rows
`1-464` with `8` CPUs and `12gb` after a `16`-CPU attempt for that half was
held before user code. Both active workers are running with no hold at the
status snapshot. Latest exact summary count: `13024/13920`.

Final status, `2026-08-12T17:08:00+0800`: the exact Vanda execution is complete
and locally audited. The remote output tree reached `13920/13920` summaries,
reporters, trajectories, and flux files, and PBS showed no remaining `sim_dbtl`
jobs. Local sync initially missed latent JSON files, causing the audit to mark
the post-migration outputs invalid for missing `latents`; after syncing JSON
artifacts, `src/yeast_validation/analyze_simulated_dbtl_acceleration.py completion-audit`
passed with `13920/13920` valid exact tasks, `0` missing, `0` failed, `0`
duplicates, `0` hash mismatches, and `0` invalid outputs. The final analysis
completed and wrote `data/simulated_dbtl_final_digital_proof.md`,
`data/simulated_dbtl_final_claim_audit_moderate.csv`, and the supporting
tables/figures. For the moderate target tier, the final claim outcome is
`NEUTRAL`: the canonical full-reporter biosensor hybrid is better than Bayesian
optimisation on cultures/rounds and quality, but it does not reduce cultures or
rounds relative to conventional DBTL and has essentially matched utility
(`+0.00343` mean best-utility delta).
The hard-tier sensitivity analysis also completed and stayed `NEUTRAL`:
canonical full-reporter hybrid target attainment `0.15`, median censored
cultures `25.0`, median censored rounds `4.0`, mean cultures saved versus
conventional DBTL `-1.9`, and mean best-utility delta versus conventional DBTL
`0.00343`.

Corrected wall-clock re-analysis, `2026-08-12`: no new Yeast9 simulations were
launched. The analysis in
`src/yeast_validation/analyze_simulated_dbtl_wallclock_reanalysis.py` distinguishes total
physical cultures from sequential biological depth. The dependency audit found
that manifest-generation proposal batches used placeholder observations
(`utility=0.0`) rather than exact outcomes, so digital-twin post-calibration
candidates can be evaluated as one parallel verification batch without
information leakage. Conventional DBTL and Bayesian optimisation remain
adaptive baselines by deployment semantics. Outputs:
`data/simulated_dbtl_wallclock_reanalysis.md`,
`data/simulated_dbtl_wallclock_final_table_moderate.csv`,
`data/simulated_dbtl_wallclock_summary_moderate.csv`,
`data/simulated_dbtl_wallclock_world_summary_moderate.csv`, and figures
`figures/simulated_dbtl_wallclock_*.png/.svg`. For the canonical full-reporter
biosensor hybrid with cached top-16 final verification, target attainment is
`0.4833`, mean physical cultures `23.47`, mean sequential biological stages
`1.967`, and mean scenario calendar time excluding compute `17.7` days.
Conventional DBTL has target attainment `0.5333`, mean cultures `20.67`, mean
stages `3.2`, and `31.0` days. The corrected conclusion is verdict `B`:
wall-clock depth improves, but only by spending more parallel culture capacity;
target-attainment reliability is not improved versus conventional DBTL.

## Best-Found Utility Follow-Up

Follow-up analysis, `2026-08-12`: the completed exact campaign was reused to
answer a complementary fixed-budget question:

```text
same exact evaluated universe + same operational budget
-> highest verified strain-environment utility found
```

No new Yeast9 simulations or HPC jobs were launched. The analysis script is:

```bash
MPLCONFIGDIR=/private/tmp/ydt_matplotlib .venv/bin/python src/yeast_validation/analyze_simulated_dbtl_best_found_followup.py
```

New outputs:

```text
data/simulated_dbtl_best_found_followup.md
data/simulated_dbtl_fixed_budget_best_found.csv
data/simulated_dbtl_fixed_budget_best_found_summary.csv
data/simulated_dbtl_fixed_budget_paired_effects_vs_blackbox.csv
data/simulated_dbtl_fixed_budget_winners.csv
data/simulated_dbtl_best_found_pareto.csv
data/simulated_dbtl_blackbox_hybrid_world_comparison.csv
data/simulated_dbtl_blackbox_hybrid_ranking_correlation.csv
data/simulated_dbtl_sensor_information_flow.csv
data/simulated_dbtl_live_reporter_feasibility_audit.csv
data/simulated_dbtl_sensor_assimilation_proxy_metrics.csv
data/simulated_dbtl_sensor_assimilation_proxy_deltas.csv
figures/simulated_dbtl_followup_*.svg
```

At the full `24`-culture budget, the ATP-only biosensor hybrid has the highest
mean best verified utility across all variants (`1.8861`). Among canonical
headline workflows, the full-reporter biosensor hybrid is highest (`1.8842`),
but the margin is small versus conventional DBTL (`1.8808`) and black-box twin
(`1.8641`). Moderate target attainment remains better for conventional DBTL
(`0.5333`) than the full-reporter hybrid (`0.4833`).

At matched two-stage or `18`-day budgets, the one-shot digital-twin workflows
outperform conventional/BO because they complete calibration plus parallel
verification within that budget. Mean best verified utility at two stages is
`1.8842` for full-reporter hybrid, `1.8641` for black-box, `1.8531` for
modular no-biosensor, and `1.6942` for conventional DBTL. At `18` days,
conventional and BO have only completed the initial stage (`1.2214` mean best
utility).

Black-box versus modular/hybrid comparison by world does not show a decisive
hybrid win. Full-reporter hybrid minus black-box mean utility deltas are:
`+0.1069` in ATP-limited, `+0.0103` in baseline, `+0.0632` in mixed mechanism,
`+0.0161` in pathway-damage, `+0.0012` in precursor-competition, and `-0.0768`
in strong oxidative burden worlds. Campaign-level Spearman correlations with
black-box best utility are high: `0.7786` for modular no-biosensor and `0.8127`
for full-reporter hybrid. The tie therefore persists across harder worlds
enough that modularity should be claimed mainly as interpretability and
auditability, not broad optimisation superiority.

The biosensor information-flow audit found that reporters were allowed for the
biosensor method, but exact reporter values were not available during virtual
search, did not update regulatory state online, and did not drive later
candidate selection from previous exact cultures. Saved reporter files contain
one culture-level row, not 10%, 20%, or 30% reporter trajectories. Therefore
the completed exact benchmark tested reporter-enabled training/calibration and
reporter-condition branches, not true early live biosensor assimilation.

A bounded same-culture reporter proxy was run from saved reporter summaries.
Mean deltas versus nominal design-only prediction were small but favorable for
full/pathway reporters: full reporters changed utility RMSE by `-0.0115`,
Spearman by `+0.0304`, and top-20% recovery by `+0.0393`; pathway reporter
changed RMSE by `-0.0194`, Spearman by `+0.0487`, and top-20% recovery by
`+0.0601`. ATP and oxidative reporters were nearly neutral. This proxy should
not be interpreted as the requested early live reporter time-series test.

## Full Prospective On-Demand DBTL Benchmark

The prospective on-demand benchmark has now been run as a full powered
world-seed comparison rather than a finite-pool acquisition replay. The frozen
protocol is recorded in `data/prospective_full_run_frozen_manifest.json` and
compares only `conventional_dbtl` with `pretrained_hybrid_digital_twin`.
Campaigns cover `10` seeds in `baseline_world` and `10` seeds in
`strong_oxidative_burden_world`. Conventional DBTL uses four online physical
batches of `4` cultures; the hybrid performs `6,000` virtual evaluations and
then verifies `8` physical cultures.

The full Vanda run completed with memory-safe exact subprocess workers. The
accepted array was `1306345[]`; earlier in-process arrays were held or
interrupted after producing valid prospective exact sidecars. The aggregation
script is:

```bash
.venv/bin/python src/yeast_validation/aggregate_prospective_hpc_shards.py --overwrite
```

The canonical aggregate contains exactly `20/20` completed world-seed shards,
`480` exact physical cultures, `69,120` Yeast9 LP solves, `120,000` hybrid
virtual evaluations, and `0` solver failures. Of the `480` exact rows, `268`
were fresh simulations and `212` were valid prospective sidecar cache hits.
The old finite-pool exact cache was not accessed as a result source.

Final status is `FULL_POWERED_PROSPECTIVE_BENCHMARK_VALID`, with all `11/11`
acceptance gates passing in `data/prospective_pilot_acceptance.csv`. Key
anti-leakage checks pass: `exact_outcome_accessed=0` for virtual evaluations,
`old_result_accessed=0`, no mock rows, all exact rows use `yeast_gem_lp`, and
conventional adaptive proposals occur after prior exact observations.

The paired biological outcome favors the hybrid in the combined analysis. Mean
best final product is `1.190680` for conventional DBTL and `1.977756` for the
hybrid. The paired hybrid-minus-conventional final-product difference across
the `20` world-seed pairs is mean `+0.787075`, median `+0.117703`, bootstrap CI
`[+0.083642,+1.565145]`, with `13` hybrid wins and `7` losses. Mean paired
productivity difference is `+0.065590`; mean paired product-AUC difference is
`+1.525483`.

The resource and stage comparison is unambiguous: the hybrid uses `8` fewer
physical cultures and `3` fewer biological stages per paired campaign. Search
performance is reported two ways. Raw full-horizon best-so-far AUC is not a
matched-budget headline because conventional has `16` culture points and the
hybrid has `8`; it is therefore preserved as a full-trajectory diagnostic.
Over the shared `8`-culture horizon, hybrid final-product best-so-far AUC is
higher by mean `+3.402982` and median `+1.266266`, with bootstrap CI
`[+0.890797,+6.092893]`.

World-separated final-product deltas are positive but individually uncertain:
`+0.976554` in `baseline_world` with CI `[-0.276445,+2.397854]`, and
`+0.597597` in `strong_oxidative_burden_world` with CI
`[-0.166118,+1.488991]`. The correct bounded conclusion is that the hybrid
shows a combined prospective advantage with fewer physical cultures and fewer
biological stages, while the two-world split remains too small for strong
world-specific superiority claims.

New or updated outputs include:

```text
src/yeast_validation/run_prospective_campaign_worker.py
src/yeast_validation/run_prospective_exact_worker.py
src/yeast_validation/aggregate_prospective_hpc_shards.py
src/yeast_validation/prepare_prospective_dbtl_hpc_bundle.py
data/prospective_dbtl_final_report.md
data/prospective_full_hpc_completion_audit.csv
data/prospective_campaign_summary.csv
data/prospective_paired_statistical_comparisons.csv
data/prospective_world_paired_summary.csv
data/prospective_best_so_far_by_culture.csv
data/prospective_best_so_far_by_stage.csv
data/prospective_best_so_far_auc.csv
data/prospective_best_so_far_auc_common_budget.csv
data/prospective_relative_threshold_efficiency.csv
results/prospective_dbtl_benchmark/hpc_shards/
results/prospective_dbtl_benchmark/on_demand_exact_rollouts/
figures/prospective_12_best_final_product_vs_cultures_full.svg
figures/prospective_13_best_productivity_vs_cultures_full.svg
figures/prospective_14_best_product_auc_vs_cultures_full.svg
figures/prospective_15_paired_final_product_scatter.svg
figures/prospective_16_world_paired_final_product_difference.svg
figures/prospective_17_cultures_to_relative_threshold.svg
figures/prospective_18_virtual_vs_physical_accounting.svg
```

## Final Boundary-Condition Experiments

The final boundary-condition validation was executed on Vanda rather than
reduced for local runtime. The completed workflow used exact Yeast9/pFBA
reranking and hidden exact simulator verification, with process-level
parallelism across independent candidates/campaigns. Local-vs-HPC validation on
a fixed candidate set matched final products to numerical tolerance and
preserved ranking order.

The primary generated and aggregation scripts are:

```text
src/yeast_validation/run_final_data_sufficiency_sweep.py
src/yeast_validation/generate_final_validation_hpc_shortlists.py
src/yeast_validation/run_final_validation_rerank_worker.py
src/yeast_validation/select_final_validation_hybrid_exact.py
src/yeast_validation/run_final_validation_hidden_exact_worker.py
src/yeast_validation/run_final_validation_conventional_worker.py
src/yeast_validation/analyze_final_validation_boundary_conditions.py
```

The Vanda run completed `3900/3900` rerank summaries,
`624/624` hybrid hidden-exact rollouts, and `6/6` new conventional campaigns.
The aggregate endpoint files are:

```text
data/final_validation_prospective/final_validation_exact_endpoints_common8.csv
data/final_validation_prospective/final_validation_data_sufficiency_common8_comparisons.csv
data/final_validation_prospective/final_validation_data_sufficiency_summary_by_N.csv
data/final_validation_prospective/final_validation_data_sufficiency_summary_by_N_worldseed_averaged.csv
data/final_validation_prospective/final_validation_predictive_overfitting_summary.csv
data/final_validation_prospective/final_validation_physiology_robustness_common8_comparisons.csv
data/final_validation_prospective/final_validation_physiology_robustness_summary_by_world.csv
data/final_validation_prospective/final_validation_physiology_robustness_summary_overall.csv
```

The final common-budget accounting is by independent culture trajectory, not by
time point. One culture is one fixed environment/strain trajectory with `49`
time points and `48` intervals. The data-sufficiency sweep retrained the
unchanged architecture on `77`, `58`, `39`, `20`, and `10` complete training
cultures. The `58`, `39`, `20`, and `10` settings used three independent
trajectory subsampling seeds each. Validation/test splits remained
trajectory-level.

The matched prospective endpoint is the common `8`-culture physical budget. The
raw earlier campaign in which hybrid used `8` cultures and conventional DBTL
continued to `16` cultures is preserved only as secondary context.

### Demonstrated Computational Result

At the common `8`-culture prospective budget, the full-data hybrid
(`77` training cultures) beat conventional DBTL across the four tested
physiology worlds. Across `12` paired world-seed campaigns, mean best final
product was `1.731517` for the hybrid and `0.812550` for conventional DBTL:
mean paired delta `+0.918967`, bootstrap CI `[+0.682314,+1.231545]`, win rate
`12/12`. World-specific mean deltas were `+0.910448` in `baseline_world`,
`+1.391879` in `strong_oxidative_burden_world`, `+0.682904` in
`atp_limited_world`, and `+0.690637` in
`pathway_bottleneck_damage_world`.

### Data Sufficiency

The exact-verified prospective advantage persisted to the smallest tested
learning set of `10` independent culture trajectories:

| Training cultures | Paired runs | Mean hybrid | Mean conventional | Mean delta | Win rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `18` | `2.494413` | `1.213108` | `+1.281305` | `18/18` |
| `20` | `18` | `2.774576` | `1.213108` | `+1.561468` | `18/18` |
| `39` | `18` | `2.491475` | `1.213108` | `+1.278367` | `18/18` |
| `58` | `18` | `2.488880` | `1.213108` | `+1.275773` | `18/18` |
| `77` | `6` | `2.364272` | `1.213108` | `+1.151164` | `6/6` |

The N-curve is not monotonic, so these values should not be overfit into a
smooth scaling law. The supported conclusion is narrower and stronger: this
synthetic benchmark did not find a practical minimum above `10` cultures. It
does not establish behavior below `10` cultures or prove that real yeast will
have the same sample requirement.

Trajectory-level predictive errors did not show a large memorization gap.
Product normalized RMSE train/validation pairs were `0.140161/0.151164` at
`10` cultures, `0.143868/0.157255` at `20`, `0.139483/0.154559` at `39`,
`0.138383/0.148188` at `58`, and `0.139642/0.152332` at `77`.

### Biological Grounding

The mechanistically strongest part of the generator remains the Yeast9 metabolic
layer: stoichiometric mass balance, exchange bounds, quasi-steady intracellular
metabolism, dynamic FBA-style extracellular updates, and pFBA-like parsimonious
flux selection. The biologically plausible but not quantitatively calibrated
layer includes oxygen, ATP burden, oxidative burden, finite pathway capacity,
bottleneck damage/recovery, strain edits through constraints, and noisy delayed
reporters. The synthetic layer includes exact latent-state dimension, coupling
strengths, thresholds, clipping/Hill forms, reporter filters, noise scales, and
hand-selected pathway-capacity relationships.

The robustness experiment intentionally stressed the high-risk synthetic
physiology assumptions with four coherent worlds rather than random parameter
noise: baseline, stronger oxidative burden, ATP-limited physiology, and
pathway-bottleneck damage. The qualitative hybrid-over-DBTL conclusion survived
all four within-world train/evaluate tests.

### Remaining Real-World Uncertainty

The final computational claim is now bounded as follows: with the same
prospective budget of `8` physical cultures, the hybrid strategy found better
hidden-exact-verified designs than conventional sequential DBTL in the tested
synthetic worlds, and the advantage persisted when the learner had as few as
`10` independent culture trajectories available for training.

The remaining empirical question is whether real yeast provides sufficiently
informative trajectories, at a feasible number of cultures, for the learned
regulatory component to achieve the same prospective DBTL advantage observed in
the synthetic benchmark.

## Current-System Complexity Audit

The current-system complexity audit was executed as a cached-data analysis only.
No new Yeast9/pFBA trajectories, learner retraining, DBTL campaigns, or
prospective simulations were run. The command is:

```bash
MPLCONFIGDIR=/tmp/codex-mpl \
  .venv/bin/python src/yeast_validation/run_current_system_complexity_audit.py \
  --bootstrap 300 \
  --subsample-reps 200
```

Primary outputs:

```text
data/current_system_complexity_audit/current_system_dataset_inventory.csv
data/current_system_complexity_audit/current_system_state_space_inventory.csv
data/current_system_complexity_audit/current_system_complexity_fingerprint.csv
data/current_system_complexity_audit/current_system_complexity_fingerprint_compact.csv
data/current_system_complexity_audit/current_system_complexity_spectra.csv
data/current_system_complexity_audit/current_system_complexity_subsampling.csv
data/current_system_complexity_audit/current_system_complexity_subsampling_summary.csv
data/current_system_complexity_audit/current_system_nonlinear_id_sensitivity.csv
data/current_system_complexity_audit/current_system_metabolic_nominal_vs_empirical.csv
data/current_system_complexity_audit/current_system_intervention_response_coefficients.csv
figures/current_system_complexity_product_observable_spectra.svg
figures/current_system_complexity_hidden_metabolic_spectra.svg
figures/current_system_complexity_dynamic_intervention_spectra.svg
```

The primary dataset is the canonical `125`-culture, `49`-timepoint
`gem_state_space` real-Yeast9 generator dataset. Observable state is biomass,
product, and reporter trajectories. Hidden physiological state is `z_ox`,
`z_atp`, `z_bottle`, `z_er`, and `E_PSY/E_DES/E_CYC`. Metabolic state is the
cached selected Yeast9/pFBA flux and active constraint vectors. Intervention
state is environment plus parsed strain-edit specifications from cached exact
prospective ledgers.

Compact fingerprint:

| Layer | Entropy rank | Participation rank | Nonlinear ID | 95% PCs / dirs | 99% PCs / dirs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product raw | `1.017` | `1.005` | TwoNN `0.523`; LB `0.151` | `1` | `1` |
| Product normalized | `1.018` | `1.005` | TwoNN `0.373`; LB `0.195` | `1` | `1` |
| Observable cell | `6.928` | `3.286` | TwoNN `19.824`; LB `8.697` | `38` | `80` |
| Hidden physiology | `2.789` | `2.406` | TwoNN `0.306`; LB `0.414` | `3` | `4` |
| Metabolic flux state | `2.864` | `2.348` | TwoNN `0.309`; LB `0.162` | `3` | `4` |
| Dynamical observable state | `2.108` | `1.522` | TwoNN `9.225`; LB `6.323` | `3` | `4` |
| Intervention response | `1.923` | `1.823` | TwoNN `2.712`; LB `2.907` | `2` | `2` |

Product trajectories are nearly one-dimensional: raw product PC1 explains
`99.7626%`, amplitude-normalized product PC1 explains `99.7750%`, and slope PC1
explains `95.5271%`. Hidden physiology, hidden-plus-capacity, hidden-plus-flux,
selected fluxes, and active constraints all collapse to roughly `3-4` PCs for
`99%` variance. Yeast9 nominally contains `4131` reactions and `2806`
metabolites, but the empirical visited metabolic state in the cached generator
is low-dimensional.

The observable-cell table is broader: entropy rank `6.928`, participation rank
`3.286`, PC95 `38`, and PC99 `80`. This reflects weak high-dimensional reporter
and noise tail structure; the first two observable PCs already explain
`77.24%` of variance. Culture-level subsampling shows this tail grows with
sample count, while product, hidden, and metabolic dimensions are stable from
`25` cultures upward.

The intervention-response audit used `960` cached exact candidate outcomes from
repaired prospective benchmark ledgers. The global standardized response
Jacobian is effectively two-dimensional: the first two singular directions
explain `99.82%` of response variation, and `2` directions reach both `95%` and
`99%`.

Interpretation: the current synthetic system is empirically low dimensional in
the quantities that matter most for product prediction, hidden physiology,
metabolic state, dynamics, and DBTL intervention response. The `N=10` training
culture result is therefore moderately impressive but not surprising enough to
claim real-yeast scarce-data sufficiency. It is consistent with a low-order
synthetic design landscape.

External-model comparison protocol: future rxncon, WM_S288C, or YEASTRACT-
derived datasets must be downsampled to `125` independent trajectories and `49`
time points; channels must be standardized the same way; constant features must
be removed; PCA/effective-rank, delay-embedding ranks, TwoNN, and Levina-Bickel
IDs must be recomputed identically. A next model should count as substantially
more complex only if at least two biologically central layers exceed the current
system by `>2x` entropy rank and `>1.5x` participation rank while clearing the
current finite-culture uncertainty bound. Nominal node/reaction count alone is
not sufficient.

## External Model Screen: Published Yeast rxncon CDC Model

The first external high-complexity screen was completed with the published
*S. cerevisiae* rxncon cell-cycle model from Muenzner, Klipp, and Krantz,
*Nature Communications* 2019 (`10.1038/s41467-019-08903-w`). Model provenance:
`external_models/rxncon_screening/models/CDC_S_cerevisiae.xls`,
`rxncon/models` commit `793c1407e36c64715c1e278fc03b2a3925fc28db`, rxncon
runtime commit `2204ac365f8ac79afcae06cd4398efe5e6cea04d`.

This was a regulation-only screen. No Yeast9, pFBA, LP solver, ML training, or
DBTL campaign was run. The executable audit is
`src/yeast_validation/run_rxncon_external_complexity_screen.py`. The original compiler hit a
`pyeda` segfault on this Python/macOS stack; the only compatibility change was
to replace Venn-set satisfier enumeration with a pure-Python DNF
partial-assignment enumerator and skip the pyeda validation pass. Boolean rule
construction and perturbation semantics stayed on the published rxncon code
paths.

The perturbable compiled BoolNet contains `3874` synchronous deterministic
Boolean targets: `1095` reaction targets, `2063` state targets, `358` knockout
targets, and `358` overexpression targets. The supported explicit inputs used
were `[Nutrients]`, `[Pheromone]`, `[HU]`, `[LatA]`, and `[Nocodazole]`. The
screen generated `1000` trajectories with `49` timepoints using direct input
programs, single KO/OE perturbations, environment-plus-genetic combinations,
and paired genetic perturbations. Boolean simulation was cheap: `3.84` seconds
for the full panel, about `0.00384` seconds per trajectory, with peak RSS about
`579 MB`.

Matched `125 x 49` complexity was bootstrapped over `10` complete-trajectory
subsets. Results:

| Layer | Entropy rank | Participation rank | 95% PCs / dirs | 99% PCs / dirs |
| --- | ---: | ---: | ---: | ---: |
| Internal regulatory state | `40.06` [`35.92`,`43.86`] | `31.22` [`26.35`,`35.34`] | `46.5` | `51.5` |
| Dynamical internal state | `40.16` [`35.87`,`44.03`] | `31.04` [`26.18`,`35.13`] | `46.5` | `51.5` |
| Observable/phenotypic state | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` |
| Dynamical observable state | `1.073` [`1.061`,`1.089`] | `1.027` [`1.022`,`1.034`] | `1` | `2` |
| Intervention response | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` |

Decision: rxncon passes the predeclared high-complexity gate for internal
regulatory dynamics because both internal state and dynamical internal state
exceed the current system by far more than `2x` entropy rank and `1.5x`
participation rank. It does not yet provide a high-dimensional observable or
intervention-response DBTL landscape: documented phenotype outputs collapse to
rank approximately `1` under this panel. The next experiment should not train a
digital twin yet; first define a biologically documented rxncon objective from
richer module activity/timing, or screen `WM_S288C` if the requirement is high
observable/intervention complexity rather than high hidden regulatory
complexity.

## rxncon Observable-Tier Complexity Follow-Up

The observable follow-up reused the same published rxncon BoolNet, perturbation
panel, and cached trajectories. No regulatory rules, topology, contingencies,
inputs, KO/OE logic, or update semantics were changed. No Yeast9, pFBA, ML, or
DBTL was run. The executable audit is
`src/yeast_validation/run_rxncon_observable_complexity_audit.py`, with outputs in
`data/rxncon_observable_complexity_audit/`.

The previous phenotype collapse is now explained quantitatively. The original
observable vector was the conservative set of explicit macroscopic CDC/global
states, such as `[CD]`, `[ND]`, `[SEP]`, `[SEG]`, DNA replication, spindle,
cytokinesis, morphology, stress, and error flags. In the representative
matched subset, that trajectory representation had `48` varying flattened
features but entropy rank `1.00`, participation rank `1.00`, PC95 `1`, PC99
`1`, and PC1 explained `100%`. Most of those flags are constant or synchronized
under the current perturbation panel.

Three nested observable tiers were tested:

| Tier | Signals | Observable contract |
| --- | ---: | --- |
| A conservative | `22` | explicit macroscopic CDC/global states only |
| B systems module | `30` | Tier A plus eight derived module activity traces from measurable CDC marker states |
| C rich marker | `725` | Tier A plus curated explicit rxncon marker states for cyclins/CDKs, replication, APC/exit, checkpoints, spindle/morphogenesis, and mating/polarity; transcription-delay bookkeeping counters excluded |

Matched `125 x 49` observable complexity over `10` complete-trajectory subsets:

| Tier / representation | Entropy rank | Participation rank | PC95 | PC99 |
| --- | ---: | ---: | ---: | ---: |
| A trajectory | `1.00` [`1.00`,`1.00`] | `1.00` [`1.00`,`1.00`] | `1` | `1` |
| A Hankel lag-8 | `1.079` [`1.064`,`1.091`] | `1.029` [`1.023`,`1.035`] | `1` | `2` |
| B trajectory | `8.06` [`7.56`,`8.46`] | `7.31` [`6.92`,`7.78`] | `8` | `9` |
| B temporal features | `8.73` [`8.35`,`9.30`] | `7.03` [`6.65`,`7.62`] | `9` | `11.5` |
| B Hankel lag-8 | `8.13` [`7.66`,`8.36`] | `7.32` [`6.95`,`7.67`] | `8` | `9` |
| C trajectory | `26.06` [`18.63`,`28.71`] | `20.63` [`13.40`,`23.71`] | `28.5` | `32.5` |
| C temporal features | `25.83` [`18.47`,`28.77`] | `20.50` [`13.30`,`23.61`] | `28.5` | `32.5` |
| C Hankel lag-8 | `26.40` [`18.94`,`29.34`] | `20.70` [`13.46`,`23.84`] | `28.5` | `33.5` |

Intervention-response complexity also becomes multidimensional when the
response vector uses accepted observables:

| Tier / response | Entropy rank | Participation rank | PC95 dirs | PC99 dirs |
| --- | ---: | ---: | ---: | ---: |
| A temporal/combined | `1.00` | `1.00` | `1` | `1` |
| B temporal features | `9.06` [`8.46`,`9.68`] | `7.45` [`6.80`,`8.06`] | `10` | `11.5` |
| C temporal features | `26.50` [`18.11`,`28.87`] | `20.93` [`12.41`,`23.55`] | `30.5` | `34.5` |

Trajectory-family diagnostics: Tier A had only `2` observable endpoint and
trajectory patterns; Tier B had `145` endpoints and `161` trajectory/temporal
patterns; Tier C had `242` endpoints and `244` trajectories. Internal state had
`339` unique final/trajectory patterns for context. Endpoint and trajectory
distances remained almost perfectly correlated, so this is not primarily a
"same endpoint, different path" ambiguity. The useful difficulty comes from a
multidimensional measured regulatory phenotype and intervention-response
surface.

Decision: rxncon now **passes** as a high-dimensional learning/design problem
if the observable contract includes at least the Tier B systems-biology marker
panel. Tier B is an acceptable practical benchmark (`D_observable` about `8-9`,
`D_intervention` about `9`); Tier C is a strong rich-observation benchmark
(`D_observable` about `26`, `D_intervention` about `26`). It still fails under
the original conservative macroscopic phenotype-only contract.

Recommended smallest future benchmark, not yet executed: `N_train = 10, 25, 50`
complete trajectories, three learner seeds, one held-out-combination split, one
harder perturbational extrapolation split, `4-6` exact verification cultures
per method, and few campaign seeds. Compare hybrid/state-space, direct
black-box, conventional sequential DBTL, and random/space-filling selection.
Use Tier B as the default observation contract and reserve Tier C as a
rich-observation sensitivity/upper-observability condition.

## rxncon -> Yeast9 Generator Gate: v1 Interface and Stop Decision

The first rxncon-to-GSM generator gate is implemented in
`src/yeast_validation/run_rxncon_gsm_generator_gate.py`. It reused the published rxncon
yeast CDC model and the existing Yeast9 beta-carotene/pFBA backend. It did not
train a model, run DBTL, launch prospective selection, or create the canonical
historical dataset.

Inputs were deliberately restricted to learner-facing environment variables:
`[Nutrients]`, `[Pheromone]`, `[HU]`, `[LatA]`, `[Nocodazole]`, temperature, pH,
DO, and glucose uptake. The hidden lockbox contains raw rxncon nodes, module
truth, GSM interface controls, true bounds, fluxes, event-trigger decisions,
future phenotypes, and generator labels. The observable preview contains only
environment/time/reporters.

The frozen v1 interface has ten GSM controls: carbon uptake, oxygen uptake, ATP
maintenance, growth allocation, stress/product-loss load, precursor
availability, resource/translation capacity, and heterologous PSY/DES/CYC
capacities. A direct GPR overlap inventory found `127` rxncon/Yeast9 label-match
rows, but the active v1 mapping is still dominated by higher-level physiological
channels rather than direct GPR capacity rules.

Six reporters were generated as mixed delayed nonlinear projections:
`R_stress`, `R_resource`, `R_checkpoint`, `R_pathway_capacity`, `R_morphology`,
and `R_energy`. Each reporter depends on at least four contributors. The maximum
single reporter/raw-node correlation was `0.678`; the maximum reporter/reporter
correlation was `0.633`; mean hidden-PC reconstruction `R^2` from reporters was
`0.262`. These pass the non-direct-exposure and informative-but-noninvertible
checks.

Layer complexity on the `60`-culture, `49`-timepoint environment-only pilot:

| Layer | Entropy rank | Participation rank | PC95 | PC99 |
| --- | ---: | ---: | ---: | ---: |
| rxncon state | `5.82` | `4.57` | `6` | `12` |
| regulatory modules | `4.91` | `4.07` | `5` | `10` |
| GSM interface | `5.36` | `4.27` | `6` | `8` |
| biosensors, noiseless | `3.31` | `2.50` | `4` | `6` |
| biosensors, noisy | `16.38` | `6.32` | `42` | `55` |

The environment-to-interface intervention response had entropy rank `5.64` and
participation rank `5.04`; environment-to-biosensor response had entropy rank
`5.07` and participation rank `3.61`. This is above the old two-dimensional
intervention-response landscape, but still far below the high-dimensional KO/OE
rxncon screen.

A three-culture dense-vs-event-triggered Yeast9 smoke test completed. Dense
reference replay used `108` LP solves per culture; sparse replay used `39`,
reducing LP solves by `63.9%`. Mean dense-vs-sparse errors were biomass RMSE
`5.6e-5`, product RMSE `4.1e-6`, and final-product absolute difference
`5.9e-6`. This supports sparse replay as a compute strategy, subject to a larger
validation subset before canonical dataset generation.

Decision: stop before ML and before canonical dataset generation. The v1
environment-only generator is more complex than the old hand-designed hidden
physiology at the interface layer, but it does not yet transmit the published
rxncon model's high-dimensional internal dynamics into observable culture
behavior. Noisy reporters create a large effective-rank tail
(`16.38` versus `3.31` noiseless entropy rank), so reporter noise must not be
counted as biological complexity. The next action should be a generator-design
revision, not training.

## rxncon Process-Variability Learning Robustness

The hidden process-variability experiment preserved the v1 published rxncon ->
GSM -> Yeast9 generator and added only upstream hidden process variables. The
deployable input contract was nominal `T_set`, `pH_set`, and `DO_set` only.
Realized process variables, systematic group IDs, rxncon state, regulatory
modules, GSM controls, bounds, and fluxes remained lockbox quantities. The
workflow is `src/yeast_validation/run_rxncon_process_variability_experiment.py`; outputs are
under `data/rxncon_process_variability/`.

Worlds and size:

| World | Cultures | Timepoints | Hidden process variation |
| --- | ---: | ---: | --- |
| `perfect_control` | `125` | `49` | none |
| `random_process` | `125` | `49` | culture-level T/pH/DO/kLa/carbon/nitrogen deviations |
| `random_systematic_process` | `125` | `49` | random deviations plus batch/plate/reader/position biases |

Expanded dense-vs-sparse validation used `12` full cultures. Dense replay used
`144` LP solves per culture and event-triggered replay used `42`, a `70.8%`
reduction. Median final-product difference was `1.22e-5`, maximum
`3.22e-5`. The canonical sparse generation used `15,750` exact Yeast9 LP solves.

The key scientific finding is that the strict `T_set,pH_set,DO_set` contract did
not perturb any supported rxncon external input. Nutrient sufficiency remained
on, and pheromone/HU/LatA/nocodazole remained off. Therefore raw rxncon state
and regulatory modules had zero varying features in all three worlds. Hidden
process variation entered Yeast9/GSM through realized T, pH, DO, carbon/feed,
nitrogen, and kLa, not through arbitrary hidden-node wiring.

Complexity summary:

| World | D_regulatory | D_interface | D_metabolic | D_observable bio | D_product |
| --- | ---: | ---: | ---: | ---: | ---: |
| perfect | `0` | `1.00` | `1.98` | `1.06` | `1.01` |
| random | `0` | `2.00` | `1.99` | `1.07` | `1.01` |
| random + systematic | `0` | `1.99` | `1.97` | `1.09` | `1.01` |

Noisy observable rank was about `37`, but this is a measurement-noise tail and
is not counted as biological complexity. Product trajectories remained nearly
rank-1.

Replicate variability increased as intended. Product within-nominal replicate
RMSE was effectively `0` in perfect control, `4.91e-4` under random process
variation, and `7.58e-4` under random plus systematic variation. Product
within/between variation ratios were `0`, `0.159`, and `0.226`.

Product NRMSE averaged over non-training splits:

| World | State-space + reporters | State-space no reporters | Direct polynomial | Nearest-env | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| perfect | `0.816` | `0.345` | `0.605` | `0.201` | `0.605` |
| random | `0.769` | `0.306` | `0.601` | `0.408` | `0.599` |
| random + systematic | `0.758` | `0.354` | `0.539` | `0.214` | `0.519` |

Reporter supervision was not beneficial for product prediction in this run:
real reporters, shuffled reporters, and random smooth auxiliary labels all
underperformed the no-reporter state-space model. The no-reporter state-space
model beat the polynomial direct model, but the nearest-environment baseline was
strongest in two of three worlds, matching the low-dimensional product
landscape. The full-information oracle only modestly improved over the direct
polynomial model.

Decision: this is a useful negative/constraint-setting experiment. A compact
model can learn the low-order product/biomass response under modest hidden
process variability, and the sparse Yeast9 strategy is accurate, but this
specific `T_set,pH_set,DO_set` process-variability contract does not test
learning over an active rxncon regulatory manifold. Future rxncon-coupled tests
must include supported external regulatory inputs as part of the nominal
experimental contract if the goal is to stress published rxncon dynamics rather
than only Yeast9 process variability.


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

The hidden-history experiment stopped at the cheap generator gate. Fifty
generator-only histories were screened across baseline, nutrient withdrawal,
pheromone, HU, LatA, Nocodazole, and sequential-pulse families. Selection used
only transfer-state, module, and frozen-interface diversity and persistence.
History IDs and all hidden truth remain in
`data/rxncon_hidden_history_live_biosensor/` lockbox files; they were not
learner inputs.

The screen found no usable persistent interface memory. Transfer rxncon state
entropy rank was 4.72, but after all histories were placed under the same
post-transfer external inputs, production rxncon, module, and interface
entropy ranks were each approximately 1.0. The HU pulse retained `[dNTP]`
only; because `[dNTP]` is absent from the existing interface, it cannot affect
Yeast9 through this frozen generator. Extending pulses to 200 steps did not
create additional downstream memory. The history/interface gate consequently
failed, so the prescribed 20--50 culture exact-Yeast9 pilot and all subsequent
assimilation analyses were correctly not run. The next scientifically distinct
step would require a predeclared, biologically justified persistent-state
mechanism or a different published model; it must not be introduced by tuning
against biosensor or product outcomes.
### Time-Varying DO Intervention Validation

This gated stage reused the active rxncon -> frozen GSM interface -> exact
Yeast9/pFBA path. The Phase A schedule manifest contains static DO 20/40/80,
three switch positions in each direction, and matched-exposure low/high pulse
timings, repeated over three existing continuous backgrounds: 45 cultures and
49 timepoints. The exact sparse ledger records 2,286 LP solves. Dense/sparse
validation passed with a 6.42e-6 median final-product difference and 70.8%
median LP reduction. The timing gate passed because switch timing effects were
reproducible and reached 0.783 times the median static spread; matched pulse
timing alone was smaller (about 0.068 of static spread at the median).

Phase B then compared an observable-only memoryless coordinate polynomial, a
causal recurrent model, and a dynamic state-space learner with current DO in
its transition. Splits were by complete culture/schedule: 21 train, 12
validation, and 12 late-timing test cultures. Validation selected the dynamic
state-space model, which achieved test trajectory NRMSE 0.477 versus 0.614 for
the coordinate baseline. The learner saw no rxncon nodes, regulatory modules,
GSM controls, fluxes, constraints, or biosensors. No prospective DBTL run was
started. Reproduction files are under
`data/rxncon_dynamic_intervention_validation/`; the learner implementation is
`src/yeast_validation/run_rxncon_dynamic_intervention_learner.py`.
