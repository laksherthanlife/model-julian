# Data Generator Specification and Validity Argument

**Yeast Digital Twin — Synthetic Architecture Validation**

Status: reference document. Scope: every data generator used anywhere in this
repository, what it assumes, why those assumptions are defensible, and what
would falsify them.

---

## 0. Why this document exists

The acceleration claim in this project has the form:

> A frozen pretrained hybrid digital twin reaches a better strain-environment
> design, at equal or lower physical-culture cost, than a conventional adaptive
> DBTL workflow, when both are run against the same hidden ground-truth
> organism.

That claim is only as strong as the hidden ground-truth organism. If the
generator is trivially learnable, the result is a statement about a toy. If the
generator is secretly aligned with the twin's inductive bias, the result is
circular. If the generator's assumptions are undeclared, the result is
unfalsifiable.

This document therefore does three things, per generator:

1. **Specifies** the generator exactly — every equation, constant, and solve.
2. **Classifies** every assumption into one of four evidence classes.
3. **States** what observation would show the assumption to be wrong, and
   whether the repository already contains a test for it.

### Assumption evidence classes

| Class | Meaning | Standard of proof required |
| --- | --- | --- |
| **M — Mechanistic** | Inherited from an externally validated artefact (the community consensus GEM, mass balance, LP optimality). Not invented here. | Provenance + checksum + version |
| **L — Literature-shaped** | Functional form and order of magnitude taken from published yeast physiology; exact constants tuned here. | Citation + stated deviation |
| **D — Declared design** | Chosen by this project to create a specific difficulty. Not a biological claim. Must be declared, frozen, and hidden from the learner. | Freeze manifest + leakage audit |
| **A — Arbitrary** | No external justification; chosen for numerical convenience. Must be shown not to drive the headline result. | Sensitivity or ablation evidence |

A generator is "reasonable" for this project's purpose if (a) its M-layer is a
real, checksummed, community model; (b) its L-layer is order-of-magnitude
defensible; (c) its D-layer is declared and provably hidden from the learner;
and (d) its A-layer is demonstrably not load-bearing. Sections 6–8 argue each
of these.

---

## 1. Generator inventory

Five distinct generators exist. Only **G3** underwrites the acceleration claim.

| ID | Generator | Script(s) | Backend | Stochastic? | Experiments served |
| --- | --- | --- | --- | --- | --- |
| **G0** | Minimal single-latent ODE positive control | `run_minimal_validation.py`, `run_concrete_experiment_chain.py` | closed form | yes (obs. noise) | Goals 1–4 scaffolding, concrete chain |
| **G1** | Analytic fixed-environment product ODE | `run_fixed_environment_validation.py` | closed form | yes (obs. noise) | 1A, 1B, 1C, 2A, 2B, 2C |
| **G2** | Reduced-stoichiometry dFBA state machine | `run_dfba_state_machine_validation.py` | declared reduced surrogate | yes (obs. noise) | 3A, 3B, 3C, 3D, 3E, 3F |
| **G2b** | Physiological-regime ODE testbed | `ode_regime_core.py`, `generate_ode_regime_dataset.py` | RK4 closed form | deterministic | ODE regime discovery |
| **G3** | **Exact Yeast9 dynamic staged-pFBA** | `gem_backend.py`, `design_benchmark_exact.py` | real GEM + GLPK LP | **deterministic** | GEM pilot, dynamic capacity, GEM state-space, deployment benchmark, final DBTL freeze, harder-shift, **prospective DBTL benchmark** |

**Reading rule.** G0–G2b are methods scaffolding and negative-result
infrastructure. They are documented here for completeness and because the
negative results they produced are what forced the project onto G3. Do not cite
G0–G2 as evidence for the acceleration claim. Cite G3.

---

## 2. Generator G3 — Exact Yeast9 dynamic staged-pFBA

This is the generator behind the big validation experiments. It is the "hidden
wet lab."

### 2.1 Deployment contract

```
(strain edit vector d, fixed environment e = [T, pH, DO])
    -> complete beta-carotene trajectory B(t), t = 0 .. 12 h
```

One culture receives one fixed environment for its whole trajectory. Time is
the output coordinate, never an input the learner gets to steer.

### 2.2 Layer M — the stoichiometric core

| Property | Value |
| --- | --- |
| Asset | `yeast-GEM.xml` |
| SBML model id | `yeastGEM_v9__46__0__46__2` (yeast-GEM v9.0.2) |
| SHA-256 | `9fd2c572cace73c2ea835205617313554d1bf89f4ef077f49defbdfa219a4ad7` |
| Reactions / metabolites / genes | 4131 / 2806 / 1161 |
| Compartments | 14 (`c, ce, e, er, erm, g, gm, lp, m, mm, n, p, v, vm`) |
| Exchange reactions | 274 |
| Objective | `r_2111` (growth) |
| Glucose exchange | `r_1714` |
| Oxygen exchange | `r_1992` |
| ATP maintenance | `r_4046` |
| Solver | GLPK via optlang 1.9.1, cobrapy 0.31.1, libSBML 5.21.1 |
| Baseline (unmodified) max growth | 0.0858440 h⁻¹ |
| Blocked reactions | 1004 |
| Mass-balance warnings in asset | 200 |

Published Yeast9 v9.0.0 reports 4130 reactions / 2805 metabolites / 1162 genes;
the v9.0.2 patch release used here differs by +1 reaction, +1 metabolite,
−1 gene. This is a patch-level difference in the community model, not a local
modification, and the checksum in `data/yeast_gem_audit.csv` pins it exactly.

**Why this matters.** Everything stoichiometric in the generator — carbon
balance, cofactor coupling, respiratory chain, the entire competing metabolic
network — is not authored by this project. It is inherited from a
community-curated consensus model whose reactions are 93.8% mass- and
charge-balanced upstream. The generator cannot be accused of having invented a
metabolism that happens to suit its own twin.

### 2.3 Layer M/D — heterologous pathway installation

Four reactions are added to the consensus model (`install_beta_carotene_pathway`):

| Reaction | Stoichiometry | Class |
| --- | --- | --- |
| `BETA_PHYTOENE_SYNTHASE` | 2 GGPP (`s_1311`) → phytoene + 2 PPi (`s_0633`) | L (lumped crtB/crtYB chemistry) |
| `BETA_PHYTOENE_DESATURASE` | phytoene → lycopene | L (lumped crtI, 4 desaturation steps) |
| `BETA_LYCOPENE_CYCLASE` | lycopene → beta-carotene | L (lumped crtYB cyclase activity) |
| `DM_beta_carotene_c` | beta-carotene → ∅ | D (accumulation sink outside biomass objective) |

Native GGPP formation (`r_0461`) is **reused, not duplicated** — the pathway
drains the model's own isoprenoid pool, so precursor competition with native
sinks is a real LP-level competition rather than a free source. Reaction-ID
collisions raise rather than silently overwrite.

Literature correspondence: the canonical yeast beta-carotene route expresses
`crtE` (GGPP synthase), `crtYB` (bifunctional phytoene synthase / lycopene
cyclase) and `crtI` (phytoene desaturase) from *Xanthophyllomyces dendrorhous*.
Reported performance spans 4 mg/gDCW (single copy) to 32 mg/gDCW (copy-number
plus truncated HMG1) and 477.9 mg/L with hydrophobic-substrate extraction. The
generator's three-step lump is a coarse-graining of that route, not a different
route.

**Declared deviations (honest list):**

- The desaturase is one lumped reaction, not four sequential desaturations, so
  no partially desaturated intermediates exist.
- The lumped steps are marked `not_formula_balanced_lumped_step` in
  `data/beta_carotene_gem_pathway_manifest.csv`. They are audited to not create
  ATP, reducing equivalents, carbon, or precursor-producing cycles — i.e. they
  cannot be exploited by the LP as a free-energy loophole.
- `DM_beta_carotene_c` is an intentional non-mass-balanced sink representing
  intracellular accumulation. Product is a *state*, not an excreted metabolite.

### 2.4 Layer D — per-interval constraint map

Each of the 48 intervals rebuilds the model and applies (`apply_interval_constraints`):

```
glucose lower bound      = -q_glc                          (default 10 mmol/gDW/h)
oxygen capacity          = max(0.25, q_O2,100 * DO/100)    (default q_O2,100 = 12)
                           * max(0.20, 1 - 0.45 z_ox)
ATPM lower bound         = m_ATP * f_env(T,pH) * (1 + 1.35 z_atp)
                           * 1.25 if state in {energy_limited, combined_overload}
f_env(T,pH)              = (1 + 0.018 |T - 30|) * (1 + 0.12 |pH - 5|)
pathway_ub               = 1000 * cap(state) * max(0.05, 1 - 0.62 z_bottle - 0.18 z_ox)
                           * 0.55 if state in {pathway_limited, combined_overload}
PSY/DES/CYC upper bounds = min(pathway_ub, V_j,max * E_j)
```

with `V_PSY,max = 0.34`, `V_DES,max = 0.28`, `V_CYC,max = 0.23` and
`E_j ∈ [0.08, 1]` the dynamic enzyme capacities of §2.6.

**Literature anchors.** `m_ATP = 0.70 mmol ATP gDW⁻¹ h⁻¹` sits within 11% of the
measured aerobic maintenance requirement of 0.63 ± 0.04 mmol ATP gDW⁻¹ h⁻¹ for
*S. cerevisiae* at near-zero growth rate. `q_glc = 10 mmol gDW⁻¹ h⁻¹` and
`q_O2,100 = 12 mmol gDW⁻¹ h⁻¹` are within the range used for aerobic
glucose-sufficient yeast batch culture. `T_opt = 30 °C` and `pH_opt = 5.0` are
the standard laboratory optima for *S. cerevisiae*.

**Declared deviations.** The temperature and pH terms enter as a *maintenance
multiplier* (`f_env`), not as an Arrhenius growth law or an enzyme-kinetic
temperature model. The DO→oxygen-uptake map is linear in DO. Both are
phenomenological stand-ins for effects the GEM has no representation of.

### 2.5 Layer M — the staged LP protocol

Per interval, `solve_staged` runs three LPs in lexicographic order
(Protocol A, the default):

1. **Growth**: maximise `r_2111` → `mu_max`. Then pin
   `lb(r_2111) = gamma(state) * mu_max`.
2. **Product**: maximise `DM_beta_carotene_c` subject to the preserved growth
   floor → `v_beta`. Then pin
   `lb(DM_beta_carotene_c) = 0.999 * v_beta`.
3. **pFBA**: minimise total flux subject to both floors.

Stage 3 is standard parsimonious FBA, present specifically to remove the
alternate-optima degeneracy that would otherwise make the "true" flux
distribution ill-defined and the reported flux features arbitrary. The
outer loop is the classical static-optimisation-approach (SOA) discretisation of
dynamic FBA: solve an LP on each interval, integrate the extracellular /
state variables forward with Euler, repeat.

**Solve accounting.** 48 intervals × 3 LPs = **144 LP solves per culture**,
recorded per row as `n_growth_optimizations`, `n_product_optimizations`,
`n_pfba_optimizations`, `n_actual_lp_solves`, `n_surrogate_evaluations` (always
0 for G3), `n_infeasible_solves`, `n_unbounded_solves`. This is not decoration:
it is the mechanism by which "the generator actually ran the GEM" is auditable
rather than asserted.

**Protocol sensitivity is measured, not assumed.**
`data/gem_allocation_protocol_comparison.csv` runs A, B and C on the same 27
environments:

| Protocol | mean growth flux | mean beta flux | final product | normalized shape PC1 |
| --- | ---: | ---: | ---: | ---: |
| A (lexicographic product) | 0.2527 | 0.19198 | 0.017096 | 0.99136 |
| B (fixed heterologous capacity fraction) | 0.3425 | 0.05860 | 0.005402 | 0.99999 |
| C (weighted, state-dependent target) | 0.2527 | 0.19198 | 0.017096 | 0.99136 |

The choice of allocation protocol changes the growth/product split by ~3×, so it
is a real modelling decision and is declared as such (class **D**, field
`allocation_protocol`). It does *not* change the qualitative conclusion that a
single shape component dominates.

### 2.6 Layer L/D — the outer dynamics (this is the hard part of the task)

Between LP solves the generator integrates four burden states, three enzyme
capacities, biomass and titer.

**Burdens** (`burden_terms`, `dt = 0.25 h`, all clipped to `[0, 1.5]`):

```
dz_ox/dt     = 0.16 * (v_O2 / O2_cap) + 0.22 * v_beta        - 0.36 * rho * z_ox
dz_atp/dt    = 0.14 * v_ATPM + 0.11 * mu + 0.12 * v_beta
                 - 0.05 * (v_O2 / O2_cap)                     - 0.38 * rho * z_atp
dz_bottle/dt = 0.85 * v_GGPP + 2.40 * C_phy + 3.00 * C_lyc
                 - 1.45 * v_beta                              - 0.28 * rho * z_bottle
dz_er/dt     = 0.08 * |T - 30| / 8 + 0.07 * |pH - 5|          - 0.30 * z_er
```

`rho = 2.8` while the culture is in any stressed or recovering state, else 1.0.
Congestion terms are

```
C_phy = max(0, v_PSY - v_DES) + max(0, ub_PSY - ub_DES) * sat(PSY)
C_lyc = max(0, v_DES - v_CYC) + max(0, ub_DES - ub_CYC) * sat(DES)
```

i.e. burden accrues when an upstream carotenoid step outruns the step below it —
either in realised flux or in available capacity.

**Enzyme capacities** (`capacity_update_terms`):

```
dE_j/dt = s_j * env_j * m_j(state) * (1 - E_j)  -  D_j * E_j
D_PSY   = d_PSY + a_ox,PSY z_ox + a_atp,PSY z_atp + a_bot,PSY z_bottle + k_phy C_phy
D_DES   = d_DES + a_ox,DES z_ox + a_atp,DES z_atp + a_bot,DES z_bottle
            + 0.35 k_phy C_phy + k_lyc C_lyc
D_CYC   = d_CYC + a_ox,CYC z_ox + a_atp,CYC z_atp + a_bot,CYC z_bottle
            + 0.40 k_lyc C_lyc
```

Selected (baseline world) values: `s = (0.070, 0.060, 0.055)`,
`d = (0.018, 0.024, 0.022)`, oxidative sensitivities `(0.040, 0.125, 0.160)`,
congestion sensitivities `k_phy = 0.320`, `k_lyc = 0.280`, `E_min = 0.08`.
`env_j` are Gaussian temperature/pH windows per enzyme with mild DO modulation.

**Hidden regulator** (`next_state`) — a 7-state machine with hysteresis:

```
states: balanced, productive, oxidative_stress, energy_limited,
        pathway_limited, combined_overload, recovering
high:  z_ox > 0.62,  z_atp > 0.58,  z_bottle > 0.60
low:   z_ox < 0.42,  z_atp < 0.38,  z_bottle < 0.40
minimum dwell: 2 intervals (0.5 h) before leaving a stressed state
>=2 high -> combined_overload; else first high wins; leaving stress -> recovering;
all low -> productive; otherwise balanced
```

The state then feeds back into the next interval's LP through `gamma(state)`
(growth fraction preserved, 0.43–0.72), `cap(state)` (pathway cap, 0.30–1.10),
and the ATPM multiplier. **This closed loop — flux → burden → discrete state →
constraints → flux — is the thing the digital twin has to learn.** It is why the
task is not a static FBA lookup.

**Product and biomass:**

```
dX/dt = mu * X
dB/dt = v_beta * X - k_deg * (1 + 2 z_ox) * B
```

**Biological motivation for the burden couplings.** Carotenoid overproduction in
yeast is documented to be self-limiting through intermediate accumulation:
increasing `crtI` dosage drives lycopene accumulation, and "elevated lycopene
concentrations impair normal yeast growth," forcing selective amplification of
`crtE`/`crtYB` only. That is exactly the `C_lyc → z_bottle → capacity damage`
loop, and it is the reason the generator penalises *imbalance between steps*
rather than total pathway flux. Beta-carotene is also chemically susceptible to
autoxidation, motivating the oxygen- and burden-dependent degradation term.

**Counter-evidence, stated deliberately.** The same study that reports
lycopene-driven growth impairment also reports that increased *beta-carotene*
accumulation "did not show significant effects on cell growth." The generator's
direct `0.22 * v_beta` term in `dz_ox` is therefore stronger than at least one
published observation supports. It is classified **L (weak)** and is one of the
three parameters flagged in §7 as requiring a sensitivity ablation before the
mechanism is claimed as biology rather than as declared difficulty.

### 2.7 Layer D — design space and edit semantics

Ten edit multipliers (`stage4_design.EDIT_COLUMNS`) are mapped onto config
changes by `cfg_for_candidate`:

| Edit | Acts on |
| --- | --- |
| `PSY/DES/CYC_capacity_multiplier` | `V_j,max` (× sink gain × √precursor) and `E_j(0)` (×√mult) |
| `precursor_supply_multiplier` | native GGPP reaction `r_0461` bounds (× mult), plus √ on pathway caps |
| `competing_sink_multiplier` | sink gain `1 + 0.22 max(0, 1 - m)` on all three caps |
| `ATP_support_multiplier` | `m_ATP / m`, and ATP sensitivities `/ m` |
| `oxygen_support_multiplier` | `q_O2,100 × m`, oxidative sensitivities `/ m` |
| `glucose_uptake_multiplier` | `q_glc × m` |
| `export_capacity_multiplier` | `k_deg / m` |
| `product_degradation_multiplier` | `k_deg × m` |

Environment domain: `T ∈ [27, 33] °C`, `pH ∈ [4.5, 5.5]`, `DO ∈ [20, 80] %`.

Critically, `apply_precursor_edit` modifies **the native GEM reaction's bounds**,
not a synthetic bypass — so a precursor edit still has to propagate through the
real stoichiometric network and the LP, and can be compensated or defeated by it.

### 2.8 Layer D — the hidden worlds

`regime_configs()` freezes seven biologically-labelled parameter worlds plus two
legacy ones. Each carries a `cfg`, per-burden `burden_scale`, a global
`repair_scale`, a `pathway_damage_scale`, a declared `degradation_rule` and a
written `biological_interpretation`.

| World | Mechanism it encodes | Key scaling | Campaigns run |
| --- | --- | --- | ---: |
| `baseline_world` | reference organism | all 1.00 | **10** |
| `strong_oxidative_burden_world` | high DO supports respiration early, damages later | ox 1.80, repair 0.76, damage 1.28 | **10** |
| `atp_limited_world` | maintenance competes with production for ATP | atp 1.70, repair 0.82 | 0 |
| `precursor_competition_world` | native isoprenoid sinks outcompete the pathway | bottleneck 1.72 | 0 |
| `pathway_bottleneck_damage_world` | PSY/DES/CYC decay differently; limiting step *moves* | damage 1.72, repair 0.70 | 0 |
| `compensatory_metabolism_world` | alternative capacity buffers obvious single edits | atp 1.22, bottleneck 1.25 | not in `PROSPECTIVE_WORLDS` |
| `mixed_multi_mechanism_world` | all of the above shift together | ox 1.45, atp 1.48, bottleneck 1.55 | 0 |

**Scope warning.** `PROSPECTIVE_WORLDS` declares six worlds, but
`data/prospective_campaign_manifest.csv` shows the completed 20-campaign run used
**two**: `baseline_world` (10 campaigns) and `strong_oxidative_burden_world`
(10 campaigns). Four declared worlds have zero campaigns. The generalisation
claim currently rests on one source world and one shifted world. Either run the
remaining four or scope the claim to "one oxidative shift" in the text.

**Update (Latent Capacity Mismatch pilot).** No new dynamics code is needed to
address this — `atp_limited_world` is already fully specified above with zero
campaigns run, and is a *non-oxidative* shift (maintenance/ATP, not
oxygen/redox), satisfying the "at least one qualitatively different world"
requirement for the new benchmark. The Stage D pilot activates it alongside
`baseline_world` and `strong_oxidative_burden_world`; see
`LATENT_CAPACITY_MISMATCH_DESIGN.md`. This does not retroactively address the
scope warning for the *existing* headline 20-campaign result above, which still
rests on two worlds.

**Argument for this design.** These are not adversarial perturbations chosen to
break the twin. Each is a named biological failure mode with a stated
interpretation, frozen before campaigns run
(`data/design_benchmark_regime_manifest.csv`), and each is applied identically to
*both* the conventional and the hybrid workflow. The `compensatory_metabolism_world`
in particular is designed to *disadvantage* a naive static-GEM ranking, which is
the twin's nearest competitor — i.e. the world set is chosen to make the
comparison harder, not easier.

### 2.9 Layer L — extracellular glucose mass balance (opt-in, addresses G3-18)

As of the Latent Capacity Mismatch Benchmark hardening pass, G3 carries an
**opt-in** extracellular glucose pool, `GEMCultureConfig.substrate_limited`
(default `False`, so every existing headline artifact reproduces bit-for-bit)
and `GEMCultureConfig.s_glc0` (default `1000.0`, arbitrary internal units
consistent with the existing `X`/`glucose_uptake` unit system — not claimed
molar).

```
S_glc(0) = s_glc0
glucose_lb(t) = -glucose_uptake                          if substrate_limited = False   (unchanged G3 behaviour)
glucose_lb(t) = -min(glucose_uptake, S_glc(t) / (dt * X(t)))   if substrate_limited = True
S_glc(t+dt) = max(0, S_glc(t) - dt * v_glc_uptake(t) * X(t))
```

where `v_glc_uptake(t) = max(0, -v_glucose_exchange(t))` is the *realised*
post-LP glucose exchange flux (not the bound). `S_glc` is tracked and reported
in every trajectory table (`traj_rows["S_glc"]`) regardless of the flag, so the
depleting-vs-non-depleting comparison is directly auditable; only the
constraint-limiting behaviour is gated by `substrate_limited`.

This does not turn G3 into a full bioreactor model (no feed, no maintenance
death term, no diauxic shift) — it is the minimal addition needed to let a
culture leave the unrestricted-exponential regime, per G3-18's own
recommendation (§10). Implemented identically in `gem_backend.run_dynamic_culture`
and `design_benchmark_exact.exact_dynamic_rollout` (the two duplicated outer
loops). See `tests/test_gem_substrate_balance.py` for the regression guard
(default-off behaviour unchanged) and the depletion/monotonicity checks.

### 2.10 Layer L — observation noise (addresses G3-20)

The internal deterministic trajectory (`z_*`, fluxes, constraint bounds,
`B_total`/`X`/`S_glc` as computed by the LP) remains exactly as before — this is
still the audited ground truth. A new, separate, opt-in post-hoc module,
`src/yeast_validation/gem_observation_noise.py`, adds seeded Gaussian measurement noise
(`sigma_product=0.03`, `sigma_biomass=0.02`, `sigma_glc=0.03`, fractions of each
trajectory's own dynamic range) plus one multiplicative per-culture
biological-variability draw (`sigma=0.015`), producing `{col}_observed` columns
(`B_total_true`/`B_total_observed`, etc.) alongside the untouched deterministic
columns. Reproducible given a seed; never applied to `z_*`/internal state.

Note this complements, rather than introduces, noise in G3: the reporter layer
(`R_ox`, `R_atp`, `R_E_PSY`, `R_E_DES`, `R_E_CYC` in
`generate_gem_state_space_dataset.build_reporters`) already had seeded
lag+noise before this pass — G3-20's "deterministic, no measurement noise"
claim held for the primary product/biomass trajectory but not for reporters.
This section's addition closes the gap on product/biomass/glucose.

---

## 3. Generator G1 — analytic fixed-environment product ODE (Exp 1A–2C)

**Contract:** `e = [I, O, S] -> P(t)`, 61 points, `dt = 0.20`, `k_P = 0.18`.

```
dP/dt = F(t, e) - k_P P,   P(0) = 0
```

| Exp | Rate law | Purpose |
| --- | --- | --- |
| 1A | `F = softplus(-0.45 + 1.45I + 0.85O - 1.25S) + 0.04` | constant rate; can environment set magnitude? |
| 1B | `F = V (1 - e^{-t/tau})`, `tau = 0.35 + softplus(-0.35 + 1.35S - 0.90I)` | magnitude + startup |
| 1C | `F = V (1 - e^{-t/tau_on}) e^{-t/tau_off}`, `V` includes `+1.25 IO - 1.05 IS` | nonlinear interaction, rise and decline |

Experiment 2 adds two hidden latent states with environment-set setpoints and
first-order kinetics:

```
s_A = sigmoid(-0.75 + 0.95I - 0.35O + 1.65S + 1.05 IS)
s_B = sigmoid(-0.55 - 0.45I + 0.80O + 1.25S - 0.85 IO + 0.75 OS)
dz_A/dt = 1.05 s_A (1 - z_A) - 0.82 (1 - s_A) z_A
dz_B/dt = 0.58 s_B (1 - z_B) - 0.24 (1 - s_B) z_B
F = F_base(e,t) * clip(1 - 0.42 z_A - 0.35 z_B - 0.38 z_A z_B, 0.12, 1.15)
```

Reporters are lagged first-order filters of the latents plus noise:
`dR_A/dt = 0.90 (z_A - R_A)`, `dR_B/dt = 0.55 (z_B - R_B)`, and a slower mean
reporter at 0.70. Observation noise: product `sigma = 0.020`, reporter
`sigma = 0.018`.

**Split construction.** 5×5×5 grid on `{0.10, 0.28, 0.46, 0.64, 0.80}` plus 72
Latin-hypercube training points, with the corner `I > 0.65 AND S > 0.65`
excised from training and reserved as `heldout_combination`; extrapolation
points are pushed outside `[0.10, 0.80]` on one axis each. Counts:
`train 192 / validation 24 / interpolation 30 / heldout_combination 30 /
extrapolation 36`. Dataset seeds `(101, 202, 303)`, model seeds `(11, 22, 33)`.

**Assumption class: entirely D.** G1 makes no claim to be yeast. Its purpose is
to establish that the *evaluation protocol* is sound — trajectory-level splits,
combinatorial holdout, reporters as training-only labels — before any biology is
introduced. `data/safeguards.csv` records **24/24 critical safeguards passing**,
including "no reporter values used at deployment in Experiments 2A–2C
(prediction functions accept only e and t)" and "figures use saved model
predictions rather than true-parameter ODE curves."

Headline held-out-combination RMSE: 1A 0.0069, 1B 0.0116, 1C 0.0142, 2B 0.0047.

**Do not cite G1 as biological evidence.** The writeup already states this:
"the hidden states were designed by the project, so recovery of those states did
not prove anything about real yeast."

---

## 4. Generator G2 — reduced dFBA state machine (Exp 3A–3F)

**Contract:** `e = [T, pH, DO] -> B(t)`, 49 points, `dt = 0.25`, 2 replicates,
dataset seeds `(701, 702, 703)`.

G2 is structurally identical to G3's outer loop but replaces the 144 LPs with a
closed-form `reduced_flux_solution`:

```
phi_T   = exp(-(T-30)^2 / (2*5^2)),  phi_pH = exp(-(pH-5)^2 / (2*0.75^2))
low_DO  = sigmoid((24 - DO)/5.5),    high_DO = sigmoid((DO - 66)/6)
mu_max  = 0.47 phi_T phi_pH * clip(1 - 0.42 low_DO - 0.20 high_DO, 0.25, 1.12)
            * (1 - 0.22 z_ox - 0.18 z_atp)
mu      = gamma(state) * mu_max                      (realised biomass flux)
v_prec  = 0.72 phi_T (0.55 + 0.45 phi_pH) (1 - 0.18 low_DO)
beta_cap= 0.38 phi_T (0.78 + 0.22 phi_pH) (1 - 0.62 z_bottle - 0.18 z_ox)
v_beta  = clip(0.62 v_prec - 0.18 mu, 0, beta_ub(state) * beta_cap)
repair  = 0.18 z_ox + 0.12 z_atp + 0.10 z_bottle
rho_atp = (v_ATPM + 0.88 mu + 0.55 v_beta + 0.25 repair) / atp_supply
```

Biomass is logistic (`k_x = 6.0`) with state-dependent death (0.000–0.032 h⁻¹),
and product degradation carries a state multiplier up to 12× in
`combined_overload`. Environment grid: 5×5×5 over `T ∈ {24..36}`,
`pH ∈ {4.0..6.0}`, `DO ∈ {10..80}`, plus six explicit extrapolation points.
Splits are assigned by named holdout regions (`highT_lowDO`, `highT_lowpH`,
`highDO_offpH`, `three_way_corner`) — i.e. by *stated biological corner*, not by
random masking.

**The `z_er` decoy.** A fourth burden state is driven purely by environmental
deviation and is deliberately given **no causal path** into any flux or
constraint. Its reporter `R_ER` is therefore environment-correlated but
causally useless. `dfba_safeguards.csv` records a finite-difference test:
perturbing `z_er` leaves all causal fluxes unchanged. This is the generator's
built-in trap for reporter-supervision methods that latch onto correlation.

**Assumption class: D with L-shaped functional forms.** Gaussian environmental
windows and sigmoid DO penalties are the standard phenomenological shapes for
microbial environmental response; the coefficients are declared, not measured.
G2's whole role was to test whether adding biological *structure* changed which
model architecture wins. `dfba_safeguards.csv`: **16/16 safeguards pass**.

**Result it produced — a negative one.** Best held-out RMSE was achieved by
`coordinate_conditioned_mlp` in 3A, 3B and 3F; reporter supervision was not a
robust win. The project's response was to audit the generator rather than tune
architectures, which is the correct order of operations and is documented in
§5 of the technical writeup.

---

## 5. Generator G2b — physiological-regime ODE testbed

A deliberately separate, fully deterministic 7-state RK4 ODE
(`X, A, R, E, S, C, P` = biomass, allocation, precursor, energy, stress,
congestion, product) over `t ∈ [0, 48] h`, 73 points, with environment domain
matched to G3 (`T ∈ [27,33]`, `pH ∈ [4.5,5.5]`, `DO ∈ [20,80]`).

Its purpose is to provide a testbed where **eight named physiological regimes**
(`balanced_sustained`, `delayed_pathway_activation`, `precursor_limited_plateau`,
`atp_limited_slow`, `oxidative_stress_collapse`, `congestion_recovery`,
`growth_dominated_low_product`, `mixed_transition`) genuinely occur and are
labelled — precisely the shape diversity that G3 was measured to *lack* (§6.3).

`mu_max = 0.105 h⁻¹`, `k_product = 0.235`, death base `0.002 h⁻¹`. Environmental
responses are Gaussian in T (`opt 30.2, sigma 1.55`) and pH (`opt 5.05, sigma 0.28`)
with a Hill response in DO (`K = 35, n = 2.2`).

**Assumption class: D, explicitly.** The module docstring states it "is
intentionally separate from the Yeast9/pFBA pipeline… a transparent synthetic
testbed." It is a diagnostic instrument, not evidence.

---

## 6. Internal validity evidence already in the repository

This is the part of the argument that does not depend on anyone believing the
parameter choices.

### 6.1 The generator provably ran the real model

| Check | Source | Result |
| --- | --- | --- |
| Selected XML loads; augmented model feasible | `data/gem_pilot_acceptance_criteria.csv` | pass |
| Every interval completes all optimization stages | same | pass |
| Zero surrogate evaluations | same | pass |
| LP solve counts recorded honestly | same | pass |
| No flux-solution reuse flag | same | pass |
| Dynamic burdens alter at least one active GEM constraint | same | pass |
| Altered constraints change at least one meaningful flux | same | pass |
| Deterministic, reproducible from a clean process | same | pass (runner uses no random numbers) |
| 144 LP solves/culture, 0 infeasible, 0 unbounded | `data/gem_dynamic_capacity_solver_accounting.csv` | 48/48/48 per culture across 27 environments |

### 6.2 The decoy state is provably non-causal

`data/gem_pilot_er_isolation.csv`: `er_isolation_passed = True`,
`er_isolation_max_abs_causal_difference = 0.0`. A finite perturbation of `z_er`
changes **no** causal output, exactly.

### 6.3 The generator's own weaknesses are measured and reported

`data/gem_dynamic_capacity_acceptance.csv` runs 16 gates. Twelve pass. **Four
fail**, and the run is classified `gem_capacity_feedback_too_weak` with
`state_space_ml_unlocked = False`:

| Failed gate | Value |
| --- | --- |
| `normalized_PC1_below_99` | amplitude-normalized product PC1 = **0.99805** |
| `normalized_PC1_below_stronger_98` | same |
| `three_distinct_shape_families` | not achieved |
| `thirty_percent_limiting_reaction_switches` | not achieved |

This is the single most important fact about G3's honesty: **the project's own
audit says the product-shape family is close to one-dimensional.** It did not
hide this, and the diagnostic state-space comparison that followed reported a
negative result for the original hypothesis — `polynomial_environment_pca`
reached mean trajectory-normalized RMSE 0.002492 versus 0.023712 for the
product-only state-space model, paired best-state-space-minus-best-direct delta
`+0.008895`, win fraction 0.0, bootstrap CI `[0.007374, 0.011401]`, conclusion
`state_space_matches_direct_models` / `reporter_supervision_not_supported`.

### 6.4 Why a low-dimensional *prediction* target does not invalidate the *design* claim

This is the crux, and it should be stated explicitly in the paper.

PC1 ≈ 99.8% is a statement about the **product trajectory shape** across
environments at fixed strain. The acceleration claim is about the **design
objective landscape** over the 10-dimensional edit space × 3-dimensional
environment, under a hidden regulator that is not observable from any single
culture. Those are different objects. Evidence that the design landscape is
*not* trivial:

- `data/design_benchmark_exact_pool_audit.csv` (strong_dynamic_regulation, 48
  exact candidates): product range 1.012, objective range 1.504,
  **30 distinct regulator state sequences** across 40 distinct strains and 16
  distinct environments, active-constraint diversity 4, 46/48 feasible, 0 growth
  collapses.
- `data/prospective_pilot_acceptance.csv`: candidate final-product range across
  the campaign = **5.967** — the objective is far from constant over the design
  space.

So: the *trajectory* is compressible; the *design response surface* is not.
The twin is being asked to do the second thing.

### 6.5 The comparison is leakage-audited

`data/prospective_pilot_acceptance.csv` — all 11 gates pass, final status
`FULL_POWERED_PROSPECTIVE_BENCHMARK_VALID`:

| Gate | Evidence |
| --- | --- |
| A exact backend | `yeast_gem_lp` |
| B no surrogate/mock lab | `mock_rows = 0; surrogate_evaluations = 0` |
| C on-demand exact | `fresh = 268; prospective_cache_hits = 212; total = 480` |
| D conventional adaptivity | `proposal_rows = 480; adaptive_stages = 60` |
| E hybrid blind virtual search | `virtual_rows = 120000` |
| F historical transfer, new edits | `transfer_rows = 480` |
| G solver health | `infeasible = 0; unbounded = 0; errors = 0; skipped = 0` |
| H biological validity | `480/480 valid` |
| I accounting complete | `cultures = 480; virtual = 120000; stages = 100; lp = 69120` |
| J reproducibility frozen | `campaigns = 20` |

`data/prospective_leakage_audit.csv` additionally confirms the virtual search
recorded no exact outcomes and the old exact caches were not used as a source.
480 cultures × 144 LPs = 69,120 — the accounting closes exactly.

Campaign design (full, not pilot): `conventional_batches = [4, 4, 4, 4]`
(4 adaptive stages x 4 cultures = 16 physical cultures),
`hybrid_verification_batch = 8`, `hybrid_virtual_evaluations = 6000`,
`mock_mode = False`. Per campaign 16 + 8 = 24 cultures; 24 x 20 = 480; 480 x 144
= 69,120 LP solves; 6000 x 20 = 120,000 virtual evaluations. Every count in the
acceptance table reconciles.

Headline: conventional mean best final product 1.191 vs hybrid 1.978; paired
mean delta 0.787, bootstrap CI `[0.084, 1.565]`; hybrid wins 13/20 campaigns;
median delta 0.118.

### 6.6 Provenance discrepancy found during this audit (action required)

`run_gem_dynamic_capacity.parameter_sweep()` swept six points around
`selected_parameter_config()` and selected the multiplier tuple
`(capacity 1.15, damage 1.00, synthesis 1.20)`. The selected parameters were
written to `data/gem_capacity_selected_parameters.csv`:

```
psy/des/cyc_base_capacity   = 0.391 / 0.322 / 0.2645
psy/des/cyc_synthesis_base  = 0.084 / 0.072 / 0.066
```

and those are the values quoted in
`YEAST_DIGITAL_TWIN_COMPREHENSIVE_TECHNICAL_WRITEUP.md` section 7.

However `selected_parameter_config()` in `src/yeast_validation/run_gem_dynamic_capacity.py`
still returns the **pre-sweep base point**:

```
psy/des/cyc_base_capacity   = 0.34  / 0.28  / 0.23
psy/des/cyc_synthesis_base  = 0.070 / 0.060 / 0.055
```

and `design_benchmark_exact.regime_configs()` calls that function directly to
build `base`, from which every DBTL world is derived by `dataclasses.replace`.

**Consequence.** The DBTL benchmark worlds are parameterised from the sweep's
base point, not the sweep's winner. Capacities are 13% lower and synthesis rates
17% lower than the writeup states. Decay rates, oxidative sensitivities,
congestion sensitivities and `E_min` are unaffected (they match exactly).

**This is not a correctness bug** — the base point is a legitimate, declared,
frozen parameterisation and the worlds rescale from it consistently. But the
*provenance claim* "the selected dynamic-congestion parameter set was chosen by a
small deterministic generator-only pilot" is not true of the benchmark worlds as
run. Resolve by one of:

1. Correcting the writeup and this document to say the DBTL worlds use the
   sweep base point (cheapest, no recompute), **or**
2. Updating `selected_parameter_config()` to the swept winner and rerunning the
   benchmark (expensive; changes headline numbers), **or**
3. Recording both explicitly in the freeze manifest and stating that the sweep
   selected the capacity/dynamics *regime*, and the benchmark uses the base
   point of that regime.

Option 1 or 3 is recommended. Whichever is chosen, the discrepancy must not
survive into the paper unnoted.

**Resolution for the Latent Capacity Mismatch Benchmark (this audit pass).**
Option 3. The existing headline DBTL result (§6.5, delta 0.787/CI/median 0.118)
is left untouched — it is not recomputed or renormalised here, and its
provenance caveat above stands as written. The **new** benchmark's freeze
manifest (Stage E, not yet produced) will record explicitly which parameter
set it uses (the sweep base point, matching `selected_parameter_config()` as
it exists today) so the code and the documentation agree for the new work,
without silently changing the old one.

---

## 7. Master assumption ledger

| ID | Assumption | Class | Support | If wrong, what breaks | Tested here? |
| --- | --- | --- | --- | --- | --- |
| G3-01 | Stoichiometry, cofactor coupling and GPRs are those of yeast-GEM v9.0.2 | **M** | Community consensus model, checksummed | Nothing local; would be a community-model error | Asset audit + SHA-256 |
| G3-02 | Beta-carotene route can be lumped into PSY / DES / CYC + sink | **L** | Canonical crtE/crtYB/crtI route | Intermediate-specific effects (e.g. phytoene toxicity) are invisible | Pathway manifest; no intermediate assay exists |
| G3-03 | Lumped steps create no thermodynamic loophole | **M** | Audited: no ATP, redox, carbon or precursor cycle | LP could farm free product | Manifest justification field |
| G3-04 | Product accumulates intracellularly (sink, not excretion) | **D** | Carotenoids are intracellular in yeast | Extraction/partitioning effects absent | — |
| G3-05 | `m_ATP = 0.70 mmol gDW⁻¹ h⁻¹` | **L** | Measured 0.63 ± 0.04 aerobic | ATP-pressure world is mis-scaled | Within 11% of measurement |
| G3-06 | `q_glc = 10`, `q_O2,100 = 12 mmol gDW⁻¹ h⁻¹` | **L** | Aerobic batch range | Absolute rates shift; relative ranking likely stable | Swept via edit multipliers |
| G3-07 | DO maps linearly to oxygen uptake capacity | **A** | none | Oxygen worlds mis-shaped | **Not yet ablated** |
| G3-08 | T and pH act as a *maintenance* multiplier `f_env` | **A/L** | Optima 30 °C / pH 5 are literature; the multiplicative-ATPM form is not | Environmental response shape wrong | Optima grounded; form not ablated |
| G3-09 | Lexicographic growth-then-product allocation (Protocol A) | **D** | none — a modelling convention | 3× shift in growth/product split | **Measured**: A/B/C comparison table |
| G3-10 | pFBA resolves remaining degeneracy | **M** | Standard method | Flux features become arbitrary | Solve accounting |
| G3-11 | Product flux generates oxidative burden (`0.22 v_beta`) | **L (weak)** | Lycopene accumulation impairs growth; beta-carotene autoxidises — **but** one study reports no significant growth effect from beta-carotene accumulation | Central burden loop may be stronger than reality | **Ablatable**: `GEMCultureConfig.oxidative_beta_coupling` (default 0.22, exposed as a first-class field; set to 0.0 to remove the term). See `tests/test_gem_oxidative_ablation.py`. Not yet run as a full campaign ablation — see §7.3 of the Latent Capacity Mismatch design. |
| G3-12 | Step-imbalance (congestion) damages downstream enzyme capacity | **L** | crtI over-dosage → lycopene accumulation → growth impairment → rebalancing required | The "moving limiting step" worlds lose their basis | Congestion terms logged per interval |
| G3-13 | 7-state discrete regulator with hysteresis and 2-interval dwell | **D** | none — declared difficulty | The task stops being a hidden-regime task | Frozen in regime manifest; 30 distinct sequences observed |
| G3-14 | Burden thresholds 0.62 / 0.58 / 0.60 (hi), 0.42 / 0.38 / 0.40 (lo) | **A** | none | State occupancy distribution shifts | **Not ablated** |
| G3-15 | Enzyme capacity floor `E_min = 0.08` | **A** | prevents irrecoverable death | Cultures could permanently collapse | Chosen by generator-only pilot |
| G3-16 | Edits act as bounded multipliers on capacities/bounds, not as new reactions | **D** | Keeps the design space finite and auditable | Real edits (deletions, new routes) unrepresented | Edit universe frozen |
| G3-17 | Precursor edits act on native `r_0461` bounds | **M/D** | Forces effects through the real network | A synthetic bypass would trivialise precursor edits | Code comment + implementation |
| G3-18 | **No extracellular mass balance**: glucose bound is constant, biomass grows exponentially with no carrying capacity or death | **A — most significant limitation** | none | No stationary phase, no substrate exhaustion, no diauxie; late-horizon behaviour is not a real fed-batch | **Modeled, opt-in**: `GEMCultureConfig.substrate_limited` (default `False`, reproduces old behaviour bit-for-bit). See §2.9. Still no feed, maintenance death term, or diauxic shift — this is depletion only, not a full bioreactor model. |
| G3-19 | Worlds differ by parameter scaling, not by network topology | **D** | Keeps the M-layer constant across worlds | Distribution shift is parametric only | Regime manifest |
| G3-20 | The generator is deterministic (no measurement noise in G3) | **D** | Makes leakage auditing exact | Real assay noise absent; benchmark is optimistic about measurement | **Partially addressed.** Reporters already carried seeded lag+noise (`generate_gem_state_space_dataset.build_reporters`, predates this audit). Product/biomass/glucose trajectories now have an opt-in seeded observation layer (`src/yeast_validation/gem_observation_noise.py`, §2.10) producing `*_observed` columns alongside the still-deterministic `*_true`/internal state — the internal ground truth used for leakage auditing remains exact and untouched. |
| G2-01 | Reduced surrogate stands in for LP when no GEM asset is present | **D** | Explicitly labelled `reduced_stoichiometric_surrogate` in every row | — | `MissingMetabolicAsset` raised otherwise |
| G2-02 | `z_er` is environment-correlated but causally isolated | **D** | Deliberate decoy | The reporter-quality result would be vacuous | Finite-difference isolation test |
| G1-01 | Product ODE `dP/dt = F(t,e) - k_P P` | **D** | Not biological | — | 24/24 safeguards |
| G1-02 | Reporters are lagged first-order filters + Gaussian noise | **D/L** | Fluorescent maturation lag is real; constants are invented | Reporter-quality frontier is a protocol result, not a biological one | Shuffled-reporter control |

---

## 8. What the generator does not model (state this in the paper)

1. **No extracellular mass balance.** Glucose availability is a fixed uptake
   bound re-applied every interval; it is never depleted. Biomass integrates as
   `dX/dt = mu X` with no carrying capacity and no death term in G3. Over the
   12 h horizon this is a growing, substrate-sufficient culture by construction.
   The generator is therefore a *constraint-based dynamic simulator*, not a
   fed-batch or bioreactor model. **This is the strongest honest caveat and
   should appear in the limitations section verbatim.**
2. **No measurement noise in G3.** The exact oracle is deterministic. Real
   DBTL rounds carry assay noise, biological replicate variance, and strain
   construction failure. The benchmark measures search efficiency under a
   noiseless oracle, which is an upper bound on both methods.
3. **No transcriptional or proteomic layer.** Enzyme capacity is a scalar per
   step, not a protein pool with allocation constraints. No ecYeast/GECKO-style
   enzyme constraint is applied.
4. **No pathway intermediates as states.** Phytoene and lycopene exist as LP
   metabolites but do not accumulate as dynamic pools; congestion is inferred
   from flux and capacity mismatch rather than measured concentration.
5. **No strain construction cost or failure rate.** Every proposed design is
   assumed buildable.
6. **No spatial or population heterogeneity.** Single well-mixed compartment,
   single genotype.
7. **Worlds differ parametrically, not topologically.** No world adds or removes
   reactions, so the twin never faces a genuinely novel network.

---

## 9. Falsifiability — what would invalidate the claim

State these as pre-registered failure conditions.

| # | Falsifier | Status |
| --- | --- | --- |
| F1 | A static single-LP GEM ranking (no dynamics) matches the hybrid twin's design performance | Included as `static_gem_public_rank` / `static_gem_parallel` baseline |
| F2 | The advantage disappears when the ablated burden coupling `0.22 v_beta` is set to 0 | **Not yet run — recommended (see §10)** |
| F3 | The advantage disappears in a world the twin was not pretrained on | **Weakly addressed**: baseline delta 0.977 vs strong-oxidative delta 0.598 — advantage shrinks but survives. Only 1 shifted world of 6 declared has been run (see §2.8) |
| F4 | The conventional workflow was under-powered (not genuinely adaptive) | Audited: `adaptive_stages = 60`, `conventional_later_proposals_after_prior_exact = 60` |
| F5 | Any exact outcome reached the hybrid's virtual search | Audited: leakage + old-cache-collision, both clean |
| F6 | Results depend on the allocation protocol | Measured: A/C identical, B differs 3× in split — **rerun the headline under Protocol B to close this** |
| F7 | The paired advantage is not distinguishable from zero | Bootstrap CI `[0.084, 1.565]` excludes zero; median delta 0.118 is much smaller than the mean — the effect is driven by a subset of campaigns and should be reported as such |

---

## 10. Recommended additions before publication

Ordered by how much they strengthen the validity argument per unit of compute.

1. **Burden-coupling ablation (F2).** Rerun a reduced campaign set with
   `oxidative_generation = 0.16 * oxygen_stress` only (drop the `0.22 v_beta`
   term). This is the single weakest literature link (G3-11) and the one a
   reviewer will attack. Cost: one world × a few seeds.
2. **Protocol B headline replicate (F6).** The allocation protocol is a declared
   convention with a measured 3× effect. Showing the ranking survives Protocol B
   converts G3-09 from a declared assumption into a tested one.
3. **Report the median alongside the mean.** Mean delta 0.787 vs median 0.118
   means the distribution is skewed. Report both, plus the per-campaign win
   count (13/20), or a reviewer will find it.
4. **Add a substrate mass balance, or explicitly bound the claim to the
   exponential-growth regime.** G3-18 is the limitation most likely to be raised.
   Either fix it or scope the claim.
5. **Ablate the state-machine thresholds (G3-14).** A ±0.05 perturbation of the
   hysteresis thresholds, showing the ranking is stable, retires an entire class
   of "you tuned the regulator" objections.
6. **Run at least one non-oxidative shifted world.** `atp_limited_world` and
   `precursor_competition_world` are already frozen and cost nothing to declare;
   running one of them turns "generalises across worlds" from a design intention
   into a measurement (see §2.8 scope warning).
7. **Resolve the parameter provenance discrepancy in §6.6** before submission.

---

## 11. Reproducibility manifest

```
GEM asset       yeast-GEM.xml  (yeastGEM v9.0.2)
SHA-256         9fd2c572cace73c2ea835205617313554d1bf89f4ef077f49defbdfa219a4ad7
Path            $YEAST_GEM_PATH (discovery: models/, then raw asset root)
Solver          GLPK via optlang 1.9.1
Libraries       cobra 0.31.1, python-libsbml 5.21.1, pandas 2.3.3, numpy 2.5.1
Simulator tags  yeast9_dynamic_pfba_on_demand_v1
                final_dbtl_exact_sparse_dynamic_pfba_v1
Regulator tag   gem_backend_decision_tree_next_state_v1
Backend tag     yeast_gem_lp     (EXACT_BACKEND)
Horizon         TIME_POINTS = 49, dt = 0.25 h  -> 12 h, 48 intervals
LP per culture  144  (EXPECTED_LP_SOLVES = (49-1) * 3)
Campaign seeds  prospective 86001..86020; final DBTL 51001..51003
World seeds     9301..9307 (+ legacy 9101, 9102)
Pool build seed 9201
G1 seeds        dataset (101,202,303), model (11,22,33)
G2 seeds        dataset (701,702,703), model (17,29,43)
G2b seed        20260724
```

Every exact trajectory row additionally carries `gem_source_checksum`,
`gem_model_id`, `solver_name`, `metabolic_backend`, `hidden_regulator`,
`capacity_mode`, and the full LP-solve counter set, so any downstream table can
be traced back to the model instance that produced it.

---

## 12. References

Model and methods

- Zhang C. et al. (2024) *Yeast9: a consensus genome-scale metabolic model for
  S. cerevisiae curated by the community.* Molecular Systems Biology.
  https://link.springer.com/article/10.1038/s44320-024-00060-7
- yeast-GEM repository, SysBioChalmers.
  https://github.com/SysBioChalmers/yeast-GEM
- Lu H. et al. (2019) *A consensus S. cerevisiae metabolic model Yeast8 and its
  ecosystem.* Nature Communications.
  https://www.nature.com/articles/s41467-019-11581-3
- Mahadevan R., Edwards J.S., Doyle F.J. (2002) *Dynamic flux balance analysis
  of diauxic growth in E. coli* — the static-optimisation-approach discretisation
  used by this generator's outer loop.
  https://www.researchgate.net/publication/11186886_Dynamic_Flux_Balance_Analysis_of_Diauxic_Growth_in
- Parsimonious FBA (pFBA), as implemented in `cobra.flux_analysis.pfba`.
  https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/parsimonious/index.html

Physiology and parameters

- Boender L.G.M. et al. / Vos T. et al. (2016) *Maintenance-energy requirements
  and robustness of Saccharomyces cerevisiae at aerobic near-zero specific growth
  rates.* Microbial Cell Factories — NGAM 0.63 ± 0.04 mmol ATP gDW⁻¹ h⁻¹.
  https://link.springer.com/article/10.1186/s12934-016-0501-z
- Hagman A. et al. (2015) *A study on the fundamental mechanism and the
  evolutionary driving forces behind aerobic fermentation in yeast.* PLOS ONE —
  aerobic specific uptake rates and the Crabtree effect.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0116942

Beta-carotene pathway

- López J. et al. (2020) *Engineering Saccharomyces cerevisiae for the
  overproduction of beta-ionone and its precursor beta-carotene.* Frontiers in
  Bioengineering and Biotechnology — crtE/crtYB/crtI copy-number balancing,
  4 → 12 → 32 mg/gDCW, lycopene accumulation impairs growth, mevalonate pathway
  rate-limiting.
  https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.578793/full
- Sun L. et al. (2021) *Metabolic engineering of Saccharomyces cerevisiae for
  production of beta-carotene from hydrophobic substrates.* FEMS Yeast Research —
  X. dendrorhous crtE/crtYB/crtI, 477.9 mg/L, 46.5 mg/gDCW.
  https://academic.oup.com/femsyr/article/21/1/foaa068/6041025
- *Advances in the biosynthesis of beta-carotene and its derivatives in yeast.*
  Bioresource Technology (2025).
  https://www.sciencedirect.com/science/article/pii/S0960852425009022
- *Beta-carotene autoxidation: oxygen copolymerization, non-vitamin A products,
  and immunological activity.* Canadian Journal of Chemistry.
  https://cdnsciencepub.com/doi/10.1139/cjc-2013-0494

Metabolic burden

- *Burden imposed by heterologous protein production in two major industrial
  yeast cell factories.* Frontiers in Fungal Biology (2022).
  https://public-pages-files-2025.frontiersin.org/journals/fungal-biology/articles/10.3389/ffunb.2022.827704/pdf
- *Engineering strategies for enhanced heterologous protein production by
  Saccharomyces cerevisiae.* Microbial Cell Factories (2024).
  https://link.springer.com/article/10.1186/s12934-024-02299-z

Internal sources

- `src/yeast_validation/gem_backend.py`, `src/yeast_validation/design_benchmark_exact.py`,
  `src/yeast_validation/run_dfba_state_machine_validation.py`,
  `src/yeast_validation/run_fixed_environment_validation.py`, `src/yeast_validation/ode_regime_core.py`
- `data/gem_pilot_acceptance_criteria.csv`, `data/gem_dynamic_capacity_acceptance.csv`,
  `data/gem_dynamic_capacity_solver_accounting.csv`, `data/gem_pilot_er_isolation.csv`,
  `data/gem_allocation_protocol_comparison.csv`, `data/safeguards.csv`,
  `data/dfba_safeguards.csv`, `data/design_benchmark_exact_pool_audit.csv`,
  `data/prospective_pilot_acceptance.csv`, `data/prospective_leakage_audit.csv`,
  `data/yeast_gem_audit.csv`, `data/yeast_gem_selected_asset.csv`
- `YEAST_DIGITAL_TWIN_COMPREHENSIVE_TECHNICAL_WRITEUP.md`
