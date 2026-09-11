# A Hybrid Regulation-Metabolism Digital Twin for Accelerating Simulated Yeast Bioprocess DBTL

*Development, Validation, Negative Results, and Deployment-Style Evaluation*

## Abstract

Bioprocess design-build-test-learn (DBTL) is constrained by sequential biological feedback: strain construction, cultivation, measurement, analysis, and redesign often must occur in ordered rounds even when individual cultures can be run in parallel. This project evaluated whether a biosensor-informed regulation-metabolism digital twin could accelerate simulated yeast beta-carotene DBTL by learning physiological regulation, coupling learned controls to mechanistic Yeast9 dynamic pFBA, performing large virtual strain-by-environment searches, and physically verifying selected designs in parallel.

The development proceeded through falsifiable stages. Synthetic fixed-environment experiments showed that reporter supervision can help when informative hidden physiological states are deliberately present, with held-out reporter-supervised product RMSE as low as 0.0047. However, increasing biological realism did not make latent state-space models superior predictors. In an expanded exact Yeast9 dynamic-capacity dataset, direct environment-to-trajectory models outperformed state-space variants; the best state-space versus best direct trajectory-normalized RMSE delta was +0.008895, with state-space win fraction 0.0. Exact finite-pool optimization also did not favor the hybrid: space filling had the lowest mean exact-pool regret.

The final validation therefore tested complete simulated DBTL workflows rather than prediction RMSE. A non-mock Yeast9 pilot and full Vanda run completed 13,920/13,920 exact hidden-wet-lab tasks with zero missing, failed, duplicate, hash-mismatch, or invalid outputs. The main comparison was conventional iterative DBTL versus a biosensor-informed modular hybrid digital twin that performs parallel calibration, virtual strain-by-environment exploration, and parallel final verification.

In strict cultures-to-target accounting, the hybrid did not reduce total physical cultures relative to conventional DBTL. For the moderate target, conventional DBTL attained target in 0.533 of campaigns, while the canonical full-reporter hybrid attained 0.483; mean cultures saved versus conventional was -0.3. However, dependency auditing showed that post-calibration hybrid proposals did not require newly revealed exact wet-lab outcomes and could be collapsed into one parallel verification stage without future-information leakage. Under the stated high-parallel-capacity scenario, the hybrid reduced mean sequential biological depth from 3.20 to 1.97 stages and scenario calendar time from 31.0 to 17.7 days, a 42.9% simulated time reduction and 1.75x speedup. The strongest supported practical result is therefore reduced sequential experimental depth, not fewer total cultures.


## Introduction

The engineering problem addressed here is not merely prediction of a product curve. In yeast bioprocess DBTL, useful design decisions often require repeated sequential cycles of strain construction, cultivation, measurement, analysis, and redesign. Even a laboratory that can run many cultures simultaneously is limited when the next designs cannot be selected until the current biological round is measured.

The project therefore distinguishes total experimental breadth from sequential experimental depth. Breadth is the number of physical cultures consumed. Depth is the number of biological feedback stages that must occur in sequence. A digital twin may fail to reduce total cultures and still accelerate DBTL if it replaces later biological feedback rounds with computation and a larger parallel verification batch.

Equation (1): `T_DBTL approx N_stages T_culture + T_planning`

Equation (2): `T_twin approx T_calibration + T_compute + T_verification`

The proposed workflow uses an initial parallel culture batch to characterize physiology, uses biosensors to expose otherwise hidden regulatory state, learns a regulatory model, connects that model to mechanistic metabolism, performs many virtual strain-by-environment experiments, and verifies a selected batch in parallel. The central hypothesis refined over the project was that a digital twin may accelerate DBTL not necessarily by requiring fewer cultures, but by replacing sequential biological rounds with computational search and parallel verification.

![Figure 1. Overall digital-twin hypothesis and deployment architecture.](figures/scientific_report/fig01_architecture.pdf)


## Overview of the Scientific Development

The scientific development is best read as a sequence of hypotheses tested and revised. The early synthetic experiments asked whether hidden physiology and reporters could improve trajectory prediction. The dFBA state-machine and real Yeast9 phases then tested whether increasing biological realism made latent state-space models necessary. Those experiments produced important negative results. The project therefore shifted from asking whether the hybrid was the best predictor to asking whether learned regulation could provide a compact interface into mechanistic metabolism and whether that interface could improve a complete DBTL workflow.

![Figure 2. Scientific development and hypothesis chain.](figures/scientific_report/fig02_hypothesis_chain.pdf)

**Table 1. Major scientific questions and outcomes**
| Question | Expected outcome | Observed outcome | Consequence |
| --- | --- | --- | --- |
| Do hidden states justify state-space models? | Reporter-supervised latent models should generalize better. | Supported only in deliberately synthetic systems. | Move to biologically grounded generators. |
| Does mechanistic realism make latent dynamics necessary? | State-space models should improve as simulator complexity rises. | Not supported; direct models remained strongest. | Reframe from prediction to digital-twin workflow. |
| Can learned regulation drive mechanistic metabolism? | Compact controls should reproduce exact pFBA when correct. | Oracle replay matched original GEM trajectory to numerical precision. | Use compact interface in hybrid design. |
| Does hybrid search dominate finite-pool optimization? | Hybrid UCB should beat classical baselines. | Space filling had lowest exact-pool regret. | Use deployment-style rather than equal-query benchmark. |
| Can the twin reduce sequential DBTL time? | Virtual search should collapse biological feedback depth. | Supported after wall-clock correction, with more cultures. | Final bounded claim. |


## Question 1: Does Hidden Physiological State Justify a State-Space Model?

The first question was whether hidden physiological state can improve prediction of production trajectories and whether biosensors make that state learnable. If different physiological states produce different dynamics under similar environmental conditions, explicitly learning latent state should improve held-out generalization. If reporters provide informative measurements of those hidden states, reporter-supervised models should outperform product-only models.

The synthetic fixed-environment generator held one environment vector constant for each culture and generated a complete product trajectory through a dynamic product equation. Later synthetic variants introduced hidden physiological states and reporter channels. Splits were culture-level, including held-out environmental combinations, so no time point from a held-out culture appeared in training.

Equation (3): `dP/dt = F(t, e, z) - k_P P`

Equation (4): `R_k(t) = h_k(z(t)) + noise + lag`

The synthetic experiments supported the local reporter hypothesis. In the held-out-combination setting, the best reporter-supervised model in Experiment 2B reached product RMSE of approximately 0.0047. Reporter corruption controls showed that clean, sparse, missing, noisy, mild-lag, gain/offset, and combined conditions could remain useful by the predeclared improvement criterion, whereas slow lag was harmful. This established that reporter supervision can help when informative hidden state is deliberately present.

The interpretation was bounded. Synthetic latent variables were constructed by the simulator, so recovering or exploiting them did not prove that comparable latent variables exist in yeast. The result motivated the next question: does the advantage survive when the production system becomes biologically grounded?

![Figure 3. Synthetic reporter supervision on held-out combinations.](figures/scientific_report/fig03_reporter.pdf)

**Table 2. Reporter-quality frontier on held-out combinations**
| Reporter condition | Product RMSE | Paired product-only RMSE | Relative improvement |
| --- | --- | --- | --- |
| missing | 0.0042 | 0.1087 | 0.961 |
| sparse | 0.0047 | 0.1087 | 0.957 |
| clean | 0.0080 | 0.1087 | 0.927 |
| mild_lag | 0.0133 | 0.1087 | 0.878 |
| noisy | 0.0160 | 0.1087 | 0.852 |
| gain_offset | 0.0310 | 0.1087 | 0.715 |
| combined | 0.0833 | 0.1087 | 0.235 |
| slow_lag | 0.1331 | 0.1087 | -0.224 |


## Question 2: Does Increasing Biological Realism Make Dynamic Latent Models Necessary?

The second hypothesis was that early direct models might have succeeded because the synthetic generator was too simple. If so, adding mechanistic metabolism and dynamic physiological constraints should make latent state more valuable. The project tested this first with a dFBA state-machine surrogate and then with exact Yeast9 dynamic pFBA.

The state-machine generator mapped fixed temperature, pH, and dissolved oxygen to hidden oxidative, ATP, pathway, and bottleneck states. These states modified metabolic constraints, and product and biomass emerged from the constrained metabolic system. Direct models nevertheless remained competitive or superior, which motivated a generator audit rather than neural tuning.

The real Yeast9 transition made the simulator substantially more mechanistic. The selected Yeast9 asset loaded with approximately 4,131 reactions, 2,806 metabolites, and 1,161 genes. The beta-carotene augmentation reused native GGPP formation and added pathway reactions for phytoene synthase, phytoene desaturase, lycopene cyclase, and beta-carotene demand. In each interval of a simulated culture, the verifier applied environment and physiological constraints, solved maximum growth, preserved a state-dependent growth fraction, optimized beta-carotene production, ran parsimonious FBA, updated biomass and product, and used solved metabolic behavior to update physiological burdens and pathway capacities.

![Figure 4. Exact Yeast9 dynamic-pFBA simulator loop.](figures/scientific_report/fig04_yeast9.pdf)

Dynamic pathway capacity states for PSY, DES, and CYC were introduced because the first real-GEM trajectories were flux-rich but product-shape simple. The selected dynamic-congestion run completed 3,888 real Yeast9 LP solves and changed pathway capacities substantially, but accumulated product remained highly compressible: in the full 49-point grid, amplitude-normalized product PC1 explained 99.8049% of variance. The final dynamic-capacity classification was `gem_capacity_feedback_too_weak`.

The definitive direct-versus-state-space diagnostic used 125 fixed environments, 49 time points, 48 intervals, and 18,000 actual LP solves. Direct models led held-out-combination prediction. The best state-space versus best direct trajectory-normalized RMSE delta was +0.008895 state-space-minus-direct; the state-space win fraction was 0.0, with bootstrap upper confidence bound 0.011401. This rejected the stronger hypothesis that mechanistic realism would automatically make state-space models superior predictors.

![Figure 5. Direct models versus state-space models on held-out real-GEM trajectories.](figures/scientific_report/fig05_state_space.pdf)

**Table 3. Direct versus state-space diagnostic**
| Model | Held-out trajectory-normalized RMSE |
| --- | --- |
| Polynomial env-to-PCA | 0.002492 |
| Multi-output MLP | 0.005622 |
| Coordinate-conditioned MLP | 0.008639 |
| Mechanistic-rate state-space | 0.012561 |
| ATP reporter state-space | 0.020308 |
| Oxidative reporter state-space | 0.020457 |
| Product-only state-space | 0.023712 |
| Combined reporter state-space | 0.029127 |


## Question 3: Can a Modular Regulation-Metabolism Digital Twin Be Built?

The revised hypothesis was that learning physiology need not directly improve product prediction. Instead, a learned regulatory model could predict the compact metabolic constraints under which a mechanistic genome-scale model should operate. In this architecture, the primary hybrid does not simply use a neural network to predict final product. The learned model predicts regulatory and metabolic controls; Yeast9 produces the metabolic outcome.

Equation (5): `c(t) = g_theta(e, z(t));  v(t) = pFBA_Yeast9(c(t));  P(t+dt) = P(t) + dt f_product(v(t), X(t))`

The reporter-grounded teacher took environment as input and learned product, oxidative reporter, ATP reporter, pathway reporter, compact-interface, and selected flux-summary heads. Dense Latin-hypercube teacher pseudo-data were then generated inside the original support for distillation. The hybrid student predicted reporter channels, compact interface controls, and flux summaries from environment. Reporter information was used for training and declared reporter-condition workflows, but hidden simulator state and future exact outcomes were prohibited.

Exact replay validated the compact interface. Oracle compact controls reproduced the original GEM trajectory to numerical precision in the smoke culture, with B_total RMSE 2.61e-16. The one-culture oracle/teacher/hybrid replay completed 432 LP stages with zero infeasible, unbounded, solver-error, skipped-interval, or surrogate counts. Preliminary completed-culture metrics showed hybrid-control pFBA close to teacher-control pFBA, but this architecture validation did not yet prove optimization or DBTL acceleration.

![Figure 6. Teacher to compact interface to Yeast9 hybrid architecture.](figures/scientific_report/fig06_hybrid.pdf)


## Question 4: Can the Digital Twin Identify Better Strain x Environment Designs?

The next question moved from prediction to engineering. Strain and environment must be optimized jointly because pathway capacity, precursor supply, oxygen availability, ATP demand, growth allocation, and product accumulation interact. The design space therefore included competing sinks, precursor supply, PSY/DES/CYC capacities, ATP or cofactor support, oxygen support, export, glucose uptake, and product degradation or loss.

The benchmark distinguished cheap virtual evaluations from exact Yeast9 verification. The exact finite-pool benchmark froze weak and strong dynamic-regulation regimes and evaluated random parallel search, space filling, static GEM ranking, black-box Bayesian optimization, direct trajectory UCB, and hybrid digital twin UCB against a declared finite exact pool.

The result was another important negative finding. All 96 finite-pool candidates completed exact dynamic pFBA, consuming 13,824 actual LP solves with zero solver errors or surrogate evaluations. Space filling had the lowest mean exact-pool regret, followed by direct trajectory UCB and then hybrid digital twin UCB. The hybrid architecture therefore did not automatically create a superior optimizer in a small bounded candidate pool.

![Figure 7. Exact finite-pool optimization comparison.](figures/scientific_report/fig07_finite_pool.pdf)

**Table 4. Exact finite-pool optimization ranking**
| Rank | Method | Mean exact-pool regret | Mean best observed objective |
| --- | --- | --- | --- |
| 1 | Space filling | 0.265236 | 2.477068 |
| 2 | Direct trajectory UCB | 0.303118 | 2.439186 |
| 3 | Hybrid digital twin UCB | 0.303241 | 2.439062 |
| 4 | Black-box BO | 0.344624 | 2.397679 |
| 5 | Random parallel | 0.421701 | 2.320602 |
| 6 | Static GEM parallel | 0.499934 | 2.242369 |


## Question 5: Can a Digital Twin Accelerate the Complete DBTL Workflow?

Equal-query optimization is not the natural deployment model for a digital twin. A deployed twin is useful because virtual experiments are cheap and can be run at enormous scale. The operational hypothesis became: perform physical calibration, run many virtual experiments, and verify a small physical batch.

This question is fundamentally about time and dependency. Conventional DBTL is iterative: design, build, test, learn, redesign, build, test, and learn again. Even if each batch contains many parallel cultures, the next batch cannot be chosen until the previous biological batch has completed and been measured. The hybrid deployment is structurally different: parallel calibration cultures, train or calibrate the twin, large virtual strain x environment exploration, and parallel final verification.

Intermediate deployment-style experiments showed both promise and failure. Cached deployment was operationally cheaper but quality-inferior. Open-ended scientist-versus-hybrid campaigns were mixed: the scientist often found the best individual design, while the hybrid often had stronger mean batch quality. Readiness audits also found effective-phenotype collapse, where distinct requested edits mapped to repeated phenotypes. Those failures motivated intervention-space redesign and a definitive benchmark with explicit culture accounting.


## Definitive Simulated DBTL Benchmark

The definitive benchmark treated exact Yeast9 dynamic pFBA as a hidden wet lab. It froze six biological worlds, ten campaign seeds, hidden culture-to-culture physiology, seven reporter conditions, five workflows, physical culture accounting, virtual evaluation accounting, and strict method-access boundaries. The five workflows were conventional DBTL, Bayesian optimisation, black-box digital twin, modular hybrid without biosensors, and biosensor-informed modular hybrid.

The prepared full manifest contained 15,840 workflow culture requests and 13,920 unique exact hidden-wet-lab tasks. The accepted Vanda exact pilot covered 18 tasks across six worlds, three campaign seeds, all five methods, and all seven reporter conditions. It passed all acceptance gates: complete exact dynamic pFBA status, non-mock `yeast_gem_lp` backend, 144 LP solves per task, zero solver errors, correct physical-culture charge, reporter consistency, latent variability, and no hidden-state or outcome access. Mean pilot runtime was 142.143 seconds.

The full execution completed 13,920/13,920 exact tasks. The strict local audit found 13,920 valid tasks and zero missing, failed, duplicate, hash-mismatch, or invalid outputs. This established that the final result is an exact simulated wet-lab result rather than a surrogate analysis.

The purpose of the multiple methods and reporter branches was benchmark rigor. The main engineering comparison in the definitive result is conventional iterative DBTL versus the biosensor-informed modular hybrid digital twin. Black-box and no-biosensor workflows remain important controls, but they do not define the primary deployment claim.

![Figure 8. Definitive simulated DBTL benchmark design.](figures/scientific_report/fig08_definitive.pdf)

**Table 5. Definitive workflow definitions**
| Workflow | Role in benchmark | Physical/virtual distinction |
| --- | --- | --- |
| Conventional DBTL | Adaptive public DBTL baseline | Physical cultures drive later choices |
| Bayesian optimisation | Outcome-only adaptive optimizer | Physical observations update BO |
| Black-box digital twin | Virtual search without modular interface | Virtual evaluations are not cultures |
| Modular hybrid, no biosensors | Regulation-metabolism interface without reporter benefit | Parallel post-calibration verification |
| Biosensor modular hybrid | Canonical full-reporter twin | Reporter-informed virtual search plus exact verification |


## Result 1: Does the Digital Twin Reduce Total Physical Experiments?

The first analysis asked a resource-efficiency question: how many physical cultures are required before the engineering threshold is reached? This strict cultures-to-target endpoint did not support total physical-culture reduction versus conventional DBTL. For the moderate target, conventional DBTL attained target in 0.533 of campaigns, while the canonical full-reporter hybrid attained 0.483. The hybrid median censored cultures-to-target was 25.0 and median censored rounds-to-target was 4.0; mean cultures saved versus conventional was -0.3. For the hard target, the hybrid attained 0.150 versus 0.367 for conventional and saved -1.9 cultures versus conventional.

The quality result was different: mean best-utility delta versus conventional was small and positive at 0.003432. Thus, the strong claim that the digital twin reduces total physical culture consumption while maintaining quality was not supported. The strict outcome was `NEUTRAL`, not a hybrid win. This result remains important because total cultures are real resources; the later wall-clock analysis does not erase that cost.

**Table 6. Strict culture-efficiency result**
| Target | Conventional target attainment | Hybrid target attainment | Hybrid median cultures | Hybrid median rounds | Cultures saved vs conventional | Quality delta vs conventional | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Moderate | 0.533 | 0.483 | 25.0 | 4.0 | -0.30 | 0.0034 | NEUTRAL |
| Hard | 0.367 | 0.150 | 25.0 | 4.0 | -1.90 | 0.0034 | NEUTRAL |


## Experimental Breadth Versus Sequential Depth

The strict cultures-to-target metric did not fully answer the original engineering question because the project was motivated by time. It treats many cultures in one parallel batch too similarly to the same number of cultures spread across several dependent rounds. The final analysis therefore separated experimental breadth from experimental depth.

Experimental breadth is the total number of physical cultures run. Experimental depth is the number of sequential biological stages whose outcomes must be observed before the next decisions can be made. One hundred cultures run simultaneously can represent approximately one biological culture-duration, whereas twenty cultures distributed across four dependent DBTL rounds can require approximately four biological culture-durations. This distinction assumes the stated high-parallel-capacity laboratory scenario; it should not be read as unlimited laboratory capacity.

The workflow histories were audited to determine whether later digital-twin proposal batches genuinely depended on previous exact wet-lab outcomes. The manifest-generation logic used placeholder observations for digital-twin proposal batches, not newly revealed exact outcomes. Therefore later hybrid proposals could have been generated before the first verification batch completed and can be collapsed into one parallel post-calibration verification stage without future-information leakage. Conventional DBTL remains genuinely adaptive because later design decisions depend on previous physical culture outcomes.

![Figure 9. Conventional DBTL versus hybrid digital-twin dependency graph.](figures/scientific_report/fig09_dependency.pdf)


## Main Result: Parallel Wall-Clock Acceleration

The corrected wall-clock analysis asked whether virtual strain x environment experimentation could replace sequential biological feedback with computation and parallel verification. This is the primary result of the final exact benchmark.

Equation (6): `Speedup = T_conventional / T_twin`

Conventional DBTL used a mean 20.67 cultures, 3.20 sequential biological stages, and 31.0 scenario days excluding compute. The biosensor-informed modular hybrid used more cultures, 23.47, but only 1.97 sequential biological stages and 17.7 scenario days. This is a 38.5% reduction in sequential biological depth, a 42.9% reduction in simulated calendar time, and a 1.75x wall-clock speedup.

The central final result is therefore topological. The hybrid digital twin did not accelerate DBTL by using fewer total cultures, finding dramatically higher-utility designs, or proving every architectural component superior. It accelerated DBTL by converting sequential biological feedback depth into parallel experimental breadth and virtual computation.

![Figure 10. Sequential biological stages by workflow.](figures/scientific_report/fig10_stages.pdf)

![Figure 11. Scenario calendar time by workflow.](figures/scientific_report/fig11_days.pdf)

![Figure 12. Total cultures versus sequential biological depth.](figures/scientific_report/fig12_pareto.pdf)

**Table 7. Conventional DBTL versus hybrid digital-twin wall-clock result**
| Metric | Conventional DBTL | Hybrid digital twin |
| --- | --- | --- |
| Moderate target attainment | 0.533 | 0.483 |
| Mean physical cultures | 20.67 | 23.47 |
| Mean sequential biological stages | 3.20 | 1.97 |
| Scenario calendar days excluding compute | 31.0 | 17.7 |
| Mean best verified quality | 1.828 | 1.884 |
| Stage reduction | reference | 38.5% |
| Calendar-time reduction | reference | 42.9% |
| Wall-clock speedup | 1.00x | 1.75x |


## Supporting Result: Best-Found Utility At Fixed Budget

A cache-only follow-up asked whether the accelerated workflow produced much worse engineering designs when the same budget was spent. This is a supporting endpoint, not a replacement for the time-to-target analysis. At the full matched 24-culture budget, conventional DBTL reached mean best verified utility 1.8808 and the full-reporter hybrid reached 1.8842; the ATP-only hybrid variant reached 1.8861, and the black-box twin control reached 1.8641. These values indicate that the wall-clock acceleration was not achieved by accepting a dramatically poorer final solution.

The same fixed-budget analysis reinforces the depth mechanism. At two biological stages, the full-reporter hybrid reached mean best utility 1.8842, while conventional DBTL reached 1.6942. At approximately 18 scenario days, the hybrid again reached 1.8842, while conventional DBTL had only completed the initial stage and reached 1.2214. These are fixed-depth and fixed-time comparisons within the frozen evaluated universe; they do not prove global optimality.

![Figure 13. Best verified utility at matched biological depth and calendar time.](figures/scientific_report/fig13_best_utility.pdf)

**Table 8. Fixed-budget best-found utility support**
| Workflow | 24-culture best utility | 2-stage best utility | 18-day best utility | Target attainment at 24 cultures |
| --- | --- | --- | --- | --- |
| Conventional DBTL | 1.8808 | 1.6942 | 1.2214 | 0.533 |
| Hybrid digital twin | 1.8842 | 1.8842 | 1.8842 | 0.483 |
| ATP-only hybrid variant | 1.8861 | 1.8861 | 1.8861 | 0.550 |
| Black-box twin control | 1.8641 | 1.8641 | 1.8641 | 0.467 |


## Reporter And Architecture Controls In The Final Benchmark

Physiological reporter branches were included in the definitive benchmark, but the final workflow did not perform live time-resolved reporter assimilation. Exact reporter values were not available during virtual search, did not update the regulatory state online, and did not drive later candidate selection from previous physical cultures. Consequently, the final benchmark should not be used as a definitive test of the independent value of live biosensors.

A bounded proxy analysis found that culture-level pathway-reporter summaries contained some additional information about eventual utility: pathway reporters improved utility RMSE by approximately 0.0194, Spearman ranking by approximately 0.0487, and top-20% recovery by approximately 0.0601 relative to nominal design-only prediction. Full reporters showed smaller positive signals. This is preliminary evidence that physiological measurements can contain useful additional information, but proper live early-time reporter assimilation remains future work.

Additional digital-twin control architectures showed similar stage-depth behavior, indicating that the present benchmark primarily validates the deployment advantage of virtual experimentation rather than a unique performance advantage of the modular architecture. This does not invalidate the hybrid result, but it bounds the architecture-specific claim.

**Table 9. Reporter information regime in the final benchmark**
| Question | Answer from final cache |
| --- | --- |
| Were reporters allowed for the biosensor method? | Yes |
| Were exact reporters available during virtual search? | No |
| Did exact reporters update regulatory state online? | No |
| Did later candidate selection use previous exact reporters? | No |
| Were early reporter time series saved? | No; reporter files contain one culture-level row |
| Pathway reporter proxy delta | RMSE -0.0194; Spearman +0.0487; top-20 +0.0601 |
| Full reporter proxy delta | RMSE -0.0115; Spearman +0.0304; top-20 +0.0393 |


## Discussion

The project progressively rejected stronger hypotheses and arrived at a narrower operational claim. Hypothesis 1, that latent state improves product prediction, was supported in simple synthetic systems but not in the real-GEM benchmark. Hypothesis 2, that increasing mechanistic realism would make state-space modeling advantageous, was not supported; accumulated product trajectories remained highly compressible. Hypothesis 3, that a learned regulatory interface can drive mechanistic metabolism, was technically supported by exact replay. Hypothesis 4, that hybrid optimization should require fewer cultures, was not supported against conventional DBTL. Hypothesis 5, that a digital twin can reduce sequential DBTL time by converting biological feedback rounds into virtual search and parallel verification, was supported under the corrected deployment assumptions, with tradeoffs.

What was not shown is as important as what was shown. The project did not show universal state-space superiority, fewer total physical cultures, universal reporter advantage, or unique hybrid superiority over every digital surrogate. The final exact benchmark also did not test live early-time reporter assimilation, so biosensor-specific conclusions must remain bounded.

What was shown is a technically viable regulation-metabolism hybrid, exact Yeast9 hidden-wet-lab benchmarking with complete solver and cache accounting, joint strain x environment virtual experimentation, and a plausible reduction in sequential DBTL wall-clock time under a stated high-parallel-capacity deployment scenario. The final positive result appears only after several easier positive stories failed. It is therefore best understood as a workflow result: the digital twin changes the dependency graph of experimentation.

![Figure 14. Supported, bounded, and unsupported hypotheses.](figures/scientific_report/fig14_hypotheses.pdf)

**Table 10. Supported, bounded, and unsupported claims**
| Claim | Status | Evidence |
| --- | --- | --- |
| Reporter supervision can help hidden-state trajectory prediction. | Supported synthetically | Experiment 2B and reporter-quality frontier. |
| State-space models are best for real-GEM product prediction. | Unsupported | State-space win fraction 0.0 in diagnostic benchmark. |
| Compact learned controls can drive exact Yeast9 replay. | Supported technically | Oracle replay B_total RMSE 2.61e-16. |
| Hybrid search dominates exact finite-pool optimization. | Unsupported | Space filling had lowest mean exact-pool regret. |
| Hybrid reduces total physical cultures versus conventional DBTL. | Unsupported | Moderate cultures saved versus conventional -0.3. |
| Hybrid reduces sequential biological depth under parallel deployment. | Supported with tradeoffs | 3.20 to 1.97 stages; 31.0 to 17.7 days. |
| Live biosensor assimilation drives the final DBTL result. | Not tested definitively | No time-resolved exact reporter assimilation in candidate selection. |


## Limitations

- The validation is entirely computational; Yeast9 dynamic pFBA is still a model, not a real culture.
- Hidden biological worlds and culture-to-culture variability are simulated.
- Calendar-time numbers depend on the specified culture-duration and operational scenario assumptions.
- The wall-clock speedup assumes sufficient parallel culture capacity for calibration and verification batches.
- The twin used slightly more total cultures and had slightly lower target-attainment probability than conventional DBTL in the moderate target analysis.
- The final benchmark did not perform live time-resolved reporter assimilation, so biosensor-specific deployment value remains unresolved.
- Modularity did not clearly outperform black-box twins on the headline stage-depth metrics.
- Product trajectories were often low-dimensional, making direct baselines strong.
- Intervention-space actionability was a recurring challenge before final redesign.
- No physical engineered yeast validation has yet been performed.

## Conclusion

This work investigated whether a hybrid regulation-metabolism digital twin could accelerate yeast bioprocess DBTL. Several stronger hypotheses were rejected: state-space models were not consistently better predictors, reporters were not universally beneficial, the hybrid did not dominate classical optimization, and total culture reduction was not demonstrated against conventional DBTL.

The final exact simulated DBTL benchmark nevertheless revealed a different operational advantage. Under the stated high-parallel-capacity scenario, the hybrid digital-twin workflow reduced mean biological stage depth from 3.20 to 1.97 stages and scenario calendar time from 31.0 to 17.7 days, corresponding to a 42.9% simulated time reduction and a 1.75x wall-clock speedup. It did so while using more total cultures, not fewer.

The strongest supported practical claim is therefore that virtual strain x environment experimentation can reduce sequential biological DBTL depth by replacing dependent physical feedback rounds with computation and parallel verification. Physical yeast validation, practical laboratory parallelization, and the independent value of live biosensor assimilation remain future questions.
