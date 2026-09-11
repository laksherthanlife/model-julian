# Notebook sequence

Open these notebooks in numerical order from the repository root. Each is an end-to-end scientific phase, not a plotting dashboard.

1. `01_problem_and_generator_validation.ipynb` — controlled problem and generator validity.
2. `02_state_space_and_biosensor_validation.ipynb` — recoverability and biosensor ambiguity.
3. `03_hybrid_gem_model.ipynb` — dynamic GEM state-space interface.
4. `04_model_comparisons_and_distillation.ipynb` — fair baselines, leakage controls, and distillation.
5. `05_dbtl_decision_benchmarks.ipynb` — matched-budget design and deployment benchmarks.
6. `06_generator_realism_and_rxncon.ipynb` — complexity and external-generator checks.
7. `07_biosensor_information_experiments.ipynb` — reporter value, assimilation, and edit identifiability.
8. `08_final_prospective_validation.ipynb` — frozen prospective and boundary-condition evidence.

Every notebook:

- names the scientific question and the deployable/lockbox interface;
- imports the actual generator/model/evaluation functions from `src/yeast_validation`;
- loads cached evidence by default;
- has a visible `REGENERATE` gate with the exact function call and configuration required to reproduce expensive work;
- interprets both positive and negative results before motivating the next phase.

Do not use a notebook to invoke a command-line experiment runner. Add reusable implementation to `src/yeast_validation`, then call those functions directly from the relevant notebook.

