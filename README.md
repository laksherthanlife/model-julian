# Yeast Digital Twin: notebook-first synthetic validation

This repository tests a bounded scientific hypothesis: whether an observable, hybrid dynamic model can improve trajectory prediction and strain–environment design decisions in carefully specified synthetic yeast-like systems.

It is a computational validation project, not a claim that the synthetic latent variables are real yeast physiology or that performance transfers directly to wet-lab strains.

## Start here

Read the notebooks in order. They are the primary research record: each one states its question, exposes the implementation used to generate or evaluate the evidence, loads the matching cached artifact by default, and explains what motivated the next phase.

| Notebook | Scientific question |
| --- | --- |
| `01_problem_and_generator_validation.ipynb` | Is the controlled synthetic generator appropriate for a bounded test? |
| `02_state_space_and_biosensor_validation.ipynb` | Can latent state be inferred, and do biosensors resolve ambiguity? |
| `03_hybrid_gem_model.ipynb` | Does the result persist with a dynamic GEM state-space generator? |
| `04_model_comparisons_and_distillation.ipynb` | Are hybrid comparisons fair, and can a metabolic teacher be distilled without leakage? |
| `05_dbtl_decision_benchmarks.ipynb` | Do model predictions improve DBTL decisions under matched budgets? |
| `06_generator_realism_and_rxncon.ipynb` | Are complexity and external-generator validity sufficient for the benchmark? |
| `07_biosensor_information_experiments.ipynb` | Which biosensor information changes deployable uncertainty or edit inference? |
| `08_final_prospective_validation.ipynb` | Does the frozen workflow survive prospective, sparse-data, and shifted-world checks? |

## Reproduction model

Each notebook has an explicit `REGENERATE = False` cell.

- With the default value, it loads the saved evidence bundle. This is the normal readable path.
- Set it to `True` only for the marked regeneration cell. Expensive GEM, exact-replay, and rxncon work is never started implicitly.
- The regeneration cell imports and calls the same implementation functions used by the experiment; it never shells out to a hidden runner.
- Seeds, split semantics, observable/lockbox distinctions, candidate budgets, and model configurations remain in the imported functions and are described in the notebook.

## Repository layout

```text
notebooks/              Scientific narrative and executable experiment phases
src/yeast_validation/   Reusable generators, model interfaces, training and evaluation helpers
data/                   Input tables and cached generated evidence
results/                Checkpoints, exact rollouts, predictions, and larger generated outputs
artifacts/              Optional restored/exported artifact bundles (ignored by git)
tests/                  Lean unit and integration coverage for the helper interfaces
webapp/                 Separate observable-data demonstration application
```

The former command-only script wrappers have been removed. The implementation is deliberately importable from `src/yeast_validation`; notebooks are the only intended research entry point.

## Major conclusions and limitations

The evidence supports only conditional computational conclusions:

- recoverability depends on the observation interface and on whether the generator contains identifiable state;
- reporter channels can improve inference in specific information regimes, but shuffled, noisy, missing, and irrelevant controls remain essential;
- surrogate accuracy is not itself a DBTL claim, so virtual ranking is separated from declared exact verification;
- more complex GEM/rxncon and hidden-process worlds are used as robustness checks rather than silently treated as equivalent generators;
- the strongest final evidence is still bounded by synthetic assumptions, frozen candidate libraries, solver behavior, and the lack of wet-lab calibration.

Long-form scientific context and retained negative results are in [PHYSIOLOGY_QUEST_VALIDATION.md](PHYSIOLOGY_QUEST_VALIDATION.md) and [YEAST_DIGITAL_TWIN_COMPREHENSIVE_TECHNICAL_WRITEUP.md](YEAST_DIGITAL_TWIN_COMPREHENSIVE_TECHNICAL_WRITEUP.md).

## Environment and tests

Create the supplied environment, then launch Jupyter from the repository root:

```bash
.venv/bin/python -m jupyter lab
.venv/bin/python -m pytest tests
```

The optional web app remains independent of the notebooks; it uses the same cleaned state-space helper where available, but it does not expose hidden simulator truth or run expensive validation campaigns.

