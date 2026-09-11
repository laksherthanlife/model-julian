#!/usr/bin/env python3
"""Build the research notebooks as readable, phase-by-phase records.

The notebooks deliberately keep orchestration visible.  Shared modules remain
the home for the Yeast9 solver and reusable model classes, but a reader can see
the experiment contract, data hand-offs, model inputs/outputs, ranking, and
analysis without opening a monolithic ``run_all`` wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


COMMON_SETUP = '''# Run this notebook from the repository root.
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

REGENERATE = False  # Cached artifacts are the default; no expensive solve runs implicitly.

def artifact(relative_path: str) -> Path:
    """Fail with a useful message rather than silently replacing evidence."""
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing cached artifact: {path}")
    return path

def show(frame, n=8):
    # `print` keeps this notebook usable in a plain Python kernel as well as Jupyter.
    print(frame.head(n).to_string(index=False))
    print(f"{len(frame):,} rows × {len(frame.columns):,} columns")
'''


SPECS = [
    {
        "file": "01_problem_and_generator_validation.ipynb",
        "title": "1. Problem framing and controlled-generator validation",
        "question": "Can observable culture trajectories support a useful digital twin in a bounded synthetic system, before asking any learner to infer hidden state?",
        "generator": "The fixed-environment generator simulates complete cultures under temperature, pH, and dissolved-oxygen conditions. It emits observable biomass/product trajectories and retains synthetic latent state only as a lockbox evaluation target.",
        "model": "No learned model is the primary object here. The experiment first verifies that the generator has nontrivial trajectories, environment effects, and declared failure worlds.",
        "screen": "There is no strain ranking in this phase. The screen is a generator audit: does each world meet its declared trajectory and split checks?",
        "comparison": "Compare the three generator worlds and their audit metrics, rather than comparing a learner to an oracle.",
        "analysis": "A passing audit makes the synthetic benchmark usable; it does not establish biological realism or wet-lab validity.",
        "module": "run_fixed_environment_validation",
        "config": '''from yeast_validation import run_fixed_environment_validation as fixed

cfg = fixed.Config()
print("culture seeds:", cfg.dataset_seeds)
print("model seeds:", cfg.model_seeds)
print("observable inputs:", ["temperature", "pH", "DO", "biomass", "product"])
print("lockbox quantities: synthetic latent state and generator internals")
''',
        "generate": '''# This is the explicit generator hand-off: a culture-level design becomes trajectories.
# `fast=True` is for notebook inspection only; it does not overwrite cached evidence.
traj, parameters, manifest = fixed.generate_dataset(cfg.dataset_seeds[0], cfg)
show(traj)
''',
        "artifacts": [("metrics", "data/canonical_metrics.csv")],
        "group": "metrics.groupby('experiment', dropna=False).mean(numeric_only=True)",
        "rerun": "fixed.run_all(fast=False, write=True)",
    },
    {
        "file": "02_state_space_and_biosensor_validation.ipynb",
        "title": "2. State-space recovery and biosensor validation",
        "question": "Can a deployable state-space model infer future-relevant state, and does a mechanistically relevant reporter resolve product-only ambiguity?",
        "generator": "The same controlled cultures are partitioned by complete culture. Product, biomass, environment, and declared reporter channels are observable; latent generator state is lockboxed.",
        "model": "The notebook makes the training contract explicit: fit only on training cultures, pass observable history into the state-space learner, and score held-out complete cultures.",
        "screen": "Reporter conditions are screened as experimental observation interfaces: informative, shuffled, noisy, missing, and irrelevant channels are all retained.",
        "comparison": "Compare product-only and reporter-informed models on identical culture splits; controls diagnose whether an apparent reporter gain is information rather than leakage.",
        "analysis": "The result is conditional on this observation interface. A reporter is useful only if it improves held-out future prediction beyond the negative controls.",
        "module": "run_fixed_environment_validation",
        "config": '''from yeast_validation import run_fixed_environment_validation as fixed

cfg = fixed.Config()
print("split unit: complete culture")
print("training inputs: environment + observed trajectory/reporter history")
print("evaluation target: future product/biomass; latent state is evaluation-only")
''',
        "generate": '''# Generate a small inspection dataset, then expose the train/validation/test hand-off.
traj, parameters, manifest = fixed.generate_dataset(cfg.dataset_seeds[0], cfg)
show(traj)
print(traj.groupby("split").size() if "split" in traj else "Split labels are assigned by the experiment runner.")
''',
        "artifacts": [("metrics", "data/canonical_metrics.csv")],
        "group": "metrics.groupby(['experiment', 'model'], dropna=False).mean(numeric_only=True)",
        "rerun": "fixed.run_all(fast=False, write=True)",
    },
    {
        "file": "03_hybrid_gem_model.ipynb",
        "title": "3. Hybrid dynamic-GEM state-space model",
        "question": "Does the state-space approach remain useful when metabolic trajectories arise from a dynamic Yeast9/GEM constraint system rather than the simple controlled ODE?",
        "generator": "At every interval, the generator applies environmental and dynamic metabolic constraints, solves staged LP/pFBA, then integrates biomass and β-carotene forward. Fluxes, bounds, and causal metabolic state are lockboxed.",
        "model": "The hybrid learner consumes only environment plus observable biomass/product/reporter history. Its control head produces six compact metabolic controls; the frozen GSM surrogate maps controls and environment to flux predictions during training.",
        "screen": "This phase screens model variants on held-out whole cultures, with the same observable interface for every deployable comparison.",
        "comparison": "Compare product and biomass error by experiment and model class, never by giving a competitor lockbox fluxes or bounds.",
        "analysis": "This is the first phase where the ML-to-GSM hand-off is explicit: learned controls constrain a metabolic solve; they are not themselves biological ground truth.",
        "module": "run_dfba_state_machine_validation",
        "config": '''from yeast_validation import run_dfba_state_machine_validation as dfba
from yeast_validation import gem_gsm_surrogate as surrogate

cfg = dfba.DFBAConfig()
print("GSM control head:", surrogate.CONTROL_COLUMNS)
print("GSM/surrogate flux outputs:", surrogate.SURROGATE_OUTPUT_COLUMNS)
print("observable: environment, biomass, product, selected reporters")
print("lockbox: reaction bounds, fluxes, burden states, enzyme capacities")
''',
        "generate": '''# The data generator is intentionally visible: it returns each hand-off table.
dataset = dfba.generate_dataset(cfg.dataset_seeds[0], cfg, fast=True)
traj, reporters, fluxes, manifest = dataset[:4]
show(manifest[[c for c in ["culture_id", "split_type", "temperature", "pH", "DO"] if c in manifest]])
show(traj)
''',
        "artifacts": [("metrics", "results/experiment_3b_state_machine_dfba/summary_metrics.csv")],
        "group": "metrics.groupby(['model', 'split'], dropna=False).mean(numeric_only=True)",
        "rerun": "dfba.run_all(fast=False, write=True, backend=None)",
    },
    {
        "file": "04_model_comparisons_and_distillation.ipynb",
        "title": "4. Fair model comparison and reporter-grounded distillation",
        "question": "Can a deployable learner inherit useful metabolic structure from a computational teacher without hidden-state leakage, and does that improve on matched baselines?",
        "generator": "The dynamic-GEM cultures from the preceding phase provide observable trajectories and a separately declared teacher target set. The notebook keeps train/test culture identity and channel availability visible.",
        "model": "Teacher and student are separate: the teacher/surrogate may use declared metabolic targets in training; the deployed student gets only the observation interface. The student’s control trajectory is then evaluated through the GSM surrogate or exact replay.",
        "screen": "Candidate/model screening uses the same split and observation contract. Cached exact replay is kept separate from cheap surrogate ranking.",
        "comparison": "Compare student variants, baselines, clean-teacher ablations, and observation-noise controls—not a pooled metric from unequal inputs.",
        "analysis": "A good surrogate ranking is not an exact-GEM guarantee. The notebook therefore presents trajectory metrics and replay/verification evidence as distinct claims.",
        "module": "run_reporter_grounded_hybrid_distillation",
        "config": '''from yeast_validation import run_reporter_grounded_hybrid_distillation as distill
from yeast_validation import gem_gsm_surrogate as surrogate

print("student-visible controls:", distill.GEM_APPLIED_CONTROL_COLUMNS)
print("student-visible reporters:", distill.REPORTER_COLUMNS)
print("surrogate output channels:", surrogate.SURROGATE_OUTPUT_COLUMNS)
''',
        "generate": '''# Load the declared training interface rather than calling an opaque experiment wrapper.
targets = pd.read_csv(artifact("data/gem_dynamic_capacity_constraints.csv"))
visible = [c for c in distill.REPORTER_COLUMNS + ["temperature", "pH", "DO"] if c in targets]
show(targets[[c for c in ["culture_id", "interval_index"] + visible if c in targets]])
''',
        "artifacts": [("metrics", "data/hybrid_student_metrics.csv")],
        "group": "metrics.groupby(['model', 'split'], dropna=False).mean(numeric_only=True)",
        "rerun": "raise RuntimeError('Run the explicitly parameterized distillation pipeline from its documented configuration; this notebook does not hide it behind run_all().')",
    },
    {
        "file": "05_dbtl_decision_benchmarks.ipynb",
        "title": "5. DBTL decision benchmark",
        "question": "Under a fixed intervention library, candidate pool, and exact-evaluation budget, does the digital twin select better strain–environment designs than conventional routes?",
        "generator": "Each candidate explicitly joins an environment with a sparse strain-edit specification. Virtual scoring is a cheap screen; the exact GSM/dynamic oracle is the separate verifier.",
        "model": "The learned model produces a control trajectory and virtual predicted outcome. It does not get to change the candidate pool, budget, or exact objective after seeing the answer.",
        "screen": "Rank the same declared pool for every method. Record shortlist size, edit mapping, solver calls, and which candidates advance to exact replay.",
        "comparison": "Compare exact verified outcomes under equal budgets—not raw surrogate scores. Acceptance and accounting tables make the comparison auditable.",
        "analysis": "This is a bounded decision-quality result. It says nothing directly about a wet-lab strain until the oracle and observation interface are biologically calibrated.",
        "module": "design_benchmark_exact",
        "config": '''pool = pd.read_csv(artifact("data/design_benchmark_candidate_pool.csv"))
print("candidate count:", len(pool))
print("candidate fields:", list(pool.columns))
show(pool)
''',
        "generate": '''# A candidate is the generator input for this phase: explicit environment + edit list.
candidate_columns = [c for c in ["candidate_id", "temperature", "pH", "DO", "edits", "candidate_json"] if c in pool]
show(pool[candidate_columns])
''',
        "artifacts": [("acceptance", "data/design_benchmark_acceptance.csv")],
        "group": "acceptance.groupby(['status_type', 'regime_id', 'winner_method', 'status'], dropna=False).size().rename('n').reset_index()",
        "rerun": "raise RuntimeError('Exact DBTL replay is intentionally not a one-line notebook side effect. Use the recorded candidate pool and declared evaluator configuration.')",
    },
    {
        "file": "06_generator_realism_and_rxncon.ipynb",
        "title": "6. Generator realism and rxncon → GSM transfer",
        "question": "Do the conclusions survive a richer causal chain—realized process variation, regulatory/rxncon state, GSM controls, and Yeast9—rather than one narrowly structured generator?",
        "generator": "The nominal environment is observable. Realized process history drives rxncon/regulatory state, which produces the hidden GSM-interface controls used in the Yeast9 rollout. Bounds, fluxes, and hidden state remain lockboxed.",
        "model": "Learning is evaluated only from the declared observable trajectories/reporters. The hidden rxncon-to-GSM interface is not made into a training feature merely because the generator can export it.",
        "screen": "Generator gates quantify interface complexity, process-variation effects, and dense-versus-sparse GSM rollout consistency before any performance claim.",
        "comparison": "Compare baseline, process-variation, and shifted-world outcomes with their controls and failure modes visible.",
        "analysis": "This widens structural validity. It is still synthetic and is not a claim that rxncon is a complete yeast regulatory ground truth.",
        "module": "run_rxncon_process_variability_experiment",
        "config": '''from yeast_validation import run_rxncon_process_variability_experiment as process

print("observable nominal inputs: temperature, pH, DO, time, reporters")
print("lockbox chain: realized process → rxncon state → GSM interface → bounds/fluxes")
print("world definitions:", [w.name for w in process.WORLDS])
''',
        "generate": '''# The named generator entry point exposes the causal hand-off in its return tables.
# The cached summary below is the primary evidence; do not launch the full campaign implicitly.
print("Process variability generator uses", process.N_TIMEPOINTS, "time points per culture.")
''',
        "artifacts": [("summary", "data/rxncon_process_variability/process_variability_main_summary_interpreted.csv")],
        "group": "summary.groupby('world', dropna=False).mean(numeric_only=True)",
        "rerun": "raise RuntimeError('The rxncon/GSM campaign is expensive. Invoke its documented, parameterized generator only after choosing a world and output directory.')",
    },
    {
        "file": "07_biosensor_information_experiments.ipynb",
        "title": "7. Biosensor information experiments",
        "question": "Which reporter channels reduce decision-relevant uncertainty or make edit effects identifiable, and which apparent gains disappear under leakage and negative controls?",
        "generator": "This phase uses rxncon/GSM cultures with reporter observations separated from hidden regulatory histories and GSM-interface controls.",
        "model": "Assimilation and edit-inference models receive only the selected reporter/history interface. The notebook explicitly separates reporter-supervision and live-biosensor-identifiability analyses.",
        "screen": "Screen reporter sets and edit hypotheses under shared cultures, with shuffled/noisy/missing/irrelevant controls retained as first-class results.",
        "comparison": "Compare uncertainty, prediction, or edit-identification metrics at the same observation budget and culture split.",
        "analysis": "A channel matters only when it changes a deployable inference or decision; correlation with product alone is not enough.",
        "module": "run_rxncon_reporter_supervision_diagnostic",
        "config": '''from yeast_validation import run_rxncon_reporter_supervision_diagnostic as diagnostic
from yeast_validation import run_rxncon_edit_identifiability_live_biosensor as ident

print("diagnostic module:", diagnostic.__name__)
print("edit-identifiability module:", ident.__name__)
print("contract: reporters observable; regulatory state and GSM controls lockboxed")
''',
        "generate": '''# Read the explicit reporter experiment record before interpreting an aggregate metric.
metrics = pd.read_csv(artifact("data/rxncon_reporter_supervision_diagnostic/reporter_diagnostic_summary_table.csv"))
show(metrics)
''',
        "artifacts": [("metrics", "data/rxncon_reporter_supervision_diagnostic/reporter_diagnostic_summary_table.csv")],
        "group": "metrics.groupby(['condition_family', 'reporter_mode', 'reporter_set'], dropna=False).mean(numeric_only=True)",
        "rerun": "raise RuntimeError('Run the reporter diagnostic and identifiability campaigns with their explicit declared arguments; they are not hidden notebook side effects.')",
    },
    {
        "file": "08_final_prospective_validation.ipynb",
        "title": "8. Frozen prospective, shifted-world, and boundary-condition validation",
        "question": "After the intervention library, candidate budgets, model choice, and evaluation protocol are frozen, does the workflow retain a decision advantage in prospective and harder-world tests?",
        "generator": "Prospective campaigns keep candidate selection separate from the exact verifier. Data-sufficiency and shifted-world suites alter the declared training amount or world, not the outcome metric after the fact.",
        "model": "The selected hybrid and conventional routes operate under their predeclared data and candidate interfaces. Exact replay remains the arbiter for candidate-level claims.",
        "screen": "Generate rankings from frozen checkpoints, select the allowed shortlist, then verify that shortlist using the declared real-GSM/hidden-world evaluator.",
        "comparison": "Compare prospective campaigns, training-set sizes, and world shifts using exact outcomes and declared accounting—not only in-sample predictive fit.",
        "analysis": "This is the strongest but still bounded result: it supports conditional computational performance inside these generators, budgets, and exact checks—not wet-lab transfer.",
        "module": "run_final_validation_prospective",
        "config": '''from yeast_validation import run_final_validation_prospective as final
from yeast_validation import run_prospective_dbtl_benchmark as prospective

print("final evaluator:", final.__name__)
print("prospective campaign definition:", prospective.__name__)
print("freeze boundary: candidate library, budgets, checkpoints, and outcome metric")
''',
        "generate": '''# The final phase begins with a frozen campaign definition, not a newly optimized pool.
summary = pd.read_csv(artifact("data/final_validation_prospective/final_validation_predictive_overfitting_summary.csv"))
show(summary)
''',
        "artifacts": [("summary", "data/final_validation_prospective/final_validation_predictive_overfitting_summary.csv")],
        "group": "summary.groupby('training_cultures_n', dropna=False).mean(numeric_only=True)",
        "rerun": "raise RuntimeError('Final prospective validation has frozen campaign inputs and distributed exact verification. Use the recorded campaign manifests rather than rerunning a hidden wrapper.')",
    },
]


def build(spec: dict) -> dict:
    cells = [
        md(f'''# {spec["title"]}

## Goal

{spec["question"]}

This notebook is a readable research record. It follows the actual hand-offs in order and loads the saved evidence by default; it does **not** hide the experiment behind a one-cell runner.'''),
        md('''## Pipeline at a glance

```text
goal → declared generator → observable/lockbox split → model setup & training
     → candidate or condition screen → matched comparison → interpretation
```

Each section below corresponds to one of these hand-offs.'''),
        code(COMMON_SETUP),
        md('''## 1. Experimental contract

The experiment has a declared observation boundary. “Observable” means the learner may use it; “lockbox” means it may be generated and audited but must not be used as a deployable feature.'''),
        code(spec["config"]),
        md(f'''## 2. Data generator

{spec["generator"]}

The next cell exposes the generator’s first concrete hand-off. It is deliberately small/inspection-only where generating the full campaign is expensive.'''),
        code(spec["generate"]),
        md(f'''## 3. Model setup and training contract

{spec["model"]}

Training is not automatically started in this notebook. The cached training/evaluation artifacts below are the evidence record; regeneration must be an intentional, parameterized action.'''),
        code('''# Make the experiment hand-off inspectable before looking at aggregate metrics.
for name, relative_path in ''' + repr([(name, path) for name, path in spec["artifacts"]]) + ''':
    path = artifact(relative_path)
    print(f"{name}: {path.relative_to(ROOT)}")
'''),
        md(f'''## 4. Screening / selection stage

{spec["screen"]}

The screen is intentionally shown separately from final verification, so a virtual score cannot be mistaken for an exact outcome.'''),
        code('''# Load the primary evidence table and inspect its schema before aggregation.
''' + "\n".join(f"{name} = pd.read_csv(artifact({path!r}))\nshow({name})" for name, path in spec["artifacts"])),
        md(f'''## 5. Matched comparison

{spec["comparison"]}'''),
        code('''# Aggregate only over fields that exist in this version of the cached record.
comparison = ''' + spec["group"] + '''
show(comparison.reset_index() if hasattr(comparison, "reset_index") else comparison)
'''),
        md('''## 6. Analysis view

The plot is intentionally generic: it exposes every numeric evidence column so the reader can select the metric relevant to the claim, rather than hard-coding an attractive subset.'''),
        code('''numeric = ''' + spec["artifacts"][0][0] + '''.select_dtypes("number")
if numeric.shape[1]:
    ax = numeric.plot(kind="box", rot=45, figsize=(11, 4), title="Cached evidence: numeric metric distribution")
    ax.set_ylabel("recorded metric value")
    plt.tight_layout()
else:
    print("This artifact has no numeric columns to plot.")
'''),
        md(f'''## 7. Interpretation, scope, and next hand-off

{spec["analysis"]}

### Reproduction boundary

The cells above reveal the inputs and artifacts without launching an expensive campaign. To regenerate, use the explicit command below only after reviewing its declared inputs and output destination.'''),
        code('''if REGENERATE:
    # This guard prevents accidental solver/campaign execution.
    ''' + spec["rerun"]),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


for spec in SPECS:
    (NB / spec["file"]).write_text(json.dumps(build(spec), indent=2) + "\n", encoding="utf-8")
    print("wrote", spec["file"])
