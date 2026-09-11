# Yeast Digital Twin Handover App

Run from the repository root:

```bash
.venv/bin/python -m webapp.ydt_app.server --port 8765
```

Open http://127.0.0.1:8765.

The packaged **Use example dataset** is a cleaned 125-culture active-rxncon
observable dataset with 49 timepoints per culture. It contains only culture ID,
time, external controls, observed product, observed biomass, and noisy reporter
columns; simulator bookkeeping and noiseless reporter truth were removed.

The app accepts CSV and TSV culture tables, maps arbitrary columns to culture
ID, time, environment/control, product, biomass, reporters, and genotype/edit
roles, validates culture-level structure, and trains several observable-data
candidate models. The current candidates include direct ridge trajectory
baselines, the repository's product-only continuous state-space model, and an
observation-conditioned observable state-space model. The observer consumes
truncated product/biomass/reporter histories with explicit masks and predicts
the remaining trajectory. It does not receive raw rxncon states, GSM controls,
fluxes, or other privileged simulator values.

Models are selected by validation cultures, while the test cultures remain
untouched. Culture IDs, not timepoint rows, are the independent split unit.
The app can export and reload a trained model JSON, reuse a saved schema, make
open-loop predictions, update predictions from a partial culture CSV, and rank
bounded virtual conditions. The prediction and observer models are not Yeast9
hybrids: exact Yeast9 verification is not exposed for arbitrary uploaded
schemas. The backend keeps an explicit metabolic adapter boundary for future
calibration.

Targeted tests:

```bash
.venv/bin/python -m pytest -q tests/test_ydt_app.py
```
