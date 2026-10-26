# AttackCast

> Where the intrusion goes next, ranked by evidence.

An analyst knows an intrusion is underway and has seen a few ATT&CK techniques,
but not where in the campaign the attacker is or what is coming. AttackCast
ranks the plausible next techniques, locates the campaign on the kill chain, and
scores possible orderings of what has already been observed — each answer
carrying the evidence and the uncertainty behind it.

The name is deliberate. Weather forecasting is the understood model for
probabilistic, multi-outcome prediction: nobody hears "70% chance of rain" and
expects certainty. Measured top-1 accuracy here is **0.369** against top-5 of
**0.756**, with **6.02** distinct techniques typically following any repeated
two-technique context. The shortlist is the result, not its first row.

---

## Quick start

```bash
python -m pip install lightgbm xgboost scikit-learn pandas numpy scipy fastapi uvicorn joblib
python src/serve/artifacts.py                   # build the model artifacts (~65s)
python -m uvicorn src.serve.app:app --reload    # http://127.0.0.1:8000/docs
```

For the interface:

```bash
npm install --prefix frontend
npm run build --prefix frontend                 # FastAPI then serves it at /
# or, for development with hot reload:
npm run dev --prefix frontend                   # http://localhost:5173
```

The artifact build **re-runs the documented cross-validation before writing
anything** and refuses to continue unless it reproduces the published figure. If
it writes, the shipped model is provably the evaluated model.

---

## What it does

| Capability | Question | Measured support |
| :--- | :--- | :--- |
| **Forecast** | What might they do next? | top-5 **0.756**, MRR **0.5386**, 5-fold CV |
| **Kill chain** | Where in the campaign are they? | display only; tactics do not improve accuracy |
| **Orderings** | In what order did this happen? | exploratory; no measured accuracy |

Each prediction returns the transition counts and mined rules behind it, how
much training support the context had, and the caveats that apply.

---

## Architecture

```
src/serve/          FastAPI service
  contract.py       Pydantic request/response schemas
  artifacts.py      artifact build (with the CV gate) and load
  predictor.py      forecasting, kill-chain localisation, beam-search orderings
  app.py            nine endpoints; serves the built frontend at /
frontend/           Vite + React + TypeScript + Tailwind
  src/components/charts/   hand-built SVG charts (no charting dependency)
artifacts/          generated; the model binary is gitignored
```

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/predict` | ranked candidates with evidence, context and caveats |
| `POST /api/stage` | 14-tactic kill-chain localisation |
| `POST /api/reconstruct` | beam search over orderings |
| `GET /api/catalogue` | 109 techniques with names and tactics |
| `GET /api/meta` | model card, including the negative results |
| `GET /api/eda[/{id}[/image]]` | exploratory-analysis charts |
| `GET /api/health` | liveness and calibration temperature |

---

## How to read the numbers

Three properties of this model shape the whole interface:

1. **The task is multi-modal.** Several next techniques are genuinely plausible,
   so the interface never commits to one. Top-5 is the figure to quote.
2. **Calibration is corrected but imperfect.** The raw model was over-confident
   (expected calibration error 0.0808). Temperature scaling fitted on
   out-of-fold predictions brings that to 0.0167, but it corrects the average
   and not individual thin contexts — so every forecast shows its training
   support, and each candidate's model estimate sits beside its observed rate.
3. **Ordering is reported, not proven.** The training data is co-occurrence
   lists from incident reports. The ordering is demonstrably non-random
   (p = 0.005), but non-random is not chronological.

A single observed technique is out of distribution — every training sequence has
at least two — so that case falls back to the first-order transition model and
says so.

---

## Tests

```bash
python tests/test_serve_artifacts.py    # 49 checks: artifacts, calibration, fallback
python tests/test_api.py                # 62 checks: endpoints, validation, disclosure
python tests/test_prefixspan.py         # custom PrefixSpan vs the reference package
python tests/test_negative_rules.py     # negative-rule mining
npm run typecheck --prefix frontend
```

---

## The research behind it

The modelling work, including what was tried and did not help, is documented in
`docs/`:

| Document | Contents |
| :--- | :--- |
| `ACCURACY_INVESTIGATION.md` | what limits accuracy, with reproduction commands |
| `DECISIONS_LOG.md` | every design decision with its rationale (DEC-01 to DEC-25) |
| `RESULTS.md` | dataset statistics and model comparisons |
| `WEB_APP_PLAN.md` | this application's design and build log |

Four approaches were measured and did not improve the model: mined pattern
features, tactic abstraction, graph embeddings with Node2Vec, and fixed-weight
ensembling. They are reported in the application itself, under Explore → The
evaluation, because an evaluation listing only successes invites the question of
what was omitted.

The held-out test split has never been loaded. Every figure quoted is
cross-validated on the development set.

---

## Data

Training data is the Center for Threat-Informed Defense sightings dataset, not
redistributed here. ATT&CK technique names and tactic mappings under
`data/reference/` are derived from the MITRE ATT&CK knowledge base.
