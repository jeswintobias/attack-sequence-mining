# AttackCast — Web Application Plan

> **AttackCast** — where the intrusion goes next, ranked by evidence.

The name is deliberate. Weather forecasting is the culturally understood
model for probabilistic, multi-outcome prediction: nobody hears "70% chance
of rain" and expects certainty. This model's defining property is top-1 0.369
against top-5 0.756, with 6.02 plausible successors per context, so a name
built on forecasting states the honest framing before anyone asks. A name like
"Attack Predictor" would have promised what the evidence does not support.

Status: planned, not started. Decisions here are recorded in DECISIONS_LOG as
DEC-25.

---

## 1. What the application does

An analyst knows an intrusion is underway and has observed some ATT&CK
techniques — say PowerShell execution (`T1059`) — but does not know where in the
campaign the attacker is or what is coming. The application answers three
questions, in descending order of how well our evidence supports them:

| Capability | Question answered | Backed by | Measured support |
| :--- | :--- | :--- | :--- |
| **A. Next-step prediction** | "What are they likely to do next?" | LightGBM softprob | **0.756 top-5**, 0.5386 MRR (5-fold CV) |
| **B. Stage localisation** | "Where in the kill chain are they?" | ATT&CK tactic multi-hot (DEC-20) | display only; tactics add nothing predictively |
| **C. Path reconstruction** | "In what order did this happen?" | Markov-2 transition scores + beam search | not separately evaluated |

A and B are the headline features. C is valuable but unvalidated, and the
interface must label it as exploratory.

### Why ranked possibilities, not a single answer

Top-1 accuracy is **0.369** while top-5 is **0.756**, and a reused last-2
context is followed by **6.02 distinct techniques on average** (section 3 of
`ACCURACY_INVESTIGATION.md`). Several next steps are genuinely plausible, so a
UI that commits to one answer would misrepresent the model. Presenting a ranked
set with calibrated confidence is the statistically honest design, and
communicating that uncertainty is the central interaction-design requirement
rather than a caveat bolted on afterwards.

### Explainability comes from the models that lost on accuracy

PrefixSpan rules (DEC-10), negative rules (DEC-12/13) and the tactic mapping
(DEC-20) did not improve ranking accuracy. They are, however, the only
human-readable components in the project, so they become the "why" layer behind
the LightGBM ranking:

- Markov-2: "`T1059 -> T1086` was observed 43 times in training."
- PrefixSpan: "the rule `T1078, T1059 -> T1086` fired at confidence 0.74."
- Tactics: the kill-chain stepper showing inferred position.

This turns four documented negative results into interface assets without
overstating any of them.

---

## 2. Which model is served

**The sightings model, on its own.** The user's input is an ATT&CK technique, so
the label space has to be ATT&CK techniques. The CPTC model predicts
attack-stage | service symbols and cannot consume that input.

**The two datasets are not merged.** See DEC-25 for the reasoning; in short the
ontologies are disjoint, their repeat semantics contradict each other (0% versus
88-93%), and their intrinsic difficulty differs by a factor of 2.3 in bits, so
pooling would let the easier dataset dominate and inflate apparent performance.
CPTC stays in the project as cross-dataset evidence that the method transfers
(DEC-23), and it is reported as such in the model card — not served.

---

## 3. Technology stack

### Backend

| Component | Choice | Reason |
| :--- | :--- | :--- |
| Framework | **FastAPI** | async, Pydantic v2 validation, auto-generated OpenAPI docs at `/docs` that double as a demonstration asset |
| Server | uvicorn | standard ASGI server; one process also serves the built frontend |
| Model loading | joblib | LightGBM booster plus the vocabulary and encoder artifacts |
| Storage | none | prediction is stateless; add SQLite only if query history is wanted |

### Frontend

| Layer | Choice | Reason |
| :--- | :--- | :--- |
| Framework | **React 18 + Vite + TypeScript** | fast dev loop, and a typed client mirrors the Pydantic contract |
| Styling | **Tailwind CSS** | utility-first, keeps the design system in one place |
| Components | **shadcn/ui** (Radix primitives) | keyboard and screen-reader accessible by default, which is assessable HCI work rather than decoration |
| Charts | **ECharts** | the only mainstream option that renders the transition **heatmap** properly; Recharts cannot |
| Graph | **Cytoscape.js** | purpose-built for attack-graph layouts |
| Icons | Lucide | one consistent set |
| Motion | Framer Motion, used sparingly | transitions that aid orientation, nothing decorative |

**Lower-effort alternative**, if the schedule tightens: Jinja2 templates +
Tailwind + HTMX served directly from FastAPI. No build step, but the heatmap and
graph views become materially harder.

### Design direction

Avoid the dark neon "hacker console" aesthetic — it reads as a student project
and hurts legibility. Target an analyst tool: light-first with a restrained dark
mode, a single accent colour, generous whitespace, and monospace reserved for
technique identifiers. The signature visual is a horizontal 14-step kill-chain
stepper with confidence bands, which communicates stage and uncertainty in one
glance.

---

## 4. Implementation stages

### Stage 1 — Model service layer  **[DONE 2026-10-26]**

The only stage containing genuine unknowns: nothing is currently persisted, as
every script fits a model in memory and discards it.

- `src/serve/train_artifacts.py`: fit on the 2,178-sequence development set and
  persist to `artifacts/` — LightGBM booster, feature vocabulary, label encoder,
  Markov-1 and Markov-2 tables, PrefixSpan rules, tactic map.
- Write a **model card** artifact holding the CV metrics, the paired CIs, the
  documented limitations and the dataset provenance.
- Define the Pydantic request and response contract.
- Add a regression test asserting the **loaded artifacts reproduce 0.5386
  sequence MRR**, so the served model is provably the evaluated model. Without
  this the application's numbers are unverifiable.

### Stage 2 — FastAPI backend  **[DONE 2026-10-26]**

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/predict` | ranked top-k with probabilities, back-off level, and which rules fired |
| `POST /api/reconstruct` | beam search over plausible orderings of an observed set |
| `GET /api/stage` | tactic localisation for a technique set |
| `GET /api/eda/{chart_id}` | serve the existing EDA CSVs as JSON |
| `GET /api/meta` | model card: metrics, limitations, provenance |

### Stage 3 — Frontend foundation

Vite + React + TypeScript + Tailwind + shadcn/ui. Header tab switching between
**Predict** and **Explore**. Design tokens, light and dark themes, typography
scale, and explicit loading, empty and error states.

### Stage 4 — Prediction experience

Where most of the effort belongs.

- Searchable technique multi-select showing **names, not just identifiers** —
  the single largest usability gain available, since nobody recalls that
  `T1086` is PowerShell.
- Ranked candidates with probability bars and tactic badges.
- Kill-chain stepper showing inferred position.
- Cytoscape path graph: observed path solid, predicted branches dashed.
- "Why" panel exposing Markov counts, fired PrefixSpan rules, co-occurrence lift.
- Uncertainty disclosure: top-5 coverage, Markov back-off level, an explicit
  flag when the prefix has no training support, and a standing note that
  predictions follow *reported* order rather than proven execution order.
- A worked example preloaded on first visit, so the application demonstrates
  itself without any typing.

### Stage 5 — EDA tab and shipping

- Chart gallery from the 34 existing CSV/PNG pairs in `results/eda/figures/`,
  rendered interactively where it pays: transition heatmap, length distribution,
  Zipf curve, technique frequency.
- A model-performance panel carrying the CV table, the paired CIs **and the
  negative results**. The methodological rigour is the strongest part of this
  project and should be visible in the product, not buried in a report.
- Build the frontend and serve it statically from FastAPI as a single process;
  add a Dockerfile and a README with exact run commands.

---

## 5. Risks

1. **Stage 1 absorbs the contingency.** Persisting artifacts and proving they
   reproduce the evaluated metric is more delicate than it sounds. Everything
   else depends on it.
2. **Path reconstruction is factorial.** k observed techniques admit k!
   orderings. Cap at k <= 8 with a beam width near 50, and state the cap in the
   interface.
3. **The graph view can consume the schedule.** The ranked list and the
   kill-chain stepper carry most of the value; the Cytoscape graph is third
   priority within Stage 4 and should be cut first if time runs short.
4. **Do not let the application imply certainty.** The interface will be read as
   a claim about real attacker behaviour. Every screen that shows a prediction
   needs the reported-order caveat within reach.


---

## 6. Stage 1 outcome (2026-10-26)

Built: `src/serve/contract.py` (Pydantic schemas), `src/serve/artifacts.py`
(build and load), `src/serve/predictor.py` (serving logic),
`tests/test_serve_artifacts.py` (49 checks). Artifacts land in `artifacts/`;
the 5.8 MB model binary is gitignored and rebuilt in about 65 seconds.

**The guarantee that matters.** The build re-runs the documented 5-fold
cross-validation before writing anything and refuses to continue unless it
reproduces the published figure. It reproduced **0.5386 exactly** (delta
0.0000), so the shipped configuration is provably the evaluated configuration.
`test_card_metric_matches_documentation` re-checks this from the artifact.

### Two defects found while building, both now fixed

**1. A single observed technique was out of distribution.** Every training
prefix has length >= 2, so the learned model had never seen a one-technique
input -- which is precisely the scenario the application is built around ("all
we know is that they ran PowerShell"). It returned confident nonsense: 90.6% for
`T1132` whose observed transition rate after `T1059` is 8.8%. Inputs below two
techniques now use the first-order transition distribution, declared as
`context.source = "first_order_fallback"`. Probabilities on that path match the
empirical rates exactly (36.8% stated against 859/2333 = 36.8% observed), which
the tests assert candidate by candidate.

**2. The raw model was badly over-confident.** Nothing in the project's
evaluation had ever tested calibration -- only ranking. Measured on out-of-fold
predictions, mean stated confidence was 0.468 against an observed accuracy of
0.387, for an expected calibration error of **0.0808**. A single temperature,
fitted on out-of-fold log-probabilities so it is not tuned on data the final
model also fitted, brings this to **0.0167** (temperature 1.157), with mean
confidence 0.404 against accuracy 0.387.

| Calibration | Raw | Temperature-scaled |
| :--- | ---: | ---: |
| Expected calibration error | 0.0808 | **0.0167** |
| Mean stated confidence | 0.468 | 0.404 |
| Observed accuracy | 0.387 | 0.387 |

### Residual limitation the interface must respect

Temperature scaling corrects calibration *on average*; it does not fix
individual sparse contexts. `["T1078", "T1059"]` still yields 82.2% for `T1132`
despite thin second-order support. This is why `ContextInfo.confidence` is
derived from **training support** rather than from the probability value, and
why every candidate ships its observed transition counts. Stage 4 must lead with
rank, support-based confidence and the raw counts, and treat the probability as
secondary.

### Capabilities verified end to end

- **Next-step prediction** with per-candidate evidence: order-1 and order-2
  transition counts, fired PrefixSpan rules, tactic badges, technique names
  (103 of 109 resolve; the 6 unnamed are deprecated or restructured ATT&CK ids
  and fall back to the raw identifier).
- **Stage localisation** over all 14 tactics in kill-chain order, with predicted
  probability mass per tactic, and a caveat stating that tactics are display
  only (DEC-20).
- **Path reconstruction** by beam search, reporting `exhaustive` when the beam
  covered every permutation, with a caveat that it carries no measured accuracy.
- **Technique catalogue** for the picker: 109 entries with names, tactics,
  training frequency and whether the model can predict them.


---

## 7. Stage 2 outcome (2026-10-26)

Built `src/serve/app.py` and `tests/test_api.py` (62 checks, all passing).
Run it with:

```bash
python -m uvicorn src.serve.app:app --reload   # http://127.0.0.1:8000/docs
```

`.claude/launch.json` carries an `attackcast-api` configuration for the same
thing.

### Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| GET | `/api/health` | liveness, class count, calibration temperature |
| GET | `/api/meta` | full model card including the negative results |
| POST | `/api/predict` | ranked candidates with evidence, context and caveats |
| POST | `/api/stage` | 14-tactic kill-chain localisation |
| POST | `/api/reconstruct` | beam search over orderings |
| GET | `/api/catalogue` | 109 techniques with names and tactics, for the picker |
| GET | `/api/eda` | chart index with kind and row counts |
| GET | `/api/eda/{id}` | chart data as records |
| GET | `/api/eda/{id}/image` | pre-rendered PNG fallback |

Artifacts load once in the lifespan handler, so the process fails fast at
startup if they are missing and no request pays for model loading. CORS is open
only to the Vite dev server origins. The built frontend mounts at `/` in Stage 5
and is mounted last so it can never shadow an `/api` route.

### Disclosure behaviours, enforced by tests

- Every prediction returns `caveats` and a `context` block with the back-off
  level, the first- and second-order support counts, and the techniques excluded
  as already observed. Exclusions are reported rather than silently applied.
- `context.confidence` is derived from **training support**, not from the
  probability, because temperature scaling fixes average calibration and not
  individual sparse contexts.
- `context.source` declares the first-order fallback whenever fewer than two
  techniques are supplied, and the tests assert that those probabilities equal
  the empirical transition rates candidate by candidate.
- `/api/stage` carries a caveat stating that tactics were measured not to
  improve accuracy; `/api/reconstruct` carries one stating it has no measured
  accuracy at all.
- Submitting only unknown techniques returns 422 with a pointer to
  `/api/catalogue` rather than a confident answer over nothing.

### One defect found and fixed

The EDA chart classifier originally treated any CSV with a leading unnamed
column as heatmap data, which wrongly marked the Zipf curve, the top-20 table
and three others as matrices. A matrix is now identified by its column names
largely coinciding with its row labels, which holds for the three transition and
co-occurrence matrices and for nothing else. Without this the Explore tab would
have tried to render a frequency curve as a heatmap.
