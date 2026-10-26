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

### Stage 3 — Frontend foundation  **[DONE 2026-10-26]**

Vite + React + TypeScript + Tailwind + shadcn/ui. Header tab switching between
**Predict** and **Explore**. Design tokens, light and dark themes, typography
scale, and explicit loading, empty and error states.

### Stage 4 — Prediction experience  **[DONE 2026-10-26]**

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

### Stage 5 — EDA tab and shipping  **[DONE 2026-10-26]**

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


---

## 8. Stage 3 outcome (2026-10-26)

`frontend/` holds a Vite 8 + React 19 + TypeScript + Tailwind 4 application.
Typecheck is clean and the production build is 74 kB gzipped.

```bash
npm install --prefix frontend
npm run dev --prefix frontend      # http://localhost:5173, proxies /api to :8000
npm run build --prefix frontend    # dist/ is then served by FastAPI at /
```

`.claude/launch.json` carries `attackcast-api` and `attackcast-web`.

### Design tokens come from a validated palette

Colours are the reference data-visualisation palette, re-checked with the
palette validator against these exact surfaces. All six checks pass in both
modes; the light mode carries a contrast warning on aqua, yellow and magenta,
all below 3:1 on the light surface.

| Check | Light | Dark |
| :--- | :--- | :--- |
| Lightness band | pass | pass |
| Chroma floor | pass | pass |
| CVD separation (worst adjacent) | pass, dE 9.1 | pass, dE 8.4 |
| Normal-vision floor | pass, dE 19.6 | pass, dE 19.3 |
| Contrast vs surface | **warn** on 3 slots | pass |

**Binding consequence for Stage 5:** any chart using the aqua, yellow or
magenta slots must ship visible direct labels or a table view. That is the
relief rule, not a preference.

The accent is categorical slot 1, which is also the sequential hue, so the
interface and the charts never disagree about what "primary" looks like.

### Confidence is deliberately not a traffic light

High confidence is not "good" and low confidence is not "danger", so a
red/amber/green scale would miscommunicate. Confidence renders as three ordinal
steps of the single sequential blue, no step lighter than 250 on the light
surface or darker than 600 on dark, and the text label is always present so
colour is reinforcement rather than the only channel. Amber and red are reserved
strictly for warnings and errors, and every status use pairs an icon with a
label because two status steps sit below 3:1 on light by design.

### What was built

- **Design tokens** in `src/styles.css`: surfaces, ink, hairlines, the accent,
  the sequential ramp, confidence steps, status colours and the eight
  categorical slots, defined for both themes and exposed to Tailwind as roles.
- **Theme switching** with light, dark and system. The resolved theme is stamped
  on `<html>` in `index.html` before first paint, so there is no flash; "system"
  keeps following the OS while selected.
- **Header** implementing the ARIA tabs pattern by hand: roving tabindex,
  arrow-key, Home and End navigation, `aria-controls` wired to the panels.
  Verified in the browser: ArrowLeft moves both selection and focus, End jumps
  to the last tab.
- **Typed API client** mirroring the Pydantic contract, with `ApiError`
  normalising FastAPI's two `detail` shapes into one readable message and
  carrying actionable hints (an unreachable API suggests the uvicorn command; a
  503 suggests rebuilding artifacts).
- **State components** for loading, error and empty, plus a skip link and
  visible focus rings throughout.
- **Model strip** in the chrome, fed by `/api/meta`: top-5 0.756, top-1 0.369,
  MRR 0.5386, 101 classes, 2,178 sequences, and "test split never used". Putting
  the model card in the chrome means no screen can show a prediction without its
  provenance one glance away.
- **A standing footer caveat** that predictions rank *reported* order and that
  the shortlist, not the top answer, is the thing to read.

### Verified in the browser

Both themes render correctly, the Explore tab lists all 18 EDA charts with the
right kind classification (3 heatmaps, 5 series, 10 tables), and the keyboard
tab pattern behaves. The Forecast tab is an empty state until Stage 4.


---

## 9. Stage 4 outcome (2026-10-26)

The Forecast tab is complete. Typecheck clean, production bundle 79 kB gzipped.

### The organising decision: never show the model's number alone

Stage 1 measured that temperature scaling corrects calibration *on average*
(expected calibration error 0.0808 to 0.0167) but does not repair individual
sparse contexts. The interface is built around that finding rather than in spite
of it. Each candidate row places the model estimate **directly beside the
observed transition rate**, so a disagreement is visible rather than hidden.

The worked example shows exactly why this matters. For `T1078 -> T1059` the top
candidate `T1132` reads:

```
1  T1132  Data Encoding  [Command and Control]
   ████████████████████████  82%   observed 8.8%

   T1078 → T1059 → T1132   seen 0 of 1 times    (0.0%)
   T1059 → T1132           seen 205 of 2,333    (8.8%)
```

An analyst can see at a glance that the two-technique context was observed
**once**, that the model is extrapolating, and that the one-technique evidence
says 8.8%. The Confidence panel reinforces it: "Only the last technique has
usable support; the two-technique context is too thin."

### What was built

- **Technique picker** — an ARIA combobox searching **names, identifiers and
  tactics**, so "PowerShell" finds `T1059`. Arrow keys move the active option,
  Enter commits, Escape closes, Backspace on an empty query removes the last
  chip. With no query it offers the techniques training actually supports, ranked
  by frequency. Chips are numbered because the model consumes an ordered prefix,
  not a set.
- **Ranked candidates** — rank is the most prominent element, since rank is what
  the evaluation measured. Probability bars are scaled against the top candidate
  so the ranking stays readable when the distribution is flat. Each row expands
  to show order-1 and order-2 transition counts and any mined PrefixSpan rules
  that fired. A footer states the shortlist's coverage and repeats that measured
  top-5 accuracy is 0.756.
- **Kill-chain stepper** — fourteen tactics in order, with two deliberately
  separate channels: a filled node means the prefix touched that tactic, and a
  bar carries the share of predicted mass. Observation and prediction are
  different claims and are not merged into one ramp. Labels are abbreviated with
  full names on hover, and the whole figure carries a sentence-form `aria-label`.
- **Confidence panel** — support counts for both contexts, the back-off
  explanation in plain language, the excluded already-observed techniques with
  the reason, and a warning when the single-technique fallback is in use.
- **Possible orderings** — beam-search results, labelled exploratory, stating
  whether the search was exhaustive.
- **Caveats** — the model card's seven caveats, collapsed but one click away.
- **Worked example preloaded** (`T1078`, `T1059`) so the application
  demonstrates itself without anyone typing an identifier from memory.

### One layout defect found and fixed

The kill chain was first placed in the narrow left column of the two-column
grid, where fourteen steps need about 812 px and had roughly 470 px. It clipped
at twelve tactics and scrolled horizontally — unacceptable for the signature
visual. It now spans the full row below the grid. Verified in the browser:
`scrollWidth` 1062 equals `clientWidth` 1062, so all fourteen steps fit without
scrolling at desktop width, and the container still scrolls gracefully on narrow
viewports.

### Not built

The Cytoscape path graph was third priority within this stage and is deferred.
The ranked list and the kill-chain stepper carry the value; the graph would add
visual interest rather than information, and the stage is complete without it.


---

## 10. Stage 5 outcome (2026-10-26)

The Explore tab is complete and the application ships. Typecheck clean,
production bundle 84 kB gzipped.

### Charts are hand-built, not a library

ECharts was planned and dropped. The mark specifications this project follows
are precise — bars capped at 24px with a 4px rounded data-end square at the
baseline, 2px lines, markers of at least 8px carrying a 2px surface ring, a 2px
surface gap between touching fills, hairline recessive gridlines — and meeting
them exactly through a library's theming layer is more work than drawing the
SVG. Hand-building also keeps every colour a design token, so light and dark
stay in step with no second palette, and it avoids roughly 1 MB of dependency
for three chart types.

Four components in `src/components/charts/`: horizontal bars, a line chart with
a snapping crosshair and optional log scale, a sequential heatmap, and a data
table.

### The palette constraint, honoured

Stage 3 recorded that three categorical slots sit below 3:1 on the light
surface. Rather than discharge that with extra relief, **series are restricted
to slots 1 and 2** (blue and orange), both of which clear contrast in both
modes. Any chart needing more than two series folds the remainder into the table
view. The heatmap uses the single sequential blue ramp, light to dark, with the
lightest step allowed to recede toward the surface because near-zero should look
like nothing.

Every chart has a **table view behind a toggle**, so no value is reachable only
by hovering, and the relief obligation is discharged structurally.

### Form chosen per chart, not inferred

A registry maps each chart id to a form chosen from the data's job: magnitude
comparisons become bars, the long-tailed rank-frequency distribution becomes a
log-scaled line, pairwise matrices become heatmaps, and anything whose job is
lookup stays a table. Charts absent from the registry render as tables, which is
a correct default rather than a failure.

### The evaluation is part of the product

Explore splits into **The dataset** and **The evaluation**. The second renders
the model card: headline metrics, the multi-modality explanation for why top-1
is low, macro recall, fold spread, the calibration before and after temperature
scaling, and **the four documented negative results** with their decision
references. An evaluation that lists only what worked invites the question of
what was left out, so what did not work is shown too.

### Shipping

- `README.md` covering the quick start, the architecture, how to read the
  numbers, the tests and the research behind it.
- `Dockerfile`: a two-stage build, Node for the frontend and Python 3.12 slim
  for the runtime, with `libgomp1` for LightGBM and a health check. Artifacts
  are copied rather than built in-image, because building them requires training
  data that is not redistributed — and the host build gates on reproducing the
  documented metric.
- `.dockerignore` excluding the datasets, caches and build outputs.

### Verified in the browser

Both sections render, the chart filter works, the transition heatmap draws a
complete 20x20 matrix with its sequential legend, the first bar chart shows
757K / 175K / 108K / 7.8K with compact axis labels, and all four negative
results display with their decision references.
