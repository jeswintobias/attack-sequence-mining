"""AttackCast API — where the intrusion goes next, ranked by evidence.

Run from the repository root:
  python src/serve/artifacts.py          # once, to build artifacts
  uvicorn src.serve.app:app --reload     # http://127.0.0.1:8000/docs

Design notes
------------
Artifacts are loaded once in the lifespan handler, so a request never pays for
model loading and the process fails fast at startup if artifacts are missing.

Every prediction response carries the model card's caveats and a `context` block
describing how much training support the prefix had. That is deliberate: the
most likely way this API misleads a caller is by returning a confident-looking
probability over a context the training data barely covers, so the support
travels with the answer rather than being discoverable only via `/api/meta`.

The interactive docs at `/docs` are generated from the Pydantic contract and are
a usable demonstration surface on their own, before any frontend exists.
"""

from __future__ import annotations

import csv
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval"), str(SRC / "serve")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from artifacts import ARTIFACT_DIR
from contract import (PredictRequest, PredictResponse, ReconstructRequest,
                      ReconstructResponse, StageRequest, StageResponse)
from predictor import AttackPathPredictor

EDA_DIR = config.EDA_RESULTS_DIR / "figures"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# Human titles for the EDA assets, so the Explore tab does not show filenames.
EDA_TITLES = {
    "01_length_class_share": "Sequence length distribution (all rows)",
    "02_length_hist_ge4": "Length histogram, multi-stage sequences",
    "03_top20_techniques": "Twenty most frequent techniques",
    "04_zipf_rank_freq": "Technique frequency against rank (Zipf)",
    "05_monthly_volume": "Sighting volume by month",
    "06_length_by_year": "Sequence length by year",
    "07_top10_repeated_seqs": "Ten most repeated sequences",
    "08_first_last_techniques": "Opening and closing techniques",
    "09_cooccurrence_lift": "Technique co-occurrence lift",
    "10_transition_asymmetry": "Transition asymmetry between technique pairs",
    "10_transition_count": "Transition counts between techniques",
    "11_next_label_dist": "Next-technique label distribution",
    "12_subtechnique_depth": "Sub-technique depth",
    "13_diversity_ratio": "Technique diversity per sequence",
    "14_order_signal_pair_details": "Order-signal permutation test, pair detail",
    "14_order_signal_results": "Order-signal permutation test, results",
    "15_order_control_pair_details": "Order control (monotonic dropped), pair detail",
    "15_order_control_results": "Order control, results",
}

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (ARTIFACT_DIR / "model_card.json").exists():
        raise RuntimeError(
            f"No artifacts in {ARTIFACT_DIR}. Run: python src/serve/artifacts.py"
        )
    predictor = AttackPathPredictor()
    STATE["predictor"] = predictor
    card = predictor.artifacts.card
    print(f"AttackCast ready: {card['sequences']:,} sequences, "
          f"{card['label_classes']} classes, "
          f"seq MRR {card['evaluation']['measured']['seq_mrr']:.4f}, "
          f"calibration T={predictor.temperature:.3f}")
    yield
    STATE.clear()


app = FastAPI(
    title="AttackCast",
    summary="Where the intrusion goes next, ranked by evidence.",
    description=(
        "Ranks plausible next ATT&CK techniques from a partially observed "
        "intrusion, locates the campaign on the kill chain, and scores possible "
        "orderings of what has already been seen.\n\n"
        "**Read the shortlist, not the top answer.** Measured top-1 accuracy is "
        "0.369 against top-5 of 0.756, and a reused two-technique context is "
        "followed by 6.02 distinct techniques on average. Every response carries "
        "its caveats and the training support behind the context."
    ),
    version="0.2.0",
    lifespan=lifespan,
    contact={"name": "attack-sequence-mining"},
    license_info={"name": "See repository"},
)

# The Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _predictor() -> AttackPathPredictor:
    predictor = STATE.get("predictor")
    if predictor is None:
        raise HTTPException(503, "Model artifacts are not loaded")
    return predictor


# ---------------------------------------------------------------- schemas
class HealthResponse(BaseModel):
    status: str
    application: str
    model_loaded: bool
    label_classes: int
    calibration_temperature: float


class CatalogueEntry(BaseModel):
    technique: str
    name: str | None
    tactics: list[str]
    predictable: bool
    train_frequency: float


class EdaChart(BaseModel):
    id: str
    title: str
    kind: str = Field(
        description="'matrix' for square heatmap data, 'indexed_table' for a "
                    "table with row labels, 'table' for plain columns")
    rows: int
    has_image: bool


class EdaChartData(BaseModel):
    id: str
    title: str
    kind: str
    columns: list[str]
    records: list[dict]
    index: list[str] | None = Field(
        default=None, description="Row labels, for matrix charts only")


def _classify(fieldnames: list[str], index: list[str]) -> str:
    """Distinguish a square heatmap from a table that merely has row labels.

    A leading unnamed column alone is not enough: `04_zipf_rank_freq` has one but
    is a curve, not a heatmap. A matrix is identified by its column names being
    largely the same set as its row labels, which is true of the transition
    matrices and of nothing else here.
    """
    if not index:
        return "table"
    if fieldnames and len(set(fieldnames) & set(index)) >= max(2, len(fieldnames) // 2):
        return "matrix"
    return "indexed_table"


def _read_chart(path: Path) -> tuple[list[str], list[dict], list[str] | None, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        records = list(reader)
    index = None
    if fieldnames and fieldnames[0] == "":
        label = fieldnames[0]
        index = [row.pop(label) for row in records]
        fieldnames = fieldnames[1:]
    return fieldnames, records, index, _classify(fieldnames, index or [])


# ---------------------------------------------------------------- endpoints
@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    predictor = _predictor()
    return HealthResponse(
        status="ok",
        application="AttackCast",
        model_loaded=True,
        label_classes=predictor.artifacts.card["label_classes"],
        calibration_temperature=predictor.temperature,
    )


@app.get("/api/meta", tags=["meta"],
         summary="Model card: metrics, calibration, limitations, provenance")
def meta():
    """The full model card.

    Includes the documented negative results. They are part of the record: a
    reader should be able to see what was tried and did not work, not only what
    shipped.
    """
    return _predictor().artifacts.card


@app.post("/api/predict", response_model=PredictResponse, tags=["predict"],
          summary="Rank plausible next techniques")
def predict(request: PredictRequest):
    """Rank the techniques that might come next.

    Fewer than two observed techniques routes to the first-order transition
    model, because every training prefix has length two or more; the response
    declares this in `context.source`. Techniques already observed are excluded,
    which is a property of the training data's deduplication rather than of real
    attacker behaviour.
    """
    predictor = _predictor()
    unknown = [t for t in request.observed if t not in predictor.known]
    if len(unknown) == len(request.observed):
        raise HTTPException(
            422,
            f"None of the submitted techniques appear in the training vocabulary: "
            f"{unknown}. See /api/catalogue for the {len(predictor.vocabulary)} "
            f"techniques this model knows.",
        )
    return predictor.predict_next(request.observed, top_k=request.top_k)


@app.post("/api/stage", response_model=StageResponse, tags=["predict"],
          summary="Locate the campaign on the ATT&CK kill chain")
def stage(request: StageRequest):
    """Kill-chain position from observed and predicted tactics.

    Display only. Tactic features were measured not to improve prediction
    accuracy, so this orients an analyst rather than adding evidence.
    """
    return _predictor().localise_stage(request.observed)


@app.post("/api/reconstruct", response_model=ReconstructResponse, tags=["predict"],
          summary="Score plausible orderings of an observed technique set")
def reconstruct(request: ReconstructRequest):
    """Beam search over orderings of an unordered observation set.

    There are k! orderings, so the search is bounded. The response reports
    whether the beam was wide enough to cover every permutation. Unlike
    next-step prediction, this capability has no measured accuracy.
    """
    return _predictor().reconstruct_path(
        request.observed, top_n=request.top_n, beam_width=request.beam_width)


@app.get("/api/catalogue", response_model=list[CatalogueEntry], tags=["reference"],
         summary="Techniques the model accepts, with names and tactics")
def catalogue():
    """Everything the technique picker needs, including names.

    Names matter for usability: nobody recalls that T1086 is PowerShell.
    """
    return _predictor().catalogue()


@app.get("/api/eda", response_model=list[EdaChart], tags=["explore"],
         summary="List the available exploratory-analysis charts")
def eda_index():
    charts = []
    for path in sorted(EDA_DIR.glob("*.csv")):
        fieldnames, records, index, kind = _read_chart(path)
        charts.append(EdaChart(
            id=path.stem,
            title=EDA_TITLES.get(path.stem, path.stem.replace("_", " ")),
            kind=kind,
            rows=len(records),
            has_image=(EDA_DIR / f"{path.stem}.png").exists(),
        ))
    return charts


@app.get("/api/eda/{chart_id}", response_model=EdaChartData, tags=["explore"],
         summary="Chart data as records")
def eda_chart(chart_id: str):
    path = EDA_DIR / f"{chart_id}.csv"
    if not path.exists() or path.parent != EDA_DIR:
        raise HTTPException(404, f"No such chart: {chart_id}")
    fieldnames, records, index, kind = _read_chart(path)
    return EdaChartData(
        id=chart_id,
        title=EDA_TITLES.get(chart_id, chart_id.replace("_", " ")),
        kind=kind,
        columns=fieldnames,
        records=records,
        index=index,
    )


@app.get("/api/eda/{chart_id}/image", tags=["explore"],
         summary="Pre-rendered chart image, as a fallback")
def eda_image(chart_id: str):
    path = EDA_DIR / f"{chart_id}.png"
    if not path.exists() or path.parent != EDA_DIR:
        raise HTTPException(404, f"No image for chart: {chart_id}")
    return FileResponse(path, media_type="image/png")


# The built frontend is served from the same process in Stage 5. Mounted last so
# it never shadows an /api route, and only when a build is present.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
