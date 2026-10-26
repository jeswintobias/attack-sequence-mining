"""Tests for the AttackCast API.

Run from the repository root:
  python tests/test_api.py

Uses FastAPI's TestClient, so no server needs to be running. The lifespan
handler runs inside the client context, which also exercises artifact loading.

These tests guard the contract the frontend will be written against, plus the
disclosure behaviours that keep the API honest: caveats on every prediction,
training support reported alongside probabilities, and the out-of-distribution
fallback declared rather than hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval"), str(SRC / "serve")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient

from app import app

FAILURES: list[str] = []


def check(condition: bool, message: str):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def test_meta(client):
    print("\nmeta endpoints")
    health = client.get("/api/health")
    check(health.status_code == 200, "health returns 200")
    body = health.json()
    check(body["application"] == "AttackCast", "application is named AttackCast")
    check(body["model_loaded"] is True, "model reports loaded")
    check(body["calibration_temperature"] > 1.0,
          f"calibration temperature exposed ({body['calibration_temperature']})")

    card = client.get("/api/meta")
    check(card.status_code == 200, "meta returns 200")
    body = card.json()
    check(body["application"] == "AttackCast", "card carries the application name")
    check(abs(body["evaluation"]["measured"]["seq_mrr"] - 0.5386) < 0.0015,
          "card exposes the evaluated metric")
    check(body["test_data_loaded"] is False, "card states the test split was unused")
    check(len(body["known_negative_results"]) >= 4,
          "card publishes the negative results")
    check(body["calibration"]["ece_after"] < body["calibration"]["ece_before"],
          "card shows calibration improved")


def test_predict(client):
    print("\nPOST /api/predict")
    response = client.post("/api/predict",
                           json={"observed": ["T1078", "T1059", "T1082"], "top_k": 5})
    check(response.status_code == 200, "returns 200 for a valid prefix")
    body = response.json()
    check(len(body["candidates"]) == 5, "returns top_k candidates")
    probabilities = [c["probability"] for c in body["candidates"]]
    check(probabilities == sorted(probabilities, reverse=True), "ranked descending")
    check(body["context"]["source"] == "learned_model",
          "three techniques use the learned model")
    check(body["context"]["confidence"] in ("high", "medium", "low"),
          f"support-based confidence reported ({body['context']['confidence']})")
    check(len(body["caveats"]) >= 4, "caveats returned with the prediction")
    check(all(c["technique"] not in {"T1078", "T1059", "T1082"}
              for c in body["candidates"]),
          "observed techniques are excluded from candidates")
    check(body["context"]["excluded_already_observed"] != [],
          "exclusions are reported, not silent")
    names = [c for c in body["candidates"] if c.get("name")]
    check(len(names) >= 3, f"candidates carry names ({len(names)}/5)")


def test_predict_validation(client):
    print("\nrequest validation")
    check(client.post("/api/predict", json={"observed": []}).status_code == 422,
          "empty observation list is rejected")
    check(client.post("/api/predict",
                      json={"observed": ["powershell"]}).status_code == 422,
          "non-ATT&CK identifiers are rejected")
    check(client.post("/api/predict",
                      json={"observed": ["T1059"], "top_k": 99}).status_code == 422,
          "top_k above the cap is rejected")
    unknown = client.post("/api/predict", json={"observed": ["T9999"]})
    check(unknown.status_code == 422,
          "an entirely unknown vocabulary is rejected with guidance")
    check("catalogue" in unknown.json()["detail"],
          "the rejection points at /api/catalogue")
    lower = client.post("/api/predict",
                        json={"observed": ["t1078", "t1059.001"], "top_k": 3})
    check(lower.status_code == 200,
          "lowercase and sub-technique input is normalised, not rejected")


def test_predict_fallback(client):
    print("\nout-of-distribution fallback")
    response = client.post("/api/predict", json={"observed": ["T1059"], "top_k": 5})
    check(response.status_code == 200, "single technique is accepted")
    body = response.json()
    check(body["context"]["source"] == "first_order_fallback",
          "fallback is declared in the response")
    check(body["context"]["observed_count"] == 1, "observed count reported")
    for candidate in body["candidates"]:
        order1 = next((t for t in candidate["transitions"] if t["order"] == 1), None)
        if order1:
            check(abs(candidate["probability"] - order1["probability"]) < 0.01,
                  f"{candidate['technique']} probability equals its empirical rate")


def test_stage(client):
    print("\nPOST /api/stage")
    response = client.post("/api/stage", json={"observed": ["T1078", "T1059"]})
    check(response.status_code == 200, "returns 200")
    body = response.json()
    check(len(body["tactics"]) == 14, "all 14 tactics returned")
    check([t["index"] for t in body["tactics"]] == list(range(14)),
          "tactics in kill-chain order")
    total = sum(t["predicted_weight"] for t in body["tactics"])
    check(abs(total - 1.0) < 1e-6 or total == 0.0, "weights form a distribution")
    check(any("DEC-20" in c for c in body["caveats"]),
          "stage view discloses that tactics do not improve accuracy")


def test_reconstruct(client):
    print("\nPOST /api/reconstruct")
    response = client.post("/api/reconstruct",
                           json={"observed": ["T1059", "T1078", "T1082"], "top_n": 3})
    check(response.status_code == 200, "returns 200")
    body = response.json()
    check(len(body["orderings"]) == 3, "returns top_n orderings")
    check(body["exhaustive"] is True, "3 techniques with default beam is exhaustive")
    for ordering in body["orderings"]:
        check(sorted(ordering["sequence"]) == ["T1059", "T1078", "T1082"],
              "each ordering permutes the input")
    check(client.post("/api/reconstruct",
                      json={"observed": ["T1059"]}).status_code == 422,
          "fewer than two techniques is rejected")
    check(client.post("/api/reconstruct",
                      json={"observed": [f"T10{i:02d}" for i in range(9)]}
                      ).status_code == 422,
          "more than eight techniques is rejected")


def test_catalogue(client):
    print("\nGET /api/catalogue")
    response = client.get("/api/catalogue")
    check(response.status_code == 200, "returns 200")
    body = response.json()
    check(len(body) > 100, f"covers the vocabulary ({len(body)})")
    powershell = [row for row in body if row["technique"] == "T1059"]
    check(bool(powershell), "T1059 present")
    check(powershell[0]["name"] == "Command and Scripting Interpreter",
          "T1059 resolves to its ATT&CK name")
    check(all("tactics" in row and "predictable" in row for row in body),
          "entries carry tactics and predictability")


def test_eda(client):
    print("\nGET /api/eda")
    index = client.get("/api/eda")
    check(index.status_code == 200, "index returns 200")
    charts = index.json()
    check(len(charts) >= 15, f"all EDA assets listed ({len(charts)})")
    kinds = {c["kind"] for c in charts}
    check(kinds <= {"table", "indexed_table", "matrix"},
          f"kinds are from the declared set ({kinds})")
    matrices = [c for c in charts if c["kind"] == "matrix"]
    check(len(matrices) == 3,
          f"only genuinely square data is marked matrix ({len(matrices)})")
    check(all(c["id"].startswith(("09", "10")) for c in matrices),
          "the matrices are the co-occurrence and transition charts")

    data = client.get("/api/eda/10_transition_count")
    check(data.status_code == 200, "matrix chart data returns 200")
    body = data.json()
    check(len(body["columns"]) == len(body["index"]),
          "transition matrix is square")
    check(body["title"] != body["id"], "chart carries a human title")

    zipf = client.get("/api/eda/04_zipf_rank_freq").json()
    check(zipf["kind"] == "indexed_table",
          "a curve with row labels is not mistaken for a heatmap")

    check(client.get("/api/eda/does_not_exist").status_code == 404,
          "unknown chart returns 404")
    check(client.get("/api/eda/01_length_class_share/image").status_code == 200,
          "chart image is served")


def test_openapi(client):
    print("\nOpenAPI surface")
    spec = client.get("/openapi.json")
    check(spec.status_code == 200, "openapi.json is served")
    body = spec.json()
    check(body["info"]["title"] == "AttackCast", "spec is titled AttackCast")
    paths = set(body["paths"])
    expected = {"/api/health", "/api/meta", "/api/predict", "/api/stage",
                "/api/reconstruct", "/api/catalogue", "/api/eda",
                "/api/eda/{chart_id}", "/api/eda/{chart_id}/image"}
    check(expected <= paths, f"all endpoints documented ({len(expected)})")


def main():
    print("=" * 70)
    print("ATTACKCAST API TESTS")
    print("=" * 70)
    with TestClient(app) as client:
        test_meta(client)
        test_predict(client)
        test_predict_validation(client)
        test_predict_fallback(client)
        test_stage(client)
        test_reconstruct(client)
        test_catalogue(client)
        test_eda(client)
        test_openapi(client)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        sys.exit(1)
    print("ALL API TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
