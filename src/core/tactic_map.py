"""Technique -> ATT&CK tactic multi-label mapping, extracted from local data.

Source
------
`data/reference/attack_technique_tactics.csv`, built by
`src/core/build_tactic_reference.py` from the 14 MITRE ATT&CK Enterprise tactic
pages. 212 techniques across all 14 tactics.

The AIT-ADS Wazuh exports also embed a `rule.mitre` block giving technique ids
with their tactics, and `extract_from_wazuh` still reads it -- but it covers
only 8 techniques, so it serves as a cross-check on the reference table rather
than as the mapping itself.

Why multi-label
---------------
A technique belongs to several tactics in general: T1078 above sits in four of
them. Treating tactics as 14 mutually exclusive buckets would force an arbitrary
choice per technique and inject label noise, so the mapping is exposed as a
multi-hot vector over tactics and never as a single tactic id.

Coverage caveat
---------------
Wazuh ships detection rules for a subset of ATT&CK, so this mapping covers only
the techniques those rules mention. `coverage_report` measures that against a
given technique vocabulary; techniques with no mapping get an all-zero vector
plus an explicit `tactic_unknown` flag, so a model can distinguish "no tactic
applies" from "tactic not recorded here".
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

MITRE_BLOCK = re.compile(r'"mitre":\s*(\{.*?\})\s*,\s*"', re.DOTALL)
CACHE_PATH = config.PROCESSED_DATA_DIR / "technique_tactics.csv"
REFERENCE_PATH = config.BASE_DIR / "data" / "reference" / "attack_technique_tactics.csv"

# The 14 ATT&CK Enterprise tactics, in kill-chain order. Fixed here so the
# multi-hot column order is stable across runs and folds.
TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]


def extract_from_wazuh(ait_dir: Path | None = None) -> pd.DataFrame:
    """Scan the Wazuh exports for technique/tactic pairs.

    Files are tens of megabytes each, so blocks are matched with a regex rather
    than parsing every alert.
    """
    ait_dir = config.AIT_ADS_DIR if ait_dir is None else ait_dir
    pairs = defaultdict(set)
    files = sorted(ait_dir.glob("*_wazuh.json"))
    if not files:
        raise FileNotFoundError(f"No *_wazuh.json files under {ait_dir}")
    for path in files:
        text = path.read_text(errors="replace")
        for raw in MITRE_BLOCK.findall(text):
            try:
                block = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ids = block.get("id") or []
            tactics = block.get("tactic") or []
            for technique in ids:
                parent = str(technique).split(".")[0]      # collapse subtechniques
                for tactic in tactics:
                    pairs[parent].add(str(tactic))
    records = [
        {"technique": technique, "tactic": tactic}
        for technique, tactics in sorted(pairs.items())
        for tactic in sorted(tactics)
    ]
    return pd.DataFrame(records)


def load_mapping(source: str = "reference") -> dict[str, set[str]]:
    """Technique -> set of tactics.

    `source="reference"` uses the checked-in ATT&CK table (the mapping used for
    modelling). `source="wazuh"` uses the local AIT-ADS extraction, kept only as
    a cross-check because it covers 8 techniques.
    """
    if source == "reference":
        if not REFERENCE_PATH.exists():
            raise FileNotFoundError(
                f"{REFERENCE_PATH} missing; run python src/core/build_tactic_reference.py"
            )
        frame = pd.read_csv(REFERENCE_PATH)
    elif source == "wazuh":
        frame = pd.read_csv(CACHE_PATH) if CACHE_PATH.exists() else extract_from_wazuh()
    else:
        raise ValueError(f"Unknown mapping source: {source!r}")
    mapping = defaultdict(set)
    for technique, tactic in zip(frame["technique"], frame["tactic"]):
        mapping[str(technique)].add(str(tactic))
    return dict(mapping)


def tactic_columns() -> list[str]:
    return [f"tactic_{t.replace(' ', '_').lower()}" for t in TACTIC_ORDER] + ["tactic_unknown"]


def multi_hot(techniques, mapping: dict[str, set[str]] | None = None) -> np.ndarray:
    """Multi-hot tactic matrix for a list of techniques, plus an unknown flag."""
    mapping = load_mapping() if mapping is None else mapping
    index = {t: i for i, t in enumerate(TACTIC_ORDER)}
    out = np.zeros((len(techniques), len(TACTIC_ORDER) + 1), dtype=np.float32)
    for row, technique in enumerate(techniques):
        tactics = mapping.get(str(technique))
        if not tactics:
            out[row, -1] = 1.0
            continue
        hit = False
        for tactic in tactics:
            j = index.get(tactic)
            if j is not None:
                out[row, j] = 1.0
                hit = True
        if not hit:
            out[row, -1] = 1.0
    return out


def coverage_report(vocabulary, source: str = "reference") -> dict:
    """How much of a technique vocabulary this mapping actually covers."""
    mapping = load_mapping(source)
    vocabulary = list(vocabulary)
    covered = [t for t in vocabulary if mapping.get(str(t))]
    counts = [len(mapping[str(t)]) for t in covered]
    return {
        "vocabulary": len(vocabulary),
        "covered": len(covered),
        "coverage": len(covered) / max(len(vocabulary), 1),
        "mean_tactics_per_covered_technique": float(np.mean(counts)) if counts else 0.0,
        "multi_tactic_share": float(np.mean([c > 1 for c in counts])) if counts else 0.0,
        "mapped_techniques_total": len(mapping),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src" / "eval"))
    from cv import load_development_examples

    examples = load_development_examples()
    vocabulary = sorted(
        set(examples["next"].astype(str)) | {t for p in examples["prefix"] for t in p}
    )
    for source in ("reference", "wazuh"):
        try:
            report = coverage_report(vocabulary, source)
        except FileNotFoundError as exc:
            print(f"{source}: unavailable ({exc})")
            continue
        print(f"[{source}] vocabulary={report['vocabulary']} "
              f"covered={report['covered']} ({report['coverage']:.1%})  "
              f"mean tactics/technique={report['mean_tactics_per_covered_technique']:.2f}  "
              f"multi-tactic share={report['multi_tactic_share']:.1%}")

    mapping = load_mapping()
    missing = [t for t in vocabulary if not mapping.get(t)]
    print(f"\nunmapped techniques ({len(missing)}): {missing}")
    columns = tactic_columns()
    matrix = multi_hot(vocabulary)
    print(f"\nmulti-hot matrix: {matrix.shape} over columns {len(columns)}")
    totals = pd.Series(matrix.sum(axis=0), index=columns).sort_values(ascending=False)
    print("\ntechniques per tactic in our vocabulary:")
    print(totals[totals > 0].to_string())
