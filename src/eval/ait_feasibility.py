"""Can AIT-ADS supply ATT&CK technique sequences for next-step prediction?

Run from the repository root:
  python src/eval/ait_feasibility.py

AIT-ADS is the dataset the project roadmap named for Phases 1-2 ("download
AIT-ADS, extract alert structures, build temporal host-level attack action
graphs"). Unlike `sightings_v2`, its alerts carry real microsecond timestamps
and the repository ships ground-truth attack windows, so in principle it could
test the one hypothesis the sightings data cannot: whether reporting order is
execution order.

This script measures whether that is actually achievable, by answering:

1. How many alerts are there, and how many carry an ATT&CK technique at all?
2. Which techniques appear, and do they overlap the 109-technique vocabulary the
   models were trained on?
3. Can alerts be attributed to the labelled attack phases by timestamp?
4. After attribution, how many usable technique sequences come out, and how much
   do the 8 scenarios differ from one another?

Writes `results/ait/` and prints a verdict. Reads only AIT files; the sightings
test split is untouched.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "eval")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

OUT_DIR = config.RESULTS_DIR / "ait"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_RE = re.compile(r'"id":\s*\[([^\]]*)\]')
TECHNIQUE_RE = re.compile(r'T\d{4}')
TS_RE = re.compile(r'"@timestamp":\s*"([^"]+)"')
MITRE_MARK = '"mitre"'


def _to_epoch(text: str) -> float:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return float("nan")


def scan_wazuh(path: Path):
    """Stream one Wazuh export, pulling only what the feasibility question needs."""
    total = tagged = 0
    techniques = Counter()
    events = []            # (epoch, technique) for alerts carrying ATT&CK ids
    with path.open(errors="replace") as handle:
        for line in handle:
            total += 1
            if MITRE_MARK not in line:
                continue
            marker = line.find(MITRE_MARK)
            block = line[marker:marker + 400]
            match = ID_RE.search(block)
            if not match:
                continue
            ids = TECHNIQUE_RE.findall(match.group(1))
            if not ids:
                continue
            tagged += 1
            stamp = TS_RE.search(line)
            epoch = _to_epoch(stamp.group(1)) if stamp else float("nan")
            for technique in ids:
                parent = technique.split(".")[0]
                techniques[parent] += 1
                events.append((epoch, parent))
    return total, tagged, techniques, events


def main():
    labels = pd.read_csv(config.AIT_ADS_LABELS_PATH)
    scenarios = sorted(labels["scenario"].unique())

    print("=" * 78)
    print("1. ALERT VOLUME AND ATT&CK TAGGING")
    print("=" * 78)
    rows, all_events, all_techniques = [], {}, Counter()
    for scenario in scenarios:
        path = config.AIT_ADS_DIR / f"{scenario}_wazuh.json"
        if not path.exists():
            print(f"  {scenario}: missing {path.name}")
            continue
        total, tagged, techniques, events = scan_wazuh(path)
        all_events[scenario] = events
        all_techniques.update(techniques)
        rows.append({
            "scenario": scenario,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "alerts": total,
            "attack_tagged": tagged,
            "tagged_pct": round(100 * tagged / max(total, 1), 3),
            "distinct_techniques": len(techniques),
        })
        print(f"  {scenario:<16} {total:>9,} alerts   {tagged:>7,} ATT&CK-tagged "
              f"({100*tagged/max(total,1):>6.3f}%)   {len(techniques):>2} techniques",
              flush=True)
    volume = pd.DataFrame(rows)
    volume.to_csv(OUT_DIR / "alert_volume.csv", index=False)
    print(f"\n  TOTAL {volume['alerts'].sum():,} alerts, "
          f"{volume['attack_tagged'].sum():,} ATT&CK-tagged "
          f"({100*volume['attack_tagged'].sum()/max(volume['alerts'].sum(),1):.3f}%)")

    print("\n" + "=" * 78)
    print("2. TECHNIQUE VOCABULARY AND OVERLAP WITH THE TRAINED MODELS")
    print("=" * 78)
    from cv import load_development_examples

    examples = load_development_examples()
    sightings_vocab = set(examples["next"].astype(str)) | {
        t for p in examples["prefix"] for t in p}
    ait_vocab = set(all_techniques)
    overlap = ait_vocab & sightings_vocab
    print(f"  AIT techniques:        {len(ait_vocab)}  {sorted(ait_vocab)}")
    print(f"  sightings vocabulary:  {len(sightings_vocab)}")
    print(f"  overlap:               {len(overlap)}  {sorted(overlap)}")
    print(f"  AIT-only:              {sorted(ait_vocab - sightings_vocab)}")
    pd.DataFrame({
        "technique": sorted(ait_vocab),
        "ait_alert_count": [all_techniques[t] for t in sorted(ait_vocab)],
        "in_sightings_vocabulary": [t in sightings_vocab for t in sorted(ait_vocab)],
    }).to_csv(OUT_DIR / "technique_overlap.csv", index=False)

    print("\n" + "=" * 78)
    print("3. ATTRIBUTING TAGGED ALERTS TO LABELLED ATTACK PHASES")
    print("=" * 78)
    attribution, sequences = [], {}
    for scenario in scenarios:
        events = all_events.get(scenario)
        if not events:
            continue
        windows = labels.loc[labels["scenario"] == scenario]
        inside = 0
        phase_techniques = defaultdict(list)
        for epoch, technique in events:
            if np.isnan(epoch):
                continue
            hit = windows.loc[(windows["start"] <= epoch) & (epoch <= windows["end"])]
            if len(hit):
                inside += 1
                phase = hit.iloc[0]
                phase_techniques[(float(phase["start"]), phase["attack"])].append(technique)
        ordered = sorted(phase_techniques)
        # One sequence per scenario: phases in time order, first-occurrence dedup
        sequence, seen = [], set()
        for key in ordered:
            for technique in phase_techniques[key]:
                if technique not in seen:
                    seen.add(technique)
                    sequence.append(technique)
        sequences[scenario] = sequence
        attribution.append({
            "scenario": scenario,
            "tagged_alerts": len(events),
            "inside_attack_window": inside,
            "inside_pct": round(100 * inside / max(len(events), 1), 2),
            "phases_with_attack_alerts": len(phase_techniques),
            "labelled_phases": len(windows),
            "derived_sequence_length": len(sequence),
        })
        print(f"  {scenario:<16} {len(events):>7,} tagged   {inside:>6,} in-window "
              f"({100*inside/max(len(events),1):>5.1f}%)   "
              f"{len(phase_techniques)}/{len(windows)} phases covered   "
              f"sequence length {len(sequence)}")
    attribution_frame = pd.DataFrame(attribution)
    attribution_frame.to_csv(OUT_DIR / "phase_attribution.csv", index=False)

    print("\n" + "=" * 78)
    print("4. USABLE SEQUENCES, AND HOW MUCH THE SCENARIOS DIFFER")
    print("=" * 78)
    for scenario, sequence in sequences.items():
        print(f"  {scenario:<16} {sequence}")
    pd.DataFrame([{"scenario": s, "sequence": str(q), "length": len(q)}
                  for s, q in sequences.items()]).to_csv(
        OUT_DIR / "derived_sequences.csv", index=False)

    usable = [q for q in sequences.values() if len(q) >= config.MIN_SEQ_LEN]
    distinct = {tuple(q) for q in sequences.values()}
    print(f"\n  sequences of length >= {config.MIN_SEQ_LEN}: {len(usable)} of {len(sequences)}")
    print(f"  distinct sequences: {len(distinct)}")
    print(f"  prefix/next training examples obtainable: "
          f"{sum(max(len(q) - 2, 0) for q in usable)}")

    print("\n  phase order across scenarios (is it one script?):")
    phase_orders = {}
    for scenario in scenarios:
        windows = labels.loc[labels["scenario"] == scenario].sort_values("start")
        phase_orders[scenario] = tuple(windows["attack"])
    for order in sorted(set(phase_orders.values())):
        who = [s for s, o in phase_orders.items() if o == order]
        print(f"    {len(who)} scenario(s) {who}:")
        print(f"      {' -> '.join(order)}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  training examples in sightings development set: {len(examples):,}")
    print(f"  training examples obtainable from AIT:          "
          f"{sum(max(len(q) - 2, 0) for q in usable)}")
    print(f"  technique vocabulary overlap: {len(overlap)} of {len(sightings_vocab)}")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
