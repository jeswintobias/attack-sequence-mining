"""Load CPTC-2018 Suricata alerts into the project's prefix/next sequence format.

Source
------
`data/cptc_2018/suricata_alert_t*.json.gz` - six gzipped Splunk exports, one per
competition team, from the MIT-licensed `tudelft-cda-lab/SAGE` repository. Each
record wraps the real Suricata alert in a `_raw` JSON string.

Why this dataset
----------------
It is the only public dataset found that has both properties `sightings_v2`
lacks and AIT-ADS lacks respectively: **real microsecond timestamps** and
**many independent attack campaigns** (six teams attacking the same
infrastructure, so attack paths are genuinely independent rather than one
scripted run repeated).

Label space
-----------
Suricata signatures are mapped to SAGE's `MicroAttackStage` taxonomy, whose
tables are read **statically** from `sage_ref/alert_signatures.py` with `ast` -
the third-party module is parsed as data and never imported or executed. The
taxonomy is semantically close to ATT&CK tactics (HOST_DISC, SERVICE_DISC,
BRUTE_FORCE_CREDS, PUBLIC_APP_EXP, LATERAL_MOVEMENT, DATA_EXFILTRATION, ...),
which is what makes results broadly comparable to ours on ATT&CK techniques.

Sequence construction
---------------------
A trace is one `(team, victim host)` pair, matching SAGE, which keeps `dst_ip`
as the victim and treats each team's file as one attacker. Alerts for a pair are
ordered by timestamp and reduced to a symbol sequence. This is a simpler
episode rule than SAGE's frequency-peak segmentation, so results here are
**comparable in task shape but not a strict reproduction** of their numbers.
"""

from __future__ import annotations

import ast
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

CPTC_DIR = config.BASE_DIR / "data" / "cptc_2018"
SAGE_REF = CPTC_DIR / "sage_ref"
TEAMS = ("t1", "t2", "t5", "t7", "t8", "t9")

# Stages that carry no attack semantics and are dropped from sequences.
DROP_STAGES = {"NON_MALICIOUS", "INIT"}

# Keyword-list tables in alert_signatures.py, and the stage each implies.
KEYWORD_TABLES = {
    "endpointDoS_signatures": "END_POINT_DOS",
    "networkDoS_signatures": "NETWORK_DOS",
    "bruteforce_signatures": "BRUTE_FORCE_CREDS",
    "servicedisc_signatures": "SERVICE_DISC",
    "info_disc_signatures": "INFO_DISC",
    "vuln_disc_signatures": "VULN_DISC",
    "hostdisc_signatures": "HOST_DISC",
    "def_evasion_signatures": "DEFENSE_EVASION",
}


def _stage_name(node) -> str | None:
    """Resolve a `MicroAttackStage.X` AST node to the bare name 'X'."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "MicroAttackStage":
            return node.attr
    return None


def load_signature_mapping() -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Parse SAGE's signature tables without importing them.

    Returns (exact signature -> stage, [(keyword, stage)]). `ccdc_combined` is
    skipped: it is the CCDC competition's table, not CPTC's.
    """
    path = SAGE_REF / "alert_signatures.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Fetch signatures/alert_signatures.py from "
            "https://github.com/tudelft-cda-lab/SAGE"
        )
    tree = ast.parse(path.read_text())
    exact: dict[str, str] = {}
    keywords: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "usual_mapping" and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                stage = _stage_name(value)
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and stage:
                    exact[key.value] = stage
        elif name in KEYWORD_TABLES and isinstance(node.value, (ast.List, ast.Tuple)):
            stage = KEYWORD_TABLES[name]
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    keywords.append((element.value, stage))
    return exact, keywords


class SignatureMapper:
    """Signature string -> attack stage, exact match first then keyword match."""

    def __init__(self):
        self.exact, self.keywords = load_signature_mapping()
        self._cache: dict[str, str | None] = {}

    def __call__(self, signature: str) -> str | None:
        if signature in self._cache:
            return self._cache[signature]
        stage = self.exact.get(signature)
        if stage is None:
            for keyword, candidate in self.keywords:
                if keyword in signature:
                    stage = candidate
                    break
        self._cache[signature] = stage
        return stage


RAW_RE = re.compile(r'"_raw":\s*"((?:[^"\\]|\\.)*)"')


def _parse_timestamp(text: str) -> float:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    except ValueError:
        return float("nan")


def read_team_alerts(team: str) -> pd.DataFrame:
    """Extract the fields needed for sequence construction from one team's file."""
    path = CPTC_DIR / f"suricata_alert_{team}.json.gz"
    with gzip.open(path, "rt", errors="replace") as handle:
        payload = json.load(handle)
    records = []
    for entry in payload:
        raw = entry.get("_raw")
        if not raw:
            continue
        try:
            alert = json.loads(raw)
        except json.JSONDecodeError:
            continue
        signature = alert.get("alert", {}).get("signature")
        if not signature:
            continue
        records.append({
            "team": team,
            "timestamp": _parse_timestamp(alert.get("timestamp", "")),
            "src_ip": alert.get("src_ip"),
            "dest_ip": alert.get("dest_ip"),
            "dest_port": alert.get("dest_port"),
            "signature": signature,
            "category": alert.get("alert", {}).get("category"),
            "severity": alert.get("alert", {}).get("severity"),
        })
    return pd.DataFrame(records)


def load_alerts(teams=TEAMS, mapper: SignatureMapper | None = None) -> pd.DataFrame:
    """All teams' alerts with attack stages attached."""
    mapper = SignatureMapper() if mapper is None else mapper
    frames = []
    for team in teams:
        frame = read_team_alerts(team)
        frame["stage"] = frame["signature"].map(mapper)
        frames.append(frame)
        print(f"  {team}: {len(frame):>8,} alerts, "
              f"{frame['stage'].notna().sum():>8,} mapped "
              f"({100 * frame['stage'].notna().mean():.1f}%)", flush=True)
    return pd.concat(frames, ignore_index=True)


def load_stage_values() -> dict[str, int]:
    """Parse the MicroAttackStage enum values, statically, for severity tiers."""
    path = SAGE_REF / "attack_stages.py"
    tree = ast.parse(path.read_text())
    values: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MicroAttackStage":
            for item in node.body:
                if (isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                        and isinstance(item.value, ast.Constant)):
                    values[item.targets[0].id] = int(item.value.value)
    return values


def severity_tiers() -> dict[str, int]:
    """Low / medium / high tier per stage.

    SAGE encodes severity in the digit count of the enum value: single digits are
    reconnaissance, two digits are exploitation and privilege escalation, three
    digits are impact. Its sub-sequence cutter uses exactly this
    (`len(str(mcat))`), so the same rule is reproduced here.
    """
    return {name: len(str(value)) for name, value in load_stage_values().items()}


# Ports grouped into coarse services, so a stage against different services is a
# different symbol. SAGE groups ports similarly; this is a simplified table and
# is the main deliberate deviation from their preprocessing.
SERVICE_PORTS = {
    80: "http", 443: "https", 8080: "http-alt", 8443: "https-alt",
    22: "ssh", 23: "telnet", 21: "ftp", 25: "smtp", 110: "pop3", 143: "imap",
    53: "dns", 445: "smb", 139: "netbios", 135: "rpc", 3389: "rdp",
    1433: "mssql", 3306: "mysql", 5432: "postgres", 1521: "oracle",
    161: "snmp", 389: "ldap", 5900: "vnc", 6379: "redis", 27017: "mongo",
}


def _service(port) -> str:
    try:
        port = int(port)
    except (TypeError, ValueError):
        return "unknown"
    return SERVICE_PORTS.get(port, "high-port" if port >= 1024 else "low-port")


def build_sequences(alerts: pd.DataFrame, min_length: int | None = None,
                    window: int = 150, include_service: bool = True) -> pd.DataFrame:
    """Episode sub-sequences per attacker-victim host pair.

    Reproduces SAGE's trace definition in simplified form:

    1. Group alerts by (attacker host, victim host) -- the attacker is `src_ip`,
       not the team, which is what makes the trace count large.
    2. Aggregate consecutive alerts of the same symbol inside a `window`-second
       bucket into one episode. SAGE segments episodes by frequency peaks; this
       fixed-window rule is the documented simplification.
    3. Order episodes by time and cut the sequence wherever the severity tier
       drops, which is SAGE's marker for a new attack attempt.

    Repeats across episodes are preserved: unlike the sightings pipeline there is
    no first-occurrence deduplication, because revisiting a stage later in a
    campaign is real behaviour here. Prefix masking must therefore be disabled
    when evaluating on this dataset.
    """
    min_length = config.MIN_SEQ_LEN if min_length is None else min_length
    tiers = severity_tiers()

    usable = alerts.dropna(subset=["stage", "timestamp", "src_ip", "dest_ip"]).copy()
    usable = usable.loc[~usable["stage"].isin(DROP_STAGES)]
    if include_service:
        usable["symbol"] = usable["stage"] + "|" + usable["dest_port"].map(_service)
    else:
        usable["symbol"] = usable["stage"]
    usable = usable.sort_values("timestamp", kind="stable")

    rows = []
    for (attacker, victim), group in usable.groupby(["src_ip", "dest_ip"], sort=True):
        times = group["timestamp"].to_numpy()
        symbols = group["symbol"].tolist()
        stages = group["stage"].tolist()
        # Episode aggregation: a new episode starts when the symbol changes or the
        # window elapses since the current episode began.
        episodes, anchor = [], None
        for t, symbol, stage in zip(times, symbols, stages):
            if (episodes and episodes[-1][0] == symbol and anchor is not None
                    and t - anchor < window):
                continue
            episodes.append((symbol, stage, float(t)))
            anchor = float(t)
        if len(episodes) < min_length:
            continue
        # Cut where the severity tier decreases.
        tier_list = [tiers.get(stage, 1) for _, stage, _ in episodes]
        cuts = [i for i in range(len(episodes) - 1) if tier_list[i] > tier_list[i + 1]]
        bounds, start = [], 0
        for cut in cuts:
            bounds.append((start, cut + 1))
            start = cut + 1
        bounds.append((start, len(episodes)))
        for part, (a, b) in enumerate(bounds):
            chunk = episodes[a:b]
            if len(chunk) < min_length:
                continue
            rows.append({
                "attacker": attacker,
                "victim": victim,
                "part": part,
                "sequence": tuple(symbol for symbol, _, _ in chunk),
                "n_episodes": len(chunk),
                "duration_hours": (chunk[-1][2] - chunk[0][2]) / 3600.0,
            })
    return pd.DataFrame(rows)


def to_examples(sequences: pd.DataFrame) -> pd.DataFrame:
    """Expand sequences into the (prefix, next) format the models consume."""
    records = []
    for seq_id, row in enumerate(sequences.itertuples(index=False)):
        sequence = row.sequence
        for cut in range(2, len(sequence)):
            records.append({
                "prefix": tuple(sequence[:cut]),
                "next": sequence[cut],
                "prefix_len": cut,
                "full_seq": tuple(sequence),
                "dup_count": 1,
                "seq_id": seq_id,
                "attacker": row.attacker,
                "length_bucket": str(len(sequence)) if len(sequence) <= 6 else "7+",
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading CPTC-2018 alerts (6 teams)...")
    mapper = SignatureMapper()
    print(f"signature tables: {len(mapper.exact):,} exact entries, "
          f"{len(mapper.keywords)} keyword rules\n")
    alerts = load_alerts(mapper=mapper)
    print(f"\nTOTAL {len(alerts):,} alerts, {alerts['stage'].notna().sum():,} mapped "
          f"({100 * alerts['stage'].notna().mean():.1f}%)")
    print(f"distinct signatures: {alerts['signature'].nunique():,}  "
          f"distinct stages: {alerts['stage'].nunique()}")

    out = config.RESULTS_DIR / "cptc"
    out.mkdir(parents=True, exist_ok=True)
    for include_service in (False, True):
        label = "stage|service" if include_service else "stage only"
        sequences = build_sequences(alerts, include_service=include_service)
        if not len(sequences):
            print(f"\n[{label}] no sequences")
            continue
        examples = to_examples(sequences)
        lengths = sequences["sequence"].map(len)
        print(f"\n[{label}]")
        print(f"  traces: {len(sequences):,}  distinct: {sequences['sequence'].nunique():,}")
        print(f"  length: mean={lengths.mean():.2f} median={lengths.median():.0f} "
              f"max={lengths.max()}")
        print(f"  symbols (label classes): {examples['next'].nunique()}")
        print(f"  prefix/next examples: {len(examples):,}")
        repeats = examples.apply(lambda r: r["next"] in set(r["prefix"]), axis=1).mean()
        print(f"  examples whose target repeats an earlier symbol: {repeats:.1%}")
        suffix = "service" if include_service else "stage"
        sequences.assign(sequence=sequences["sequence"].map(list)).to_csv(
            out / f"cptc_sequences_{suffix}.csv", index=False)
        examples.to_pickle(config.PROCESSED_DATA_DIR / f"cptc_examples_{suffix}.pkl")
    print(f"\nSaved to {out} and data/processed/cptc_examples_*.pkl")
    print("\nPaper reference (arXiv 2408.09888): 555 traces, avg length 5.78, 148 symbols")
