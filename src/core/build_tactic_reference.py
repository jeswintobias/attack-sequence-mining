"""Write the ATT&CK Enterprise technique -> tactic reference table.

Provenance
----------
Technique lists were read from the 14 MITRE ATT&CK Enterprise tactic pages at
`https://attack.mitre.org/tactics/<TAxxxx>/` on 2026-10-25. Each list is the set
of techniques that page shows as belonging to that tactic. Sub-technique ids are
collapsed to their parent (`T1021.001 -> T1021`) to match DEC-01.

This is reference data from an external authority, not a measurement, and is
checked in so the mapping is auditable and the experiments reproducible without
network access. Regenerate by re-reading those pages if ATT&CK is updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

# tactic id -> (tactic name, techniques listed on that tactic's page)
TACTIC_TECHNIQUES = {
    "TA0043": ("Reconnaissance", """T1595 T1592 T1589 T1590 T1591 T1598 T1682 T1597 T1596 T1593
        T1681 T1594"""),
    "TA0042": ("Resource Development", """T1650 T1583 T1586 T1584 T1587 T1585 T1683 T1588 T1608"""),
    "TA0001": ("Initial Access", """T1659 T1189 T1190 T1133 T1200 T1566 T1091 T1195 T1199 T1078
        T1669"""),
    "TA0002": ("Execution", """T1197 T1651 T1059 T1609 T1610 T1699 T1675 T1203 T1574 T1698 T1674
        T1559 T1106 T1677 T1053 T1648 T1129 T1072 T1569 T1127 T1204 T1047"""),
    "TA0003": ("Persistence", """T1098 T1197 T1547 T1037 T1671 T1554 T1136 T1543 T1546 T1668 T1133
        T1525 T1556 T1112 T1137 T1653 T1542 T1053 T1505 T1176 T1205 T1078"""),
    "TA0004": ("Privilege Escalation", """T1548 T1134 T1098 T1547 T1037 T1543 T1484 T1611 T1546
        T1068 T1055 T1053 T1078"""),
    "TA0005": ("Defense Evasion", """T1134 T1696 T1197 T1612 T1622 T1678 T1140 T1006 T1480 T1211
        T1564 T1574 T1070 T1202 T1036 T1027 T1542 T1055 T1620 T1014 T1679 T1684 T1218 T1216 T1221
        T1205 T1127 T1535 T1078 T1497 T1220"""),
    "TA0006": ("Credential Access", """T1557 T1110 T1555 T1212 T1187 T1606 T1056 T1556 T1111 T1621
        T1040 T1003 T1528 T1697 T1649 T1558 T1539 T1552"""),
    "TA0007": ("Discovery", """T1087 T1010 T1217 T1580 T1538 T1526 T1619 T1613 T1622 T1652 T1482
        T1083 T1615 T1680 T1654 T1046 T1135 T1040 T1201 T1120 T1069 T1057 T1012 T1018 T1518 T1082
        T1614 T1016 T1049 T1033 T1007 T1124 T1673 T1497"""),
    "TA0008": ("Lateral Movement", """T1210 T1534 T1570 T1563 T1021 T1091 T1072 T1080 T1550"""),
    "TA0009": ("Collection", """T1557 T1560 T1123 T1119 T1185 T1115 T1530 T1602 T1213 T1005 T1039
        T1025 T1074 T1114 T1056 T1113 T1125"""),
    "TA0011": ("Command and Control", """T1071 T1092 T1659 T1132 T1001 T1568 T1573 T1008 T1665
        T1105 T1104 T1095 T1571 T1572 T1090 T1219 T1205 T1102"""),
    "TA0010": ("Exfiltration", """T1700 T1020 T1030 T1048 T1041 T1011 T1052 T1567 T1029 T1537"""),
    "TA0040": ("Impact", """T1531 T1485 T1486 T1565 T1491 T1561 T1667 T1499 T1657 T1495 T1490
        T1498 T1496 T1489 T1529"""),
}

REFERENCE_PATH = config.BASE_DIR / "data" / "reference" / "attack_technique_tactics.csv"


def build() -> pd.DataFrame:
    records = []
    for tactic_id, (tactic_name, blob) in TACTIC_TECHNIQUES.items():
        for token in blob.split():
            records.append({
                "technique": token.split(".")[0],
                "tactic": tactic_name,
                "tactic_id": tactic_id,
            })
    frame = pd.DataFrame(records).drop_duplicates().sort_values(["technique", "tactic"])
    return frame.reset_index(drop=True)


if __name__ == "__main__":
    frame = build()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REFERENCE_PATH, index=False)
    per_technique = frame.groupby("technique")["tactic"].nunique()
    print(f"Wrote {len(frame)} technique/tactic pairs for "
          f"{per_technique.size} techniques -> {REFERENCE_PATH}")
    print(f"tactics per technique: mean={per_technique.mean():.2f} "
          f"max={per_technique.max()} multi-tactic share={(per_technique > 1).mean():.1%}")
    print(f"tactics covered: {frame['tactic'].nunique()} of 14")
    print("\nmost multi-tactic techniques:")
    print(per_technique.sort_values(ascending=False).head(5).to_string())
