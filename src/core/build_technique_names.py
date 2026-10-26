"""Write the ATT&CK Enterprise technique id -> human-readable name table.

Provenance
----------
Read from https://attack.mitre.org/techniques/enterprise/ on 2026-10-26, which
lists every Enterprise technique with its identifier and name. Sub-techniques
are excluded, matching DEC-01's collapse to parent techniques.

The web application needs these names: nobody recalls that `T1086` is
PowerShell, so the technique picker has to search on names rather than
identifiers. Checked in so the application works without network access.
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

REFERENCE_PATH = config.BASE_DIR / "data" / "reference" / "attack_technique_names.csv"

RAW = """
T1548|Abuse Elevation Control Mechanism
T1134|Access Token Manipulation
T1531|Account Access Removal
T1087|Account Discovery
T1098|Account Manipulation
T1650|Acquire Access
T1583|Acquire Infrastructure
T1595|Active Scanning
T1557|Adversary-in-the-Middle
T1700|AI-Enabled Exfiltration
T1696|Anti-Automation Evasion
T1071|Application Layer Protocol
T1010|Application Window Discovery
T1560|Archive Collected Data
T1123|Audio Capture
T1119|Automated Collection
T1020|Automated Exfiltration
T1197|BITS Jobs
T1547|Boot or Logon Autostart Execution
T1037|Boot or Logon Initialization Scripts
T1217|Browser Information Discovery
T1185|Browser Session Hijacking
T1110|Brute Force
T1612|Build Image on Host
T1115|Clipboard Data
T1651|Cloud Administration Command
T1671|Cloud Application Integration
T1580|Cloud Infrastructure Discovery
T1538|Cloud Service Dashboard
T1526|Cloud Service Discovery
T1619|Cloud Storage Object Discovery
T1059|Command and Scripting Interpreter
T1092|Communication Through Removable Media
T1586|Compromise Accounts
T1554|Compromise Host Software Binary
T1584|Compromise Infrastructure
T1609|Container Administration Command
T1613|Container and Resource Discovery
T1659|Content Injection
T1136|Create Account
T1543|Create or Modify System Process
T1555|Credentials from Password Stores
T1485|Data Destruction
T1132|Data Encoding
T1486|Data Encrypted for Impact
T1530|Data from Cloud Storage
T1602|Data from Configuration Repository
T1213|Data from Information Repositories
T1005|Data from Local System
T1039|Data from Network Shared Drive
T1025|Data from Removable Media
T1565|Data Manipulation
T1001|Data Obfuscation
T1074|Data Staged
T1030|Data Transfer Size Limits
T1622|Debugger Evasion
T1491|Defacement
T1678|Delay Execution
T1140|Deobfuscate/Decode Files or Information
T1610|Deploy Container
T1587|Develop Capabilities
T1652|Device Driver Discovery
T1699|Direct Prompt Injection
T1006|Direct Volume Access
T1686|Disable or Modify System Firewall
T1685|Disable or Modify Tools
T1561|Disk Wipe
T1484|Domain or Tenant Policy Modification
T1482|Domain Trust Discovery
T1689|Downgrade Attack
T1189|Drive-by Compromise
T1568|Dynamic Resolution
T1667|Email Bombing
T1114|Email Collection
T1573|Encrypted Channel
T1499|Endpoint Denial of Service
T1611|Escape to Host
T1585|Establish Accounts
T1675|ESXi Administration Command
T1546|Event Triggered Execution
T1668|Exclusive Control
T1480|Execution Guardrails
T1048|Exfiltration Over Alternative Protocol
T1041|Exfiltration Over C2 Channel
T1011|Exfiltration Over Other Network Medium
T1052|Exfiltration Over Physical Medium
T1567|Exfiltration Over Web Service
T1190|Exploit Public-Facing Application
T1203|Exploitation for Client Execution
T1212|Exploitation for Credential Access
T1687|Exploitation for Defense Impairment
T1068|Exploitation for Privilege Escalation
T1211|Exploitation for Stealth
T1210|Exploitation of Remote Services
T1133|External Remote Services
T1008|Fallback Channels
T1083|File and Directory Discovery
T1222|File and Directory Permissions Modification
T1657|Financial Theft
T1495|Firmware Corruption
T1187|Forced Authentication
T1606|Forge Web Credentials
T1592|Gather Victim Host Information
T1589|Gather Victim Identity Information
T1590|Gather Victim Network Information
T1591|Gather Victim Org Information
T1683|Generate Content
T1615|Group Policy Discovery
T1200|Hardware Additions
T1564|Hide Artifacts
T1665|Hide Infrastructure
T1574|Hijack Execution Flow
T1525|Implant Internal Image
T1070|Indicator Removal
T1202|Indirect Command Execution
T1698|Indirect Prompt Injection
T1105|Ingress Tool Transfer
T1490|Inhibit System Recovery
T1056|Input Capture
T1674|Input Injection
T1559|Inter-Process Communication
T1534|Internal Spearphishing
T1570|Lateral Tool Transfer
T1680|Local Storage Discovery
T1654|Log Enumeration
T1036|Masquerading
T1556|Modify Authentication Process
T1578|Modify Cloud Compute Infrastructure
T1666|Modify Cloud Resource Hierarchy
T1112|Modify Registry
T1601|Modify System Image
T1111|Multi-Factor Authentication Interception
T1621|Multi-Factor Authentication Request Generation
T1104|Multi-Stage Channels
T1106|Native API
T1599|Network Boundary Bridging
T1498|Network Denial of Service
T1046|Network Service Discovery
T1135|Network Share Discovery
T1040|Network Sniffing
T1095|Non-Application Layer Protocol
T1571|Non-Standard Port
T1027|Obfuscated Files or Information
T1588|Obtain Capabilities
T1137|Office Application Startup
T1003|OS Credential Dumping
T1201|Password Policy Discovery
T1120|Peripheral Device Discovery
T1069|Permission Groups Discovery
T1566|Phishing
T1598|Phishing for Information
T1647|Plist File Modification
T1677|Poisoned Pipeline Execution
T1653|Power Settings
T1542|Pre-OS Boot
T1690|Prevent Command History Logging
T1057|Process Discovery
T1055|Process Injection
T1572|Protocol Tunneling
T1090|Proxy
T1682|Query Public AI Services
T1012|Query Registry
T1620|Reflective Code Loading
T1219|Remote Access Tools
T1563|Remote Service Session Hijacking
T1021|Remote Services
T1018|Remote System Discovery
T1091|Replication Through Removable Media
T1496|Resource Hijacking
T1207|Rogue Domain Controller
T1014|Rootkit
T1688|Safe Mode Boot
T1053|Scheduled Task/Job
T1029|Scheduled Transfer
T1113|Screen Capture
T1597|Search Closed Sources
T1596|Search Open Technical Databases
T1593|Search Open Websites/Domains
T1681|Search Threat Vendor Data
T1594|Search Victim-Owned Websites
T1679|Selective Exclusion
T1505|Server Software Component
T1648|Serverless Execution
T1489|Service Stop
T1129|Shared Modules
T1684|Social Engineering
T1072|Software Deployment Tools
T1518|Software Discovery
T1176|Software Extensions
T1608|Stage Capabilities
T1528|Steal Application Access Token
T1697|Steal Application Specific Passwords
T1649|Steal or Forge Authentication Certificates
T1558|Steal or Forge Kerberos Tickets
T1539|Steal Web Session Cookie
T1553|Subvert Trust Controls
T1195|Supply Chain Compromise
T1218|System Binary Proxy Execution
T1082|System Information Discovery
T1614|System Location Discovery
T1016|System Network Configuration Discovery
T1049|System Network Connections Discovery
T1033|System Owner/User Discovery
T1216|System Script Proxy Execution
T1007|System Service Discovery
T1569|System Services
T1529|System Shutdown/Reboot
T1124|System Time Discovery
T1080|Taint Shared Content
T1221|Template Injection
T1205|Traffic Signaling
T1537|Transfer Data to Cloud Account
T1127|Trusted Developer Utilities Proxy Execution
T1199|Trusted Relationship
T1552|Unsecured Credentials
T1535|Unused/Unsupported Cloud Regions
T1550|Use Alternate Authentication Material
T1204|User Execution
T1078|Valid Accounts
T1125|Video Capture
T1673|Virtual Machine Discovery
T1497|Virtualization/Sandbox Evasion
T1600|Weaken Encryption
T1102|Web Service
T1669|Wi-Fi Networks
T1047|Windows Management Instrumentation
T1220|XSL Script Processing
"""


def build() -> pd.DataFrame:
    records = []
    for line in RAW.strip().splitlines():
        technique, _, name = line.partition("|")
        technique, name = technique.strip(), name.strip()
        if technique and name:
            records.append({"technique": technique, "name": name})
    return (pd.DataFrame(records)
            .drop_duplicates("technique")
            .sort_values("technique")
            .reset_index(drop=True))


def load_names() -> dict[str, str]:
    """Technique id -> name. Falls back to the raw table if the CSV is absent."""
    if REFERENCE_PATH.exists():
        frame = pd.read_csv(REFERENCE_PATH)
    else:
        frame = build()
    return dict(zip(frame["technique"].astype(str), frame["name"].astype(str)))


if __name__ == "__main__":
    frame = build()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REFERENCE_PATH, index=False)
    print(f"Wrote {len(frame)} technique names -> {REFERENCE_PATH}")

    sys.path.insert(0, str(ROOT / "src" / "eval"))
    from cv import load_development_examples

    examples = load_development_examples()
    vocabulary = sorted(set(examples["next"].astype(str))
                        | {t for p in examples["prefix"] for t in p})
    names = load_names()
    missing = [t for t in vocabulary if t not in names]
    print(f"vocabulary coverage: {len(vocabulary) - len(missing)}/{len(vocabulary)} "
          f"({(len(vocabulary) - len(missing)) / len(vocabulary):.1%})")
    if missing:
        print(f"unnamed techniques ({len(missing)}): {missing}")
