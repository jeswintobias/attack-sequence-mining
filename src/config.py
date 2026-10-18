"""
Configuration module for Attack Sequence Mining (V1).
Stores all global parameters, paths, random seeds, and MITRE ATT&CK mapping tables.
Every design choice is documented for viva defense.
"""

from pathlib import Path

# ==========================================
# 1. Reproducibility & Global Paths
# ==========================================
# Fixed random seed to ensure strict experimental reproducibility across splits and baselines.
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
SIGHTINGS_DATA_PATH = BASE_DIR / "data" / "sightings_v2_public.csv"
AIT_ADS_DIR = BASE_DIR / "data" / "ait_ads"
AIT_ADS_LABELS_PATH = AIT_ADS_DIR / "labels.csv"
DATA_PATH = SIGHTINGS_DATA_PATH
RESULTS_DIR = BASE_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Sequence Mining & Preprocessing Settings
# ==========================================
# Minimum sequence length: Filter out very short sequences (lengths < 4)
# Rationale: Predicting next steps requires sufficient prefix context (length >= 2)
# and at least one step ahead.
MIN_SEQ_LEN = 4

# Flag to collapse sub-techniques to parent technique (e.g., T1595.002 -> T1595)
# Rationale for viva: Collapsing reduces sparsity and groups granular tool variations
# into macro-tactical behaviors, whereas keeping subtechniques preserves fine-grained tooling.
COLLAPSE_SUBTECHNIQUES = True

# Whole-sequence stratified split ratios (60/20/20 by length bucket)
# Rationale for viva: Splits unique sequences so no full attack playbook
# appears in more than one set. Stratification by length bucket ensures
# balanced length distributions.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

# Top-K evaluation values for prediction baselines
TOP_K_VALUES = [1, 3, 5]

# Association rule mining (Stage C) and PrefixSpan (Stage D) parameters
MIN_SUPPORT = 0.1
MAX_PATTERN_LENGTH = 5

# ==========================================
# 3. Ground-Truth to MITRE ATT&CK Mapping
# ==========================================
# Mapping each AIT-ADS attack phase to its corresponding MITRE ATT&CK technique:
# - network_scans: Active scanning of IP ranges (T1595.001)
# - service_scans: Scanning open service ports (T1046)
# - dirb: Web content/directory brute forcing (T1595.003 - Wordlist Scanning)
# - wpscan: CMS/vulnerability scanning (T1595.002 - Vulnerability Scanning)
# - webshell: Persistence via web shell upload (T1505.003 - Server Software Component: Web Shell)
# - cracking: Offline/online credential brute forcing (T1110.001 - Password Guessing)
# - reverse_shell: Interactive remote shell execution (T1059.004 - Unix Shell)
# - privilege_escalation: Exploiting local vulnerabilities for root (T1068)
# - service_stop: Terminating defensive/security services (T1489)
# - dnsteal: Exfiltrating collected data over covert DNS queries (T1048.003)
ATTACK_TO_MITRE = {
    "network_scans": "T1595.001",
    "service_scans": "T1046",
    "dirb": "T1595.003",
    "wpscan": "T1595.002",
    "webshell": "T1505.003",
    "cracking": "T1110.001",
    "reverse_shell": "T1059.004",
    "privilege_escalation": "T1068",
    "service_stop": "T1489",
    "dnsteal": "T1048.003",
}
