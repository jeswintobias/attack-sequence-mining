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

BASE_DIR = Path(__file__).resolve().parents[2]
SIGHTINGS_DATA_PATH = BASE_DIR / "data" / "sightings_v2_public.csv"
AIT_ADS_DIR = BASE_DIR / "data" / "ait_ads"
AIT_ADS_LABELS_PATH = AIT_ADS_DIR / "labels.csv"
DATA_PATH = SIGHTINGS_DATA_PATH
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RESULTS_DIR = BASE_DIR / "results"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
BASELINES_RESULTS_DIR = RESULTS_DIR / "baselines"
ASSOCIATION_RESULTS_DIR = RESULTS_DIR / "association_rules"
PREFIXSPAN_RESULTS_DIR = RESULTS_DIR / "prefixspan"
NEGATIVE_RULES_RESULTS_DIR = RESULTS_DIR / "negative_rules"
LIGHTGBM_RESULTS_DIR = RESULTS_DIR / "lightgbm"
XGBOOST_RESULTS_DIR = RESULTS_DIR / "xgboost"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"
CV_RESULTS_DIR = RESULTS_DIR / "cv"

# Ensure all result subdirectories exist
ALL_RESULT_DIRS = [
    RESULTS_DIR,
    EDA_RESULTS_DIR,
    BASELINES_RESULTS_DIR,
    ASSOCIATION_RESULTS_DIR,
    PREFIXSPAN_RESULTS_DIR,
    NEGATIVE_RULES_RESULTS_DIR,
    LIGHTGBM_RESULTS_DIR,
    XGBOOST_RESULTS_DIR,
    COMPARISON_RESULTS_DIR,
    CV_RESULTS_DIR,
]
for d in ALL_RESULT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

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

# Rule out candidates already present in the prefix before ranking.
# Rationale for viva: first-occurrence deduplication (DEC-03) means a technique
# appears at most once per sequence, so the target is never a member of its own
# prefix (verified 0/8,329 train and 0/2,776 validation examples). Ranking a
# prefix member wastes a top-k slot on an outcome the label construction has
# excluded. Caveat: this is a property of our deduplication choice, not of real
# attacker behaviour -- adversaries do repeat techniques -- so the masked
# metrics describe the task as defined here. Applied to every model so the
# comparison stays on one footing.
MASK_SEEN_TECHNIQUES = True

# ==========================================
# 1b. Cross-validated evaluation protocol
# ==========================================
# Model selection and comparative reporting run on a development set that merges
# the original train and validation splits (2,178 unique sequences), using
# N_CV_FOLDS grouped, length-stratified folds. Measured rationale: a single
# 545-sequence validation holdout has SE 0.0090 on sequence MRR (95% CI
# half-width 0.0176), so the smallest unpaired difference it can resolve is
# about 0.036 -- while the three LightGBM searches differed by only 0.0047.
# Five folds over the development set halve that SE to 0.0045. The test split
# stays sealed for a single confirmatory run.
N_CV_FOLDS = 5
DEV_SPLITS = ("train", "val")
SEALED_SPLITS = ("test",)

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
