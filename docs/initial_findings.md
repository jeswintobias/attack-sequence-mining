# Initial Findings, Architectural Decisions & EDA Implementation Plan

**Project**: Mining Attack Sequences and Missing-Event Rules to Predict the Next Attacker Step  
**Workspace**: `attack-sequence-mining`  
**Date**: October 2026  
**Status**: Stage A Inception & Exploratory Data Analysis (EDA)

---

## 1. Project Background & Problem Formulation

In enterprise security, intrusion detection systems (IDS), endpoint detection and response (EDR) sensors, and threat intelligence feeds observe adversaries executing multi-stage attacks across the **Cyber Kill Chain**. 

Our research focuses on:
1. **Sequential Pattern Mining**: Discovering ordered, frequent technique progressions ($T_1 \to T_2 \to T_3 \dots$) rather than unordered itemsets.
2. **Next-Step Prediction**: Given an observed prefix of adversary actions ($T_1 \dots T_k$), predicting the upcoming attacker technique ($T_{k+1}$) with calibrated confidence scores before high-impact objectives (credential dumping, lateral movement, exfiltration) occur.
3. **Missing-Event Rules (Post-V1 Extension)**: Mining negative implications ($X \wedge \neg \mathcal{M} \implies Y$) when expected intermediate security alarms fail to fire due to deliberate stealth (in-memory execution, direct syscalls, living-off-the-land tools) or sensor blind spots.

---

## 2. Dataset Synthesis & The "Dual-Source" Architecture

During our initial technical investigation, two primary dataset sources were acquired and evaluated:

### Source A: MITRE ATT&CK Sightings Ecosystem v2 (`data/sightings_v2_public.csv`)
* **Origin**: Publicly released by the Center for Threat-Informed Defense (CTID) / MITRE Engenuity.
* **Scale**: Exactly **1,048,575 rows** (matching Excel's max row boundary), ~25.7 MB.
* **Columns**: `tid` (Python stringified lists, e.g., `"['T1059', 'T1105']"`) and `sighting_date` (format `%m/%d/%y`).
* **Characteristics**:
  * **~72% Singletons**: 757,291 rows contain only a single isolated technique.
  * **~7,800 Rows with Length $\ge 4$**: Yielding ~2,900 unique multi-stage attack sequences.
  * **278 Rows with Internal Repetition**: Techniques repeated within the same sighting list.
  * **Corrupted Date Anomalies**: Timestamp parsing reveals out-of-range years (1969, 1980, 2040).
  * **Ordering Caveat**: Each row is an aggregated incident sighting; list order is a proxy for reporting co-occurrence rather than microsecond physical execution.

### Source B: AIT Alert Data Set (AIT-ADS Benchmark - Landauer et al., CSET 2024)
* **Origin**: Austrian Institute of Technology testbed with multi-sensor telemetry (Wazuh HIDS, AMiner, Suricata).
* **Scale**: 8 full multi-step attack scenarios (`fox`, `harrison`, `russellmitchell`, `santos`, `shaw`, `wardbeck`, `wheeler`, `wilson`), 16 raw JSON alert streams (~2.8 GB, 2.65M alerts), and an 80-row ground-truth schedule (`labels.csv`).
* **Characteristics**:
  * **100% Physical Timestamps**: Verified down to the millisecond across 10 network hosts.
  * **Ground-Truth Missing Events**: Scenario `wheeler` demonstrates true adversarial omission (skipping password cracking and jumping directly from web shell to reverse shell).
  * **Sample Size Tradeoff**: Macro-level scenario steps produce only ~41 unique sequences, making macro-only test splits statistically fragile.

### The Jeffrey Ullman Synthesis (The Dual Strategy):
Rather than relying on noisy sightings or tiny scenario sample sizes in isolation:
1. **Macro Training & Prior Calibration**: We mine broad technique rules, baselines, and PrefixSpan patterns on `sightings_v2_public.csv` (~7,800 multi-stage sequences).
2. **Prior Probability Weighting via Singletons**: The 757,291 singletons are **not discarded from the project**—they are retained in pandas memory to compute the **global unigram base rates** $P(T)$, powering Bayesian back-off in Markov models.
3. **Temporal Ground-Truth Validation**: We validate predictions and evaluate missing-event rules against the physical ground-truth testbed of AIT-ADS.

---

## 3. Core Design Principles (Strict In-Memory Processing)

As explicitly mandated:
* **Zero Modification to Raw Files**: `data/sightings_v2_public.csv` remains completely immutable on disk.
* **DataFrame-Level Transformations**: All parsing (`ast.literal_eval`), date sanitization, singleton isolation, deduplication, and subtechnique collapsing are executed dynamically inside pandas DataFrames in memory.
* **Full Data Retention**: Singletons are isolated into a dedicated DataFrame for prior estimation, while sequences of length $\ge 4$ are routed to the sequential mining pipeline.

---

## 4. Comprehensive EDA Implementation Plan

Our Exploratory Data Analysis (EDA) will be implemented in `src/eda.py` and visualized in `v1.ipynb`:

```
+----------------------------------------------------------------------------------------------------+
|                                    EDA EXECUTION PIPELINE                                          |
+--------------------+---------------------------------------------------+---------------------------+
| Stage              | Actions & Metrics Computed                        | Output Deliverable        |
+--------------------+---------------------------------------------------+---------------------------+
| 1. Safe Ingestion  | - ast.literal_eval on 'tid'                       | Clean In-Memory DataFrame |
|    & Cleaning      | - pd.to_datetime with %m/%d/%y                    | Out-of-range date flags   |
|                    | - Flag & isolate junk years (<2000, >2026)        |                           |
+--------------------+---------------------------------------------------+---------------------------+
| 2. Cardinality &   | - Compute exact sequence length distribution      | Length distribution table |
|    Singleton Split | - Isolate singletons (len == 1) for Prior table   | results/eda_lengths.csv   |
|                    | - Filter sequences with length >= 4               |                           |
+--------------------+---------------------------------------------------+---------------------------+
| 3. Technique       | - Global technique counts across all 1.04M rows   | Top-30 Techniques CSV     |
|    Prevalence      | - Technique counts within length >= 4 sequences   | results/top_techniques.csv|
|                    | - Comparison of prior P(T) vs sequence P(T)       |                           |
+--------------------+---------------------------------------------------+---------------------------+
| 4. Repetition &    | - Detect intra-list repetitions (278 target rows) | Deduped Sequence DB       |
|    Deduplication   | - Deduplicate consecutive identical techniques    | In-memory                 |
+--------------------+---------------------------------------------------+---------------------------+
| 5. Subtechnique    | - Measure frequency of parent vs subtechniques    | Collapsing comparison     |
|    Analysis        | - Compare collapse_subtechniques = True vs False  | table                     |
+--------------------+---------------------------------------------------+---------------------------+
| 6. Visualizations  | - High-res sequence length histogram              | results/eda_plots.png     |
|                    | - Top-20 technique frequency horizontal bar chart |                           |
|                    | - Temporal sighting volume over time (clean dates)|                           |
+--------------------+---------------------------------------------------+---------------------------+
```

---

## 5. Methodological Justification for Viva Defense

Every design choice is grounded in peer-reviewed research from our research corpus:

1. **Why Filter Singletons for Sequence Models?**  
   * *Reference*: **Roelofs et al. (ACM SAC 2024, ING Bank)**.  
   * Over 99% of raw alerts in enterprise SOCs are isolated, low-fidelity commodity alarms. Sequential pattern mining requires multi-stage transitions ($T_1 \to T_2$); filtering for $\text{length} \ge 4$ isolates true multi-stage intrusions from background noise.
2. **Why Use Singletons for Bayesian Back-off?**  
   * *Reference*: **Li et al. (DeepAG, IEEE TDSC 2023)**.  
   * Unseen prefixes in test sets suffer from zero-frequency sparsity. Backing off to the global unigram prior $P(T)$ computed over 1 million real-world sightings provides calibrated baseline predictions.
3. **Why Split by Whole Sequence?**  
   * *Reference*: **Khan et al. (ACM DTRAP 2026)**.  
   * Splitting at the sequence level (60/20/20 stratified by length bucket) ensures that no full attack sequence appears in more than one set. This prevents the model from memorising exact attack playbooks.

---

## 6. Next Immediate Steps
1. Execute `src/eda.py` across `data/sightings_v2_public.csv` entirely in-memory.
2. Save EDA statistics and publication-quality plots to `results/`.
3. Update `config.py` with refined parameters based on EDA findings.
4. Finalize Stage A prefix-disjoint split on the validated multi-stage sequences.
