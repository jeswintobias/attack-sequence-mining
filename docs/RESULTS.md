# Experimental Results & Dataset Benchmarks (V1)

*Note: All numerical metrics reported in this document strictly originate from
code outputs generated during actual pipeline execution. Every statement is
labelled MEASURED (from code output) or ASSUMPTION (design choice).*

---

## 1. Exploratory Data Analysis (EDA) Summary

*Source*: `data/sightings_v2_public.csv` (Center for Threat-Informed Defense / MITRE Engenuity)  
*Raw File Status*: Immutable on disk; all transformations executed in-memory.

| Metric | Measured Value | Viva Significance |
| :--- | :--- | :--- |
| **Total Raw Rows** | **1,048,575** (MEASURED) | Exactly matches Excel's maximum row boundary. |
| **Singletons (Length = 1)** | **757,291 (72.22%)** (MEASURED) | Isolated commodity detections; retained for global unigram prior estimation $P(T)$. |
| **Short Transitions (Length 2–3)** | **283,503 (27.04%)** (MEASURED) | Intermediate detections insufficient for length $\ge 4$ sequence mining. |
| **Multi-Stage Sequences (Length $\ge 4$, collapsed)** | **7,537 rows / 2,721 unique** (MEASURED) | Genuine multi-stage attack campaigns isolated for sequential pattern mining. |
| **Intra-List Repeated Techniques** | **278 rows** (MEASURED) | Deduplicated using first-occurrence rule (not just consecutive repeats). |
| **Corrupted Date Rows** | **14 rows** (MEASURED) | Out-of-range years: 1969 (9), 1980 (3), 1970 (1), 2040 (1). |
| **Fine-Grained Techniques** | **378** (MEASURED) | Unique technique / subtechnique tokens observed. |
| **Collapsed Parent Techniques** | **185** (MEASURED) | Macro-level tactical techniques after stripping subtechniques. |

### 1.1 Deduplication Rule

The project uses **first-occurrence deduplication**: for each sighting row, if a technique
appears more than once anywhere in the list, only the first occurrence is kept.  
Example: `[T1, T2, T1, T3]` → `[T1, T2, T3]`. (MEASURED — implemented in
`deduplicate_first_occurrence()` in `src/preprocessing.py` and `src/split.py`)

This is **not** consecutive-only deduplication. The existing code also contains a
`deduplicate_consecutive()` function that removes only adjacent repeats, but
the pipeline uses `deduplicate_first_occurrence()`. (MEASURED)

### 1.2 Length Statistics (Length $\ge 4$, Collapsed Subtechniques)

| Scope | N | Mean | Median | Std | Skew | Min | Q25 | Q50 | Q75 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **All rows** | 7,537 | 6.28 | 6.0 | 2.09 | 1.07 | 4 | 5.0 | 6.0 | 8.0 | 31 |
| **Unique sequences** | 2,721 | 7.12 | 7.0 | 2.47 | 0.86 | 4 | 5.0 | 7.0 | 9.0 | 31 |

(MEASURED — `results/length_stats_ge4.csv`)

**Chart 2 note**: The existing Chart 2 (`02_length_hist_ge4.png`) counts **rows** (not
unique sequences) and uses **uncollapsed** subtechniques, which gives a mean of ~5.5
and median of 5. The collapsed-subtechnique, first-occurrence-deduped row-level stats
are mean 6.28 / median 6.0, and unique-sequence stats are mean 7.12 / median 7.0.
The difference arises because (a) collapsing reduces sequence length, and (b) short
sequences are over-represented among duplicates. (MEASURED)

---

## 2. Global Technique Priors (Singleton Rows, Collapsed)

These base rates are computed from the 757,291 singleton rows with subtechniques
collapsed to parent, and power **Baseline 2** (unigram prior fallback).
Saved to `results/global_technique_priors.csv`. (MEASURED)

---

## 3. Stage A: 60/20/20 Whole-Sequence Stratified Split

Execution parameters: `MIN_SEQ_LEN = 4`, `COLLAPSE_SUBTECHNIQUES = True`,
`RANDOM_SEED = 42`, split ratio 60/20/20. (ASSUMPTION — design choice)

Stratification buckets: {4, 5, 6, 7+} by sequence length. (ASSUMPTION)

### 3.1 Set Sizes

| Set | Unique Sequences | Dup-Weighted Rows | Total (Prefix, Next) Examples | Dup-Weighted Examples | Bucket 4 | Bucket 5 | Bucket 6 | Bucket 7+ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train** | 1,633 | 4,291 | 8,329 | 17,786 | 281 | 237 | 222 | 893 |
| **Val** | 545 | 1,676 | 2,776 | 7,398 | 94 | 79 | 74 | 298 |
| **Test** | 543 | 1,570 | 2,832 | 7,099 | 93 | 79 | 74 | 297 |

(MEASURED — `results/data_stats.csv`)

### 3.2 Integrity Assertions

| Assertion | Result |
| :--- | :--- |
| No full sequence in train ∩ val | **PASS** (MEASURED) |
| No full sequence in train ∩ test | **PASS** (MEASURED) |
| No full sequence in val ∩ test | **PASS** (MEASURED) |
| Sets partition all 2,721 unique sequences | **PASS** (MEASURED) |

### 3.3 Leakage Analysis

| Metric | Value |
| :--- | :--- |
| Share of val (last, next) pairs seen as consecutive items in any train sequence | **0.9697** (MEASURED) |
| Share of test (last, next) pairs seen as consecutive items in any train sequence | **0.9640** (MEASURED) |

These high values are expected: although no full sequence appears across sets, most
local bigram transitions are shared because attackers reuse common technique pairs
(e.g., T1055 → T1070 is a standard injection-then-cleanup pattern). This is not
data leakage — the model must still learn the correct transition from the full
prefix context. (ASSUMPTION — interpretation)

---

## 4. Order-Signal Permutation Test (Train Sequences Only)

**Goal**: Test whether the ordering of techniques within sighting lists is
statistically non-random, without claiming it is chronological. (ASSUMPTION —
we do not know whether the original list order reflects physical execution time;
we test only whether it is distinguishable from a random permutation.)

**Method**: Mean transition asymmetry over the 30 most frequent techniques,
compared to a null distribution from 200 random shuffles. Following the
directional analysis approach in Li et al. (DeepAG, IEEE TDSC 2023).

| Metric | Value |
| :--- | :--- |
| Real mean asymmetry | **0.6199** (MEASURED) |
| Number of qualifying pairs (total ≥ 20) | **140** (MEASURED) |
| Shuffled mean | **0.0929** (MEASURED) |
| Shuffled 2.5th percentile | **0.0812** (MEASURED) |
| Shuffled 97.5th percentile | **0.1045** (MEASURED) |
| Empirical p-value | **0.0050** (MEASURED) |

**Interpretation**: The real asymmetry (0.62) is far outside the 95% CI of the
shuffled null distribution [0.08, 0.10]. The p-value of 0.005 rejects the null
hypothesis that the ordering is random at the 1% significance level.

**What this shows**: The technique order within sighting lists carries
statistically significant directional signal — transitions A→B and B→A occur
at very different rates. (MEASURED)

**What this does NOT show**: We cannot conclude that the list order represents
physical chronological execution order. The sighting data aggregates co-occurring
techniques from incident reports, and the original reporting order may reflect
analyst conventions, kill-chain-aware tagging, or automated tool output rather
than microsecond-level execution timestamps. (ASSUMPTION — caveat)

---

## 5. Stage B: Prediction Baselines (Validation Only)

All counts from train sequences, weighted by duplicate count. Evaluation strictly
on validation set. Test set never touched.

### 5.1 Overall Metrics

| Model | uw_top1 | uw_top3 | uw_top5 | uw_MRR | wt_top1 | wt_top3 | wt_top5 | wt_MRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **B1_Majority** | 0.1001 | 0.2453 | 0.3728 | 0.2510 | 0.0761 | 0.1960 | 0.2729 | 0.2110 |
| **B2_UnigramPrior** | 0.0432 | 0.0998 | 0.1084 | 0.1115 | 0.0257 | 0.1171 | 0.1204 | 0.1002 |
| **B3_Markov1 (k=0.001)** | 0.1661 | 0.4308 | 0.5875 | 0.3558 | 0.3715 | 0.6746 | 0.7586 | 0.5515 |
| **B4_Markov2 (m=5)** | 0.2839 | 0.5191 | 0.6538 | 0.4415 | 0.4850 | 0.7324 | 0.8097 | 0.6288 |

(MEASURED — `results/baselines_results.csv`)

**Tuned hyperparameters** (on validation MRR):
- Markov-1 add-k: k = 0.001 (all tested k values gave identical MRR; smallest selected). (MEASURED)
- Markov-2 back-off threshold: m = 5 (best uw_MRR = 0.4415 on validation). (MEASURED)

### 5.2 Majority Baseline from Train Labels

The majority class in training labels (dup-weighted) is **T1070** at **7.56%**.
(MEASURED)

The old figure of 6.22% came from all rows (not train-only) and used uncollapsed
subtechniques. It must not be reused. (MEASURED — conflict resolved)

### 5.3 Metrics by Prefix Length (Validation)

| Model | Prefix Len | uw_top1 | wt_top1 | uw_top3 | wt_top3 | uw_top5 | wt_top5 | uw_MRR | wt_MRR |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1_Majority | 2 | 0.0917 | 0.0525 | 0.2550 | 0.1617 | 0.4183 | 0.2768 | 0.2554 | 0.1834 |
| B1_Majority | 3 | 0.1046 | 0.0483 | 0.2606 | 0.1450 | 0.3817 | 0.2058 | 0.2594 | 0.1906 |
| B1_Majority | 4 | 0.1153 | 0.0826 | 0.2683 | 0.2184 | 0.3792 | 0.2787 | 0.2634 | 0.2161 |
| B1_Majority | 5+ | 0.0964 | 0.1051 | 0.2259 | 0.2381 | 0.3466 | 0.3097 | 0.2409 | 0.2386 |
| B2_UnigramPrior | 2 | 0.0440 | 0.0149 | 0.1101 | 0.0447 | 0.1229 | 0.0489 | 0.1176 | 0.0841 |
| B2_UnigramPrior | 3 | 0.0550 | 0.0358 | 0.1138 | 0.3395 | 0.1193 | 0.3413 | 0.1249 | 0.1741 |
| B2_UnigramPrior | 4 | 0.0399 | 0.0180 | 0.0887 | 0.0503 | 0.0998 | 0.0539 | 0.1062 | 0.0736 |
| B2_UnigramPrior | 5+ | 0.0389 | 0.0301 | 0.0931 | 0.0573 | 0.1004 | 0.0610 | 0.1050 | 0.0775 |
| B3_Markov1 | 2 | 0.1780 | 0.4499 | 0.4165 | 0.5961 | 0.5578 | 0.6641 | 0.3569 | 0.5693 |
| B3_Markov1 | 3 | 0.1908 | 0.1963 | 0.4624 | 0.7035 | 0.6110 | 0.7858 | 0.3802 | 0.4750 |
| B3_Markov1 | 4 | 0.1596 | 0.5151 | 0.4124 | 0.7091 | 0.5987 | 0.8089 | 0.3498 | 0.6420 |
| B3_Markov1 | 5+ | 0.1522 | 0.3572 | 0.4300 | 0.6880 | 0.5862 | 0.7747 | 0.3468 | 0.5410 |
| B4_Markov2 | 2 | 0.2881 | 0.5227 | 0.5284 | 0.7291 | 0.6642 | 0.7959 | 0.4434 | 0.6373 |
| B4_Markov2 | 3 | 0.2734 | 0.3222 | 0.5138 | 0.7070 | 0.6606 | 0.7936 | 0.4344 | 0.5401 |
| B4_Markov2 | 4 | 0.2838 | 0.5704 | 0.5277 | 0.7680 | 0.6718 | 0.8534 | 0.4501 | 0.6899 |
| B4_Markov2 | 5+ | 0.2866 | 0.5192 | 0.5142 | 0.7317 | 0.6397 | 0.8056 | 0.4406 | 0.6475 |

(MEASURED — `results/baselines_by_prefix_len.csv`)

### 5.4 Unseen Context Handling

| Model | Handling |
| :--- | :--- |
| **B1_Majority** | Always predicts the single most frequent train label. Context-independent. |
| **B2_UnigramPrior** | Always predicts from singleton global prior. Context-independent. |
| **B3_Markov1 (k=0.001)** | If last technique not seen in train bigrams, falls back to singleton unigram prior. Otherwise uses add-k=0.001 smoothing. |
| **B4_Markov2 (m=5)** | If bigram context has < 5 observations, backs off to 1st-order Markov. If that context is also unseen, falls back to singleton unigram prior. |

(ASSUMPTION — design choices; follows the Bayesian back-off structure
described in Li et al., DeepAG, IEEE TDSC 2023)
