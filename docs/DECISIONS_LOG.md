# Project Decisions & Viva Defense Log

This document serves as the official, running audit log of all architectural, methodological, and experimental design decisions made in the `attack-sequence-mining` project. Every decision records context, alternatives considered, justification for defense, code location, and validation status.

---

### Decision Log Summary Table

| ID | Topic | Decision | Primary Rationale / Viva Defense | Code Location | Validation Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **DEC-01** | Technique Granularity | Collapse subtechniques to parent techniques (`Txxxx`) | Mitigates severe sparsity across 378 fine-grained IDs; standardizes on 185 parent tactics. | `src/core/preprocessing.py`, `src/core/split.py` | Validated |
| **DEC-02** | Data Splitting | 60/20/20 whole-sequence stratified split by length buckets `{4, 5, 6, 7+}` | Prevents sequence contamination across splits; stratifies campaign complexity. | `src/core/split.py` | Validated (0 overlap) |
| **DEC-03** | Intra-Sequence Deduplication | First-occurrence deduplication (`[A, B, A, C] -> [A, B, C]`) | Preserves earliest appearance of tactic without cyclic loops corrupting prefix Markov transitions. | `src/core/preprocessing.py` | Validated |
| **DEC-04** | Singletons Handling | Retain length=1 rows strictly for global unigram prior $P(T)$ | Prevents skewing sequence mining while providing unigram smoothing base rates for unseen contexts. | `src/models/baselines.py` | Validated |
| **DEC-05** | Order Signal Testing | Permutation test over top-30 technique pairs with 200 shuffles | Tests statistical directionality without falsely claiming physical chronological execution. | `src/eda/order_signal.py` | Validated ($p=0.005$) |
| **DEC-06** | Order Artifact Control | Drop monotonic sequences & check ascending ID share among strong pairs | Confirms that high asymmetry is not an artifact of ascending technique ID sorting (measured 55.0%). | `src/eda/order_signal_control.py` | Validated ($p=0.005$) |
| **DEC-07** | Frequent Itemset Mining | FP-Growth on unordered itemsets at supports 0.01, 0.03, 0.05 ($\text{conf} \ge 0.5$) | Mines co-occurrence rules; benchmarks against Apriori; supports justified by dataset sequence counts. | `src/mining/association_rules.py` | Validated |
| **DEC-08** | Sequential Mining Engine | Implement custom PrefixSpan using projected databases | Eliminates external package dependencies in production; complies with course requirements. | `src/mining/prefixspan.py` | Validated |
| **DEC-09** | PrefixSpan Verification | Automated test comparing custom PrefixSpan to pip package on sample & full train | Proves exact pattern and support count equivalence before using mined rules in predictors. | `tests/test_prefixspan.py` | Validated (Exact match) |
| **DEC-11** | Evaluation Invariant | Headline metrics unweighted over unique sequences; test set isolated | Prevents dominance by repeated commodity incidents; ensures fair generalization benchmarking. | `src/models/baselines.py`, `src/models/prefixspan_predictor.py` | Validated |
| **DEC-12** | Negative Rule Mining | Mine rules of form $X \text{ and not } M \implies Y$ with Fisher tests & BH FDR | Fulfills 'beyond class' requirement; accounts for absence of expected companion tactics. | `src/mining/negative_rules.py` | Validated ($p_{\text{adj}} \le 0.05$) |
| **DEC-13** | Negative Rule Predictor | Model B6: Inject top $n_{\text{inject}}$ negative rules into B5 PrefixSpan ranking | Boosts validation unweighted MRR from 0.4400 to 0.4612 and top-1 from 0.2824 to 0.2965. | `src/models/negative_predictor.py` | Validated (Tuned on val) |
| **DEC-14** | LightGBM Feature Predictor | Boosted-tree ranking with prefix presence, last-4 positions, GroupKFold early stopping | Outperforms Markov-2 and PrefixSpan on validation unique-sequence MRR (0.5407). | `src/models/lightgbm_predictor.py` | Validated |
| **DEC-15** | Sequence-Balanced Macro Metrics | Sequence-balanced macro recall and macro F1 over validation-observed classes | Resolves class imbalance without distortion from long sequences generating more prefixes. | `src/core/evaluation_metrics.py`, `src/eval/compare_validation_models.py` | Validated |
| **DEC-16** | XGBoost Optuna Comparison | Multiclass XGBoost tuned via 8-trial Optuna GroupKFold | Validation unique-sequence MRR 0.5025; provides tree-based model diversity. | `src/models/xgboost_predictor.py` | Validated |

---

### Detailed Decision Records

#### DEC-01: Collapsing Subtechniques to Parent Techniques
- **Context**: The raw dataset contains 378 distinct tokens, including subtechniques like `T1059.001`, `T1059.003`.
- **Alternatives Considered**: 
  1. Train on raw subtechniques. (Rejected: extreme vocabulary sparsity; majority of subtechniques have $<5$ occurrences).
  2. Hierarchy-aware tree embeddings. (Rejected: adds unnecessary model complexity for baselines).
- **Decision**: Collapse all subtechniques to parent IDs (`Txxxx`). 185 unique parent techniques remain.
- **Viva Defense**: High-level attack progression is governed by parent tactics and techniques. Collapsing increases sample support per state and ensures reliable transition probabilities.

#### DEC-02: 60/20/20 Whole-Sequence Stratified Split
- **Context**: How to divide multi-stage sequences ($\ge 4$ techniques) into Train, Validation, and Test.
- **Alternatives Considered**:
  1. Random prefix-level split. (Rejected: creates severe data leakage because prefixes from the same sequence would appear in both train and test).
  2. Temporal split by incident year. (Rejected: dataset has corrupted dates and reporting delay biases).
- **Decision**: Whole-sequence split (60% train, 20% val, 20% test) stratified across sequence length buckets $\{4, 5, 6, 7+\}$ with `RANDOM_SEED = 42`.
- **Validation**: Zero overlap of sequences across sets (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).

#### DEC-03: First-Occurrence Deduplication Rule
- **Context**: In 278 rows, attackers re-executed the same technique later in the incident (e.g. `[T1055, T1082, T1055]`).
- **Alternatives Considered**:
  1. Keep all duplicates. (Creates infinite self-loops in Markov transitions).
  2. Consecutive-only deduplication (`deduplicate_consecutive`). (Leaves non-consecutive cycles).
- **Decision**: Use `deduplicate_first_occurrence()`: retain only the first appearance of any technique.
- **Viva Defense**: In sequential pattern mining, each technique represents the initial achievement of a capability within that intrusion campaign.

#### DEC-04: Handling Singleton Rows for Global Prior
- **Context**: 757,291 rows (72.2%) are single isolated technique detections.
- **Alternatives Considered**:
  1. Discard completely. (Wastes vast epidemiological data on base occurrence rates).
  2. Include in sequential mining. (Degenerates sequence length distributions).
- **Decision**: Isolate multi-stage attacks ($\ge 4$) for mining, but use singletons to fit the global unigram prior $P(T)$ used for unseen context smoothing.
- **Viva Defense**: Follows Bayesian back-off principles: empirical unigram base rates serve as the weakly informative prior when higher-order Markov context is completely unseen.

#### DEC-05: Permutation Test for Order Signal
- **Context**: Raw incident lists lack sub-second timestamps. Does list ordering carry genuine directional signal?
- **Alternatives Considered**:
  1. Assume ordering is physical execution time without verification. (Methodologically unsound).
  2. Ignore ordering and treat all sightings as unordered sets. (Abandons sequential prediction).
- **Decision**: Implement a permutation test measuring mean pairwise transition asymmetry against a null distribution of 200 random shuffles.
- **Finding**: Real asymmetry 0.6199 vs shuffled 0.0929 ($p = 0.0050$). Reject null hypothesis of random ordering.
- **Viva Defense**: We demonstrate that reporting order contains strong statistical directionality ($A \rightarrow B \neq B \rightarrow A$), while explicitly avoiding the unsubstantiated claim that it represents physical execution chronology.

#### DEC-06: Controlled Permutation Test (Ruling Out ID-Sorting Bias)
- **Context**: Sighting lists might be artificially ordered by numeric technique ID (e.g. `T1018` before `T1059`).
- **Alternatives Considered**:
  1. Accept initial permutation test without controlling for ID ordering. (Vulnerable to reviewer critique).
- **Decision**: Drop all monotonic sequences (IDs strictly ascending or descending) and re-test. Also measure the proportion of strong asymmetric pairs following ascending ID order.
- **Finding**: Dropping monotonic sequences retained 1,572 unique sequences; real asymmetry was 0.5918 ($p = 0.0050$). Among strong pairs ($>0.5$), exactly 55.0% (44/80) followed ascending ID order—consistent with random 50/50 chance.
- **Viva Defense**: Proves conclusively that directional signal is driven by adversary behavioral dependencies, not by automated ID-sorting pipelines.

#### DEC-07: FP-Growth vs Apriori Itemset Mining
- **Context**: Stage C requires itemset association rule mining on train itemsets with three justified support levels and runtime comparison.
- **Decision**:
  - Supports: 0.01 (1%), 0.03 (3%), 0.05 (5%) based on train transaction count ($N=4,291$).
  - Max pattern size: 4. Minimum confidence: $\ge 0.5$.
  - Benchmark against Apriori with 3-run timing averages.
- **Finding**: Apriori proved slightly faster on this sparse, small-transaction dataset ($0.015\text{s}$ vs $0.023\text{s}$ at 1%), demonstrating that FP-tree construction overhead exceeds candidate generation costs at small $N$.

#### DEC-08 & DEC-09: Custom PrefixSpan Implementation & Verification Test
- **Context**: Need sequential pattern mining on ordered subsequences without using external libraries in production.
- **Decision**:
  - Custom recursive projected-database algorithm in `src/prefixspan.py` supporting duplicate sequence weights.
  - Strict verification test suite in `src/test_prefixspan.py` that imports pip `prefixspan` strictly within the test harness.
- **Verification**: Exact match achieved across all patterns and support counts on both toy database and the full training set (270 patterns at 5% support).

#### DEC-10: Pattern-Based Next-Step Predictor Architecture
- **Context**: Building a next-step technique predictor from PrefixSpan sequential rules.
- **Decision**:
  - Rule structure: Antecedent $A \rightarrow$ Consequent $x$.
  - Matching condition: Antecedent ends at the last technique of the prefix ($A[-1] == p[-1]$) and $A[:-1]$ is a subsequence of $p[:-1]$.
  - Back-off: If no rule fires or to fill out full top-$k$ rankings, back off to tuned Second-Order Markov (**Markov-2**, $m=5$).
  - Validation Tuning: Tuned `min_support` and `min_conf` on validation unweighted MRR. Optimal: `min_support=0.01`, `min_conf=0.7`.
- **Finding**: On duplicate-weighted traffic, PrefixSpan predictor beats Markov-2 on top-1 accuracy (0.5561 vs 0.4850, +7.11%) and MRR (0.6658 vs 0.6288, +3.70%).

#### DEC-11: Metric Priority & Test Set Isolation
- **Context**: Avoiding metric gaming and data leakage.
- **Decision**:
  - Primary headline metrics: Unweighted top-1, top-3, top-5, and MRR over unique validation sequences.
  - Secondary metrics: Duplicate-weighted metrics.
  - Absolute ban on inspecting or testing the test set until user explicitly authorizes "go" after STOP 3.
- **Viva Defense**: Unweighted metrics treat each distinct attack pattern equally, preventing a single widespread automated malware family from artificially inflating accuracy.

#### DEC-12: Negative Association Rule Mining ($X \text{ and not } M \implies Y$)
- **Context**: Project requires contributing an element beyond class (along with our own PrefixSpan).
- **Decision**:
  - Mine rules where $X$ is present in prefix, but companion technique $M$ is absent, implying next technique $Y$.
  - Evaluate statistical significance via $2 \times 2$ Fisher exact tests.
  - Apply Benjamini-Hochberg (BH) False Discovery Rate (FDR) control at $\alpha \in \{0.01, 0.05\}$.
- **Viva Defense**: Rigorous statistical control ensures only non-spurious absent-tactic rules are retained, capturing alternative attacker pathways when expected steps are skipped.

#### DEC-13: Model B6 Negative Rule Injection Predictor
- **Context**: Evaluating whether negative rules provide complementary predictive signal when combined with PrefixSpan sequential rules.
- **Decision**:
  - Construct Model B6: Take B5 PrefixSpan ranking (with Markov-2 back-off) and inject top $n_{\text{inject}} \in \{1, 2, 3\}$ candidate consequents from fired negative rules.
  - 48-configuration tuning grid on validation unweighted MRR.
- **Finding**: Optimal config (`min_expect=0.6, min_count=5, min_lift=1.5, alpha=0.05, n_inject=3`) achieves validation unweighted MRR of **0.4612** (vs B4 Markov-2's 0.4415 and B5 PrefixSpan's 0.4400), with unweighted top-1 increasing to **0.2965** (vs B4's 0.2839) and unweighted top-5 reaching **0.6913** (vs B4's 0.6538).


#### DEC-14: LightGBM Feature-Based Predictor and Hyperparameter Search
- **Context**: Compare a boosted-tree model with Markov and sequence-mining predictors on next-technique ranking.
- **Features**: Technique presence in each prefix, the last four ordered techniques, and prefix length. Training prefix rows are weighted to sum to the source sequence's duplicate count.
- **Searches**: A 12-setting fixed grid, 12-trial Optuna TPE, and eight-candidate `HalvingRandomSearchCV`; grouped folds keep each complete sequence in one fold. Early stopping monitors sequence-level MRR on an inner group holdout (patience 30, max 400 trees). Halving allocates `n_estimators` as its resource, so early stopping applies on its final refit only.
- **Measured results**: Grid best validation unique-sequence MRR **0.5407** (best iteration 101); Optuna training-CV MRR **0.5110**, validation MRR **0.5375** (best iteration 149); halving training-CV MRR **0.5081**, validation MRR **0.5362** (best iteration 90). Full top-k results and configuration details are in `docs/LIGHTGBM_SEARCH_GUIDE.md` and the corresponding files under `results/`.
- **Limitations**: These are validation-stage model-selection results, not test estimates. The grid directly selected on validation and is especially optimistic. No held-out test or AIT-ADS data was loaded. Reported list order does not establish chronological causation.


## Results filename clarity (2026-10-20)

Renamed negative-rule and STOP3 validation outputs to identify their split, metric, and experiment purpose. Updated the generating scripts to write the new names. The split-half artifact is explicitly named as validation sensitivity analysis; it is not a test-set result. Existing CSV contents were preserved unchanged.


#### DEC-15: Sequence-Balanced Macro Metrics
- **Context**: Add class-sensitive metrics to compare common and rarer next-technique labels without changing the existing ranking task.
- **Decision**: Report macro recall and macro F1 over target labels observed in validation. Also report sequence-balanced versions by giving every prefix from a full sequence weight `1 / number_of_prefixes_in_sequence`; retain prefix-example and duplicate-weighted metrics as supporting views. Create a consolidated sequence-level comparison for B1-B6 and refreshed LightGBM grid/halving runs.
- **Validation**: Re-ran baseline, PrefixSpan, negative-rule, LightGBM grid, and HalvingRandomSearchCV validation workflows. Test data was not loaded. Optuna could not be rerun because its optional package is not installed; its existing result is excluded from the refreshed comparison.
- **Viva justification**: Macro metrics expose class imbalance; sequence balancing prevents longer reported sequences from dominating because they generate more prefixes. They remain secondary to top-k and MRR because classes with sparse support have noisy estimates.


#### DEC-16: XGBoost Optuna Model Comparison
- **Context**: Add a gradient-boosted tree alternative and avoid deciding from LightGBM alone.
- **Decision**: Train multiclass XGBoost using the shared prefix features. Use Optuna TPE (8 trials), 3-fold GroupKFold by full sequence, sequence-level MRR as tuning objective, and an inner group holdout with early stopping on multiclass log loss. Report sequence-balanced top-k, MRR, macro recall, and macro F1 on the external validation set.
- **Measured result**: Best training CV sequence MRR 0.4656; external validation sequence top-1 0.3356, top-3 0.5822, top-5 0.7248, MRR 0.5025, macro recall 0.1505, macro F1 0.1446; best iteration 115. The test set was not loaded.
- **Viva justification**: Grouped folds prevent prefixes from a single full sequence crossing train/validation boundaries. Macro metrics complement ranking scores under class imbalance. Search is a modest initial comparison (8 trials), and validation is still used for model selection rather than final unbiased testing.
