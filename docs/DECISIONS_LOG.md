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
| **DEC-13** | Negative Rule Predictor | Model B6: Inject top $n_{\text{inject}}$ negative rules into B5 PrefixSpan ranking | Mined rules are statistically significant, but injecting them does **not** improve prediction. Paired bootstrap CI contains zero; the sign reverses once prefix masking is applied to all models. | `src/models/negative_predictor.py` | **Not supported** (negative result) |
| **DEC-14** | LightGBM Feature Predictor | Boosted-tree ranking with prefix presence, last-4 positions, GroupKFold early stopping | Outperforms Markov-2 and PrefixSpan on validation unique-sequence MRR (0.5407). | `src/models/lightgbm_predictor.py` | Validated |
| **DEC-15** | Sequence-Balanced Macro Metrics | Sequence-balanced macro recall and macro F1 over validation-observed classes | Resolves class imbalance without distortion from long sequences generating more prefixes. | `src/core/evaluation_metrics.py`, `src/eval/compare_validation_models.py` | Validated |
| **DEC-16** | XGBoost Optuna Comparison | Multiclass XGBoost tuned via Optuna GroupKFold, at protocol parity with LightGBM | Original 8-trial run was handicapped by a log-loss stopping rule and a 160-tree ceiling; re-run with sequence-MRR stopping and leaf-wise growth offered. | `src/models/xgboost_predictor.py` | Re-run 2026-10-25 |
| **DEC-18** | Cross-Validated Protocol | 5-fold grouped, length-stratified CV over a development set merging train+val; test sealed | A single 545-sequence holdout has SE 0.0090 and cannot resolve differences below 0.0355; the three LightGBM searches differed by 0.0047. CV halves SE; paired comparison reaches a 0.0069 half-width. | `src/core/cv.py`, `src/eval/cv_harness.py` | Validated |
| **DEC-19** | Accuracy Ceiling Investigation | Documented negative results: Naive Bayes, pattern features, candidate ranker, stacking, blending, graph similarity | Boosted trees already exceed the context-lookup ceiling by 0.106, so the limit is not Bayes-irreducible; but mined patterns are deterministic functions of an already-complete prefix encoding and add nothing. | `src/eval/diagnostics.py`, `docs/ACCURACY_INVESTIGATION.md` | Validated (negative) |
| **DEC-20** | Tactic Abstraction | Multi-hot over 14 ATT&CK tactics from the MITRE tactic pages (91.7% vocabulary coverage); tested as input features and as hierarchical output reweighting | Tactics are the one tested feature source carrying information not derivable from the sightings data, and they target class sparsity directly. Measured effect on MRR is +0.0013 across CV seeds, indistinguishable from zero; output-side reweighting actively hurts. | `src/core/tactic_map.py`, `src/eval/tactic_experiments.py` | **Not supported** (negative) |
| **DEC-21** | Graph + Tactic + Interaction Pipeline | Node2Vec graph features (A), tactic transitions (B), recency interactions (C), masking (D), XGBRanker head | Built and measured in full. A+B+C add +0.0012 MRR (CI straddles zero); both ranker variants are significantly worse than the 511-feature softprob model (-0.036, CI [-0.043, -0.029]). Only D delivered. | `src/models/graph_features.py`, `src/models/pipeline_ranker.py`, `src/eval/pipeline_final.py` | **Not supported** (negative) |
| **DEC-17** | Prefix Candidate Masking | Rule out candidates already present in the prefix before ranking, for every model | First-occurrence dedup (DEC-03) makes the target impossible to be a prefix member (verified 0/8,329 train, 0/2,776 val); leaving them in wasted top-k slots and silently advantaged models whose features could learn the constraint. | `src/core/config.py`, `src/models/baselines.py`, `src/models/lightgbm_predictor.py` | Validated |

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
- **Original finding (2026-10-20, superseded)**: the selected config reached validation unweighted MRR **0.4612** vs B5's 0.4400, and this was logged as a validated improvement.
- **Correction (2026-10-25)**: that claim was wrong on two counts, both visible in our own outputs.
  1. **It was never statistically supported.** The paired bootstrap over the 545 validation sequences gave mean difference +0.0108 with 95% CI **[-0.0055, +0.0268]** — the interval contains zero. The configuration was also the winner of a 48-cell grid scored on the same validation MRR, so even the point estimate is optimistic.
  2. **The effect reverses once a confound is removed.** B6 gained partly by displacing candidates already present in the prefix, which first-occurrence deduplication (DEC-03) guarantees can never be the target. With that masking applied to *every* model (DEC-17), B6 scores unweighted MRR **0.4727 vs B5's 0.4755** and sequence MRR **0.4540 vs 0.4588** — worse on both, with bootstrap mean difference **-0.0047**, CI **[-0.0176, +0.0078]**, and `b6_beats_b5_uw_mrr = False`.
- **Collateral cost, present in both runs**: injection roughly halves class coverage. Sequence-balanced macro recall drops from **0.1890 (B5) to 0.0783 (B6)** and duplicate-weighted MRR from **0.6817 to 0.5678**. Optimising a single unweighted ranking metric concentrated predictions on fewer classes.
- **Rule selectivity problem**: the chosen rule set fires on **2,770 of 2,776** validation prefixes (99.8%). A condition that holds almost always cannot discriminate between candidates, which is the mechanism behind the null result.
- **What still stands**: the *mining* contribution is intact. Rules of the form $X \wedge \neg M \Rightarrow Y$ are discovered with Fisher exact tests under Benjamini-Hochberg FDR control, and significant rules exist. The negative result is specifically that **injecting them into a ranking does not improve next-technique prediction on this dataset**.
- **Viva defense**: report this as a negative result with the evidence that produced it — the bootstrap, the B7 positive control (which also sits level with B5), and the sign reversal under masking. A rule set that is statistically significant but not predictively useful is a legitimate, well-evidenced finding; claiming the 0.4612 figure as an improvement would not survive the bootstrap we ourselves ran.


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
- **Validation**: Re-ran baseline, PrefixSpan, negative-rule, LightGBM grid, and HalvingRandomSearchCV validation workflows. Test data was not loaded. Optuna could not be rerun at the time because its optional package was not installed. (Superseded on 2026-10-25: optuna was installed and both Optuna searches re-ran; `compare_validation_models.py` now lists every search that produced a summary, so `LightGBM_Optuna` is included.)
- **Viva justification**: Macro metrics expose class imbalance; sequence balancing prevents longer reported sequences from dominating because they generate more prefixes. They remain secondary to top-k and MRR because classes with sparse support have noisy estimates.


#### DEC-16: XGBoost Optuna Model Comparison
- **Context**: Add a gradient-boosted tree alternative and avoid deciding from LightGBM alone.
- **Decision**: Train multiclass XGBoost using the shared prefix features, with Optuna TPE over 3-fold GroupKFold by full sequence and sequence-level MRR as the tuning objective.
- **Original run (2026-10-20, superseded)**: 8 trials, early stopping on **multiclass log loss** under a **160-tree ceiling**. Best training CV sequence MRR 0.4656; validation sequence MRR 0.5025, macro recall 0.1505; best iteration 115. This was logged as showing XGBoost trailing LightGBM by about 0.038 MRR.
- **Defect found (2026-10-25)**: the early-stopping rule did not measure the reported objective, and the ceiling was binding.
  - Log loss is non-monotone on this 94-class problem. On the inner holdout it dipped at round 114, firing patience-20 early stopping at round 135 and reporting 115 trees — while its own global minimum lay at round **196** and sequence MRR was still rising at round 400. The stop fired on noise, beneath a ceiling that would have truncated it anyway.
  - Measured divergence over rounds 115/134/196/400 on that holdout: log loss 2.2733/2.2781/2.2556/2.2699 against sequence MRR 0.4736/0.4741/0.4856/0.4923.
  - Because the same rule ran inside every CV fold, the search ranked configurations by truncated models, so the selected hyperparameters were not reliably best either.
  - Depthwise growth at `max_depth=4` also caps a tree at 16 leaves, against the 63 LightGBM may use, so the two models were not offered equivalent capacity.
- **Correction**: stop on sequence MRR, raise the ceiling to 400 with patience 30, offer `lossguide` growth with `max_leaves` alongside `depthwise`, and match LightGBM's fold count (3) and trial budget (12) so neither harness is favoured.
- **Measured result after correction**: best training CV sequence MRR **0.4830**; validation sequence top-1 **0.3547**, top-3 **0.6051**, top-5 **0.7456**, MRR **0.5233**, macro recall **0.1963**, macro F1 **0.1880**; best iteration 109. The test set was not loaded. Selected config: `depthwise, max_depth=4, learning_rate=0.1866`. The search had `lossguide` available and did not pick it — the gain came from the stopping metric, which let a much higher learning rate (0.187 vs 0.054) be judged on ranking quality rather than calibration.
- **Consequence for the comparison**: the gap to the best LightGBM configuration narrows from ~0.038 to ~0.018 MRR, which is inside this split's uncertainty. The earlier "LightGBM clearly beats XGBoost" reading was a property of the harness, not of the algorithms.
- **Viva justification**: Grouped folds prevent prefixes from one full sequence crossing fold boundaries. Macro metrics complement ranking scores under class imbalance. The general lesson is the defensible one: **select on the metric you report**. Early stopping on a proper scoring rule while reporting a ranking metric silently truncated the model, and the noise in that scoring rule made the truncation point arbitrary.
- **Still outstanding**: CV folds score with `refit=False`, so a fold's model is fit on 85% of its training portion with only the labels present there, while the final model uses 100% — training CV (0.4830) is therefore systematically pessimistic against validation (0.5233) and the two are not on the same scale.


#### DEC-17: Prefix Candidate Masking (2026-10-25)
- **Context**: First-occurrence deduplication (DEC-03) keeps only the first appearance of a technique in a sequence. A direct consequence is that the prediction target can never be a member of its own prefix. Verified: **0 of 8,329** training examples and **0 of 2,776** validation examples violate this.
- **Problem found**: no model exploited the constraint explicitly. The Markov and PrefixSpan predictors ranked over the full 173-technique vocabulary and regularly spent top-k slots on techniques the prefix had already consumed. The boosted-tree models, by contrast, receive `seen_*` presence indicators and could learn the constraint from data — so part of their measured advantage was them learning a property of our label construction that the baselines were never given.
- **Alternatives considered**:
  1. Leave the behaviour as-is. (Rejected: the model comparison stays confounded, and the baselines are handicapped for a reason unrelated to their modelling assumptions.)
  2. Add a "technique already seen" feature to the Markov models. (Rejected: the constraint is exact, not probabilistic; a hard mask is the correct encoding and needs no estimation.)
  3. Remove the dedup rule instead. (Rejected: DEC-03 is load-bearing for the Markov transition estimates; changing it would invalidate Stages B-D.)
- **Decision**: rule out prefix members before ranking, in every evaluator, behind `config.MASK_SEEN_TECHNIQUES` so unmasked numbers remain reproducible. Applied at one choke point per evaluator: `mask_seen()` inside `evaluate_baseline` for B1-B7, and `_seen_mask()` / `_ranked_labels()` for the tree models, which read the mask off the `seen_*` feature block. Model-selection metrics (CV scoring and early stopping) use the masked ranking too, so tuning and reporting optimise the same quantity.
- **Measured effect** (validation, sequence-balanced MRR): B4 Markov-2 **0.4312 to 0.4583**; B5 PrefixSpan **0.4311 to 0.4588**; B1 Majority **0.2497 to 0.3254**. The gain for the tree models is negligible (XGBoost +0.0008), confirming they had already learned the constraint.
- **Consequences for earlier conclusions**:
  - The tree models' margin over Markov-2 shrinks, because roughly a quarter of it was this artifact rather than better modelling.
  - B6's negative-rule gain **disappears and reverses sign** (see DEC-13), since part of that gain was incidentally displacing already-seen candidates.
- **Viva defense**: this is a property of the task as we constructed it, not a claim about adversary behaviour — real attackers do repeat techniques, and a deployment on non-deduplicated traces must drop the mask. It is reported as a consequence of DEC-03 rather than as a modelling contribution. The honest framing is that it removes a confound from the comparison; it does not make the predictors better at forecasting real intrusions.


#### DEC-18: Cross-Validated Evaluation Protocol (2026-10-25)
- **Context**: Model selection and comparative reporting both used the single 545-sequence validation holdout, and seven of the ten rows in the comparison table had hyperparameters chosen on the split they were scored on.
- **Problem measured**: per-sequence MRR has sd 0.2093, so a 545-sequence holdout carries SE 0.0090, a 95% CI half-width of 0.0176, and a smallest unpaired detectable difference of **0.0355**. The observed spread of the three LightGBM searches was **0.0047**, which is 7.5x below that threshold. Those rankings were not measurable.
- **Alternatives considered**:
  1. Keep the single holdout and simply report CIs. (Rejected: honest, but still cannot resolve the differences being chased.)
  2. Enlarge the validation split at the expense of test. (Rejected: the final test estimate is already at a 0.0176 half-width; shrinking it makes the one unbiased number worse.)
  3. Leave-one-sequence-out CV. (Rejected: 2,178 refits per configuration for a variance gain that 5 folds already largely capture.)
- **Decision**: merge the original train and validation splits into a development set of **2,178 unique sequences** and use `N_CV_FOLDS = 5` `StratifiedGroupKFold`, grouped by full sequence and stratified by the same length buckets as DEC-02, for all model selection and all comparative reporting. Compare models by **paired per-sequence differences** with a bootstrap over sequences. Keep the 543 test sequences sealed for one confirmatory run.
- **Validation**: invariants are machine-checked by `verify_folds` -- no sequence contributes prefixes to both sides of a fold, folds partition every development example exactly once, and per-fold length-bucket shares agree to four decimals. Reproducing known models through the harness gives sequence MRR B1 0.3069, B4 0.4487, LightGBM 0.5386, each slightly below its published holdout figure, which is the expected optimism of a split that was tuned on. Paired B4-vs-LightGBM is -0.0898 with CI [-0.0967, -0.0830], a half-width of **0.0069** against the previous 0.0355.
- **Consequence**: each fold trains on 1,742 sequences rather than 1,633, so absolute numbers are not directly comparable to the earlier tables. Masking is enforced centrally in the harness so no adapter can omit it or apply it differently.
- **Viva defense**: the protocol makes extensive experimentation safe. Selection happens inside the development folds, the sealed test split is spent once, and the reported precision is stated rather than assumed. The weaker point to concede is that the final test estimate still has a 0.0176 half-width, so the test run can confirm large effects (trees over Markov, about +0.09) but can never adjudicate differences of 0.015 such as LightGBM against XGBoost.

#### DEC-19: Accuracy Ceiling Investigation (2026-10-25)
- **Context**: after the DEC-16 and DEC-17 corrections the four best models clustered within 0.0047 MRR, so the binding constraint needed identifying before more modelling.
- **Method and findings**: full detail and reproduction commands in `docs/ACCURACY_INVESTIGATION.md`; measurements in `results/diagnostics/`.
  - **Not an irreducible ceiling.** A leave-one-out empirical-Bayes estimate caps context-lookup prediction at 0.4474 MRR (best context: last two techniques). LightGBM reaches 0.5534 on the same fold, exceeding it by **0.106**. The trees generalise across contexts rather than memorising them.
  - **The target is multi-modal.** Reused last-2 contexts are followed by 6.02 distinct techniques on average, which caps top-1 for any method. Reporting consequence: MRR and top-5 are the defensible headline; top-1 should not be called "accuracy".
  - **Richer context hurts.** The lookup ceiling falls from 0.4474 (last-2) to 0.3420 (full ordered prefix) because context reuse collapses from 62% to 16%.
  - **Naive Bayes is not competitive**: best variant 0.4654 MRR, at or below Markov-2 and about 0.09 below LightGBM. Its independence assumption fails precisely where DEC-07 measured correlated co-occurrence, and its bag-of-techniques form discards the order DEC-05 showed to be informative.
  - **Complementarity exists but fixed blending cannot use it.** Markov-2 is uniquely correct on 7.1% of the examples LightGBM misses (oracle union 0.4703 top-1 against 0.3830), yet the best fixed-weight blend gains **+0.0005**, with the weight swept on the evaluation data.
  - **Mined patterns add nothing as features**: -0.0059 from 478 pattern and context columns. A PrefixSpan antecedent match is a deterministic function of the prefix, and the 511 `seen_*`/`last-k` features already determine the prefix completely.
  - **The candidate-ranker reformulation lost about 0.04 MRR** in every variant, collapsing into a reweighted Markov-2 (`m2_prob` gain 691 against 71 for the next feature). Candidate identity added +0.0006 and graph similarity +0.0037.
- **Caveats recorded rather than buried**: the ranker is untuned while LightGBM has a 12-configuration grid behind it, so that comparison is not yet like-for-like; and the stacking result is anomalous and unexplained, since `lgb_prob` scores 0.5534 alone yet receives gain 10.1 and degrades the ranker.
- **Correction logged**: an earlier claim that the ranker beat LightGBM on macro recall by 0.09 compared the ranker's fold-0 score against LightGBM's 5-fold mean. On the same fold the gap is about +0.03, within noise.
- **Viva defense**: this is a set of negative results obtained under a protocol that can actually detect the effects in question, which is a stronger position than an unreplicated positive. It also narrows the remaining hypotheses to external knowledge (DEC-20) and to the reporting-order limitation of the sightings data, which only the timestamped AIT-ADS traces can test.
- **What is explicitly NOT retracted**: the custom PrefixSpan implementation and its exact-match verification (DEC-08, DEC-09) and the negative-rule significance testing under BH FDR control (DEC-12). The finding is narrower: mined patterns are not useful *as predictive features for a model that already sees the full prefix*.

#### DEC-20: ATT&CK Tactic Mapping (2026-10-25)
- **Context**: tactic abstraction is the one proposed feature branch that introduces knowledge not derivable from the sightings data, and it attacks the measured class-sparsity bottleneck directly -- roughly 600 examples per tactic against about 89 per technique, with 47 of 94 techniques holding fewer than 5 examples.
- **Decision**: represent tactics as a **multi-hot vector over the 14 Enterprise tactics plus an explicit `tactic_unknown` flag**, never as a single tactic id. Techniques are genuinely multi-tactic: T1078 Valid Accounts spans Initial Access, Persistence, Privilege Escalation and Defense Evasion. Forcing one bucket per technique would inject label noise, and the unknown flag keeps "no tactic applies" distinguishable from "not recorded in our source".
- **Source investigation**: `config.py` carried a comment claiming ATT&CK mapping tables, but no table existed. The AIT-ADS Wazuh exports embed a `rule.mitre` block giving technique ids with their tactics, and `src/core/tactic_map.py` extracts it; measured coverage is only **8 techniques** of the 94-107 vocabulary, with mean 3.25 tactics each. Too sparse to use.
- **Mapping obtained**: `data/reference/attack_technique_tactics.csv`, 241 technique/tactic pairs over 212 techniques across all 14 tactics, read from the MITRE ATT&CK Enterprise tactic pages (provenance in `src/core/build_tactic_reference.py`). Coverage of our 109-technique vocabulary is **91.7%**; 9 unmapped ids get the explicit `tactic_unknown` flag.
- **Measured result (5-fold CV, paired over 2,178 sequences)**: prefix tactic features give +0.0032 MRR, CI [+0.0003, +0.0063]; hierarchical output reweighting gives -0.0032 at lambda 0.5, -0.0101 at lambda 1.0 and -0.0246 at lambda 2.0, all significantly **worse**. The `lambda = 0` row reproduces the baseline exactly, confirming the implementation.
- **Why the +0.0032 is not believed**: it was one of five comparisons against the same baseline (t = 2.11 against a Bonferroni threshold of 2.58, so it fails correction), and it does not replicate across fold partitions -- re-running under CV seeds 42/7/2024 gives +0.0032 / +0.0008 / -0.0001, mean **+0.0013**, with signs disagreeing and two of three not significant.
- **Status**: **not supported**. Tactic abstraction does not improve next-technique ranking on this dataset. Full detail in `docs/ACCURACY_INVESTIGATION.md` section 9.
- **Viva defense**: this was the best-motivated hypothesis in the investigation and the prediction was wrong, which is worth saying plainly. The seed-42 result would have been reported as a success under the earlier practice of selecting and reporting on a single split; it was caught by the same discipline that overturned DEC-13, applied to a result this project produced and wanted to be true. One thread is left open: the macro-recall gain (0.2300 to 0.2429) was not stability-checked, because macro recall is not a per-sequence quantity and does not fit the paired bootstrap.
