# What Limits Next-Technique Accuracy (2026-10-25)

Every number here is MEASURED from executed code. Reproduce with:

```bash
python src/core/cv.py                 # fold construction and invariant checks
python src/eval/diagnostics.py        # sections 2-5 below
```

Outputs land in `results/cv/` and `results/diagnostics/`. The test split is not
loaded by any script in this document.

---

## 1. The problem this document addresses

After correcting the XGBoost harness (DEC-16) and applying prefix masking
(DEC-17), the best models clustered tightly:

| Model | validation sequence MRR |
| :--- | ---: |
| LightGBM grid | 0.5410 |
| LightGBM Optuna | 0.5379 |
| LightGBM halving | 0.5363 |
| XGBoost Optuna | 0.5233 |

Three independent hyperparameter searches over the same features landed within
**0.0047** of each other. That is the signature of a saturated hypothesis class,
not of a tuning problem. The question is what the binding constraint actually is.

---

## 2. The evaluation could not see the differences being chased

Per-sequence MRR has standard deviation **0.2093** on this data. That fixes the
precision of every evaluation design:

| Design | Sequences | SE | 95% CI half-width | Min. unpaired detectable difference |
| :--- | ---: | ---: | ---: | ---: |
| Single validation holdout | 545 | 0.0090 | 0.0176 | **0.0355** |
| Final test split | 543 | 0.0090 | 0.0176 | 0.0356 |
| 5-fold CV over development set | 2,178 | 0.0045 | 0.0088 | 0.0178 |

(MEASURED — `results/diagnostics/resolving_power.csv`)

The spread among the three LightGBM searches was 0.0047, **7.5x below** what the
single holdout could resolve. Those rankings were noise. This is the first
finding and it invalidates several earlier conclusions rather than extending
them; the cross-validated protocol in DEC-18 exists because of it.

---

## 3. We are not near an irreducible ceiling

A leave-one-out empirical-Bayes estimate gives the best score achievable by
predicting from a context's observed successor distribution — the limit of what
Markov and PrefixSpan style lookup can reach. For each example, labels are
ranked by pooled `(context -> label)` counts with that example's own
contribution removed.

| Context | Contexts | Reuse rate | Distinct labels when reused | LOO top-1 | LOO MRR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| last technique | 98 | 71.4% | 12.43 | 0.2379 | 0.4047 |
| last 2 | 852 | 62.4% | 6.02 | 0.2973 | **0.4474** |
| last 3 | 3,325 | 46.0% | 3.05 | 0.3084 | 0.4224 |
| full ordered prefix | 7,365 | 15.6% | 2.43 | 0.2173 | 0.3420 |
| prefix as a set | 5,185 | 30.0% | 2.49 | 0.2983 | 0.4132 |

(MEASURED — `results/diagnostics/lookup_ceiling.csv`)

**LightGBM reaches 0.5534 sequence MRR on the same fold, which is 0.106 above
the best row in this table.** The boosted trees are not memorising contexts;
they pool statistical strength across similar ones and have already passed the
ceiling of lookup-based prediction. So there is no "we have hit the Bayes limit"
story, and feature or model work is not futile in principle.

Two consequences:

- **Richer context does not help by itself.** Moving from last-2 to the full
  ordered prefix *lowers* the lookup ceiling from 0.4474 to 0.3420, because
  reuse collapses from 62% to 16%. Sparsity beats specificity on this data.
- **The task is genuinely multi-modal.** Reused last-2 contexts are followed by
  **6.02 distinct techniques on average**. Several next techniques are
  legitimately plausible, so top-1 has a low intrinsic bound no model can lift.
  This is why top-1 sits near 0.38 while top-5 reaches 0.78.

### Reporting consequence

Top-1 should not be called "accuracy" or used as the headline. **MRR and top-5
are the defensible targets**, and top-5 is already 0.78. Chasing top-1 on this
dataset is partly chasing irreducible ambiguity.

---

## 4. Naive Bayes: tested, not competitive

A high-bias model is a reasonable instinct for 8,329 rows, so it was measured
rather than argued about (fold 0 of the development CV):

| Model | seq top-1 | seq top-5 | seq MRR | seq macro recall |
| :--- | ---: | ---: | ---: | ---: |
| BernoulliNB (a=1.0) | 0.2851 | 0.6674 | 0.4590 | 0.1498 |
| BernoulliNB (a=0.1) | 0.2842 | 0.6785 | 0.4606 | 0.2887 |
| ComplementNB (a=0.5) | 0.2526 | 0.6499 | 0.4219 | 0.2055 |
| MultinomialNB (a=0.5) | 0.2860 | 0.6843 | 0.4654 | 0.2133 |
| B4 Markov-2 | 0.2867 | 0.6881 | 0.4553 | 0.2912 |
| **LightGBM grid9** | **0.3830** | **0.7774** | **0.5534** | **0.3059** |

(MEASURED — `results/diagnostics/naive_bayes_and_references.csv`)

Every variant lands at or below Markov-2 and about 0.09 MRR below LightGBM. Two
specific reasons, both verifiable from earlier project findings:

1. The conditional-independence assumption is violated **exactly where this data
   has structure**. Techniques co-occur in strongly correlated clusters, which is
   what the FP-Growth lift analysis (DEC-07) measured directly.
2. Bernoulli NB over `seen_*` indicators is a bag-of-techniques model and
   discards order, while the permutation test (DEC-05, p = 0.005) established
   that order carries signal.

Retained as a documented baseline. Not a path to accuracy.

---

## 5. The models make different mistakes, but a fixed blend cannot use that

Top-1 agreement against LightGBM on fold 0:

| Other model | Both correct | Only LightGBM | Only other | Neither | Union ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B4 Markov-2 | 0.2381 | 0.1611 | **0.0711** | 0.5297 | 0.4703 |
| BernoulliNB (a=0.1) | 0.2219 | 0.1773 | 0.0725 | 0.5284 | 0.4716 |
| ComplementNB (a=0.5) | 0.1908 | 0.2084 | 0.0675 | 0.5333 | 0.4667 |

(MEASURED — `results/diagnostics/model_complementarity.csv`)

Markov-2 is uniquely correct on **7.1%** of examples LightGBM misses. An oracle
combiner would reach 0.4703 top-1 against LightGBM's 0.3830, so complementary
signal demonstrably exists. But a fixed-weight blend does not recover it:

| LightGBM weight | seq top-1 | seq top-5 | seq MRR |
| ---: | ---: | ---: | ---: |
| 1.0 | 0.3830 | 0.7774 | 0.5534 |
| 0.9 | 0.3833 | 0.7748 | **0.5539** |
| 0.8 | 0.3800 | 0.7748 | 0.5517 |
| 0.6 | 0.3882 | 0.7604 | 0.5524 |
| 0.0 | 0.2867 | 0.6881 | 0.4567 |

(MEASURED — `results/diagnostics/blend_sweep.csv`)

Best blend is **+0.0005 MRR**, with the weight swept on the evaluation data
itself — so even that is optimistic. Conclusion: plain ensembling and voting are
not worth pursuing. Any gain from complementarity has to come from a model that
learns *when* to trust each source, conditional on context.

---

## 6. Feature probes: the prefix encoding is saturated

The 511 `seen_*` / `last-k` features **fully determine the prefix**. That single
observation predicts most of what follows, and it was tested four ways. All
numbers are fold 0 of the development CV, against LightGBM's 0.5534.

| Probe | seq MRR | Δ | seq macro recall |
| :--- | ---: | ---: | ---: |
| LightGBM, 511 features (reference) | 0.5534 | — | 0.3059 |
| + 478 mined-pattern / context features | 0.5475 | **−0.0059** | 0.3085 |
| Candidate ranker, 28 mining features | 0.5147 | −0.0387 | 0.3224 |
| Candidate ranker + candidate identity | 0.5153 | −0.0381 | 0.3210 |
| Candidate ranker + stacked LightGBM probability | 0.5106 | −0.0428 | 0.3183 |
| Candidate ranker + graph similarity | 0.5173 | −0.0361 | 0.3372 |
| Candidate ranker, `rank:pairwise` | 0.5101 | −0.0433 | 0.3156 |
| Candidate ranker, `binary:logistic` | 0.2486 | −0.3048 | 0.2266 |

### Why mined patterns add nothing

A PrefixSpan antecedent match is a **deterministic function of the prefix**, and
the model already receives a complete encoding of the prefix. Pattern indicators
were therefore never new information, only new dimensions. Measured effect:
−0.0059, i.e. noise in the unhelpful direction.

This does **not** retract the mining work. PrefixSpan's implementation and its
exact-match verification (DEC-08, DEC-09) and the negative-rule significance
testing (DEC-12) stand. The finding is narrower and worth stating precisely:
**mined patterns are not useful as predictive features for a model that already
sees the full prefix.**

### The candidate-ranker reformulation did not pay off

The ranker was built to escape the 94-class softprob bottleneck (47 of 94
classes hold fewer than 5 training examples; each boosting round fits 94
one-vs-rest functions). One ranking objective over 28 dense candidate-specific
features replaced it. See `src/models/candidate_ranker.py`.

It lost about 0.04 MRR in every variant. Feature importance explains why:
`m2_prob` carries gain 691 against 71 for the next feature, so the model
collapsed into a reweighted Markov-2. Adding candidate identity (+0.0006) and
graph similarity (+0.0037) barely moved it.

Two caveats kept on the record rather than buried:

- The ranker is **untuned** while LightGBM has a 12-configuration grid behind
  it, so this is not yet a like-for-like comparison.
- The stacking result is **anomalous and unexplained**: `lgb_prob` alone scores
  0.5534, yet the ranker assigns it gain 10.1 and gets *worse* by including it.
  Something about how `rank:ndcg` exploits features here is not understood.

Its one consistent advantage is rare-class coverage — macro recall 0.32-0.34
against LightGBM's 0.3059. Modest, and within noise on a single fold.

### Correction to an earlier claim

An earlier note in this investigation reported the ranker beating LightGBM on
macro recall by 0.09 (0.32 vs 0.23). That compared the ranker's fold-0 score
against LightGBM's **5-fold mean**. On the same fold LightGBM scores 0.3059, so
the real gap is about +0.03, within noise. The original comparison was invalid.

---

## 7. Graph-feature architecture: measured premises

A proposed architecture combined node embeddings, tactic abstraction, and graph
distance into a dense vector for the ranker. The transition graph was measured
first (fold-0 fit rows):

| Property | Value |
| :--- | ---: |
| Nodes | 107 |
| Directed edges | 909 |
| Density | 0.080 |
| Reachable ordered pairs | 84.3% |
| Mean hop distance | 2.83 |
| Reachable pairs within 1 / 2 / 3 hops | 9.5% / 41.8% / 75.4% |

Assessment, organised by the test from section 6 — **does this branch bring
information from outside the prefix encoding?**

- **Node embeddings (Node2Vec / random walk): mostly no.** An embedding
  compresses the same 107x107 transition structure the ranker already receives
  raw as `m1/m2/m3_prob` and `lift`. Tested in its cheap form (cosine similarity
  of transition profiles): +0.0037 MRR, +0.013 macro recall, with similarity
  feature gains of ~2 against `m2_prob`'s 691. Note also that 128-dimensional
  embeddings over 107 nodes is parameter *expansion*, not compression; 8-16
  dimensions would be the sensible range. Its one real contribution is a
  similarity prior that could help rare classes, which is unproven here.
- **Tactic abstraction: yes, and it is the only branch that does.** The ATT&CK
  tactic taxonomy is human knowledge that is *not derivable from the sightings
  data*, so by the section 6 argument it is the only proposed branch capable of
  adding information. It also attacks the measured bottleneck directly: roughly
  600 examples per tactic against about 89 per technique.
- **Graph distance to candidate: lowest value.** An earlier prediction that
  distance would have no dynamic range was wrong — distances do spread over 1-3
  hops. But hop count is an *unweighted* summary of edges whose weighted forms
  are already features, and it should be dominated the same way similarity was.
  A `-log(transition probability)` weighted shortest path ("cheapest attack
  path") would at least have dynamic range and would connect to the attack-graph
  literature.

### Implementation note: tactics are multi-label

Techniques belong to several tactics. T1078 Valid Accounts maps to Initial
Access, Persistence, Privilege Escalation and Defense Evasion. Treating tactics
as 14 mutually exclusive classes would force an arbitrary per-technique choice
and inject label noise, so the mapping must be a **multi-hot vector over 14
tactics**, never a single tactic id. See `src/core/tactic_map.py`.

### Mapping availability

`config.py` carried a comment claiming ATT&CK mapping tables, but no table
exists in the repository. The AIT-ADS Wazuh exports embed a usable
`rule.mitre` block and were extracted as a local source, but they cover only
**8 techniques** of the 94-107 vocabulary — far too sparse to use. The
authoritative ATT&CK mapping is therefore still required before the tactic
branch can be tested.

---

## 8. Summary of what is and is not the constraint

**Not the constraint:**

- Model family. Naive Bayes, Markov, PrefixSpan, LightGBM, XGBoost and a
  learning-to-rank reformulation were all measured.
- Hyperparameters. Three searches landed within 0.0047.
- Mined patterns as features. Deterministic functions of an already-complete
  prefix encoding.
- Ensembling by fixed weights. +0.0005.
- An irreducible Bayes limit. Trees already exceed the lookup ceiling by 0.106.

**Plausibly the constraint:**

- **Absence of external knowledge.** Every feature tried so far is a re-encoding
  of the same dataset. The tactic taxonomy is the first candidate for genuinely
  new information.
- **Intrinsic multi-modality of the target.** 6.02 distinct successors per
  reused last-2 context caps top-1 regardless of method.
- **Reporting order is not execution order.** The sightings data aggregates
  co-occurring techniques from incident reports. DEC-05 established the ordering
  is not random (p = 0.005), but "not random" is not "predictable". Testing this
  properly needs the timestamped AIT-ADS traces, which no feature engineering on
  this dataset can substitute for.

A null result here is a legitimate finding, not a failure, provided it is
reported with the measurements above and with the cross-validated protocol of
DEC-18 rather than from a single noisy holdout.

---

## 9. Tactic abstraction: tested under the cross-validated protocol

Reproduce with:

```bash
python src/core/build_tactic_reference.py     # reference table, external data
python src/core/tactic_map.py                 # coverage report
python src/eval/tactic_experiments.py --full  # 5 folds + paired CIs
```

### Mapping

`data/reference/attack_technique_tactics.csv` holds 241 technique/tactic pairs
over 212 techniques across all 14 Enterprise tactics, read from the MITRE
ATT&CK tactic pages (see `src/core/build_tactic_reference.py` for provenance).
Coverage of our 109-technique vocabulary is **100 techniques (91.7%)**; the 9
unmapped ids (T1207, T1222, T1366-T1369, T1406, T1553, T1562) receive an
explicit `tactic_unknown` flag so "no tactic applies" stays distinguishable from
"not present in our source". 17.0% of covered techniques are multi-tactic,
which is why the encoding is multi-hot (DEC-20).

### Designs tested

- **A, prefix tactics**: 32 prefix-level columns added to the softprob model —
  tactic multi-hot of the whole prefix, of the last technique, the count of
  distinct tactics reached, and the furthest kill-chain index reached.
- **B, hierarchical**: 14+1 binary tactic heads predict the next technique's
  tactic (each head has hundreds of positives, against ~89 per technique), then
  `score(c) = P_technique(c) * tactic_score(c)^lambda`. `lambda = 0` recovers the
  baseline exactly and is retained as an implementation check.

### Result (5 folds, means over folds)

| Model | seq top-1 | seq top-5 | seq MRR | seq macro recall |
| :--- | ---: | ---: | ---: | ---: |
| baseline, 511 features | 0.3694 | 0.7559 | 0.5386 | 0.2300 |
| A, + 32 tactic features | 0.3739 | 0.7581 | **0.5418** | **0.2429** |
| B, lambda = 0 (check) | 0.3694 | 0.7559 | 0.5386 | 0.2300 |
| B, lambda = 0.25 | 0.3683 | 0.7554 | 0.5379 | 0.2276 |
| B, lambda = 0.5 | 0.3671 | 0.7499 | 0.5353 | 0.2252 |
| B, lambda = 1.0 | 0.3643 | 0.7335 | 0.5285 | 0.2216 |
| B, lambda = 2.0 | 0.3614 | 0.7008 | 0.5140 | 0.2181 |

Paired against the baseline over 2,178 sequences:

| Design | mean diff | 95% CI | nominally significant |
| :--- | ---: | :--- | :--- |
| A, prefix tactics | +0.0032 | [+0.0003, +0.0063] | yes |
| B, lambda = 0 | 0.0000 | [0.0000, 0.0000] | no (exact check passes) |
| B, lambda = 0.25 | −0.0007 | [−0.0023, +0.0008] | no |
| B, lambda = 0.5 | −0.0032 | [−0.0054, −0.0011] | yes (worse) |
| B, lambda = 1.0 | −0.0101 | [−0.0130, −0.0074] | yes (worse) |
| B, lambda = 2.0 | −0.0246 | [−0.0283, −0.0212] | yes (worse) |

(MEASURED — `results/tactics/tactic_cv_summary.csv`,
`results/tactics/tactic_paired_vs_baseline.csv`)

### Design A's significance does not survive scrutiny

Design A produced the only nominally significant improvement in this entire
investigation, so it was checked two further ways before being believed.

**Multiple comparisons.** Design A was one of five real comparisons against the
same baseline (`lambda = 0` is a tautology). Its statistic is t = 2.11; the
Bonferroni threshold for five comparisons at alpha = 0.05 is |t| > 2.58. It does
**not** survive correction.

**Replication across fold partitions.** Re-running the same comparison under
three CV seeds, which changes both the partition and the estimator seeds:

| CV seed | baseline | + tactics | difference | nominally significant |
| ---: | ---: | ---: | ---: | :--- |
| 42 | 0.5386 | 0.5418 | **+0.0032** | yes |
| 7 | 0.5335 | 0.5343 | +0.0008 | no |
| 2024 | 0.5361 | 0.5360 | −0.0001 | no |

Mean across seeds **+0.0013**, the signs do not agree, and two of three seeds
are not significant. The seed-42 result was fold-partition luck.

### Conclusion

**Tactic abstraction does not improve next-technique ranking on this dataset.**
The effect on MRR is +0.0013 across seeds, indistinguishable from zero, and
putting the tactic signal on the output side (design B) actively hurts, more so
the more it is weighted.

This is the strongest negative result in the investigation because it was the
best-motivated hypothesis: tactics are the one tested feature source carrying
information not derivable from the sightings data, and they target the measured
class-sparsity bottleneck directly. The prediction was wrong.

One unresolved thread: design A's macro recall gain (0.2300 to 0.2429, +5.6%
relative) was not subjected to the same stability check, because macro recall is
not a per-sequence quantity and does not fit the paired bootstrap. If rare-class
coverage is the objective rather than MRR, that is worth testing properly.

### Note on method

The seed-42 result would have been reported as a success under the project's
earlier practice of selecting and reporting on one split. It was caught by the
same discipline that overturned the B6 negative-rule claim (DEC-13) — applied
here to a result this investigation itself produced and wanted to be true.

---

## 10. The full A+B+C+D pipeline

A four-stage pipeline was specified and built in full:

| Stage | What it contributes | Implementation |
| :--- | :--- | :--- |
| **A** graph | Node2Vec embedding closeness, hop distance, and `-log P` shortest-path cost from prefix to candidate | `src/models/graph_features.py` |
| **B** tactics | Candidate tactic multi-hot, tactic-level transition probability and count, overlap with the prefix's tactics, kill-chain progression delta | `src/models/pipeline_ranker.py` |
| **C** interactions | Recency-decayed transition evidence (lambda 0.5 and 0.8), nearest contributing prefix position, transition breadth, recency-weighted lift | `src/models/pipeline_ranker.py` |
| **D** masking | Structurally impossible candidates removed before ranking | `src/core/config.py`, enforced in `src/eval/cv_harness.py` |
| head | XGBRanker, `rank:ndcg`, over the assembled (prefix, candidate) rows | `src/models/pipeline_ranker.py` |

Reproduce with:

```bash
python -m pip install gensim          # Node2Vec skip-gram, Part A only
python src/models/pipeline_ranker.py          # fold-0 ablation
python src/eval/pipeline_final.py             # 5-fold paired verdict
```

### Part A is real Node2Vec

Second-order biased random walks (`p = 1.0`, `q = 0.5`, 20 walks per node of
length 20) fed to gensim skip-gram word2vec, 16 dimensions. The dimension is a
deliberate choice: with about 107 nodes, the 128 dimensions used in the Node2Vec
literature would be parameter *expansion* rather than compression. `q < 1`
biases walks outward, which suits attack progression. The candidate's raw
embedding is also passed through as 16 features, giving the model continuous
candidate identity rather than the integer id tried earlier.

Hop distance and path cost come from Dijkstra over the fold's own graph, the
latter with `-log(transition probability)` edge weights so it measures the cost
of the most likely route rather than a bare hop count.

### Fold-0 ablation

| Variant | seq top-1 | seq top-3 | seq top-5 | seq MRR | macro recall | macro F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base, mining features only | 0.3549 | 0.5905 | 0.7159 | **0.5148** | 0.3238 | 0.2960 |
| A, + graph | 0.3507 | 0.5875 | 0.7117 | 0.5117 | 0.3261 | 0.3109 |
| A + B, + tactics | 0.3512 | 0.5915 | 0.7139 | 0.5115 | 0.3235 | 0.3165 |
| A + C, + interactions | 0.3565 | 0.5923 | 0.7043 | 0.5139 | 0.3322 | 0.3215 |
| A + B + C, full | 0.3565 | 0.5903 | 0.7125 | 0.5137 | **0.3371** | **0.3222** |
| *LightGBM softprob (reference)* | *0.3830* | *0.6360* | *0.7774* | *0.5534* | *0.3059* | — |

(MEASURED — `results/pipeline/pipeline_fold0_summary.csv`)

Two things are visible immediately, and they point in opposite directions:

- **On ranking metrics nothing moves.** Every variant sits between 0.5115 and
  0.5148 MRR, a spread of 0.0033, which is noise. Adding the graph group
  *lowered* MRR from 0.5148 to 0.5117. All variants remain about 0.04 below the
  plain softprob model.
- **On class metrics the stages do accumulate.** Macro F1 rises monotonically
  with each group added, 0.2960 to 0.3222, and macro recall reaches 0.3371
  against the softprob model's 0.3059.

So the pipeline is not a ranking improvement, but it is a different operating
point: it spreads prediction across more classes at the cost of peak accuracy on
the common ones.

### 5-fold verdict

| Model | seq top-1 | seq top-3 | seq top-5 | seq MRR | macro recall | macro F1 | MRR fold sd |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LightGBM softprob** | **0.3694** | **0.6360** | **0.7559** | **0.5386** | 0.2299 | 0.2269 | 0.0132 |
| Pipeline, mining only | 0.3458 | 0.5740 | 0.6936 | 0.5014 | 0.2369 | 0.2321 | 0.0081 |
| Pipeline, A+B+C | 0.3466 | 0.5723 | 0.6899 | 0.5026 | **0.2420** | **0.2367** | 0.0105 |

Paired per-sequence MRR, bootstrap over all 2,178 development sequences:

| Comparison | mean diff | 95% CI | significant |
| :--- | ---: | :--- | :--- |
| Pipeline A+B+C − Pipeline base | +0.0012 | [−0.0015, +0.0040] | **no** |
| Pipeline A+B+C − LightGBM softprob | **−0.0360** | [−0.0430, −0.0291] | yes (worse) |
| Pipeline base − LightGBM softprob | −0.0372 | [−0.0440, −0.0302] | yes (worse) |

(MEASURED — `results/pipeline/pipeline_final_summary.csv`,
`results/pipeline/pipeline_final_paired.csv`)

**Two findings, both negative.**

1. **A + B + C add nothing.** Together, Node2Vec embeddings, tactic transitions
   and recency interactions move the ranker by +0.0012 MRR with a CI straddling
   zero. Having each group measured separately in the fold-0 ablation and jointly
   here, the conclusion is the same: these three feature families carry no
   additional ranking signal for this task.
2. **The ranker formulation itself costs about 0.036 MRR.** Both ranker variants
   are significantly worse than the plain 94-class softprob model, with CIs
   decisively excluding zero. The reformulation was my proposal, and it is
   wrong for this problem.

### The two models are genuinely different, not uniformly worse

Per-sequence comparison against the softprob model:

| | Share of sequences | Mean MRR difference |
| :--- | ---: | ---: |
| Pipeline strictly better | 38.9% | +0.1127 |
| Pipeline strictly worse | 56.8% | −0.1405 |
| Tied | 4.2% | — |

The pipeline wins outright on nearly two sequences in five, and by a large
margin when it does. It simply loses more often and by more. This is the same
complementarity measured in section 5 — real, substantial, and still not
convertible into a net gain. The pipeline is also more stable across folds
(MRR sd 0.0081-0.0105 against 0.0132), which matters if variance is a concern.

### Correcting the predicted impacts

The pipeline was specified with expected metric impacts. Measured against them:

| Stage | Predicted impact | Measured |
| :--- | :--- | :--- |
| A, graph | Keeps predictions in the right neighbourhood, avoiding severe misses | Lowered fold-0 MRR from 0.5148 to 0.5117; jointly with B and C, +0.0012 and not significant |
| B, tactics | "Surges macro recall from ~0.23 to >0.34" | Macro recall 0.2299 to 0.2420 over 5 folds. The 0.33-0.34 figures were fold-0 only, and fold 0 is a favourable fold |
| C, interactions | "Breaks the flat 511-feature plateau" | Plateau intact; the 511-feature softprob model remains the best model measured |
| D, masking | "Immediate zero-cost boosts to top-1, top-3, MRR" | **Confirmed** — +0.0271 MRR for Markov-2, +0.0277 for PrefixSpan (DEC-17). The only one of the four that delivered |

The macro-recall claim is worth isolating because it was the strongest numeric
prediction. The gain exists but is +0.012, not +0.11. The earlier appearance of a
large gain came from comparing a single favourable fold against a 5-fold mean,
the same error corrected in section 6.
