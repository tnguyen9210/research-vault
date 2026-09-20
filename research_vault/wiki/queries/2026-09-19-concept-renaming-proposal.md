---
date: 2026-09-19
question: "Should concept pages be renamed broadest-to-narrowest, and which ones have a clear enough parent to justify it?"
tags: [vault-maintenance, naming]
status: temporary
---

# Concept renaming proposal — broader-to-narrower names

> **EXECUTED 2026-09-19.** All 11 changes below are applied, with one
> amendment: `pfql-algorithm-1` was not renamed but **merged into**
> `fitted-q-iteration-pessimistic` and removed, so `concepts/` now
> holds 38 pages. Kept as the record of the rule and the reasoning;
> delete it once that is no longer wanted. Deliberately not in
> `index.md`, so a lint will report it as an orphan — expected.

## The rule used

A name is only changed when **the parent is itself an existing concept
page** and the child is a genuine specialization of it. That is the
rule both of Tuan's examples satisfy, and it is what keeps the list
short instead of forcing an arbitrary parent onto every page.

**Result: 11 renamed, 28 unchanged.**

## All 39 concepts

| current | proposed | parent | note |
|---|---|---|---|
| `best-arm-identification` | — | — | family root — parent would be *bandits*, no page |
| `budget-limited-mab` | — | — | family root — parent would be *multi-armed bandits*, no page |
| `cabai` | **`best-arm-identification-cost-aware`** | `best-arm-identification` | rename |
| `constrained-bai` | **`best-arm-identification-constrained`** | `best-arm-identification` | rename |
| `contextual-bandits` | — | — | family root |
| `coverage-coefficient` | — | — | cross-cutting: offline RL, offline CB, alignment |
| `decision-estimation-coefficient` | — | — | general complexity measure; already sorts beside its offline twin |
| `decision-offline-estimation-coefficient` | — | — | general complexity measure; already sorts beside its online twin |
| `deep-q-network` | — | — | parent would be *Q-learning* / *deep RL*, no page |
| `differentiable-function-approximation` | — | — | parent would be *function approximation*, no page |
| `eluder-dimension` | — | — | cross-cutting: bandits, RL, function approximation |
| `epsilon-first` | **`budget-limited-mab-epsilon-first`** | `budget-limited-mab` | rename |
| `epsilon-sec` | — | — | a coverage measure bounding DOEC; no single parent |
| `expectile-regression` | — | — | parent would be *regression*, no page |
| `exploitative-f-design` | — | — | parent would be *experimental design* or `oracle-efficiency`; neither is clean |
| `extrapolation-error` | — | — | parent is the **topic** [[offline-reinforcement-learning]], not a concept page |
| `fitted-q-iteration` | — | — | family root — parent would be *approximate DP*, no page |
| `fqi-finite-sample-analysis` | **`fitted-q-iteration-finite-sample-analysis`** | `fitted-q-iteration` | rename |
| `implicit-q-learning` | — | — | parent is the **topic** [[offline-reinforcement-learning]], not a concept page |
| `importance-weighting` | — | — | cross-cutting: statistics, CB, RL |
| `instance-dependent-bounds` | — | — | cross-cutting: spans bandits *and* RL by construction |
| `kube` | **`budget-limited-mab-kube`** | `budget-limited-mab` | rename |
| `linear-softmax-policy` | — | — | parent would be *RLHF* / policy parameterization, no page |
| `monte-carlo-tree-search` | — | — | family root |
| `offline-contextual-bandits` | **`contextual-bandits-offline`** | `contextual-bandits` | rename |
| `offline-regression-oracle` | — | — | parent `oracle-efficiency` is clear, but the rename repeats *oracle* and reads worse |
| `oracle-efficiency` | — | — | parent is `contextual-bandits` (its own title says so) but it has 5 children; prefixing them needs a 4th level. See the open call below |
| `overestimation-bias` | — | — | parent would be *Q-learning*; not a child of `deep-q-network` |
| `pessimism-principle` | — | — | cross-cutting: the spine of both offline RL and offline CB |
| `pessimistic-fitted-q-learning` | **`fitted-q-iteration-pessimistic`** | `fitted-q-iteration` | rename |
| `pfql-algorithm-1` | **`fitted-q-iteration-pessimistic-algorithm-1`** | `pessimistic-fitted-q-learning` | rename |
| `power-mean-mcts` | **`monte-carlo-tree-search-power-mean`** | `monte-carlo-tree-search` | rename |
| `realizability` | — | — | cross-cutting: an assumption used everywhere |
| `slg-search` | **`test-time-scaling-slg-search`** | `test-time-scaling` | rename |
| `softmax-bellman-operator` | — | — | parent would be *Bellman operator*, no page |
| `spanner-sampling` | — | — | parent would be *RLHF alignment*, no page |
| `test-time-scaling` | — | — | family root |
| `upper-confidence-bound` | — | — | cross-cutting: bandits, MCTS, RL |
| `value-based-offline-bandits` | **`contextual-bandits-offline-value-based`** | `offline-contextual-bandits` | rename |

## What the renames buy

Every family becomes alphabetically contiguous:

| family | before | after |
|---|---|---|
| offline CB | 2 pages over 15 slots | 2 / 2 |
| FQI | 4 pages over 15 slots | 4 / 4 |
| BAI | 3 pages over 4 slots | 3 / 3 |
| budget-limited | 3 pages over 21 slots | 3 / 3 |
| MCTS | 2 pages over 9 slots | 2 / 2 |
| test-time scaling | 2 pages over 4 slots | 2 / 2 |

Longest name grows from 39 to 42 characters.

Two side benefits: the FQI renames retire the abbreviation
inconsistency (`fqi-` vs `fitted-q-iteration`, `pfql-` vs
`pessimistic-fitted-q-learning`), and `cabai`, `kube` and
`pfql-algorithm-1` stop being cryptic.

## Open calls

1. **`oracle-efficiency` and its five children.** The parent is
   explicit in its own title, *Oracle Efficiency (Contextual Bandits)*,
   so `contextual-bandits-oracle-efficiency` is justified. But
   prefixing its children consistently needs a fourth level —
   `contextual-bandits-oracle-efficiency-offline-regression-oracle` is
   62 characters. Renaming the parent alone would separate it from its
   own children, which is worse than today. Recommendation: cap depth
   at three levels and leave this cluster.
2. **`epsilon-first` is the weakest of the 11.** It is explore-then-
   commit in general, and budget-specific only as this vault frames it.
   `kube`, by contrast, opens with "An online algorithm for the
   `[[budget-limited-mab]]`". Drop it if the list should be strictly
   safe, leaving 10.

## Cost, if this goes ahead

- 11 filenames, every inbound wiki link, the `index.md` entries, and
  each page's `title:` and `# heading`.
- `log.md` is append-only, so it will keep referring to the old names
  permanently. One log entry recording the full mapping keeps the
  history readable.
- Old names should be added as `aliases:` in frontmatter so existing
  links and muscle memory keep resolving.
