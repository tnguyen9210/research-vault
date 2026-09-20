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
>
> **Wave 2 also executed 2026-09-19:** the `bai`, `fqi` and `mcts`
> family prefixes are abbreviated; `cb` was rejected. See the last
> section for the rule and the reasoning.

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

## Wave 2 (executed) — abbreviating family prefixes

Wave 1 made names structural but long: `fitted-q-iteration-finite-
sample-analysis` is 41 characters. The question is whether to
abbreviate family prefixes to the standard acronyms of the field.

### The rule proposed

Abbreviate only when all four hold:

1. the acronym is **standard in the literature**, not a local coinage;
2. it is **unambiguous inside this vault**, not just in general;
3. the **parent is abbreviated too**, so the family stays consistent —
   mixing `fitted-q-iteration` with `fqi-...` is exactly the mismatch
   wave 1 removed;
4. the **full name goes into `aliases:`**, so search and existing
   links keep working.

### Applied (8 pages)

| current | proposed | chars |
|---|---|---|
| `best-arm-identification` | `bai` | 23 → 3 |
| `best-arm-identification-constrained` | `bai-constrained` | 35 → 15 |
| `best-arm-identification-cost-aware` | `bai-cost-aware` | 34 → 14 |
| `fitted-q-iteration` | `fqi` | 18 → 3 |
| `fitted-q-iteration-finite-sample-analysis` | `fqi-finite-sample-analysis` | 41 → 26 |
| `fitted-q-iteration-pessimistic` | `fqi-pessimistic` | 30 → 15 |
| `monte-carlo-tree-search` | `mcts` | 23 → 4 |
| `monte-carlo-tree-search-power-mean` | `mcts-power-mean` | 34 → 15 |

Average name length across these eight drops from about 30 characters
to about 12. All three acronyms are universal in this literature and
collide with nothing else in the vault.

### Rejected

**`cb` for contextual bandits.** Measured in the vault: **LCB appears
47 times and UCB 38, against CB's 8.** In a corpus where those three
letters usually end a confidence-bound acronym, `cb-offline-value-
based` invites a misreading, and [[upper-confidence-bound]] is itself
a page. The three affected names are also the shortest of the long
ones (18, 26, 38), so the saving is smallest where the risk is
highest. Keep `contextual-bandits-`.

**`mab`.** `budget-limited-mab` already carries the acronym;
abbreviating the rest of it gives `blmab-kube`, which is worse than
what wave 1 replaced.

**`tts`, `rl`, `dfa`.** Not standard enough to be read without effort.

### The cost worth weighing

**A filename is also its link text.** Obsidian shows a bare wiki link
as the page name, so a link to `bai` reads "bai" in the sentence
unless it is piped with explicit display text every time. Short names
improve the file list and the quick switcher; they make running prose
terser and, in places, worse. The
eight pages above are the ones where the acronym is so standard that
"BAI", "FQI" and "MCTS" read naturally in a sentence anyway — which
is the real reason to stop at these three and not generalise.

A second wave also leaves a second set of dead names in the
append-only log. Doing it now is cheaper than after the four queued
ingests, when more pages will point at these names.
