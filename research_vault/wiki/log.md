# Operation Log

Append-only. Each entry: `## [YYYY-MM-DD] <operation> | <title or summary>`  
Parse with: `grep "^## \[" wiki/log.md`

---

## [2026-08-19] lint | Vault-wide link repair: 0 broken links, 0 orphans

- Scanned all 84 wiki pages for unresolved `[[links]]`, stubs, and pages with no inbound links. Scanner now strips code spans first -- `[[slug]]` in index.md and `[[links]]` in log.md are documentation text inside backticks, not real links, and were previously false positives.
- **Broken links fixed (8 targets, 6 new pages):**
  - `authors/Carin-Lawrence.md` -- third author of [[Song2019Revisiting]], omitted at ingest
  - `concepts/pessimism-principle.md` -- referenced from `importance-weighting`; now the hub for the offline LCB idea across [[Ryu2025Improved]], [[Yin2023Offline]], PEVI, CQL
  - `concepts/upper-confidence-bound.md` -- referenced from `kube`, `budget-limited-mab`, [[TranThanh2012Knapsack]]
  - `concepts/eluder-dimension.md` -- referenced from `doec`
  - `concepts/epsilon-sec.md` -- referenced from `doec`, [[Qin2026Taming]], `topics/offline-oracle-efficient-bandits`; written from the source PDF (Definition 3, Theorem 3, Propositions 1 and 3)
  - `papers/TranThanh2010Epsilon.md` -- STUB, clearly banner-marked: no PDF in `raw/papers/`, content sourced only from what [[TranThanh2012Knapsack]] states about its predecessor
- **Orphans fixed (6 pages, 0 new content):** [[Foster2025Foundation]] now links [[spanner-sampling]], its authors [[Mhammedi-Zakaria]] / [[Rohatgi-Dhruv]], and the saved query [[2026-06-16-foster2025-sections-1-4]]; [[Ryu2025Improved]] now links [[Ryu-J-Jon]] and [[Jun-Kwang-Sung]]. All were reachable only from `index.md`.
- **Garbled index entries fixed:** the `test-time-scaling` concept line had `offline-regression-oracle`'s description appended to it, and the `test-time-scaling` topic line had a duplicated `FALCON -> OE2D` tail. Both truncated to their correct text; the orphaned tail restored to the `offline-regression-oracle` entry where it belonged.
- Updated `index.md`: +6 entries (1 paper stub, 4 concepts, 1 author)
- **Result: 0 broken links, 0 orphans across 84 pages.**
- Outstanding: [[TranThanh2010Epsilon]] needs a real ingest (AAAI 2010) to replace the stub.

## [2026-08-19] ingest | Offline RL with Differentiable Function Approximation is Provably Efficient (ICLR 2023)

- Renamed raw `2210.00750v2.pdf` -> `Yin2023Offline.pdf`
- Venue verified as ICLR 2023 via Yu-Xiang Wang's NSF project page (arXiv abs page carries no venue comment); slug uses venue year per the `Song2019Revisiting` precedent
- Created `wiki/papers/Yin2023Offline.md`
- Created concept pages: `differentiable-function-approximation`, `pessimistic-fitted-q-learning`, `instance-dependent-bounds` (the last is a HUB page, not novel to this paper -- created by request to serve the vault's dominant guarantee type)
- Created author pages: `Yin-Ming`, `Wang-Mengdi`, `Wang-Yu-Xiang`
- Updated `topics/offline-reinforcement-learning.md`: theory/practice axis, PFQL entry, 5 new open problems
- Updated concepts: `realizability` (Bellman completeness), `coverage-coefficient` (concentrability + uniform coverage), `importance-weighting`, `best-arm-identification`
- Updated papers: `Kostrikov2022Offline` (theory counterpart), `Ryu2025Improved` (shared pessimism principle)
- Updated `overview.md`: theory-of-offline-RL paragraph, two new threads (instance-dependent guarantees cross-cutting; theory/practice gap), sources 12 -> 13
- Updated `index.md`: +7 entries (1 paper, 3 concepts, 3 authors)
- Critiques recorded on the paper page: uniform coverage (Asmp 2.3) requires parameter identifiability and so excludes overparameterized nets, contradicting the "$\theta$ = network weights" motivation; rate carries $d$ = parameter count, not $\sqrt{d}$; "provably efficient" is statistical only -- the fitted-Q step is a nonconvex argmin analyzed as exactly solved
- Noted for lint: `[[pessimism-principle]]` referenced from `concepts/importance-weighting.md` does not exist (stub to create)

## [2026-08-19] ingest | Offline Reinforcement Learning with Implicit Q-Learning (ICLR 2022)

- Renamed raw `2110.06169v1.pdf` -> `Kostrikov2022Offline.pdf`
- Created `wiki/papers/Kostrikov2022Offline.md` (framing: theory gap + smooth-aggregator thread, per discussion)
- Created concept pages: `expectile-regression`, `implicit-q-learning` (AWR and in-sample-learning considered and deferred)
- Created author pages: `Kostrikov-Ilya`, `Nair-Ashvin`, `Levine-Sergey`
- Created topic page: `topics/offline-reinforcement-learning.md` — first offline RL topic in the vault
- Updated concepts: `overestimation-bias`, `softmax-bellman-operator`, `deep-q-network`, `power-mean-mcts`, `coverage-coefficient`
- Updated papers: `Song2019Revisiting`, `Dam2024Power` (thematic parallels); `topics/monte-carlo-tree-search` (related topics)
- Updated `overview.md`: offline RL paragraph, new cross-cutting thread "smooth aggregators in place of max" (Dam2024Power / Song2019Revisiting / Kostrikov2022Offline), sources 11 -> 12
- Updated `index.md`: +7 entries (1 paper, 2 concepts, 3 authors, 1 topic)
- Headline open question recorded: no finite-$\tau$ bound on $\max_{a:\pi_\beta(a|s)>0} Q^*(s,a) - V_\tau(s)$; conjecture that the gap is governed by a density-weighted coverage quantity ([[coverage-coefficient]]) rather than a support indicator
- Noted for lint: pre-existing broken link `[[Carin-Lawrence]]` in `Song2019Revisiting.md`; garbled tail text in two `index.md` entries (`test-time-scaling` concept and topic lines)

## [2026-06-16] query | Foster2025Foundation Sections 1–4 summary

- Saved `wiki/queries/2026-06-16-foster2025-sections-1-4.md`
- Covers: sampling oracle framework, coverage coefficient + lower bound (Thm 2.1), SpannerSampling algorithm + guarantee (Thm 3.1), ETH-hardness of proper exploration (Thm 4.1)
- Updated `index.md` (1 new query entry)

## [2026-06-16] ingest | Is a Good Foundation Necessary for Efficient RL? (arXiv 2025)

- Renamed raw `2503.07453v2.pdf` → `Foster2025Foundation.pdf`
- Created `wiki/papers/Foster2025Foundation.md`
- Created concept pages: `coverage-coefficient`, `linear-softmax-policy`, `spanner-sampling`
- Created author pages: `Foster-Dylan-J`, `Mhammedi-Zakaria`, `Rohatgi-Dhruv`
- Updated `topics/test-time-scaling.md`: added Foster2025Foundation entry + 2 open problems
- Updated `topics/monte-carlo-tree-search.md`: added MTSS entry + open problem on token-level coverage
- Updated `overview.md`: new "Computational theory of LM alignment" thread; active thread entry; source count 10 → 11
- Updated `index.md`: +7 entries (1 paper, 3 concepts, 3 authors)
- New thread: computational-statistical tradeoffs for LM alignment via sampling oracle. Stubs referenced: XPO (Xie et al. 2024), OnlineDPO (Guo et al. 2024)

## [2026-06-09] update | Song2019Revisiting — add Assumptions & Theorems section

- Added "## Assumptions & Theorems" to `wiki/papers/Song2019Revisiting.md`: vocabulary, Lemma 2, Theorem 3, Theorem 4 (all three parts), gradient noise argument, and a how-the-pieces-fit diagram — all in plain language with the honest-gap note.
- No new pages or links; no other files touched.

## [2026-06-09] ingest | Revisiting the Softmax Bellman Operator (ICML 2019)

- Renamed raw `1812.00456v2.pdf` → `Song2019Revisiting.pdf`
- Created `wiki/papers/Song2019Revisiting.md`
- Created concept pages: softmax-bellman-operator, overestimation-bias, deep-q-network
- Created author pages: Song-Zhao, Parr-Ronald-E (Carin omitted — less central to RL content)
- Cross-linked [[Dam2024Power]] (thematic parallel: both replace max backup with smooth aggregator)
- Updated index.md (+6 entries), overview.md (new deep RL thread), log.md
- New thread: value-based deep RL / Bellman operator variants. Stubs referenced: mellowmax (Asadi & Littman 2017), distributional RL (C51/QR-DQN), Rainbow (Hessel et al. 2018)

## [2026-06-09] update | KUBE vs. fractional KUBE comparison

- Added "## KUBE vs. Fractional KUBE" section to `concepts/kube.md`: same objective, per-step knapsack-solver fork (greedy multiset + sampling vs. fractional relaxation = UCB-on-densities), side-by-side table, and the bound-vs-practice inversion (fractional has tighter bound but KUBE wins ≤40% under diverse costs).
- Trimmed the now-redundant Fractional-KUBE bullet in "Variants" to a pointer to the new section.
- No new pages or links; no other files touched.

## [2026-06-08] ingest | Knapsack-based Optimal Policies for Budget-Limited MAB (AAAI 2012)

- Renamed raw `1204.1909v1.pdf` → `TranThanh2012Knapsack.pdf`
- Created `wiki/papers/TranThanh2012Knapsack.md`
- Created concept pages: budget-limited-mab, kube
- Created author pages: Tran-Thanh-Long (first), Rogers-Alex, Jennings-Nicholas-R (senior)
- Created topic page: budget-limited-bandits (cost/budget-constrained bandits; reward vs. pure-exploration)
- Cross-linked [[Kanarios2024Cost]] (Contrast: shared-budget regret vs. fixed-confidence cost minimization)
- Updated index.md (7 new entries)
- New thread: first budget-limited/cost-bandit family in the vault. Stubs referenced but not created: [[TranThanh2010Epsilon]] (the ε-first predecessor), [[upper-confidence-bound]]

## [2026-05-31] update | Removed stale empty venues/ directory

- Deleted empty `wiki/venues/` (contained no files); the venues category was already removed from the schema on 2026-05-27 during the Qin2026Taming ingest.
- No wiki pages, `[[links]]`, or `CLAUDE.md` references affected.
- `wiki/` now contains: papers, concepts, authors, topics, queries.

## [2026-05-28] ingest | Stochastically Constrained BAI with Thompson Sampling (arXiv 2025)

- Renamed raw file `2501.03877v1.pdf` → `raw/papers/Yang2025Stochastically.pdf`
- Created `wiki/papers/Yang2025Stochastically.md`
- Created author pages: Yang-Le, Wang-Yi
- Updated constrained-bai.md: added BFAI-TS entry + fixed-confidence vs. fixed-budget comparison table
- Updated topics/best-arm-identification.md: added Yang2025Stochastically entry
- Updated overview.md (8 papers), index.md (3 new entries), log.md

## [2026-05-28] ingest | Constrained Best Arm Identification (NeurIPS 2025)

- Renamed raw file `28162_Constrained_Best_Arm_Ide.pdf` → `raw/papers/Lardy2025Constrained.pdf`
- Created `wiki/papers/Lardy2025Constrained.md`
- Created concept page: constrained-bai
- Created author pages: Lardy-Tyron, Koolen-Wouter-M
- Updated best-arm-identification.md (concept + topic): added CBAI entry and CABAI vs. CBAI comparison table
- Updated cabai.md: added distinction from [[constrained-bai]]
- Updated overview.md (7 papers, CABAI/CBAI paragraph), index.md (5 new entries), log.md

## [2026-05-28] ingest | Improved Offline Contextual Bandits with Second-Order Bounds (COLT 2025)

- Copied raw file `2502.10826v2.pdf` → `raw/papers/Ryu2025Improved.pdf`; removed misplaced copy from `wiki/papers/`
- Created `wiki/papers/Ryu2025Improved.md`
- Created concept page: importance-weighting
- Created author pages: Ryu-J-Jon, Jun-Kwang-Sung
- Updated contextual-bandits.md (Key Papers + Variants), topics/offline-oracle-efficient-bandits.md (new Adjacent section)
- Updated overview.md (6 papers, new offline-policy-optimization thread), index.md (5 new entries), log.md

## [2026-05-28] ingest | Predicting and improving test-time scaling laws (arXiv 2026)

- Renamed raw file `2602.01485v1.pdf` → `Li2026Predicting.pdf`
- Created `wiki/papers/Li2026Predicting.md`
- Created concept pages: test-time-scaling, slg-search
- Created author pages: Li-Muheng, Qian-Jian, Mou-Wenlong
- Created topic page: test-time-scaling
- Updated overview.md (5 papers, new LLM test-time area), index.md (9 new entries)

## [2026-05-27] ingest | Power Mean Estimation in Stochastic MCTS (arXiv 2024)

- Renamed raw file `2406.02235v1.pdf` → `Dam2024Power.pdf`
- Created `wiki/papers/Dam2024Power.md`
- Created concept pages: monte-carlo-tree-search, power-mean-mcts
- Created author pages: Dam-Tuan, Maillard-Odalric-Ambrym, Kaufmann-Emilie
- Created topic page: monte-carlo-tree-search
- Updated overview.md (4 papers, new MCTS area), index.md (8 new entries), log.md

## [2026-05-27] ingest | Bypassing the Monster (Mathematics of Operations Research 2022)

- Renamed raw file `2003.12699v5.pdf` → `SimchiLevi2022Bypassing.pdf`
- Created `wiki/papers/SimchiLevi2022Bypassing.md`
- Created concept page: realizability
- Created author pages: Simchi-Levi-David, Xu-Yunzong
- Updated oracle-efficiency.md, offline-oracle-efficient-bandits.md (FALCON now linked)
- Updated overview.md (3 papers), index.md (5 new entries)

## [2026-05-27] ingest | Cost Aware Best Arm Identification (arXiv 2024)

- Renamed raw file `2402.16710v2.pdf` → `Kanarios2024Cost.pdf`
- Created `wiki/papers/Kanarios2024Cost.md`
- Created concept pages: best-arm-identification, cabai
- Created author pages: Kanarios-Kellen, Zhang-Qining, Ying-Lei
- Created topic page: best-arm-identification
- Updated overview.md (now covers 2 papers/2 areas), index.md (10 new entries)

## [2026-05-27] ingest | Taming the Monster Every Context (arXiv 2026)

- Created `wiki/papers/Qin2026Taming.md`
- Created concept pages: contextual-bandits, oracle-efficiency, offline-regression-oracle, exploitative-f-design, doec, dec
- Created author pages: Zhang-Chicheng (advisor), Qin-Hao
- Created topic page: offline-oracle-efficient-bandits
- Updated index.md (8 new entries), overview.md (1 paper ingested), log.md
- Removed venues category from schema per user preference
- Note: ε-SEC concept page deferred; referenced as [[epsilon-sec]] from doec.md

## [2026-05-27] setup | Wiki initialized

- Created folder structure: raw/papers, raw/assets, wiki/{papers,concepts,authors,venues,topics,queries}
- Created CLAUDE.md schema (ingest, query, lint workflows; page format specs)
- Created index.md and log.md
- Created overview.md as blank slate for synthesis
- Domain: ML/AI research; primary sources: academic papers
