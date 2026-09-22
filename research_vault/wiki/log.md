# Operation Log

Append-only. Each entry: `## [YYYY-MM-DD] <operation> | <title or summary>`  
Parse with: `grep "^## \[" wiki/log.md`

---

## [2026-08-19] update | Move the PFQL Algorithm 1 walkthrough to its own page

Corrects the placement from the previous entry: the Algorithm 1 detail was folded into `concepts/fqi/pessimistic-fitted-q-learning.md`, but it belongs in a dedicated file.

- Created `concepts/fqi/pfql-algorithm-1.md` — a full walkthrough, expanded well beyond what fit as a section: pseudocode block, notation table, the three structural points (backward over $h$ once, no $\gamma$, $\hat V_{H+1}=0$ as a terminal condition not a heuristic), a paragraph per line, the "pessimism inside the backup" argument with a worked penalty table, a hyperparameter/conditions table (including the Theorem 3.2 burn-in on $K$, read off the source), the linear / tabular / GLM specializations, VAFQL as a one-line diff, where the guarantee comes from, and an explicit "what Algorithm 1 does not do" section
- Trimmed `pessimistic-fitted-q-learning.md` back to its concept-page role (definition, three objectives, guarantee statement, VAFQL, related work) with a blockquote pointer to the walkthrough. 134 -> 81 lines; no content lost, just relocated.
- Retargeted the entry points at the new page: `extrapolation-error`, the saved query `2026-08-19-offline-fqi-walkthrough`, `papers/Yin2023Offline.md`, and `index.md`
- Specializations are new relative to the earlier section: linear $\Rightarrow$ PEVI exactly, one-hot $\Rightarrow$ $m(s,a)$ is the literal visitation count and $\Gamma_h \propto \beta/\sqrt{N(s,a)}$ (recovering VPVI), GLM $\Rightarrow$ the link derivative enters the width
- Vault now 91 pages; `concepts/fqi/` holds 6.

## [2026-08-19] query | Offline FQI walkthrough

- Saved `wiki/queries/2026-08-19-offline-fqi-walkthrough.md` — threads the multi-turn FQI discussion into a reading path rather than a reference page, so it complements the concept pages instead of duplicating them
- Covers, in the order the questions actually arose: what FQI fits (the $Q$-function directly, with labels from the previous iterate -- not dynamics, not a reward model); whether the algorithm depends on offline vs. online (it does not -- FQI is how you learn from data, offline/online is how you obtain it); the multi-action target question and why a fitted $Q$ can evaluate unobserved pairs at all; why $\max$ turns that into a selection effect; the two families of fix (penalize the query vs. refuse it); and Algorithm 1 of [[Yin2023Offline]] as the concrete instantiation
- Links out to [[fitted-q-iteration]], [[extrapolation-error]], [[pessimistic-fitted-q-learning]], [[implicit-q-learning]], [[pessimism-principle]], [[overestimation-bias]], [[coverage-coefficient]], [[instance-dependent-bounds]], [[offline-reinforcement-learning]]
- Backlinked from `concepts/fqi/fitted-q-iteration.md` so the query page is not an orphan (the vault's only other query page reached that state only after the 2026-08-19 lint)
- Updated `index.md`: +1 Queries entry. Vault now 90 pages.

## [2026-08-19] update | Walk Algorithm 1 of Yin2023Offline (PFQL) line by line

- Expanded `concepts/fqi/pessimistic-fitted-q-learning.md` with a **line-by-line reading of Algorithm 1**, framed against the [[fitted-q-iteration]] template: lines 4 and 9 alone are FQI, lines 5-8 are the entire modification.
- Per line: the finite-horizon structure (one backward sweep over $h$, a separate $\hat\theta_h$ per stage, no $\gamma$); the ridge term's double duty in line 4 and the fact that its argmin has no closed form; $\Sigma_h$ as the Gram matrix of *gradients* at $\hat\theta_h$, collapsing to $\sum_k\phi\phi^\top+\lambda I$ under linearity; $\Gamma_h = \beta/\sqrt{m(s,a)}$ as a $1/\sqrt{n}$ width with $n$ = effective sample size along the gradient direction; the clip at $H-h+1$ justified by rewards in $[0,1]$; and $\langle\cdot,\cdot\rangle_\mathcal{A}$ as $\arg\max_a$ written for stochastic policies.
- **New subsection "Why the pessimism goes inside the backup."** $\hat V_h$ from line 9 becomes line 4's target next iteration, so the value backed up already has $\Gamma_h$ subtracted. That is what stops [[extrapolation-error]] propagating: an unsupported action has small $m(s,a)$, large $\Gamma_h$, and loses the max. Applying pessimism only at policy extraction would give a cautious policy built on already-contaminated values -- a different, weaker algorithm. This point was not recorded anywhere in the vault.
- **New subsection "Where the guarantee comes from."** The decomposition $v^\pi - v^{\hat\pi} \le \sum_h 2\mathbb{E}_\pi[\Gamma_h]$, valid only when $|(\mathcal{P}_h\hat V_{h+1} - f(\hat\theta_h,\phi))(s,a)| \le \Gamma_h(s,a)$ (verified against the source, Section 3.1). Two consequences drawn out: the price of pessimism is the penalty along the *comparator's* trajectory, which is why the bound is instance-dependent; and the validity condition is where the nonlinearity bites, since $\Gamma_h$ is built from the very $\hat\theta_h$ it is meant to protect against.
- Cross-updated `concepts/fqi/extrapolation-error.md` (placement matters as much as penalty size) and the variants table on `concepts/fqi/fitted-q-iteration.md`.
- No new pages; still 89.

## [2026-08-19] update | Offline vs. online FQI; what FQI actually fits

Expanded `concepts/fqi/fitted-q-iteration.md` rather than adding a page -- the material is about a property of the template itself, and the Variants section already carried the undeveloped seed ("the data setting, not the algorithm").

- New section **"Offline vs. online FQI"**, built on the thesis: *FQI determines how you learn from data; offline vs. online determines how you obtain it.* The Bellman update is identical in both regimes and inspects no provenance; what differs is whether $\mathcal{D}$ can change in response to what has been learned. Includes the online loop ($\mathcal{D}_1 \to Q_1 \to$ collect with $\pi_1$ + exploration $\to \mathcal{D}_2 \to \cdots$), the correction feedback loop, and a 9-row comparison table.
- The framing that makes it click: online, the data distribution depending on the learner is usually called a difficulty, but for this failure mode it is a gift -- an erroneous $Q(s,B)=100$ is self-refuting online (acting on it generates the evidence that kills it) and self-*reinforcing* offline. Offline-specific machinery (pessimism, conservatism, support constraints) is therefore compensation for a missing feedback loop, not a different idea about Bellman regression.
- New subsection **"What is actually being fitted"**: not the dynamics $P(s'|s,a)$, not a reward model -- the $Q$-function directly, with labels generated by the previous iterate. Notes that this is why FQI is model-free while still performing dynamic programming, and that the moving-target nature of the labels is the source of its instability.
- Added the **tabular Q-learning** relation to Variants: FQI = Q-learning-style Bellman targets + batch supervised regression (the DQN link was already there; the incremental-limit framing was not).
- Updated `concepts/fqi/extrapolation-error.md` to point at the new section instead of re-deriving the online/offline asymmetry, and `topics/offline-reinforcement-learning.md` to state the regime/template distinction once at the top.
- No new pages; page count unchanged at 89.

## [2026-08-19] update | Document extrapolation error in offline FQI

- Created `concepts/fqi/extrapolation-error.md` — the train/query mismatch that defines the offline setting: FQI fits $Q$ on $(s_i,a_i)\sim d^\mu$ but its target evaluates $\max_{a'\in\mathcal{A}}Q_k(s_i',a')$ over every action, including unsupported ones
- Contents: worked 3-action example showing an unconstrained $Q_1(s_3,B)$ contaminating the iteration-2 target; why a fitted $Q$ can predict an unobserved pair at all (shared parameters; the linear case $\phi(s_3,B)^\top\theta$ makes it explicit); the three coverage cases (exact pair observed / absent but similar data / far outside support); the unseen-pair vs. unseen-action distinction; the tabular contrast showing generalization and extrapolation are the same mechanism seen from two sides; why $\max$ turns zero-mean error into a selection effect; why online RL self-corrects and offline cannot; and a worked LCB table where the penalty flips the chosen action from an unsupported $B$ to a supported $A$
- **Kept distinct from [[overestimation-bias]]**, with a comparison table on both pages. Overestimation is a statistical bias present when every action is observed and estimates are merely noisy; extrapolation error is a coverage failure that no amount of data on the wrong distribution repairs.
- Records the point that vanilla FQI maximizes over all of $\mathcal{A}$, *not* over dataset-supported actions — these are different algorithms, and the second is essentially what [[implicit-q-learning]] implements via an upper expectile. Noted on both pages.
- Also explains why concentrability / [[coverage-coefficient]] conditions are unavoidable rather than technical conveniences: FQI's greedy $\pi_k$ induces $d^{\pi_k}$, which need not resemble $d^\mu$.
- Updated `concepts/fqi/fitted-q-iteration.md`: annotated the worked example as single-action (the multi-action case is where the difficulty lives), expanded the danger section with the train/query statement and a pointer, marked coverage error as the only offline-specific member of the three-error split, and sharpened the IQL row of the variants table
- Cross-linked from `overestimation-bias`, `implicit-q-learning`, `pessimistic-fitted-q-learning`, `pessimism-principle`, `coverage-coefficient`
- `concepts/fqi/` now holds 5 pages. `index.md` +1 entry.

## [2026-08-19] update | Document Fitted Q-Iteration; group the FQI family under concepts/fqi/

- Created `concepts/fqi/fitted-q-iteration.md` — the base template the vault referenced everywhere but never defined: Bellman relabelling + least-squares refit, the ADP view $Q_{k+1}\approx\Pi_\mathcal{F}\mathcal{T}Q_k$ and why $\Pi_\mathcal{F}\mathcal{T}$ need not contract (hence Bellman completeness, not just [[realizability]]), the backward reward-propagation intuition, a worked 3-transition example, the offline extrapolation failure mode, and the statistical / approximation / coverage error decomposition
- **Structural change: `concepts/` may now contain topical subfolders.** Created `concepts/fqi/` and moved four pages into it: `fitted-q-iteration`, `deep-q-network`, `implicit-q-learning`, `pessimistic-fitted-q-learning` — i.e. the template and the things that ARE FQI instantiations. General primitives (`expectile-regression`, `overestimation-bias`, `pessimism-principle`, `softmax-bellman-operator`) stayed flat because other clusters reference them too.
- Wikilinks are unaffected: Obsidian resolves `[[slug]]` by filename, not path, and the move introduced no new basename collisions. Verified 0 broken links after the move.
- Updated `CLAUDE.md`: subfolder entry in the directory layout, a naming-table row, and a "Concept subfolders" rule block (links stay `[[slug]]` and never include the folder; filenames unique vault-wide; create a subfolder only at 4+ pages with a clear parent page; anything walking the wiki must recurse).
- Cross-linked the new page from `pessimistic-fitted-q-learning`, `implicit-q-learning`, `deep-q-network`, `pessimism-principle`, `overestimation-bias`, `expectile-regression`, and `topics/offline-reinforcement-learning` — the last now frames the field as "FQI with one of its three steps modified" (target / regression / policy extraction), with a table of instances on the FQI page
- Updated `index.md`: +1 entry, plus a note under Concepts that FQI-family pages live in `concepts/fqi/`
- Noted for a future lint: `best-arm-identification`, `monte-carlo-tree-search`, and `test-time-scaling` each exist as BOTH a concept and a topic page, so `[[slug]]` for those three is already ambiguous in Obsidian. Pre-existing, unrelated to this change.

## [2026-08-19] ingest | Epsilon-First Policies for Budget-Limited Multi-Armed Bandits (AAAI 2010)

- Renamed raw `1817-8271-1-PB.pdf` -> `TranThanh2010Epsilon.pdf`
- Replaced the `papers/TranThanh2010Epsilon.md` STUB (created in the 2026-08-19 lint pass) with a full page written from the source
- Created concept page: `epsilon-first`
- Created author pages: `Chapman-Archie`, `Munoz-de-Cote-Enrique`; added the 2010 paper to `Tran-Thanh-Long`, `Rogers-Alex`, `Jennings-Nicholas-R`
- **ATTRIBUTION CORRECTION.** The vault credited [[TranThanh2012Knapsack]] with introducing the [[budget-limited-mab]] and its unbounded-knapsack characterization. The source shows both are from the 2010 paper ("we introduce a new version of the MAB, a budget-limited MAB with an overall budget"; "a MAB with an overall budget limit reduces to an unbounded knapsack problem"), along with the reward-density statistic and the density-ordered greedy step. Fixed in: `concepts/budget-limited-mab.md` (`introduced_by` and body), `papers/TranThanh2012Knapsack.md` (TL;DR, Problem, Connections), `topics/budget-limited-bandits.md`, `index.md`, `overview.md`. 2012's contributions -- [[kube]], fractional KUBE, $O(\ln B)$, matching lower bound -- are unchanged.
- **Precision fix.** The $O(B^{2/3})$ rate is NOT stated in the 2010 paper. Corollary 2 gives $L \le 2\varepsilon B D_{\max} + 2B\sqrt{(-\ln\delta)\sum_j c_j/(\varepsilon B)}$ w.p. $(1-\delta)^k$; optimizing over $\varepsilon$ yields $\varepsilon^* \propto B^{-1/3}$ and hence $B^{2/3}$. That optimization, and the "provably stuck" claim, are [[TranThanh2012Knapsack]]'s. Noted on both paper pages and on `concepts/kube.md`.
- Updated `concepts/upper-confidence-bound.md`: the 2010 paper uses UCB as an exploration subroutine and finds it no better than uniform -- evidence that the $B^{2/3}$ ceiling is caused by the phase split, not the sampling rule
- Updated `overview.md`: budget-limited paragraph re-attributed; sources 13 -> 14
- Updated `index.md`: +3 entries (1 concept, 2 authors), 2 entries corrected, stub description replaced
- No outstanding stubs remain in the vault.

## [2026-08-19] lint | Vault-wide link repair: 0 broken links, 0 orphans

- Scanned all 84 wiki pages for unresolved `[[links]]`, stubs, and pages with no inbound links. Scanner now strips code spans first -- `[[slug]]` in index.md and `[[links]]` in log.md are documentation text inside backticks, not real links, and were previously false positives.
- **Broken links fixed (8 targets, 6 new pages):**
  - `authors/Carin-Lawrence.md` -- third author of [[Song2019Revisiting]], omitted at ingest
  - `concepts/pessimism-principle.md` -- referenced from `importance-weighting`; now the hub for the offline LCB idea across [[Ryu2025Improved]], [[Yin2023Offline]], PEVI, CQL
  - `concepts/upper-confidence-bound.md` -- referenced from `kube`, `budget-limited-mab`, [[TranThanh2012Knapsack]]
  - `concepts/eluder-dimension.md` -- referenced from `doec`
  - `concepts/epsilon-sec.md` -- referenced from `doec`, [[Qin2026Taming]], `topics/offline-oracle-efficient-bandits`; written from the source PDF (Definition 3, Theorem 3, Propositions 1 and 3)
  - `papers/TranThanh2010Epsilon.md` -- STUB, clearly banner-marked: no PDF in `raw/papers/`, content sourced only from what [[TranThanh2012Knapsack]] states about its predecessor
- **Orphans fixed (6 pages, 0 new content):** [[Foster2025Foundation]] now links [[spanner-sampling]], its authors Zakaria Mhammedi / Dhruv Rohatgi, and the saved query [[2026-06-16-foster2025-sections-1-4]]; [[Ryu2025Improved]] now links J. Jon Ryu and Kwang-Sung Jun. All were reachable only from `index.md`.
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
- Noted for lint: pre-existing broken link `Lawrence Carin` in `Song2019Revisiting.md`; garbled tail text in two `index.md` entries (`test-time-scaling` concept and topic lines)

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

## [2026-08-30] update | Dissolve overview.md; create smooth-aggregators topic
- overview.md deleted: per-area synthesis verified already present in topic/concept pages; consensus/debate bullets likewise
- Its one unique thread became [[smooth-aggregators]] (power-mean MCTS / softmax DQN / expectile IQL; unified analysis open)
- Related-Topics links added from [[monte-carlo-tree-search]] and [[offline-reinforcement-learning]]; index.md updated
- Schema: overview.md removed from layout and from the ingest workflow (steps renumbered)

## [2026-08-30] update | Boundary rules encoded; twins resolved; authors/ removed
- CLAUDE.md: new Page-Type Boundary Rules section (concept-vs-topic test, topic-when-earned, hygiene, no author pages); no-delete rule amended for redundant/regenerable pages
- Twins audit: topic/concept basename collisions found (BAI, MCTS, TTS). BAI topic earned -> renamed [[cost-aware-bai]], re-definition stripped. MCTS and TTS topics (2 papers each, below threshold) merged into their concept pages and deleted; per-paper open questions already lived on paper pages
- authors/ (36 pages) deleted: list + regenerable themes only; the one cross-paper insight (Kaufmann bridging MCTS<->BAI) already on [[Dam2024Power]]. Author links unlinked to plain names in 17 files; Authors section dropped from index

## [2026-08-30] update | Flatten concepts/fqi/ into concepts/
- Six FQI-family pages moved to top-level `concepts/`; subfolder removed; links unchanged (basename resolution)
- Schema: cluster-subfolder mechanism replaced by the hub-page convention — a family is expressed by its hub ([[fitted-q-iteration]]) + member links, never a folder

## [2026-08-30] update | Concept naming rules encoded; dec/doec spelled out
- CLAUDE.md: five concept-naming rules (citable-name kebab, acronym policy, aliases:, hub-prefix restriction, name-shape-encodes-type); concept template gains `aliases:`
- Renamed [[decision-estimation-coefficient]] and [[decision-offline-estimation-coefficient]] (were dec/doec); links updated; aliases added to 10 concept pages

## [2026-09-01] ingest | Notes on Fitted Q-iteration (Jiang 2020, lecture note)
- Non-Zotero source; created [[fqi-finite-sample-analysis]] — Tuan's verbatim line-by-line walkthrough of the finite-sample analysis (A/B/C decomposition, Lemma 1, coverage step, Pythagorean identity, Bernstein fast rate, assumption map)
- Inline math delimiters normalized \( \) -> $ $ per formatting rules; wording unchanged
- Linked from the [[fitted-q-iteration]] hub; index updated
- raw/papers/Jiang2020Fitted.pdf PENDING — the PDF still needs to be dropped into raw/papers/ (only extracted text was available at ingest)

## [2026-09-16] query | Value-based offline contextual bandits (in Jun's CSED703Q note framework)
- Saved [[2026-09-16-offline-cb-value-based]]: DM/DR warm-up, greedy (Lemma 1') vs LCB-greedy (Lemma 2') mirroring the note's Lemma 1/2, unification (LCB-greedy = PES over A^X with the DM estimator), instantiations (tabular C*, linear PEVI H=1 + l_p sets, general F Bellman-consistent pessimism, NeuraLCB), Xiao et al. on necessity of pessimism, instance-dependent/fast rates, side-by-side table, reading order with verify-before-citing flags
- Source note: raw/papers/Jun2026Offline.pdf PENDING (text-only)
- Terminology guard added: "offline" (logged data) vs the vault's offline-oracle-efficient (online) topic
- §1 restates the note's problem definition (1.1, near-verbatim) and adds a policy- vs value-based subsection (1.2: what is estimated, what each assumes, role of mu, explicit vs implicit policy class, granularity of pessimism, computation, selection vs learning). Two transcription remarks recorded: the note's x~D in v(pi) is read as nu; T vs n unified to n

## [2026-09-17] update | Direct-method provenance verified on [[2026-09-16-offline-cb-value-based]]
- Checked against the PDFs: "direct method (DM)" is coined in dudik2011Doubly (§1, §2.1; journal version Dudík–Erhan–Langford–Li, Statist. Sci. 2014); the pre-name "regression approach" with the sqrt bound is Beygelzimer & Langford 2009 (Thm 6.1); the L2 form of Lemma 1' is Murphy 2005 (JMLR), restated as (3.1) in Qian & Murphy 2011, whose Thm 3.1 (margin -> fast rates) predates Hu–Kallus–Uehara 2021
- Page edits: provenance paragraphs in §2 and §3; §7 margin bullet re-credited to Qian & Murphy; §1.3 cites the Qian–Murphy toy example for the selection mismatch; reading order gains an "origins" item (renumbered 7–11)
- Not in Zotero: Murphy 2005, Qian & Murphy 2011, Beygelzimer & Langford 2009, Wang–Agarwal–Dudík 2017 (cited author–year)
- §2 rewritten as a self-contained technical section (2.1 method with sample split; 2.2 intuition; 2.3 analysis — Lemma 2.1 bias/variance, Lemma 2.2 transfer with C_2(pi) and C_F(pi), Thm 2.3 DM error, Thm 2.4 realizable regression rate 32/3·ln(|F|/δ)/n from (U) of [[fqi-finite-sample-analysis]], Cor 2.5, Prop 2.6 misspecification floor; 2.4 what it establishes; 2.5 limitations; 2.6 DR extension with Prop 2.7 bias/variance in bandit form; 2.7 provenance). Earlier over-claim that SWITCH is minimax-optimal removed (the verified claim is that IPS/DR match the lower bound)
- Whole page then rebuilt as a self-contained document (§0 summary; §1 setup with notation table + standing assumptions + two approaches; §2 shared toolkit — Hoeffding/Bernstein/self-normalized, the two selection lemmas 2.4–2.5 that yield MaxIPW/PES and greedy/LCB alike, Pythagorean identity, Bernstein condition, realizable rate 12·ln(N/δ)/n with proof, version space, coverage C_2/C_∞/C_F + transfer lemma incl. linear case, tabular/linear widths; §3 direct method + DR; §4 greedy incl. L2 form (Murphy 2005) and two-action tightness; §5 LCB-greedy + "LCB-greedy = PES over A^X"; §6 tabular/linear/version-space/neural; §7 necessity; §8 Bernstein widths + hard-gap 1/n theorem with proof; §9 table; §10 provenance/reading/flags). Constants re-derived on-page (12 instead of the FQI page's 32/3; b=3 in Bernstein). Policy-side fast-rate claim softened: 1/n also holds under a policy-value gap, which is class-dependent
- Rewritten once more around the literature's own formulation (the note now supplies only notation + §1.1): value-based = regression → greedy, pessimistic form = argmax_a f̂ − b(x,a) with an uncertainty quantifier; "estimate v̂(π) then select" documented as the policy-optimization view that coincides with greedy/LCB when Π is unrestricted (Prop 7.1), is cost-sensitive classification with imputed rewards when Π is restricted (Thm 7.2), and is policy-level (version-space) pessimism otherwise. Analysis organized around the plug-in decomposition (Lemma 3.13, uniform coverage) and the pessimism lemma (Lemma 3.14 = Jin et al. Thm 4.2 at H=1, single-policy coverage). Verified against PDFs today: Brandfonbrener et al. definitions/action-stability, Rashidinejad Def 1 + Thms 4–5 + Prop 1, Jin Thm 4.2 + bonus, Xie Eq 3.2 + Def 1, Xiao index rules + weighted-minimax, Li–Ma–Srebro abstract (π̂_∞ = LCB dominates π̂_2 = BCP), Hu–Kallus–Uehara abstract rates; corresponding flags cleared in §10.3
- Transcription-remarks paragraph dropped from §1.1: the page no longer presents the section as a quotation, so notational deviations from the note belong in this log rather than in the text. The deviations remain: the note's x~D inside v(pi) is read as nu (Jun's own Overleaf section writes J(pi)=E_{x~nu}[...], confirming the slip); T vs n unified to n; and the note's given policy class Pi is replaced by all deterministic policies, so pi* is the global optimum — now carried by assumption (A5) alone
- Li–Ma–Srebro 2022 read in full: page corrected — PUNC (ℓ∞ rule) generalizes *tabular* LCB and is a policy-level rule; the pointwise linear rule (PEVI / Thm 5.3) is a context-wise enlarged ℓ2 set outside their family, looser by d and Jensen but valid for all test distributions; π̂_2 = BCP = the version-space rule; their coverage quantity is ‖Σ_D^{-1/2} E_x φ(x,π*(x))‖_q; Thm 2 lower bounds over CB_q(Λ) and adaptive minimax optimality recorded in §5.4/§10.1

## [2026-09-17] update | Intuition and provenance for the greedy bounds on [[2026-09-16-offline-cb-value-based]]
- §4.2 gains two paragraphs. "What the two theorems say": Thm 4.1 charges the model error at exactly two actions, the $\pi^*$ term is (up to a factor 2) what the pessimistic rule also pays by Lemma 3.14, so the chosen-action term is the entire difference between the two approaches; Thm 4.2 answers a different question, converting Thm 3.6's average squared error under $d^\mu$ into a decision bound, the square root being the Cauchy–Schwarz step of Lemma 3.9(a) — hence $1/n$ squared error becomes $1/\sqrt n$ regret. "Provenance": Thm 4.2 is Murphy 2005 / Beygelzimer–Langford 2009 Thm 6.1 restated with the second-moment coefficients of Def 3.8 in place of the original $\mu_{\min}^{-1/2}$ constant; Thm 4.1 is an assembly of Lemma 3.13 and the Jin et al. quantifier, not a named result in any single paper; Thm 3.6 and Cor 4.3 are likewise standard
- Two corrections in §4.3. Hoeffding does not give $b(1)=0$: $b(1)=0$ is a valid quantifier because action 1 is noiseless, while $b^{\mathrm H}$ of Prop 3.11 supplies only $b(2)$. And $C^*=(1-p)^{-1}$, not 1, since $\mu$ plays the optimal action with probability $1-p$ (the conclusion is unchanged either way)
- §10.3 gains a flag: the Singh & Yee (1994) attribution for the classical greedy-policy loss bound is from memory, not verified against the PDF
- §1.1, §2.1, §3 and §4 transcribed into the Overleaf research log (`tuan-research-log/02_offline_contextual_bandits.tex`, commits 47f2417..89ab4df) in that project's notation: $q^*$ for the mean reward, $T$ for the sample size, $\lvert\mathcal F\rvert$ for the class cardinality since $N(x,a)$ is the per-cell count there, $\gamma$ for the two-action gap since $\Delta(\cdot)$ is the suboptimality functional, and the toolkit placed after §4 rather than before it, with all cross-references by label

## [2026-09-17] update | Two symbols for the two roles of an uncertainty quantifier on [[2026-09-16-offline-cb-value-based]]
- The $\delta$-uncertainty quantifier condition (Def 3.10) serves an analysis role and an algorithmic one, and the two now have separate symbols throughout. $b$ is the analysis object: it need not be computable, any valid bound may be used, and the sharpest one gives the sharpest conclusion (Thm 4.1, which the greedy rule never evaluates). $\Gamma$ is the penalty an algorithm subtracts in (LCB), so it must be computable and must in addition be a valid quantifier — pessimism-validity in the sense of Li–Ma–Srebro. Renamed across §0, §1.2 (notation table gains the distinction), §2.1, §3.4–3.5, §4.3, §5.1–5.6, §7.1, §8.1 and §9
- Of the tabular quantifiers, $b^{\mathrm H}$ is computable and serves in either role; $b^{\mathrm B}$ involves the unknown $\sigma^2(x,a)$ and is an analysis object until the variance is estimated — noted at Prop 3.11. Prop 3.12's linear quantifier is computable
- Prop 4.4 rewritten to use both symbols, which sharpens (iii): the greedy analysis may take $b(1)=0$, but the pessimistic rule cannot, so it pays $\Gamma(1)$ and picks action 1 only when $\Gamma(1)<\Delta$ — sufficient condition $p<1/(1+2\ln(4/\delta))$, now stated. This replaces the earlier unconditional claim
- Bernstein's almost-sure bound renamed $b\to c$ in Thm 3.2 and its two applications: it collided with the quantifier symbol
- Thm 4.1 now states its hypothesis on $b$ inline instead of citing Def 3.10, and the single provenance paragraph is split into one per theorem
- Same changes carried into the Overleaf research log

## [2026-09-19] update | Greedy instantiations and Rashidinejad's Theorem 4 on [[2026-09-16-offline-cb-value-based]]
- New §4.5: Theorem 4.1 instantiated in the tabular and linear models, in the same two settings §5.3–§5.4 use for pessimism, so the comparison reduces to one symbol. Tabular greedy pays $2\sqrt{S\,C_{\mathrm{unif}}\ln(2SK/\delta)/n}$ against Theorem 5.2's $2\sqrt{S\,C^*\ln(2SK/\delta)/n}$, and the count conditions differ the same way — all $SK$ cells against $\pi^*$'s $S$. Unrelaxed, pessimism pays $\sqrt{\bar C^*}+\sqrt{\bar C^*}$ where greedy pays $\sqrt{\bar C^*}+\sqrt{\bar C_{\mathrm{unif}}}$. Both theorems are ours, derived not quoted
- New coefficient $\bar C_{\mathrm{unif}}:=\mathbb E_x[\max_a1/\mu(a\mid x)]$ in the notation table, the largest $C^\pi_2$ over deterministic $\pi$. Always $\ge K$, so the number of actions — absent from Theorem 4.1 — enters the greedy rate at $\sqrt K$ whatever the logging policy; and the regime where pessimism is fastest ($C^*\to1$) is where the greedy bound is vacuous
- New §5.7: Rashidinejad et al.'s Theorem 4 stated and proved — the algorithm's conventions at unvisited cells, the two auxiliary lemmas, the $T_1/T_2/T_3$ split, the $\mathcal X_1/\mathcal X_2/\mathcal X_3$ context partition, and the single inequality $\sum_{\mathcal X_3}\nu(x)\le\min\{1,10(C^\pi-1)\}$ that produces the $C^*-1$. Proved for an arbitrary deterministic comparator; Theorem 4 is the case $\pi=\pi^*$
- **Discrepancy in their proof:** at the well-covered-contexts step they assert $n(x,\pi(x))\ge4\,n(x,a)$, where the properties invoked give only $5/2$. Written here with $5/2$, hence $\varepsilon\ge\frac13\sqrt{L/n(x,a)}$ and exponent $L/9$ in place of $L/4$; nothing downstream changes. Recorded in §10.3
- §10.3: Rashidinejad's penalty constant $L=2000\ln(2SKn)$ moved from "from memory" to verified. Linear-model caveat added to §4.5: $1/\kappa$ is not scale-free and inherits $\nu$, so under $\phi=e_{xa}$ it is $\ge C_{\mathrm{unif}}$ and Theorem 4.6 is $\sqrt{SK}$ worse than Theorem 4.5
- Source: the Overleaf research log, whose §4 now carries the same two instantiation theorems. The vault keeps material the log has since dropped for readability — the $L_2$-form Theorem 4.2, §4.3, §4.4 and §5.5–§5.6

## [2026-09-19] update | Problem setup and notation of [[2026-09-16-offline-cb-value-based]] realigned to the research log
- §1 replaced wholesale. The old §1.1 (Jun's problem definition) and §1.2 (notation table) are gone; §1 now mirrors the Overleaf log's setup in seven parts — offline data, mean reward/value/optimal policy, learning objective, function approximation and realizability with the tabular and linear models, coverage, relation to offline RL, algorithm families. The notation table is removed per request; the definitions the body relies on ($f(x,\pi)$, $\pi_f$, $\Pi_{\mathcal F}$, the two squared losses, the $\nu\times\pi$ and $L_1$ norms, $\sigma^2$, $\mu_{\min}$, the sample split) are now stated in running text where they belong. Standing assumptions (A1)–(A5) kept — they are referenced by name in §4, §5, §6 and §8
- Whole page renotated to the log's symbols: $n\to T$, $n(x,a)\to N(x,a)$, $r\to q^*$, $\hat f\to\hat q$, $v(\pi)\to J(\pi)$, $\hat v\to\widehat J$, $\mathrm{Regret}\to\Delta$, $D_n\to\mathcal D$, $N=|\mathcal F|\to|\mathcal F|$, split size $n_1\to T_{\mathrm{reg}}$
- Three symbol collisions resolved by hand: Proposition 4.4's reward gap $\Delta\to\gamma$ and Proposition 6.5's model error $\Delta\to g$, both of which would have collided with suboptimality; and Rashidinejad's own $N$, $N(s,a)$ left untouched inside §5.7's notation-translation sentence, where they describe *their* symbols
- Scope box and frontmatter updated: notation now follows the research log, and Jun's note is cited only as the reference point for the policy-based route in §9
- Entries in this log before today use the previous notation ($n$, $r$, $\hat f$, $v$, Regret). The log is append-only, so they stand as written

## [2026-09-19] update | Split [[2026-09-16-offline-cb-value-based]] into [[offline-contextual-bandits]] and [[value-based-offline-bandits]]
- The 618-line query page is now two maintained concept pages plus a 40-line reading path, following the vault's existing pattern of [[fqi-finite-sample-analysis]] with `2026-08-19-offline-fqi-walkthrough` pointing into it
- [[offline-contextual-bandits]] (208 lines) — the setting: offline data, mean reward/value/optimal policy, learning objective, function approximation with the tabular and linear models, coverage, standing assumptions, relation to offline RL, and the value-based/policy-based taxonomy. It also carries the cross-family material, by request: the doubly robust estimator, class-restricted learning with imputed rewards, and the route comparison
- [[value-based-offline-bandits]] (483 lines) — the reward-model family alone: formulation, toolkit, greedy with its tabular and linear instantiations, pessimism with the tabular/linear/version-space/neural cases, Rashidinejad's Theorem 4 with proof, direct method, estimate-then-select, fast rates, per-result provenance
- Sections renumbered on the value-based page (old §2–§8, §10 → §1–§8) with every result renumbered to match. Externally cited results were protected first and verified afterwards: Beygelzimer & Langford's Theorem 6.1, Jin–Yang–Wang's Theorem 4.2, Qian & Murphy's Theorem 3.1 and §3, Xie's Eq. (3.2), Dudík's §1, §2.1 and §5.1.3 all kept their own numbering. Every remaining internal §- and result-reference was checked to resolve
- Two renotation errors from earlier today found and fixed while splitting: the doubly robust estimator and its variance both had the *observed* reward $r$ wrongly converted to the mean $q^*$ — the correction term is $r-\hat q(x,a)$ under the empirical sum, and the $\sigma^2$ in the variance depends on it
- Back-links added from [[contextual-bandits]], [[pessimism-principle]], [[coverage-coefficient]] and [[offline-reinforcement-learning]]. The query page keeps the open follow-ups: ingest the four uningested papers, write the policy-based page, and revisit whether a topic page is earned

## [2026-09-19] update | [[offline-contextual-bandits]] sections numbered; five stale cross-page references fixed
- Sections and subsections numbered §1–§6, matching [[value-based-offline-bandits]]: 1 Intuition, 2 Preliminaries (2.1–2.6), 3 Relation to offline RL, 4 The two families of methods (4.1–4.2), 5 Where the families meet (5.1–5.2), 6 Comparing the two routes. Key Papers / Variants & Related Concepts / Current State stay unnumbered, as on the sister page
- The page's two propositions renumbered to the sister page's `<section>.<n>` scheme: Proposition 1 → 5.1 (Dudík et al., doubly robust), Proposition 2 → 6.1 (MaxIPW and PES)
- Five references in the route-comparison table were left an off-by-one behind by the split — they still used the old query-page numbering, which shifted when §1 moved to this page. Fixed and verified against their targets: Prop. 6.4 → 5.4 (misspecification floor), Def. 3.10 → 2.10 (uncertainty quantifier), Prop. 7.1 → 6.1 (per-policy pessimism on $\mathcal A^{\mathcal X}$), §5 → §4 (pessimism, for the linear coverage quantity), Thm. 8.2 → 7.2 (hard gap). All five point at [[value-based-offline-bandits]]; a note under the table now says so, since §4 and Prop. 6.1 also exist on this page
- $\mathrm{SubOpt}$ dropped as a second name for the suboptimality: the duplicate parenthetical in §2.3 is gone and the restatement of Jin–Yang–Wang's Theorem 4.2 in [[value-based-offline-bandits]] §4.2 now uses $\Delta(\hat\pi)$. The one-time mapping in the terminology guard stays, as does the verbatim $\mathrm{SubOpt}$ inside the provenance section, which quotes authors in their own notation
- Two markup/wording defects from the split fixed in the same table: a mangled cross-page link, `[[value-based-offline-bandits]](c)` → `Lemma 2.9(c)`, and a duplicated clause, "the policy-based route of the policy-based route above"

## [2026-09-19] lint | 0 broken links, 0 orphans; index re-sorted and one missing entry added
- Scope since the last lint (2026-08-19): 3 ingests ([[differentiable-function-approximation]], [[implicit-q-learning]], the Jiang FQI note), 2 queries, 16 updates including today's three-way split of the offline-CB material
- **Links.** 63 pages, 63 distinct wiki-link targets, 0 broken and 0 orphans. The only apparent misses are the literal `[[links]]` and `[[slug]]` placeholders in the conventions prose of this log and `index.md`
- **Fixed — index completeness.** [[oracle-efficiency]] existed and was linked from other pages but had no `index.md` entry; added under Concepts
- **Fixed — index ordering.** Papers and Concepts had drifted out of alphabetical order (5 and 6 displaced entries respectively, from entries being appended next to related ones rather than in position). Both re-sorted case-insensitively; verified as a pure permutation plus the one new line. Topics and Queries were already sorted
- **Reported, not fixed — index entry length.** The convention is "under ~120 chars", but 56 of 61 entries exceed it, several over 200. The guide no longer describes practice; either it should be relaxed or the entries trimmed, and trimming would cost real information. Left for a decision
- **Reported, not fixed — template sections.** Seven concept pages lack one or more of Key Papers / Variants & Related Concepts / Current State: [[decision-estimation-coefficient]], [[decision-offline-estimation-coefficient]], [[exploitative-f-design]], [[offline-regression-oracle]], [[oracle-efficiency]] and [[pfql-algorithm-1]] are short pages that never got them, while [[fqi-finite-sample-analysis]] (2488 lines) omits them as a long technical account. Filling them is synthesis, not repair, and the second case is really the open page-shape question
- **Frontmatter** valid on all 61 pages. **Log headers** all parse except the vault's first line, `[2026-05-27] setup`, which predates the convention
- Questions worth chasing, in priority order: (1) the four papers cited author–year throughout [[value-based-offline-bandits]] — Rashidinejad et al. 2021, Jin–Yang–Wang 2021, Brandfonbrener et al. 2021, Xie et al. 2021 — still have no paper pages, which is the vault's largest citation gap; (2) does Li–Ma–Srebro's adaptive minimax optimality for PUNC survive restriction to a policy class $\Pi\subsetneq\mathcal A^{\mathcal X}$; (3) whether a topic page for offline contextual bandits is earned once those four exist; (4) the policy-based counterpart page to [[value-based-offline-bandits]]

## [2026-09-19] update | Index length guide relaxed; concept-page shape settled and all 39 pages brought into line
- **Index length.** The "~120 chars" guide was measuring the wrong thing: descriptions are median 16 words and max 23 — genuinely one line — while 57 of 61 lines exceeded the cap, and 11 of the 15 longest do so because they contain LaTeX (`$\mathcal{F}=\{f(\theta,\phi(\cdot,\cdot))\}$` is 40 characters of source for one symbol). The `- [[slug]] — ` scaffold alone costs up to 48. Replaced with a ~25-word budget on the description; every current entry passes, and it still bites on real bloat
- **Page shape settled** in `CLAUDE.md`, closing the open question of short definitional pages and long technical accounts both being filed as `concept`. Two shapes are allowed — definitional (the template, ~35–200 lines) and technical account (numbered `##` sections, a scope note, per-result provenance; [[value-based-offline-bandits]] is the reference instance) — but both must end with the same three sections: Key Papers, Variants & Related Concepts, Current State. One `#` per page
- **Seven pages brought into line.** [[decision-estimation-coefficient]], [[decision-offline-estimation-coefficient]], [[exploitative-f-design]], [[oracle-efficiency]] and [[offline-regression-oracle]] — all from the [[Qin2026Taming]] ingest — used `## Related Concepts` and had no Current State. [[fqi-finite-sample-analysis]] and [[pfql-algorithm-1]] used `## Connections` and had neither Key Papers nor Current State. Headings renamed, the two Key Papers sections added, and seven Current State sections written from what each page already establishes
- **[[fqi-finite-sample-analysis]] heading levels fixed.** Its 35 numbered sections (0–34) were `#`, not `##` — 35 h1s in one document, which is why the earlier audit read it as having two sections and no structure. Demoted to `##`, its one subsection to `###`; the title is now the only `#`
- Re-verified afterwards: 0 broken links, 0 orphans, every page indexed, all 39 concept pages carrying the three required sections
- Not done: [[fqi-finite-sample-analysis]] and [[pfql-algorithm-1]] still lack the scope note and provenance section the technical-account shape calls for, and the former's 2489 lines have no reading path. Left as follow-up rather than rewritten here

## [2026-09-19] update | Citekeys resolved for the four uningested offline-CB papers; one misattribution caught
- All four are now in Zotero with PDFs, and their citekeys are recorded in [[2026-09-16-offline-cb-value-based]]. Ingest itself is postponed
- **Wrong paper caught.** The vault attributes the value-based/policy-based terminology, action-stability, and the DR-collapses-to-value-based observation to Brandfonbrener, Whitney, Ranganath & Bruna. Zotero's `brandfonbrener2021Offline` is *Offline RL Without Off-Policy Evaluation* (NeurIPS 2021), a different paper by the same four authors in the same year; the cited one is *Offline Contextual Bandits with Overparameterized Models* (ICML 2021, arXiv 2006.15368), now `brandfonbrener2021OfflineCB`. Verified by fetching the NeurIPS PDF: it contains "action-stab" and "overparameteriz" zero times
- **Author name corrected** in [[value-based-offline-bandits]] §8.1: Rashidinejad, **Zhu**, Ma, Jiao & Russell, not Zhou. The byline was read off the mirrored PDF
- Rashidinejad's mirrored PDF md5-matches the Zotero attachment, so nothing needs re-fetching. It is arXiv v2 (3 Jul 2023), which states that part of the paper appeared at NeurIPS 2021; the numbering §4.7 was verified against is v2's and re-checked here — Definition 1, Proposition 1, Theorems 4 and 5, Lemmas 13 and 14 all present as cited
- Three of the four were keyed to arXiv revision years (`rashidinejad2023Bridging`, `jin2022Pessimism`, `xie2023Bellmanconsistent`) before the venue metadata was corrected. Worth confirming the keys are pinned before ingesting, since the citekey is the page filename, the link target and the mirror filename
- `brandfonbrener2021Offline.pdf` was fetched into the mirror to run the check above and left there; it is a legitimate library paper, simply not one this vault cites

## [2026-09-19] update | Scope notes and provenance for the three numbered concept pages; reading path for the FQI walkthrough
- Applies the technical-account rule settled earlier today to the pages that did not yet meet it
- **[[fqi-finite-sample-analysis]]** — scope note saying what it assumes ([[fitted-q-iteration]]'s template and notation) and what it leaves elsewhere ([[extrapolation-error]], [[pessimistic-fitted-q-learning]], [[pfql-algorithm-1]], [[value-based-offline-bandits]]), plus the reminder that the analysis is of plain, non-pessimistic FQI. New **Reading path** near the top mapping the 35 sections onto the note's own A/B/C decomposition, since 2489 lines with no entry point is unusable. New §35 Provenance: the argument and both rate results are Jiang's, the section titles and commentary are the page's, §32's rigor observation and §34's smooth-backup question are the page's own, and Munos & Szepesvári is characterised only as the note characterises it
- **Provenance gap recorded, not fixed:** the frontmatter points at `raw/papers/Jiang2020Fitted.pdf`, which does not exist. The note was read on 2026-09-01 and never mirrored, so nothing has been re-checked against it; quotations should be treated as from memory until the PDF is in `raw/`
- **[[pfql-algorithm-1]]** — sections numbered §1–§9 (its scope note was already adequate) and a Provenance section separating what is quoted from [[Yin2023Offline]] (the algorithm, Assumption 2.3, the hyperparameter conditions, Theorems 3.2 and 4.1, VAFQL) from what is the page's own framing (pessimism-inside-the-backup as the organising idea; the four limitations in §9, inferred from what the theorems assume rather than stated by the authors)
- **[[offline-contextual-bandits]]** — this page was numbered but had neither a scope note nor provenance, a gap the new rule itself created. Both added; §7 Provenance separates Dudík et al.'s Proposition 5.1 from Proposition 6.1, which is derived here by reading the two suboptimality lemmas over policies, and points at [[value-based-offline-bandits]] §8.3 as the single per-claim record rather than duplicating it
- **Correction to today's numbering entry.** That entry records §2 of [[offline-contextual-bandits]] as "Preliminaries". It is not, and will stay `## 2. Formal Description`. The rename was made but never reached a commit — `890b72b` has the template heading, while the log entry written by the same script says Preliminaries; every other edit from that turn survived, so the page was most likely overwritten by a stale editor buffer between the edit and the commit. On review the template name is the right one anyway: it is the schema's, and the 31 other concept pages that have the section all use it. The earlier entry stands as written, being append-only; this bullet is the correction
- Verified: 0 broken links, every §-reference on all four numbered pages resolves, one `#` per page, all 39 concept pages carrying the three required trailing sections

## [2026-09-19] update | [[fqi-finite-sample-analysis]] source repointed to the author's online note
- The frontmatter pointed at `raw/papers/Jiang2020Fitted.pdf`, which never existed. It now points at the author's copy, <https://nanjiang.cs.illinois.edu/files/cs598/note5.pdf> (UIUC CS598, note 5), and §35 and Key Papers say the same
- Verified 2026-09-19 before repointing: the URL returns 200, 206,648 bytes of `application/pdf`, and the document's title block reads "Notes on Fitted Q-iteration, Nan Jiang, August 30, 2020" — the version the walkthrough was written from. The server's `Last-Modified` is 2021-08-17, which is a re-upload, not a revision: the note's own date is unchanged
- The gap is narrowed, not closed. An unarchived course note can be revised or moved without notice, so keeping a copy in `raw/papers/` is still worth doing, and the page's content has not been re-read against the note since 2026-09-01. §35 says both

## [2026-09-19] update | Dropped the `raw/papers/` suggestion from [[fqi-finite-sample-analysis]]
- The previous entry said keeping a copy of the Jiang note in `raw/papers/` was still worth doing. That is withdrawn: the note is not going into `raw/papers/`, and the page no longer mentions the path at all. Its source is the author's URL and nothing else
- §35 keeps the substance of the risk without the pointer: a course note can be revised or moved without notice, and if the URL stops resolving the walkthrough is the only record of what it said

## [2026-09-19] update | Jun's CSED703Q note read; two policy-route claims corrected
- The note (*Offline contextual bandits*, Spring 2026, Kwang-Sung Jun) was supplied 2026-09-19. It is the vault's only source for the policy route, which until now had none, and checking the existing claims against it found two errors
- **Proposition 6.1 was under-credited.** [[offline-contextual-bandits]] §7 said it was "not taken from any single paper". It is Lemmas 1 and 2 of the note, in exactly the abstracted form the page uses — $N$ variables $X_i$ with means $\mu_i$ and widths $W^L_i,W^U_i$, MaxIPW giving $\mu_1-\mu_J\le W^L_1+W^U_J$ and PES giving $\mu_1-\mu_J\le W^L_1+W^U_1$. Now credited
- **IX and LS were conflated.** §6 said the two "trade a chosen bias $\gamma C_\gamma(\pi)$ for bounded weights". True of IX, which is bounded by $1/\gamma$ with bias exactly $\gamma C_\gamma(\pi)$; false of LS, which the note states is unbounded for fixed $b$ — "in stark contrast to IX" — and which pays $bD_b(\pi^*)$, a different quantity, with $D_b(\pi)\le C_b(\pi)$ and hence the better bound. Corrected and the two now stated separately
- Checked and correct as they stood: the IX bias and coverage entries in the comparison table, and the attributions of IX to `gabbianelli2023ImportanceWeighted` (ALT 2024), LS to Sakhi et al. (NeurIPS 2024) and the hyperparameter adaptation to [[Ryu2025Improved]] (COLT 2025), all of which match the note's own "Loose ends"
- The note is still not in `raw/papers/`; it was read from an attachment, which cannot be written to disk from here. The query page records what it settles so the content survives regardless

## [2026-09-19] update | Jun's note given its citekey, `jun2026CS703Q10`, and mirrored
- The note is in Zotero as `jun2026CS703Q10` (K.-S. Jun, *CS703Q10: Offline contextual bandits*, `bookSection` of "ML Theory", 2026), with a PDF attached. Fetched to `$PAPERS_DIR/jun2026CS703Q10.pdf` and verified as the same 9-page document read earlier today
- All three references switched from the prose description to the citekey, in backticks rather than as a wiki link, since no paper page exists yet — the vault's convention for uningested sources
- The `raw/papers/Jun2026Offline.pdf` intention is dropped. It was the wrong home: a Zotero-keyed source belongs in the PDF mirror via `fetch_paper.py`, and `raw/` is for non-Zotero ingests. No `raw/papers/` reference now remains outside this log

## [2026-09-19] update | Concept pages renamed broadest-to-narrowest; the two PFQL pages merged
- Adopts a broader-to-narrower naming convention for concept pages, applied only where the parent is itself an existing concept page and the child a genuine specialization. 11 of 39 qualified; the reasoning for all 39, including why 28 were left alone, is in [[2026-09-19-concept-renaming-proposal]]
- Renames: `offline-contextual-bandits`→[[contextual-bandits-offline]], `value-based-offline-bandits`→[[contextual-bandits-offline-value-based]], `cabai`→[[best-arm-identification-cost-aware]], `constrained-bai`→[[best-arm-identification-constrained]], `kube`→[[budget-limited-mab-kube]], `epsilon-first`→[[budget-limited-mab-epsilon-first]], `fqi-finite-sample-analysis`→[[fitted-q-iteration-finite-sample-analysis]], `power-mean-mcts`→[[monte-carlo-tree-search-power-mean]], `slg-search`→[[test-time-scaling-slg-search]]
- **Merge.** `pessimistic-fitted-q-learning` and `pfql-algorithm-1` were the same subject at two levels of detail, so instead of renaming both they are now one page, [[fitted-q-iteration-pessimistic]] (291 lines, §1–§11). `concepts/` goes from 39 pages to 38. Nothing was dropped but duplication: the algorithm display, the effective-sample-size reading of the bonus, and VAFQL each appeared on both pages. The concept page's Intuition lost its third and fourth paragraphs, which §4.3 and §8 state more fully; its Theorem 3.2 display now opens §8 ahead of the walkthrough's derivation; the two VAFQL treatments are combined in §9, keeping the walkthrough's "only line 4 changes" framing and the concept page's Theorem 4.1/4.2 detail
- Every inbound link outside this log was rewritten, and the four lines that ended up linking the merged page twice were reworded by hand. Old slugs added as frontmatter `aliases:` on all nine renamed pages, so existing links and habit keep resolving
- `index.md`: the two PFQL entries collapsed into one, all four categories re-sorted
- Entries in this log before today use the old names. The log is append-only, so they stand; 7 old slugs remain referenced here and nowhere else
- Verified: 0 broken links outside this log, 0 index mismatches, 38 concept pages all carrying the three required trailing sections, every §-reference on the merged page resolving, one `#` per page

## [2026-09-19] update | Second naming wave: `bai`, `fqi` and `mcts` family prefixes abbreviated
- Wave 1 made names structural but long — `fitted-q-iteration-finite-sample-analysis` was 41 characters. This wave abbreviates three family prefixes to the standard acronyms, under a four-part rule now recorded in [[2026-09-19-concept-renaming-proposal]]: the acronym must be standard in the literature, unambiguous *inside this vault*, applied to the parent as well as the children, and the full name kept as an alias
- Renames: `best-arm-identification`→[[bai]], `-constrained`→[[bai-constrained]], `-cost-aware`→[[bai-cost-aware]]; `fitted-q-iteration`→[[fqi]], `-finite-sample-analysis`→[[fqi-finite-sample-analysis]], `-pessimistic`→[[fqi-pessimistic]]; `monte-carlo-tree-search`→[[mcts]], `-power-mean`→[[mcts-power-mean]]. Average name length across the eight falls from about 30 characters to about 12
- **`cb` for contextual bandits was rejected on measurement.** In the vault LCB appears 47 times and UCB 38, against CB's 8, and [[upper-confidence-bound]] is its own page — so `cb-offline-value-based` invites reading "CB" as the tail of a confidence-bound acronym. Those three names are also the shortest of the long ones, so the saving was smallest where the risk was highest. `mab`, `tts`, `rl` and `dfa` rejected too: either already carried in the name, or not standard enough to read without effort
- The third part of the rule is the one that matters: abbreviating a child without its parent is exactly the `fqi-` versus `fitted-q-iteration` mismatch wave 1 removed
- Note for anyone reading the history: [[fqi-finite-sample-analysis]] has come full circle. It held that name before today, took `fitted-q-iteration-finite-sample-analysis` in wave 1, and is back. Net effect on the page is one added alias
- Tags were deliberately not touched. `tags: [fitted-q-iteration, ...]` still uses full names; tags are a free-form namespace, not page references, and changing them is a separate decision
- Verified: 0 broken links outside this log, 0 orphans, 0 index mismatches, 38 concept pages all carrying the three required trailing sections. The seven lines that link one page twice were each checked and read correctly — unlike wave 1, no two pages collapsed into one here

## [2026-09-21] update | Concept template trialled on the contextual-bandit pages; `contextual-bandits` marked `-online`
- **Template trial.** [[contextual-bandits-online]] and [[contextual-bandits-offline]] restructured to the proposed concept shape: Definition, scope note, Intuition, Formal Description, **Literature Survey organized by research direction**, Variants, Related Concepts, Current State and Open Problems, Provenance. The survey replaces the chronological Key Papers list as the page's account of its field; Key Papers stays as the short list of what the vault actually holds
- On [[contextual-bandits-offline]] the survey is §4 in five directions — value-based, policy-based, estimators that combine the two, learning within a policy class, and instance dependence with lower bounds. The old §4's two one-paragraph route sketches survive as the openings of §4.1 and §4.2, so nothing downstream broke. Writing §4.1's lineage surfaced something neither page had said: the value-based route runs back to Murphy (2005) and Beygelzimer & Langford (2009), and pessimism is a recent layer on a plug-in rule that is two decades old
- Provenance earned its place again: §4.2 is the one direction with a single source (`jun2026CS703Q10`), and the section now says so, naming the two attributions taken from that note's "Loose ends" rather than from the papers
- **`contextual-bandits` → [[contextual-bandits-online]].** The old name left "online" as the unmarked default while `contextual-bandits-offline` was marked, on a page whose definition was the interactive protocol. Now marked on both sides, with `contextual-bandits` kept as an alias. 14 inbound links rewritten; the `-offline` pages were untouched, the rewrite matching only a bare or piped link. Offline is presented as a **variant** of the online problem, by request — I had argued for "sibling protocols of one problem class", which is the more defensible taxonomy, but the variant framing is simpler and is what the pages now say
- **Rule amendment this forces.** There is now a `contextual-bandits-` family with no parent page. The naming rule should read: a prefix may name a *family* rather than a parent page, when the umbrella is not worth a page of its own
- **Where to stop.** [[decision-estimation-coefficient]] is the same asymmetry — unmarked means online, against `decision-offline-estimation-coefficient`. It is deliberately left alone: DEC is the established name in the literature and "online DEC" is not a term, so marking it would make the vault disagree with its sources to satisfy an internal convention. The rule needs both conditions: mark the default when a contrasting variant has a page **and** the marked form is itself used in the literature
- [[contextual-bandits-online]] was then rewritten to cover the interactive problem only. The batch direction is gone from its survey; in its place is a direction on **what governs the achievable regret** — DEC, DOEC, $\varepsilon$-SEC and eluder dimension, the question of which quantity decides the rate rather than how to compute the policy. Three online directions now, and no offline material outside the one Variants entry. One dangling provenance entry was removed with the batch direction: [[Ryu2025Improved]] was still credited there with nothing in the body citing it
- Verified: 0 broken links outside this log, 0 index mismatches, 38 concept pages

## [2026-09-21] update | `CLAUDE.md` naming section rewritten to match the vault
- The section had drifted out of agreement with the vault in five places after the two renaming waves: rule 1's example (`fitted-q-iteration`, now `fqi`), rule 2's three acronym examples (`kube`, `cabai`, `slg-search` — all three since prefixed), rule 4's "hub-prefix only for members with no standalone name", which wave 1 inverted, rule 5's claim that name shape encodes page type, and a `[[fitted-q-iteration]]` link pointing at a filename that no longer exists
- Rewritten as seven rules, each carrying its reasoning and a live example. Two are new and were forced by this week's work: **a prefix may name a family with no parent page** (`contextual-bandits-`, whose bare form is an alias on [[contextual-bandits-online]]), and **mark a default variant only when a contrasting variant has a page *and* the marked form is used in the literature** — which is why CB is marked and DEC is not
- Rule 7 is new and came out of the last commit: a rename plus a substantial rewrite in one commit falls below git's rename-similarity threshold and is recorded as an unrelated add + delete, so `git log --follow` dead-ends. Verified: at 25% similarity git still refused to pair them. The commit was split, and history now follows back through `a9d70ba` and `ecd34fc`
- Rule 5 of the old section said name shape distinguishes concepts from topics. It does not, and this is evidence for retiring topics: topic `budget-limited-bandits` sits beside concept `budget-limited-mab`, and topic `cost-aware-bai` is a near-anagram of concept `bai-cost-aware`. Replaced with the true statement — the directory encodes the type — plus the vault-wide unique-basename requirement
- Two numbers in the draft did not survive checking. The `cb` rejection cited "LCB 47, UCB 38, CB 8" from the 2026-09-19 measurement; re-measured across wiki pages today it is **51 / 43 / 9**, so the figures are now dated in the text. And `oracle-efficiency` was described as having five children; only three pages point at it as a parent, so the claim was replaced with one that does not depend on the count
- Also corrected two illustrative examples elsewhere in the file that name pages this vault has never held: the `concepts/` directory gloss ("attention, LoRA, RLHF") and the cross-referencing rule's own example link, `[[attention]]` — a broken link inside the rule that tells the agent to link concepts
- **Still divergent, deliberately:** the Concept Page template in `## Page Formats` still describes the old shape (Key Papers / Variants & Related Concepts / Current State). The two contextual-bandit pages now use the trial shape with a Literature Survey. Left alone until the trial is judged; an agent writing a new page today will follow the old template

## [2026-09-21] update | Topics abolished; all 40 concept pages migrated to the survey template
- **`topics/` no longer exists.** The concept/topic split is retired. It forced a filing decision on every ingest, produced near-duplicate names (topic `cost-aware-bai` beside concept `bai-cost-aware`), and split one subject across two pages a reader had to find separately. A concept page now both defines its object and surveys its literature
- **Three topics merged into the concept they were about**, each keeping its old slug as an alias so every existing link — including this log — still resolves: `cost-aware-bai` → [[bai-cost-aware]], `budget-limited-bandits` → [[budget-limited-mab]], `offline-oracle-efficient-bandits` → [[oracle-efficiency]]. The merge is what gave those three pages real surveys; the topics were already organized by research direction, which is exactly what the new template asks for
- `budget-limited-bandits` was an umbrella over *two* concepts, so it split: the shared-budget half to [[budget-limited-mab]], the cost-aware half to [[bai-cost-aware]]. The contrast between the two objectives — reward density $\mu_i/c_i$ under one budget, versus $\sqrt{c_a}$ proportions at fixed confidence — is now stated on both, because "cost-aware bandits" names both in the literature and they do not reduce to each other
- **Two topics became concepts in their own right**, [[offline-reinforcement-learning]] and [[smooth-aggregators]], both definable objects that happened to carry surveys. Filenames unchanged, so no inbound link moved
- **The finding that shaped the rest.** The vault has **14 paper pages against 40 concepts**, and 25 concept pages cite three papers or fewer. A literature survey on every concept would mean writing ~30 of them from background knowledge rather than from anything the vault holds. So: the survey goes where the material exists — 7 pages have one today — and everywhere else the page keeps `## Key Papers` until the papers arrive. `CLAUDE.md` now states this as a rule, with the reason: a page with two paper pages does not have a survey in it, and writing one anyway is how fabricated citations enter
- **All 33 remaining pages migrated structurally**: `## Variants & Related Concepts` split in two, `## Current State` → `## Current State and Open Problems`. The split was made page by page, not mechanically — a variant is a version of *this* concept (a specialization, a limiting case, a differently-parameterized sibling), everything else is a related concept. 28 pages have variants; 12 do not and the section is omitted rather than padded. [[fqi-pessimistic]] gained VAFQL and the linear/PEVI specializations, which were described in its body but never listed
- One bug worth recording: the splitter initially dropped any prose *before* the first bullet. Batch 1 was unaffected — checked, the diffs touch only headings — but [[fqi]]'s section opens with the "FQI with one step modified" table, which would have been lost silently. Fixed before that batch ran
- Rule 7 from this morning proved itself within the hour: the two topic→concept moves were `git mv` plus a full rewrite, and git recorded them as unrelated add + delete. Split into a move-only commit first, and `git log --follow` now reaches back through `0425eeb`
- `CLAUDE.md`: topic page type, page format and boundary rules removed; the concept template updated to the new shape; the ingest, index and cross-reference rules rewritten accordingly. Two mentions of topics remain, both explaining the retirement
- Verified: 0 broken links, 0 orphans, 0 index mismatches, 40 concepts in the directory and 40 in the index in alphabetical order, one `#` heading per page, every page carrying the required trailing sections

## [2026-09-22] update | `raw/` retired from the schema; paper pages wired to Zotero citekeys
- `raw/` is gone from `CLAUDE.md` — all six references. The folder had not existed for some time and nothing was ever tracked under it, so the schema was describing a storage tier the vault does not have
- The replacement rule is what practice already was: **a non-Zotero source is added to Zotero** so it gets a citekey and resolves through the same path rule as everything else — `jun2026CS703Q10`, a course lecture note, is the worked example and its PDF is in the mirror. If a source is online-only and not worth a library entry, the page's `source:` carries the URL instead, which is what [[fqi-finite-sample-analysis]] does. The vault stores no source files of its own
- **The item was larger than six lines.** Every one of the 14 paper pages carried `source: raw/papers/<name>.pdf`, pointing into the dead folder, and none carried the `citekey:` the schema asks of legacy pages. So all 14 were stale in the same way, and the fix was to replace the dead pointer with the field that actually resolves
- **Nine citekeys resolved and verified**, by exact title match against `library.json` *and* by confirming each `<citekey>.pdf` exists in the mirror — end to end through the schema's own path rule, not just a name match. Five page names differ in year from their citekey (`Kostrikov2022Offline` → `kostrikov2021Offline`, `SimchiLevi2022Bypassing` → `simchi-levi2021Bypassing`, `Yin2023Offline` → `yin2022Offline`, `Lardy2025Constrained` → `lardy2026Constrained`): the page names use the venue year, Better BibTeX uses the arXiv posting year. Same papers — titles are character-identical. `Foster2025Foundation` → `foster2025Good` for the same reason on the title word, since the title begins "Is a **Good** Foundation…"
- **Five papers are not in Zotero at all**: [[Li2026Predicting]], [[Ryu2025Improved]], [[Song2019Revisiting]], [[TranThanh2010Epsilon]], [[Yang2025Stochastically]]. Checked three ways — exact title, distinctive title phrase, and author surname (275 Li papers in the library, 38 Yang, 12 Song, none matching). Their frontmatter now says so explicitly rather than pointing at a folder that never held them. They need adding to Zotero; until then no PDF resolves for them
- A methodology note worth keeping: the first surname search returned "0 in library" for *every* author including Li and Song, which is impossible in a 1146-item ML library. The extractor was reading `author`/`creators` as a flat field when entries store `creators: [{firstName, lastName}]`. An implausible negative is worth one more check before it becomes a finding
- Verified: 0 broken links, 0 frontmatter parse failures, 0 `raw/` references outside this log, 9 of 14 paper pages resolving to a mirrored PDF

## [2026-09-22] update | Citekey-year rule recorded; four Zotero entries found still on arXiv years
- Rule added to `CLAUDE.md`, per Tuan: **the citekey's year is the venue year and it wins.** Papers enter Zotero from arXiv and the year is corrected to the venue by hand afterwards, so a page's `year:` and `venue:` should be written to agree with the citekey. Recorded alongside the reverse hazard: a citekey that *matches* on year can still be the wrong paper, which is how `brandfonbrener2021Offline` was once cited for claims belonging to `brandfonbrener2021OfflineCB`. Match on title, then check the year
- **Four of the nine wired citekeys disagree with their page's venue year**, and the rule presumes the Zotero entry has already been corrected — so these are worth looking at before the rule is applied to them:
  - `kostrikov2021Offline` — page records ICLR 2022; the Zotero entry has **no venue field at all** and date 2021-10-12, so it looks like an uncorrected arXiv import
  - `yin2022Offline` — page records ICLR 2023; likewise **no venue field**, date 2022-11-23
  - `simchi-levi2021Bypassing` — page records Mathematics of Operations Research 2022; Zotero has the venue filled but dates it 2021-07-10, which may be online-first rather than issue year
  - `lardy2026Constrained` — page records NeurIPS 2025, Zotero says 2026 with the proceedings filled. This is the one where the citekey is *later*, so here the page may be what is stale
- No page years were changed. For the first two the evidence points at Zotero being the stale side, and following the citekey there would write a year the vault knows to be wrong; the other two need Tuan's call
- **The consequence to weigh before fixing any of them:** correcting a year in Zotero makes Better BibTeX regenerate the citekey, and the citekey is this vault's page filename, link target and PDF mirror filename. `kostrikov2021Offline` becoming `kostrikov2022Offline` would silently break the `citekey:` field committed today. Pin the key in Zotero's Extra field first, as was done for `brandfonbrener2021OfflineCB`

## [2026-09-22] correction | Three citekeys committed this morning were already stale; resolved against live Zotero
- **Correcting the entry above.** It says the nine citekeys were verified "end to end through the schema's own path rule". They were verified against `library.json` and the PDF mirror — and **both are snapshots from 2026-08-29**, three weeks old. The check confirmed internal consistency with a stale snapshot, not with Zotero. The live API is the authority and was not consulted
- Tuan then corrected the venue years in Zotero, Better BibTeX regenerated the keys, and three of the nine changed — the breakage flagged an hour earlier, made real: `yin2022Offline` → **`yin2023Offline`**, `simchi-levi2021Bypassing` → **`simchi-levi2022Bypassing`**, `lardy2026Constrained` → **`lardy2025Constrained`**. All three page `year:` fields were already right; it was Zotero that came into line, and the citekeys moved with it
- Resolved by querying the Zotero API directly through `scripts/fetch_paper.py`'s own index builder rather than the stale export. All 14 pages rechecked: 9 match a live entry, the other 5 are still absent from the library
- `Kostrikov2022Offline` corrected the other way — `year: 2022` → **2021**, `venue: ICLR` → **arXiv**, per Tuan: the paper has no conference venue, the ICLR appearance being a workshop rather than the main track. The mirrored PDF is arXiv:2110.06169v1, 12 Oct 2021, with no venue banner, which is consistent. The page **filename** stays `Kostrikov2022Offline` — legacy paper pages keep their names, since renaming breaks links
- The three renamed PDFs were re-fetched so `<citekey>.pdf` resolves again; all 9 now do. Their old-named copies are still in the mirror as orphans (`yin2022Offline.pdf` and two others) — harmless in a disposable cache, cleared by `fetch_paper.py --sync-all --prune` when someone wants to
- **Two things left for Tuan, neither fixable from here:**
  - `research_vault/library.json` is the schema's documented lookup index and is three weeks stale. It is regenerated by Better BibTeX on the desktop and is read-only here, so every fuzzy-reference lookup until it is re-exported will run against 2026-08-29 data
  - `Yin2023Offline` records `venue: ICLR`; live Zotero says *Proceedings of NeurIPS*. The years agree at 2023, so the citekey rule does not settle it, and the mirrored PDF is a bare preprint with no venue banner. Left unchanged rather than guessed at
- **The general lesson, worth more than the four fixes.** A citekey is not a stable identifier: it is derived from metadata the user edits, so it changes when they fix a year. Anything in the vault keyed on it — page filenames, `citekey:` fields, link targets, mirror filenames — is only as current as the last export. Resolve against the live API when it matters, and pin the key in Zotero's Extra field for anything the vault has already written down

## [2026-09-22] update | Library export refreshed; citekeys and venues reconciled
- Pulled `af1155d`, Tuan's BBT re-export (1094 keyed items, BBT 9.0.63 → 9.0.64). The three-week gap flagged in the correction above is closed, and the export now agrees with the live API
- **All 9 wired citekeys resolve in the fresh export.** `yin2023Offline` now records *Proceedings of ICLR*, so the vault page's `venue: ICLR` was right all along and Zotero came into line — the one open venue question is settled without the page changing
- Two pages were stale in the other direction, saying `venue: arXiv` where Zotero has a real venue, years agreeing: [[Dam2024Power]] → **UAI**, [[Kanarios2024Cost]] → **RLJ**. Corrected per the citekey rule. The five pages still reading `venue: arXiv` are correct: three match Zotero, and two are papers the library does not hold
- **All four postponed ingests now resolve**: `rashidinejad2021Bridging`, `jin2021Pessimism`, `xie2021Bellmanconsistent`, `brandfonbrener2021OfflineCB`. The last is distinct in the export from `brandfonbrener2021Offline` (*Offline RL Without Off-Policy Evaluation*), which is the pair that caused the misattribution on 2026-09-19 — the pin held through a key regeneration that moved four other keys. Ingesting these is now unblocked
- The five papers without a citekey are still absent, rechecked against the new export: [[Li2026Predicting]], [[Ryu2025Improved]], [[Song2019Revisiting]], [[TranThanh2010Epsilon]], [[Yang2025Stochastically]]. [[Yang2025Stochastically]] scored 0.82 on an earlier pass, high enough to look like a near-miss; that was an artifact of comparing truncated titles. On full titles its nearest neighbour is [[Lardy2025Constrained]] at 0.65, a different paper with its own page
- `__pycache__/` added to `.gitignore` — importing `fetch_paper` to query the API left a bytecode file that was briefly staged

## [2026-09-22] update | The five missing papers added to Zotero; every paper page now resolves
- Superseding the entry above, which recorded them as absent: Tuan added all five to Zotero. Resolved against the **live API** rather than the export, which does not yet hold them — `li2026Predicting`, `ryu2025Improved`, `song2019Revisiting`, `tran-thanh2010Epsilon`, `yang2025Stochastically`. All five matched on full title exactly, and every citekey year *and* venue already agrees with what the page recorded, so nothing else needed changing
- PDFs fetched to the mirror. **All 14 paper pages now carry a citekey resolving to a mirrored PDF** — the first time that has been true. Page 1 of three of them was checked against the page title: AAAI-10 proceedings for `tran-thanh2010Epsilon`, PMLR vol. 291 (COLT 2025) for `ryu2025Improved`, and arXiv:1812.00456 for `song2019Revisiting`
- What this unblocks is the surveys written on 2026-09-21. [[budget-limited-mab]] attributes the model, the knapsack characterization and the reward-density statistic to [[TranThanh2010Epsilon]]; [[bai-cost-aware]]'s third direction rests on [[Yang2025Stochastically]]; [[smooth-aggregators]] rests on [[Song2019Revisiting]] being the only instance with a finite-parameter bound. Those claims were written from the paper pages, which were themselves written before the PDFs were reachable — they can now be checked at source, and should be
- `library.json` does not yet contain these five. It is the documented lookup path, so until the next BBT export a fuzzy reference to any of them will fail there while resolving fine against the API

## [2026-09-22] update | `concepts/inactive/` created; 10 pages parked
- Tuan is focusing on contextual bandits and RL, so three-plus families are parked: `bai` and its two children, `budget-limited-mab` and its two, `mcts` and `mcts-power-mean`, and the two `test-time-scaling` pages. **30 active concepts, 10 inactive**
- **No link changed.** `[[slug]]` resolves by basename regardless of directory, which is the property that makes this cheap — and is why it is worth doing this way rather than by renaming or archiving. Verified: 60 pages, 0 broken links, 0 basename collisions across the two directories. All 10 moves are pure `git mv` with content untouched, so each tracked as a rename
- Seven **active** pages still link into the parked set — [[upper-confidence-bound]] reaches all three of BAI, budget-limited and MCTS; [[instance-dependent-bounds]] and [[spanner-sampling]] and [[offline-reinforcement-learning]] reach in too. Those links still work. Worth knowing that the parked set is not isolated: [[upper-confidence-bound]] in particular is a hub whose contrasts point outward into it, so the active pages will keep referring to parked material
- `index.md` keeps all 40 in one catalog, with the parked ones under their own `### Inactive` heading rather than deleted or hidden — the point is a legible working set, not a smaller vault
- **The schema needed amending, not ignoring.** `CLAUDE.md` said `wiki/concepts/` is "fully flat — never a subfolder", a rule rewritten only yesterday. It now permits exactly one subfolder and says why the exception holds: `inactive/` marks *status*, not subject. Filing by topic forces a judgement call on every page and buys nothing; active/inactive is a single binary that keeps the working set legible. The lint step and the directory diagram were updated with it, including the trap that anything walking `concepts/*.md` must walk `concepts/inactive/*.md` too or silently skip ten pages
- **Six of the 14 paper pages now serve only parked concepts** — [[TranThanh2010Epsilon]], [[TranThanh2012Knapsack]], [[Kanarios2024Cost]], [[Lardy2025Constrained]], [[Yang2025Stochastically]], [[Li2026Predicting]]. They were left in `papers/`, since the request was about concepts and a paper page is a record of something read rather than a working note. Flagged in case `papers/` should get the same treatment

## [2026-09-22] update | Notation standardized across the four setting pages; `online-reinforcement-learning` written
- **The anchor is [[contextual-bandits-offline]] §2**, by Tuan's instruction, and it now says so: a correspondence table maps every role — decision point, its distribution, mean reward, optimal action-value, behavior policy, logged pairs, what the learner sees, objective, function class — across offline CB, online CB, and offline/online RL, so that a symbol means one thing on all four pages. Measured before choosing: the offline-CB pair uses $q^*$ ~120 times, $\mu$ ~150, $\nu$ ~90, against 5 uses of $f^*$ and one of $\mathcal D_{\mathcal X}$ online and 9 of $\pi_\beta$ in the RL family. The direction of travel was never in doubt
- **Changed.** Online CB: $f^* \to q^*$, $\mathcal D_{\mathcal X} \to \nu$ (which also ends the clash with $\mathcal D$ the dataset), the smoothed benchmark's base measure no longer written $\mu$, and the DEC paragraph's borrowed $g^*$ tied to $q^*(x,\cdot)$. Offline RL: $\pi_\beta \to \mu$ on the page and on [[implicit-q-learning]] and [[coverage-coefficient]]; the RL-theory literature (Rashidinejad, Jin, Xie) already writes $\mu$, so this picks the theory convention over the deep-RL one rather than departing from the sources. CB family: $f^* \to q^*$ on [[realizability]], [[oracle-efficiency]], [[offline-regression-oracle]]; $\pi_{f^*} \to \pi^*$; $\hat f \to \hat q$. Paper pages untouched — they record each paper's own notation
- **Deliberately not changed, and written down as such.** $\Lambda$ is the benchmark class online (Foster et al.) and the regularized Gram matrix on the value-based page (linear bandits); both are the field's own and never share a page. $K$ is $|\mathcal A|$ throughout the vault while the RL literature uses it for episodes — the RL pages write $T$ episodes and say how to read a quoted rate. $\pi_{\mathrm{ref}}$ on the alignment pages stays: a KL-regularized reference policy is sampling distribution *and* regularization target, which is a different role from a behavior policy, and renaming it $\mu$ would conflate the two. This follows the rule set for DEC: internal consistency is not worth making the vault disagree with its sources
- **Added to make the notation cohere, not just match.** Both online pages now state how regret and suboptimality meet: the round-$t$ term of standard regret has conditional expectation $\Delta(\pi_t)$, so $\mathbb E[\mathrm{Reg}(T)] = \sum_t \mathbb E[\Delta(\pi_t)]$ and the uniform mixture has $\Delta(\bar\pi) = \mathbb E[\mathrm{Reg}(T)]/T$. The offline RL page gained the horizon-one correspondence explicitly: $\mathcal S = \mathcal X$, $r_1 = Q^*_1 = q^*$, $d^\pi_1 = d^\pi$
- **[[online-reinforcement-learning]]** written in the standardized notation. Definition, scope, intuition, formal description with the regret, the minimax and variance-dependent rates, and the bias/optimism decomposition as the mirror of the pessimism lemma; Key Papers, Variants, Related, Current State, Provenance. **No Literature Survey**, by the rule: the vault holds one online-RL paper read in full (Cassel & Rosenberg 2026, not yet ingested) and two adjacent ones, which is not enough to organize by direction. The page says so under Current State rather than pretending
- Back-links added from [[offline-reinforcement-learning]], [[contextual-bandits-online]], [[upper-confidence-bound]] and [[pessimism-principle]], so the page has six inbound links rather than two. Index entry added in order
- Verified: 61 pages, 0 broken links, 0 collisions, frontmatter parses on all nine edited pages, `$$` parity and brace balance hold in every edited math block, no residual $f^*$, $\mathcal D_{\mathcal X}$ or $\pi_\beta$ in the active set outside the one clause of the correspondence table that names $\pi_\beta$ as the deep-RL spelling

## [2026-09-22] ingest | [[cassel2026Quantile]] and [[zanette2019Tighter]] — the two routes to optimism in tabular RL
- Both read in full from the mirrored PDFs and ingested together, at Tuan's instruction, because the first is positioned entirely against the second. No `quantile-of-means` concept page, also by instruction: the estimator lives on the paper page with its mechanism in full, and the concept-level summary sits in [[online-reinforcement-learning]]'s Formal Description
- **[[cassel2026Quantile]]** (Cassel & Rosenberg, arXiv 2606.20107). VIBE: value iteration where every $Q$-value is the $\alpha$-quantile of $B$ estimates trained on disjoint round-robin batches. No bonuses, no counts, no posterior, no distributional assumption. The optimism is not a concentration argument — it rests on Feige (2004), that a batch mean with $+1$ in the denominator falls below the true mean with probability $\ge 1/13$ using *only the first moment*; independence across batches then makes the optimistic count binomial and a Chernoff lower-tail bound finishes it. Theorem 4 gives $22\sqrt{\min\{\mathcal Q^\star, HV^\star\}HSAK\kappa^2}$ plus lower-order, matching both the minimax and the variance-dependent lower bounds
- **[[zanette2019Tighter]]** (Zanette & Brunskill, ICML 2019). EULER: optimistic value iteration with an empirical-Bernstein variance bonus plus a correction for value-function uncertainty. Theorem 1 is the *minimum* of a $\mathcal Q^*$-form and a $G^2/H$-form bound, and the advance over REGAL is that neither quantity is an input — the algorithm adapts without being told which regime it is in. Equation 13 answers Jiang & Agarwal (2018): under $\sum_h r_h \in [0,1]$ a.s., regret is $\tilde O(\sqrt{SAK})$ with **no $H$** in the dominant term
- **What ingesting them corrected, and would not have been caught otherwise.** [[online-reinforcement-learning]] was written yesterday citing both *through* Cassel, and it blurred two things the PDFs separate. First, the settings differ: Zanette's MDP has a single stationary $P$, Cassel's has $P_h$ per step, so their minimax rates are $\sqrt{H^2SKT}$ and $\sqrt{H^3SKT}$ and the numbers are not comparable. Second, **both papers write $\mathcal Q^*$ and mean different things** — Zanette takes one global $\max$ over $(s,a,h)$ (the environmental norm of Maillard et al. 2014), Cassel **sums** the per-step maxima over $h$. Cassel's "similarly to Zanette and Brunskill" is accurate as lineage and misleading as equality. The page now states both definitions side by side and says neither dominates: a max is fragile to a single high-variance pair, a sum accumulates over the horizon
- This is the third time this month that reading a source directly overturned something taken from a second-hand framing, after the Brandfonbrener misattribution and the IX/LS conflation. The pattern is consistent: what survives second-hand citation is the *result*, what gets lost is the *definition it is stated in*
- [[online-reinforcement-learning]] updated throughout — the rate paragraph rewritten with both settings and both variance functionals, a new paragraph contrasting the two ways to buy optimism (bonus versus order statistic), Key Papers replaced with the two links, and the "vault's own gap" open problem retired since the gap it named is now closed. Two sharper open problems took its place: max-versus-sum, and whether either measure predicts empirical difficulty, which neither paper tests
- Also cited from [[instance-dependent-bounds]] (its first two online-RL instances) and [[upper-confidence-bound]] (EULER as the bonus refined for MDPs, VIBE as the counter-case that optimism does not require a confidence width at all)
- Verified: 16 paper pages, 62 pages total, 0 broken links, 0 frontmatter failures, both citekeys resolving to mirrored PDFs, five inbound links each, index sorted

## [2026-09-22] update | Paper pages renamed to their Zotero citekeys; the legacy naming rule retired
- All 14 pre-Zotero pages renamed so that **the filename, the link target, the `citekey:` field and the mirrored `<citekey>.pdf` are now one string**: `Dam2024Power` → `dam2024Power`, `SimchiLevi2022Bypassing` → `simchi-levi2022Bypassing`, `TranThanh2010Epsilon` → `tran-thanh2010Epsilon`, and so on for all 16 pages. 302 link occurrences rewritten across 57 files; `log.md` deliberately untouched, since it is append-only and the retired name now lives in each page's `aliases:`
- Five of the fourteen changed more than case. `Foster2025Foundation` → `foster2025Good`, because the title begins "Is a **Good** Foundation…" and Better BibTeX takes the first substantive word. `Kostrikov2022Offline` → `kostrikov2021Offline`, the one paper with no conference venue. The rest gained a hyphen (`simchi-levi`, `tran-thanh`) or the corrected venue year
- **This overrode a schema rule written six days ago** that said to keep legacy names because "renaming breaks links". The rule's real concern was a *twin* — both spellings existing at once, which on a case-insensitive macOS filesystem is silent data loss — and nine of these renames differ only in case, so the concern was live. A rename leaves one file, not two, so the hazard does not apply; the case-only moves were done through a temporary name to keep the move unambiguous, and git recorded all 14 as renames with no add/delete pair
- `CLAUDE.md`'s "Legacy paper pages" paragraph is replaced by "Paper page filenames are citekeys, exactly", carrying forward the two cautions that remain true: never create the second spelling, and **a citekey is derived from metadata the user edits**, so pin it in Zotero's Extra field before writing a page against it and re-resolve against the live API rather than a stale export. That second caution is not theoretical — three citekeys went stale within an hour this morning
- Verified: 0 case-insensitive basename twins anywhere in the vault, 0 broken links including the log, 16/16 paper pages where filename, citekey and mirrored PDF agree, index Papers section re-sorted

## [2026-09-22] update | Explicit protocol and algorithm blocks on [[cassel2026Quantile]]
- Both written in the `\begin{aligned}` pseudocode style of [[fqi]]'s template, at Tuan's request, so the interaction loop and VIBE read as procedures rather than as prose with equations dropped into it
- **The interaction protocol** now shows what the learner sees and when: the policy committed *before* the episode, the per-step loss and successor observed *only* for the pair actually played, and the regret incurred as the episode is played. Writing it out made three things explicit that the prose had left implicit — no within-episode adaptation, bandit rather than full-information feedback, and no forgiven training phase, which is the structural difference from the batch protocol on [[offline-reinforcement-learning]]
- **Algorithm 1** is now the paper's twelve lines rather than four loose display equations, which makes it directly comparable to the FQI template: VIBE is the same backward sweep with one deletion (the bonus, entirely absent) and two substitutions (one regression becomes $B$ on disjoint data; the single $\hat Q$ in the backup becomes $q_\alpha$ over the ensemble). The round-robin line is visible as part of the mechanism rather than a footnote — it keeps batch sizes within one of each other, which is what Lemma 1's bias bound assumes when it requires $|D_b|\ge\lfloor n/B\rfloor$
- The $+1$ in both denominators is now labelled as load-bearing where it appears: it is exactly the $c\ge1/12$ of Corollary 2, and without it the constant-probability undershoot that drives optimism fails
- A structural check flagged both `aligned` blocks as malformed; that was the checker splitting on `\\` and reading `\begin{aligned}` as part of the first row. Re-checked after stripping the environment: 22 rows, every one beginning `&`, braces balanced per row. Same class of false positive as the inline-code link scan earlier this week — a crude check on LaTeX is worth re-running before it becomes a finding

## [2026-09-22] update | Paper pages brought onto the standardized notation
- Reverses a choice made during the 2026-09-21 standardization, which held that "paper pages record each paper's own notation". At Tuan's instruction they now follow the anchor in [[contextual-bandits-offline]] §2, with each page stating up front which symbols were translated so a reader can still check against the PDF
- **Translated where the difference is only spelling.** $A \to K$ for $|\mathcal A|$; episode count $K \to T$; $\star \to *$ on optima ($V^*$, $\pi^*$, $\mathcal Q^*$) — measured first, the concept pages use the asterisk 53–0 for $V$, 185–0 for $\pi$, so the two new paper pages were the outliers. Episode superscripts $\pi^k, s^k_h, D^{k,b}_h$ became $\pi_t, s^t_h, D^{t,b}_h$, matching [[online-reinforcement-learning]]'s $\pi_t$. [[cassel2026Quantile]]'s Theorem 4 therefore reads $\mathrm{Reg}(T) \le 22\sqrt{\min\{\mathcal Q^*, HV^*\}HSKT\kappa^2}+\dots$ rather than the paper's $\mathrm{regret}_K$ with $SAK$
- **[[zanette2019Tighter]] needed care that the others did not.** Its $T$ is the *total timestep* count with $T \le KH$, whereas the vault's $T$ is the episode count — so the paper's $T$ is the vault's $TH$, and every rate had to be restated rather than relabelled: $\sqrt{HSAT}$ becomes $\sqrt{H^2SKT}$, $\sqrt{SAK}$ becomes $\sqrt{SKT}$. Its instance quantity is now written $\mathcal Q^*_{\mathrm{ZB}}$ throughout, matching the name the concept page already used to keep it apart from Cassel's summed version
- **Not translated, deliberately.** The loss framing on [[cassel2026Quantile]]: the paper minimizes loss, which is *why* the optimistic quantile is the low one and why $\alpha=1/65$ is small. Rewriting it as reward maximization would break the explanation of its own hyperparameter. The page says so where the translation is declared
- **Left alone after checking.** Every remaining `\star` in the vault is a decorated *object* rather than a style choice — $\Sigma_h^{\star-1}$, $\Lambda_h^\star$, $\sigma_h^\star$ on [[yin2023Offline]] and [[fqi-pessimistic]] mark variance-weighted matrices distinct from their unweighted namesakes, and $(\star)$, $(\star\star)$ are condition labels. Two genuine stragglers were fixed: a $V^\star$ on [[kostrikov2021Offline]] and one $\mathcal Q^\star$ on [[zanette2019Tighter]]
- A check reported seven concept pages as still containing `\star` when none did. The pattern was over-escaped — `\\\\star` inside single quotes matches two literal backslashes, not one — so the filter silently matched nothing and the earlier listing went unfiltered. Third time this week a crude check has produced a false reading before a real one; the fix each time was to re-run it rather than report it
- Verified: 0 broken links, `$$` parity and brace balance on all three rewritten pages, every `aligned` row beginning `&`

## [2026-09-22] update | `CLAUDE.md`'s paper template replaced
- The schema's `## Paper Page` block described a shape no page in the vault used any more. It now carries the structure Tuan specified: **TL;DR, Intuition, Formal Problem Definition** (setting and learning protocol, learning objective, how it differs from the surrounding literature), **Assumptions**, **Method**, then Connections and Open Questions
- Two conventions settled this week are written into it rather than left as habit. The **protocol is a procedure**, in [[fqi]]'s `\begin{aligned}` pseudocode style — prose hides what the learner sees and when, and a procedure cannot. And **Assumptions carries an explicit "Not assumed" line**, which on [[cassel2026Quantile]] turned out to be the more informative half: what a reader expects to be required and is not
- **The notation rule is now stated.** Paper pages follow the vault anchor, not the paper, and must open the formal section by naming every translated symbol so the page can still be checked against the PDF. With the two exceptions this week produced: do not translate a *substantive* choice (Cassel's loss framing is why its optimistic quantile is the low one), and where a symbol means different things — [[zanette2019Tighter]]'s $T$ is total timesteps, the vault's is episodes — **restate the results rather than relabelling them**, or the claims shift by $\sqrt H$ in silence
- Two retired sections are documented rather than silently dropped. *Prior Work & Position* became the third part of the formal definition, where it sharpens the problem instead of trailing it. *Strengths & Limitations* is dissolved: strengths restate the contributions, and limitations split between what the paper assumes away and what is untested — each more useful attached to what it qualifies than collected into a verdict
- Ingest step 4 now says to re-resolve the citekey against the **live API** before writing, since filename, link target and mirrored PDF all depend on it and three keys went stale within an hour this morning. Step 5 now points at the paper page's Connections section for concept candidates, replacing the retired "New Concepts" section, and records that a concept fully explained on one paper page and used nowhere else does not need a page of its own — `quantile-of-means` being the worked example
- **[[cassel2026Quantile]] conforms exactly; [[zanette2019Tighter]] does not.** Its notation is aligned but its sections are still Problem / Method / Results / Strengths & Limitations. Left as-is rather than quietly rewritten, since restructuring it is its own change

## [2026-09-22] fix | Display math was silently not rendering on GitHub

- Tuan noticed that [[cassel2026Quantile]]'s protocol block rendered on the GitHub web view while Algorithm 1 sat there as raw LaTeX. Algorithm 1 was not the problem: its commands are all standard MathJax, it is pure ASCII, and it contains no stray `$`
- **The rule is that a `$$` opener must begin its own block.** Glue it to the end of the preceding paragraph — the natural thing to do after a lead-in ending in a colon or comma — and GitHub leaves it as literal text *and fails every later display block in the same file*. That cascade is why a correctly-separated Algorithm 1 died: a glued opener eleven lines earlier had already poisoned the file
- Established by rendering the committed files through GitHub's own API rather than by reading the spec. `fqi.md`, 8 separated openers, rendered 8/8. `contextual-bandits-offline.md`, 15 glued, rendered 0/15. The whole of `contextual-bandits-offline-value-based.md` — 30 blocks, the longest technical account in the vault — had never rendered at all
- **56 glued openers across 7 files**, fixed by inserting a blank line and nothing else: 56 insertions, 0 deletions, every `$$` count unchanged. The three contextual-bandit pages, both RL pages, [[cassel2026Quantile]] and [[zanette2019Tighter]]
- Fenced ` ```math ` blocks were the tempting fix — a code fence is a leaf block, so it cannot be poisoned and needs no blank line. Rejected: `research_vault/.obsidian` has no math plugin, so Obsidian would show every equation as raw code. `$$` with a blank line is the only form that renders in both, so the rule stays and the schema now says why
- Worth noting how long this survived. Nothing was malformed, no tool complained, and the pages look right in Obsidian — the failure was visible only in the browser, on pages nobody had scrolled through since writing them. Added to the lint sweep as step 6

## [2026-09-22] query | Quantile of Means as value-based pessimism for offline contextual bandits → [[2026-09-22-qom-value-based-pessimism]]

- Tuan asked whether [[cassel2026Quantile]]'s quantile-of-means idea adapts to pessimism for offline contextual bandits, with a Theorem 4.2-style analysis and every step verified mathematically. The answer is yes in the tabular value-based setting, and the direction needs no flip: QoM under-estimates, which is optimism for online losses and pessimism for offline rewards
- **Results.** Theorem 1 is Theorem 4.2's shape with the reward variance at the comparator's actions in place of the worst case, no penalty and no variance estimate; validity needs only non-negative rewards with finite means. Theorem 2 is the coverage form, with a new variance-weighted coverage V\* ≤ C\*/4. Corollaries give a 1/T rate for noiseless optimal actions, S/(eT) for expert data, and a heavy-tailed version
- **Two negative results.** The estimator's typical gap is about 7.8 Bernstein widths at the calibration that makes it valid for every non-negative law (Proposition 2, via the CLT), so the constant is intrinsic, not slack. And QoM-LCB cannot attain Rashidinejad's (C\*−1) rate (Proposition 3, an explicit instance). The same instance defeats every penalty that vanishes like 1/N at a noiseless cell, the Bernstein rules included
- **An erratum in the source.** Cassel & Rosenberg's Lemma 1(2) is false as stated. The corollary under their Lemma 14 drops log(1/δ) from under a square root: 2√(e−2) = 1.695 is their 1.7 and log(1/δ) = 7.66 their 8R. Checked on the rendered PDF page and against Beygelzimer et al.'s original Theorem 1, which has the log inside. For Bernoulli rewards the lemma's event holds with probability tending to δ^0.345. Flagged under Lemma 1 on [[cassel2026Quantile]]; a Bernstein-based replacement with constants 4 and 11/3 is proved on the query page
- Feige's Theorem 1 was checked against the STOC 2004 original. The 1/e improvement is not available: a claimed general proof (arXiv 2508.07316) was withdrawn by its author, and the identically-distributed note (arXiv 2509.19949) is an unrefereed preprint. So 1/13 stands and 1/e appears only as an open problem
- **`scripts/check_math.py` fixed twice.** It now skips HTML comments, after a `$$` inside a comment shifted every later pairing. And it labels two further GitHub hazards found while writing the page: an opening `$` glued to a letter, digit or dash (`Bernoulli$(1/2)$`, `horizon-$H$`), and a span ending in `)` followed by `)`. The page was written GitHub-safe in the current `$`/`$$` convention and checks at 0 of 480 spans wrong on GitHub's renderer, with 0 MathJax errors, so it converts mechanically whichever convention is chosen

## [2026-09-22] update | Math moved to GitHub's native syntax; the display-math diagnosis corrected

- **Correcting the `fix` entry above.** It said a `$$` opener glued to its paragraph "takes every later display block in the same file down with it". It does not: a glued delimiter breaks only its own block. `contextual-bandits-offline.md` rendered 0/15 because all 15 openers were glued, and [[cassel2026Quantile]]'s Algorithm 1 failed because GitHub read the `](s,a)` inside it as a link. Blank lines could never have been enough, since GitHub runs markdown over the *content* of bare-dollar math as well: `\{ \} \, \; \! \|` lose their backslash, paired `*` become emphasis, `[..](..)` becomes a link
- Measured on the 61 pages with math before today's query page: 234 of 405 display blocks and 736 of 3705 inline spans reached MathJax altered or not at all. At Tuan's choice all math now uses GitHub's native syntax, display blocks in ```` ```math ```` fences and inline spans as `` $`...`$ ``, which GitHub passes through untouched. This reverses the entry above's rejection of fences: Obsidian, with no math plugin, now shows the math as code, a price Tuan accepted
- **62 pages converted mechanically and checked three ways.** 420 display blocks and 4195 inline spans; five table norms `\|` → `\Vert`, since GFM reads `\|` in a table row as a cell pipe even inside code; three italic runs closed before a span instead of after it; one inline span that was broken over two lines joined. A diff check confirmed the math, in order, and the text around it unchanged apart from whitespace and those edits; GitHub's renderer then showed 0 of 4615 spans wrong, and MathJax 3.2.1 typeset all 4615 without error
- **`CLAUDE.md`'s Formatting Rules rewritten**: the native syntax with an example, why bare dollars are out, the three things that still break native math (a `|` in a table row, emphasis closed against a span, a backtick inside math), and a rule to run `scripts/check_math.py` on every page with math. Lint step 6 runs the checker in place of the blank-line audit. The schema's own examples were converted, and the index rule's example corrected: it is 47 characters of source in the new syntax, and had been 45, not 40
- **`scripts/check_math.py` now reports bare-dollar spans even where GitHub renders them**, and any `$` that belongs to no span, so a page cannot slip back to the old syntax unnoticed; it also names the table-pipe hazard. This log keeps its bare dollars in older entries: it is append-only, and the checker skips it
