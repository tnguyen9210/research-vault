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
