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
