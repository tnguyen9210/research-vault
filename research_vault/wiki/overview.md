# ML/AI Research Overview

*Synthesis of the research area as represented by papers in this vault. Updated incrementally with each ingest.*

**Sources ingested:** 8  
**Last updated:** 2026-05-28

---

## Current State of the Field

The vault covers two complementary areas of bandit research:

**Oracle-efficient contextual bandits (regret minimization):** Reducing contextual bandit learning to offline regression with as few oracle calls as possible. [[SimchiLevi2022Bypassing]] (FALCON, 2022) first achieved $O(\log T)$ offline oracle calls with optimal regret under realizability for discrete actions. [[Qin2026Taming]] (OE2D, 2026) then resolves the general case: $O(\log T)$ calls for arbitrary function classes without realizability, via DOEC and exploitative F-design.

**Offline policy optimization (off-policy learning):** Given a fixed log of $n$ interactions from a behavior policy, find the best policy without further interaction. [[Ryu2025Improved]] (COLT 2025) establishes new state-of-the-art bounds: PUB achieves parameter-free variance-adaptive off-policy selection via betting-based LCBs for unbounded importance-weighted rewards; the freezing score function reduces variance in small-data regimes. Key distinction from the oracle-efficient line: no online loop, no oracle calls — pure offline optimization.

**Cost-aware pure exploration (best arm identification):** Two complementary cost-aware extensions of standard BAI. [[Kanarios2024Cost]] introduces CABAI: minimize cumulative testing cost $\sum_t C_t$ to find the best arm; optimal proportions scale with $\sqrt{c_a}$. [[Lardy2025Constrained]] introduces CBAI (fixed-confidence): find the best-reward arm among those whose *mean cost* is $\leq \gamma$; handles dependent reward-cost; optimal TaS for three model families. [[Yang2025Stochastically]] addresses the fixed-budget regime (BFAI): Thompson sampling with top-two framework (BFAI-TS) achieves asymptotically optimal exponential PFS decay rate $\Gamma_{\beta^*}$ under $m$ simultaneous Gaussian constraints.

**LLM test-time scaling:** [[Li2026Predicting]] introduces tail-guided scaling law prediction and SLG Search, showing adaptive two-stage compute allocation achieves polynomial amplification over Best-of-$N$. Structural connection to BAI: state selection in SLG is a fixed-budget BAI problem.

**Planning / MCTS:** [[Dam2024Power]] opens a fourth research direction — theoretical foundations of MCTS in stochastic environments. Uses power mean value backup to fix UCT's flawed logarithmic bonus, proving $\mathcal{O}(n^{-1/2})$ convergence. Notable cross-vault link: [[Kaufmann-Emilie]] co-authored both this paper and TAS (the BAI baseline in [[Kanarios2024Cost]]).

## Active Research Threads

- **Unifying offline and online oracle design** — [[Qin2026Taming]] establishes the first formal bridge between DOEC (offline) and DEC (online)
- **Offline policy optimization** — [[Ryu2025Improved]] shows parameter-free pessimism (PUB) and aggressive variance reduction (freezing) work; doubly robust bounds and deeper trees are open
- **Complexity measures for exploration** — DOEC, DEC, Eluder dimension; their mutual relationships are active
- **BAI with practical constraints** — [[Kanarios2024Cost]] (cost-minimization) and [[Lardy2025Constrained]] (cost-threshold, dependent distributions) open two complementary directions; safety constraints and multi-fidelity are adjacent
- **Extensions beyond iid bandits** — misspecification, reward corruption, distribution shift handled in [[Qin2026Taming]]; regret-setting CABAI and partial monitoring are open
- **MCTS in stochastic environments** — [[Dam2024Power]] provides first complete convergence theory; optimal $p$ selection, adversarial extension, and deep learning integration are open
- **Test-time compute optimization** — [[Li2026Predicting]] establishes tail-extrapolation scaling law prediction; deeper trees, PRM integration, and non-Gaussian tails are open

## Emerging Consensus

- O(log T) offline oracle calls is achievable for general function classes (resolved by OE2D)
- DEC ≤ DOEC + lower-order terms — offline algorithms are at least as hard as online, not harder
- Optimal CABAI arm proportions scale with √c_a, not c_a — low-cost arms deserve more pulls than a cost-proportional rule suggests
- Simple algorithms (CO, FALCON) often near-match asymptotically optimal algorithms (CTAS, OE2D) in practice

## Active Debates / Open Questions

- Are there information-theoretic lower bounds matching DOEC?
- Can CO's optimality be proved for $K > 2$ arms?
- Can OE2D extend to RLHF and partial monitoring?
- Is there a cost-aware regret minimization analogue of CABAI?
- Can PUB's learning bound (Thm 4.3 in [[Ryu2025Improved]]) be tightened to match its selection bound — scaling with $\sqrt{\bar{\mathbb{V}}(\pi^*)}$ rather than raw second moment?

## Suggested Next Sources

- **E2D** (Foster et al. 2021a) — online-oracle framework; understanding DEC deeply complements DOEC
- **TAS** (Garivier & Kaufmann 2016) — standard BAI algorithm; direct predecessor of CTAS in [[Kanarios2024Cost]]
- **Foster & Krishnamurthy 2021** — first-order offline-oracle-efficient bandits (open problem for OE2D)
- **SquareCB** (Foster & Rakhlin 2020) — online-oracle optimal algorithm that preceded [[SimchiLevi2022Bypassing]]
