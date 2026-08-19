# ML/AI Research Overview

*Synthesis of the research area as represented by papers in this vault. Updated incrementally with each ingest.*

**Sources ingested:** 12  
**Last updated:** 2026-08-19

---

## Current State of the Field

The vault covers two complementary areas of bandit research:

**Oracle-efficient contextual bandits (regret minimization):** Reducing contextual bandit learning to offline regression with as few oracle calls as possible. [[SimchiLevi2022Bypassing]] (FALCON, 2022) first achieved $O(\log T)$ offline oracle calls with optimal regret under realizability for discrete actions. [[Qin2026Taming]] (OE2D, 2026) then resolves the general case: $O(\log T)$ calls for arbitrary function classes without realizability, via DOEC and exploitative F-design.

**Offline policy optimization (off-policy learning):** Given a fixed log of $n$ interactions from a behavior policy, find the best policy without further interaction. [[Ryu2025Improved]] (COLT 2025) establishes new state-of-the-art bounds: PUB achieves parameter-free variance-adaptive off-policy selection via betting-based LCBs for unbounded importance-weighted rewards; the freezing score function reduces variance in small-data regimes. Key distinction from the oracle-efficient line: no online loop, no oracle calls — pure offline optimization.

**Cost-aware pure exploration (best arm identification):** Two complementary cost-aware extensions of standard BAI. [[Kanarios2024Cost]] introduces CABAI: minimize cumulative testing cost $\sum_t C_t$ to find the best arm; optimal proportions scale with $\sqrt{c_a}$. [[Lardy2025Constrained]] introduces CBAI (fixed-confidence): find the best-reward arm among those whose *mean cost* is $\leq \gamma$; handles dependent reward-cost; optimal TaS for three model families. [[Yang2025Stochastically]] addresses the fixed-budget regime (BFAI): Thompson sampling with top-two framework (BFAI-TS) achieves asymptotically optimal exponential PFS decay rate $\Gamma_{\beta^*}$ under $m$ simultaneous Gaussian constraints.

**Budget-limited cumulative-reward bandits (cost-aware regret):** The cumulative-reward counterpart to the cost-aware BAI thread. [[TranThanh2012Knapsack]] introduces the [[budget-limited-mab]] — a single shared budget $B$ caps both exploration and exploitation, making the full-information optimum an *unbounded knapsack* on reward densities $\mu_i/c_i$. [[kube]] and fractional KUBE are the first algorithms achieving the optimal $O(\ln B)$ regret (with a matching lower bound), beating the $O(B^{2/3})$ $\varepsilon$-first baseline. Fractional KUBE is the budget-limited analogue of UCB. This directly answers the "cost-aware regret-minimization analogue of CABAI" question raised by the BAI thread.

**LLM test-time scaling:** [[Li2026Predicting]] introduces tail-guided scaling law prediction and SLG Search, showing adaptive two-stage compute allocation achieves polynomial amplification over Best-of-$N$. Structural connection to BAI: state selection in SLG is a fixed-budget BAI problem.

**Computational theory of LM alignment:** [[Foster2025Foundation]] opens a new thread — a formal computational framework for online LM alignment via a *sampling oracle*. Central finding: the [[coverage-coefficient]] $C_\text{cov}(\pi^*_\beta)$ is both necessary (Thm 2.1) and sufficient (SpannerSampling, Thm 3.1) for computationally efficient active exploration. Training-time interventions (OnlineDPO, XPO) cannot simultaneously achieve data and computational efficiency under ETH (Thm 4.1). Multi-turn MTSS (Thm 5.1) replaces sequence-level $C_\text{cov}$ with exponentially smaller token-level $C_\text{cond}$ under autoregressive realizability. Theoretical grounding for why inference-time compute (MCTS, BoN) beats training-time-only exploration.

**Planning / MCTS:** [[Dam2024Power]] opens a fourth research direction — theoretical foundations of MCTS in stochastic environments. Uses power mean value backup to fix UCT's flawed logarithmic bonus, proving $\mathcal{O}(n^{-1/2})$ convergence. Notable cross-vault link: [[Kaufmann-Emilie]] co-authored both this paper and TAS (the BAI baseline in [[Kanarios2024Cost]]).

**Value-based deep RL:** [[Song2019Revisiting]] introduces a new thread — replacing the max operator in DQN/DDQN target networks with the [[softmax-bellman-operator]] at inverse temperature $\tau$. Despite being a non-contraction, $\mathcal{T}_\text{soft}$ provably reduces [[overestimation-bias]] (with quantified bounds) and gradient noise, and converges to $\mathcal{T}$ exponentially fast in $\tau$. S-DQN and S-DDQN outperform their max counterparts on Atari, independent of exploration. Notable thematic parallel with [[Dam2024Power]]: both replace greedy max aggregation with a smooth operator to improve value estimation under noise.

**Offline RL:** [[Kostrikov2022Offline]] opens the [[offline-reinforcement-learning]] thread. IQL performs multi-step dynamic programming while evaluating $Q$ *only* at dataset actions, by fitting an upper $\tau$-expectile of $Q(s,\cdot)$ ([[expectile-regression]]) in place of the max; Theorem 3 gives $\lim_{\tau\to 1} V_\tau(s) = \max_{a:\pi_\beta(a|s)>0} Q^*(s,a)$, the optimum constrained to data support. State of the art on D4RL antmaze (378.0 vs. 303.6 for CQL) at $\approx 4\times$ lower compute. The empirical headline is the collapse of single-step methods on stitching tasks ($\approx 0$ on antmaze-medium/large), which is the sharpest evidence in the vault that iterated Bellman backups buy something real.

## Active Research Threads

- **Unifying offline and online oracle design** — [[Qin2026Taming]] establishes the first formal bridge between DOEC (offline) and DEC (online)
- **Offline policy optimization** — [[Ryu2025Improved]] shows parameter-free pessimism (PUB) and aggressive variance reduction (freezing) work; doubly robust bounds and deeper trees are open
- **Complexity measures for exploration** — DOEC, DEC, Eluder dimension; their mutual relationships are active
- **BAI with practical constraints** — [[Kanarios2024Cost]] (cost-minimization) and [[Lardy2025Constrained]] (cost-threshold, dependent distributions) open two complementary directions; safety constraints and multi-fidelity are adjacent
- **Cost/budget-constrained bandits across objectives** — [[budget-limited-bandits]] unifies cumulative-reward ([[TranThanh2012Knapsack]]) and pure-exploration ([[Kanarios2024Cost]]) cost-aware bandits; open frontiers: tight constants, non-stationary costs, multiple resource constraints (bandits-with-knapsacks)
- **Extensions beyond iid bandits** — misspecification, reward corruption, distribution shift handled in [[Qin2026Taming]]; regret-setting CABAI and partial monitoring are open
- **MCTS in stochastic environments** — [[Dam2024Power]] provides first complete convergence theory; optimal $p$ selection, adversarial extension, and deep learning integration are open
- **Test-time compute optimization** — [[Li2026Predicting]] establishes tail-extrapolation scaling law prediction; deeper trees, PRM integration, and non-Gaussian tails are open
- **Computational theory of LM alignment** — [[Foster2025Foundation]] establishes the sampling oracle framework and $C_\text{cov}$ lower bound; extension to nonlinear policies, tighter $T_\text{comp}$ bounds, and estimation of $C_\text{cond}$ from base model are open
- **Smooth Bellman operators in deep RL** — [[Song2019Revisiting]] motivates replacing max with softmax in DQN targets; open: does the benefit extend to Rainbow/SAC/PPO, and is there a principled cooling schedule for $\tau$?
- **Smooth aggregators in place of max (cross-cutting)** — now three independent instances in the vault: power mean in MCTS backups ([[Dam2024Power]], parameter $p$), softmax in DQN targets ([[Song2019Revisiting]], parameter $\tau$), and upper expectiles in offline RL ([[Kostrikov2022Offline]], parameter $\tau$). All three replace greedy maximization with a one-parameter family interpolating averaging-to-maximization, all three are motivated by estimation error under noise, and all three select the parameter empirically with no instance-dependent rule. Only [[Song2019Revisiting]] supplies a quantitative finite-parameter gap bound. A unified analysis looks tractable and is not in the literature.
- **Offline RL with in-sample value learning** — [[Kostrikov2022Offline]] shows multi-step DP is possible without out-of-sample queries; open: a finite-$\tau$ bound on the gap to the support-constrained optimum, and a density-aware replacement for the binary support condition (natural bridge to [[coverage-coefficient]] from [[Foster2025Foundation]])

## Emerging Consensus

- O(log T) offline oracle calls is achievable for general function classes (resolved by OE2D)
- DEC ≤ DOEC + lower-order terms — offline algorithms are at least as hard as online, not harder
- Optimal CABAI arm proportions scale with √c_a, not c_a — low-cost arms deserve more pulls than a cost-proportional rule suggests
- Simple algorithms (CO, FALCON) often near-match asymptotically optimal algorithms (CTAS, OE2D) in practice

## Active Debates / Open Questions

- Are there information-theoretic lower bounds matching DOEC?
- Can CO's optimality be proved for $K > 2$ arms?
- Can OE2D extend to RLHF and partial monitoring?
- ~~Is there a cost-aware regret minimization analogue of CABAI?~~ — yes: the [[budget-limited-mab]] / [[kube]] ([[TranThanh2012Knapsack]]). Remaining: tight constants, non-stationary rewards/costs.
- Can PUB's learning bound (Thm 4.3 in [[Ryu2025Improved]]) be tightened to match its selection bound — scaling with $\sqrt{\bar{\mathbb{V}}(\pi^*)}$ rather than raw second moment?

## Suggested Next Sources

- **E2D** (Foster et al. 2021a) — online-oracle framework; understanding DEC deeply complements DOEC
- **TAS** (Garivier & Kaufmann 2016) — standard BAI algorithm; direct predecessor of CTAS in [[Kanarios2024Cost]]
- **Foster & Krishnamurthy 2021** — first-order offline-oracle-efficient bandits (open problem for OE2D)
- **SquareCB** (Foster & Rakhlin 2020) — online-oracle optimal algorithm that preceded [[SimchiLevi2022Bypassing]]
