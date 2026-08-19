---
title: "Monte-Carlo Tree Search"
tags: [planning, reinforcement-learning, mcts]
---

# Monte-Carlo Tree Search

## Overview
Monte-Carlo Tree Search (MCTS) is an online planning framework that combines tree search with Monte-Carlo sampling. At each internal node, action selection is treated as a multi-armed bandit problem. The theoretical foundation of the dominant algorithm (UCT) is incomplete for stochastic environments — its logarithmic bonus assumes exponential concentration of regret, but actual concentration is polynomial. Fixing this is an active research area.

The vault currently covers one paper addressing the stochastic setting via power mean value estimation.

## Key Papers

- **[[Dam2024Power]]** (2024) — Stochastic-Power-UCT: uses power mean backup operator + polynomial exploration bonus; proves $\mathcal{O}(n^{-1/2})$ convergence rate for root-node value estimation in stochastic MDPs; $p = 2$ consistently best empirically
- **[[Foster2025Foundation]]** (2025) — MTSS (MultiTurnSpannerSampling): a token-level DP backward induction algorithm for autoregressive LM alignment; achieves exponentially better runtime by replacing sequence-level $C_\text{cov}(\pi^*_\beta)$ with token-level $C_\text{cond}(\pi^*_\beta)$; exploits that the KL-regularized Q-function $Q^*_{h,\beta}$ is linear under autoregressive realizability

## Open Problems

- Is $\mathcal{O}(n^{-1/2})$ the minimax-optimal convergence rate for stochastic MCTS?
- Optimal $p$ selection for power mean as a function of environment structure
- Extension of polynomial-bonus MCTS to adversarial MDP settings
- Integration of power mean backups with deep learning value networks (AlphaZero-style)
- Does MTSS's runtime advantage (token-level vs. sequence-level coverage) extend to sub-sequence-level MDPs?

## Related Topics
- [[best-arm-identification]] — BAI theory underlies action selection at each MCTS node; [[Kaufmann-Emilie]] bridges both areas
- [[offline-oracle-efficient-bandits]] — contextual bandits; distinct from MCTS (planning vs. regret minimization)
- [[offline-reinforcement-learning]] — shares the smooth-aggregator-instead-of-max design pattern ([[power-mean-mcts]] and [[expectile-regression]] are the two instances), and the same underlying problem of maximizing over noisy value estimates
