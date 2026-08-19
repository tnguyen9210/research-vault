---
title: "Budget- and Cost-Constrained Bandits"
tags: [budget-limited-mab, bandits, regret, best-arm-identification]
---

# Budget- and Cost-Constrained Bandits

## Overview

A family of bandit problems where pulling arms is *costly* and resources are *constrained*, so the agent must reason about reward *per unit cost* rather than reward alone. Two distinct objectives recur:

1. **Cumulative-reward / regret under a shared budget** — a single budget $B$ caps total spend across exploration *and* exploitation; maximize total reward (equivalently, minimize regret vs. the knapsack optimum). The [[budget-limited-mab]] is the canonical model; the right statistic is reward density $\mu_i/c_i$.
2. **Pure-exploration / fixed-confidence cost minimization** — identify the best arm with confidence $1-\delta$ while minimizing *cumulative cost*. This is cost-aware [[best-arm-identification]].

These differ fundamentally: (1) trades off cheap-but-good vs. expensive-but-best arms for accumulated reward and admits $\Theta(\ln B)$ regret; (2) is a stopping-time problem where cost reshapes the optimal sampling proportions (e.g. $\propto \sqrt{c_a}$ in [[Kanarios2024Cost]]).

## Key Papers

- [[TranThanh2010Epsilon]] (2010) — **introduces the budget-limited MAB**, its unbounded-knapsack optimum, and the reward-density statistic; solves it with an [[epsilon-first]] policy (uniform exploration on $\varepsilon B$, density-ordered greedy on the rest) and proves the first loss bound. Optimizing that bound over $\varepsilon$ gives $O(B^{2/3})$
- [[TranThanh2012Knapsack]] (2012) — [[kube]] / fractional KUBE; first $O(\ln B)$ regret with matching lower bound, by interleaving exploration into the knapsack objective instead of giving it a separate phase
- [[Kanarios2024Cost]] (2024) — cost-aware BAI (CABAI); fixed-confidence cost minimization, optimal proportions scale with $\sqrt{c_a}$

## Open Problems

- **Tight constants** for budget-limited regret — current $O(\ln B)$ bounds have loose $1/d_{\min}^2$ and $(c_{\max}/c_{\min})^2$ factors that don't predict empirical orderings.
- **Non-stationary costs/rewards** — most analyses assume static $\mu_i$; drift breaks the convergence-to-knapsack argument.
- **Multiple resource constraints** (bandits-with-knapsacks) — generalizing a single scalar budget to several simultaneous resource caps.
- **Stochastic / unknown costs** — both threads largely assume known per-arm costs.

## Related Topics

- [[best-arm-identification]] — the pure-exploration regime that the cost-aware BAI thread specializes
