---
title: "Power Mean MCTS (Stochastic-Power-UCT)"
tags: [mcts, planning, power-mean, convergence]
introduced_by: [[Dam2024Power]]
---

# Power Mean MCTS (Stochastic-Power-UCT)

**Definition:** An MCTS algorithm using the power mean $\hat{V}_t(s) = \left(\sum_a \frac{T_{s,a}}{t} \hat{Q}_a^p\right)^{1/p}$ as the value backup operator, paired with a polynomial exploration bonus, achieving $\mathcal{O}(n^{-1/2})$ convergence in stochastic MDPs.

## Intuition
The standard average mean underestimates the optimal value (weighted sum biased low), while the max overestimates it. The power mean with $p > 1$ lies between these extremes and provides a balanced estimator. As $p$ increases the backup becomes more optimistic; $p = 2$ is empirically optimal across a wide range of environments.

## Formal Description
**Power mean value backup:**

$$\hat{V}_t(s_h) = \left( \sum_{a \in \mathcal{A}_{s_h}} \frac{T_{s_h,a}(t)}{t} \left(\hat{Q}_{T_{s_h,a}(t)}(s_h, a)\right)^p \right)^{1/p}$$

**Optimal exploration bonus** (Remark 2 of [[Dam2024Power]]):

$$B_h(n, s, a) = C \frac{n^{1/4}}{T_{s,a}(n)^{1/2}}$$

obtained by setting $\alpha_i/\beta_i = 1/2$, $b_i/\beta_i = 1/4$ in the general polynomial schedule.

**Convergence (Theorem 3):**

$$\left|\mathbb{E}[\hat{V}_n(s_0)] - \tilde{V}(s_0)\right| \leq \mathcal{O}(n^{-1/2})$$

Special cases: $p = 1$ recovers Fixed-Depth-MCTS (average mean); $p \to \infty$ approaches pure max backup.

## Key Papers
- [[Dam2024Power]] — introduces Stochastic-Power-UCT with complete convergence proof for stochastic MDPs

## Variants & Related Concepts
- [[monte-carlo-tree-search]] — the broader framework
- **Fixed-Depth-MCTS** (Shah et al. 2022) — special case $p = 1$; deterministic environments only
- **Power-UCT** (Dam et al. 2019) — predecessor without stochastic convergence guarantee
- [[softmax-bellman-operator]] — the same smooth-aggregator-in-place-of-max idea in deep Q-learning ([[Song2019Revisiting]])
- [[expectile-regression]] — and again in offline RL ([[Kostrikov2022Offline]]), where the smoothing is what makes in-sample maximization possible

## Current State
Optimal $p$ selection remains open; $p = 2$ is a robust empirical default. Extension to adversarial MDPs and deep learning integration are open problems.
