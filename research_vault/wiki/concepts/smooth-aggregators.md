---
title: "Smooth Aggregators in Place of Max"
tags: [reinforcement-learning, value-estimation]
---

# Smooth Aggregators in Place of Max

## Overview

A cross-cutting pattern, now instantiated three independent times in the vault: replace greedy maximization in a value update with a **smooth one-parameter family that interpolates between averaging and maximization**. All three instances are motivated by the same failure mode — under estimation noise, the hard max selects *for* upward error (see [[overestimation-bias]]) — and all three select their parameter empirically, with no instance-dependent rule.

| Instance | Aggregator | Parameter | Where |
|---|---|---|---|
| Power-mean MCTS backups ([[mcts-power-mean]]) | power mean | $p$ (avg $p{=}1$ → max $p{\to}\infty$) | [[Dam2024Power]] |
| Softmax DQN targets ([[softmax-bellman-operator]]) | softmax / log-sum-exp | $\tau$ (avg $\tau{\to}0$ → max $\tau{\to}\infty$) | [[Song2019Revisiting]] |
| Upper expectiles in offline RL ([[expectile-regression]]) | $\tau$-expectile | $\tau$ (mean $\tau{=}0.5$ → max $\tau{\to}1$) | [[Kostrikov2022Offline]] |

Only [[Song2019Revisiting]] supplies a quantitative finite-parameter bound on the gap to the max (plus exponential convergence in $\tau$); [[Dam2024Power]] proves $\mathcal{O}(n^{-1/2})$ convergence for the MCTS estimator ($p=2$ consistently best empirically); [[Kostrikov2022Offline]] gives only the asymptotic limit $\tau \to 1$ recovering the support-constrained optimum.

## Key Papers

- [[Dam2024Power]] (2024) — power mean value backup fixes UCT's flawed logarithmic bonus in stochastic MDPs
- [[Song2019Revisiting]] (2019) — softmax Bellman operator reduces [[overestimation-bias]] and gradient noise in DQN/DDQN; the only instance with a finite-parameter gap bound
- [[Kostrikov2022Offline]] (2022) — IQL's upper-expectile value fit enables in-sample maximization for offline RL

## Open Problems

- **A unified analysis.** Three parameterized interpolations averaging→max, all motivated by estimation error under noise — a common framework quantifying the bias–variance trade-off across them looks tractable and is not in the literature.
- **Instance-dependent parameter selection.** All three tune empirically: is there a principled rule for $p$ (as a function of environment stochasticity), a cooling schedule for the softmax $\tau$, or a data-dependent expectile level?
- Does the benefit of smooth targets extend to modern value-based stacks (Rainbow, SAC-style critics)?

## Related Topics

- [[mcts]] — the power-mean instance lives in MCTS backups
- [[offline-reinforcement-learning]] — the expectile instance is IQL's core mechanism; the softmax instance targets the same overestimation pathology
