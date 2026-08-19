---
title: "Offline Reinforcement Learning"
tags: [offline-reinforcement-learning, deep-reinforcement-learning, distributional-shift]
---

# Offline Reinforcement Learning

## Overview

Offline RL learns a policy entirely from a fixed dataset $\mathcal{D}$ collected by some behavior policy $\pi_\beta$, with no further environment interaction. The binding constraint is not optimization but **distributional shift**: improving over $\pi_\beta$ requires knowing the value of actions $\pi_\beta$ did not take, and those values must be extrapolated by a function approximator that has no data to constrain it. Worse, the max in the Bellman target selects *for* the largest extrapolation error, so the bias compounds through the backup (see [[overestimation-bias]]).

The literature splits along two axes.

**How the shift is controlled.** Policy-constraint methods (BCQ, BEAR, AWAC, TD3+BC) restrict deviation from $\pi_\beta$; value-regularization methods (CQL, Fisher-BRC) push down $Q$ on out-of-distribution actions; **in-sample** methods ([[implicit-q-learning]]) never query an unseen action at all. The first two impose an explicit improvement-vs.-safety trade-off; the third dissolves it during value learning, at the cost of targeting only the support-constrained optimum.

**How many DP steps are taken.** *Single-step* methods (Onestep RL, Decision Transformer) fit $Q^{\pi_\beta}$ or clone behavior directly. They are simple and competitive when the dataset already contains near-optimal trajectories, but cannot **stitch** — compose segments of distinct sub-optimal trajectories into a better one. *Multi-step* methods iterate the Bellman backup and can. [[Kostrikov2022Offline]] makes this the sharpest empirical distinction in the area: on D4RL antmaze-medium and antmaze-large, every single-step method scores $\approx 0$ while multi-step methods score 40–70.

The vault currently covers this area through one paper; the deep-RL machinery it builds on is covered by [[deep-q-network]] and [[Song2019Revisiting]].

## Key Papers

- **[[Kostrikov2022Offline]]** (ICLR 2022) — IQL: multi-step dynamic programming with strictly in-sample value evaluation, via upper-[[expectile-regression]] of $Q(s,\cdot)$ plus AWR policy extraction. $\tau \to 1$ recovers the support-constrained optimum (Thm 3); $\tau$ interpolates SARSA to Q-learning. State of the art on D4RL antmaze (378.0 total vs. 303.6 for CQL) at roughly $4\times$ lower compute, and finetunes well online ($370.1 \to 473.7$).

Adjacent, not offline but structurally relevant:

- **[[Song2019Revisiting]]** (ICML 2019) — softmax in place of max in the online DQN target; the closest analogue to IQL's expectile, and the one with an explicit finite-temperature bound.
- **[[Foster2025Foundation]]** (2025) — formalizes [[coverage-coefficient]] as the quantity governing what a reference distribution makes learnable; the density-aware version of the support condition offline RL relies on.

## Open Problems

- **Finite-$\tau$ / finite-sample guarantees for in-sample methods.** IQL's Theorem 3 is asymptotic in $\tau$ and assumes exact solutions. No bound exists on $\max_{a:\pi_\beta(a|s)>0} Q^*(s,a) - V_\tau(s)$ at the $\tau \in \{0.7, 0.9\}$ actually used.
- **Instance-dependent characterization of dataset quality.** Support ($\pi_\beta(a|s) > 0$) is a binary condition that ignores how much mass sits near the maximizing action. A [[coverage-coefficient]]-style density-weighted quantity should govern both the achievable value and the required $\tau$ — this would explain why $\tau = 0.9$ is necessary on antmaze but not on locomotion.
- **Hyperparameter selection without online evaluation.** $\tau$ and $\beta$ are tuned per domain against online returns, which offline RL is by definition not supposed to have. Offline model selection remains largely unsolved.
- **Unified theory of smooth maximization operators.** Expectile ([[Kostrikov2022Offline]]), softmax ([[Song2019Revisiting]]), and power mean ([[Dam2024Power]]) all replace max with a tunable smooth aggregator for closely related reasons. No common analysis exists.
- **When is stitching actually needed?** The single-step/multi-step gap is enormous on antmaze and negligible on locomotion. A dataset statistic that predicts which regime you are in would be more useful than either method.

## Related Topics

- [[monte-carlo-tree-search]] — planning-time analogue of the same max-under-uncertainty problem; [[Dam2024Power]] uses a smooth backup for the same reason
- [[test-time-scaling]] — inference-time compute as an alternative to better training-time policies; [[Foster2025Foundation]] gives the coverage-theoretic argument for that trade
- [[offline-oracle-efficient-bandits]] — the bandit-theoretic counterpart: what can be learned given only an offline regression oracle over a fixed distribution
