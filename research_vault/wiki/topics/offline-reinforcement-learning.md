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

**A third axis: what is actually provable.** Practice and theory in offline RL have largely diverged. The methods people run (IQL, CQL, TD3+BC) carry no instance-dependent guarantees; the methods with sharp guarantees (PEVI, PFQL) are not run. [[Yin2023Offline]] represents the strongest current theory — [[instance-dependent-bounds]] under nonlinear ([[differentiable-function-approximation]]) models, via pessimism — while [[Kostrikov2022Offline]] represents the strongest current practice, with only an asymptotic guarantee. The two make a useful pair: they answer the same question (how to avoid trusting extrapolated values) with opposite strategies, penalize-the-uncertainty versus never-query-it.

Nearly every algorithm here is a variant of one template: [[fitted-q-iteration]] — relabel the fixed batch with Bellman targets, refit by least squares, repeat. Reading the field as "FQI with one of its three steps modified" (target, regression, or policy extraction) makes the algorithm zoo tractable; the `concepts/fqi/` cluster collects the instances.

The vault covers this area through two papers, one empirical and one theoretical; the deep-RL machinery they build on is covered by [[deep-q-network]] and [[Song2019Revisiting]].

## Key Papers

- **[[Kostrikov2022Offline]]** (ICLR 2022) — IQL: multi-step dynamic programming with strictly in-sample value evaluation, via upper-[[expectile-regression]] of $Q(s,\cdot)$ plus AWR policy extraction. $\tau \to 1$ recovers the support-constrained optimum (Thm 3); $\tau$ interpolates SARSA to Q-learning. State of the art on D4RL antmaze (378.0 total vs. 303.6 for CQL) at roughly $4\times$ lower compute, and finetunes well online ($370.1 \to 473.7$).

- **[[Yin2023Offline]]** (ICLR 2023) — PFQL: pessimistic fitted Q-iteration over [[differentiable-function-approximation]]. First instance-dependent bound for offline RL under a nonlinear class, governed by the Fisher-information-style quantity $\sum_h\mathbb{E}_{\pi^*}[\sqrt{\nabla_\theta f^\top\Sigma_h^{\star-1}\nabla_\theta f}]$; the variance-aware VAFQL variant saves a factor $H$ and is minimax-optimal up to $\sqrt{d}$ (Thm 4.2). Strictly generalizes the tabular and linear-MDP results. Caveats: uniform coverage excludes overparameterized (non-identifiable) models, the rate carries $d$ = parameter count rather than $\sqrt{d}$, and "efficient" means statistically, not computationally.

Adjacent, not offline but structurally relevant:

- **[[Song2019Revisiting]]** (ICML 2019) — softmax in place of max in the online DQN target; the closest analogue to IQL's expectile, and the one with an explicit finite-temperature bound.
- **[[Foster2025Foundation]]** (2025) — formalizes [[coverage-coefficient]] as the quantity governing what a reference distribution makes learnable; the density-aware version of the support condition offline RL relies on.

## Open Problems

- **Finite-$\tau$ / finite-sample guarantees for in-sample methods.** IQL's Theorem 3 is asymptotic in $\tau$ and assumes exact solutions. No bound exists on $\max_{a:\pi_\beta(a|s)>0} Q^*(s,a) - V_\tau(s)$ at the $\tau \in \{0.7, 0.9\}$ actually used. [[Yin2023Offline]] shows what such a bound looks like for the pessimism family — transporting it to the in-sample family is the obvious target.
- **Theory for the algorithms people run.** PFQL has the guarantee; IQL has the benchmark numbers. Neither has both. An instance-dependent bound for an in-sample method, or a competitive empirical evaluation of PFQL/VAFQL, would close the gap from either side.
- **Coverage conditions that survive overparameterization.** Uniform coverage requires parameter identifiability, which neural networks violate structurally (permutation and scaling symmetries). A quotient formulation — coverage on function space or parameter equivalence classes — is missing and looks tractable.
- **Is $d$ or $\sqrt{d}$ right for nonlinear classes?** Thm 4.2 leaves a $\sqrt{d}$ gap; whether the covering argument is loose or nonlinearity genuinely costs $\sqrt{d}$ is open.
- **Does the instance measure predict empirical difficulty?** Nobody has checked whether $\sum_h\mathbb{E}_{\pi^*}[\|\nabla_\theta f\|_{\Sigma_h^{\star-1}}]$ tracks observed hardness on D4RL. Cheap experiment, would directly link the two papers on this page.
- **Instance-dependent characterization of dataset quality.** Support ($\pi_\beta(a|s) > 0$) is a binary condition that ignores how much mass sits near the maximizing action. A [[coverage-coefficient]]-style density-weighted quantity should govern both the achievable value and the required $\tau$ — this would explain why $\tau = 0.9$ is necessary on antmaze but not on locomotion.
- **Hyperparameter selection without online evaluation.** $\tau$ and $\beta$ are tuned per domain against online returns, which offline RL is by definition not supposed to have. Offline model selection remains largely unsolved.
- **Unified theory of smooth maximization operators.** Expectile ([[Kostrikov2022Offline]]), softmax ([[Song2019Revisiting]]), and power mean ([[Dam2024Power]]) all replace max with a tunable smooth aggregator for closely related reasons. No common analysis exists.
- **When is stitching actually needed?** The single-step/multi-step gap is enormous on antmaze and negligible on locomotion. A dataset statistic that predicts which regime you are in would be more useful than either method.

## Related Topics

- [[monte-carlo-tree-search]] — planning-time analogue of the same max-under-uncertainty problem; [[Dam2024Power]] uses a smooth backup for the same reason
- [[test-time-scaling]] — inference-time compute as an alternative to better training-time policies; [[Foster2025Foundation]] gives the coverage-theoretic argument for that trade
- [[offline-oracle-efficient-bandits]] — the bandit-theoretic counterpart: what can be learned given only an offline regression oracle over a fixed distribution
- [[instance-dependent-bounds]] — the guarantee type that separates [[Yin2023Offline]] from the worst-case GFA literature, and that [[Kostrikov2022Offline]] lacks
