---
title: "Best Arm Identification"
tags: [bandits, pure-exploration, fixed-confidence]
introduced_by: [[Kanarios2024Cost]]
---

# Best Arm Identification

**Definition:** Fixed-confidence pure exploration problem: given $K$-armed stochastic bandit, find the arm with highest expected reward with probability $\geq 1 - \delta$ while minimizing expected sample complexity.

## Intuition
Unlike regret minimization (balance exploration and exploitation throughout), BAI dedicates all pulls to exploration and stops when sufficiently confident. Models the testing phase: run sequential trials until certain which option is best, then commit to it.

## Formal Description
$K$ arms with reward distributions from natural exponential family. Policy $\pi$ = (sampling rule, stopping time $\tau_\delta$, decision rule $\hat{a}$). $\delta$-PAC requires $P(\hat{a} \neq a^*(\mu)) \leq \delta$ and $\tau_\delta < \infty$ a.s.

**Lower bound (Kaufmann et al. 2016):**

$$
\mathbb{E}[\tau_\delta] \geq T^*(\mu) \log\frac{1}{\delta}, \qquad
T^*(\mu)^{-1} = \sup_{w \in \Sigma_K}\ \inf_{\lambda:\, a^*(\lambda) \neq a^*(\mu)} \sum_a w_a\, d(\mu_a, \lambda_a)
$$

Optimal: TAS (Track-And-Stop, Garivier & Kaufmann 2016) achieves the lower bound asymptotically by tracking empirical proportions $w^*(\hat{\mu})$.

**Chernoff GLR statistic** used for stopping:

$$Z_{a,b}(t) = N_a(t)\,d(\hat{\mu}_a,\, \hat{\mu}_{a,b}) + N_b(t)\,d(\hat{\mu}_b,\, \hat{\mu}_{a,b})$$

where $\hat{\mu}_{a,b}$ is the pull-weighted mixture mean.

## Key Papers
- [[Kanarios2024Cost]] — extends BAI with per-arm costs (CABAI); optimal proportions shift from $w_a \propto 1/\Delta_a^2$ to $w_a \propto \sqrt{c_a}/\Delta_a^2$
- [[Lardy2025Constrained]] — CBAI: each arm has joint (reward, cost) distribution; goal is best-reward arm with mean cost $\leq \gamma$; handles dependent distributions; asymptotically optimal TaS

## Variants & Related Concepts
- [[cabai]] — cost-aware BAI; minimize cumulative cost not rounds; $w^*_a \propto \sqrt{c_a}$
- [[constrained-bai]] — cost-threshold BAI; find best arm with mean cost $\leq \gamma$; handles dependent reward-cost
- **BAI with safety constraints** (Wang et al. 2022) — agent constrained; distinct from CABAI/CBAI where any arm can be pulled
- **Multi-fidelity BAI** — costs are known a priori and controllable; differs from CABAI's random unknown costs

## Current State
Theoretical foundation is mature: matching lower/upper bounds via TAS. Active extensions: cost-awareness ([[cabai]], [[constrained-bai]]), safety constraints, multi-fidelity, and dependent arm distributions. Regret-minimization and BAI are studied as largely separate paradigms.
