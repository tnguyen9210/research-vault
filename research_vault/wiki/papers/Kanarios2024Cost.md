---
title: "Cost Aware Best Arm Identification"
authors: [Kellen Kanarios, Qining Zhang, Lei Ying]
year: 2024
venue: arXiv
arxiv: "2402.16710"
tags: [best-arm-identification, cabai, pure-exploration, bandits, cost-aware]
source: raw/papers/Kanarios2024Cost.pdf
---

# Cost Aware Best Arm Identification

**TL;DR:** Introduces CABAI, a generalization of fixed-confidence BAI where each arm has a cost distribution and the goal is to identify the best-reward arm at minimum cumulative cost; optimal arm proportions scale with $\sqrt{c_a}$ (not $c_a$), invalidating standard BAI algorithms.

## Problem

Standard [[best-arm-identification]] minimizes sample complexity (rounds). Real product pipelines separate testing from deployment — the cost of each prototype trial matters. Standard BAI algorithms like TAS ignore heterogeneous per-arm costs and are provably sub-optimal in this setting. CABAI formalizes: given $K$ arms each with a reward distribution $\nu_{\mu_a}$ and a cost distribution $\nu_{c_a}$, find a $\delta$-PAC algorithm minimizing expected cumulative cost $\mathbb{E}[J(\tau_\delta)] = \mathbb{E}[\sum_{k=1}^{\tau_\delta} C_k]$.

## Method

**Lower bound (Theorem 1):** For any $\delta$-PAC algorithm:

$$
\mathbb{E}[J(\tau_\delta)] \geq T^*(\mu) \log\frac{1}{\delta} + o\!\left(\log\frac{1}{\delta}\right)
$$

$$
T^*(\mu)^{-1} = \sup_{w \in \Sigma_K} \inf_{\lambda:\, a^*(\lambda) \neq a^*(\mu)} \sum_a \frac{w_a}{c_a} d(\mu_a, \lambda_a)
$$

2-armed Gaussian closed form (Corollary 1): lower bound = $\dfrac{2(\sqrt{c_1} + \sqrt{c_2})^2}{(\mu_1 - \mu_2)^2} \log\dfrac{1}{\delta}$.

Key insight: optimal proportions satisfy $w^*_a \propto \sqrt{c_a}$ — low-cost arms should be pulled *more* (square-root, not linear in cost).

**CTAS — Cost-adapted Track And Stop (Algorithm 1):**
- Sampling rule: match empirical cost proportion $\hat{w}_a = \hat{c}_a N_a(t)/J(t)$ to $w^*(\hat{\mu})$ via largest-deficit-first; forced exploration pulls least-pulled arm when any $N_a < \sqrt{t}$
- Stopping rule: Chernoff GLR $Z(t) = \max_a \min_{b \neq a} Z_{a,b}(t) > \log(Bt^\alpha/\delta)$
- Asymptotically optimal: $\limsup_{\delta \to 0} \mathbb{E}[J(\tau_\delta)] / \log(1/\delta) \leq \alpha T^*(\mu)$ (Theorem 2)
- Bottleneck: bilevel optimization (ComputeProportions) at every step — ~1700–4600s over 1000 trajectories

**CO — Chernoff-Overlap (Algorithm 2):**
- Sampling rule: pull $a_t = \arg\min_{a \in \mathcal{R}} \sqrt{c_a} \cdot N_a(t)$ (square-root rule; no $w^*$ computation)
- Stopping rule: eliminate arm $a$ when $Z_{a^*(t),a}(t) > \log(Bt^\alpha/\delta)$; stop when one arm remains
- Proved asymptotically cost-optimal for 2-armed Gaussian (Theorem 3); empirically near-optimal for $K > 2$ and Bernoulli/Poisson families
- ~20–50x faster than CTAS (58–96s vs. 1700–4600s)

## Results

| Algorithm | Asymp. optimal? | Gaussian (s) | Bernoulli (s) | Poisson (s) |
|-----------|----------------|-------------|---------------|-------------|
| CO | 2-arm Gaussian | 85 | 58 | 96 |
| CTAS | Yes | 1712 | 1995 | 3260 |
| TAS | No (ignores cost) | 2410 | 2780 | 4633 |
| d-LUCB | No | 82 | 60 | 101 |

*1000 trajectories, $\mu=[1.5,1,0.5]$, $c=[1,0.1,0.01]$, $\delta=10^{-6}$*

CTAS optimal proportions $w^*=(0.23, 0.72, 0.05)$ vs. TAS $(0.46, 0.46, 0.08)$: cost-awareness heavily shifts pulls toward cheap arms.

## Strengths & Limitations

**Strengths:** Clean lower bound derivation; CTAS is the first asymptotically optimal CABAI algorithm; $\sqrt{c_a}$ insight is non-trivial and experimentally validated; CO avoids bilevel optimization entirely while remaining near-optimal in practice.

**Limitations:** CTAS's bilevel optimization per step is expensive in practice. CO's optimality gap for $K > 2$ is open. Requires exponential family assumption. No regret-minimization analogue.

## Connections

- [[best-arm-identification]] — CABAI extends fixed-confidence BAI with heterogeneous per-arm costs
- [[cabai]] — the novel problem formulation introduced here
- [[Kanarios-Kellen]] — first author
- [[Zhang-Qining]] — second author (U. Michigan; distinct from advisor [[Zhang-Chicheng]])
- [[Ying-Lei]] — third/senior author
- **Generalizes:** TAS (Garivier & Kaufmann 2016) — standard BAI baseline, sub-optimal here
- **Related:** BAI with safety constraints (Wang et al. 2022) — agent is free here; constraint is cost not safety

## Open Questions

- Can CO's asymptotic optimality be proved for $K > 2$ or general exponential families?
- Can CABAI be extended to the regret minimization setting with costs?
- Is there an ETC variant with carefully chosen costs?
- First-order CABAI where cost scales with $\Delta$ not $\Delta^2$?
