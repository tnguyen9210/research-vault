---
title: "Cost Aware Best Arm Identification (CABAI)"
tags: [bandits, pure-exploration, cost-aware, cabai]
introduced_by: [[Kanarios2024Cost]]
---

# Cost Aware Best Arm Identification (CABAI)

**Definition:** Pure exploration problem where each arm $a$ has both a reward distribution $\nu_{\mu_a}$ and a cost distribution $\nu_{c_a}$; goal is to identify the best-reward arm with probability $\geq 1 - \delta$ at minimum expected cumulative cost $J(\tau_\delta) = \sum_{k=1}^{\tau_\delta} C_k$.

## Intuition
Models the testing/deployment split in product development: during testing, each trial incurs a cost (prototype cost, ad platform fee, clinical trial dose), while reward is what matters post-deployment. Standard [[best-arm-identification]] ignores this cost heterogeneity. CABAI separates the testing objective (cost) from the deployment objective (reward).

## Formal Description
Instance $(\mu, c)$: $\mu = (\mu_1, \ldots, \mu_K)$ reward means, $c = (c_1, \ldots, c_K)$ cost means. Arms from natural exponential family (Assumption 1). Costs bounded positive (Assumption 2). At round $t$, observe $(R_t, C_t) \sim \nu_{\mu_{A_t}} \times \nu_{c_{A_t}}$.

**Lower bound (Theorem 1 of [[Kanarios2024Cost]]):**

$$
\mathbb{E}[J(\tau_\delta)] \geq T^*(\mu)\log\frac{1}{\delta}, \qquad
T^*(\mu)^{-1} = \sup_{w \in \Sigma_K}\ \inf_{\lambda:\, a^*(\lambda) \neq a^*(\mu)} \sum_a \frac{w_a}{c_a}\,d(\mu_a, \lambda_a)
$$

**2-armed Gaussian (Corollary 1):**

$$\mathbb{E}[J(\tau_\delta)] \geq \frac{2(\sqrt{c_1} + \sqrt{c_2})^2}{(\mu_1 - \mu_2)^2}\,\log\frac{1}{\delta}$$

Non-trivial structural result: optimal cost proportion $w^*_a \propto \sqrt{c_a}$ — low-cost arms should be sampled *more*, scaling as square root, not linearly in cost.

**Algorithms:**
- **CTAS** (Algorithm 1): tracks $w^*(\hat{\mu})$ via largest-deficit sampling + forced exploration; Chernoff stopping; asymptotically optimal (Theorem 2)
- **CO** (Algorithm 2): model-free square-root sampling rule $\arg\min \sqrt{c_a}\,N_a(t)$; elimination-based stopping; optimal for 2-arm Gaussian (Theorem 3), ~20–50$\times$ faster than CTAS

## Key Papers
- [[Kanarios2024Cost]] — introduces CABAI, lower bound, CTAS, and CO

## Variants & Related Concepts
- [[best-arm-identification]] — CABAI reduces to standard BAI when all $c_a$ equal
- **BAI with safety constraints** — constraint is on safety, not cost; agent can be restricted from certain arms

## Current State
Open: CO's optimality for $K > 2$ arms; cost-aware regret minimization; ETC variant for CABAI.
