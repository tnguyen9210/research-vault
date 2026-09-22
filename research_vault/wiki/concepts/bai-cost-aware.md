---
title: "Cost-Aware Best Arm Identification"
aliases: [CABAI, cost-aware BAI, cabai, best-arm-identification-cost-aware, cost-aware-bai]
tags: [bandits, pure-exploration, cost-aware, cabai]
introduced_by: [[Kanarios2024Cost]]
---

# Cost-Aware Best Arm Identification

**Definition:** [[bai]] in which each arm $a$ carries a cost as well as a reward, so that the sampling rule must account for what a pull *costs* and not only what it reveals. In the central formulation, CABAI, arm $a$ has a reward distribution $\nu_{\mu_a}$ and a cost distribution $\nu_{c_a}$, and the goal is to identify the best-reward arm with probability $\ge 1-\delta$ at minimum expected cumulative cost $J(\tau_\delta) = \sum_{k=1}^{\tau_\delta} C_k$.

> **Scope.** Cost-aware pure exploration as an area, and CABAI in full. **Left to other pages:** the cost-free paradigm, its transportation-cost lower bound and Track-and-Stop, all on [[bai]]; the constrained formulation CBAI, on [[bai-constrained]]; the cumulative-reward setting where one shared budget caps exploration *and* exploitation, on [[budget-limited-mab]].

## Intuition

Models the testing/deployment split in product development: during testing each trial incurs a cost — prototype cost, ad platform fee, clinical trial dose — while reward is what matters after deployment. Standard [[bai]] ignores this heterogeneity and counts every pull the same. Once pulls are priced, the question changes from "how many samples?" to "how much money?", and the optimal allocation changes with it.

## Formal Description

Instance $(\mu, c)$: $\mu = (\mu_1, \ldots, \mu_K)$ reward means, $c = (c_1, \ldots, c_K)$ cost means. Arms from a natural exponential family (Assumption 1); costs bounded positive (Assumption 2). At round $t$, observe $(R_t, C_t) \sim \nu_{\mu_{A_t}} \times \nu_{c_{A_t}}$.

**Lower bound (Theorem 1 of [[Kanarios2024Cost]]):**

$$
\mathbb{E}[J(\tau_\delta)] \geq T^*(\mu)\log\frac{1}{\delta}, \qquad
T^*(\mu)^{-1} = \sup_{w \in \Sigma_K}\ \inf_{\lambda:\, a^*(\lambda) \neq a^*(\mu)} \sum_a \frac{w_a}{c_a}\,d(\mu_a, \lambda_a)
$$

**2-armed Gaussian (Corollary 1):**

$$\mathbb{E}[J(\tau_\delta)] \geq \frac{2(\sqrt{c_1} + \sqrt{c_2})^2}{(\mu_1 - \mu_2)^2}\,\log\frac{1}{\delta}$$

The structural consequence is the one to remember: the optimal cost proportion is $w^*_a \propto \sqrt{c_a}$. Low-cost arms should be sampled *more*, scaling as the square root of cost, not linearly in it.

## Literature Survey

The vault's most developed pure-exploration thread. What organizes it is not the algorithms — all three papers inherit the same machinery, transportation-cost lower bounds and Track-and-Stop — but the **role cost plays in the problem statement**. Each role reshapes the optimal allocation differently, and the three are not special cases of one another.

### Cost as the objective

The allocation is chosen to minimize what identification *costs*. [[Kanarios2024Cost]] introduces CABAI and proves the $T^*(\mu)$ lower bound above, whose solution gives the $\sqrt{c_a}$ proportions. Two algorithms realize it: **CTAS** (Algorithm 1) tracks $w^*(\hat\mu)$ by largest-deficit sampling with forced exploration and a Chernoff stopping rule, and is asymptotically optimal (Theorem 2); **CO** (Algorithm 2) drops the plug-in optimization for the model-free rule $\arg\min_a \sqrt{c_a}\,N_a(t)$ with elimination-based stopping, is optimal for the 2-arm Gaussian case (Theorem 3), and runs roughly 20–50$\times$ faster. That a rule this simple is near-optimal empirically is the paper's practical claim; that it is *provably* optimal only for $K=2$ is its main gap.

### Cost as a feasibility constraint

Here cost does not price the search — it restricts which arm counts as the answer. [[Lardy2025Constrained]] introduces CBAI in the fixed-confidence regime: each arm has a joint (reward, cost) distribution, and the target is the best-reward arm whose *mean cost* satisfies $\mathbb{E}[C_k] \le \gamma$. Reward and cost may be dependent, which CABAI assumes away. The characteristic time $T^*$ comes from a transportation-cost interface, and asymptotically optimal Track-and-Stop variants are given for Gaussian arms (fixed and unknown covariance) and for non-parametric arms on $[0,1]^2$. The answer set gains an element — all arms may be infeasible — which has no counterpart in CABAI.

The two are worth holding side by side, since both are described as "cost-aware BAI" and they are not the same problem:

| | CABAI ([[Kanarios2024Cost]]) | CBAI ([[Lardy2025Constrained]]) |
|---|---|---|
| **Cost role** | Optimization metric: minimize $\sum_t C_t$ | Constraint: find an arm with $\mathbb{E}[C_k] \leq \gamma$ |
| **Optimal allocation** | $w^*_a \propto \sqrt{c_a}$ (square-root rule) | Via the $c_1$/$c_2$ transportation interface |
| **Answer set** | Always one of the $K$ arms | $K$ arms $\cup$ `None` (all infeasible) |
| **Cost–reward dependence** | Independent by assumption | Explicitly modeled |

### Cost under a fixed budget, with several constraints

The third role fixes the spend in advance and asks for the best answer within it, which turns the problem from a stopping-time question into an allocation question. [[Yang2025Stochastically]] studies fixed-budget BAI with $K$ arms and $m$ *stochastic* constraints, and gives BFAI-TS: Thompson sampling inside a top-two framework parameterized by $\beta$, with an asymptotically optimal exponential convergence rate $\Gamma_{\beta^*}$ for the probability of false selection. This is the only one of the three where the horizon is known and the confidence is the output rather than the input.

### What the three share, and where it stops

All three reuse the cost-free paradigm's machinery — a transportation-cost lower bound, then a tracking or top-two rule that drives the empirical allocation to the bound's optimizer. What none of them has is the non-asymptotic side: every optimality claim here is as $\delta \to 0$ or as the budget grows, and no finite-sample cost bound exists for any of the three formulations.

## Variants

- [[bai]] — the cost-free paradigm; CABAI reduces to it when all $c_a$ are equal
- [[bai-constrained]] — cost as a feasibility constraint rather than the objective; see the table above
- **Fixed-budget cost-aware BAI** — [[Yang2025Stochastically]]'s regime, where the spend is fixed and the error probability is what is optimized
- **BAI with safety constraints** — the constraint restricts which arms may be *pulled* rather than which may be returned, so exploration itself is limited

## Related Concepts

- [[budget-limited-mab]] — the cumulative-reward counterpart: one shared budget over exploration *and* exploitation, reward density $\mu_i/c_i$, and $\Theta(\ln B)$ regret instead of a stopping time
- [[instance-dependent-bounds]] — the guarantee type every result here is stated in: the rate is a functional of the specific instance $(\mu, c)$, not a worst case

## Current State and Open Problems

The three cost roles are each resolved asymptotically, and the $\sqrt{c_a}$ rule is the area's one memorable structural fact. The gaps are concentrated in the same two places for all three: finite $K$, and finite samples.

- **CO beyond two arms.** Its asymptotic optimality is proved only for $K = 2$ Gaussian arms; whether it extends to general $K$ or to general exponential families is open, and is the most direct question the area poses.
- **Non-asymptotic sample complexity.** No finite-$\delta$ cost bound exists for CABAI, CBAI or BFAI — every optimality claim is a limit.
- **Cost-aware *regret* minimization.** All three formulations stop; none addresses the non-stopping setting where cost is paid indefinitely. [[budget-limited-mab]] is the nearest cumulative-reward analogue, but under a shared budget rather than per-pull pricing.
- **ETC analogues for CABAI.** Explore-then-commit variants are absent, which is notable given how well the model-free CO rule performs.
- **Fixed-budget CBAI, and multiple constraints.** CBAI exists only in fixed-confidence form; and handling several cost constraints currently costs time exponential in their number.

## Provenance

*Sourced.* Everything attributed to [[Kanarios2024Cost]], [[Lardy2025Constrained]] and [[Yang2025Stochastically]] comes from their paper pages, written against the PDFs at ingest.

*This page's judgment, not a citation.* The organization of the survey by cost's *role* rather than by chronology or algorithm, the claim that the three formulations are not special cases of one another, and the observation that the missing non-asymptotic theory is common to all three.

*Merged.* This page absorbed the topic page `cost-aware-bai` on 2026-09-21, whose slug is kept as an alias, together with the cost-aware half of the topic `budget-limited-bandits` (whose slug went to [[budget-limited-mab]], its main subject).
