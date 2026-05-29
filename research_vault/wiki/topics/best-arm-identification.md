---
title: "Best Arm Identification"
tags: [bandits, pure-exploration]
---

# Best Arm Identification

## Overview
Best arm identification (BAI) is the fixed-confidence pure exploration paradigm: run sequential trials on K arms and stop when confident the best arm has been identified (with probability ≥ 1−δ), at minimum expected sample cost. Unlike regret minimization, all pulls serve exploration and the algorithm terminates. The theoretical foundation is mature: Kaufmann et al. (2016) established matching lower bounds, and TAS achieves them asymptotically.

The vault currently covers a cost-aware generalization ([[cabai]]) where each arm has a heterogeneous testing cost, motivating a fundamentally different optimal allocation rule (√c_a vs. uniform).

## Key Papers

- **[[Kanarios2024Cost]]** (2024) — introduces CABAI: BAI where each arm has a cost distribution; lower bound shows optimal arm proportions scale with $\sqrt{c_a}$; proposes CTAS (asymptotically optimal via bilevel optimization) and CO (low-complexity, model-free, near-optimal empirically)
- **[[Lardy2025Constrained]]** (2025) — introduces CBAI (fixed-confidence): each arm has a joint (reward, cost) distribution; goal is best-reward arm with mean cost $\leq \gamma$; handles dependent reward-cost; derives $T^*$ via transportation cost interface; asymptotically optimal TaS for Gaussian (fixed/unknown covariance) and non-parametric $[0,1]^2$ arms
- **[[Yang2025Stochastically]]** (2025) — introduces BFAI-TS (fixed-budget): $k$ arms, $m$ stochastic constraints; Thompson sampling with top-two framework parameterized by $\beta$; asymptotically optimal exponential convergence rate $\Gamma_{\beta^*}$ for PFS

## The CABAI vs. CBAI Distinction

Both papers place cost at the center of BAI, but with different objectives:

| | CABAI ([[Kanarios2024Cost]]) | CBAI ([[Lardy2025Constrained]]) |
|---|---|---|
| **Cost role** | Optimization metric: minimize $\sum_t C_t$ | Constraint: find arm with $\mathbb{E}[C_k] \leq \gamma$ |
| **Optimal allocation** | $w^*_a \propto \sqrt{c_a}$ (square-root rule) | Via $c_1$/$c_2$ transportation interface |
| **Answer set** | Always one of the $K$ arms | $K$ arms $\cup$ `None` (all infeasible) |
| **Cost-reward dependence** | Independent by assumption | Explicitly modeled |

## Open Problems

- Can CO's asymptotic optimality (CABAI) be established for $K > 2$ arms or general exponential families?
- Cost-aware BAI in the regret (non-stopping) setting
- ETC (Explore-Then-Commit) analogues for CABAI
- Fixed-budget variant of CBAI
- Multiple cost constraints in CBAI (currently exponential complexity in number of constraints)
- Non-asymptotic sample complexity for CBAI

## Related Topics
- [[offline-oracle-efficient-bandits]] — regret minimization setting; distinct goal (cumulative regret over T rounds rather than fixed-confidence identification)
