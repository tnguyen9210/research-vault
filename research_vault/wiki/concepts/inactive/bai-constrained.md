---
title: "Constrained Best Arm Identification (CBAI)"
tags: [best-arm-identification, pure-exploration, cost-constraint, bandits]
aliases: [constrained-bai, best-arm-identification-constrained]
introduced_by: [[lardy2025Constrained]]
---

# Constrained Best Arm Identification (CBAI)

**Definition:** Fixed-confidence BAI where each arm has a joint distribution over (reward, cost) and the goal is to identify the arm with highest mean reward among arms whose **mean cost is below a threshold** $`\gamma \in \mathbb{R}`$, or report `None` if no arm is feasible.

## Intuition

Standard BAI finds the best arm. CBAI finds the best *economically viable* arm — one whose average cost satisfies a budget constraint. Models A/B testing in online platforms where each policy has both a performance metric (e.g., conversion rate) and an economic cost (e.g., discount spend), and the goal is the best policy that is sustainable.

Crucially, reward and cost are **jointly distributed per arm** and may be dependent. A policy that improves reward may also increase cost — this correlation changes the structure of the optimal exploration strategy.

## Formal Description

Bandit $`\boldsymbol{\nu} = (\nu_1, \ldots, \nu_K)`$ where $`\nu_k`$ is a bivariate distribution with mean $`\boldsymbol{m}(\nu_k) = (m_{k,1}, m_{k,2})`$. Given threshold $`\gamma`$:

```math
i^*(\boldsymbol{\nu}) = \arg\max_{\{i:\, m_{i,2} \leq \gamma\}} m_{i,1}, \quad \text{or } \texttt{None} \text{ if } m_{i,2} > \gamma \text{ for all } i
```

Instance-dependent lower bound: $`\mathbb{E}[\tau_\delta] \geq T^*(\boldsymbol{\nu}) \cdot \mathrm{KL}(\delta \| 1 - \delta)`$, where $`T^*`$ is computed via the $`c_1`$/$`c_2`$ transportation cost interface in [[lardy2025Constrained]].

## Key Papers

- [[lardy2025Constrained]] — **fixed-confidence** CBAI; information-theoretic lower bound via transportation cost interface; optimal TaS for Gaussian and non-parametric arms; handles dependent reward-cost distributions
- [[yang2025Stochastically]] — **fixed-budget** BFAI with $`m`$ constraints; BFAI-TS (Thompson sampling + top-two); asymptotically optimal exponential convergence rate $`\Gamma_{\beta^*}`$; Gaussian arms, known variances

## The Fixed-Confidence vs. Fixed-Budget Distinction

| | Fixed-confidence ([[lardy2025Constrained]]) | Fixed-budget ([[yang2025Stochastically]]) |
|---|---|---|
| **Input** | Confidence $`\delta`$ | Budget $`n`$ |
| **Objective** | Minimize $`\mathbb{E}[\tau_\delta]`$ | Minimize PFS $`= 1 - P_{n,1}`$ |
| **Algorithm** | TaS (frequentist GLR) | BFAI-TS (Bayesian Thompson sampling) |
| **Dependence** | Full joint distributions | Independent objective + constraints |

## Variants

- [[bai-cost-aware]] — **cost minimization** formulation: minimize $`\sum_t C_t`$ (cumulative testing cost) to find the best arm ([[kanarios2024Cost]]). Key distinction: CABAI's cost is the *optimization metric*; CBAI/BFAI's cost is a *feasibility constraint on arm means*.

## Related Concepts

- [[bai]] — unconstrained BAI; constrained BAI adds per-arm feasibility constraints

## Current State and Open Problems

Two complementary regimes now solved (2025): fixed-confidence ([[lardy2025Constrained]]) and fixed-budget ([[yang2025Stochastically]]). Open: multiple cost constraints (CBAI), non-Gaussian distributions (BFAI), non-asymptotic bounds for both.
