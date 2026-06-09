---
title: "Budget-Limited Multi-Armed Bandit"
tags: [multi-armed-bandits, regret, budget-limited-mab, bandits]
introduced_by: [[TranThanh2012Knapsack]]
---

# Budget-Limited Multi-Armed Bandit

**Definition:** A multi-armed bandit in which each pull of arm $i$ incurs a known cost $c_i$, and a single shared budget $B$ caps the *total* cost of all pulls — across both exploration and exploitation — so the agent must learn a *sequence* of (possibly different) arms that maximizes total reward within $B$.

## Intuition

Standard MAB rewards "find the best arm, then pull it forever." The budget-limited MAB breaks this because arms cost different amounts: a slightly-worse arm that is much cheaper can yield more total reward per unit budget. The right per-arm statistic is therefore the **reward density** $\mu_i / c_i$, not the mean $\mu_i$. Crucially, the budget limits exploitation too (not just an upfront exploration phase), so the agent cannot afford a long pure-exploration warmup.

The full-information optimum is the solution to an **unbounded knapsack** problem: knapsack capacity $= B$, item $i$ has weight $c_i$ and value $\mu_i$, choose pull-counts $\{x_i\}$ maximizing $\sum_i x_i\mu_i$ s.t. $\sum_i x_i c_i \le B$. Learning the policy = learning this knapsack solution online while estimating the $\mu_i$.

## Formal Description

$K$ arms; pulling arm $i$ costs $c_i$ and returns reward with mean $\mu_i$ (bounded support, WLOG $[0,1]$, $c_i \ge 1$). An algorithm $A$ produces pull-counts $N_i^A(B)$ with $\sum_i N_i^A(B)\, c_i \le B$ almost surely. Expected return $\mathbb{E}[G(A)] = \sum_i \mathbb{E}[N_i^A(B)]\,\mu_i$. With $A^* = \arg\max_A \sum_i \mathbb{E}[N_i^A(B)]\mu_i$ the (unachievable) optimum, the **regret** is

$$
R(A) = \mathbb{E}[G(A^*)] - \mathbb{E}[G(A)].
$$

The optimal density arm is $I^* = \arg\max_i \mu_i/c_i$; the minimal density gap $d_{\min} = \min_{j\neq I^*}\{\mu_{I^*}/c_{I^*} - \mu_j/c_j\}$ controls the leading regret constant. The best achievable regret is $\Theta(\ln B)$ — matching standard MAB's $\Theta(\ln T)$ since equal costs give $B/c = T$.

## Key Papers

- [[TranThanh2012Knapsack]] — introduces the model alongside the $\varepsilon$-first baseline; provides the first $O(\ln B)$ algorithms ([[kube]]) and the matching lower bound
- [[TranThanh2010Epsilon]] — the earlier $\varepsilon$-first treatment, $O(B^{2/3})$ regret (superseded)

## Variants & Related Concepts

- [[best-arm-identification]] — pure-exploration cousin; budget-limited MAB is cumulative-reward (regret), not identification
- [[Kanarios2024Cost]] / cost-aware BAI — attaches per-arm costs but in the fixed-confidence pure-exploration regime (minimize cost to identify the best arm), a different objective from the shared-budget cumulative-reward setting here
- [[upper-confidence-bound]] — the confidence-width machinery KUBE inherits
- Knapsack / unbounded knapsack — the full-information optimal policy is a knapsack solution

## Current State

The seminal framing for cost-constrained cumulative-reward bandits. The $O(\ln B)$ rate is settled (matching lower bound), but the leading constants are loose. Active descendants generalize to budgeted contextual bandits, bandits-with-knapsacks (BwK, multiple resource constraints), and non-stationary costs/rewards — the static-reward assumption is the main open frontier flagged here.
