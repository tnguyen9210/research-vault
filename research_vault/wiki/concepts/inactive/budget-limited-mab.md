---
title: "Budget-Limited Multi-Armed Bandit"
aliases: [budget-limited-bandits]
tags: [multi-armed-bandits, regret, budget-limited-mab, bandits]
introduced_by: [[TranThanh2010Epsilon]]
---

# Budget-Limited Multi-Armed Bandit

**Definition:** A multi-armed bandit in which each pull of arm $i$ incurs a known cost $c_i$, and a single shared budget $B$ caps the *total* cost of all pulls — across both exploration and exploitation — so the agent must learn a *sequence* of (possibly different) arms that maximizes total reward within $B$.

> **Scope.** The shared-budget cumulative-reward setting. **Left to other pages:** the pure-exploration regime where cost is minimized to reach a confidence level, on [[bai-cost-aware]]; the policies themselves, on [[budget-limited-mab-epsilon-first]] and [[budget-limited-mab-kube]].

## Intuition

Standard MAB rewards "find the best arm, then pull it forever." The budget-limited MAB breaks this because arms cost different amounts: a slightly-worse arm that is much cheaper can yield more total reward per unit budget. The right per-arm statistic is therefore the **reward density** $\mu_i / c_i$, not the mean $\mu_i$. Crucially, the budget limits exploitation too — not just an upfront exploration phase — so the agent cannot afford a long pure-exploration warmup.

The full-information optimum is the solution to an **unbounded knapsack** problem: capacity $= B$, item $i$ has weight $c_i$ and value $\mu_i$, choose pull-counts $\{x_i\}$ maximizing $\sum_i x_i\mu_i$ subject to $\sum_i x_i c_i \le B$. Learning the policy = learning this knapsack solution online while estimating the $\mu_i$.

## Formal Description

$K$ arms; pulling arm $i$ costs $c_i$ and returns reward with mean $\mu_i$ (bounded support, WLOG $[0,1]$, $c_i \ge 1$). An algorithm $A$ produces pull-counts $N_i^A(B)$ with $\sum_i N_i^A(B)\, c_i \le B$ almost surely. Expected return $\mathbb{E}[G(A)] = \sum_i \mathbb{E}[N_i^A(B)]\,\mu_i$. With $A^* = \arg\max_A \sum_i \mathbb{E}[N_i^A(B)]\mu_i$ the (unachievable) optimum, the **regret** is

$$
R(A) = \mathbb{E}[G(A^*)] - \mathbb{E}[G(A)].
$$

The optimal density arm is $I^* = \arg\max_i \mu_i/c_i$; the minimal density gap $d_{\min} = \min_{j\neq I^*}\{\mu_{I^*}/c_{I^*} - \mu_j/c_j\}$ controls the leading regret constant. The best achievable regret is $\Theta(\ln B)$ — matching standard MAB's $\Theta(\ln T)$, since equal costs give $B/c = T$.

## Literature Survey

A short thread: two papers, two years apart, that between them pose the problem and close its rate. It is included as a survey rather than a list because the *reason* the second improves on the first is the transferable lesson.

### Posing the model, and the cost of phasing exploration

[[TranThanh2010Epsilon]] introduces the model, the unbounded-knapsack characterization of its optimum, and the reward-density statistic. Its policy, [[budget-limited-mab-epsilon-first]], spends $\varepsilon B$ on uniform exploration and then runs density-ordered greedy on the remainder. Optimizing the resulting loss bound over $\varepsilon$ gives $O(B^{2/3})$ — and that exponent is not an artifact of the analysis but of the *phase split*: a budget fraction committed to uniform exploration is spent whether or not the estimates needed it.

### Interleaving exploration into the objective

[[TranThanh2012Knapsack]] gives the first $O(\ln B)$ algorithms, [[budget-limited-mab-kube]] and fractional KUBE, with a matching lower bound. The mechanism is the one worth carrying elsewhere: rather than giving exploration its own phase, put the confidence bound *inside* the knapsack objective, so that uncertainty is priced alongside value and exploration happens only where it changes the knapsack solution. The same paper supplies the $O(B^{2/3})$ characterization of $\varepsilon$-first, which is how the two results are known to be separated rather than merely unequal.

### The neighbouring objectives

Pricing arm pulls admits several non-equivalent problem statements, and the vault holds two of them. Here, one shared budget caps exploration and exploitation together and the objective is cumulative reward, which makes reward density $\mu_i/c_i$ the governing statistic and gives a $\Theta(\ln B)$ rate. In [[bai-cost-aware]], the problem is a stopping time: identify the best arm at confidence $1-\delta$ while minimizing cost, and the governing statistic becomes $\sqrt{c_a}$ sampling proportions ([[Kanarios2024Cost]]). These are genuinely different problems and the transfer between them is weak — which is worth stating, because "cost-aware bandits" in the literature names both.

Bandits-with-knapsacks, which generalizes the single scalar budget to several simultaneous resource caps, is the natural next generalization and has no page here.

## Variants

- [[budget-limited-mab-epsilon-first]] — the first policy family proposed for this model; capped at $O(B^{2/3})$ by its phase split
- [[budget-limited-mab-kube]] — KUBE and fractional KUBE, the $O(\ln B)$ answer
- **Bandits-with-knapsacks (BwK)** — several simultaneous resource constraints in place of one scalar budget
- **Budgeted contextual bandits** — the same cost structure with per-round context

## Related Concepts

- [[bai]] — the pure-exploration cousin; this setting is cumulative-reward, not identification
- [[bai-cost-aware]] — cost in the fixed-confidence regime; see the third survey direction for why the two do not reduce to each other
- [[upper-confidence-bound]] — the confidence-width machinery KUBE inherits
- [[instance-dependent-bounds]] — the $d_{\min}$-dependent form every rate here takes

## Current State and Open Problems

The seminal framing for cost-constrained cumulative-reward bandits, and the $O(\ln B)$ rate is settled with a matching lower bound. What is not settled is everything quantitative below that rate, and every assumption the model makes about costs.

- **Tight constants.** The $O(\ln B)$ bounds carry loose $1/d_{\min}^2$ and $(c_{\max}/c_{\min})^2$ factors that do not predict empirical orderings — so the theory ranks algorithms it cannot rank in practice.
- **Non-stationary costs and rewards.** Both analyses assume static $\mu_i$; drift breaks the convergence-to-knapsack argument that the whole approach rests on.
- **Stochastic or unknown costs.** Both threads assume per-arm costs are known exactly, which is the least realistic of the model's assumptions.
- **Multiple resource constraints.** Generalizing to BwK is the standard next step and is not covered here.

## Provenance

*Sourced.* Everything attributed to [[TranThanh2010Epsilon]] and [[TranThanh2012Knapsack]] comes from their paper pages, written against the PDFs at ingest. The [[Kanarios2024Cost]] contrast in the third direction comes from [[bai-cost-aware]].

*This page's judgment, not a citation.* The reading of $O(B^{2/3})$ as a consequence of the phase split rather than of the analysis, the claim that transfer between the shared-budget and fixed-confidence objectives is weak, and the observation that the loose constants are what stops the theory predicting empirical orderings.

*Merged.* This page absorbed the topic page `budget-limited-bandits` on 2026-09-21; its slug is kept as an alias, and its cost-aware half went to [[bai-cost-aware]].
