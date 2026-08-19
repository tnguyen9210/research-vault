---
title: "Upper Confidence Bound (UCB)"
tags: [bandits, exploration, confidence-bounds, regret]
introduced_by: [[TranThanh2012Knapsack]]
---

# Upper Confidence Bound (UCB)

**Definition:** The optimism-in-the-face-of-uncertainty rule for online decision-making: maintain a high-probability upper bound on each arm's mean, $\bar\mu_i + \text{width}_i$, and pull the arm maximizing it.

## Intuition

Optimism resolves the explore/exploit tension without an explicit exploration schedule. Pulling the arm with the highest optimistic estimate means either the arm really is good — a good pull — or it is not, in which case the estimate falls and the bound tightens. Either outcome is progress, so the number of pulls of a suboptimal arm is bounded by how long its confidence interval can stay above the optimum's mean.

The width shrinks as $\sqrt{\log t / n_i}$, so arms pulled rarely stay optimistic and get revisited, while well-sampled arms are judged on their means. Regret is then gap-dependent, $O(\sum_{i\neq i^*}\log T/\Delta_i)$ — one of the earliest [[instance-dependent-bounds]].

Note the sign convention relative to offline problems: online, an over-estimate costs one round and is self-correcting, so optimism is safe. Offline there is no correcting round, so the same confidence machinery is used with the opposite sign — see [[pessimism-principle]].

## Formal Description

For a stochastic $K$-armed bandit with rewards in $[0,1]$, UCB1 (Auer et al. 2002) pulls

$$
i_t = \arg\max_{i}\ \bar\mu_{i,t-1} + \sqrt{\frac{2\log t}{n_{i,t-1}}},
$$

with $\bar\mu_{i,t-1}$ the empirical mean and $n_{i,t-1}$ the pull count. This achieves $\mathbb{E}[R_T] = O\big(\sum_{i:\Delta_i>0}\log T/\Delta_i\big)$, matching the Lai–Robbins lower bound up to constants.

**Structured generalizations** replace the count-based width with a geometry-aware one: $\|\phi\|_{\Sigma^{-1}}$ in linear bandits, and $\|\nabla_\theta f\|_{\Sigma^{-1}}$ under [[differentiable-function-approximation]]. The pattern — an empirical estimate plus a width measured in the inverse information matrix — recurs throughout this vault in both signs.

**Budget-limited variant.** When each arm carries a cost $c_i$ and a shared budget $B$ caps total spend, the relevant object is the reward *density* $\mu_i/c_i$ rather than $\mu_i$. Fractional [[kube]] applies UCB to densities and is described in [[TranThanh2012Knapsack]] as the budget-limited analogue of UCB.

## Key Papers

- Lai & Robbins (1985) — the asymptotic instance-dependent lower bound UCB matches
- Auer, Cesa-Bianchi & Fischer (2002) — UCB1; the finite-time logarithmic regret analysis
- [[TranThanh2010Epsilon]] — uses UCB purely as an *exploration* subroutine inside an [[epsilon-first]] shell, and finds it performs no better than uniform sampling: the $B^{2/3}$ ceiling comes from the phase split, not the sampling rule
- [[TranThanh2012Knapsack]] — fractional [[kube]] as the budget-limited analogue of UCB; [[kube]] itself solves a UCB-augmented knapsack each step
- Abbasi-Yadkori et al. (2011) — self-normalized confidence sets for linear bandits; the source of the elliptical-bonus form reused throughout

## Variants & Related Concepts

- [[kube]] / [[budget-limited-mab]] — UCB adapted to reward densities under a shared budget
- [[pessimism-principle]] — the offline mirror image: same confidence widths, subtracted rather than added
- [[best-arm-identification]] — pure exploration uses confidence bounds toward a different objective (identify the best arm) with a different optimality notion (sample complexity, not regret)
- **Thompson sampling** — the Bayesian alternative; comparable regret, often better empirically (see [[Yang2025Stochastically]])
- [[instance-dependent-bounds]] — UCB's gap-dependent regret is the canonical example

## Current State

Foundational and settled for the basic stochastic bandit. The active work is in generalizing the confidence width — to structured function classes, to heavy-tailed or unbounded rewards (betting-based bounds, see [[Ryu2025Improved]]), to cost- and budget-constrained settings ([[TranThanh2012Knapsack]], [[Kanarios2024Cost]]), and to sequential settings where the same machinery becomes an exploration bonus in RL.
