---
title: "Epsilon-First Policies"
tags: [multi-armed-bandits, exploration, budget-limited-mab, explore-then-commit]
introduced_by: [[TranThanh2010Epsilon]]
---

# Epsilon-First Policies

**Definition:** An explore-then-commit scheme for budgeted decision problems: dedicate a fixed fraction $\varepsilon$ of the resource to pure exploration, then spend the remaining $(1-\varepsilon)$ exploiting the resulting estimates, with no further learning.

## Intuition

The appeal is analytic tractability. Because the two phases are disjoint, estimate quality at the end of exploration is a clean function of $\varepsilon$ alone, and the exploitation step can be analyzed as a fixed optimization on those estimates. That decomposition is what makes a loss bound provable when interleaved schemes are still hard to analyze — it is why [[TranThanh2010Epsilon]] could give the first guarantee for the [[budget-limited-mab]].

The weakness is structural, not incidental. Splitting the budget means:

1. **Exploration earns nothing.** The $\varepsilon B$ spent learning is a pure write-off, whereas an interleaved policy collects reward while it learns.
2. **Exploitation learns nothing.** Estimates freeze at the phase boundary, so a mistake made with the exploration budget is paid for over the entire remaining $(1-\varepsilon)B$.
3. **$\varepsilon$ must be set in advance**, and the optimal value depends on quantities not yet estimated — in the budget-limited MAB, on the spread of reward densities $D_{\max}$.

These are exactly the losses that [[kube]] recovers by folding exploration into the exploitation objective, which is how it improves $O(B^{2/3})$ to $O(\ln B)$.

## Formal Description

In the [[budget-limited-mab]] instantiation of [[TranThanh2010Epsilon]]: split budget $B$ into $\varepsilon B$ and $(1-\varepsilon)B$.

**Explore.** Pull arms uniformly in sequence until $\varepsilon B$ is exhausted, giving $n_i \ge \big\lfloor \varepsilon B/\sum_{j=1}^k c_j\big\rfloor$ pulls of every arm. Uniform sampling is chosen because equal $n_i$ makes the accuracy analysis uniform across arms.

**Commit.** Solve the unbounded knapsack on the estimates $\hat\mu_i$ with density-ordered greedy, and pull that combination.

**Guarantee (Corollary 2).** With probability $(1-\delta)^k$,

$$
L(A_{\varepsilon\text{-first}}) \le \underbrace{2\varepsilon B\, D_{\max}}_{\text{cost of exploring}} + \underbrace{2B\sqrt{\frac{(-\ln\delta)\sum_j c_j}{\varepsilon B}}}_{\text{cost of imprecision}} .
$$

The two terms move in opposite directions in $\varepsilon$, which is the whole design tension made explicit. Balancing them gives $\varepsilon^* \propto B^{-1/3}$ and a loss of order $B^{2/3}$ — an optimization performed by [[TranThanh2012Knapsack]] rather than in the original paper.

**The $B^{2/3}$ barrier.** The $\varepsilon B$ term is linear in the exploration budget and the estimation term falls only as $\varepsilon^{-1/2}$, so no choice of $\varepsilon$ does better than $B^{2/3}$. The ceiling is a property of the phase split, not of the sampling rule inside it — which is why swapping uniform exploration for UCB exploration does not help, as [[TranThanh2010Epsilon]] found empirically.

## Key Papers

- [[TranThanh2010Epsilon]] — introduces the budgeted $\varepsilon$-first policy; Theorem 1 (any exploration policy) and Corollary 2 (uniform pull)
- [[TranThanh2012Knapsack]] — shows the family is stuck at $O(B^{2/3})$ and replaces it with interleaved [[kube]] at $O(\ln B)$, with a matching lower bound
- Auer, Cesa-Bianchi & Fischer (2002) — $\varepsilon_n$-greedy, the decaying-$\varepsilon$ relative that keeps exploring forever; the baseline $\varepsilon$-first is compared against

## Variants & Related Concepts

- **$\varepsilon_n$-greedy / $\varepsilon$-greedy** — exploration probability decays over time rather than being confined to a prefix; keeps learning throughout, but in a budget-limited setting can lock onto a wrong arm and lack the budget to recover
- **Explore-then-commit (ETC)** — the same idea in standard bandits, where the phase length is chosen in rounds rather than as a budget fraction; known to be suboptimal against UCB for the same structural reason
- [[kube]] — the interleaved successor; exploration bonus enters the knapsack objective instead of occupying its own phase
- [[upper-confidence-bound]] — can be dropped into the exploration phase, but does not lift the $B^{2/3}$ ceiling
- [[budget-limited-mab]] — the setting this family was designed for

## Current State

Superseded as an algorithm, retained as a baseline and a teaching case. The reason it stays interesting is the clean separation it exposes: the $B^{2/3}$ ceiling is caused by the *phase split*, independent of how well either phase is executed. That makes it the standard illustration of why exploration and exploitation should be interleaved rather than scheduled, in budgeted settings and beyond. It remains attractive when a guarantee is needed and the interleaved analysis is intractable, since the two-phase decomposition is far easier to bound.
