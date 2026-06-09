---
title: "KUBE (Knapsack-based Upper Confidence Bound Exploration)"
tags: [budget-limited-mab, ucb, knapsack, regret, kube, bandits]
introduced_by: [[TranThanh2012Knapsack]]
---

# KUBE (Knapsack-based Upper Confidence Bound Exploration)

**Definition:** An online algorithm for the [[budget-limited-mab]] that, at each step, solves a UCB-augmented unbounded-knapsack problem over the arms and pulls an arm with probability proportional to its multiplicity in that knapsack solution — jointly exploring and exploiting without an explicit phase split.

## Intuition

If the true means were known, the optimal budget-limited policy is the unbounded-knapsack solution (pull arms in proportion to the optimal pull-counts). KUBE estimates this online: replace true densities $\mu_i/c_i$ with **UCB densities** $\frac{\hat\mu_i + \sqrt{2\ln t/n_i}}{c_i}$, re-solve the knapsack each step, and sample the next arm from the solution's multiplicity distribution. Sampling proportionally makes KUBE's expected reward equal the knapsack value under current estimates; UCB inflation keeps under-sampled (cheap-information) arms in contention. As estimates converge, the pull sequence converges to the true optimal set.

## Formal Description

After an initial phase pulling each arm once, at step $t > K$ with residual budget $B_t$, solve

$$
\max_{\{m_{i,t}\}} \sum_{i=1}^K m_{i,t}\Big(\hat\mu_{i,n_i} + \sqrt{\tfrac{2\ln t}{n_{i,t}}}\Big)
\quad\text{s.t.}\quad \sum_{i=1}^K m_{i,t}\, c_i \le B_t,\; m_{i,t}\in\mathbb{Z}_{\ge 0},
$$

via **density-ordered greedy** (sort by UCB density, fill greedily; $O(K\ln K)$ per step). Let $M^*(B_t)=\{m^*_{i,t}\}$; pull arm $i$ with probability

$$
P(i(t)=i) = \frac{m^*_{i,t}}{\sum_k m^*_{k,t}}.
$$

Update $\hat\mu_{i(t)}$ and set $B_{t+1} = B_t - c_{i(t)}$; stop when $B_t < \min_j c_j$.

**Fractional KUBE** swaps density-ordered greedy for the **fractional relaxation** of the unbounded knapsack, which optimally allocates all budget to the single highest-density arm. So fractional KUBE deterministically pulls $\arg\max_i\big(\hat\mu_i/c_i + \sqrt{2\ln t/n_i}/c_i\big)$ — exactly the **budget-limited analogue of [[upper-confidence-bound]]** — at $O(K)$ per step.

## KUBE vs. Fractional KUBE

**Same objective.** Both minimize regret on the [[budget-limited-mab]], target the same $O(\ln B)$ rate, and share the *same* full-information optimum (the unbounded-knapsack solution on true densities $\mu_i/c_i$). There is no objective-level difference — only an algorithmic one in *how the per-step knapsack is approximated*, which then cascades into the next-arm rule and the per-step cost. (Their regret *bounds* differ in constants, but that is a property of the analysis, not the goal.)

Each step both build the UCB density $\rho_i = \frac{\hat\mu_i + \sqrt{2\ln t/n_i}}{c_i}$ and face the same unbounded-knapsack subproblem (NP-hard exactly). They diverge only in the approximation used:

| | **KUBE** | **Fractional KUBE** |
|---|---|---|
| Knapsack solver | density-ordered **greedy** | **fractional relaxation** |
| Solution shape | a *multiset* $\{m^*_i\}$ — several arms, integer counts | all budget on **one** arm, $\arg\max_i \rho_i$ |
| Next-arm rule | **sample** arm $i$ w.p. $m^*_i / \sum_k m^*_k$ | **deterministically** pull $\arg\max_i \rho_i$ |
| Per-step cost | $O(K\ln K)$ | $O(K)$ |
| Reduces to | knapsack-aware UCB | plain [[upper-confidence-bound]] (budget-limited) |

**Why KUBE randomizes.** Density-ordered greedy returns a multiset that typically mixes several arms; sampling proportional to multiplicity makes KUBE's *expected* single-step reward equal the average reward of following the whole knapsack solution, so the realized pull frequencies converge to the true optimal set as $\hat\mu_i \to \mu_i$. The greedy keeps a *spread* of arms in play, which the optimal knapsack genuinely needs when costs are heterogeneous.

**Why fractional KUBE is just UCB.** The fractional unbounded knapsack puts the entire budget on the single highest-density item, so "sample proportional to the solution" degenerates to "pull that one arm w.p. 1" — i.e. $\arg\max_i \rho_i$, which is exactly UCB with each term divided by cost. No mixing, no randomization.

**The inversion (bound vs. practice).** Fractional KUBE has the *tighter* regret bound (it drops the $(c_{\max}/c_{\min})^2$ factor KUBE carries) and is *cheaper*, yet **KUBE wins empirically** — up to 40% lower regret under moderately diverse costs, ~30% under extremely diverse costs. Reason: density-ordered greedy is a *better approximation* of the true integer knapsack, so KUBE converges to the optimal set faster; the fractional relaxation discards the mixing structure the real optimum needs when costs differ. When costs are **homogeneous** the two coincide (greedy stops after ≈ one item ≈ fractional) and performance matches. So the bounds order them one way and practice the opposite — a textbook bound-looseness symptom the authors concede; do not read the cleaner bound as the better algorithm. See [[TranThanh2012Knapsack]].

## Key Papers

- [[TranThanh2012Knapsack]] — introduces KUBE and fractional KUBE; proves $O(\ln B)$ regret for both and a matching $\Omega(\ln B)$ lower bound

## Variants & Related Concepts

- **Fractional KUBE** — the UCB-on-densities variant; full contrast in [§ KUBE vs. Fractional KUBE](#kube-vs-fractional-kube) above
- [[upper-confidence-bound]] — fractional KUBE reduces to UCB when costs are equal; KUBE generalizes UCB to the knapsack setting
- [[budget-limited-mab]] — the problem KUBE solves
- $\varepsilon$-first ([[TranThanh2010Epsilon]]) — the baseline KUBE beats; $\varepsilon$-first is $O(B^{2/3})$, KUBE is $O(\ln B)$

## Current State

The reference algorithm for budget-limited MABs. The density-ordered-greedy version remains the empirically stronger choice when arm costs are heterogeneous; the fractional/UCB version is preferred when per-step compute matters or costs are near-homogeneous (where the two coincide).
