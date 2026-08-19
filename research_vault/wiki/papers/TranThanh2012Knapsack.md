---
title: "Knapsack-based Optimal Policies for Budget-Limited Multi-Armed Bandits"
authors: [Long Tran-Thanh, Archie Chapman, Alex Rogers, Nicholas R. Jennings]
year: 2012
venue: AAAI
arxiv: "1204.1909"
tags: [budget-limited-mab, multi-armed-bandits, regret, ucb, knapsack, kube, bandits]
source: raw/papers/TranThanh2012Knapsack.pdf
---

# Knapsack-based Optimal Policies for Budget-Limited Multi-Armed Bandits

**TL;DR:** Introduces [[kube]] and fractional KUBE for the [[budget-limited-mab]] (the model itself is from the same group's earlier [[TranThanh2010Epsilon]]), where each arm pull costs $c_i$ and a single budget $B$ caps *both* exploration and exploitation. Both achieve $O(\ln B)$ regret — the first algorithms to do so — and this rate is shown to be asymptotically optimal.

## Problem

In the [[budget-limited-mab]] — introduced by the same group in [[TranThanh2010Epsilon]], together with its unbounded-knapsack characterization — pulling arm $i$ costs $c_i$ and the agent has one shared budget $B$ that limits the *total* cost of pulls across both exploration and exploitation phases. This breaks the standard MAB intuition: because costs are heterogeneous, the optimal full-information policy is **not** "pull the highest-mean arm repeatedly," but rather to pull a *sequence* of arms maximizing total reward within budget. The relevant per-arm quantity becomes **reward density** $\mu_i / c_i$, not the mean $\mu_i$.

Prior work either (a) budget-limited *only the exploration phase* (Antos et al. 2008; Bubeck et al. 2009; Guha & Munagala 2007), or (b) used a budget-limited [[epsilon-first]] policy ([[TranThanh2010Epsilon]]) that splits $B$ into $\varepsilon B$ exploration and $(1-\varepsilon)B$ exploitation. The $\varepsilon$-first method is sensitive to the choice of $\varepsilon$ and provably stuck at $O(B^{2/3})$ regret, far from the $O(\ln B)$ theoretical optimum. (The $O(B^{2/3})$ rate is this paper's characterization; the 2010 paper states only an $\varepsilon$- and $\delta$-parameterized bound, which yields $B^{2/3}$ when optimized over $\varepsilon$.)

## Method

The full-information optimum is the solution to an **unbounded knapsack** problem: choose non-negative integer pull-counts $\{x_i\}$ maximizing $\sum_i x_i \mu_i$ s.t. $\sum_i x_i c_i \le B$. KUBE makes this a *learning* algorithm:

**KUBE** (per step $t > K$, after pulling each arm once):
1. Solve the UCB-augmented knapsack — maximize $\sum_i m_{i,t}\big(\hat\mu_{i,n_i} + \sqrt{2\ln t / n_{i,t}}\big)$ s.t. $\sum_i m_{i,t} c_i \le B_t$ — using **density-ordered greedy** (sorts arms by UCB density $\frac{\hat\mu_i + \sqrt{2\ln t/n_i}}{c_i}$, fills greedily; $O(K\ln K)$ per step).
2. Let $M^*(B_t) = \{m^*_{i,t}\}$ be the resulting multiset. Pull arm $i$ with probability $P(i(t)=i) = m^*_{i,t} / \sum_k m^*_{k,t}$.
3. Update $\hat\mu$, decrement budget $B_{t+1} = B_t - c_{i(t)}$. Stop when $B_t < \min_j c_j$.

Sampling proportional to knapsack multiplicity means KUBE's *expected* reward equals that of the knapsack solution under current estimates — so as estimates converge, the pull sequence converges to the true optimal set, while UCB inflation keeps under-sampled arms in contention (joint explore/exploit, no explicit split).

**Fractional KUBE** replaces density-ordered greedy with the **fractional relaxation** (which optimally puts all budget on the single highest-density arm). This makes fractional KUBE the **budget-limited analogue of UCB** — it just pulls $\arg\max_i \big(\hat\mu_i/c_i + \sqrt{2\ln t/n_i}/c_i\big)$ — at $O(K)$ per step instead of $O(K\ln K)$.

## Results

- **Theorem 1 / Theorem 7 (regret upper bounds):** both KUBE and fractional KUBE have regret $O(\ln B)$. Fractional KUBE's constant is smaller, dominated by $\frac{8}{d_{\min}^2}\big(\sum_{\Delta_j>0}\Delta_j + \sum_{\delta_j>0}\frac{\delta_j}{c_{I^*}}\big)\ln\frac{B}{c_{\min}}$; KUBE additionally carries a $(c_{\max}/c_{\min})^2$ factor. Here $d_{\min}$ is the minimal density gap, $\Delta_j = \mu_{I^*} - \mu_j$, $\delta_j = c_j - c_{\min}$.
- **Theorem 8 (lower bound):** for any algorithm there is an instance with regret $\ge C\ln B$. Proof reduces standard MAB to budget-limited MAB (equal costs $\Rightarrow B/c = T$), inheriting the Lai–Robbins bound. Hence $O(\ln B)$ is asymptotically optimal.
- **Empirical (100 arms, truncated-Gaussian rewards):** both algorithms hug $C\ln(B/c_{\min})$ with small constant $C \in [4,7]$; $\varepsilon$-first stays above the $O(B^{2/3})$ line. KUBE beats fractional KUBE by **up to 40%** (moderately diverse costs) / **~30%** (extremely diverse), since density-ordered greedy approximates the knapsack better than the fractional relaxation. Both beat $\varepsilon$-first by up to 70% (KUBE) / 50% (fractional).

## Strengths & Limitations

**Strengths:** Foundational framing — recognizing the full-info optimum as unbounded knapsack and density as the right statistic is the key insight. First $O(\ln B)$ algorithms with a matching lower bound. Clean, well-motivated (wireless sensor energy budgets).

**Limitations:** Constants are loose ($1/d_{\min}^2$, $(c_{\max}/c_{\min})^2$) and the authors admit the bounds aren't tight — tellingly, fractional KUBE has the *better* bound but *worse* empirical performance, so the bounds don't explain practice. Stationary rewards only (no drift). The knapsack solver is approximate (NP-hard exactly), so "optimal" is up to the greedy approximation.

## Connections

- [[budget-limited-mab]] — the problem model this paper introduces
- [[kube]] — the core algorithm (and its fractional variant)
- [[upper-confidence-bound]] — fractional KUBE is the budget-limited analogue; UCB confidence width $\sqrt{2\ln t / n}$ is reused
- [[Kanarios2024Cost]] — also attaches per-arm costs, but pure-exploration/fixed-confidence (minimize cost to *identify* best arm) vs. this paper's cumulative-reward/regret under a *shared* budget; contrast in what "cost-aware" means
- **Extends:** [[TranThanh2010Epsilon]] — the paper that introduced the [[budget-limited-mab]] and its knapsack framing; this paper keeps the model and replaces its [[epsilon-first]] algorithm
- [[Tran-Thanh-Long]], [[Rogers-Alex]], [[Jennings-Nicholas-R]] — authors

## Open Questions

- Can the constants be tightened so the bound predicts the KUBE-vs-fractional gap (currently fractional has the better bound but loses in practice)?
- Non-stationary extension: the convergence argument relies on static $\mu_i$ — how to handle drifting rewards (authors flag this as future work)?
- Does the density-ordered-greedy approximation gap ever dominate the learning regret for adversarially chosen costs?
