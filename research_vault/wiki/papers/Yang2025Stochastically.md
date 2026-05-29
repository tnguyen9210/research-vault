---
title: "Stochastically Constrained Best Arm Identification with Thompson Sampling"
authors: [Le Yang, Siyang Gao, Cheng Li, Yi Wang]
year: 2025
venue: arXiv
arxiv: "2501.03877"
tags: [best-arm-identification, constrained-bai, pure-exploration, thompson-sampling, fixed-budget]
source: raw/papers/Yang2025Stochastically.pdf
---

# Stochastically Constrained Best Arm Identification with Thompson Sampling

**TL;DR:** Introduces BFAI-TS — a Thompson Sampling algorithm for fixed-budget Best Feasible Arm Identification with $m$ stochastic constraints. Adapts the top-two TS framework via parameter $\beta$ to avoid over-sampling the best feasible arm; proves asymptotically optimal exponential convergence rate $\Gamma_{\beta^*}$ for the probability of false selection.

## Problem

**Best Feasible Arm Identification (BFAI):** $k$ arms, each pull yielding an $(m+1)$-dimensional sample $\mathbf{X}_{t,i} = [X_{t,i0}, X_{t,i1}, \ldots, X_{t,im}]^\top$ where $X_{t,ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$ with known variances. Arm $i$ is *feasible* if $\mu_{ij} \leq \gamma_j$ for all $j = 1, \ldots, m$. Goal: within a fixed budget of $n$ rounds, find $I^* = \arg\max_{i \in \mathcal{F}} \mu_{i0}$, minimizing the probability of false selection (PFS) $1 - P_{n,1}$.

This is the **fixed-budget** analogue of [[constrained-bai]] (which is fixed-confidence). The paper calls it BFAI; the same problem structure appears in the simulation optimization literature as constrained ranking-and-selection (R&S).

Four arm categories drive the analysis:
- $I^*$: best feasible arm
- $\mathcal{F}_w$: feasible but suboptimal arms (better reward needed to eliminate)
- $\mathcal{I}_b$: infeasible arms with $\mu_{i0} \geq \mu_{I^*,0}$ — "dangerous" (would win if feasible; need to confirm infeasibility)
- $\mathcal{I}_w$: infeasible arms with $\mu_{i0} < \mu_{I^*,0}$ — need both inferior objective and infeasibility confirmed

## Method

**Naive TS for BFAI fails.** Pulling proportional to $P_{t,i}$ (posterior probability of being best feasible) allocates too many samples to $I^*$ asymptotically, degrading convergence. Fix: introduce parameter $\beta \in (0,1)$.

**BFAI-TS Algorithm:**  
At each round $t$, sample $\theta \sim \Pi_t$ (product of independent Gaussian posteriors):
1. Get feasible set $F = \{i : \theta_{ij} \leq \gamma_j, \forall j\}$; set $I_t^{(1)} = \arg\max_{i \in F} \theta_{i0}$ (uniform if $F = \emptyset$)
2. With probability $\beta$: pull $I_t^{(1)}$
3. With probability $1 - \beta$: resample until finding $I_t^{(2)} \neq I_t^{(1)}$ (best feasible from remaining); pull $I_t^{(2)}$

This mirrors top-two sampling for standard BAI (Russo 2020), adapted so feasibility is checked at the posterior sample level. The sampling probability of arm $i$ is:
$$\phi_{t,i} = \frac{c_t}{k} + (1-\beta) P_{t,i} \sum_{i' \neq i} \left(\frac{P_{t,i'}}{1 - P_{t,i'}}(1 - c_t) + \frac{c_t}{k-1}\right) + P_{t,i}\beta(1-c_t)$$
where $c_t$ = probability all posterior samples are infeasible.

**Optimal sampling rates** $\alpha_i^\beta$ are defined by the balance condition:
$$\mathcal{R}_i = \mathcal{R}_{i'} \quad \forall i \neq i' \neq 1, \qquad \text{where} \quad \mathcal{R}_i = \frac{(\mu_{i0}-\mu_{10})^2}{(\sigma_{10}^2/\alpha_i^\beta + \sigma_{i0}^2/\beta)}\mathbf{1}\{i \in \mathcal{F}_w \cup \mathcal{I}_w\} + \alpha_i^\beta \sum_{j \in \mathcal{M}_I^i} \frac{(\mu_{ij}-\gamma_j)^2}{\sigma_{ij}^2}\mathbf{1}\{i \in \mathcal{I}_b \cup \mathcal{I}_w\}$$

**Rate of posterior convergence $\Gamma_\beta$** (eq. 4.2):
$$\Gamma_\beta = \min_{i \neq 1} \left(\frac{(\mu_{i0}-\mu_{10})^2}{2(\sigma_{10}^2/\alpha_i^\beta + \sigma_{i0}^2/\beta)}\mathbf{1}\{i \in \mathcal{F}_w \cup \mathcal{I}_w\} + \alpha_i^\beta \!\!\sum_{j \in \mathcal{M}_I^i}\!\! \frac{(\mu_{ij}-\gamma_j)^2}{2\sigma_{ij}^2}\mathbf{1}\{i \in \mathcal{I}_b \cup \mathcal{I}_w\},\ \min_{j \in \mathcal{M}_F^i} \beta\frac{(\mu_{1j}-\gamma_j)^2}{2\sigma_{1j}^2}\right)$$

The first term governs discrimination between $I^*$ and arm $i$ (either by objective or constraint violation). The second term governs confirming $I^*$'s own constraints are satisfied.

**Theorem 4.1** (Optimal sample allocation): $N_{n,i}/n \xrightarrow{p} \alpha_i^\beta$ for all $i$.

**Theorem 4.2** (Asymptotic optimality):
1. For any $\beta \in (0,1)$: $\lim_{n \to \infty} -\frac{1}{n}\log(1 - P_{n,1}) = \Gamma_\beta$ — BFAI-TS achieves the fastest possible posterior convergence rate among all algorithms allocating $\beta$ fraction to $I^*$.
2. With $\beta = \beta^*$ (the maximizer of $\Gamma_\beta$): achieves globally optimal rate $\Gamma_{\beta^*}$.

## Results

Five synthetic experiments ($k = 10$ or $50$ arms, $m = 1$ to $4$ constraints) and a dose-finding clinical trial (secukinumab, 5 dosage arms, 2 performance measures). PFS at selected sample sizes:

| Algorithm | Exp 1 ($n$=3400) | Exp 2 ($n$=3500) | Dose ($n$=8000) |
|---|---|---|---|
| BFAI-TS ($\beta=\beta^*$) | **0.01** | **0.01** | **0.01** |
| BFAI-TS ($\beta$=0.5) | 0.02 | 0.03 | 0.05 |
| OCBA-CO | 0.04 | 0.05 | 0.10 |
| TF-LUCB | 0.13 | 0.46 | 0.09 |
| BFAI-TS-1 ($\beta$=1, no top-two) | 0.13 | 0.13 | 0.06 |
| MD-UCBE | 0.05 | 0.13 | 0.19 |
| Uniform | 0.46 | 0.59 | 0.27 |

PFS on log scale shows linear decay for BFAI-TS — confirming the exponential convergence rate $\Gamma_{\beta^*}$. Setting $\beta = 1$ (BFAI-TS-1) degrades substantially, confirming the necessity of the top-two framework.

## Strengths & Limitations

**Strengths:** First TS-based algorithm for BFAI with provable asymptotic optimality. Handles $m$ simultaneous constraints natively. The $\beta$ parameter is interpretable and can be adaptively tuned via estimated $\mathcal{R}_i$. Strong empirical performance across diverse problem structures.

**Limitations:** Fixed-budget only (no fixed-confidence guarantees). Requires known variances $\sigma_{ij}^2$ (though posteriors concentrate regardless). Assumes Gaussian distributions; non-parametric extension is open. Asymptotic result — finite-budget behavior is empirical only.

## Connections

- [[constrained-bai]] — complementary regime: this paper is **fixed-budget** (fixed $n$, minimize PFS); [[Lardy2025Constrained]] is **fixed-confidence** (minimize $\mathbb{E}[\tau_\delta]$). Same problem structure, different theoretical frameworks.
- [[best-arm-identification]] — BFAI extends fixed-budget BAI to $m$ stochastic constraints
- [[cabai]] — distinct: CABAI minimizes cumulative testing cost; BFAI minimizes false selection probability within a fixed budget
- [[Yang-Le]] — first author
- [[Wang-Yi]] — senior author
- **Extends:** Russo 2020 (top-two TS for unconstrained BAI) — BFAI-TS adds feasibility tracking to the top-two TS framework
- **Related:** Katz-Samuels & Scott 2019 (TF-LUCB, fixed-confidence top-feasible arm ID); OCBA-CO (Lee et al. 2012, constrained R&S)

## Open Questions

- Finite-budget non-asymptotic PFS bounds for BFAI-TS?
- Extension to unknown variances $\sigma_{ij}^2$?
- Non-Gaussian (sub-Gaussian, exponential family) arm distributions?
- Can $\beta^*$ be computed efficiently without exhaustive search over $\beta$?
