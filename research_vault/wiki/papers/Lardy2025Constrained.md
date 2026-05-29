---
title: "Constrained Best Arm Identification"
authors: [Tyron Lardy, Christina Katsimerou, Wouter M. Koolen]
year: 2025
venue: NeurIPS
arxiv: "2412.08031"
tags: [best-arm-identification, constrained-bai, pure-exploration, bandits, cost-constraint]
source: raw/papers/Lardy2025Constrained.pdf
---

# Constrained Best Arm Identification

**TL;DR:** Introduces CBAI — fixed-confidence BAI where each arm has a **joint** (reward, cost) distribution and the goal is to find the best-reward arm among those with mean cost $\leq \gamma$, or report `None` if all are infeasible. Derives $T^*(\boldsymbol{\nu})$ via a transportation cost interface and proposes an asymptotically optimal TaS algorithm covering three model families including dependent reward-cost distributions.

## Problem

Standard BAI ignores cost. [[Kanarios2024Cost]] (CABAI) minimizes *cumulative testing cost* to find the best arm. This paper poses a different question: among arms whose *mean cost* is below a threshold $\gamma$, which has the highest expected reward?

Formally: $K$-armed bandit $\boldsymbol{\nu} \in \mathcal{M}^K$ where each $\nu_k$ is a **bivariate** distribution on $(\text{reward}, \text{cost})$ with mean $\boldsymbol{m}(\nu_k) = (m_{k,1}, m_{k,2})$. The answer is:
$$i^*(\boldsymbol{\nu}) = \arg\max_{\{i:\, m_{i,2} \leq \gamma\}} m_{i,1}, \quad \text{or } \texttt{None} \text{ if all arms infeasible}$$

Key novelty: reward and cost can be **dependent** ($\Sigma_{12} \neq 0$). Prior constrained BAI work (Wang et al. 2022, Katz-Samuels & Scott 2019) assumes independence or linear/monotone relationships. Dependence changes $T^*$ and can make uniform sampling outperform independence-assuming strategies.

## Method

**Lower bound (Theorem 2.1, applying Garivier & Kaufmann 2016):** For any $\delta$-correct algorithm:
$$\mathbb{E}_\nu[\tau_\delta] \geq T^*(\boldsymbol{\nu}) \cdot \mathrm{KL}(\delta \| 1 - \delta)$$
$$T^*(\boldsymbol{\nu})^{-1} = \max_{\boldsymbol{w} \in \triangle_K} \min_{\substack{\nu' \in \mathcal{M}^K \\ i^*(\nu') \neq i^*(\boldsymbol{\nu})}} \sum_{k=1}^K w_k \, \mathrm{KL}(\nu_k \| \nu'_k)$$

**Transportation cost interface (Interface 2.1):** The key computational abstraction. Define:
- $c_1(\nu_i, \nu_j, w)$: minimum KL cost to make arm $j$ feasible **and** better than arm $i$
- $c_2(\nu)$: minimum KL cost to flip the feasibility status of $\nu$ across threshold $\gamma$

These two functions suffice to compute $T^*$ and $\boldsymbol{w}^*$ via two nested binary searches:

**Theorem 2.2:** $T^*(\boldsymbol{\nu}) = \sum_{j=1}^K \tilde{w}_j(\tilde{C}^*) / \tilde{C}^*$, where $\tilde{C}^*$ is the unique solution in $[0, \tilde{C}_{\max}]$ of:
$$\sum_{j \neq i^*} \frac{c_{1,i^*}(\nu_{i^*}, \nu_j, \tilde{w}_j(\tilde{C}^*))}{c_{1,j}(\nu_{i^*}, \nu_j, \tilde{w}_j(\tilde{C}^*))} = 1$$

Same computational cost as standard TAS (two binary searches). The `None` case reduces to a simpler 1d thresholding problem.

**Three arm model families** with explicit $c_1, c_2$:
1. $\mathcal{M}_{G,\Sigma}$ — Gaussian with fixed $\Sigma$: $\mathrm{KLinf}(\nu, \boldsymbol{\lambda}) = \frac{1}{2}\|\boldsymbol{\mu} - \boldsymbol{\lambda}\|^2_{\Sigma^{-1}}$
2. $\mathcal{M}_G$ — Gaussian with unknown $\Sigma$: $\mathrm{KLinf}(\nu, \boldsymbol{\lambda}) = \frac{1}{2}\ln(1 + \|\boldsymbol{\mu} - \boldsymbol{\lambda}\|^2_{\Sigma^{-1}})$
3. $\mathcal{M}_B$ — non-parametric on $[0,1]^2$: KLinf via concave linear program

**TaS algorithm:** Plug-in Track-and-Stop — estimates $\boldsymbol{\nu}$ online, tracks $\boldsymbol{w}^*(\hat{\boldsymbol{\nu}}(n))$ via C-Tracking, stops at GLR statistic $\Lambda_n \geq \beta(\delta, n)$. Stopping thresholds (Theorem 3.1):
$$\beta_{G,\Sigma}(\delta, n) = \ln\frac{K}{\delta} + 4\ln\frac{\ln K}{4} + 8\ln(4 + \ln n/2)$$
$$\beta_G(\delta, n) = \ln\frac{K}{\delta} + 2\ln n + 4\ln\ln n + 2\ln\!\left(\ln\frac{K}{\delta} + 2\ln n + 4\ln\ln n\right)$$

**Theorem 3.2 (Asymptotic optimality):**
$$\lim_{\delta \to 0} \frac{\mathbb{E}_\nu[\tau_\delta]}{\ln(1/\delta)} = T^*(\boldsymbol{\nu})$$

## Results

Five-arm Gaussian bandit ($\Sigma = [0.1, 0.05; 0.05, 0.09]$, $\delta = 0.01$, four instances):

| Algorithm | Easy | Hard | All feasible | None feasible |
|---|---|---|---|---|
| TaS-EV (empirical cov.) | 79.4 | 3291 | 199 | 234 |
| TaS (known $\Sigma$) | 76.3 | 3218 | 190 | 223 |
| Oracle ($\boldsymbol{w}^*$ known) | 89.6 | 5398 | 230 | 270 |
| TopTwo-TCI | 68.8 | 2860 | 175 | 242 |
| TaS-1d (ignores cost) | 96.6 | 4816 | 186 | 3293 |

TaS-1d (unconstrained BAI) is catastrophically bad on the None-feasible instance (3293 vs. 223) — it cannot detect infeasibility. TaS-EV (uses empirical covariance) closely matches TaS with known $\Sigma$.

**Impact of dependence:** For two-arm Gaussian with correlation $\rho$, $T^*(\boldsymbol{\nu}_\rho)$ decreases as $\rho$ increases. For large $\rho > 0$, the independence-assuming strategy $T_{w_{\text{ind}}}$ is *worse* than uniform sampling $T_{w_{\text{unif}}}$ — dependence has a first-order effect on optimal proportions that cannot be ignored.

## Strengths & Limitations

**Strengths:** First constrained BAI paper to handle dependent reward-cost distributions in full generality. Transportation cost interface is model-agnostic and computationally clean. All three model families have provably optimal algorithms. The `None` answer (all arms infeasible) is handled natively.

**Limitations:** Fixed-confidence only; no fixed-budget CBAI. Multiple-constraint extension faces exponentially many cases in $c_1$. Stopping threshold for $\mathcal{M}_G$ contains $\ln n$ (vs. $\ln \ln n$ for $\mathcal{M}_{G,\Sigma}$) — sharper threshold requires future work.

## Connections

- [[constrained-bai]] — the novel problem formulation introduced here
- [[Kanarios2024Cost]] — complementary: CABAI minimizes cumulative cost; CBAI constrains mean arm cost. Both use GLR stopping; different lower bounds and optimal proportions.
- [[best-arm-identification]] — CBAI extends fixed-confidence BAI to cost-constrained bivariate arms
- [[cabai]] — contrast: cost-minimization vs. cost-threshold objective
- [[Lardy-Tyron]] — first author
- [[Koolen-Wouter-M]] — senior author; co-developer of mixture martingale tools used in stopping thresholds
- **Extends:** Garivier & Kaufmann 2016 (TAS) to bivariate arm distributions
- **Related:** Wang et al. 2022 (BAI + safety constraints) — assumes independence and linear/monotone cost-reward; this paper removes both restrictions

## Open Questions

- Can the $\mathcal{M}_G$ stopping threshold be tightened from $\ln n$ to $\ln \ln n$ correction?
- Fixed-budget variant of CBAI?
- Multiple cost constraints ($m_{k,j} \leq \gamma_j$ for $j = 1, \ldots, d$) — currently exponential complexity in $d$?
- Non-asymptotic (finite-$\delta$) sample complexity bounds?
