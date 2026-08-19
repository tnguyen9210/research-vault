---
title: "Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing"
authors: [J. Jon Ryu, Jeongyeol Kwon, Benjamin Koppe, Kwang-Sung Jun]
year: 2025
venue: COLT
arxiv: "2502.10826"
tags: [contextual-bandits, offline-learning, importance-weighting, pessimism, confidence-bounds]
source: raw/papers/Ryu2025Improved.pdf
---

# Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing

**TL;DR:** PUB achieves parameter-free, variance-adaptive off-policy selection by applying a betting-based LCB to unbounded importance-weighted rewards; freezing, a score-function variant for policy learning, reduces variance and consistently wins in small-data regimes.

## Problem

Given an offline log $D_n = \{(x_t, a_t, r_t)\}_{t=1}^n$ collected under a fixed behavior policy $\pi_\text{ref}(a|x)$, find a policy $\hat{\pi}$ minimizing offline regret:

$$\mathrm{Reg}_n(\hat{\pi}) \triangleq \mu(\pi^*) - \mu(\hat{\pi}), \quad \mu(\pi) \triangleq \mathbb{E}_{(x,a,r) \sim p(x)\pi(a|x)p(r|a,x)}[r].$$

The central challenge: the [[importance-weighting|IW estimator]] $\hat{\mu}_n^\text{IW}(\pi) = \frac{1}{n}\sum_t w_t^\pi r_t$ with $w_t^\pi = \pi(a_t|x_t)/\pi_\text{ref}(a_t|x_t)$ is unbiased but has unbounded variance when $\pi$ deviates from $\pi_\text{ref}$. Existing pessimism methods either lack finite-sample efficiency guarantees or require hyperparameter tuning.

## Method

**UP-LCB for $[0,\infty)$-valued random variables.** The core technical contribution. For a nonneg process $(Y_t)$ with unknown mean $\mu$, construct a hypothetical one-sided stock market $\boldsymbol{x}_t(\nu) = [Y_t/\nu,\; 1]^\top$ and let Cover's Universal Portfolio (UP) act as the betting strategy. The resulting wealth process $W_n^\text{UP}(Y_{1:n}; \nu)$ defines a time-uniform LCB:

$$\hat{\mu}_\text{UP}^{(\delta)}(Y_{1:n}) \triangleq \min\!\left\{\nu > 0 : W_n^\text{UP}(Y_{1:n};\nu) \leq \frac{1}{\delta}\right\}.$$

UP tracks the best Constantly Rebalanced Portfolio (CRP) up to a $\sqrt{\pi(n+1)}$ factor, enabling automatic adaptation to the underlying variance without knowing it in advance.

**PUB (Pessimism via semi-Unbounded-coin-Betting) — off-policy selection.** Apply UP-LCB to the IW rewards $\{\tilde{r}_t^\pi\}_{t=1}^n$ for each candidate policy and select the policy with the highest LCB:

$$\hat{\pi}_\text{UP} \triangleq \arg\max_{\pi \in \Pi}\; \hat{\mu}_\text{UP}^{(\delta)}(\tilde{r}_{1:n}^\pi).$$

No hyperparameter tuning required; Cover's UP automatically finds the optimal betting fraction.

**Score-function framework — off-policy learning.** For large (possibly infinite) $\Pi$, solve:

$$\hat{\pi}_n \triangleq \arg\max_{\pi \in \Pi} \sum_{t=1}^n \phi(\beta \tilde{r}_t^\pi), \tag{1.3}$$

where $\phi: \mathbb{R}_+ \to \mathbb{R}$ is a *score function* satisfying Assumption 4.1 (a mild sandwiching condition). Three instances:
- **Logarithmic smoothing** (Sakhi et al. 2024): $\phi^\text{LS}(x) = \ln(1+x)$
- **Clipping**: $\phi^\text{clip}(x) = \ln(1 + (x \wedge 1))$
- **Freezing**: $\phi^\text{freeze}(x) = \ln(1 + x \cdot \mathbf{1}\{x \leq 1\})$ — zeros out large IW samples entirely

## Results

**Selection (Theorem 3.3):** With probability $\geq 1 - 2\delta$, letting $F_n^{(\delta)} \triangleq \ln\frac{\sqrt{\pi(n+1)}}{\delta^2}$:

$$\mu(\pi^*) - \mu(\hat{\pi}) \leq 2\sqrt{\frac{F_n^{(\delta')}}{n}\, \mathbb{W}_{b_n^{(\delta')}}[\tilde{r}_1^{\pi^*}]},$$

where $\mathbb{W}_b[Y] \triangleq \mathbb{E}\!\left[\frac{(Y - \mathbb{E}[Y])^2}{1 + b\frac{Y - \mathbb{E}[Y]}{\mathbb{E}[Y]}}\right]$ is the $b$-smoothed variance. This is strictly tighter than prior bounds scaling with raw second moment $1 + \mathbb{E}[(\tilde{r}_1^{\pi^*})^2]$ (Gabbianelli et al. 2024, Sakhi et al. 2024).

**Learning (Theorem 4.3):** For any score function satisfying Assumption 4.1:

$$\mu(\pi^*) - \mu(\hat{\pi}_n) \leq \beta \mathbb{E}\!\left[\frac{(\tilde{r}^{\pi^*})^2}{c_1 + c_2 \tilde{r}^{\pi^*}}\right] + \frac{2}{\beta n}\ln\frac{|\Pi|}{\delta} - F_\beta(\phi),$$

where $F_\beta(\phi) \geq 0$ is the *negative influence* of $\phi$. The bound recovers Logarithmic Smoothing (LS) as a special case and satisfies $F_\beta(\phi^\text{freeze}) \geq F_\beta(\phi^\text{clip}) \geq F_\beta(\phi^\text{LS})$, meaning freezing and clipping can only improve the guarantee.

**Experiments:** PUB is best or statistically indistinguishable from best across all 9 dataset/size combinations (PenDigits, SatImage, JPVowel). LS+freezing consistently outperforms plain LS in small-data ($n = 0.01\times$ dataset) regime.

## Strengths & Limitations

**Strengths:** Parameter-free variance adaptation (key practical advantage); first LCB with finite-sample guarantee for $[0,\infty)$-valued random variables; general score-function framework unifies prior approaches.

**Limitations:** UP incurs a $\log n$ overhead in the confidence bound width; optimal rate for doubly robust estimators is unknown; learning bound (Thm 4.3) does not yet match the selection bound (Thm 3.3) in terms of variance term.

## Connections

- [[importance-weighting]] — IW estimator $\tilde{r}_t^\pi = w_t^\pi r_t$ is the central primitive
- [[contextual-bandits]] — offline variant: policy class fixed, no further environment interaction
- [[offline-oracle-efficient-bandits]] — adjacent line: those results minimize online regret via offline oracle calls; this paper optimizes from a fixed log
- [[SimchiLevi2022Bypassing]] — FALCON also uses offline data but addresses regret in an online setting under realizability; this paper requires neither realizability nor online interaction
- **Shares the pessimism principle:** [[Yin2023Offline]] — PFQL applies a lower-confidence penalty to offline *policy learning* in sequential MDPs, with the uncertainty measured by gradient geometry rather than betting-based IW concentration; both deliver [[instance-dependent-bounds]] adapted to the data at hand
- [[instance-dependent-bounds]] — PUB's variance adaptivity is the bandit-side instance of this guarantee type

## Open Questions

- Can the learning bound be tightened to match the selection bound (scale with $\sqrt{\bar{\mathbb{V}}(\pi^*)}$ rather than raw second moment)?
- Are similar or stronger bounds achievable for doubly robust (DR) estimators?
- Can the $\log n$ factor in PUB's guarantee be removed with additional assumptions (e.g., known variance)?
- Is there an analogue of freezing for infinite-horizon RL?
