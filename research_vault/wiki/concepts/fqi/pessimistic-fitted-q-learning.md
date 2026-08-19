---
title: "Pessimistic Fitted Q-Learning (PFQL)"
tags: [offline-reinforcement-learning, pessimism, fitted-q-iteration, differentiable-function-approximation, variance-awareness]
introduced_by: [[Yin2023Offline]]
---

# Pessimistic Fitted Q-Learning (PFQL)

**Definition:** Backward fitted Q-iteration over a [[differentiable-function-approximation]] class in which each Q-estimate is penalized by a data-driven uncertainty bonus $\Gamma_h(s,a) = \beta\sqrt{\nabla_\theta f(\hat\theta_h,\phi)^\top\Sigma_h^{-1}\nabla_\theta f(\hat\theta_h,\phi)}$, where $\Sigma_h$ is the Gram matrix of parameter gradients on the offline data.

## Intuition

PFQL fuses two well-worn ingredients and gets something neither gives alone.

**[[fitted-q-iteration]]** supplies computational realism: a squared-error regression at each step, solvable by SGD, matching what neural FQI and DQN-style critics actually do. Compare the maxmin objectives that general-function-approximation theory produces, which are not runnable.

**Pessimism** supplies safety under distributional shift. Where the data is thin, the fitted model extrapolates and $Q$ is unreliable — usually upward, since downstream maximization selects large errors (see [[overestimation-bias]]). Subtracting an uncertainty penalty makes the algorithm prefer actions it has evidence for, the offline mirror of optimism-under-uncertainty in online RL.

**The bonus has a clean reading.** Define $m(s,a) := (\nabla_\theta f^\top\Sigma_h^{-1}\nabla_\theta f)^{-1}$. This is the *effective sample size at $(s,a)$ along the gradient direction* $\nabla_\theta f$, and the penalty is exactly $\beta/\sqrt{m(s,a)}$ — the familiar $1/\sqrt{n}$ confidence width, with $n$ replaced by a local, direction-aware sample count. Under linearity $\nabla_\theta f = \phi$ and this is the standard elliptical bonus $\|\phi\|_{\Sigma^{-1}}$.

**One wrinkle absent in the linear case.** $m(s,a)$ is evaluated at the *estimated* $\hat\theta_h$, so the uncertainty measure is only meaningful once $\hat\theta_h$ is close to $\theta^*_h$ — plugging in an arbitrary $\theta$ gives a meaningless, possibly harmful $\Gamma_h$. In linear MDPs $\phi^\top(\Sigma^\text{linear}_h)^{-1}\phi$ carries no such dependence and is always valid. This implicit constraint is a real source of difficulty in the analysis.

## Formal Description

**Algorithm 1 of [[Yin2023Offline]].** Given $\mathcal{D} = \{(s^k_h,a^k_h,r^k_h,s^k_{h+1})\}$, set $\hat V_{H+1}\leftarrow 0$ and for $h = H,\dots,1$:

$$
\hat\theta_h \leftarrow \arg\min_{\theta\in\Theta}\ \sum_{k=1}^K\big[f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]^2 + \lambda\|\theta\|_2^2
$$

$$
\Sigma_h \leftarrow \sum_{k=1}^K \nabla_\theta f(\hat\theta_h,\phi_{h,k})\nabla_\theta^\top f(\hat\theta_h,\phi_{h,k}) + \lambda I_d
$$

$$
\Gamma_h(\cdot,\cdot) \leftarrow \beta\sqrt{\nabla_\theta f(\hat\theta_h,\phi)^\top\Sigma_h^{-1}\nabla_\theta f(\hat\theta_h,\phi)} + \tilde{O}(1/K)
$$

then $\bar Q_h \leftarrow f(\hat\theta_h,\phi) - \Gamma_h$, clip to $[\,\cdot\,, H-h+1]$, and set $\hat\pi_h$ greedy with $\hat V_h$ accordingly. Theorem 3.2 takes $\beta = 8dH\iota$.

> **Line-by-line reading:** [[pfql-algorithm-1]] walks Algorithm 1 one line at a time — what the ridge term does, why the argmin has no closed form, $\Gamma_h$ as an effective-sample-size width, the hyperparameter conditions, the linear/tabular/GLM specializations, and why the pessimism has to sit *inside* the backup rather than at policy extraction.

**Guarantee (Thm 3.2).** Under realizability + Bellman completeness and uniform coverage, with $\epsilon_\mathcal{F}=0$ and $K$ large enough, with probability $1-\delta$:

$$
v^{\pi^*} - v^{\hat\pi} \;\le\; 16dH\sum_{h=1}^H \mathbb{E}_{\pi^*}\Big[\sqrt{\nabla_\theta^\top f(\theta^*_h,\phi)\,\Sigma_h^{\star-1}\,\nabla_\theta f(\theta^*_h,\phi)}\Big]\cdot\iota + \tilde{O}(C'_\text{hot}/K).
$$

### VAFQL — the variance-aware variant

Reweight each residual by an estimated conditional variance $\hat\sigma^2_h(s,a)$, with $\sigma^{\star 2}_h := \max\{1, \text{Var}_{P_h}V^*_{h+1}\}$:

$$
\hat\theta_h \leftarrow \arg\min_{\theta\in\Theta}\ \sum_{k=1}^K \frac{\big[f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]^2}{\hat\sigma^2_h(s^k_h,a^k_h)} + \lambda\|\theta\|_2^2 .
$$

High-noise transitions get down-weighted. Theorem 4.1 replaces $\Sigma^\star_h$ with $\Lambda^\star_h = \sum_k \nabla f\nabla f^\top/\sigma^{\star2}_h + \lambda I_d$ and — crucially — $\beta = 8d\iota$ rather than $8dH\iota$, a **net saving of a factor $H$** since $\text{Var}_{P_h}V^*_{h+1}\le H^2$. On deterministic transitions $\sigma^\star_h\approx0$, $\Lambda^{\star-1}_h\to0$, and the rate improves to $O(1/K)$. Theorem 4.2 gives a matching lower bound up to $\sqrt{d}$.

## Key Papers

- [[Yin2023Offline]] — introduces PFQL and VAFQL; first instance-dependent offline RL guarantee under nonlinear function approximation
- Jin et al. (2021b) — PEVI, the linear-MDP special case PFQL strictly generalizes
- Yin & Wang (2021) — VPVI, the tabular predecessor
- Yin et al. (2022) — variance-aware linear offline RL, recovered by VAFQL up to $\sqrt{d}$
- Ernst et al. (2005), Riedmiller (2005) — fitted Q-iteration and neural FQI, the practical lineage
- Xie et al. (2021a) — PSPI: weaker (single-policy) coverage but a slower $O(n^{-1/3})$ rate and a maxmin objective

## Variants & Related Concepts

- [[pfql-algorithm-1]] — line-by-line walkthrough of Algorithm 1
- [[fitted-q-iteration]] — the base template; PFQL is FQI with an uncertainty penalty subtracted before the greedy step
- [[differentiable-function-approximation]] — the class PFQL is analyzed over
- [[instance-dependent-bounds]] — the guarantee type it achieves
- [[implicit-q-learning]] — the opposing design in offline RL: rather than penalize uncertainty at out-of-sample actions, never evaluate them. Empirically strong, theoretically much weaker
- [[importance-weighting]] / [[Ryu2025Improved]] — pessimism via betting-based LCBs, applied to offline policy *selection* rather than learning
- [[overestimation-bias]] — the failure mode pessimism corrects
- [[extrapolation-error]] — the offline-specific failure the penalty is really aimed at; PFQL's $\beta\|\nabla_\theta f\|_{\Sigma_h^{-1}}$ is a computable instance of the generic width $b(s,a)$
- **Model-based/model-free bridge:** FQI is a batch Q-learning update but also an instantiation of approximate value iteration, so PFQL unifies the value-iteration (PEVI, VPVI) and fitted-Q lineages

## Current State

The reference theoretical algorithm for offline RL beyond linear models. Its guarantee is the sharpest available in that setting and comes with a near-matching lower bound. What it does not offer: any computational claim (the fitted-Q step is a nonconvex argmin, analyzed as exactly solved), tolerance of overparameterized classes (uniform coverage fails under non-identifiability), or a $\sqrt{d}$ rate. Empirically PFQL is not run — the practical offline RL field uses [[implicit-q-learning]], CQL, and TD3+BC, none of which carry instance-dependent guarantees. Closing that theory/practice gap is the standing challenge in this line.
