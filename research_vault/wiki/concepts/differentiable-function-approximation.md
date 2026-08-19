---
title: "Differentiable Function Approximation (DFA)"
tags: [function-approximation, offline-reinforcement-learning, sample-complexity, nonlinear-models]
introduced_by: [[Yin2023Offline]]
---

# Differentiable Function Approximation (DFA)

**Definition:** A parametric function class $\mathcal{F} = \{f(\theta,\phi(\cdot,\cdot)) : \theta\in\Theta\subset\mathbb{R}^d\}$ over a feature map $\phi:\mathcal{S}\times\mathcal{A}\to\Psi\subset\mathbb{R}^m$, with $\Theta,\Psi$ compact and $f$ three-times differentiable in $\theta$ with jointly continuous derivatives — expressive enough to include nonlinear models, structured enough that gradient geometry supports [[instance-dependent-bounds]].

## Intuition

RL theory has been stuck between two poles. Linear models are analyzable because least squares has a closed form, but describe nothing practitioners run. General function approximation covers everything but, having no structure, yields only worst-case bounds that cannot distinguish an easy MDP from a hard one.

DFA picks the structure that a worst-case analysis discards: **local derivative information**. If $f$ is smooth in $\theta$, then near the true parameter the model behaves like its linearization $\nabla_\theta f$, and all the machinery built for linear models — Gram matrices, elliptical confidence sets, effective sample size — transfers, with $\phi$ replaced by $\nabla_\theta f(\theta,\phi)$. Nonlinearity survives in the curvature terms, which then have to be controlled rather than assumed away.

The class is a strict generalization of the standard choices:

| Instantiation | Recovers |
|---|---|
| $f(\theta,\phi) = \langle\theta,\phi\rangle$ | linear MDPs |
| $\phi$ one-hot | tabular MDPs |
| $f(\theta,\phi) = g(\langle\theta,\phi\rangle)$, $g$ a link | generalized linear models |

## Formal Description

**Definition 1.1 of [[Yin2023Offline]].** With $\mathcal{S},\mathcal{A}$ arbitrary, $\phi:\mathcal{S}\times\mathcal{A}\to\Psi\subset\mathbb{R}^m$, $\Theta\subset\mathbb{R}^d$, both compact:

$$
\mathcal{F} := \{f(\theta,\phi(\cdot,\cdot)) : \mathcal{S}\times\mathcal{A}\to\mathbb{R},\ \theta\in\Theta\}
$$

such that (i) $f(\theta,\phi)$ is three-times differentiable in $\theta$ for any $\phi$, and (ii) $f, \partial_\theta f, \partial^2_{\theta\theta}f, \partial^3_{\theta\theta\theta}f$ are jointly continuous in $(\theta,\phi)$.

Compactness plus continuity yields constants $C_\Theta, B_\mathcal{F}, \kappa_1,\kappa_2,\kappa_3$ bounding $\|\theta\|_2$, $|f|$, and the first three derivatives.

**The induced instance measure.** Writing $g_\theta(s,a) := \nabla_\theta f(\theta,\phi(s,a))$ and $\Sigma_h := \sum_k g_\theta(s^k_h,a^k_h)g_\theta(s^k_h,a^k_h)^\top + \lambda I_d$ (a Fisher information matrix of the observed data), learning hardness is governed by

$$
\sum_{h=1}^H \mathbb{E}_{\pi^*,h}\Big[\sqrt{g_\theta(s,a)^\top \Sigma_h^{-1} g_\theta(s,a)}\Big].
$$

Its reciprocal-square, $m(s,a) := (g_\theta^\top\Sigma_h^{-1}g_\theta)^{-1}$, is the **effective sample size at $(s,a)$ along the gradient direction** — the natural generalization of counting visits in the tabular case and of the elliptical norm $\|\phi\|_{\Sigma^{-1}}$ in the linear case.

**Where nonlinearity bites.** In a Taylor expansion of the fitted-Q stationarity condition, the curvature term $\Delta\Sigma^s_h = \sum_k[\text{residual}_k]\cdot\nabla^2_{\theta\theta}f$ appears and breaks positive definiteness of the expansion matrix. Under linearity $\nabla^2_{\theta\theta}f\equiv0$ and it vanishes; handling it is the technical content of [[Yin2023Offline]].

**The identifiability caveat.** Guarantees over DFA require a coverage condition of the form $\mathbb{E}_{\mu}[(f(\theta_1,\phi)-f(\theta_2,\phi))^2]\ge\kappa\|\theta_1-\theta_2\|_2^2$, which says distinct parameters must produce measurably distinct functions. **Overparameterized neural networks violate this by construction** — permutation and scaling symmetries make distinct $\theta$ realize the identical function. So despite the "$\theta$ is the network weights" motivation, the theorems' honest headline instance is GLM, not deep networks.

## Key Papers

- Zhang et al. (2022a) — introduces differentiable function classes for off-policy *evaluation*; obtains asymptotic guarantees via Prohorov's theorem
- [[Yin2023Offline]] — adopts DFA for policy *learning*; first non-asymptotic, instance-dependent offline RL bound over a nonlinear class, via [[pessimistic-fitted-q-learning]]
- Jin et al. (2021b) — PEVI for linear MDPs; the special case $f = \langle\theta,\phi\rangle$
- Chen & Jiang (2019), Xie et al. (2021a) — general function approximation; the weaker-assumption, worst-case-bound alternative

## Variants & Related Concepts

- [[instance-dependent-bounds]] — what the differentiable structure is *for*
- [[pessimistic-fitted-q-learning]] — the algorithm analyzed over this class
- [[realizability]] — required alongside Bellman completeness (Assumption 2.1)
- [[coverage-coefficient]] — DFA's uniform coverage is a coverage condition depending jointly on the MDP and the function class, unlike pure concentrability which depends on the MDP alone
- **Eluder dimension** (Russo & Van Roy 2013; Jin et al. 2021a) — an alternative structural handle; its tractable instances are still essentially linear-in-features, which is the comparison [[Yin2023Offline]] draws
- **Bilinear classes** (Du et al. 2021), **linear Bellman complete** (Zanette et al. 2020) — other structural generalizations of linear models

## Current State

Young and under-explored. [[Yin2023Offline]] establishes offline policy learning; OPE came earlier (Zhang et al. 2022a). Online RL, reward-free exploration, and representation learning under DFA are all open, and the authors explicitly flag them. The two live obstacles are the $d$-dependence of current bounds (versus $\sqrt{d}$ in the linear case, an artifact of covering arguments that may or may not be removable) and the identifiability condition, which excludes the overparameterized regime the framework is most often motivated by. A quotient formulation — coverage stated on function space or on parameter equivalence classes rather than on $\theta$ — is the obvious missing idea.
