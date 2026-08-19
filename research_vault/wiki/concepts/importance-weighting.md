---
title: "Importance Weighting"
tags: [contextual-bandits, offline-learning, statistics]
introduced_by: [[Ryu2025Improved]]
---

# Importance Weighting

**Definition:** Importance weighting (IW) reweights samples collected under a behavior policy $\pi_\text{ref}$ to estimate the expected reward of a different target policy $\pi$, using the density ratio $w_t^\pi = \pi(a_t|x_t)/\pi_\text{ref}(a_t|x_t)$ as a correction factor.

## Intuition

When we collect data using $\pi_\text{ref}$ but want to evaluate $\pi$, actions that $\pi$ prefers but $\pi_\text{ref}$ rarely takes are underrepresented. IW compensates by upweighting those samples in proportion to how much more likely $\pi$ would have chosen them.

## Formal Description

Given offline log $D_n = \{(x_t, a_t, r_t)\}_{t=1}^n$ with $(x_t, a_t, r_t) \sim p(x)\pi_\text{ref}(a|x)p(r|a,x)$, define:

$$w_t^\pi \triangleq \frac{\pi(a_t|x_t)}{\pi_\text{ref}(a_t|x_t)}, \qquad \tilde{r}_t^\pi \triangleq w_t^\pi r_t.$$

The IW estimator of policy value $\mu(\pi) = \mathbb{E}_{x,a \sim p(x)\pi(a|x)}[r(x,a)]$ is:

$$\hat{\mu}_n^\text{IW}(\pi) \triangleq \frac{1}{n}\sum_{t=1}^n \tilde{r}_t^\pi.$$

This estimator is unbiased: $\mathbb{E}[\hat{\mu}_n^\text{IW}(\pi)] = \mu(\pi)$. However, $w_t^\pi$ is unbounded when $\pi_\text{ref}(a_t|x_t) \to 0$, causing high or infinite variance.

## Key Papers

- [[Ryu2025Improved]] — applies betting-based LCB to $\tilde{r}_t^\pi$; proves first finite-sample guarantee for unbounded IW rewards; introduces freezing to control variance via score-function pessimism

## Variants & Related Concepts

- **Doubly Robust (DR) estimator** — augments IW with a reward model to reduce variance; unbiased if either the model or the weights are correct
- **Clipped IW** — truncates $w_t^\pi$ at a threshold; biased but variance-controlled
- **Logarithmic smoothing** (Sakhi et al. 2024) — uses $\phi^\text{LS}(x) = \ln(1+x)$ as the score function in the pessimistic objective
- **Freezing** — $\phi^\text{freeze}(x) = \ln(1 + x \cdot \mathbf{1}\{x \leq 1\})$; zeros out large IW samples; reduces variance at cost of bias, preferred in small-data regimes
- [[contextual-bandits]] — offline policy optimization is the primary application of IW in this vault
- [[pessimism-principle]] — applying pessimism (select policy with highest LCB on $\mu(\pi)$) to IW estimates yields variance-adaptive guarantees
- [[pessimistic-fitted-q-learning]] — the sequential-RL counterpart: the same pessimism principle, but the uncertainty is an elliptical/gradient-geometry penalty on the fitted Q-function rather than a concentration bound on IW estimates

## Current State

Active research area. Key open problems: achieving doubly-robust guarantees with variance adaptation; removing bounded-probability-ratio assumptions; matching selection and learning bounds in the same framework. [[Ryu2025Improved]] represents the current state of the art for unbounded IW rewards.
