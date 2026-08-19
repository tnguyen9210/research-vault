---
title: "Linear Softmax Policy"
tags: [rlhf, language-model-alignment, policy-parameterization]
introduced_by: [[Foster2025Foundation]]
---

# Linear Softmax Policy

**Definition:** Given a feature map $\phi: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^d$ and a reference policy $\pi_\text{ref}$, the linear softmax policy class is:
$$\Pi = \{\pi_\theta : \theta \in \Theta\}, \quad \pi_\theta(y|x) \propto \pi_\text{ref}(y|x) \cdot \exp\!\left(\beta^{-1}\langle\theta, \phi(x,y)\rangle\right).$$

## Intuition

This is the natural policy class for RLHF: it starts from the reference distribution and tilts it in the direction of high reward, with $\beta > 0$ controlling how far the learned policy can deviate from $\pi_\text{ref}$ (the KL penalty). When $\beta$ is small, the policy can deviate far; when $\beta$ is large, the policy stays close to $\pi_\text{ref}$.

This parameterization corresponds operationally to a deep LM where all but the last layer are frozen (the "lazy kernel" / NTK regime of Malladi et al. 2023).

## Formal Description

The optimal KL-regularized policy under linear softmax is $\pi_{\theta^*}$ where $\theta^* \in \mathbb{R}^d$ satisfies the reward linearity condition: $r^*(x,y) = \langle\theta^*, \phi(x,y)\rangle$. In this case:
$$J_\beta(\pi^*_\beta) - J_\beta(\pi_\theta) = \frac{1}{\beta}\mathbb{E}_{\pi_\theta}\!\left[\langle\theta^* - \theta, \phi(x,y)\rangle^2\right] + \text{lower order}.$$

The KL-regularized objective $J_\beta(\pi) = J(\pi) - \beta \cdot D_\text{KL}(\pi \| \pi_\text{ref})$ induces strong convexity, giving a fast $\tilde{O}(1/\varepsilon^2)$ sample complexity rate (vs. $\tilde{O}(1/\varepsilon)$ for unregularized policies).

For autoregressive sequence models ($\mathcal{Y} = \mathcal{A}^H$), the analogous class is:
$$\pi^{\text{auto}}_{h,\theta}(a_h | x, a_{1:h-1}) \propto \pi_{h,\text{ref}}(a_h | x, a_{1:h-1}) \cdot \exp\!\left(\beta^{-1}\langle\theta_h, \phi_h(x, a_{1:h})\rangle\right).$$
Under this parameterization, the KL-regularized Q-function $Q^*_{h,\beta}(x, a_{1:h})$ is linear in $\phi_h$, enabling efficient DP backward induction (MTSS).

## Key Papers

- [[Foster2025Foundation]] — primary source; establishes computational-statistical tradeoffs for this class; SpannerSampling and MTSS are specialized to it

## Variants & Related Concepts

- [[coverage-coefficient]] — $C_\text{cov}(\pi^*_\beta)$ measures how much $\pi^*_\beta$ deviates from $\pi_\text{ref}$; central computational complexity measure for this class
- [[realizability]] — Assumption 1.1: $\pi^*_\beta \in \Pi$; enables statistical efficiency
- [[contextual-bandits]] — alignment with linear softmax policies is a KL-regularized contextual bandit with a structured policy class

## Current State

Studied as the simplest nontrivial policy class for alignment theory. [[Foster2025Foundation]] gives sharp computational-statistical tradeoffs for this class. Extension to nonlinear/transformer policy classes is a primary open problem.
