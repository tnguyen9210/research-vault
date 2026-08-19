---
title: "Coverage Coefficient"
tags: [rlhf, active-exploration, computational-complexity, language-model-alignment]
introduced_by: [[Foster2025Foundation]]
---

# Coverage Coefficient

**Definition:** The coverage coefficient $C_\text{cov}(\pi^*_\beta)$ measures how well the reference policy $\pi_\text{ref}$ covers the optimal KL-regularized policy $\pi^*_\beta$:
$$C_\text{cov}(\pi^*_\beta) := \sup_{x \in \mathcal{X},\, y \in \mathcal{Y}} \frac{\pi^*_\beta(y|x)}{\pi_\text{ref}(y|x)}.$$

## Intuition

If the base model $\pi_\text{ref}$ places very low probability on the optimal response $y^*$, then $C_\text{cov}$ is large — the algorithm must either search a vast space or get lucky. It quantifies the "hidden knowledge" problem: how much of what matters is already encoded in the pre-trained model.

A small $C_\text{cov}$ means $\pi_\text{ref}$ already "knows" roughly where good responses are; a large $C_\text{cov}$ means the optimal responses are rare under $\pi_\text{ref}$, making active exploration expensive.

## Formal Description

In the [[linear-softmax-policy]] parameterization, since $\pi^*_\beta(y|x) \propto \pi_\text{ref}(y|x) \cdot \exp(\beta^{-1}\langle\theta^*, \phi(x,y)\rangle)$, we have:
$$C_\text{cov}(\pi^*_\beta) \leq \exp(R_\text{max}/\beta),$$
so coverage is always at most exponential in $R_\text{max}/\beta$. When rewards are not too large relative to KL regularization, $C_\text{cov}$ can be $O(1)$.

The *token-level* counterpart for autoregressive policies is the conditional coverage coefficient:
$$C_\text{cond}(\pi^*_\beta) := \max_{h \in [H]} \sup_{x,\, a_{1:h}} \frac{\pi^*_{h,\beta}(a_h | x, a_{1:h-1})}{\pi_{h,\text{ref}}(a_h | x, a_{1:h-1})}.$$
This satisfies $C_\text{cond}(\pi^*_\beta) \leq 2$ while $C_\text{cov}(\pi^*_\beta)$ can be as large as $2^H$ — an exponential gap.

## Key Papers

- [[Foster2025Foundation]] — introduces $C_\text{cov}$ as the central computational complexity measure; proves it lower bounds $T_\text{comp}$ (Thm 2.1); SpannerSampling achieves $T_\text{comp} = \tilde{O}(C_\text{cov})$ per round (Thm 3.1); MTSS replaces $C_\text{cov}$ with $C_\text{cond}$ (Thm 5.1)

## Variants & Related Concepts

- [[linear-softmax-policy]] — the policy class where $C_\text{cov}$ is studied
- [[realizability]] — analogous assumption: $\pi^*_\beta \in \Pi$; coverage is a *computational* condition whereas realizability is *statistical*
- [[contextual-bandits]] — in standard contextual bandits, importance weights play a similar coverage role; $C_\text{cov}$ is the LM alignment analogue
- [[dec]] — DEC and DOEC measure statistical complexity of bandit algorithms; $C_\text{cov}$ measures computational complexity in the sampling oracle framework
- [[implicit-q-learning]] — offline RL's support condition $\pi_\beta(a|s) > 0$ is the binary, density-free analogue of $C_\text{cov}$: [[Kostrikov2022Offline]] proves convergence to the support-constrained optimum but cannot say how large $\tau$ must be, precisely because a support indicator carries no density information

## Current State

Active area. [[Foster2025Foundation]] establishes $C_\text{cov}$ as the right complexity measure for the sampling oracle framework. Empirical evidence (Brown et al. 2024; Snell et al. 2024; Wu et al. 2024) suggests existing base models have favorable coverage on tasks of interest. Open: can $C_\text{cond}$ be estimated efficiently from $\pi_\text{ref}$ before running alignment? A second opening is transfer in the other direction — importing a density-weighted coverage quantity into offline RL to replace the support condition in [[Kostrikov2022Offline]].
