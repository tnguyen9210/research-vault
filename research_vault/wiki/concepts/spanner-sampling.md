---
title: "SpannerSampling"
tags: [rlhf, active-exploration, language-model-alignment, algorithms]
introduced_by: [[Foster2025Foundation]]
---

# SpannerSampling

**Definition:** A two-phase online alignment algorithm that achieves optimal data efficiency and computational efficiency (up to the [[coverage-coefficient]] lower bound) by using inference-time rejection sampling rather than training-time policy updates.

## Intuition

The key idea is *improper exploration*: SpannerSampling never queries the reward oracle with samples from a linear softmax policy $\pi_\theta \in \Pi$. Instead it uses $\pi_\text{ref}$ to build a small *spanner* — a compact set of response pairs that covers the feature space in directions relevant to $\pi^*_\beta$ — then explores in directions *not* already covered. This breaks the proper-exploration barrier (Thm 4.1 in [[Foster2025Foundation]]).

The truncation in the exploration phase is the critical mechanism: it restricts exploration to directions outside the spanner's coverage, ensuring that only $\tilde{O}(C_\text{cov}(\pi^*_\beta))$ oracle calls suffice per round via rejection sampling from $\pi_\text{ref}$.

## Formal Description

**Phase 1 — Spanner construction.** Let $\varphi(x,y_1,y_2) = \phi(x,y_1) - \phi(x,y_2)$ be the *relative feature*. For each prompt $x^t$, sample i.i.d. pairs $(y_1^{t,i}, y_2^{t,i}) \sim \pi_\text{ref}(\cdot|x^t)^{\otimes 2}$ until $\|\varphi(x^t, y_1^{t,i}, y_2^{t,i})\|_{\Sigma^{-1}_\text{span}} > \nu$ (i.e., the pair is "informative"). Accept it, query rewards $(r_1^{t,i}, r_2^{t,i})$, update $\Sigma_\text{span} \leftarrow \Sigma_\text{span} + \varphi\varphi^\top$, and move to the next prompt. The spanner $\Psi_\text{span}$ has size $|\Psi_\text{span}| \leq \text{poly}(d, \nu^{-1})$.

**Phase 2 — Exploration.** For each round $t$:
1. Fit $\theta^t = \arg\min_{\theta} \sum_{(x,y_1,y_2,r_1,r_2) \in \mathcal{D}^t} (\langle\theta, \varphi(x,y_1,y_2)\rangle - (r_1 - r_2))^2$.
2. Define truncated reward: $r^t(x,y,y') = \langle\theta^t, \varphi(x,y,y')\rangle \cdot \mathbf{1}[\|\varphi(x,y,y')\|_{\Sigma^{-1}_\text{span}} \leq \nu]$.
3. Sample $y_1^t \sim \text{SoftmaxSampler}_{\beta, M_\text{rej}, \delta_\text{rej}}(r^t(\cdot, \cdot, y_2^t); x^t, \pi_\text{ref})$ via rejection sampling from $\pi_\text{ref}$.

SoftmaxSampler draws $N = 4M\log(4\delta^{-1})$ candidates from $\pi_\text{ref}$ and accepts with probability $\exp(\beta^{-1} f(y_i)) / \hat{Z}_M$. This requires $\tilde{O}(C_\text{cov}(\pi^*_\beta))$ samples from $\pi_\text{ref}$ per call.

**Guarantees (Thm 3.1):**
$$T_\text{data}(\varepsilon,\delta) = \tilde{O}\!\left(\frac{R_\text{max}^2}{\beta} \cdot \frac{d^2 \log^2(\delta^{-1})}{\min\{\varepsilon,\beta\}}\right), \quad T_\text{comp}(\varepsilon,\delta) = \tilde{O}\!\left(C_\text{cov}(\pi^*_\beta) \cdot \frac{R_\text{max}^2}{\beta^2} \cdot T_\text{data}^2\right).$$

## Key Papers

- [[Foster2025Foundation]] — introduces SpannerSampling; proves Thm 3.1 (upper bound) and Thm 2.1 (matching $C_\text{cov}$ lower bound); also introduces MTSS (multi-turn extension)

## Variants & Related Concepts

- [[coverage-coefficient]] — governs $T_\text{comp}$; SpannerSampling achieves the lower bound in $C_\text{cov}$
- [[linear-softmax-policy]] — the policy class for which SpannerSampling is designed
- [[monte-carlo-tree-search]] — MTSS (MultiTurnSpannerSampling) is the token-level DP extension; structurally analogous to backward induction in MDPs; [[Dam2024Power]] studies MCTS convergence in stochastic environments
- [[test-time-scaling]] — SpannerSampling is a formal instance of inference-time exploration beating training-time-only methods

## Current State

SpannerSampling is the first algorithm matching the $C_\text{cov}$ computational lower bound. Its polynomial dependence on $C_\text{cov}$ in $T_\text{comp}$ may not be tight. Extension to nonlinear policies is the primary open question.
