---
title: "Realizability"
tags: [contextual-bandits, function-approximation, statistical-learning]
introduced_by: [[SimchiLevi2022Bypassing]]
---

# Realizability

**Definition:** The assumption that the true reward function $f^*(x,a) = \mathbb{E}[r(a) \mid x]$ belongs to the hypothesis class $\mathcal{F}$ used by the learner, i.e., $f^* \in \mathcal{F}$.

## Intuition
Realizability says the model is well-specified: the function class $\mathcal{F}$ is rich enough to contain the ground truth. This is a strong but standard assumption in supervised learning (e.g., linear regression assuming a linear data-generating process). In contextual bandits, it lets the learner treat offline regression errors as directly informative about the true reward, enabling tighter guarantees without needing to account for approximation error.

## Formal Description
Given function class $\mathcal{F} = \{f : \mathcal{X} \times \mathcal{A} \to [0,1]\}$, the realizability assumption is:

$$f^*(x,a) := \mathbb{E}[r_t(a) \mid x_t = x] \in \mathcal{F} \quad \forall\, x \in \mathcal{X},\, a \in \mathcal{A}$$

In the realizable setting, the learner competes with the globally optimal policy $\pi_{f^*} = \arg\max_a f^*(x,a)$, which is guaranteed to be the best possible policy. In the agnostic setting, the learner only competes with the best policy *in* a policy class $\Pi$, which may be strictly worse.

## Key Papers
- [[SimchiLevi2022Bypassing]] — realizability enables FALCON to achieve optimal regret with $O(\log T)$ offline oracle calls; the implicit policy distribution argument relies on $f^* \in \mathcal{F}$
- [[Qin2026Taming]] — does not require realizability for the main result; treats misspecification as an extension (Appendix B)
- [[Yin2023Offline]] — the RL form (Assumption 2.1): realizability of $Q^*_h$ in the differentiable class *plus* Bellman completeness $\sup_V\inf_{f\in\mathcal{F}}\|f - \mathcal{P}_h(V)\|_\infty \le \epsilon_\mathcal{F}$. In sequential problems realizability alone is insufficient — the class must also be closed under Bellman backups, or errors compound across the horizon

## Variants & Related Concepts
- **Agnostic setting** — no realizability; algorithm competes with best in-class policy; typically requires stronger oracle (online regression or cost-sensitive classification)
- **Misspecification** — approximate realizability; $f^* \approx f$ for some $f \in \mathcal{F}$ with bounded error $\varepsilon$; OE2D ([[Qin2026Taming]]) handles this
- [[offline-regression-oracle]] — realizability makes offline regression error directly bound bandit regret
- **Bellman completeness** — the sequential strengthening required in RL ([[Yin2023Offline]], Assumption 2.1); strictly stronger than realizability and not implied by it
- [[differentiable-function-approximation]] — a structured class over which realizability plus completeness yields [[instance-dependent-bounds]] rather than worst-case ones

## Current State
Realizability is a standard assumption in the realizable contextual bandit literature. Whether it can be dropped while maintaining offline-oracle efficiency and $O(\log T)$ calls is largely resolved by [[Qin2026Taming]] via DOEC, at the cost of more complex per-context optimization.
