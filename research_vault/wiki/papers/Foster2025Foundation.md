---
title: "Is a Good Foundation Necessary for Efficient Reinforcement Learning? The Computational Role of the Base Model in Exploration"
authors: [Dylan J. Foster, Zakaria Mhammedi, Dhruv Rohatgi]
year: 2025
venue: arXiv
arxiv: "2503.07453"
tags: [rlhf, active-exploration, computational-complexity, linear-softmax, language-model-alignment]
source: raw/papers/Foster2025Foundation.pdf
---

# Is a Good Foundation Necessary for Efficient Reinforcement Learning?

**TL;DR:** Proves that coverage of the base policy $\pi_\text{ref}$ is *necessary* for computationally efficient active exploration in LM alignment, then gives the first algorithm (SpannerSampling) that achieves the resulting lower bound via inference-time computation — a regime that training-time interventions like DPO/XPO cannot match under ETH.

## Problem

Online LM alignment requires *active exploration* — deliberately generating diverse responses to gather informative reward signal. Existing algorithms (OnlineDPO, XPO) are either data-inefficient (passive exploration) or computationally intractable (require enumerating the exponentially large response space $\mathcal{Y}$). The paper asks: **how can access to the pre-trained base model $\pi_\text{ref}$ improve computational efficiency, and what are the fundamental limits?**

## Method

**Setup.** Policy class: *linear softmax policies* $\pi_\theta(y|x) \propto \pi_\text{ref}(y|x) \cdot \exp(\beta^{-1}\langle\theta, \phi(x,y)\rangle)$, where $\phi: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^d$ is a feature embedding and $\beta > 0$ is a KL-regularization parameter. The objective is KL-regularized reward: $J_\beta(\pi) = J(\pi) - \beta \cdot D_\text{KL}(\pi \| \pi_\text{ref})$. The learner interacts via a *sampling oracle* — either weak (sample $y \sim \pi_\text{ref}(\cdot|x)$) or strong (sample $y \sim \pi_\theta(\cdot|x)$). Complexity is measured by $T_\text{data}$ (reward oracle calls) and $T_\text{comp}$ (sampling oracle calls).

**Coverage coefficient.** Define:
$$C_\text{cov}(\pi^*_\beta) := \sup_{x \in \mathcal{X},\, y \in \mathcal{Y}} \frac{\pi^*_\beta(y|x)}{\pi_\text{ref}(y|x)}.$$
This measures how much the optimal policy deviates from the base model — the "hidden knowledge" problem.

**Result 1 — Coverage is computationally necessary (Thm 2.1).** For any online alignment algorithm:
$$T_\text{comp}(\varepsilon, \delta) \geq \Omega\!\left(\min\!\left\{e^{\beta^2 d/2},\, e^{\beta^{-1}/2},\, C^*\right\}\right)$$
unless $T_\text{data}(\varepsilon, \delta) \geq |\mathcal{Y}|/8$. Informally: if the base model doesn't cover $\pi^*_\beta$, any algorithm must either enumerate the response space or make exponentially many oracle queries.

**Result 2 — SpannerSampling (Thm 3.1).** Two-phase algorithm using only the *weak* oracle:

- *Spanner phase*: Sample pairs $(y_1, y_2) \sim \pi_\text{ref}(\cdot|x)^{\otimes 2}$; accept if $\|\varphi(x,y_1,y_2)\|_{\Sigma^{-1}_\text{span}} > \nu$ where $\varphi(x,y_1,y_2) = \phi(x,y_1) - \phi(x,y_2)$ is the *relative feature*. Accumulate until the second moment matrix $\Sigma_\text{span}$ covers the feature space in $\pi^*_\beta$-weighted directions. The resulting spanner $\Psi_\text{span}$ has size $\text{poly}(d, \nu^{-1})$ regardless of $T_\text{span}$.

- *Exploration phase*: Fit $\theta^t$ via least squares on the accumulated dataset; then use a *truncated softmax policy* $\bar\pi_{\theta^t}(y|x) \propto \pi_\text{ref}(y|x) \cdot \exp(\beta^{-1}\langle\theta^t, \varphi(x,y,y')\rangle) \cdot \mathbf{1}[\|\varphi(x,y,y')\|_{\Sigma^{-1}_\text{span}} \leq \nu]$, which zeroes out directions already covered by the spanner. Sampling from $\bar\pi_{\theta^t}$ uses rejection sampling from $\pi_\text{ref}$ via SoftmaxSampler, costing $\tilde{O}(C_\text{cov}(\pi^*_\beta))$ oracle queries per round.

Guarantees:
$$T_\text{data}(\varepsilon,\delta) = \tilde{O}\!\left(\frac{R_\text{max}^2}{\beta}\right) \cdot \frac{d^2 \log^2(\delta^{-1})}{\min\{\varepsilon,\beta\}}, \qquad T_\text{comp}(\varepsilon,\delta) = \tilde{O}\!\left(C_\text{cov}(\pi^*_\beta) \cdot \frac{R_\text{max}^2}{\beta^2}\right) \cdot T_\text{data}^2(\varepsilon,\delta).$$
$T_\text{comp}$ matches the lower bound; $T_\text{data}$ achieves a fast $\frac{1}{\varepsilon^2}$ rate (vs. $\frac{1}{\varepsilon}$ for XPO) due to strong convexity from regularization.

**Result 3 — Training-time interventions cannot be efficient (Thm 4.1).** A *proper* algorithm is one that always queries the reward oracle with $y \sim \pi_{\theta^t}(\cdot|x)$ for some $\theta^t \in \Theta$. Under ETH (Randomized Exponential Time Hypothesis), no proper algorithm can be simultaneously data-efficient ($T_\text{data} \leq \text{poly}(d, \beta^{-1}, \varepsilon^{-1}, \delta^{-1})$) and compute-efficient ($T_\text{comp} \leq \text{poly}(d, \exp(\beta^{-1}), \varepsilon^{-1}, \delta^{-1})$). This covers OnlineDPO, XPO, and any algorithm that solves $\pi^t = \arg\min_{\pi \in \Pi} L^t_\mathcal{D}(\pi)$. Proof: reduction from Max-$k$-DNF.

**Result 4 — Multi-turn exploration (Thm 5.1).** For autoregressive LMs ($\mathcal{Y} = \mathcal{A}^H$, $\pi_\text{ref}$ factored over tokens), under autoregressive realizability ($\pi^*_\beta \in \Pi_\text{auto}$), the MTSS algorithm replaces the sequence-level coverage $C_\text{cov}(\pi^*_\beta)$ with the token-level *conditional coverage coefficient*:
$$C_\text{cond}(\pi^*_\beta) := \max_{h \in [H]} \sup_{x \in \mathcal{X},\, (a_1,\dots,a_h) \in \mathcal{A}^h} \frac{\pi^*_{h,\beta}(a_h | x, a_{1:h-1})}{\pi_{h,\text{ref}}(a_h | x, a_{1:h-1})}.$$
This can be exponentially smaller: $C_\text{cond}(\pi^*_\beta) \leq 2$ while $C_\text{cov}(\pi^*_\beta) \geq 2^H$. Key insight: when $\pi^*_\beta$ is autoregressive, the KL-regularized value function $Q^*_{h,\beta}(x, a_{1:h})$ is *linear* in $\phi_h(x, a_{1:h})$, enabling efficient DP backward induction.

## Results

- SpannerSampling: $T_\text{data} = \tilde{O}(d^2/\varepsilon^2)$, $T_\text{comp} = \tilde{O}(C_\text{cov} \cdot R^2_\text{max}/\beta^2) \cdot T_\text{data}^2$ — both optimal up to poly factors.
- MTSS: replaces $C_\text{cov}$ with $C_\text{cond}$, achieving exponential runtime improvement under autoregressive realizability.
- Thm 4.1 (proper algorithms): no polynomial-time proper algorithm exists under ETH.

## Strengths & Limitations

**Strengths:** Sharp computational-statistical separation; first algorithm matching the coverage lower bound; ETH-conditional hardness of training-time interventions is a strong result; multi-turn result connects to classical RL ($Q^*$-realizability).

**Limitations:** Linear softmax policy class only (nonlinear/transformer case is open); polynomial dependence on problem parameters in $T_\text{comp}$ may not be tight; autoregressive realizability ($\pi^*_\beta \in \Pi_\text{auto}$) is a nontrivial assumption for Thm 5.1; ETH is a computational assumption, not information-theoretic.

## Connections

- [[coverage-coefficient]] — the central complexity measure; governs $T_\text{comp}$ lower and upper bounds
- [[linear-softmax-policy]] — the policy parameterization studied throughout
- [[contextual-bandits]] — alignment is cast as a contextual bandit with KL regularization
- [[monte-carlo-tree-search]] — MTSS's multi-turn DP is a token-level analogue; [[Dam2024Power]] studies MCTS convergence in a related stochastic setting
- [[test-time-scaling]] — SpannerSampling formalizes *why* inference-time compute (rather than training-time) enables computationally efficient exploration; direct theoretical backing for MCTS/BoN approaches
- [[realizability]] — Assumption 1.1 (policy realizability) parallels FALCON's realizability; [[SimchiLevi2022Bypassing]]
- [[dec]] — DEC/DOEC are complexity measures for oracle-efficient bandits; $C_\text{cov}$ plays an analogous role in the sampling oracle framework
- **Extends:** XPO (Xie et al. 2024) — proves XPO cannot be computationally efficient (Thm 4.1)
- **Extends:** OnlineDPO (Guo et al. 2024) — shows it is data-inefficient due to passive exploration

## Open Questions

- Can SpannerSampling extend beyond linear softmax to general nonlinear/transformer policies?
- Is the polynomial dependence on $C_\text{cov}$ in $T_\text{comp}$ tight, or can it be improved?
- Does the multi-turn result (Thm 5.1) extend to sub-sequence-level MDPs (e.g., sentence-level steps)?
- Is there a training-time intervention that achieves computational efficiency under a weaker assumption than ETH?
- Can $C_\text{cond}$ be estimated empirically from the base model before running alignment?
