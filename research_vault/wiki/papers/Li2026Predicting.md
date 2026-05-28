---
title: "Predicting and improving test-time scaling laws via reward tail-guided search"
authors: [Muheng Li, Jian Qian, Wenlong Mou]
year: 2026
venue: arXiv
arxiv: "2602.01485"
tags: [test-time-scaling, llm, best-of-n, scaling-laws, search, reward-modeling]
source: raw/papers/Li2026Predicting.pdf
---

# Predicting and Improving Test-Time Scaling Laws via Reward Tail-Guided Search

**TL;DR:** Introduces a tail-extrapolation method to predict Best-of-$N$ scaling laws from $m \ll N$ samples, and SLG Search — a two-stage algorithm that uses these predictions to adaptively allocate compute, achieving polynomial compute amplification over BoN with provable vanishing regret vs. a full-information oracle.

## Problem

Best-of-$N$ (BoN) — generate $N$ responses and select the highest-reward one — is the dominant test-time scaling strategy for LLMs, but lacks principled guidance on: how to choose $N$, how to allocate budget across prompts, and whether adaptive multi-stage strategies can improve efficiency. The value function $V_N(s) = \mathbb{E}[\max_{1 \leq i \leq N} R_s^{(i)}]$ is expensive to estimate directly (requires $N$ samples), and no theory governed when or how much multi-stage search helps over flat BoN.

## Method

### Part 1: Predicting Scaling Laws via Tail Extrapolation

**Key insight:** for large $N$, $V_N(s)$ is governed entirely by the upper tail of the reward distribution $F_s$.

**Gaussian tail assumption (Assumption 1):** For $r \geq r_\alpha = F_s^{-1}(1-\alpha)$, the reward density coincides with $\phi(r; \mu, \sigma^2)$.

**Algorithm 1** (tail estimation from $m$ samples):
1. Extract tail set $\mathcal{D}_\text{tail} = \{r \in \mathcal{D} : r \geq \hat{r}_{\alpha/2}\}$
2. Estimate $(\hat{\mu}, \hat{\sigma})$ via Method of Moments on truncated normal:

$$\hat{\sigma} = \sqrt{\sigma^2_\text{tail}/\delta(z)}, \quad \hat{\mu} = \hat{\mu}_\text{tail} - \hat{\sigma}\lambda(z)$$

where $z = \Phi^{-1}(1-\alpha/2)$, $\lambda(z)$ is the inverse Mills ratio.

3. Predict: $\hat{V}_N(s) = \hat{\mu} + \hat{\sigma} E(N)$, where $E(N)$ is the expected maximum of $N$ i.i.d. standard normals.

**Theorem 1** (estimation error): For $m \geq c_1 \log(1/\delta)$, with probability $\geq 1 - \delta$:

$$\left|\hat{V}_N(s) - V_N(s)\right| \leq c_2 \left\{\sqrt{\frac{\log N \cdot \log(1/\delta)}{m}} + \left(1 - \frac{\alpha}{2}\right)^N\right\}$$

Error decays as $\mathcal{O}(\sqrt{\log N / m})$; the second term is negligible for large $N$.

### Part 2: SLG Search Algorithm

**Algorithm 2** — Scaling-Law Guided (SLG) Search with prompt $x$, budget $N$, width $K$, estimation samples $m$:
1. Generate $K$ intermediate states $\{s_i\}$ from $x$
2. Sample $m$ responses from each $s_i$, estimate $\hat{V}_N(s_i)$ via Algorithm 1
3. Identify best state: $\hat{I} = \arg\max_{i \in [K]} \hat{V}_N(s_i)$
4. Allocate remaining budget $N - K \cdot m$ to $s_{\hat{I}}$; return highest-reward response

**Theorem 2** (vanishing regret vs. oracle): With $K = \lfloor N/2m \rfloor$:

$$V_N(\mathcal{A}_\text{SLG}) \geq V_{\lfloor N/2m \rfloor}(\mathcal{A}^*) - c\frac{\log N}{\sqrt{m}}$$

Setting $m \asymp (\log N)^\alpha$ with $\alpha > 2$ makes the gap vanish.

**Corollary 1** (polynomial compute amplification): SLG with budget $N$ matches BoN with budget $N^{1+\gamma}$:

$$V_N(\mathcal{A}_\text{SLG}) \geq V_{N^{1+\gamma}}(\mathcal{A}_\text{BoN}), \quad \gamma := \frac{\sqrt{2}t}{4(t+1)\sqrt{1+t^2}} > 0$$

where $t = \sigma_1/\sigma_0$ is the variance ratio between response and state generation.

**Proposition 1** (Gaussian advantage over BoN):

$$V_N(\mathcal{A}_\text{SLG}) - V_N(\mathcal{A}_\text{BoN}) \geq \frac{\sqrt{2}t}{4(t+1)}\,\sigma_0\sqrt{\log N}$$

## Results

Evaluated on AMC23, AIME24, AIME25 with Llama-3.2-1B-Instruct and Qwen2.5-7B-Instruct, scored by Skywork-Reward-V2-Llama-3.1-8B:
- SLG consistently outperforms BoN under identical compute budgets across all benchmarks and model scales
- Gain grows with budget $N$ (logarithmic amplification)
- Robust across different reward models and hyperparameter settings
- Optimal estimation budget $m \approx \frac{1}{5}(\ln N)^3$, search width $K \approx \frac{N}{2m(N)}$

## Strengths & Limitations

**Strengths:** Clean theoretical framework; non-asymptotic guarantees; polynomial compute amplification is a strong practical claim; only 2 sequential rounds of interaction (highly parallelizable); validated on real LLMs.

**Limitations:** Gaussian tail assumption may not hold for all reward models or prompts. Two-stage structure limits to one intermediate state selection; deeper trees unexplored. Reward model quality is a bottleneck. No extension to process reward models (PRMs).

## Connections

- [[test-time-scaling]] — the broader phenomenon this paper characterizes and improves
- [[slg-search]] — the algorithm introduced here
- [[Li-Muheng]] — first author
- [[Qian-Jian]] — second author
- [[Mou-Wenlong]] — third/senior author
- **Related (structural):** [[best-arm-identification]] — SLG's state selection phase is structurally a fixed-budget BAI problem; concentrate on the best "arm" (intermediate state) after a pilot round
- **Related (structural):** [[Kanarios2024Cost]] — CABAI's cost-aware allocation parallels SLG's budget-aware state selection

## Open Questions

- Can SLG extend to deeper multi-stage trees (beyond two stages)?
- What is the optimal tail fraction $\alpha$ as a function of the reward model?
- Can the Gaussian tail assumption be relaxed to heavier-tailed distributions?
- Extension to process reward models (PRMs) where intermediate steps are scored?
- Is there a regret lower bound showing BoN requires polynomially more budget than SLG?
