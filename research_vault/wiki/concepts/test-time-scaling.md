---
title: "Test-Time Scaling"
tags: [llm, inference, scaling-laws, best-of-n]
introduced_by: [[Li2026Predicting]]
---

# Test-Time Scaling

**Definition:** The phenomenon where LLM output quality improves as more compute is allocated at inference time (e.g., by generating more candidate responses), independent of model size or training compute.

## Intuition
Training-time scaling (larger models, more data) improves a model's base capabilities. Test-time scaling leverages the stochastic nature of generation: even a fixed model can produce better answers by sampling more and selecting the best. The key question is how to allocate a fixed inference budget most efficiently.

## Formal Description
Given prompt $x$ and stochastic policy $\pi_\text{ref}$, the value function of a state $s$ under budget $N$ is:

$$V_N(s) = \mathbb{E}\!\left[\max_{1 \leq i \leq N} R_s^{(i)}\right], \quad R_s^{(i)} \overset{\text{i.i.d.}}{\sim} F_s$$

**Best-of-$N$ (BoN):** generate $N$ responses from prompt $x$, return highest-reward one. Simple but wasteful — ignores intermediate state structure.

**Scaling law:** empirically $V_N(\mathcal{A}_\text{BoN}) \approx C(\log N)^\gamma$ for $\gamma \in (0,1)$, meaning gains diminish logarithmically with budget.

## Key Papers
- [[Li2026Predicting]] — tail extrapolation predicts $V_N(s)$ from $m \ll N$ samples; SLG Search achieves polynomial compute amplification over BoN
- [[Foster2025Foundation]] — the theoretical case for inference-time compute: SpannerSampling meets the [[coverage-coefficient]] lower bound with only a sampling oracle, while training-time interventions (DPO/XPO) cannot be simultaneously data- and compute-efficient under ETH

## Variants & Related Concepts
- [[test-time-scaling-slg-search]] — adaptive two-stage search using predicted scaling laws
- **Tree of Thoughts / MCTS** — deeper tree-structured search; [[mcts]] is a related planning framework
- **Process Reward Models (PRMs)** — score intermediate steps rather than final responses; complement to outcome reward models
- [[coverage-coefficient]] — the quantity governing when inference-time exploration is computationally feasible

## Current State
Active research area (2024–2026). BoN is the dominant practical approach; adaptive multi-stage methods (SLG) are emerging with theoretical backing, and [[Foster2025Foundation]] supplies the computational-theory grounding for why inference-time compute beats training-time-only exploration. Structural connection: state selection in SLG is a fixed-budget [[bai]] problem. Per-paper open questions (deeper trees, PRM integration, tail assumptions, estimating $C_\text{cond}$) live on the paper pages.
