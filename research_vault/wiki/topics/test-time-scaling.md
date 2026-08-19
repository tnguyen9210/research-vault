---
title: "Test-Time Scaling"
tags: [llm, inference, scaling-laws]
---

# Test-Time Scaling

## Overview
Test-time scaling refers to improving LLM output quality by allocating more compute at inference, without changing model weights. The dominant strategy is Best-of-$N$ (BoN): generate $N$ candidates, return the highest-reward one. BoN is simple but stateless — it ignores intermediate generation structure and wastes compute on unpromising trajectories.

The emerging approach is adaptive multi-stage search: use a small pilot budget to estimate the scaling potential of candidate intermediate states, then concentrate remaining compute on the most promising one. [[Li2026Predicting]] provides the first principled framework for this, grounded in tail distribution theory.

## Key Papers

- **[[Li2026Predicting]]** (2026) — introduces SLG Search: predicts BoN scaling laws via Gaussian tail extrapolation from $m \ll N$ samples; proves polynomial compute amplification over BoN ($V_N(\mathcal{A}_\text{SLG}) \geq V_{N^{1+\gamma}}(\mathcal{A}_\text{BoN})$); validated on AMC/AIME benchmarks with Llama and Qwen models
- **[[Foster2025Foundation]]** (2025) — provides the *theoretical* justification for why inference-time compute beats training-time-only exploration: SpannerSampling achieves the [[coverage-coefficient]] lower bound using only the weak sampling oracle; training-time interventions (DPO/XPO) cannot be simultaneously data- and compute-efficient under ETH (Thm 4.1); multi-turn MTSS achieves exponentially better runtime via token-level $C_\text{cond}$

## Open Problems

- Can SLG extend beyond two stages to deeper multi-stage trees?
- What are the right tail assumptions for non-standard reward models?
- Can process reward models (PRMs) be incorporated into the scaling law framework?
- Is there a lower bound showing BoN fundamentally requires polynomially more budget than adaptive search?
- Can the polynomial dependence on $C_\text{cov}$ in SpannerSampling's $T_\text{comp}$ be tightened?
- Can $C_\text{cond}$ be estimated from the base model before running alignment?

## Related Topics
- [[monte-carlo-tree-search]] — deeper tree-structured planning; [[Dam2024Power]] shows convergence rates in stochastic MDPs; SLG is a shallower, LLM-focused analogue; MTSS (from [[Foster2025Foundation]]) is a DP backward-induction algorithm over token-level MDPs
- [[best-arm-identification]] — SLG's state selection is structurally a BAI problem; [[Kanarios2024Cost]] studies cost-aware BAI
