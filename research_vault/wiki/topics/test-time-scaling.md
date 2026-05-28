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

## Open Problems

- Can SLG extend beyond two stages to deeper multi-stage trees?
- What are the right tail assumptions for non-standard reward models?
- Can process reward models (PRMs) be incorporated into the scaling law framework?
- Is there a lower bound showing BoN fundamentally requires polynomially more budget than adaptive search?

## Related Topics
- [[monte-carlo-tree-search]] — deeper tree-structured planning; [[Dam2024Power]] shows convergence rates in stochastic MDPs; SLG is a shallower, LLM-focused analogue
- [[best-arm-identification]] — SLG's state selection is structurally a BAI problem; [[Kanarios2024Cost]] studies cost-aware BAI
