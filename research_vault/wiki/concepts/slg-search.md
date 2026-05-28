---
title: "Scaling-Law Guided (SLG) Search"
tags: [llm, test-time-scaling, search, best-of-n]
introduced_by: [[Li2026Predicting]]
---

# Scaling-Law Guided (SLG) Search

**Definition:** A two-stage test-time search algorithm that uses tail-based scaling law predictions to identify the most promising intermediate state from a prompt, then concentrates the remaining compute budget on that state.

## Intuition
BoN wastes compute sampling uniformly from the prompt. SLG first generates $K$ candidate intermediate states (e.g., first 100 tokens), spends a small estimation budget $m$ on each to predict their scaling potential $\hat{V}_N(s_i)$, then exploits the best state with the remaining budget. The gain comes from avoiding unpromising intermediate states early.

## Formal Description
**Input:** Prompt $x$, total budget $N$, search width $K$, estimation samples $m$.

1. Generate $K$ intermediate states $\{s_i\}_{i=1}^K$ from $x$
2. For each $s_i$: sample $m$ responses, estimate $\hat{V}_N(s_i)$ via tail extrapolation (Algorithm 1 of [[Li2026Predicting]])
3. Select best state: $\hat{I} = \arg\max_{i \in [K]} \hat{V}_N(s_i)$
4. Sample $N - Km$ responses from $s_{\hat{I}}$; return highest-reward response

Optimal schedule: $m(N) \approx \frac{1}{5}(\ln N)^3$, $K(N) \approx \frac{N}{2m(N)}$.

**Vanishing regret (Theorem 2 of [[Li2026Predicting]]):**

$$V_N(\mathcal{A}_\text{SLG}) \geq V_{\lfloor N/2m \rfloor}(\mathcal{A}^*) - c\frac{\log N}{\sqrt{m}}$$

**Polynomial compute amplification (Corollary 1):**

$$V_N(\mathcal{A}_\text{SLG}) \geq V_{N^{1+\gamma}}(\mathcal{A}_\text{BoN}), \quad \gamma > 0$$

## Key Papers
- [[Li2026Predicting]] — introduces SLG Search with full theoretical analysis

## Variants & Related Concepts
- [[test-time-scaling]] — the broader context
- [[best-arm-identification]] — SLG's state selection phase is structurally a fixed-budget BAI problem: pilot $K$ arms, then exploit the best
- [[monte-carlo-tree-search]] — deeper tree-structured alternative; SLG uses only 2 stages

## Current State
SLG is a new algorithm (Feb 2026). Open: deeper multi-stage extensions, PRM integration, non-Gaussian tail distributions.
