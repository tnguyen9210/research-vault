---
title: "Contextual Bandits"
tags: [bandits, sequential-decision-making, reinforcement-learning]
---

# Contextual Bandits

**Definition:** A sequential decision-making problem where at each round $t$, a learner observes context $x_t \in \mathcal{X}$, takes action $a_t \in \mathcal{A}$, and receives reward $r_t$. The goal is to minimize cumulative regret versus the best action per context in hindsight.

## Intuition

A one-step version of online RL: no planning beyond the current step, but must balance **exploration** (learning $f^*$) with **exploitation** (taking high-reward actions). Unlike multi-armed bandits, the best arm can vary by context.

## Formal Description

- Context space $\mathcal{X}$, action space $\mathcal{A}$, unknown reward $f^*: \mathcal{X} \times \mathcal{A} \to [0,1]$
- Realizability: $f^* \in \mathcal{F}$ (learner's reward function class)
- Contexts drawn iid from $\mathcal{D}_\mathcal{X}$ (unless context-shift setting)

$$\Lambda\text{-Regret}(T, \mathrm{ALG}) = \sum_{t=1}^T \mathrm{Reg}(p_t \mid x_t), \quad \mathrm{Reg}(p \mid x) = \max_{\lambda \in \Lambda}\,\mathbb{E}_{a \sim \lambda}[f^*(x,a)] - \mathbb{E}_{a \sim p}[f^*(x,a)]$$

Benchmark class $\Lambda$ determines the regret notion:
- $\Lambda = \{\delta_a : a \in \mathcal{A}\}$ → standard regret (best fixed action per context)
- $\Lambda = \Delta_h^\mu(\mathcal{A})$ → $h$-smoothed regret (best $h$-smooth distribution per context)

## Key Papers

- [[Qin2026Taming]] — first offline-oracle-efficient algorithm for general function approximation with $O(\log T)$ oracle calls
- [[Ryu2025Improved]] — offline policy optimization from logged data; PUB (pessimism via betting) for off-policy selection; freezing score function for learning in small-data regimes

## Variants & Related Concepts

- [[oracle-efficiency]] — computational tractability via oracle reductions
- [[offline-regression-oracle]] — practical oracle model
- [[importance-weighting]] — core primitive for offline/off-policy evaluation
- [[dec]] — complexity measure for online-oracle-efficient algorithms
- [[doec]] — complexity measure for offline-oracle-efficient algorithms

## Current State

Mature area for simple settings (linear, discrete). Active frontiers: general function approximation with few oracle calls (partially addressed by [[Qin2026Taming]]), partial monitoring, non-iid contexts, RLHF extensions.
