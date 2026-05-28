---
title: "Offline-Oracle-Efficient Contextual Bandits"
tags: [contextual-bandits, oracle-efficiency, offline-regression, regret-bounds]
---

# Offline-Oracle-Efficient Contextual Bandits

## Overview

A research line making contextual bandit algorithms computationally tractable by reducing them to a small number of calls to an **offline regression oracle** — a standard supervised ML algorithm trained on batch data. The core challenge: the bandit algorithm must balance exploration and exploitation using only periodic batch model updates, not a continuously updated online learner.

The key open problem until 2026: can we handle **general reward function approximation** with only O(log T) oracle calls? Prior work was stuck at either restricted settings (discrete/linear) for few calls, or O(T) calls for general classes.

## Key Papers (Chronological)

- **UCCB** (Xu & Zeevi 2020): first to handle general function classes with offline oracle, but requires O(T) calls — matches online oracle in call count, undermining the practical motivation
- **[[SimchiLevi2022Bypassing]]** (Simchi-Levi & Xu 2022): O(log T) calls for **discrete action spaces** under realizability; first to achieve optimal regret with offline oracle; introduces separate Low Regret and Good Coverage conditions
- **Linear FALCON** (Xu & Zeevi 2020, Sec. 4): O(log T) calls for **per-context linear reward**
- **E2D.Off** (Foster et al. 2024): general function classes, O(T) calls, information-theoretic framework
- [[Qin2026Taming]] (2026): **O(log T) calls + general function classes** — closes the gap; introduces exploitative F-design and DOEC

## The Core Tension: LR + GC

Every offline-oracle-efficient bandit algorithm must simultaneously achieve:
- **Low Regret (LR):** take near-optimal actions under the current reward estimate
- **Good Coverage (GC):** collect data that covers all benchmark distributions Λ

FALCON and Linear FALCON handled these separately for special cases. [[Qin2026Taming]] unifies them via a single minimax optimization ([[exploitative-f-design]]), generalizing to arbitrary function classes.

## Complexity Landscape

| Algorithm | Oracle calls | Setting |
|-----------|-------------|---------|
| FALCON [[SimchiLevi2022Bypassing]] | O(log T) | Discrete actions (realizability) |
| Linear FALCON | O(log T) | Per-context linear |
| UCCB, E2D.Off | O(T) | General |
| **OE2D** [[Qin2026Taming]] | **O(log T)** | **General** |

## Adjacent: Offline Policy Optimization (Off-Policy Learning)

A distinct but related problem: given a fixed log $D_n$ from behavior policy $\pi_\text{ref}$, find the best policy without further interaction. No oracle calls; variance control is the central challenge rather than exploration.

- **[[Ryu2025Improved]]** (Ryu, Kwon, Koppe, Jun — COLT 2025): PUB achieves parameter-free variance-adaptive off-policy selection via betting-based LCB; freezing score function improves learning in small-data regimes. Does not require realizability or an online loop.

Key distinction from the oracle-efficient line: oracle-efficient bandits minimize regret over $T$ online rounds using offline oracle calls; offline policy optimization assumes all data is pre-collected and minimizes offline regret.

## Open Problems

- Lower bounds on [[doec]]: when is offline-oracle-efficient learning information-theoretically hard?
- First-order offline-oracle-efficient algorithms (sub-√T regret under favorable conditions)
- Extensions to partial monitoring, RLHF, non-iid context distributions
- Tighter structural characterizations of [[doec]] beyond [[epsilon-sec]]
- Online-to-offline reduction for first-order algorithms (cf. Foster & Krishnamurthy 2021)

## Related Topics

- [[dec]] and [[doec]] as dual complexity measures (online vs. offline oracle)
- Online-oracle-efficient bandits (E2D, SquareCB — not yet in vault)
