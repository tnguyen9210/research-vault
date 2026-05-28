---
title: "Oracle Efficiency (Contextual Bandits)"
tags: [contextual-bandits, computational-complexity, regression]
---

# Oracle Efficiency

**Definition:** An oracle-efficient contextual bandit algorithm reduces the learning problem to a small number of calls to a regression or optimization oracle, enabling practical implementation with standard ML libraries.

## Intuition

Direct contextual bandit algorithms are computationally hard in general (related to agnostic classification). Oracle efficiency sidesteps this: instead of solving the bandit problem from scratch, the algorithm calls a black-box learner a few times and combines the results.

## Two Oracle Types

**Online regression oracle** O_on(F): receives a stream of (context, action, reward) tuples online, outputs predictors with small cumulative squared loss. Requires O(T) calls. Used by SquareCB (Foster & Rakhlin 2020), E2D (Foster et al. 2021a).

**Offline regression oracle** O_off(F): receives a batch of iid (context, action, reward) tuples, outputs a predictor minimizing out-of-sample prediction error. More practical — standard ERM, ridge regression, logistic regression all qualify. Complexity governed by [[doec]].

## Why Offline Oracle Is Preferred in Practice

1. Richer statistical learning theory guarantees
2. Any standard supervised ML algorithm qualifies (sklearn, xgboost, neural nets)
3. Few calls (O(log T)) means low overhead; compatible with periodic retraining

## Oracle Call Counts (Key Results)

| Algorithm | Oracle | Calls | Setting |
|-----------|--------|-------|---------|
| E2D (Foster et al. 2021a) | Online | T | General |
| FALCON (Simchi-Levi & Xu 2022) | Offline | log T | Discrete actions |
| Linear FALCON (Xu & Zeevi 2020) | Offline | log T | Linear reward |
| UCCB (Xu & Zeevi 2020) | Offline | T | General |
| E2D.Off (Foster et al. 2024) | Offline | T | General |
| OE2D [[Qin2026Taming]] | Offline | log T | **General** |

## Key Papers

- [[SimchiLevi2022Bypassing]] — FALCON: first optimal offline-oracle-efficient algorithm; O(log T) calls for discrete actions under realizability
- [[Qin2026Taming]] — OE2D: extends FALCON to general action spaces and drops realizability; introduces DOEC

## Related Concepts

- [[offline-regression-oracle]]
- [[dec]] — governs online-oracle efficiency
- [[doec]] — governs offline-oracle efficiency
- [[contextual-bandits]]
