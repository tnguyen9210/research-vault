---
title: "Offline Regression Oracle"
tags: [oracle-efficiency, regression, contextual-bandits]
---

# Offline Regression Oracle

**Definition:** An algorithm $\mathcal{O}_\mathrm{off}(\mathcal{F})$ that takes a batch of $T$ iid (context, action, reward) tuples drawn from distribution $\mathcal{D}$ and outputs $\hat{f} \in \mathcal{F}$ minimizing out-of-sample squared prediction error, with error bound $\mathrm{Reg}_\mathrm{off}(\mathcal{F}, T, \delta)$.

## Formal Description

With probability at least $1 - \delta$:

$$\mathbb{E}_{\mathcal{D}}\!\left[(\hat{f}(x,a) - f^*(x,a))^2\right] \lesssim \mathrm{Reg}_\mathrm{off}(\mathcal{F}, T, \delta)$$

Key properties:
- $\mathrm{Reg}_\mathrm{off}$ is monotonically decreasing in $T$ (more data → better)
- $\mathrm{Reg}_\mathrm{off}$ is monotonically increasing in $\delta$ (looser confidence → worse)
- For finite $\mathcal{F}$ with ERM: $\mathrm{Reg}_\mathrm{off} = \tilde{O}(\log|\mathcal{F}| / T)$

## Why It Matters

Offline oracles are strictly more practical than online regression oracles:
- Standard ERM, regularized least squares, logistic regression all qualify
- Implemented directly by any supervised ML library
- Statistical learning theory provides clean uniform convergence guarantees

## Contrast with Online Oracle

An online regression oracle $\mathcal{O}_\mathrm{on}(\mathcal{F})$ receives tuples in a streaming fashion and must predict before seeing labels. Online guarantees are harder to obtain and harder to implement. Offline oracles require only that data is iid from some fixed distribution — satisfied when using historical batches from prior epochs.

## Key Papers

- [[Qin2026Taming]] — uses offline oracle with $O(\log T)$ calls for general function classes; shows $\mathrm{Reg}_\mathrm{off}$ governs regret alongside [[doec]]

## Related Concepts

- [[oracle-efficiency]]
- [[doec]] — the complexity measure that governs regret when using offline oracles
- [[contextual-bandits]]
