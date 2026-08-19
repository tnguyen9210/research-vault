---
title: "Eluder Dimension"
tags: [function-approximation, complexity-measure, bandits, learning-theory]
introduced_by: [[Qin2026Taming]]
---

# Eluder Dimension

**Definition:** A complexity measure for a function class $\mathcal{G}$ capturing how long an adversary can keep producing points whose value remains uncertain — the length of the longest sequence of actions each of which is $\varepsilon$-"independent" of its predecessors, in the sense that two functions in $\mathcal{G}$ agreeing on all earlier points can still differ by more than $\varepsilon$ on it.

## Intuition

Statistical dimensions like VC or Rademacher complexity measure how hard a class is to *fit*. Eluder dimension measures something different and more suited to sequential decision-making: how hard the class is to *pin down by adaptive querying*. A class has small Eluder dimension when observing a few well-chosen points forces the value at every remaining point, so exploration cannot be dragged out.

The name is literal — it counts how long a point can "elude" being determined by what you have already seen. For linear classes it is $\tilde{O}(d)$; the mechanism generalizes the linear-algebraic fact that $d$ independent observations determine a $d$-dimensional linear function.

## Formal Description

Informally, $\text{Edim}(\mathcal{G},\varepsilon)$ is the length of the longest sequence $a_1,\dots,a_n$ such that each $a_i$ is $\varepsilon$-independent of $\{a_1,\dots,a_{i-1}\}$: there exist $g,g'\in\mathcal{G}$ with $\sqrt{\sum_{j<i}(g(a_j)-g'(a_j))^2}\le\varepsilon$ yet $|g(a_i)-g'(a_i)|>\varepsilon$. (Formal statement: Russo & Van Roy 2013; Definition 6, Appendix C.3.1 of [[Qin2026Taming]].)

**Role in this vault.** Eluder dimension enters as a *sufficient condition* for the sharper measures actually being studied. Proposition 1 of [[Qin2026Taming]]: for $\Lambda = \{\delta_a : a\in\mathcal{A}\}$,

$$
\text{SEC}_\varepsilon(\mathcal{G},\Lambda) \lesssim \text{Edim}(\mathcal{G},\sqrt{\varepsilon})\log^2(1/\varepsilon),
\qquad\text{hence}\qquad
\text{doec}_{\gamma,\varepsilon}(\mathcal{G},\Lambda) \lesssim \tfrac{1}{\gamma}\text{Edim}(\mathcal{G},\sqrt{\varepsilon})\log^3(1/\varepsilon).
$$

So small Eluder dimension bounds [[epsilon-sec]], which bounds [[doec]]. Applied to per-context generalized linear rewards (where $\text{Edim} \le \tilde{O}(d\log\frac{1}{\varepsilon})$), this gives OE2D a $\tilde{O}(\sqrt{dT\log|\mathcal{F}|})$ regret bound while cutting oracle calls from $O(T)$ to $O(\log T)$.

## Key Papers

- Russo & Van Roy (2013) — introduces Eluder dimension for optimistic algorithms in bandits and RL
- Jin et al. (2021a) — Bellman Eluder dimension; the RL extension
- [[Qin2026Taming]] — uses it to certify small [[epsilon-sec]] and hence small [[doec]] (Proposition 1)
- [[Yin2023Offline]] — argues the tractable instances of Eluder dimension are still essentially linear-in-features, motivating [[differentiable-function-approximation]] as an alternative structural handle

## Variants & Related Concepts

- [[epsilon-sec]] — bounded by Eluder dimension; in turn bounds [[doec]]
- [[doec]] / [[dec]] — the complexity measures Eluder dimension is used to certify
- [[differentiable-function-approximation]] — a competing structural assumption; smoothness rather than query-independence
- **Bilinear classes** (Du et al. 2021), **linear Bellman complete** (Zanette et al. 2020) — sibling structural conditions
- [[realizability]] — assumed alongside; Eluder dimension constrains the class's geometry, realizability its expressiveness

## Current State

A standard tool for proving sequential learnability beyond linear models, but with a known limitation the vault records twice: its concrete tractable instances remain close to linear-in-features (a point made in Wen & Van Roy 2013 §4.1 and cited by [[Yin2023Offline]]), and it is a *sufficient* rather than necessary condition — [[Qin2026Taming]] shows [[epsilon-sec]] can be exponentially loose relative to [[doec]], so the chain from Eluder dimension down to the true complexity can be doubly loose.
