---
title: "$\\varepsilon$-Sequential Extrapolation Coefficient ($\\varepsilon$-SEC)"
tags: [contextual-bandits, complexity-measure, oracle-efficiency, coverage]
introduced_by: [[Qin2026Taming]]
---

# $\varepsilon$-Sequential Extrapolation Coefficient ($\varepsilon$-SEC)

**Definition:** A passive measure of exploration difficulty: the worst-case accumulated coverage of each benchmark distribution $\lambda_i$ by the mixture of distributions chosen up to and including step $i$. Introduced in [[Qin2026Taming]] as a modification of Xie et al. (2022)'s SEC, and used there to upper bound [[doec]].

## Intuition

To learn efficiently from an offline regression oracle you need the data you gather to be informative about every benchmark distribution you might want to evaluate. $\varepsilon$-SEC asks: if distributions arrive adversarially one at a time, how badly can the running mixture fail to cover the newest arrival, summed over the sequence?

Small $\varepsilon$-SEC means an adversary cannot keep presenting distributions that the accumulated data fails to cover — coverage is achieved *incidentally*, without designing for it. That is exactly why it is a **passive** measure, and why it can be loose: [[doec]] permits active experimental design, deliberately choosing an exploration distribution $p$, whereas $\varepsilon$-SEC only tracks what accumulates.

## Formal Description

**Coverage.** For function class $\mathcal{G}$ and $\varepsilon > 0$, the coverage of behavior distribution $p$ over target $q$ is

$$
\text{Coverage}_\varepsilon(p,q;\mathcal{G}) = \sup_{g,g'\in\mathcal{G}} \frac{\big(\mathbb{E}_{a\sim q}[g(a)-g'(a)]\big)^2}{\varepsilon + \mathbb{E}_{a\sim p}[(g(a)-g'(a))^2]}.
$$

Smaller means samples from $p$ are more informative for evaluating $q$.

**Definition 3 of [[Qin2026Taming]].** With $\lambda_{1:i} := \sum_{j=1}^i \lambda_j$,

$$
\text{SEC}_\varepsilon(\mathcal{G},\Lambda) = \sup_{N\in\mathbb{N}}\ \sup_{\lambda_1,\dots,\lambda_N\in\Lambda}\ \sum_{i=1}^N \text{Coverage}_{N\varepsilon}(\lambda_{1:i},\lambda_i;\mathcal{G}).
$$

Two deliberate differences from the original SEC (Xie et al. 2022): the regularization parameter is $N\varepsilon$, scaling with the number of terms rather than fixed at 1; and coverage is measured of $\lambda_{1:i}$ on $\lambda_i$ rather than $\lambda_{1:i-1}$ on $\lambda_i$. [[Qin2026Taming]] reports that this off-by-one adjustment is what makes the quantity usable for constructing an exploration distribution certifying low DOEC, and lets it give guarantees where optimism-based approaches do not suffice.

**Theorem 3.** $\text{doec}_{\gamma,\varepsilon}(\mathcal{G},\Lambda) \le \frac{10}{\gamma}\,\text{SEC}_{\gamma\varepsilon}(\mathcal{G},\Lambda)$.

**Proposition 1** (grounding). For $\Lambda = \{\delta_a : a\in\mathcal{A}\}$, $\text{SEC}_\varepsilon(\mathcal{G},\Lambda) \lesssim \text{Edim}(\mathcal{G},\sqrt{\varepsilon})\log^2(1/\varepsilon)$ — so small [[eluder-dimension]] certifies small $\varepsilon$-SEC.

**Proposition 3** (the caveat). There are instances with $\text{SEC}_\varepsilon \ge 2^{k-2}$ while $\text{doec}\lesssim\sqrt{k}/\gamma$ — an *exponential* gap. Active experimental design can be strictly better than passive coverage, so $\varepsilon$-SEC alone does not characterize when offline-oracle-efficient contextual bandits are feasible.

## Key Papers

- Xie et al. (2022) — the original SEC, used to unify optimism-based RL analyses under low coverability and small Bellman Eluder dimension
- [[Qin2026Taming]] — introduces $\varepsilon$-SEC; Theorem 3 (upper bounds [[doec]]), Proposition 1 (Eluder grounding), Proposition 3 (exponential looseness)
- Agarwal et al. (2024) — nonlinear $F$-design; the $\gamma\to0$ pure-exploration specialization generalizes their Theorem 4.2

## Variants & Related Concepts

- [[doec]] — the quantity $\varepsilon$-SEC bounds; strictly sharper, because it allows active design
- [[eluder-dimension]] — bounds $\varepsilon$-SEC (Proposition 1)
- [[dec]] — the online counterpart in the DEC/DOEC family
- [[exploitative-f-design]] — the per-context optimization OE2D actually solves, which is where active design enters
- [[coverage-coefficient]] — same underlying question (how well does one distribution cover another?) in the LM alignment setting

## Current State

Useful but known to be lossy. It is the main structural tool for certifying bounded [[doec]] and connects the DOEC framework to the established Eluder-dimension literature, but [[Qin2026Taming]] itself demonstrates the exponential gap and lists "tighter structural characterizations of DOEC beyond $\varepsilon$-SEC" as an open problem. The gap is not a technical artifact — it reflects a genuine distinction between passive coverage and active experimental design.
