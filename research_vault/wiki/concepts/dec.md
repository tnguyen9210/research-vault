---
title: "Decision Estimation Coefficient (DEC)"
tags: [contextual-bandits, complexity-measure, oracle-efficiency]
---

# Decision Estimation Coefficient (DEC)

**Definition:** (Foster et al. 2021a) For reward class $G: \mathcal{A} \to \mathbb{R}$, benchmark $\Lambda \subseteq \Delta(\mathcal{A})$, $\gamma > 0$, reference $\hat{g}$:

$$
\mathrm{dec}_\gamma(G, \hat{g}, \Lambda) = \inf_{p \in \Delta(\mathcal{A})}\ \sup_{g^* \in G}\ \mathbb{E}_{a \sim p}\!\left[
\max_{\lambda \in \Lambda} \mathbb{E}_{a' \sim \lambda}[g^*(a')] - g^*(a) - \gamma\,(\hat{g}(a) - g^*(a))^2
\right]
$$

$$
\mathrm{dec}_\gamma(G, \Lambda) = \max_{\hat{g} \in G}\ \mathrm{dec}_\gamma(G, \hat{g}, \Lambda)
$$

## Intuition

DEC measures the fundamental cost of exploration when using an **online** regression oracle. A small DEC means there exists an action distribution $p$ that simultaneously looks near-optimal under the worst-case $g^*$ AND provides good squared-error signal for estimating $g^*$. It is the primary complexity measure for online-oracle-efficient interactive decision making.

## Role in E2D Algorithm

The E2D algorithm (Foster et al. 2021a) achieves:

$$\mathrm{Reg}(T, \mathrm{E2D}) \leq \min_{\gamma > 0} \left( T \cdot \max_x\,\mathrm{dec}_\gamma(F_x, \Lambda) + \gamma \cdot \mathrm{Reg}_\mathrm{on}(\mathcal{F}, T, \delta) \right)$$

Key values: $\mathrm{dec}(F_x, \Lambda) \lesssim |\mathcal{A}|$, $d/h$, $1/h$ for discrete, linear, $h$-smoothed settings respectively.

## Key Difference from DOEC

DEC's exploration cost $\mathbb{E}_{a \sim p}[(\hat{g}(a) - g^*(a))^2]$ references the **ground-truth $g^*$** — this prevents reduction to offline regression (you can't minimize this without knowing $g^*$). [[doec]] replaces this with $\mathrm{Coverage}_\varepsilon$, removing the dependence on $g^*$ and enabling offline oracle reduction.

## DOEC-DEC Bridge ([[Qin2026Taming]], Theorem 5)

Any distribution $p$ certifying $\mathrm{doec}_{\gamma,\varepsilon}(G,\Lambda) \leq V$ also certifies $\mathrm{dec}_\gamma(G,\Lambda) \leq V + 1/\gamma + \gamma\varepsilon$.

Consequently:

$$\mathrm{dec}_\gamma(G, \Lambda) \leq \mathrm{doec}_{\gamma,\varepsilon}(G, \Lambda) + \frac{1}{\gamma} + \gamma\varepsilon$$

This bridges the design principles of offline- and online-oracle efficient algorithms for the first time.

## Key Papers

- [[Qin2026Taming]] — establishes $\mathrm{dec} \leq \mathrm{doec}$ + lower-order terms

## Related Concepts

- [[doec]] — offline-oracle analogue; upper bounds DEC
- [[contextual-bandits]]
- [[oracle-efficiency]]
