---
title: "Decision-Offline Estimation Coefficient (DOEC)"
tags: [contextual-bandits, complexity-measure, oracle-efficiency]
introduced_by: [[Qin2026Taming]]
---

# Decision-Offline Estimation Coefficient (DOEC)

**Definition:** (Qin & Zhang 2026, Definition 2) For function class $G: \mathcal{A} \to [0,1]$, benchmark distributions $\Lambda \subseteq \Delta(\mathcal{A})$, $\gamma > 0$, $\varepsilon \geq 0$:

$$
\mathrm{doec}_{\gamma,\varepsilon}(\hat{g}, G, \Lambda) = \min_{p \in \mathrm{co}(\Lambda)}\ \max_{\lambda \in \Lambda}
\left[ \mathbb{E}_{a \sim \lambda}[\hat{g}(a)] - \mathbb{E}_{a \sim p}[\hat{g}(a)] + \frac{1}{\gamma}\,\mathrm{Coverage}_\varepsilon(p, \lambda;\, G) \right]
$$

$$
\mathrm{doec}_{\gamma,\varepsilon}(G, \Lambda) = \max_{\hat{g} \in G}\ \mathrm{doec}_{\gamma,\varepsilon}(\hat{g}, G, \Lambda)
$$

## Intuition

DOEC is the minimax value of the [[exploitative-f-design]] problem. It measures: given the worst-case reward estimate $\hat{g}$, what is the minimum achievable cost when simultaneously exploiting and exploring? A small DOEC means there always exists an action distribution that both exploits and explores efficiently.

## Key Difference from DEC

[[dec]] uses $\mathbb{E}_{a \sim p}[(\hat{g}(a) - g^*(a))^2]$ as the exploration cost — this references the unknown $g^*$, preventing direct reduction to offline regression.

DOEC uses $\mathrm{Coverage}_\varepsilon(p, \lambda;\, G)$, which does **not** reference $g^*$ — enabling actual reduction to [[offline-regression-oracle]].

## Bounds

**Upper bound via $\varepsilon$-SEC (Theorem 3):**

$$\mathrm{doec}_{\gamma,\varepsilon}(G, \Lambda) \leq \frac{10}{\gamma}\,\mathrm{SEC}_\varepsilon(G, \Lambda)$$

**Upper bound via Eluder dimension (for $\Lambda = \{\delta_a\}$):**

$$\mathrm{doec}_{\gamma,\varepsilon}(G, \Lambda) \lesssim \frac{1}{\gamma}\,\mathrm{Edim}(G, \sqrt{\varepsilon}) \cdot \log^2(1/\varepsilon)$$

**Concrete examples (finite $\mathcal{F}$, ERM oracle):**

| Setting | doec bound | Resulting regret |
|---------|-----------|-----------------|
| Discrete, $\lvert\mathcal{A}\rvert$ actions | $\tilde{O}(\lvert\mathcal{A}\rvert/\gamma \cdot \log(1/\varepsilon))$ | $\tilde{O}(\sqrt{\lvert\mathcal{A}\rvert T \log\lvert\mathcal{F}\rvert})$ |
| Per-context linear (dim $d$) | $\tilde{O}(d/\gamma \cdot \log^3(1/\varepsilon))$ | $\tilde{O}(\sqrt{dT \log\lvert\mathcal{F}\rvert})$ |
| $h$-smoothed regret | $\log(1+1/\varepsilon)/(h\gamma)$ | $\tilde{O}(\sqrt{T/h \cdot \log\lvert\mathcal{F}\rvert})$ |

**$\varepsilon$-SEC can be loose:** Proposition 3 of [[Qin2026Taming]] constructs examples where $\mathrm{SEC}_\varepsilon \geq 2^{k-2}$ but $\mathrm{doec} \lesssim \sqrt{k}/\gamma$ — exponential gap. Active experimental design (choosing $p$) can be strictly better than passive coverage.

## DOEC-DEC Bridge

Any $p$ certifying $\mathrm{doec}_{\gamma,\varepsilon}(G,\Lambda) \leq V$ also certifies $\mathrm{dec}_\gamma(G,\Lambda) \leq V + 1/\gamma + \gamma\varepsilon$ (Theorem 5, [[Qin2026Taming]]).

## Role in Regret (Theorem 1)

OE2D's regret scales as:

$$\mathrm{Reg} \lesssim \tilde{O}\!\left(\sqrt{T \cdot \max_x\,\mathbb{E}[\mathrm{doec}_{\gamma,\varepsilon}(F_{x_t}, \Lambda)] \cdot \mathrm{Reg}_\mathrm{off}(\mathcal{F}, T, \delta)}\right)$$

## Key Papers

- [[Qin2026Taming]] — introduced DOEC

## Related Concepts

- [[dec]] — online-oracle analogue; DOEC $\geq$ DEC (up to lower-order terms)
- [[epsilon-sec]] — upper bounds DOEC; may be exponentially loose
- [[exploitative-f-design]] — DOEC is the minimax value of this problem
- [[eluder-dimension]] — bounds $\varepsilon$-SEC, which bounds DOEC (for discrete $\Lambda$)
