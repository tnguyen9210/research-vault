---
title: "Taming the Monster Every Context: Complexity Measure and Unified Framework for Offline-Oracle Efficient Contextual Bandits"
authors: [Hao Qin, Chicheng Zhang]
year: 2026
venue: arXiv
arxiv: "2602.09456"
tags: [contextual-bandits, oracle-efficiency, offline-regression, regret-bounds, exploration-exploitation, doec]
source: raw/papers/Qin2026Taming.pdf
---

# Taming the Monster Every Context

**TL;DR:** Introduces OE2D, the first algorithm achieving $\sqrt{T}$-regret for contextual bandits with general function approximation using only $O(\log T)$ calls to an offline regression oracle, along with a new complexity measure DOEC that characterizes when this is feasible.

## Problem

Prior offline-oracle-efficient contextual bandit algorithms had a fundamental gap: either they worked only for restricted settings — discrete actions (FALCON) or linear rewards (Linear FALCON) — or they required $O(T)$ oracle calls for general function classes (UCCB, E2D.Off), which is computationally prohibitive. No unified framework existed for **general reward function approximation** with **few oracle calls**. Additionally, no formal connection existed between the design principles of offline- and online-oracle-efficient algorithms.

## Method

**Algorithm: OE2D (Offline Estimation to Decision)**

Operates in epochs. Each epoch $m$:
1. Call offline oracle on previous epoch's data → reward estimate $\hat{f}_m$
2. For each context $x_t$, solve the **exploitative F-design** problem (Eq. 3) → action distribution $p_t$
3. Sample $a_t \sim p_t$ and observe reward $r_t$

Total oracle calls: $O(\log T)$; reducible to $O(\log \log T)$ with small-epoch schedule when $T$ is known.

**Exploitative F-design** (the core primitive):

$$
p_t = \arg\min_{p \in \mathrm{co}(\Lambda)} \max_{\lambda \in \Lambda}
\left[ \mathbb{E}_{a \sim \lambda}[\hat{f}_m(x_t, a)] - \mathbb{E}_{a \sim p}[\hat{f}_m(x_t, a)] + \frac{1}{\gamma_m} \mathrm{Coverage}_\varepsilon(p, \lambda;\, F_{x_t}) \right]
$$

- First term: exploitation — $p_t$ should maximize expected reward under $\hat{f}_m$
- Second term: exploration — $p_t$ should cover all benchmark distributions $\lambda \in \Lambda$

The solution $p_t$ simultaneously satisfies two key properties (Lemma 1):
- **Low Regret (LR):** $\widetilde{\mathrm{Reg}}_m(p_t \mid x_t) \leq \mathrm{doec}_{\gamma,\varepsilon}(F_{x_t}, \Lambda)$
- **Good Coverage (GC):** $\mathrm{Coverage}_\varepsilon(p_t, \lambda;\, F_{x_t}) \leq \gamma \cdot \mathrm{doec} + \gamma \cdot \widetilde{\mathrm{Reg}}_m(\lambda \mid x_t),\quad \forall \lambda \in \Lambda$

**DOEC — new complexity measure (Definition 2):**

$$
\mathrm{doec}_{\gamma,\varepsilon}(G, \Lambda) = \max_{\hat{g} \in G}\ \min_{p \in \mathrm{co}(\Lambda)}\ \max_{\lambda \in \Lambda}
\left[ \mathbb{E}_{a \sim \lambda}[\hat{g}(a)] - \mathbb{E}_{a \sim p}[\hat{g}(a)] + \frac{1}{\gamma} \mathrm{Coverage}_\varepsilon(p, \lambda;\, G) \right]
$$

Unlike [[dec]], DOEC does not reference the ground-truth reward — enabling actual reduction to offline regression.

## Results

Regret of OE2D (finite $\mathcal{F}$, ERM oracle):

| Setting | Regret | Notes |
|---------|--------|-------|
| Discrete action space | $\tilde{O}\!\left(\sqrt{\lvert\mathcal{A}\rvert T \log\lvert\mathcal{F}\rvert}\right)$ | Matches FALCON |
| Per-context linear reward (dim $d$) | $\tilde{O}\!\left(\sqrt{dT \log\lvert\mathcal{F}\rvert}\right)$ | Matches Linear FALCON |
| $h$-smoothed regret | $\tilde{O}\!\left(\sqrt{T/h \cdot \log\lvert\mathcal{F}\rvert}\right)$ | **First offline-oracle-efficient result** |

**DOEC-DEC bridge (Theorem 5):** Any distribution $p$ certifying $\mathrm{doec} \leq V$ also certifies $\mathrm{dec} \leq V + 1/\gamma + \gamma\varepsilon$. Consequently $\mathrm{dec}_\gamma(G,\Lambda) \leq \mathrm{doec}_{\gamma,\varepsilon}(G,\Lambda) + \text{lower-order terms}$. First formal connection between offline- and online-oracle algorithm design principles.

**Extensions** (Appendix B): model misspecification, adversarial reward corruption, context distribution shift (first offline-oracle result beyond iid contexts).

**DOEC bounds:**
- $\varepsilon$-SEC upper bounds DOEC: $\mathrm{doec} \leq (10/\gamma)\,\mathrm{SEC}_\varepsilon$ (Theorem 3)
- $\varepsilon$-SEC is bounded by Eluder dimension: $\mathrm{SEC}_\varepsilon \lesssim \mathrm{Edim}(G, \sqrt{\varepsilon}) \cdot \log^2(1/\varepsilon)$ (Proposition 1)
- $\varepsilon$-SEC can be exponentially loose vs. DOEC (Proposition 3) — active design matters

## Strengths & Limitations

**Strengths:** Unified framework; first general function class + offline oracle + $O(\log T)$ calls result; $h$-smoothed regret result is genuinely new; DOEC-DEC bridge is conceptually important.

**Limitations:** Exploitative F-design is a minimax problem per context — computational cost depends on function class. Lower bounds on DOEC are open. Finer structural characterizations of DOEC (e.g., via star number) are open.

## Connections

- [[doec]] — new complexity measure introduced here; governs OE2D's regret
- [[dec]] — existing measure for online-oracle algorithms; shown to be $\geq$ DOEC (up to lower-order terms)
- [[exploitative-f-design]] — the key algorithmic primitive
- [[offline-regression-oracle]] — the computational model
- [[oracle-efficiency]] — the broader research context
- [[contextual-bandits]] — the problem setting
- [[epsilon-sec]] — bounds DOEC from above (Theorem 3)
- [[Zhang-Chicheng]] — second author (advisor)
- [[Qin-Hao]] — first author
- **Generalizes:** [[SimchiLevi2022Bypassing]] (FALCON), Linear FALCON (Xu & Zeevi 2020)
- **Offline-oracle counterpart of:** E2D (Foster et al. 2021a)

## Open Questions

- Do lower bounds on DOEC imply information-theoretic lower bounds for contextual bandit learning?
- Can DOEC be characterized via the value function star number (cf. Foster et al. 2020b)?
- Can OE2D extend to partial monitoring, RL, RLHF (Li et al. 2025; Wang et al. 2023)?
- Is there a first-order offline-oracle-efficient algorithm (cf. Foster & Krishnamurthy 2021)?
- Are there tighter structural characterizations of DOEC beyond $\varepsilon$-SEC?
