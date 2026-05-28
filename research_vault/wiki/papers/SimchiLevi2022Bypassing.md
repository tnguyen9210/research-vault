---
title: "Bypassing the Monster: A Faster and Simpler Optimal Algorithm for Contextual Bandits under Realizability"
authors: [David Simchi-Levi, Yunzong Xu]
year: 2022
venue: Mathematics of Operations Research
arxiv: "2003.12699"
tags: [contextual-bandits, oracle-efficiency, offline-regression-oracle, realizability, falcon]
source: raw/papers/SimchiLevi2022Bypassing.pdf
---

# Bypassing the Monster

**TL;DR:** Introduces FALCON, the first algorithm achieving optimal $\tilde{O}(\sqrt{KT\log|\mathcal{F}|})$ regret for stochastic contextual bandits with general function approximation using only $O(\log T)$ calls to an offline regression oracle — resolving an open problem from 2012 under the [[realizability]] assumption.

## Problem

Two prior lines of work each achieved one of two desiderata but not both:
- **Computationally tractable but suboptimal:** RegCB (Foster et al. 2018) uses $O(T^{3/2})$ offline oracle calls and achieves $\tilde{O}(T)$ regret in the worst case.
- **Optimal but intractable:** Regressor Elimination (Agarwal et al. 2012) achieves $\tilde{O}(\sqrt{KT\log|\mathcal{F}|})$ but requires $\Omega(|\mathcal{F}|)$ computation per round.
- **Online oracle, optimal:** SquareCB (Foster & Rakhlin 2020) achieves optimality but requires an *online* regression oracle — a strictly stronger and less practical assumption than offline.

Open question since Agarwal et al. (2012): *Is there an offline-regression-oracle-based algorithm achieving optimal regret for general contextual bandits?*

## Method

**FALCON (and FALCON+)** operates in epochs. Key elements:

- **Realizability assumption:** true reward $f^*(x,a) = \mathbb{E}[r(a) \mid x]$ belongs to function class $\mathcal{F}$ (Assumption 1).
- **Dual interpretation:** algorithm implicitly maintains a *dense distribution over all (possibly improper) policies* — no explicit enumeration of $\mathcal{F}$ needed.
- **Offline oracle:** standard least-squares regression oracle; not cost-sensitive classification (which is NP-hard in general).
- **$O(\log T)$ calls:** epoch schedule with doubling — oracle called once per epoch; FALCON+ reduces to $O(\log \log T)$ when $T$ is known.

The algorithm achieves all sufficient conditions for regret optimality in the universal policy space *implicitly*, bypassing the "monster" computational barrier. The realizability assumption is what makes this implicit satisfaction possible.

**Regret guarantee (finite $\mathcal{F}$):**

$$
\mathrm{Reg}_T \leq \tilde{O}\!\left(\sqrt{KT \log|\mathcal{F}|}\right)
$$

**Comparison table (Table 1 of paper):**

| Algorithm | Regret | Oracle calls | Oracle type |
|-----------|--------|-------------|-------------|
| Regressor Elimination (2012) | optimal | $\Omega(\lvert\mathcal{F}\rvert)$ per round | — |
| ILOVETOCONBANDITS (2014) | optimal | $\tilde{O}(\sqrt{KT/\log\lvert\Pi\rvert})$ | classification |
| RegCB (2018) | suboptimal | $O(T^{3/2})$ | offline regression |
| SquareCB (2020) | optimal | $O(T)$ | **online** regression |
| **FALCON (this paper)** | **optimal** | **$O(\log T)$** | **offline** regression |

## Results

- Finite $\mathcal{F}$, i.i.d. contexts: $\tilde{O}(\sqrt{KT\log|\mathcal{F}|})$ regret with $O(\log T)$ offline regression oracle calls
- Known $T$ (FALCON+): oracle calls further reduced to $O(\log \log T)$
- Infinite $\mathcal{F}$: regret bounded by a function of the offline estimation error; universally optimal when oracle is statistically optimal

## Strengths & Limitations

**Strengths:** First paper to resolve the open problem of optimal + offline-oracle-efficient contextual bandits. Algorithm is simple. Analysis via dual policy distribution is novel and elegant.

**Limitations:** Requires [[realizability]] ($f^* \in \mathcal{F}$) — no misspecification tolerance. Restricted to discrete (finite) action spaces. [[Qin2026Taming]] later removes both restrictions via DOEC and exploitative F-design.

## Connections

- [[realizability]] — key assumption enabling the result; without it, FALCON's implicit optimality conditions do not hold
- [[oracle-efficiency]] — FALCON is the milestone offline-oracle result for discrete actions
- [[offline-regression-oracle]] — the oracle type; standard ERM qualifies
- [[contextual-bandits]] — the problem setting
- [[offline-oracle-efficient-bandits]] — this paper's central place in that research line
- [[Simchi-Levi-David]] — first author
- [[Xu-Yunzong]] — second author
- **Generalized by:** [[Qin2026Taming]] — extends to general action spaces and drops realizability via DOEC

## Open Questions (as of this paper)

- Can FALCON extend beyond discrete action spaces? *(resolved by [[Qin2026Taming]])*
- Can realizability be relaxed to misspecification tolerance? *(partially addressed in [[Qin2026Taming]] appendix)*
- First-order algorithms (sub-$\sqrt{T}$ regret under favorable gap conditions)?
