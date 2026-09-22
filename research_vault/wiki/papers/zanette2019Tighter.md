---
title: "Tighter Problem-Dependent Regret Bounds in Reinforcement Learning without Domain Knowledge using Value Function Bounds"
authors: [Andrea Zanette, Emma Brunskill]
year: 2019
venue: ICML
arxiv: "1901.00210"
tags: [online-reinforcement-learning, exploration, optimism, regret-bounds, instance-dependent-bounds]
citekey: zanette2019Tighter
---

# Tighter Problem-Dependent Regret Bounds in RL

**TL;DR:** EULER is an optimistic tabular RL algorithm whose regret automatically tightens on easy MDPs — scaling with the variance of the optimal value function rather than the horizon — while still matching the worst-case minimax rate, and crucially **without being told** which regime it is in. As a by-product it answers an open problem of Jiang & Agarwal (2018): with total reward bounded by 1, regret has no horizon dependence in the dominant term.

## Problem

Worst-case regret bounds for episodic RL are tight but pessimistic: in practice algorithms do far better than they predict, and the bounds give no account of *what makes an MDP hard*. Problem-dependent bounds would, but prior ones came with a catch — REGAL (Bartlett & Tewari 2009) achieves $`\tilde O(\Phi S\sqrt{K\,TH})`$ where $`\Phi`$ upper-bounds the optimal value range, but $`\Phi`$ is **an input to the algorithm**. Requiring domain knowledge you do not have defeats the purpose. The question is whether one algorithm can be minimax-optimal on hard instances and automatically better on easy ones, with no prior knowledge of which it faces.

## Method

Notation follows [[contextual-bandits-offline]] §2 and the horizon-$`H`$ extension on [[offline-reinforcement-learning]]. **One translation needs care.** The paper writes $`A`$ for $`|\mathcal A|`$, which the vault writes $`K`$; and it writes $`K`$ for episodes and $`T \le KH`$ for the *total number of timesteps*, whereas the vault's $`T`$ **is** the episode count. So the paper's $`T`$ equals the vault's $`TH`$, and every rate below is restated in vault symbols: the paper's $`\sqrt{HSAT}`$ reads $`\sqrt{H^2SKT}`$ here.

**Setting.** Undiscounted finite-horizon episodic MDP $`(\mathcal S, \mathcal A, P, r, H)`$ with **stationary** transitions $`P(s'\mid s,a)`$ — no $`h`$-dependence, the one structural difference from [[cassel2026Quantile]] — and $`r(s,a) \in [0,1]`$. $`T`$ episodes of fixed length $`H`$, so $`TH`$ timesteps in all. Policies and value functions are time-indexed, $`V^\pi_h`$.

**The two instance quantities.** The **maximum conditional value variance**

```math
\mathcal Q^*_{\mathrm{ZB}} \;:=\; \max_{s,a,h}\Big[ \mathrm{Var}\,r_h(s,a) \;+\; \mathrm{Var}_{s'\sim P(\cdot\mid s,a)} V^*_{h+1}(s') \Big]
```

— identical to the **environmental norm** of Maillard et al. (2014) — and the **max return** $`G`$, a deterministic upper bound on $`\sum_{h} r_h(s_h,\pi_h(s_h))`$ over all policies and starting states (Definition 1).

**EULER** (Episodic Upper Lower Exploration in Reinforcement learning) is optimism-under-uncertainty with a two-part bonus: an empirical-Bernstein estimate of the conditional value variance, plus a correction term that explicitly accounts for value-function uncertainty by maintaining **both** a pointwise over- and under-estimate of $`V^*`$ — the upper and lower of the name. Azar et al. (2017) used a Bernstein–Friedman reward bonus with a different correction; the change of correction term is what converts a worst-case bound into a problem-dependent one. Same computational complexity as value iteration.

## Results

**Theorem 1.** W.p. $`\ge 1-\delta`$, the regret $`\mathrm{Reg}(T)`$ of EULER is at most the **minimum** of

```math
\tilde O\Big(\sqrt{\mathcal Q^*_{\mathrm{ZB}}\, S K T H} + \sqrt S\,SKH^2(\sqrt S + \sqrt H)\Big)
\quad\text{and}\quad
\tilde O\Big(\sqrt{G^2 S K T} + \sqrt S\,SKH^2(\sqrt S + \sqrt H)\Big).
```

Both forms are kept because the second is tighter than substituting $`\mathcal Q^*_{\mathrm{ZB}} \le G^2`$ by a factor $`H`$. **EULER is given neither $`\mathcal Q^*_{\mathrm{ZB}}`$ nor $`G`$.**

**Corollary 1.1 (worst case).** Since $`r \in [0,1]`$ gives $`G^2 \le H^2`$: $`\tilde O(\sqrt{H^2 S K T} + \sqrt S SKH^2(\sqrt S+\sqrt H))`$, matching the $`\Omega(\sqrt{H^2 S K T})`$ lower bound (Jaksch et al. 2010; Osband & Van Roy 2016) and the minimax bound of Azar et al. (2017) in the dominant term.

**Equation 13 — the horizon-free bound.** Jiang & Agarwal (2018) ask whether a horizon dependence is *necessary* in tabular episodic RL when rewards are positive with $`\sum_{h} r_h \in [0,1]`$ almost surely (the usual per-step-bounded setting is this one after dividing by $`H`$). That assumption forces $`G \le 1`$, and Theorem 1's second form gives

```math
\tilde O\Big(\sqrt{S K T} + \sqrt S\,SKH^2(\sqrt S + \sqrt H)\Big) ,
```

with **no $`H`$ in the dominant term** — evidence that long-horizon domains are not intrinsically harder when total reward is bounded.

**Which MDPs are easy (§6).** Classes with small $`\mathcal Q^*_{\mathrm{ZB}}`$, all handled by the same uninformed algorithm: deterministic domains, single-goal MDPs, and high-stochasticity domains. The paper also introduces $`\Phi_{\mathrm{succ}}`$, the range of $`V^*`$ restricted to the *successor states* of a given pair, and notes $`\Phi \ge \mathrm{rng}\,V^* \ge \Phi_{\mathrm{succ}}`$ — so the successor-restricted range is always the sharpest of the three, and EULER adapts to it without being told.

## Strengths & Limitations

**Strengths.** Adaptivity without domain knowledge is the substantive advance over REGAL: one algorithm, minimax-optimal in the worst case and automatically tighter on structured instances. The $`\min`$ of two bounds gives genuinely different handles on easy problems. Answering the Jiang–Agarwal question is a clean by-product rather than a forced application.

**Limitations.** Tabular and **stationary** transitions, so the rate is not directly comparable to time-inhomogeneous results. No experiments. The lower-order term $`\sqrt S SKH^2(\sqrt S + \sqrt H)`$ is large and dominates until $`T`$ is substantial. $`\mathcal Q^*_{\mathrm{ZB}}`$ is a max over all $`(s,a,h)`$, so a single high-variance pair anywhere in the MDP spoils the bound however benign the rest of the domain is.

## Connections

- [[online-reinforcement-learning]] — the setting page; this is the count-based branch of its optimism mechanism
- [[cassel2026Quantile]] — obtains a variance-dependent rate **without bonuses**, and positions itself against this paper. Two differences matter when comparing the rates: transitions are time-inhomogeneous there and stationary here, and its $`\mathcal Q^*`$ **sums** the per-step maxima over $`h`$ where this one takes a single max over $`(s,a,t)`$
- [[instance-dependent-bounds]] — the guarantee type; this is one of its two canonical online-RL instances
- [[upper-confidence-bound]] — the optimism principle and the bonus machinery, here refined with empirical Bernstein plus a value-uncertainty correction
- [[fqi]] — EULER is optimistic value iteration; the backup is the same template
- Maillard et al. (2014) for the environmental norm, Azar et al. (2017), Bartlett & Tewari (2009), Jiang & Agarwal (2018), Zhou et al. (2023) — cited author–year, no pages here

## Open Questions

- **Is the lower-order term necessary?** $`\sqrt S SKH^2(\sqrt S+\sqrt H)`$ is far from the leading term and the paper does not claim it tight.
- **Max versus sum.** $`\mathcal Q^*_{\mathrm{ZB}}`$ as a global max is fragile — one high-variance pair sets the bound. Whether a distributional or averaged instance measure gives a more useful account of hardness is open, and [[cassel2026Quantile]]'s summed version is a different answer rather than a strictly better one.
- **Beyond tabular and stationary.** The empirical-Bernstein bonus needs counts, which is exactly the obstacle motivating the ensemble alternative.
- **Does the instance measure predict empirical difficulty?** No experiments here; whether $`\mathcal Q^*_{\mathrm{ZB}}`$ or $`\Phi_{\mathrm{succ}}`$ tracks observed hardness is untested — the same gap [[offline-reinforcement-learning]] records on its side.
