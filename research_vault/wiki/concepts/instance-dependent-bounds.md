---
title: "Instance-Dependent Bounds"
tags: [sample-complexity, learning-theory, bandits, reinforcement-learning, minimax]
introduced_by: [[Yin2023Offline]]
---

# Instance-Dependent Bounds

**Definition:** A performance guarantee whose leading term is a functional of the *particular* problem instance — gaps, variances, coverage, gradient geometry — rather than a supremum over a class of instances. Two problems with the same worst-case parameters receive different, and differently sized, bounds.

> Hub page. This is a guarantee *type* rather than a single technique, and most papers in this vault are best read through it. Individual instantiations live on their own pages.

## Intuition

A minimax bound answers "how bad can this get over all instances in the class?" A useful answer for establishing that a problem is learnable at all, and the right currency for lower bounds. But it is silent on the question a practitioner actually has: *is my problem easy?*

The classic symptom, noted by Zanette & Brunskill (2019): algorithms routinely perform far better than their problem-independent bounds predict, because real instances are not the adversarial ones. A worst-case bound cannot express that, so it cannot guide algorithm design toward exploiting easiness.

An instance-dependent bound replaces the sup with a functional evaluated at the instance:

$$
\underbrace{\sup_{\mathcal{M}\in\mathfrak{M}} \text{Err}(\mathcal{A},\mathcal{M}) \le \frac{C(\mathfrak{M})}{\sqrt{n}}}_{\text{worst-case}}
\qquad\longrightarrow\qquad
\underbrace{\text{Err}(\mathcal{A},\mathcal{M}) \le \frac{\mathcal{C}(\mathcal{M})}{\sqrt{n}}\ \ \forall \mathcal{M}}_{\text{instance-dependent}}
$$

The bar is that $\mathcal{C}(\mathcal{M})$ be **computable from the instance** and **genuinely discriminating** — a bound that reduces to the worst case for every instance has gained nothing. The strongest form is *instance-optimality*: a matching lower bound at each instance, not merely over the class.

## Formal Description

The instance functional takes recognizably different forms by setting.

**Gap-dependent (bandits, BAI).** Hardness is carried by reward gaps $\Delta_i = \mu^* - \mu_i$: UCB's $O(\sum_i \log T/\Delta_i)$ regret, and in fixed-confidence BAI the characteristic time $T^*(\mu)$ that appears in the $\log(1/\delta)$ lower bound. Track-and-stop-style algorithms are asymptotically optimal *at each instance*. See [[best-arm-identification]].

**Variance-dependent.** Replace a range bound $V_\text{max}$ with the actual conditional variance, so low-noise instances converge faster — Bernstein rather than Hoeffding. In offline RL this is the difference between $H$ and $\sqrt{\text{Var}}$ scaling, and it collapses to a fast $O(1/K)$ rate on deterministic transitions.

**Geometry-dependent (function approximation).** Hardness is the position of the target policy's features relative to the data's information matrix:

$$
\sum_{h=1}^H \mathbb{E}_{\pi^*}\Big[\sqrt{g^\top \Sigma_h^{-1} g}\Big],
\qquad \Sigma_h = \sum_k g_k g_k^\top + \lambda I,
$$

with $g = \phi$ in linear MDPs and $g = \nabla_\theta f$ under [[differentiable-function-approximation]]. The reciprocal $(g^\top\Sigma_h^{-1}g)^{-1}$ is an effective sample size in the direction that matters.

**Coverage-dependent.** How well the data (or a reference policy) covers what you need to learn — concentrability, single-policy concentrability, and [[coverage-coefficient]] $C_\text{cov}$ for LM alignment. Distinct from the above in that it is a property of the *data source* jointly with the target, not of the reward structure.

## Key Papers

Instance-dependent results in this vault, by setting:

- [[Yin2023Offline]] — offline RL under nonlinear function approximation; Fisher-information-style gradient geometry; the reason this page exists. Includes a lower bound matching to $\sqrt{d}$
- [[Ryu2025Improved]] — PUB: parameter-free *variance-adaptive* off-policy selection via betting-based LCBs; adapts to the instance without knowing its variance
- [[Kanarios2024Cost]] — CABAI: optimal sampling proportions scale as $\sqrt{c_a}$, a property of the instance's cost profile
- [[Lardy2025Constrained]] — CBAI: asymptotically optimal track-and-stop for three model families
- [[Yang2025Stochastically]] — BFAI-TS: instance-dependent exponential decay rate $\Gamma_{\beta^*}$ in the fixed-budget regime
- [[TranThanh2012Knapsack]] — $O(\ln B)$ budget-limited regret with a *matching* lower bound
- [[Dam2024Power]] — $\mathcal{O}(n^{-1/2})$ convergence for stochastic MCTS
- [[Foster2025Foundation]] — $C_\text{cov}$ as necessary *and* sufficient: an instance functional that is tight in both directions

Papers whose central open problem is the *absence* of such a bound:

- [[Kostrikov2022Offline]] — IQL's Theorem 3 is asymptotic in $\tau$ with a binary support condition; no bound at the $\tau$ actually used, and no instance functional to explain why $\tau=0.9$ is needed on antmaze but not locomotion
- [[Song2019Revisiting]] — has a finite-$\tau$ gap bound, but under stylized assumptions (all-equal $Q^*$, i.i.d. noise) rather than instance quantities

## Variants & Related Concepts

- **Minimax / worst-case bounds** — the complement; the right tool for establishing lower bounds and for comparing algorithm classes, and what instance-dependent results are measured against
- **Instance-optimality** — the strong form: a matching per-instance lower bound. Achieved asymptotically in BAI, and up to $\sqrt{d}$ in [[Yin2023Offline]]
- **Complexity measures** — [[dec]], [[doec]], Eluder dimension: instance functionals for whole problem classes rather than single instances
- [[differentiable-function-approximation]] — chosen precisely because its structure supports an instance functional
- [[coverage-coefficient]] — the coverage-flavored member of this family

## Current State

The dominant standard in bandit theory, where instance-optimality is routinely achieved and expected. Considerably less settled in RL with function approximation: linear models have had instance-dependent treatments for several years, [[Yin2023Offline]] extends this to a nonlinear class, and general function approximation remains worst-case only. Two live frontiers:

1. **Empirical validation.** These functionals are proved to separate instances, but almost never *measured*. Whether $\sum_h\mathbb{E}_{\pi^*}[\|\nabla_\theta f\|_{\Sigma_h^{\star-1}}]$ predicts observed difficulty on a real benchmark is untested and cheap to test.
2. **Reaching the empirically successful algorithms.** The methods with instance-dependent guarantees (PFQL, PEVI) are not the methods people run (IQL, CQL, TD3+BC). Producing an instance-dependent bound for an algorithm someone actually uses would be worth more than tightening one for an algorithm nobody does.
