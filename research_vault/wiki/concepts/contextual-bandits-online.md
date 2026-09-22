---
title: "Contextual Bandits (Online)"
aliases: [contextual bandits, contextual-bandits]
tags: [bandits, sequential-decision-making, reinforcement-learning]
---

# Contextual Bandits

**Definition:** A sequential decision-making problem where at each round $t$, a learner observes context $x_t \in \mathcal{X}$, takes action $a_t \in \mathcal{A}$, and receives reward $r_t$. The goal is to minimize cumulative regret versus the best action per context in hindsight. *Online* is marked in the name because the same problem class also has a batch protocol, [[contextual-bandits-offline]].

> **Scope.** The interactive protocol only — the learner acts, observes its own reward, and pays for what it does not try. **Left to other pages:** the batch variant, [[contextual-bandits-offline]], where that feedback loop is absent and coverage replaces exploration; the reduction to supervised learning, [[oracle-efficiency]]; the realizability assumption itself, [[realizability]]. The Provenance section at the end says what is sourced and what is not.

## Intuition

A one-step version of online RL: no planning beyond the current step, but must balance **exploration** (learning $q^*$) with **exploitation** (taking high-reward actions). Unlike multi-armed bandits, the best arm can vary by context.

## Formal Description

Notation follows [[contextual-bandits-offline]] §2, the vault's anchor for every bandit and RL page; only what this setting adds is stated here.

The instance is the offline one — context space $\mathcal X$, finite action set $\mathcal A$ with $K := |\mathcal A|$, context distribution $\nu \in \Delta(\mathcal X)$, reward kernel $\rho$ on $[0,1]$, and mean reward $q^*(x,a) := \mathbb E[r \mid x,a]$ with $V^*(x) := \max_a q^*(x,a)$ — but there is no behavior policy and no dataset. At each round $t = 1, \dots, T$ the learner observes $x_t \sim \nu$, chooses an action distribution $p_t \in \Delta(\mathcal A)$ from everything seen so far, plays $a_t \sim p_t$, and observes $r_t \sim \rho(\cdot \mid x_t, a_t)$. Nothing else is revealed: the rewards of the $K-1$ actions not played are never seen, which is what makes exploration necessary and what the batch setting never has to pay for. (Contexts are iid unless a context-shift variant is stated.)

**Realizability.** As offline: a class $\mathcal F \subseteq \{f : \mathcal X \times \mathcal A \to [0,1]\}$ with $q^* \in \mathcal F$. See [[realizability]].

**Regret.** The learner is judged cumulatively, against a benchmark class $\Lambda \subseteq \Delta(\mathcal A)$ of action distributions, using the offline page's shorthand $f(x,p) := \sum_a p(a) f(x,a)$:

$$
\mathrm{Reg}_\Lambda(T) := \sum_{t=1}^T \Big( \max_{\lambda \in \Lambda} q^*(x_t, \lambda) \;-\; q^*(x_t, p_t) \Big).
$$
The benchmark class fixes which regret is meant:
- $\Lambda = \{\delta_a : a \in \mathcal A\}$, the point masses — **standard regret**, against the best action per context, so the round-$t$ term is $V^*(x_t) - q^*(x_t, p_t)$;
- $\Lambda$ = the $h$-smooth distributions over $\mathcal A$ with respect to a fixed base measure — **$h$-smoothed regret**, in which the general-action-space results are stated.

**Regret against suboptimality.** The offline objective $\Delta(\hat\pi) = J(\pi^*) - J(\hat\pi)$ is one number about one output policy; regret is a sum over $T$ rounds about the whole trajectory of play. They meet through the context distribution. Writing $\pi_t$ for the round-$t$ policy (the rule $x \mapsto p_t$, fixed before $x_t$ is drawn), the conditional expectation of the round-$t$ term is exactly $\Delta(\pi_t)$, so $\mathbb E[\mathrm{Reg}(T)] = \sum_t \mathbb E[\Delta(\pi_t)]$, and the uniform mixture $\bar\pi$ of $\pi_1, \dots, \pi_T$ has $\Delta(\bar\pi) = \mathbb E[\mathrm{Reg}(T)]/T$. That online-to-batch conversion is how a regret bound becomes a suboptimality bound, and it is the bridge the oracle-efficiency line walks when it compares the two settings.

**Symbol collisions, kept deliberately.** $\Lambda$ here is the benchmark class, following Foster et al.; on [[contextual-bandits-offline-value-based]] it is the regularized Gram matrix, following the linear-bandit literature. Both are the field's own notation and the two never appear on one page. Likewise $\delta_a$ is a point mass while a bare $\delta$ is the confidence level.

## Literature Survey

Three directions, all concerned with the interactive problem. The first is the vault's main contextual-bandit thread; the third is a stub.

### Oracle-efficient learning under realizability

The organizing question: contextual bandits are computationally hard in general (the direct problem is related to agnostic classification), so can the problem be reduced to a *small number of calls* to a supervised-learning oracle? This is the vault's main contextual-bandit thread; [[oracle-efficiency]] holds it in full.

The split that matters is which oracle. An **online** regression oracle sees a stream and needs $O(T)$ calls — the route taken by SquareCB (Foster & Rakhlin 2020) and E2D (Foster et al. 2021a), with [[decision-estimation-coefficient]] as the governing complexity measure. An **offline** oracle takes a batch and is satisfied by ordinary ERM, so any supervised learner qualifies; this is [[offline-regression-oracle]], and it is the practically useful one.

On the offline route the progression is legible: FALCON ([[simchi-levi2022Bypassing]], 2022) achieved $O(\log T)$ calls for discrete actions under realizability, and OE2D ([[qin2026Taming]], 2026) reached general action spaces at the same call count while dropping realizability. Its complexity measure is [[decision-offline-estimation-coefficient]], and its algorithmic primitive is [[exploitative-f-design]], whose minimax value DOEC turns out to be. How to bound DOEC in a given class is unsettled: the [[epsilon-sec]] route is simplest but admits an exponential gap, and the [[eluder-dimension]] route covers only discrete benchmarks.

### What governs the achievable regret

A parallel line asks not how to compute the policy but what quantity decides how well any algorithm can do. [[decision-estimation-coefficient]] (Foster et al. 2021a) is the answer for the online oracle: it measures the cost of exploration as a minimax game between choosing an action distribution and an adversary picking the reward function, and it upper-bounds the regret of E2D. Its structural limitation is that the exploration cost references the unknown $g^* = q^*(x,\cdot)$, the true reward restricted to the current context, which is what blocks a reduction to batch regression — [[decision-offline-estimation-coefficient]] removes that dependence, and Theorem 5 of [[qin2026Taming]] relates the two for the first time.

Bounding either measure for a concrete class is the open part. [[epsilon-sec]] is a passive coverage measure that upper-bounds DOEC but can be exponentially loose, which is the formal statement that *active* experimental design beats passive coverage. [[eluder-dimension]] gives the other route and applies when the benchmark is discrete.

### Alignment and language-model applications

Contextual-bandit machinery transfers to language-model alignment, with coverage again in the role exploration plays online. [[foster2025Good]] argues coverage is necessary and sufficient for computationally efficient alignment, and that [[spanner-sampling]] matches the [[coverage-coefficient]] lower bound while training-time interventions are ETH-hard; [[linear-softmax-policy]] is the policy class in that line. One paper and three concepts — thin, and the weakest-supported direction here.

## Variants

- [[contextual-bandits-offline]] — the batch variant: one fixed logged dataset, no interaction, coverage in place of exploration
- $h$-smoothed regret — the benchmark class $\Lambda$ above, which changes what "best action" means and with it the achievable rates
- Linear and discrete-action instantiations — the settled cases, against which the general-function-approximation work in §1 of the survey is measured

## Related Concepts

- [[online-reinforcement-learning]] — the horizon-$H$ version: the same regret summed over episodes, with the per-context trade-off compounded through transitions
- [[oracle-efficiency]] — computational tractability via oracle reductions, and the hub for the first direction above
- [[offline-regression-oracle]] — the practical oracle model
- [[realizability]] — the assumption every result in the first direction rests on
- [[importance-weighting]] — core primitive for off-policy evaluation
- [[decision-estimation-coefficient]] / [[decision-offline-estimation-coefficient]] — the online and offline complexity measures
- [[upper-confidence-bound]] — the optimism principle in the context-free case

## Current State and Open Problems

Mature for the simple settings — linear rewards, discrete actions — where the algorithms and their rates are settled. The live frontier in this vault is general function approximation with few oracle calls, which [[qin2026Taming]] largely closes for the offline oracle.

What is open, in the vault's own terms:

- **Bounding DOEC.** Whether $\varepsilon$-SEC or Eluder dimension is the right route for a given class; Proposition 3 of [[qin2026Taming]] constructs an exponential gap for the first.
- **Empirical behaviour.** Every oracle-efficiency result here is theoretical; the vault has no source on how these algorithms actually perform.
- **Non-iid contexts and partial monitoring.** Named as frontiers but with no page and no paper behind them.

## Provenance

*Sourced.* Everything attributed to [[qin2026Taming]], [[simchi-levi2022Bypassing]] and [[foster2025Good]] comes from their paper pages, written against the PDFs at ingest.

*Cited author–year, no paper page.* SquareCB (Foster & Rakhlin 2020) and E2D (Foster et al. 2021a), including the attribution of DEC to the latter. [[decision-estimation-coefficient]] records the same gap; neither has been checked directly.

*This page's judgment, not a citation.* That the area is "mature for linear and discrete actions", and the relative weight given to the three directions, which reflects what this vault happens to hold rather than the field's own proportions.
