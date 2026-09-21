---
title: "Contextual Bandits (Online)"
aliases: [contextual bandits, contextual-bandits]
tags: [bandits, sequential-decision-making, reinforcement-learning]
---

# Contextual Bandits

**Definition:** A sequential decision-making problem where at each round $t$, a learner observes context $x_t \in \mathcal{X}$, takes action $a_t \in \mathcal{A}$, and receives reward $r_t$. The goal is to minimize cumulative regret versus the best action per context in hindsight. *Online* is marked in the name because the same problem class also has a batch protocol, [[contextual-bandits-offline]].

> **Scope.** The interactive protocol only — the learner acts, observes its own reward, and pays for what it does not try. **Left to other pages:** the batch variant, [[contextual-bandits-offline]], where that feedback loop is absent and coverage replaces exploration; the reduction to supervised learning, [[oracle-efficiency]]; the realizability assumption itself, [[realizability]]. The Provenance section at the end says what is sourced and what is not.

## Intuition

A one-step version of online RL: no planning beyond the current step, but must balance **exploration** (learning $f^*$) with **exploitation** (taking high-reward actions). Unlike multi-armed bandits, the best arm can vary by context.

## Formal Description

- Context space $\mathcal{X}$, action space $\mathcal{A}$, unknown reward $f^*: \mathcal{X} \times \mathcal{A} \to [0,1]$
- Realizability: $f^* \in \mathcal{F}$ (learner's reward function class)
- Contexts drawn iid from $\mathcal{D}_\mathcal{X}$ (unless context-shift setting)

$$\Lambda\text{-Regret}(T, \mathrm{ALG}) = \sum_{t=1}^T \mathrm{Reg}(p_t \mid x_t), \quad \mathrm{Reg}(p \mid x) = \max_{\lambda \in \Lambda}\,\mathbb{E}_{a \sim \lambda}[f^*(x,a)] - \mathbb{E}_{a \sim p}[f^*(x,a)]$$

Benchmark class $\Lambda$ determines the regret notion:
- $\Lambda = \{\delta_a : a \in \mathcal{A}\}$ → standard regret (best fixed action per context)
- $\Lambda = \Delta_h^\mu(\mathcal{A})$ → $h$-smoothed regret (best $h$-smooth distribution per context)

## Literature Survey

Three directions, all concerned with the interactive problem. The first is the vault's main contextual-bandit thread; the third is a stub.

### Oracle-efficient learning under realizability

The organizing question: contextual bandits are computationally hard in general (the direct problem is related to agnostic classification), so can the problem be reduced to a *small number of calls* to a supervised-learning oracle? This is the vault's main contextual-bandit thread; [[oracle-efficiency]] holds it in full.

The split that matters is which oracle. An **online** regression oracle sees a stream and needs $O(T)$ calls — the route taken by SquareCB (Foster & Rakhlin 2020) and E2D (Foster et al. 2021a), with [[decision-estimation-coefficient]] as the governing complexity measure. An **offline** oracle takes a batch and is satisfied by ordinary ERM, so any supervised learner qualifies; this is [[offline-regression-oracle]], and it is the practically useful one.

On the offline route the progression is legible: FALCON ([[SimchiLevi2022Bypassing]], 2022) achieved $O(\log T)$ calls for discrete actions under realizability, and OE2D ([[Qin2026Taming]], 2026) reached general action spaces at the same call count while dropping realizability. Its complexity measure is [[decision-offline-estimation-coefficient]], and its algorithmic primitive is [[exploitative-f-design]], whose minimax value DOEC turns out to be. How to bound DOEC in a given class is unsettled: the [[epsilon-sec]] route is simplest but admits an exponential gap, and the [[eluder-dimension]] route covers only discrete benchmarks.

### What governs the achievable regret

A parallel line asks not how to compute the policy but what quantity decides how well any algorithm can do. [[decision-estimation-coefficient]] (Foster et al. 2021a) is the answer for the online oracle: it measures the cost of exploration as a minimax game between choosing an action distribution and an adversary picking the reward function, and it upper-bounds the regret of E2D. Its structural limitation is that the exploration cost references the unknown $g^*$, which is what blocks a reduction to batch regression — [[decision-offline-estimation-coefficient]] removes that dependence, and Theorem 5 of [[Qin2026Taming]] relates the two for the first time.

Bounding either measure for a concrete class is the open part. [[epsilon-sec]] is a passive coverage measure that upper-bounds DOEC but can be exponentially loose, which is the formal statement that *active* experimental design beats passive coverage. [[eluder-dimension]] gives the other route and applies when the benchmark is discrete.

### Alignment and language-model applications

Contextual-bandit machinery transfers to language-model alignment, with coverage again in the role exploration plays online. [[Foster2025Foundation]] argues coverage is necessary and sufficient for computationally efficient alignment, and that [[spanner-sampling]] matches the [[coverage-coefficient]] lower bound while training-time interventions are ETH-hard; [[linear-softmax-policy]] is the policy class in that line. One paper and three concepts — thin, and the weakest-supported direction here.

## Variants

- [[contextual-bandits-offline]] — the batch variant: one fixed logged dataset, no interaction, coverage in place of exploration
- $h$-smoothed regret — the benchmark class $\Lambda$ above, which changes what "best action" means and with it the achievable rates
- Linear and discrete-action instantiations — the settled cases, against which the general-function-approximation work in §1 of the survey is measured

## Related Concepts

- [[oracle-efficiency]] — computational tractability via oracle reductions, and the hub for the first direction above
- [[offline-regression-oracle]] — the practical oracle model
- [[realizability]] — the assumption every result in the first direction rests on
- [[importance-weighting]] — core primitive for off-policy evaluation
- [[decision-estimation-coefficient]] / [[decision-offline-estimation-coefficient]] — the online and offline complexity measures
- [[upper-confidence-bound]] — the optimism principle in the context-free case

## Current State and Open Problems

Mature for the simple settings — linear rewards, discrete actions — where the algorithms and their rates are settled. The live frontier in this vault is general function approximation with few oracle calls, which [[Qin2026Taming]] largely closes for the offline oracle.

What is open, in the vault's own terms:

- **Bounding DOEC.** Whether $\varepsilon$-SEC or Eluder dimension is the right route for a given class; Proposition 3 of [[Qin2026Taming]] constructs an exponential gap for the first.
- **Empirical behaviour.** Every oracle-efficiency result here is theoretical; the vault has no source on how these algorithms actually perform.
- **Non-iid contexts and partial monitoring.** Named as frontiers but with no page and no paper behind them.

## Provenance

*Sourced.* Everything attributed to [[Qin2026Taming]], [[SimchiLevi2022Bypassing]] and [[Foster2025Foundation]] comes from their paper pages, written against the PDFs at ingest.

*Cited author–year, no paper page.* SquareCB (Foster & Rakhlin 2020) and E2D (Foster et al. 2021a), including the attribution of DEC to the latter. [[decision-estimation-coefficient]] records the same gap; neither has been checked directly.

*This page's judgment, not a citation.* That the area is "mature for linear and discrete actions", and the relative weight given to the three directions, which reflects what this vault happens to hold rather than the field's own proportions.
