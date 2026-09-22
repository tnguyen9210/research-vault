---
title: "Smooth Aggregators in Place of Max"
tags: [reinforcement-learning, value-estimation]
---

# Smooth Aggregators in Place of Max

**Definition:** The substitution of a smooth, one-parameter family interpolating between averaging and maximization for the hard $`\max`$ in a value update. The parameter tunes how aggressively the update commits to the best-looking action, and at its two limits the family recovers the mean and the maximum exactly.

> **Scope.** The pattern itself and what is known about it across instances. **Left to other pages:** each instance's own setting — [[mcts-power-mean]], [[softmax-bellman-operator]], [[expectile-regression]] — and the failure mode motivating all three, on [[overestimation-bias]].

## Intuition

Under estimation noise, the hard $`\max`$ selects *for* upward error: the action that looks best is disproportionately likely to be the one whose estimate is most inflated, so the maximum of noisy estimates is biased above the maximum of true values. A smooth aggregator averages some of that noise away, at the price of no longer targeting the greedy value. The parameter is the dial between the two errors, and in every instance below it is set empirically.

## Formal Description

Each instance replaces $`\max_a Q(s,a)`$ with an operator $`\mathcal{M}_\alpha`$ satisfying $`\mathcal{M}_\alpha \to \mathbb{E}`$ at one limit of $`\alpha`$ and $`\mathcal{M}_\alpha \to \max`$ at the other, and monotone in between:

| Instance | Aggregator | Parameter | Limits | Where |
|---|---|---|---|---|
| Power-mean MCTS backups ([[mcts-power-mean]]) | power mean | $`p`$ | avg $`p{=}1`$ → max $`p{\to}\infty`$ | [[dam2024Power]] |
| Softmax DQN targets ([[softmax-bellman-operator]]) | softmax / log-sum-exp | $`\tau`$ | avg $`\tau{\to}0`$ → max $`\tau{\to}\infty`$ | [[song2019Revisiting]] |
| Upper expectiles in offline RL ([[expectile-regression]]) | $`\tau`$-expectile | $`\tau`$ | mean $`\tau{=}0.5`$ → max $`\tau{\to}1`$ | [[kostrikov2021Offline]] |

## Literature Survey

The three instances arose independently, in three subfields, for the same stated reason. There is no unifying paper — the survey below is organized by what each instance actually proves, because that is where they differ most and where the gap is.

### The instances, and what each one proves

[[song2019Revisiting]] (softmax DQN targets) is the only instance supplying a **quantitative finite-parameter bound** on the gap to the max, together with exponential convergence in $`\tau`$. It is therefore the reference point: it shows that the bias a smooth aggregator introduces can be controlled explicitly rather than merely assumed small.

[[dam2024Power]] (power-mean MCTS) proves $`\mathcal{O}(n^{-1/2})`$ convergence for the estimator, and finds $`p = 2`$ consistently best empirically — but the parameter's optimal value is not connected to any property of the environment.

[[kostrikov2021Offline]] (upper expectiles, IQL) gives only the asymptotic limit: $`\tau \to 1`$ recovers the support-constrained optimum. Nothing is proved at the $`\tau \in \{0.7, 0.9\}`$ actually used, which is the weakest of the three positions and also the most consequential, since IQL is the most used of the three methods.

### Why it is the same pattern and not a coincidence

The three differ in where in the pipeline the $`\max`$ sits — a tree backup, a bootstrapped TD target, a regression objective over logged actions — and in what the noise is: sampling noise over rollouts, function-approximation error, and dataset sparsity respectively. What is common is the structure of the problem, not the setting: an expectation is estimated, a maximum is taken over the estimates, and the maximum inherits the estimates' upward tail. Any pipeline with those three steps admits the same fix, which is the reason to expect the pattern to recur beyond these three.

### What is missing

No common analysis exists. Each instance's bias–variance trade-off is analysed in its own terms, with its own noise model, and none of the three results transfers to another. That is a gap rather than a direction — but it is the specific gap a unifying paper would fill, and the three instances are close enough to make it look tractable.

## Variants

- [[mcts-power-mean]] — power-mean backups in tree search
- [[softmax-bellman-operator]] — softmax in the DQN target
- [[expectile-regression]] — upper expectiles for in-sample maximization
- **Mellowmax and other log-sum-exp variants** — the same family as the softmax instance, not separately paged here

## Related Concepts

- [[overestimation-bias]] — the failure mode all three address
- [[mcts]] — where the power-mean instance lives
- [[offline-reinforcement-learning]] — where the expectile instance lives; the softmax instance targets the same pathology in the online case
- [[fqi]] — the template the value-update instances modify

## Current State and Open Problems

A pattern with three independent confirmations and no theory of its own. The open problems are the same in each instance, which is itself the argument that they should be studied together.

- **A unified analysis.** Three parameterized interpolations from averaging to max, all motivated by estimation error under noise, with no common framework quantifying the bias–variance trade-off across them. This looks tractable and is not in the literature.
- **Instance-dependent parameter selection.** All three tune empirically. Is there a principled rule for $`p`$ as a function of environment stochasticity, a cooling schedule for the softmax $`\tau`$, or a data-dependent expectile level? Absent this, the pattern is a heuristic with three success stories.
- **Do smooth targets help modern value-based stacks?** Rainbow, SAC-style critics and distributional critics already mitigate overestimation by other means; whether the aggregator still buys anything on top is untested.

## Provenance

*Sourced.* Everything attributed to [[dam2024Power]], [[song2019Revisiting]] and [[kostrikov2021Offline]] comes from their paper pages, written against the PDFs at ingest, including the limits in the table and each paper's guarantee.

*This page's judgment, not a citation.* That these three are instances of one pattern — no paper says so — the argument in "Why it is the same pattern", and the claim that a unifying analysis looks tractable.

*Converted.* This page was the topic `smooth-aggregators` until 2026-09-21; the filename is unchanged, so inbound links were unaffected.
