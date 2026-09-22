---
title: "Offline Reinforcement Learning"
tags: [offline-reinforcement-learning, deep-reinforcement-learning, distributional-shift]
---

# Offline Reinforcement Learning

**Definition:** Learning a policy entirely from a fixed dataset $`\mathcal{D}`$ collected by some behavior policy $`\mu`$, with no further environment interaction. The binding constraint is not optimization but **distributional shift**: improving over $`\mu`$ requires knowing the value of actions $`\mu`$ did not take.

> **Scope.** The sequential ($`H > 1`$) problem and its algorithm families. **Left to other pages:** the $`H = 1`$ case, on [[contextual-bandits-offline]]; the algorithmic template nearly everything here specializes, on [[fqi]]; the mechanism the theory side rests on, on [[pessimism-principle]]; the specific failure mode, on [[extrapolation-error]] and [[overestimation-bias]].

## Intuition

The values of unseen actions must be extrapolated by a function approximator that has no data to constrain them. Worse, the $`\max`$ in the Bellman target selects *for* the largest extrapolation error, so the bias compounds through the backup rather than averaging out — see [[overestimation-bias]]. Every family below is a different answer to the same question: what do you do about a value you cannot check?

Nearly every algorithm here is a variant of one template, [[fqi]] — relabel the fixed batch with Bellman targets, refit by least squares, repeat. The template is regime-agnostic: FQI determines how you learn from data, offline versus online determines how you obtain it, and the Bellman update is byte-identical either way. What the offline regime removes is the feedback loop in which a wrong value gets tested and corrected, and this field's distinctive machinery — pessimism, conservatism, support constraints — exists to compensate for that absence. Reading the area as "FQI with one of its three steps modified" (target, regression, or policy extraction) is what makes the algorithm zoo tractable.

## Formal Description

Notation follows [[contextual-bandits-offline]] §2 and extends it to horizon $`H`$; only the extension is stated here. The behavior policy is $`\mu`$ throughout, as in the bandit case — the deep-RL literature's $`\mu`$ is the same object.

An episodic MDP $`(\mathcal S, \mathcal A, \rho, P, H)`$: a state space $`\mathcal S`$, with $`S := |\mathcal S|`$ in the tabular case; a finite action set $`\mathcal A`$ with $`K := |\mathcal A|`$; at each step $`h \in [H]`$ a reward kernel $`\rho_h(\cdot \mid s,a)`$ on $`[0,1]`$ with mean $`r_h(s,a) := \mathbb E[r \mid s,a,h]`$, and a transition kernel $`P_h(\cdot \mid s,a)`$. Episodes start from $`s_1 \sim \nu`$. A policy is $`\pi = (\pi_h)_{h \in [H]}`$ with $`\pi_h : \mathcal S \to \Delta(\mathcal A)`$. The learner sees only

```math
\mathcal D := \big\{(s^t_h, a^t_h, r^t_h, s^t_{h+1})\big\}_{t \in [T],\ h \in [H]},
```

$`T`$ trajectories collected by $`\mu`$, and outputs $`\hat\pi`$ without further interaction. Write $`d^\pi_h`$ for the distribution of $`(s_h, a_h)`$ under $`\pi`$, so $`d^\mu_h`$ is the distribution of the logged pairs at step $`h`$ — the step-$`h`$ version of the bandit page's $`d^\pi(x,a) = \nu(x)\,\pi(a \mid x)`$.

**Values.** $`Q^\pi_h(s,a)`$ and $`V^\pi_h(s)`$ are the expected reward-to-go from step $`h`$ under $`\pi`$; $`Q^*_h`$ and $`V^*_h`$ the optimal ones, with $`V^*_h(s) = \max_a Q^*_h(s,a)`$ and $`\pi^*`$ greedy in $`Q^*`$. The value of a policy is $`J(\pi) := \mathbb E_{s_1 \sim \nu}[V^\pi_1(s_1)]`$, and the objective is the same suboptimality as before,

```math
\Delta(\hat\pi) := J(\pi^*) - J(\hat\pi).
```

**The horizon-one correspondence.** At $`H = 1`$ the state is the context, $`\mathcal S = \mathcal X`$; there is no transition; $`r_1 = Q^*_1 = q^*`$ and $`V^*_1 = V^*`$; $`d^\pi_1 = d^\pi`$; and $`J`$ and $`\Delta`$ are the bandit page's exactly. Every quantity on this page reduces to its bandit counterpart this way, which is why [[contextual-bandits-offline]] is the right place to first meet coverage — there it appears without bootstrapping or error propagation.

**Coverage.** Since $`\pi^*`$ may visit pairs $`\mathcal D`$ never contains, $`\Delta(\hat\pi)`$ is not controllable without an assumption relating $`d^\mu_h`$ to $`d^{\pi^*}_h`$. Which one is assumed is the single largest axis of variation in the theory: uniform coverage bounds the ratio $`d^\pi_h / d^\mu_h`$ for *every* policy, single-policy coverage only for $`\pi^*`$, and the latter is what [[pessimism-principle]] buys. See [[coverage-coefficient]].

## Literature Survey

Four directions. The first two organize the empirical literature and are close to orthogonal — a method picks one answer from each. The third is the field's central embarrassment rather than a research direction, but it is what a reader most needs to know. The fourth is where the vault's own material is thinnest.

### How the distributional shift is controlled

Three answers, in increasing order of how much they restrict the value-learning step itself.

**Policy-constraint methods** (BCQ, BEAR, AWAC, TD3+BC) restrict how far $`\hat\pi`$ may deviate from $`\mu`$. **Value-regularization methods** (CQL, Fisher-BRC) leave the policy free but push $`Q`$ down on out-of-distribution actions. Both impose an explicit improvement-versus-safety trade-off, tuned by a coefficient. **In-sample methods** ([[implicit-q-learning]]) never query an unseen action at all: IQL fits an upper [[expectile-regression]] of $`Q(s,\cdot)`$ over the actions actually present, then extracts a policy by advantage-weighted regression. That dissolves the trade-off during value learning rather than balancing it — at the cost of targeting only the support-constrained optimum, which is weaker than $`\pi^*`$ whenever $`\mu`$ misses good actions entirely.

The theory side adds a fourth answer that the empirical literature largely does not use: **pessimism**, subtracting an uncertainty penalty before the value enters the next target ([[fqi-pessimistic]]). It is the only one of the four with a matching lower bound.

### How many dynamic-programming steps are taken

*Single-step* methods (Onestep RL, Decision Transformer) fit $`Q^{\mu}`$ or clone behavior directly. *Multi-step* methods iterate the Bellman backup. The difference is **stitching** — composing segments of distinct sub-optimal trajectories into a trajectory better than any in the data — which single-step methods structurally cannot do.

[[kostrikov2021Offline]] makes this the sharpest empirical distinction in the area: on D4RL antmaze-medium and antmaze-large, every single-step method scores $`\approx 0`$ while multi-step methods score 40–70. On locomotion tasks the gap nearly vanishes. That the gap is enormous on one benchmark family and negligible on another is the most useful unexplained fact here, and it is what makes "which regime am I in?" a live question for a practitioner rather than a taxonomy detail.

### What is actually provable, and for which algorithms

Practice and theory in offline RL have largely diverged, and the divergence is clean enough to state as a pair. The methods people run — IQL, CQL, TD3+BC — carry no instance-dependent guarantees. The methods with sharp guarantees — PEVI, PFQL — are not run.

[[yin2023Offline]] represents the strongest current theory: [[instance-dependent-bounds]] under a nonlinear ([[differentiable-function-approximation]]) class via pessimism, governed by the Fisher-information-style quantity $`\sum_h\mathbb{E}_{\pi^*}\big[\sqrt{\nabla_\theta f^\top\Sigma_h^{\star-1}\nabla_\theta f}\big]`$, with a variance-aware variant (VAFQL) that saves a factor $`H`$ and is minimax-optimal up to $`\sqrt{d}`$. [[kostrikov2021Offline]] represents the strongest current practice, with only an asymptotic guarantee: $`\tau \to 1`$ recovers the support-constrained optimum, and nothing is proved at the $`\tau \in \{0.7, 0.9\}`$ actually used.

The two make a useful pair precisely because they answer the same question with opposite strategies — penalize-the-uncertainty versus never-query-it — and neither has what the other has.

### The bandit and planning specializations

The $`H = 1`$ case, [[contextual-bandits-offline]], removes bootstrapping and error propagation, so the coverage story appears in isolation and the two algorithm families separate cleanly into value-based and policy-based. It is the right place to understand what coverage *is* before meeting it inside a backup. On the planning side, [[mcts]] faces the same max-under-uncertainty problem at a different point in the pipeline, and [[dam2024Power]] applies a smooth backup for the same reason IQL applies an expectile — the pattern is collected on [[smooth-aggregators]].

## Variants

- [[contextual-bandits-offline]] — the $`H = 1`$ case
- [[implicit-q-learning]] — the in-sample family's representative
- [[fqi-pessimistic]] — the pessimism family's representative, and the theory side's
- **Offline-to-online finetuning** — IQL's reported $`370.1 \to 473.7`$; the regime where the dataset is a starting point rather than the whole problem
- **Batch-constrained and conservative variants** (BCQ, BEAR, CQL, TD3+BC) — named throughout above, no pages here

## Related Concepts

- [[online-reinforcement-learning]] — the interactive counterpart: same MDP, same $`J`$ and $`\Delta`$, but the learner chooses what data it gets, so optimism replaces pessimism and regret replaces suboptimality
- [[fqi]] — the template nearly every algorithm here modifies
- [[extrapolation-error]] / [[overestimation-bias]] — the failure mode, and why the $`\max`$ compounds it
- [[pessimism-principle]] — the mechanism behind the provable branch
- [[coverage-coefficient]] — what makes $`\Delta(\hat\pi)`$ controllable at all
- [[instance-dependent-bounds]] — the guarantee type separating [[yin2023Offline]] from worst-case work
- [[smooth-aggregators]] — the cross-cutting pattern IQL's expectile instantiates
- [[deep-q-network]] / [[song2019Revisiting]] — the deep-RL machinery these methods build on
- [[test-time-scaling]] — inference-time compute as an alternative to a better trained policy

## Current State and Open Problems

The area has a working empirical playbook and a sharp theory, with very little overlap between them. The open problems below are mostly restatements of that gap from different directions; the two marked as cheap are the ones a reader could actually run.

- **Finite-$`\tau`$, finite-sample guarantees for in-sample methods.** IQL's Theorem 3 is asymptotic in $`\tau`$ and assumes exact solutions. No bound exists on $`\max_{a:\mu(a|s)>0} Q^*(s,a) - V_\tau(s)`$ at the $`\tau`$ values used in practice. [[yin2023Offline]] shows what such a bound looks like for the pessimism family; transporting it is the obvious target.
- **Theory for the algorithms people run.** PFQL has the guarantee, IQL has the benchmark numbers, neither has both. Closing this from either side — an instance-dependent bound for an in-sample method, or a competitive empirical evaluation of PFQL/VAFQL — would be the single most valuable contribution here.
- **Coverage conditions that survive overparameterization.** Uniform coverage requires parameter identifiability, which neural networks violate structurally through permutation and scaling symmetries. A quotient formulation — coverage on function space, or on parameter equivalence classes — is missing and looks tractable.
- **Is $`d`$ or $`\sqrt{d}`$ right for nonlinear classes?** [[yin2023Offline]]'s Theorem 4.2 leaves a $`\sqrt{d}`$ gap; whether the covering argument is loose or nonlinearity genuinely costs $`\sqrt{d}`$ is open.
- **Does the instance measure predict empirical difficulty?** *(cheap)* Nobody has checked whether $`\sum_h\mathbb{E}_{\pi^*}[\|\nabla_\theta f\|_{\Sigma_h^{\star-1}}]`$ tracks observed hardness on D4RL. It would directly link the two papers on this page.
- **When is stitching actually needed?** *(cheap)* The single-step/multi-step gap is enormous on antmaze and negligible on locomotion. A dataset statistic predicting which regime you are in would be more useful than either method.
- **Instance-dependent dataset quality.** Support ($`\mu(a|s) > 0`$) is binary and ignores how much mass sits near the maximizing action. A [[coverage-coefficient]]-style density-weighted quantity should govern both the achievable value and the required $`\tau`$, and would explain why $`\tau = 0.9`$ is necessary on antmaze but not on locomotion.
- **Hyperparameter selection without online evaluation.** $`\tau`$ and $`\beta`$ are tuned per domain against online returns, which offline RL is by definition not supposed to have. Offline model selection remains largely unsolved.
- **Unified theory of smooth maximization operators.** Expectile, softmax and power mean all replace $`\max`$ with a tunable smooth aggregator for closely related reasons, with no common analysis. See [[smooth-aggregators]].

## Provenance

*Sourced.* Everything attributed to [[kostrikov2021Offline]], [[yin2023Offline]], [[song2019Revisiting]] and [[foster2025Good]] comes from their paper pages, written against the PDFs at ingest — including the D4RL numbers and theorem references.

*Cited by name, no paper page.* BCQ, BEAR, AWAC, TD3+BC, CQL, Fisher-BRC, Onestep RL, Decision Transformer and PEVI. These are named to place the families and are taken from how [[kostrikov2021Offline]] positions itself; none has been checked at source, and no claim here rests on one individually.

*This page's judgment, not a citation.* The reading of the whole area as "FQI with one step modified", the grouping into four directions, the claim that the first two axes are close to orthogonal, and the two problems marked cheap.

*Converted.* This page was the topic `offline-reinforcement-learning` until 2026-09-21; the filename is unchanged, so inbound links were unaffected.
