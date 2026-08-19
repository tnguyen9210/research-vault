---
title: "Extrapolation Error (Offline RL)"
tags: [offline-reinforcement-learning, distributional-shift, function-approximation, coverage, bellman-operator]
introduced_by: [[Kostrikov2022Offline]]
---

# Extrapolation Error (Offline RL)

**Definition:** The error incurred when a fitted $Q$-function is evaluated at state-action pairs the offline dataset does not support. Because the Bellman target maximizes over the *whole* action space while the regression only trains on *observed* actions, the two distributions differ — and the $\max$ preferentially selects whichever unsupported action the approximator happened to overvalue.

## Intuition

The single sentence that explains most of offline RL:

> **[[fitted-q-iteration]] trains $Q$ on the data distribution but queries $Q$ outside it.**

The regression step fits $Q_{k+1}$ on inputs $(s_i,a_i) \sim d^\mu$ — whatever the behavior policy actually did. The target step evaluates $\max_{a'\in\mathcal{A}} Q_k(s_i',a')$ — over *every* action, including ones $\mu$ never took. Nothing forces those two sets to coincide, and in offline RL they usually do not.

A fitted $Q$ is a **function**, not a lookup table, so it returns a number for any input. That is a feature: it is how FQI generalizes at all. It is also the hazard, because a number is produced whether or not there is evidence behind it.

### Two statements that must be kept apart

| Question | Answer |
|---|---|
| *Can* the fitted model produce $Q(s_3,B)$ when $(s_3,B)\notin\mathcal{D}$? | **Yes.** A function approximator is defined off its training points. |
| *Is* that value accurate? | **Not necessarily.** This is precisely the offline RL difficulty. |

Conflating the ability to *output* a value with having *statistical evidence for* it is the mistake the whole subfield is organized around avoiding.

## Formal Description

### Worked example

Let $\mathcal{A} = \{A, B, C\}$, $\gamma = 0.9$, and

$$
\mathcal{D} = \{(s_1,A,0,s_2),\ (s_2,A,0,s_3),\ (s_3,A,10,\text{terminal})\},
$$

with **no** samples of $(s_3,B)$ or $(s_3,C)$. Starting from $Q_0\equiv 0$, iteration 1 has targets $y(s_1,A)=0$, $y(s_2,A)=0$, $y(s_3,A)=10$. The regression loss directly constrains only

$$
Q_1(s_1,A),\qquad Q_1(s_2,A),\qquad Q_1(s_3,A),
$$

yet the iteration-2 target is

$$
y_2(s_2,A) = 0 + 0.9\max_{a'\in\{A,B,C\}} Q_1(s_3,a') .
$$

Suppose the fit happens to give $Q_1(s_3,\cdot) = (10, 4, 2)$. Then $y_2(s_2,A) = 0.9\cdot 10 = 9$ and all is well — but only because the unconstrained predictions came out small. Had the fit given $Q_1(s_3,B) = 20$ while the truth is $Q^*(s_3,B) = -5$, the target becomes $0.9\cdot 20 = 18$ instead of $9$, and that contaminated value propagates backward on every subsequent sweep.

Nothing in the algorithm detects the difference between these two cases.

### Why the model can predict $(s_3,B)$ at all

Because $Q_\theta : \mathcal{S}\times\mathcal{A}\to\mathbb{R}$ has shared parameters. The linear case makes it explicit:

$$
Q_\theta(s_3,B) = \phi(s_3,B)^\top\theta,
$$

where $\theta$ was estimated from entirely different examples. With a neural network over discrete actions, one forward pass on $s_3$ mechanically emits $|\mathcal{A}|$ numbers whether or not each action was observed there. The question is never *can it*, but *on what evidence*.

### Three coverage cases

Worth separating, because they are not equally dangerous:

1. **Exact pair observed.** $(s_3,B)$ appears in $\mathcal{D}$, many times. Directly data-supported; the safe case.
2. **Pair absent, similar data present.** $(s_3,B)\notin\mathcal{D}$ but $(s_4,B), (s_5,B),\dots$ are, with $s_4 \approx s_3$ in feature space. The fit may reasonably interpolate $Q(s_3,B)\approx Q(s_4,B)$. This is ordinary function approximation, and whether it works depends on smoothness, representation, and sample size — the usual generalization question.
3. **Pair far outside the data distribution.** Action $B$ is essentially never taken in states resembling $s_3$. The output is determined by initialization, regularization, parameter sharing, and accident. This is the dangerous case, and it is the one the theory is written for.

### Unseen pair vs. unseen action

A distinction that is easy to lose. If $B$ was never taken *at $s_3$* but is common elsewhere, $Q(s_3,B)$ may be estimable by generalization (case 2). If $B$ never appears *anywhere* in $\mathcal{D}$, there is no reward or transition information about it at all, and no function class has a principled basis for its prediction. The second is a far more severe coverage failure than the first, and "out-of-distribution action" is often used loosely for both.

### Tabular contrast

With a literal table, an unobserved $(s_3,B)$ has no estimate — you may initialize it to $0$, but that is an arbitrary choice, not an inference. There is no mechanism carrying information from $(s_4,B)$ to $(s_3,B)$. Function approximation is what creates generalization *and* what creates extrapolation; they are the same mechanism seen from two sides. This is why extrapolation error is a function-approximation phenomenon and does not arise in tabular analyses.

### Why the max makes it worse

Even a well-calibrated estimator with zero-mean error is dangerous here, because $\max$ is not an average. Suppose the truth is $Q^*(s_3,\cdot) = (10, 5, 4)$ while the estimates come out $(9.8, 7, 30)$ — the estimate for the poorly supported $C$ is absurd. FQI takes

$$
\max\{9.8,\ 7,\ 30\} = 30 .
$$

**Among uncertain predictions, the max preferentially selects those with large positive error.** Errors on unsupported actions are not averaged away; they are actively sought out. This is a selection effect on top of any estimation bias.

### Relation to [[overestimation-bias]]

These are related but distinct, and the vault keeps them on separate pages:

| | [[overestimation-bias]] | Extrapolation error |
|---|---|---|
| Setting | all actions observed, estimates noisy | some actions unsupported |
| Cause | $\mathbb{E}[\max_a(Q^* + \epsilon_a)] \ge \max_a Q^*$ | no evidence constrains $Q$ off-support |
| Present online? | yes | largely self-correcting online |
| Fix | DDQN, smooth operators ([[softmax-bellman-operator]]) | coverage assumptions, pessimism, in-sample targets |

Overestimation is a statistical bias that survives infinite data at each observed pair; extrapolation error is a *coverage* failure that infinite data on the wrong distribution never repairs.

### Why online RL escapes this

If an online agent believes $Q(s,B) = 100$, it can try $B$, observe $r = -2$, and correct. The optimistic estimate is self-refuting: acting on it generates the evidence that destroys it. Offline, the dataset is fixed. If $B$ is absent, nothing in the training loop can ever contradict the fabricated value — the same error is self-*reinforcing*. This is why optimism is right online and [[pessimism-principle]] is right offline. The underlying asymmetry is a property of the data-collection protocol, not of the Bellman regression; see the *Offline vs. online FQI* section of [[fitted-q-iteration]].

### Vanilla FQI really does maximize over unsupported actions

Worth stating plainly, because the alternative looks superficially similar. The target is

$$
y_i = r_i + \gamma\max_{a'\in\mathcal{A}} Q_k(s_i',a')
$$

and **not**

$$
y_i = r_i + \gamma\!\!\max_{\substack{a' :\ (s_i',a')\in\mathcal{D}}}\!\! Q_k(s_i',a') .
$$

Those are different algorithms. Vanilla FQI approximates the true Bellman optimality operator, so it maximizes over all of $\mathcal{A}$ by definition. The second expression is essentially the **in-sample** approach — and making that idea work smoothly, via an upper expectile rather than a hard restricted max, is exactly [[implicit-q-learning]].

### Why coverage assumptions appear in the theory

FQI returns $\pi_k(s) = \arg\max_a Q_k(s,a)$, which by construction may differ from $\mu$. So although the data come from $d^\mu$, the learned policy induces $d^{\pi_k}$. When $d^{\pi_k}$ places mass where $d^\mu$ has little, every guarantee must pay for extrapolation — which is why concentrability, single-policy concentrability, and [[coverage-coefficient]]-style conditions are unavoidable in this literature rather than technical conveniences.

### The pessimistic fix, numerically

Replace the point estimate with a lower confidence bound $Q_\text{LCB}(s,a) = \widehat{Q}(s,a) - b(s,a)$, where $b$ grows with uncertainty:

| $a$ | $\widehat{Q}(s_3,a)$ | $b(s_3,a)$ | $Q_\text{pess}(s_3,a)$ |
|---|---|---|---|
| $A$ | 10 | 1 | **9** |
| $B$ | 20 | 15 | 5 |
| $C$ | 8 | 10 | $-2$ |

Vanilla FQI takes $\max\{10,20,8\} = 20$ and commits to the unsupported $B$. Pessimistic FQI takes $\max\{9,5,-2\} = 9$ and picks $A$. The penalty does not make the estimate of $B$ more accurate — it makes the algorithm decline to act on an estimate it has no evidence for. See [[pessimistic-fitted-q-learning]] for the version where $b$ is the gradient-geometry width $\beta\|\nabla_\theta f\|_{\Sigma_h^{-1}}$.

**Where the penalty is applied matters as much as its size.** Subtracting $b$ only at policy extraction gives a cautious final policy computed from values that were *already* contaminated during the backups. To actually stop the propagation, the penalized $\hat Q$ must be what feeds the next Bellman target — so this table has to be evaluated at every stage, not once at the end. [[pessimistic-fitted-q-learning]] walks through Algorithm 1 of [[Yin2023Offline]] line by line on exactly this point.

## Key Papers

- Fujimoto, Meger & Precup (2019) — BCQ; names extrapolation error and isolates it as the central obstacle in batch RL
- Levine et al. (2020) — the offline RL survey that made distributional shift the organizing frame
- [[Kostrikov2022Offline]] — the in-sample answer: never evaluate an out-of-sample action, so extrapolation error cannot enter the target at all
- [[Yin2023Offline]] — the pessimism answer with an instance-dependent guarantee; the penalty is an effective-sample-size width along $\nabla_\theta f$
- Kumar et al. (2020) — CQL; pushes down $Q$ on out-of-distribution actions rather than penalizing at selection time
- Chen & Jiang (2019) — concentrability and completeness; where coverage conditions enter batch RL analysis formally

## Variants & Related Concepts

- [[fitted-q-iteration]] — the template in which the train/query mismatch arises
- [[overestimation-bias]] — the noise-driven cousin; see the comparison table above
- [[pessimism-principle]] — the general remedy: act on a lower confidence bound
- [[pessimistic-fitted-q-learning]] — pessimism instantiated with a computable width
- [[implicit-q-learning]] — the alternative remedy: restrict the max to in-sample actions, smoothed via [[expectile-regression]]
- [[coverage-coefficient]] — the quantity that measures how much extrapolation a dataset forces
- [[offline-reinforcement-learning]] — the setting

## Current State

The consensus diagnosis for why naive batch RL fails, and the organizing problem of the field. Two families of answer are established — penalize the uncertainty (pessimism, conservatism) or refuse the query (in-sample learning) — and the vault holds a strong representative of each. What remains unsettled is quantitative rather than conceptual: how to measure the degree of extrapolation a given dataset forces without an all-policy coverage assumption, and how to state a bound in terms of *density* near the actions that matter rather than a binary support indicator. That gap is exactly what separates [[Yin2023Offline]]'s uniform coverage and [[Kostrikov2022Offline]]'s support condition from the density-aware [[coverage-coefficient]].
