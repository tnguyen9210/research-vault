---
title: "Fitted Q-Iteration (FQI)"
tags: [reinforcement-learning, offline-reinforcement-learning, approximate-dynamic-programming, function-approximation, bellman-operator]
introduced_by: [[Yin2023Offline]]
---

# Fitted Q-Iteration (FQI)

**Definition:** An approximate dynamic programming template that turns value iteration into a sequence of supervised regressions: relabel a fixed batch of transitions with Bellman targets computed from the current $Q_k$, fit a new $Q_{k+1}$ to those targets by least squares, and repeat.

## Intuition

The whole algorithm is one sentence: **RL becomes repeated supervised learning on the same data.**

Given a batch $\mathcal{D} = \{(s_i,a_i,r_i,s_i')\}_{i=1}^N$ and a current estimate $Q_k$, each transition supplies an input $(s_i,a_i)$ and a target $y_i = r_i + \gamma\max_{a'}Q_k(s_i',a')$. Fitting $Q_{k+1}$ to those pairs is an ordinary regression problem — any function class works, which is why the original formulation is not a neural-network method at all (Ernst et al. used trees and random forests).

**Why iterate.** Each pass propagates value information backward by exactly one step. Starting from $Q_0 \equiv 0$, the first targets are just $y_i = r_i$, so $Q_1$ knows only immediate rewards. On a chain $s_1 \to s_2 \to s_3 \to \text{reward } 10$:

$$
Q_1(s_3,\cdot) \approx 10,\quad Q_2(s_2,\cdot) \approx \gamma\cdot 10,\quad Q_3(s_1,\cdot) \approx \gamma^2\cdot 10 .
$$

The reward walks backward one transition per iteration. This is why $K$ iterations are needed to see $K$ steps ahead, and why the effective horizon $1/(1-\gamma)$ sets how many are required.

**Offline by construction.** Nothing in the loop touches the environment. The same $\mathcal{D}$ is *relabelled* with fresh targets each round — $\mathcal{D}\to Q_1$, then the same $\mathcal{D}\to Q_2$, and so on. FQI is therefore the natural algorithmic skeleton for [[offline-reinforcement-learning]], and most modern offline methods are FQI with one of its three steps modified.

## Formal Description

### The template

$$
\begin{aligned}
&\textbf{Input: } \mathcal{D}=\{(s_i,a_i,r_i,s_i')\}_{i=1}^N,\ \text{function class } \mathcal{F},\ \text{discount } \gamma \\
&Q_0 \equiv 0 \\
&\textbf{for } k = 0,\dots,K-1: \\
&\qquad y_i \leftarrow r_i + \gamma\max_{a'} Q_k(s_i',a') \qquad \forall i \\
&\qquad Q_{k+1} \leftarrow \arg\min_{Q\in\mathcal{F}} \sum_{i=1}^N \big(Q(s_i,a_i) - y_i\big)^2 \\
&\textbf{return } \hat\pi(s) = \arg\max_a Q_K(s,a)
\end{aligned}
$$

Policy extraction is greedy and happens only at the end; the policy is never executed during training.

### As approximate dynamic programming

With the Bellman optimality operator $(\mathcal{T}Q)(s,a) = \mathbb{E}[r + \gamma\max_{a'}Q(s',a')\mid s,a]$, exact value iteration is $Q_{k+1} = \mathcal{T}Q_k$. FQI cannot represent $\mathcal{T}Q_k$ in general, so it does

$$
\boxed{\;Q_{k+1} \approx \Pi_{\mathcal{F}}\,\mathcal{T}Q_k\;}
$$

where $\Pi_\mathcal{F}$ is the least-squares projection onto $\mathcal{F}$ under the data distribution. The two operators pull in different directions: $\mathcal{T}$ is a $\gamma$-contraction in $\|\cdot\|_\infty$, but $\Pi_\mathcal{F}$ is a contraction only in the weighted $L_2$ norm, so the composition need not contract in either. That mismatch is the root of most divergence results for approximate value iteration, and it is why **Bellman completeness** ($\mathcal{T}\mathcal{F}\subseteq\mathcal{F}$, approximately) rather than mere [[realizability]] is the assumption that appears in the guarantees.

### Finite-horizon form

The vault's theory papers use the episodic variant: iterate *backward over the horizon* $h = H,\dots,1$ rather than over $k$, with a separate $Q_h$ per stage and $\hat V_{h+1}$ in the target. Same template, different indexing — one sweep suffices because the horizon is finite. See [[pessimistic-fitted-q-learning]], which is exactly this with an uncertainty penalty inserted at each stage.

### Worked example

$\gamma = 0.9$, $\mathcal{D} = \{(s_1,A,0,s_2),\ (s_2,A,0,s_3),\ (s_3,A,10,\text{terminal})\}$, $Q_0\equiv 0$:

| Iteration | $Q_k(s_1,A)$ | $Q_k(s_2,A)$ | $Q_k(s_3,A)$ |
|---|---|---|---|
| 1 | 0 | 0 | 10 |
| 2 | 0 | $0.9(10) = 9$ | 10 |
| 3 | $0.9(9) = 8.1$ | 9 | 10 |

Converging to $Q(s_1,A) = 0.9^2\cdot 10 = 8.1$ — dynamic programming carried out indirectly through regression.

### Why offline FQI is dangerous

The $\max_{a'}$ queries $Q_k$ at actions that may not appear in $\mathcal{D}$. Suppose the data covers action $A$ at state $s$ but almost never $B$, and the fitted model happens to extrapolate $Q(s,B) = 100$. The target becomes $y = r + \gamma\cdot 100$, the next regression internalizes it, and the error propagates backward on every subsequent sweep:

$$
\text{extrapolation error} \to \max \to \text{overestimation} \to \text{Bellman backup} \to \text{more error}
$$

Online, this is self-correcting: the agent tries $B$, observes the real reward, and the estimate falls. Offline there is no such correction — see [[overestimation-bias]] and [[pessimism-principle]]. This single failure mode is what most of the offline RL literature is organized around.

### Three sources of error

Worth separating, because different papers attack different ones:

1. **Statistical error** — $N < \infty$, so each regression has estimation error. Governed by the complexity of $\mathcal{F}$.
2. **Approximation error** — $\mathcal{T}Q_k \notin \mathcal{F}$, so $Q_{k+1} \ne \mathcal{T}Q_k$ even with infinite data. This is the $\epsilon_\mathcal{F}$ of Bellman completeness.
3. **Distribution / coverage error** — $d^\pi$ places mass where $d^\mu$ does not, forcing extrapolation. Where concentrability, coverage assumptions, and pessimism enter; see [[coverage-coefficient]].

All three compound across the $K$ sweeps, which is what error-propagation analyses (Munos 2005, 2007; Chen & Jiang 2019) quantify.

## Key Papers

- Gordon (1999), Ernst, Geurts & Wehenkel (2005) — the original fitted value/Q-iteration formulation; Ernst et al. use tree ensembles, underlining that FQI is function-class agnostic
- Riedmiller (2005) — neural FQI; the direct ancestor of DQN-style critics
- Munos (2005, 2007) — approximate value iteration and error propagation across iterations
- Chen & Jiang (2019) — information-theoretic analysis of batch RL under realizability + concentrability; the reference point that later instance-dependent work is measured against
- [[Yin2023Offline]] — [[pessimistic-fitted-q-learning]]: FQI plus a gradient-geometry penalty over [[differentiable-function-approximation]], yielding the first [[instance-dependent-bounds]] for offline RL under a nonlinear class
- Fan et al. (2020) — theoretical analysis of deep Q-learning, treating DQN as neural FQI

## Variants & Related Concepts

FQI is a template; most offline RL algorithms are FQI with **one of its three steps modified**. That is the most useful way to read the literature:

| Modified step | Mechanism | Example |
|---|---|---|
| Bellman target | replace $\max_{a'}$ with an in-sample upper expectile | [[implicit-q-learning]] |
| Bellman target | replace $\max_{a'}$ with a softmax at temperature $\tau$ | [[softmax-bellman-operator]] |
| Regression step | reweight residuals by conditional variance | VAFQL ([[Yin2023Offline]]) |
| Value estimate | subtract an uncertainty penalty | [[pessimistic-fitted-q-learning]], [[pessimism-principle]] |
| Value estimate | push down $Q$ on out-of-distribution actions | CQL (Kumar et al. 2020) |
| Policy extraction | advantage-weighted behavioral cloning | AWR step of [[implicit-q-learning]] |

- [[deep-q-network]] — DQN is an incremental, stochastic FQI: the target network $Q_{\theta^-}$ plays the role of the previous iterate $Q_k$, and SGD steps replace the full refit. Replay buffers and target networks are the practical machinery that makes the approximation work
- [[offline-reinforcement-learning]] — the data setting, not the algorithm. Keeping these separate matters: **FQI is the template, offline RL is the regime.** Offline FQI is plain FQI on fixed data; most of modern offline RL is offline FQI plus some mechanism preventing unsupported actions from looking artificially good
- [[realizability]] — necessary but not sufficient here; Bellman completeness is the operative assumption
- [[differentiable-function-approximation]] — the class over which FQI's fitted step becomes analyzable with gradient geometry

## Current State

The dominant algorithmic skeleton for value-based offline RL, and the one nearly every theoretical guarantee in the area is stated for. Its appeal over information-theoretic alternatives (the maxmin objectives that general-function-approximation analyses produce) is that the inner step is an ordinary least-squares fit, so it is actually runnable.

What remains open is the same list that makes offline RL hard. The composition $\Pi_\mathcal{F}\mathcal{T}$ has no contraction guarantee without Bellman completeness, an assumption that is strong and unverifiable in practice. Error propagation across sweeps is understood only up to worst-case concentrability in the general case; the instance-dependent treatment exists only for structured classes ([[Yin2023Offline]]). And the $\max$ in the target remains the central vulnerability — the reason the two strongest current answers, in-sample learning ([[implicit-q-learning]]) and pessimism ([[pessimistic-fitted-q-learning]]), are both best understood as edits to this template rather than departures from it.
