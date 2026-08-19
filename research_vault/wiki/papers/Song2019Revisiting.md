---
title: "Revisiting the Softmax Bellman Operator: New Benefits and New Perspective"
authors: [Zhao Song, Ronald E. Parr, Lawrence Carin]
year: 2019
venue: ICML
arxiv: "1812.00456"
tags: [deep-reinforcement-learning, bellman-operator, softmax-bellman-operator, overestimation-bias, deep-q-network, q-learning]
source: raw/papers/Song2019Revisiting.pdf
---

# Revisiting the Softmax Bellman Operator: New Benefits and New Perspective

**TL;DR:** Proves that the [[softmax-bellman-operator]] (i) converges to the standard Bellman operator exponentially fast in inverse temperature $\tau$, (ii) provably reduces [[overestimation-bias]] in DQNs with quantified bounds, and (iii) reduces gradient noise — and shows empirically that replacing max with softmax in DQN/DDQN target networks improves Atari performance independent of exploration.

## Problem

The softmax Bellman operator has long been viewed as problematic: Littman (1996) showed it is not a contraction for certain $\tau$, so iterating it may not converge to $Q^*$. Its primary use has been as an exploration device (Boltzmann exploration). Yet practitioners observed that simply swapping max for softmax in target networks improved deep Q-learning. Why?

The paper's question: **can the softmax Bellman operator be beneficial beyond exploration, and if so, why?** Prior work (Asadi & Littman 2017, mellowmax) proposed a contraction alternative but it cannot directly represent a policy, blocking compatibility with double Q-learning.

## Method

**The softmax Bellman operator** is defined as:

$$
\mathcal{T}_{\text{soft}} Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \underbrace{\frac{\exp[\tau Q(s',a')]}{\sum_{\bar{a}} \exp[\tau Q(s',\bar{a})]}}_{\text{sm}_\tau(Q(s',\cdot))_{a'}} Q(s',a'),
$$

where $\tau \ge 0$ is the inverse temperature. $\mathcal{T}_\text{soft} \to \mathcal{T}$ as $\tau \to \infty$.

**S-DQN / S-DDQN**: replace the max in the target network of DQN / DDQN with the softmax operator above. Exploration is unchanged ($\varepsilon$-greedy), and $\tau$ is selected by grid search over $\{1,5,10\}$.

**Theoretical results:**

- **Theorem 3 (convergence):** $\limsup_{k\to\infty}\mathcal{T}_{\text{soft}}^k Q_0(s,a) \le Q^*(s,a)$ and $\liminf_{k\to\infty}\mathcal{T}_{\text{soft}}^k Q_0 \ge Q^*(s,a) - \frac{\gamma(m-1)}{1-\gamma}\max\!\left\{\frac{1}{\tau+2}, \frac{2Q_{\max}}{1+\exp(\tau)}\right\}$. Crucially, the upper bound is $Q^*$; deviations below are bounded and shrink *exponentially* in $\tau$. Non-convergence means staying near $Q^*$, not diverging.

- **Theorem 4 (overestimation reduction):** Under the same assumptions as van Hasselt et al. (2016a) — estimation error $Q_t(s,a) = V^*(s) + \epsilon_a$ — the softmax overestimation errors are $\le$ those of $\mathcal{T}$ for all $\tau \ge 0$; the reduction is within $\big[\frac{\hat{\delta}(s)}{m\exp(\tau\hat{\delta}(s))},\ (m-1)\max\{\frac{1}{\tau+2},\frac{2Q_{\max}}{1+\exp(\tau)}\}\big]$; and the overestimation error increases monotonically in $\tau\in[0,\infty)$, recovering the max at $\tau=\infty$.

## Results

- **Atari (6 games, 5 seeds):** S-DQN and S-DDQN consistently achieve higher test scores than DQN and DDQN. S-DDQN vs. DDQN: best $\tau$ wins on 5/6 games, with large margins (e.g. Asterix: 10266 vs. 5524 at $\tau=5$). S-DQN also generally beats DDQN — suggesting softmax operator alone can substitute for double Q-learning's bias reduction.
- **Overestimation:** S-DQN and S-DDQN achieve lower overestimation bias than their max counterparts, validating Theorem 4. Smaller $\tau$ reduces overestimation but increases risk of underestimation.
- **Gradient noise:** S-DQN/S-DDQN exhibit lower gradient $\ell_2$ norm and variance in the final FC layer, consistent with the stabilizing effect of reducing overestimation on the Bellman target.
- **vs. mellowmax:** Softmax matches or exceeds mellowmax on 5/6 games (mellowmax is cheaper to compute but requires numerical methods to represent policy and is incompatible with double Q-learning).

**Bellman operator comparison (Table 3):**

| | Bellman optimality | Tuning | Overestimation reduction | Policy representation | Double Q-learning |
|---|---|---|---|---|---|
| Max | Yes | No | — | Yes | Yes |
| Mellowmax | No | Yes | Yes | No | No |
| **Softmax** | **No** | **Yes** | **Yes** | **Yes** | **Yes** |

## Assumptions & Theorems

### Vocabulary

- **Q-value** $Q(s,a)$: expected total future reward from state $s$, taking action $a$, then playing optimally.
- **$Q^*$**: the true, perfect Q-values under the optimal policy.
- **Bellman backup**: the update rule — Q-value now = reward now + discounted best Q-value next step. Iterating this is how Q-learning learns.
- **Max operator** $\mathcal{T}$: standard backup — greedy, always picks the single highest Q-value at the next state.
- **Softmax operator** $\mathcal{T}_\text{soft}$: modified backup — weighted average over all actions, with weights proportional to $\exp(\tau \cdot Q)$. High $\tau \to$ almost max; low $\tau \to$ uniform average.
- $m$: number of actions. $\gamma$: discount factor.

---

### Lemma 2 — prerequisite for everything

**Statement:** Assume the Q-gap $\hat\delta(s) > 0$ — meaning not all actions are equally valued in state $s$.

**Plain English:** "The problem is non-trivial — there's at least some difference between the best and worst actions." Rules out the degenerate case where all Q-values are identical (softmax = max = mean there, so nothing interesting happens). Every reasonable RL problem satisfies this.

---

### Theorem 3 — convergence ("it doesn't blow up")

**Statement:**
- **(I)** Iterating $\mathcal{T}_\text{soft}$ from any starting $Q_0$ never produces values exceeding $Q^*$: $\limsup_k \mathcal{T}_\text{soft}^k Q_0 \le Q^*$. The values may not converge exactly to $Q^*$ — they can stabilize anywhere in a range $[Q^* - \text{gap},\ Q^*]$.
- **(II)** That gap shrinks *exponentially fast* in $\tau$: $\text{gap} \le \frac{\gamma(m-1)}{1-\gamma}\max\!\left\{\frac{1}{\tau+2},\frac{2Q_{\max}}{1+\exp(\tau)}\right\}$.

**Plain English:** The standard max operator is guaranteed to converge to $Q^*$ (it's a contraction). The softmax operator has no such guarantee — Littman (1996) showed it can fail to converge for certain $\tau$. The worry: maybe softmax diverges or oscillates wildly, making it useless.

Theorem 3 kills that worry. **Softmax can't overshoot $Q^*$, and even if it doesn't converge exactly, it stays trapped in a small band just below $Q^*$.** How small? The band shrinks exponentially as $\tau$ grows. At $\tau = 10$ the band is already tiny; at $\tau = \infty$ it collapses to zero and you recover the standard operator.

The key reframe: "doesn't converge" $\neq$ "goes off to infinity." Q-values might oscillate slightly near $Q^*$, which is fine — DQN never iterates to convergence anyway, it runs stochastic gradient steps. Being near $Q^*$ is good enough.

**Assumption:** Tabular setting with exact Bellman updates and known transition probabilities. This is the main gap with practice — real DQN has function approximation and sampled transitions.

---

### Theorem 4 — overestimation reduction ("softmax is less biased")

The paper's most important practical result.

**Assumptions (same as van Hasselt et al. 2016a / DDQN):**
- **(A1)** At the optimal policy, every action is equally good in each state: $Q^*(s,a) = V^*(s)$ for all $a$. The agent is indifferent among actions.
- **(A2)** Estimation noise is i.i.d. per action: $Q_t(s,a) = V^*(s) + \epsilon_a$.

**Plain English of the setup:** Imagine all true Q-values for a state are 5, but the network estimates each with random noise — one comes out 5.3, another 4.8, another 5.6. The max operator picks 5.6 as the "best," which is higher than the true value. That's overestimation. These assumptions are stylized (real Q-values aren't all equal) but cleanly isolate the problem.

**What the theorem says — three parts:**

**(I) Softmax always overestimates less than max, for any $\tau \ge 0$:**

Softmax overestimation $\le$ max overestimation, always. **You never make things worse by switching from max to softmax** — for any temperature setting.

**(II) The reduction is quantified:**

The overestimation reduction is at least $\frac{\hat\delta(s)}{m\cdot\exp(\tau\hat\delta(s))}$ and at most $(m-1)\max\!\left\{\frac{1}{\tau+2},\frac{2Q_{\max}}{1+\exp(\tau)}\right\}$. You get a computable range — not just "epsilon better."

**(III) Overestimation increases monotonically with $\tau$:**

Lower $\tau$ = less [[overestimation-bias]], higher $\tau$ = more (approaching max at $\tau = \infty$). This gives a dial: turn $\tau$ down to reduce bias. But turning it too far pushes toward the mean, risking underestimation and losing the Bellman-optimality anchor. Experiments found $\tau \in \{5, 10\}$ works well.

---

### Gradient noise (Section 5.2 — informal argument, empirically validated)

**The claim:** Softmax also reduces gradient noise during training, as a downstream effect of reducing overestimation.

**Plain English:** The DQN gradient is proportional to the Bellman error — the gap between the network's current prediction and the target. Overestimated targets are inflated and noisy, producing large chaotic gradients. Reducing overestimation shrinks and stabilizes the targets, which stabilizes the gradients. Experiments confirm: S-DQN/S-DDQN show lower gradient $\ell_2$ norm and variance than their max counterparts.

This is not a theorem — it's a consequence of Theorem 4, argued informally then empirically validated.

---

### How the pieces fit together

```
Lemma 2: non-trivial Q-gap assumed
    ↓
Theorem 3: softmax Q-values stay near Q*
           (non-convergence ≠ divergence — safe to use)

A1 + A2: all-equal Q* + i.i.d. noise assumed
    ↓
Theorem 4: softmax overestimates less than max, always
    ↓
Gradient noise argument: less overestimation
    → smaller, more stable targets
    → better gradient signal → better training
```

**The honest gap:** Both theorems assume things that don't hold exactly in practice — tabular/known transitions (Thm 3), all-equal optimal Q-values (Thm 4). The theory motivates but doesn't fully explain the empirical gains. The qualitative mechanism (softmax reduces max-induced noise) appears to persist even with function approximation, but proving this formally is the main open problem.

## Strengths & Limitations

**Strengths:** Clean separation of exploration and operator effects (exploration fixed, $\varepsilon$-greedy throughout). The overestimation bounds are tight enough to be informative. Practical simplicity — one hyperparameter $\tau$, drop-in replacement. Provides theoretical backing for why a non-contraction operator can still improve deep RL in the presence of function approximation.

**Limitations:** Theory is for tabular / known next-state distributions; the performance guarantees are sub-optimality bounds, not superiority over max. Only tested on 6 Atari games with a relatively old DQN baseline. $\tau$ sensitivity is real (too small → underestimation, too large → approaches max and overestimation returns). No convergence guarantee — the bounds only say Q-values stay in a range near $Q^*$.

## Connections

- [[softmax-bellman-operator]] — the central concept introduced/analyzed here
- [[overestimation-bias]] — the key mechanism softmax corrects in deep Q-learning
- [[deep-q-network]] — the base algorithm (DQN/DDQN) on which S-DQN/S-DDQN are built
- [[monte-carlo-tree-search]] — loose connection via [[Dam2024Power]]: both use softmax/power-mean backups to improve on the max operator in tree/value-based planning
- **Thematic parallel:** [[Kostrikov2022Offline]] — IQL replaces the max with an upper $\tau$-expectile over dataset actions. Closest relative of this paper in the vault, with an instructive asymmetry: here the max is available and softened to reduce [[overestimation-bias]]; there the max is unavailable (out-of-sample actions cannot be queried) and the smooth statistic is the only way to approximate it. This paper supplies the finite-$\tau$ gap bound that IQL's asymptotic Theorem 3 lacks
- [[Song-Zhao]], [[Parr-Ronald-E]], [[Carin-Lawrence]] — authors

## Open Questions

- Can the performance guarantee be strengthened beyond a sub-optimality range — i.e., can we prove softmax *beats* max under function approximation?
- What is the optimal $\tau$ as a function of problem structure (number of actions, noise level)? The cooling scheme (simulated annealing) showed no improvement here.
- Does the benefit extend to modern deep RL algorithms (Rainbow, PPO, SAC) or is it specific to DQN-family architectures?
- How does $\mathcal{T}_\text{soft}$ relate to entropy-regularized RL (soft Q-learning, SAC)? The paper notes softmax policies appear as optima in entropy-regularized MDPs — are these the same phenomenon?
