---
title: "Offline Reinforcement Learning with Implicit Q-Learning"
authors: [Ilya Kostrikov, Ashvin Nair, Sergey Levine]
year: 2022
venue: ICLR
arxiv: "2110.06169"
tags: [offline-reinforcement-learning, expectile-regression, implicit-q-learning, deep-reinforcement-learning, bellman-operator, overestimation-bias, distributional-shift]
source: raw/papers/Kostrikov2022Offline.pdf
---

# Offline Reinforcement Learning with Implicit Q-Learning

**TL;DR:** [[implicit-q-learning]] (IQL) does multi-step dynamic programming in offline RL *without ever evaluating the Q-function at an out-of-sample action*, by fitting an upper $\tau$-expectile of $Q(s,\cdot)$ over dataset actions ([[expectile-regression]]) in place of the max. As $\tau \to 1$ this recovers the support-constrained optimal value function (Thm 3); at $\tau = 0.5$ it degenerates to SARSA, so $\tau$ interpolates the whole SARSA-to-Q-learning spectrum.

## Problem

Offline RL must reconcile two conflicting aims: improve over the behavior policy $\pi_\beta$, while staying close enough to it that value estimates remain trustworthy. The standard TD target

$$
r(s,a) + \gamma \max_{a'} Q_{\hat\theta}(s', a')
$$

queries $Q_{\hat\theta}$ at actions $a'$ that may never appear in the dataset $\mathcal{D}$, where the function approximator extrapolates freely and — because of the max — the errors are selected *for* being large (see [[overestimation-bias]]).

Prior work patches this from two directions:

- **Policy constraints** (BCQ, BEAR, AWAC, TD3+BC) — restrict how far $\pi$ may deviate from $\pi_\beta$.
- **Value regularization** (CQL, Fisher-BRC) — push down $Q$ on out-of-distribution actions.

Both impose an improvement-vs.-safety trade-off. A third family, **single-step** methods (Onestep RL, Decision Transformer), sidesteps the issue entirely by fitting $Q^{\pi_\beta}$ or by pure behavioral cloning — but then cannot "stitch" good sub-trajectories together, because they never iterate the Bellman backup.

The paper's question: **can we get multi-step dynamic programming without ever querying an unseen action?**

## Method

The target the paper actually wants is the *support-constrained* optimum:

$$
L(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\Big[\big(r(s,a) + \gamma \!\!\max_{\substack{a' \in \mathcal{A} \\ \text{s.t. } \pi_\beta(a'|s')>0}}\!\! Q_{\hat\theta}(s',a') - Q_\theta(s,a)\big)^2\Big].
$$

IQL estimates this max using only dataset actions, via three pieces.

**1. Value function by upper-expectile regression.** With the asymmetric squared loss $L_2^\tau(u) = |\tau - \mathbb{1}(u<0)|\, u^2$,

$$
L_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[L_2^\tau\big(Q_{\hat\theta}(s,a) - V_\psi(s)\big)\big].
\tag{5}
$$

For $\tau > 0.5$ this downweights residuals where $Q < V$, so $V_\psi(s)$ tracks an *upper* expectile of $Q(s,\cdot)$ over the action distribution — an in-sample proxy for the max.

**2. Q-function by ordinary MSE.**

$$
L_Q(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\big[(r(s,a) + \gamma V_\psi(s') - Q_\theta(s,a))^2\big].
\tag{6}
$$

The separation of (5) and (6) is the subtle and important design choice. Taking the expectile of the *TD target* directly would take an upper expectile over both action randomness **and** transition randomness $s' \sim p(\cdot|s,a)$ — so a large target could reflect a "lucky" transition rather than a good action. Routing the expectile through $V_\psi$ (actions only) and the backup through MSE (which averages over dynamics) isolates the intended statistic. This is exactly the failure mode that makes naive optimism unsound in stochastic MDPs.

**3. Policy extraction by advantage-weighted regression.**

$$
L_\pi(\phi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\exp\big(\beta(Q_{\hat\theta}(s,a) - V_\psi(s))\big)\log\pi_\phi(a|s)\big],
\tag{7}
$$

with inverse temperature $\beta$. Small $\beta$ $\to$ behavioral cloning; large $\beta$ $\to$ greedy w.r.t. $Q$. Also in-sample.

Value learning and policy extraction are fully decoupled — the policy never influences the value functions, so extraction can run concurrently (enabling online finetuning) or after. Implementation uses clipped double Q-learning (min of two critics).

## Results

**D4RL, normalized scores (Table 1).** IQL is competitive on Gym locomotion and decisive where stitching matters:

| Suite | BC | Onestep RL | TD3+BC | CQL | **IQL** |
|---|---|---|---|---|---|
| locomotion-v2 total | 466.7 | 684.6 | 677.4 | **698.5** | 692.4 |
| antmaze-v0 total | 100.2 | 125.3 | 163.8 | 303.6 | **378.0** |
| total | 566.9 | 809.9 | 841.2 | 1002.1 | **1070.4** |
| total + kitchen + adroit | 825.9 | — | — | 1240.3 | **1348.3** |
| runtime (1M updates) | 10m | ~20m | 20m | 80m | 20m |

The gap is concentrated exactly where the paper predicts. On `antmaze-medium` and `antmaze-large`, every single-step method scores $\approx 0$ (Onestep RL: 0.3, 0.0, 0.0, 0.0), while IQL scores 71.2 / 70.0 / 39.6 / 47.5. These datasets contain essentially no near-optimal trajectories, so performance requires composing sub-optimal ones — the thing single-step methods structurally cannot do.

**Toy u-maze (Fig. 2).** With 1 optimal trajectory and 99 random ones, single-step policy evaluation produces a value function that decays to zero away from the goal; IQL's closely matches $V^\star$. A clean, honest illustration of the mechanism.

**Effect of $\tau$ (Fig. 3).** Larger $\tau$ is essential on antmaze; $\tau = 0.5$ (SARSA) fails, $\tau = 0.9$ works. On locomotion, smaller $\tau$ suffices — consistent with those datasets already containing near-optimal behavior.

**Online finetuning (Table 2).** After 1M online steps: antmaze total $370.1 \to 473.7$ for IQL, vs. $151.5 \to 231.1$ for CQL and $107.7 \to 108.3$ for AWAC. The AWR extraction step is the stated reason it finetunes well.

**Runtime.** ~20 min per 1M updates on a single GTX 1080, ~4$\times$ faster than their own JAX reimplementation of CQL.

## Assumptions & Theorems

The analysis is short and worth stating precisely, because the gap between what is proved and what is run is the most interesting thing about this paper.

### Notation

$\mathbb{E}^\tau_{x\sim X}[x]$ denotes the $\tau$-expectile of $X$ ($\tau = 0.5$ is the mean). Define the fixed point of (5)–(6):

$$
V_\tau(s) = \mathbb{E}^\tau_{a\sim\mu(\cdot|s)}[Q_\tau(s,a)], \qquad
Q_\tau(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'\sim p(\cdot|s,a)}[V_\tau(s')].
$$

### Lemma 1 — expectiles reach the supremum

For a real-valued random variable $X$ with bounded support and supremum $x^*$: $\lim_{\tau\to 1} m_\tau = x^*$. Proof is by monotonicity of $m_\tau$ in $\tau$ plus boundedness.

### Lemma 2 — monotonicity in $\tau$

$\tau_1 < \tau_2 \implies V_{\tau_1}(s) \le V_{\tau_2}(s)$ for all $s$. Proved by mirroring the classical policy improvement argument.

### Corollary 2.1 — never exceeds the constrained optimum

$V_\tau(s) \le \max_{a:\,\pi_\beta(a|s)>0} Q^*(s,a)$, where $Q^*$ is the optimal value function *constrained to dataset support*. Follows because a convex combination cannot exceed a maximum.

### Theorem 3 — the main guarantee

$$
\lim_{\tau\to 1} V_\tau(s) = \max_{\substack{a\in\mathcal{A} \\ \text{s.t. } \pi_\beta(a|s)>0}} Q^*(s,a).
$$

**Plain English.** Squeeze Lemma 2 (monotone increasing in $\tau$) against Corollary 2.1 (bounded above by the constrained optimum), then apply Lemma 1 to identify the limit. IQL converges to the best policy *expressible within the data support* — not to the true $Q^*$, which is the correct and honest target for offline RL.

### The honest gap

Three assumptions separate this from what is run:

1. **Asymptotic in $\tau$.** Theorem 3 is a statement about $\tau \to 1$. Experiments use $\tau \in \{0.7, 0.9\}$. At finite $\tau$ there is **no quantitative bound** on $\max_{\text{supp}} Q^* - V_\tau$ — only the qualitative monotonicity of Lemma 2. Contrast [[Song2019Revisiting]], which does supply an explicit exponential-in-$\tau$ gap bound for the analogous softmax operator.
2. **Exact solutions.** $V_\tau, Q_\tau$ are defined as exact optima of (5)–(6). No function approximation, no finite samples, no propagation of expectile-regression error through the backup.
3. **Support, not density.** "Constrained to $\pi_\beta(a|s) > 0$" ignores *how much* mass $\pi_\beta$ places near the maximizing action. An action with $\pi_\beta = 10^{-8}$ counts as fully supported in the theorem but is invisible to finite-sample expectile regression.

Point 3 is the substantive one and connects directly to [[coverage-coefficient]]: the object that should govern the finite-$\tau$, finite-sample gap is a density-weighted coverage quantity, not a support indicator.

## Strengths & Limitations

**Strengths.** The central idea is genuinely elegant — one loss-function change to a SARSA backup buys multi-step DP. The $V$/$Q$ split is a real insight, not a hack: it correctly separates action randomness (where you want optimism) from transition randomness (where you must not be optimistic). Computational cost is near the floor for this problem class. The $\tau$ = SARSA-to-Q-learning spectrum is a clean conceptual contribution independent of the algorithm. Evaluation is honest about the one-step/multi-step distinction and fixes a real benchmarking error in prior work (the `-v0` vs `-v2` D4RL environments), which *raises* the CQL baseline they compare against.

**Limitations.** The theory is thin relative to the empirical claims — asymptotic, exact-solution, tabular-flavored, with no finite-$\tau$ rate (see honest gap above). The headline "never queries out-of-sample actions" holds for *value training only*; AWR extraction still imposes an implicit KL-style constraint, so the constraint is relocated rather than removed — the paper states this in Section 4.3, but the abstract's framing oversells it. Two coupled hyperparameters ($\tau$, $\beta$) with meaningful sensitivity and no principled selection rule; $\tau$ that works on antmaze does not transfer to locomotion. Expectiles are chosen over quantiles largely for convenience ("worked somewhat better"), with no analysis of why. Finally, the support-constrained optimum can be far from $Q^*$ on narrow datasets, and nothing here quantifies that distance.

## Connections

- [[implicit-q-learning]] — the algorithm introduced here
- [[expectile-regression]] — the estimation primitive that makes in-sample maximization possible
- [[overestimation-bias]] — the failure mode being avoided; IQL removes it at the source by never applying max to an extrapolated value
- [[deep-q-network]] — IQL is a modification of the standard TD/actor-critic backup used in the DQN family
- **Thematic parallel:** [[Song2019Revisiting]] — softmax in place of max in the Bellman target. Same move as IQL's expectile, different smoothing family, and Song supplies the finite-$\tau$ bound IQL lacks. Note the *opposite* directions: Song softens the max downward to fight overestimation with full action access; IQL softens it to approximate a max it is forbidden from computing.
- **Thematic parallel:** [[Dam2024Power]] — power mean in place of max in MCTS backups. Third instance of the same design pattern.
- **Related:** [[coverage-coefficient]] — [[Foster2025Foundation]] formalizes coverage as the quantity governing what is learnable from a reference distribution; IQL's support constraint $\pi_\beta(a|s) > 0$ is the crude, binary version of the same idea
- [[Kostrikov-Ilya]], [[Nair-Ashvin]], [[Levine-Sergey]] — authors

## Open Questions

- **The main one: a finite-$\tau$ bound.** Can one bound $\max_{a:\pi_\beta(a|s)>0} Q^*(s,a) - V_\tau(s)$ for $\tau < 1$, as a function of the behavior-policy density near the maximizing action? A natural conjecture: the gap scales with an instance-dependent quantity resembling $C_\text{cov}$ (see [[coverage-coefficient]]) rather than with a support indicator, which would explain why $\tau = 0.9$ suffices on locomotion but not on antmaze.
- **Sample complexity.** What is the finite-sample error of expectile regression at $\tau \to 1$, and how does it propagate through the Bellman backup? Upper expectiles are estimated from progressively fewer effective samples, so there should be a bias-variance optimum in $\tau$ — the paper treats $\tau$ purely as a bias knob.
- **Expectile vs. quantile vs. softmax.** Is there a principled reason to prefer expectiles here beyond MSE convenience? A unified analysis of smooth in-support maximization operators across [[Song2019Revisiting]], [[Dam2024Power]], and IQL seems within reach.
- **$\tau$ scheduling.** Would annealing $\tau \to 1$ during training dominate a fixed $\tau$? (Song found annealing did not help for softmax — does that transfer?)
- Does the $V$/$Q$ split remain necessary under low transition stochasticity, or can it be dropped in near-deterministic environments?
