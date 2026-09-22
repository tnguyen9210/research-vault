---
title: "Quantile of Means: A Bonus-Free Ensemble Method for Minimax Optimal Reinforcement Learning"
authors: [Asaf Cassel, Aviv Rosenberg]
year: 2026
venue: arXiv
arxiv: "2606.20107"
tags: [online-reinforcement-learning, exploration, optimism, ensembles, regret-bounds, instance-dependent-bounds]
citekey: cassel2026Quantile
---

# Quantile of Means

**TL;DR:** Optimism in tabular RL has always been bought with count-based bonuses. This replaces them with a fixed quantile of an ensemble of Q-estimates trained on disjoint data batches, and gets the same variance-dependent minimax rate with no bonuses, no visitation counts, no posterior, and no assumption on the reward distribution — the first provably efficient ensemble exploration in MDPs.

## Problem

Two literatures had not met. Provably efficient exploration in MDPs uses count-based uncertainty: estimate a mean, add a bonus proportional to $1/\sqrt{N_h(s,a)}$, plan against the inflated values. This is tight but needs explicit visitation counts or transition models, which do not exist in high-dimensional or continuous state spaces — so the theory gives little guidance for practice. Meanwhile deep RL explores with ensembles (Bootstrapped DQN, UCB Q-ensembles, SUNRISE) with empirical success and no theory: it was not understood how or why ensembling induces the optimism efficient exploration requires.

Cassel et al. (2025) closed this for multi-armed bandits with a batch ensemble scheme. Extending it to MDPs is not a per-state reduction: Q-estimates at step $h$ depend on later value estimates through Bellman backups, which destroys the independence the bandit analysis rests on. Viel et al. (2025) adapted the bandit scheme to MDPs for imitation learning but got suboptimal rates and needed to binarize the state space, a construction quadratic in $|\mathcal S|$ that cannot be applied heuristically under function approximation.

## Method

**The Quantile of Means (QoM) estimator.** Given iid non-negative $X_1,\dots,X_n$ with mean $\mu$, partition the sample into $B$ fixed disjoint batches $D_1,\dots,D_B$ and take the $\alpha$-quantile of the batch means:
$$
\hat\mu_\alpha \;=\; q_\alpha\!\Big( \tfrac{1}{|D_b|+1}\textstyle\sum_{X \in D_b} X,\ b \in [B] \Big),
\qquad q_\alpha(\cdot) := \hat\mu_{(\lceil \alpha B\rceil)},
$$
where $\hat\mu_{(1)} \le \dots \le \hat\mu_{(B)}$ is the sorted sequence. It generalizes Median of Means; Cassel et al. (2025)'s Minimum of Means is the case $\alpha = 1/B$.

**Why it is optimistic** is the paper's mechanism, and it is not a concentration argument. The $+1$ in the denominator is load-bearing: by Feige (2004, Thm 1), $\Pr\big[\sum_i X_i/(n+c) < \mu\big] \ge 1/13$ for any $c \ge 1/12$ — a statement needing *only the first moment*, no tail assumption (Corollary 2). Each batch is therefore optimistic with constant probability; the batches are independent, so the number of optimistic batches is Binomial$(B, \ge 1/13)$, and a Chernoff bound on its lower tail makes the $\alpha$-quantile optimistic with high probability. **Lemma 1**, with $B \ge 26\log\delta^{-1}$ and $\alpha = 1/65$, gives each of the following individually w.p. $\ge 1-\delta$: (i) *optimism*, $\hat\mu_\alpha \le \mu$; (ii) *bias*, $\hat\mu_\alpha \ge \mu - 1.7\sqrt{\sigma^2 B/\max\{1,n\}} - 9RB/\max\{1,n\}$.

The bias half uses Freedman's inequality, and this is where the "distribution-agnostic algorithm, distribution-adaptive analysis" claim comes from: swapping in Chebyshev handles unbounded variables with finite variance, and a KL bound sharpens the Bernoulli case — **all without changing the estimator**. Bonus-based methods must change the implementation to change the distributional assumption.

**Algorithm 1 (VIBE — Value Iteration with Bootstrap Ensemble).** Maintain $B$ disjoint datasets $D^{t,b}_h(s,a)$ per state-action-step, assigned by strict round-robin (each new transition goes to the batch with fewest samples, keeping batch sizes balanced). Each episode, run standard value iteration *with no bonus*, but evaluate each Q-value as the QoM across the $B$ batches:
$$
\hat Q^{t,b}_h(s,a) = \big[\hat r^{t,b}_h + \hat P^{t,b}_h \hat V^t_{h+1}\big](s,a),
\qquad
\hat V^t_h(s) = \min_a q_\alpha\big(\hat Q^{t,b}_h(s,a),\ b\in[B]\big),
$$
with the same $+1$ in both empirical estimates. (The paper minimizes loss, so the $\alpha$-quantile is the optimistic — low — end.) Two hyperparameters: $\alpha$ constant, $B$ logarithmic in $H, S, K, \delta^{-1}$. Cost is $O(SAHB)$ per episode — value iteration run $B$ times, negligible since $B$ is logarithmic.

## Results

**Theorem 4.** With $\alpha = 1/65$, $B = 26\log(5SAHK\delta^{-1})$ and $\kappa = \log(20HS^2AK\delta^{-1})$, w.p. $\ge 1-\delta$,
$$
\mathrm{regret}_K \;\le\; 22\sqrt{\min\{\mathcal Q^\star,\, HV^\star\}\,HSAK\,\kappa^2} \;+\; 1924\,H^3S^2A\kappa^3,
$$
where $\mathcal Q^\star = \sum_{h\in[H]} \max_{s,a} \mathrm{Var}_{s,a,h}\big(L_h + V^\star_{h+1}(s_{h+1})\big)$ is the **summed** maximum conditional variance. Optimal in two regimes: it matches the $\Omega(\sqrt{H^3SAK})$ minimax lower bound (Domingues et al. 2021) in the worst case, and the refined variance-dependent lower bound of Zhou et al. (2023) in low-stochasticity environments.

**Secondary result.** The quantile approach also improves the bandit case, shaving a logarithmic factor off Cassel et al. (2025) and reaching the optimal instance-dependent rate (Section C).

**The technical obstacle and its resolution.** A naive covering argument over value functions handles the Bellman dependence but yields ensemble size **linear in $S$**. Logarithmic $B$ requires replacing the empirical value function with the *optimal* one when establishing optimism (Lemma 5). In prior variance-dependent analyses that substitution is always entangled with the bonus construction; with no bonuses it stands alone, which is the paper's argument that ensembles may be the more natural primitive for exploration in MDPs.

## Strengths & Limitations

**Strengths.** The mechanism is genuinely simple and the whole proof is in the main paper. The first-moment optimism fact is reusable and surprising. The distribution-agnostic/adaptive split is a real advantage over bonuses, not a restatement of one. Computational overhead is logarithmic.

**Limitations.** Tabular only — the setting where counts *are* available, so the practical argument is made in the one regime where it is not yet needed. No experiments. The deep-RL ensemble methods it justifies in spirit (Bootstrapped DQN, SUNRISE) are not the algorithm analyzed, so the theoretical grounding is for the family, not for any deployed method. Time-inhomogeneous transitions, so the rate is not directly comparable to [[zanette2019Tighter]]'s stationary setting.

## Connections

- [[online-reinforcement-learning]] — the setting page; its optimism decomposition and variance-dependent rate come from here
- [[zanette2019Tighter]] — the count-based variance-dependent result this is measured against. **Their $\mathcal Q^*$ differ**: Zanette maxes over $(s,a,t)$, this sums the per-step maxima over $h$, so the quantities are not interchangeable and the settings (stationary vs. time-inhomogeneous) differ too
- [[upper-confidence-bound]] — the bonus paradigm this replaces
- [[pessimism-principle]] — the offline mirror; there the estimate is pushed *below* the truth for the opposite reason
- [[instance-dependent-bounds]] — variance-dependent regret is this guarantee type
- [[smooth-aggregators]] — a quantile of an ensemble is an order statistic over *estimates*, structurally adjacent to the three max-replacement instances there
- [[fqi]] — VIBE is value iteration with the regression step replaced by an ensemble; the template is unchanged
- Cassel et al. (2025), Viel et al. (2025), Feige (2004), Tiapkin et al. (2022a,b), Zhou et al. (2023), Domingues et al. (2021) — cited author–year, no pages here

## Open Questions

- **Function approximation.** Whether the quantile mechanism extends to linear or general function classes is the paper's own closing question, and the one that decides whether this is a theoretical curiosity or the justification for what deep RL already does.
- **Ensemble size under approximation.** The $V^\star$-substitution that buys logarithmic $B$ is a tabular argument; whether it survives is the technical form of the above.
- **Empirical behaviour.** No experiments. Whether VIBE is competitive with tuned bonus-based methods, or with the deep ensembles it is meant to justify, is untested.
- **Which deployed ensembles this actually covers.** Bootstrapped DQN resamples with replacement; VIBE partitions disjointly by round-robin. The gap between the two is not analyzed.
