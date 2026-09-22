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

**TL;DR:** Optimism in tabular RL has always been bought with count-based bonuses. This replaces them with a fixed quantile of an ensemble of $Q$-estimates trained on disjoint data batches, and recovers the variance-dependent minimax rate with no bonuses, no visitation counts, no posterior, and no assumption on the shape of the loss distribution — the first provably efficient ensemble exploration in MDPs.

## Intuition

To explore efficiently you need to know which of your value estimates might be wrong in your favour. The standard answer is to *measure* that: count how often you have visited each pair, convert the count into a confidence width, and subtract it so the value you plan against is optimistic. The width is the algorithm's model of its own ignorance, and building it is where the difficulty — and the count-dependence — lives.

The alternative here is to stop measuring uncertainty and start *exhibiting* it. Split the data for each state-action-step into $B$ disjoint batches and fit an estimate on each. The batches disagree exactly to the extent the data is uninformative: with plenty of data they concentrate together, with little data they scatter. So the spread of the ensemble already *is* the uncertainty, and you never have to quantify it — you just reach into the ensemble and take a low-order statistic. Acting on the $\alpha$-quantile is then optimistic for free, and it tightens automatically as the batches converge.

What makes this work rather than merely sound plausible is a single fact about sample means, and it is not a concentration inequality. A sample mean with a $+1$ added to its denominator falls strictly below the true mean with probability at least $1/13$, **whatever the distribution**, using nothing but the existence of the first moment (Feige 2004). Each batch is therefore a coin that lands "optimistic" a constant fraction of the time; the batches are independent; so the number of optimistic batches is binomial and a Chernoff bound on its lower tail says a fixed low quantile is optimistic with high probability. Optimism becomes a counting argument over independent estimates rather than a statement about tails.

## Formal Problem Definition

### Setting and learning protocol

A finite-horizon MDP $M = (\mathcal S, \mathcal A, H, P, L)$ with $|\mathcal S| = S$, $|\mathcal A| = A$, horizon $H$, **time-inhomogeneous** transitions $P_h(\cdot \mid s,a)$ for $h \in [H]$, and a loss distribution $L_h(s,a)$ with mean $\ell_h(s,a)$. Each of $K$ episodes starts from a fixed $s_1$ (stated as WLOG, purely notational) and runs for $H$ steps: at step $h$ in state $s^k_h$ the agent picks $a^k_h$, transitions to $s^k_{h+1} \sim P_h(\cdot \mid s^k_h, a^k_h)$, and observes a loss $L^k_h \sim L_h(s^k_h, a^k_h)$. Only the visited pair's loss and successor are observed. The agent commits to a deterministic Markov policy $\pi^k = (\pi^k_h)_{h\in[H]}$ before each episode.

### Learning objective

For a policy $\pi$, $V^\pi_h(s)$ is the expected **loss-to-go** from $s$ at step $h$; the optimal policy and value minimize it, $\pi^\star \in \arg\min_{\pi \in \Pi_M} V^\pi_1(s_1)$ and $V^\star = V^{\pi^\star}_1(s_1)$, with deterministic Markov policies known to be optimal among all history-dependent ones. The objective is cumulative regret over $K$ episodes,
$$
\mathrm{regret}_K \;=\; \sum_{k \in [K]} \Big( V^{\pi^k}_1(s_1) - V^\star_1(s_1) \Big).
$$

*Translation to vault notation:* this vault reserves $K$ for $|\mathcal A|$ and writes $T$ for the number of episodes, so quoted rates read $A \to K$ and $K \to T$; $\mathrm{regret}_K$ is [[online-reinforcement-learning]]'s $\mathrm{Reg}(T) = \sum_t \Delta(\pi_t)$.

### How this differs from the surrounding literature

Four departures, each of which the paper turns into part of its claim:

- **Losses, not rewards.** The problem is stated as minimization. This inverts the direction of optimism — the optimistic estimate is the *low* one, so the algorithm takes a **low** quantile ($\alpha$ small) where a reward-maximizing version would take a high one.
- **No counts anywhere.** Prior optimal methods — Zanette & Brunskill (2019), Zhou et al. (2023), and see [[zanette2019Tighter]] — require explicit visitation counts or transition models to build bonuses. The quantile-based posterior methods do not escape this: Bayes-UCBVI (Tiapkin et al. 2022b) and Tiapkin et al. (2022a) use quantiles of a **Dirichlet posterior over transitions**, which is an inherently count-based operation, and additionally assume the reward function is fully known.
- **No distributional knowledge.** Other bandit perturbation and ensemble schemes (Lu & Van Roy 2017; Kveton et al. 2019; Lee & Oh 2024) and bootstrap bounds (Hao et al. 2019) either fail to adapt to unknown reward distributions, inject artificial noise, or require symmetric rewards. Bayes-UCB (Kaufmann et al. 2012) uses posterior quantiles but still computes count-dependent confidence levels.
- **A direct extension to MDPs, not a reduction.** Cassel et al. (2025) gave the batch-ensemble scheme for bandits; Viel et al. (2025) carried it to MDPs for imitation learning but reached suboptimal rates and needed to **binarize the state space**, a construction scaling quadratically in $S$ that cannot be applied heuristically under function approximation. The open question this answers is whether a direct ensemble, with no such structural surgery, attains optimal regret.

## Assumptions

Strikingly few, and the paper's third contribution is essentially the list of assumptions it does *not* make.

- **Assumption 3 (bounded total loss).** The losses $L_h$ are non-negative and satisfy $\sum_{h\in[H]} L_h \le H$ almost surely, for every policy. This is the only assumption on the loss distribution.
- **Tabular and finite.** $S$, $A$, $H$ all finite; rates depend polynomially on them.
- **Hyperparameters are set, not tuned.** $\alpha = 1/65$ and $B = 26\log(5SAHK\delta^{-1})$ are fixed by the analysis, so $\alpha$ is a constant and $B$ logarithmic in all problem parameters.
- **Not assumed:** any knowledge of the loss distributions' shape or variance; any knowledge of $\mathcal Q^\star$ or $V^\star$; bounded support beyond Assumption 3; symmetry; a prior; any visitation counts.
- **Relaxation (Section B).** Assumption 3 can be weakened to $L_h \ge 0$, $\mathbb E[\sum_h L_h] \le H$ for all policies, and each $L_h(s,a)$ of finite but **unknown** variance — covering heavy tails — because Lemma 1 needs only non-negativity and a finite variance, not boundedness. The same regret bound survives up to constants, with one loss: the $\min\{\mathcal Q^\star, HV^\star\}$ becomes $\mathcal Q^\star$ alone, since the $HV^\star$ branch comes from bounding the variance of the optimal value function and that step needs almost-sure boundedness.

## Method

### The estimator

Given iid non-negative $X_1,\dots,X_n$ with mean $\mu$, partition into $B$ fixed disjoint batches $D_1,\dots,D_B$ and take the $\alpha$-quantile of the batch means:
$$
\hat\mu_\alpha \;=\; q_\alpha\!\Big( \tfrac{1}{|D_b|+1}\textstyle\sum_{X \in D_b} X,\ b \in [B] \Big),
\qquad q_\alpha(\hat\mu_b, b\in[B]) := \hat\mu_{(\lceil \alpha B\rceil)},
$$
with $\hat\mu_{(1)} \le \dots \le \hat\mu_{(B)}$ the sorted batch means. **Quantile of Means** generalizes Median of Means; Cassel et al. (2025)'s Minimum of Means is the special case $\alpha = 1/B$.

### Algorithm 1 — VIBE (Value Iteration with Bootstrap Ensemble)

Maintain $B$ disjoint datasets $D^{k,b}_h(s,a)$ per state-action-step, filled by **strict round-robin**: each new transition goes to whichever batch currently has fewest samples, so batch sizes stay balanced and the ensemble's variance stays uniform across members. Each episode, run ordinary backward value iteration with **no bonus term**, but evaluate each $Q$-value as the QoM across batches:
$$
\hat P^{k,b}_h(s'\mid s,a) = \!\!\sum_{s^+ \in D^{k,b}_h(s,a)}\!\! \frac{\mathbb 1\{s^+ = s'\}}{|D^{k,b}_h(s,a)|+1},
\qquad
\hat\ell^{k,b}_h(s,a) = \!\!\sum_{L \in D^{k,b}_h(s,a)}\!\! \frac{L}{|D^{k,b}_h(s,a)|+1},
$$
$$
\hat Q^{k,b}_h(s,a) = \big[\hat\ell^{k,b}_h + \hat P^{k,b}_h \hat V^k_{h+1}\big](s,a),
\qquad
\hat V^k_h(s) = \min_{a} \; q_\alpha\big(\hat Q^{k,b}_h(s,a),\ b\in[B]\big),
$$
and play greedily with respect to the same quantile. The $+1$ in both denominators is the mechanism, not a regularizer. Cost is $O(SAHB)$ per episode — value iteration run $B$ times — which is negligible overhead since $B$ is logarithmic.

### The contribution, in the order the paper claims it

1. **Instance-optimal regret for MDPs.** Variance-dependent bounds for tabular finite-horizon MDPs matching the best known results, previously reachable only through count-based bonus constructions. First provably efficient ensemble-based exploration in MDPs, and optimal at that.
2. **Better bandit bounds as a by-product.** The quantile view shaves a logarithmic factor off Cassel et al. (2025) and attains the optimal instance-dependent rate (Section C).
3. **Distribution-agnostic algorithm, distribution-adaptive analysis.** The algorithm encodes no distributional assumption and runs unchanged on bounded, sub-Gaussian and non-negative heavy-tailed losses; the *analysis* may still exploit the true concentration properties — a KL bound for Bernoulli, Chebyshev for unbounded finite variance — yielding tighter guarantees with **no algorithmic modification**. Bonus-based methods must change the implementation to change the assumption.
4. **A transparent analysis.** The complete proof is in the main paper. Without bonus terms, the key step — substituting the optimal value function for the empirical one when establishing optimism — stands alone instead of being entangled with a bonus construction.

### Main theorems

**Lemma 1 (the estimator).** For iid non-negative $X_i$ bounded a.s. by $R$, with mean $\mu$ and variance $\sigma^2$, run with $B \ge 26\log\delta^{-1}$ batches and $\alpha = 1/65$. Each of the following holds individually w.p. $\ge 1-\delta$:
- *optimism:* $\hat\mu_\alpha \le \mu$;
- *bias:* if $|D_b| \ge \lfloor n/B \rfloor$ for all $b$, then $\hat\mu_\alpha \ge \mu - 1.7\sqrt{\sigma^2 B / \max\{1,n\}} - 9RB/\max\{1,n\}$.

**Corollary 2 (why optimism holds).** For iid non-negative $X_i$ with mean $\mu$ and any $c \ge 1/12$, $\Pr\big[\sum_{i=1}^n X_i/(n+c) < \mu\big] \ge 1/13$. Follows from Feige (2004, Thm 1). This is the first-moment fact; the optimism half of Lemma 1 is a Chernoff bound on the binomial count of optimistic batches, and the bias half is the same argument with Freedman's inequality in place of Corollary 2.

**Lemma 5 (good event for optimism).** With $\alpha = 1/65$ and $B \ge 26\log(5SHK\delta^{-1})$, w.p. $\ge 1 - \delta/5$, simultaneously for all $h, s, k$: $q_\alpha\big([\hat\ell^{k,b}_h + \hat P^{k,b}_h V^\star_{h+1}](s,\pi^\star_h(s)), b\in[B]\big) \le [\ell_h + P_h V^\star_{h+1}](s, \pi^\star_h(s))$. Note $V^\star$, not $\hat V$ — this substitution is what keeps $B$ logarithmic rather than linear in $S$.

**Theorem 4 (main result).** With $\alpha = 1/65$, $B = 26\log(5SAHK\delta^{-1})$ and $\kappa = \log(20HS^2AK\delta^{-1})$, w.p. $\ge 1-\delta$,
$$
\mathrm{regret}_K \;\le\; 22\sqrt{\min\{\mathcal Q^\star,\, HV^\star\}\,HSAK\,\kappa^2} \;+\; 1924\,H^3S^2A\kappa^3,
$$
where $\mathcal Q^\star = \sum_{h\in[H]} \mathcal Q^\star_h$ and $\mathcal Q^\star_h = \max_{s,a} \mathrm{Var}_{s,a,h}\big(L_h + V^\star_{h+1}(s_{h+1})\big)$ is the **summed** maximum conditional variance. Optimal in two regimes at once: it matches the $\Omega(\sqrt{H^3SAK})$ worst-case lower bound (Domingues et al. 2021), and in low-stochasticity environments it matches the refined variance-dependent lower bounds of Zhou et al. (2023).

**Why the extension from bandits is hard.** $Q$-estimates at step $h$ depend on later value estimates through the Bellman backup, destroying the independence that makes the bandit analysis clean. A standard covering argument over value functions handles this but forces ensemble size **linear in $S$**. Logarithmic $B$ requires Lemma 5's substitution of $V^\star$ for the empirical value — a technique that in prior work is always intertwined with intricate bonus constructions, and which the absence of bonuses here isolates.

## Connections

- [[online-reinforcement-learning]] — the setting page; its optimism decomposition, variance-dependent rate and QoM summary come from here
- [[zanette2019Tighter]] — the count-based variance-dependent result this is measured against. **Their $\mathcal Q^*$ differ**: Zanette maxes over $(s,a,h)$, this sums the per-step maxima over $h$, so the quantities are not interchangeable — and the settings differ too, stationary transitions there against time-inhomogeneous here
- [[upper-confidence-bound]] — the bonus paradigm this dispenses with; evidence that a confidence width is one implementation of optimism rather than its definition
- [[pessimism-principle]] — the offline mirror, where the estimate is pushed the other way for the opposite reason
- [[instance-dependent-bounds]] — variance-dependent regret is this guarantee type
- [[smooth-aggregators]] — a quantile over an ensemble is an order statistic over *estimates*, structurally adjacent to those three max-replacement instances
- [[fqi]] — VIBE is value iteration with the regression step replaced by an ensemble; the backup template is unchanged
- Cassel et al. (2025), Viel et al. (2025), Feige (2004), Tiapkin et al. (2022a,b), Zhou et al. (2023), Domingues et al. (2021), Azar et al. (2017), Kaufmann et al. (2012) — cited author–year, no pages here

## Open Questions

- **Function approximation.** Whether the quantile mechanism extends to linear or general function classes is the paper's own closing question, and it decides whether this is a theoretical curiosity or the justification for what deep RL already does — since function approximation is exactly the regime where counts are unavailable.
- **Ensemble size under approximation.** The $V^\star$-substitution that buys logarithmic $B$ is a tabular argument; whether it survives is the technical form of the question above.
- **Tabular is the wrong proving ground for the motivation.** The case against bonuses is that counts do not exist in large state spaces — but the result is proved where they do. The argument is made in the one regime where it is not yet needed.
- **Empirical behaviour.** No experiments at all. Whether VIBE is competitive with tuned bonus-based methods, or with the deep ensembles it is meant to justify, is untested.
- **Which deployed ensembles this actually covers.** Bootstrapped DQN and SUNRISE resample *with replacement*; VIBE partitions **disjointly** by round-robin. Whether the guarantee says anything about the methods practitioners run is not analyzed, and the difference is not cosmetic — disjointness is what gives the independence the Chernoff step needs.
- **Max versus sum.** $\mathcal Q^\star$ accumulates per-step maxima over the horizon while [[zanette2019Tighter]]'s takes one global max; neither dominates, and which is the better account of instance hardness is open and untested on either side.
