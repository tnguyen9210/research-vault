---
title: "Online Reinforcement Learning"
aliases: [online RL, episodic RL, regret minimization in MDPs]
tags: [reinforcement-learning, exploration, regret, sequential-decision-making]
---

# Online Reinforcement Learning

**Definition:** Learning to act in an unknown MDP by interacting with it — the learner plays an episode, observes only the rewards and transitions its own actions produce, updates, and plays again. It is judged by **regret**, the cumulative shortfall of the policies it played against the optimal one, so every exploratory episode is paid for at the moment it is played.

> **Scope.** The episodic, finite-horizon problem, its objective, and the one mechanism every provably efficient method rests on — optimism — in the two forms the vault holds. **Left to other pages:** the batch problem where the data is fixed in advance and coverage replaces exploration, [[offline-reinforcement-learning]]; the $H=1$ case, [[contextual-bandits-online]]; the optimism principle in the context-free bandit, [[upper-confidence-bound]]; the regression template that most algorithms here instantiate, [[fqi]]; the deep-RL machinery, [[deep-q-network]]. This page has no literature survey: the vault holds too few online-RL papers to organize one honestly, and says so under Current State.

## Intuition

The agent faces the exploration–exploitation trade-off of the bandit problem compounded over a horizon: a state it never visits has a value it can never learn, and reaching that state may require a sequence of actions that each look bad in isolation. What separates this from [[offline-reinforcement-learning]] is the direction of the failure. Offline, a wrong value for an unseen action is never tested, so it must be *distrusted* — pessimism. Online, a wrong value that is too high will be acted on, tested, and corrected, so it can be *trusted provisionally* — optimism. Same confidence machinery, opposite sign, because online mistakes are self-correcting and offline ones are not.

The standard way to buy optimism is a **bonus**: estimate a mean, add a term proportional to $1/\sqrt{N_h(s,a)}$, and plan against the inflated values. This is theoretically tight and practically useless — visitation counts do not exist in high-dimensional or continuous state spaces, which is why deep RL explores with ensembles instead, until recently with no theory. The alternative the vault now holds is optimism from **order statistics**: partition the data, fit an ensemble, and act on a fixed quantile of it. No counts, no bonus, no posterior.

## Formal Description

Notation follows [[contextual-bandits-offline]] §2 and the horizon-$H$ extension on [[offline-reinforcement-learning]]; only what the online protocol changes is stated here.

The same episodic MDP $(\mathcal S, \mathcal A, \rho, P, H)$, with $S := |\mathcal S|$, $K := |\mathcal A|$, mean rewards $r_h(s,a) \in [0,1]$, transitions $P_h$, initial state $s_1 \sim \nu$, values $Q^\pi_h, V^\pi_h$ and optimal $Q^*_h, V^*_h$, and $J(\pi) := \mathbb E_{s_1 \sim \nu}[V^\pi_1(s_1)]$ — but no behavior policy and no dataset. The learner chooses a policy $\pi_t$ for each episode $t = 1, \dots, T$ from everything observed so far, runs it, and observes the trajectory $(s^t_h, a^t_h, r^t_h)_{h \in [H]}$ it produces. Nothing about unplayed actions is revealed.

**Regret.** The objective is cumulative,
$$
\mathrm{Reg}(T) := \sum_{t=1}^T \big( J(\pi^*) - J(\pi_t) \big) = \sum_{t=1}^T \Delta(\pi_t),
$$
the sum over episodes of the offline page's suboptimality of the policy actually played. (The RL literature writes $K$ for the number of episodes; this vault reserves $K$ for $|\mathcal A|$ and writes $T$ episodes, so a rate quoted from a paper as $\sqrt{H^3 S A K}$ reads $\sqrt{H^3 S K T}$ here.) The online-to-batch conversion of [[contextual-bandits-online]] applies unchanged: the uniform mixture of $\pi_1, \dots, \pi_T$ has $\Delta(\bar\pi) = \mathrm{Reg}(T)/T$.

**Minimax rate.** For tabular time-inhomogeneous MDPs with rewards in $[0,1]$ per step, $\mathrm{Reg}(T) = \tilde\Theta\big(\sqrt{H^3 S K T}\big)$ — the upper bound from bonus-based value iteration and the matching lower bound as stated in Cassel & Rosenberg (2026), citing Azar et al. (2017) and Domingues et al. (2021). The **variance-dependent** refinement replaces $H^2$ in the leading term by the maximum cumulative conditional variance of the optimal value, $\mathcal Q^* := \sum_h \max_{s,a} \mathrm{Var}_{s,a,h}\big(r_h + V^*_{h+1}(s_{h+1})\big) \le H^2$, giving $\tilde O\big(\sqrt{\min\{\mathcal Q^*, H V^*\}\, H S K T}\big)$ — the same rate in the worst case and strictly better in low-stochasticity instances. This is the online analogue of the [[instance-dependent-bounds]] the offline pages are stated in.

**The optimism decomposition.** Write $\hat V^t_1$ for the value the learner's own estimate assigns to $\pi_t$ at the start of episode $t$. Then
$$
\mathrm{Reg}(T) = \underbrace{\sum_{t} \big( \hat V^t_1(s_1) - J(\pi_t) \big)}_{\text{bias}} \;+\; \underbrace{\sum_{t} \big( J(\pi^*) - \hat V^t_1(s_1) \big)}_{\text{optimism}}.
$$
An optimistic algorithm is one whose estimate satisfies $\hat V^t_1 \ge V^*_1$ with high probability, making the second sum non-positive; the regret is then controlled entirely by the bias, the gap between what the learner believed and what it got. This is the exact mirror of the pessimism lemma on [[contextual-bandits-offline-value-based]], which arranges $\hat V \le V^*$ so that the *other* term drops. How optimism is established is the whole design question: by a bonus, whose size then dominates the bias analysis; or by an ensemble quantile, in which case the bias analysis is the standard one and the optimism argument is a binomial tail bound.

## Key Papers

- Cassel & Rosenberg (2026), *Quantile of Means* — read in full, no page yet. Value Iteration with Bootstrap Ensemble: partition each $(s,a,h)$'s data round-robin into $B = O(\log(SKHT/\delta))$ batches, plan on the $\alpha$-quantile of the $B$ Q-estimates with $\alpha$ a constant. Achieves the variance-dependent rate above with no bonuses, no counts and no distributional assumptions. The optimism comes from a first-moment fact — a batch mean with $+1$ in the denominator undershoots the true mean with probability at least $1/13$ — and a Chernoff bound over independent batches. First provably efficient ensemble exploration in MDPs.
- [[Song2019Revisiting]] — the online deep-RL end: softmax in place of max in the DQN target, with a finite-temperature bound on the gap. What the vault holds on how the value update itself behaves online, as opposed to how exploration is driven.
- [[deep-q-network]] — the algorithm family the practical side of this page lives in, and the one for which ensemble exploration was a heuristic without a guarantee until the first paper above.

## Variants

- **Infinite-horizon discounted and average-reward** — the same problem with $H$ replaced by $1/(1-\gamma)$ or by a mixing time; not paged
- **Model-based vs. model-free** — whether the learner estimates $P_h$ and plans, or estimates $Q_h$ directly; the ensemble method above is model-based in the tabular case
- **Linear and general function approximation** — where the count-based bonus stops being computable; the open frontier named below
- [[contextual-bandits-online]] — the $H = 1$ case, where the whole problem is the per-context exploration trade-off and regret is $\mathrm{Reg}_\Lambda(T)$

## Related Concepts

- [[offline-reinforcement-learning]] — the batch counterpart; the same MDP, the same $J$ and $\Delta$, but no feedback loop, so coverage replaces exploration and pessimism replaces optimism
- [[upper-confidence-bound]] — optimism in the context-free case, and the bonus construction that the count-based line generalizes to MDPs
- [[pessimism-principle]] — the mirror image: same confidence widths, subtracted rather than added, for the reason given under Intuition
- [[fqi]] — the value-iteration template; its "Offline vs. online FQI" section is the vault's statement that the Bellman update is byte-identical across the two regimes and only the data source differs
- [[instance-dependent-bounds]] — variance-dependent regret is this guarantee type stated online
- [[smooth-aggregators]] — an ensemble quantile is itself a smooth-aggregator-in-place-of-max, applied to *estimates* rather than to actions; not one of that page's three instances, but the same shape

## Current State and Open Problems

The tabular theory is complete: minimax and variance-dependent rates are matched by lower bounds, and as of 2026 both are achievable without bonuses. What the vault holds is thin — one paper read in full and two adjacent ones — so the survey is deferred rather than written from background knowledge. The open problems below are the ones the sources themselves name.

- **Ensemble optimism under function approximation.** Whether the quantile mechanism extends to linear or general function classes is the open question Cassel & Rosenberg state explicitly. It is the one that matters, since it is the setting in which counts are unavailable and ensembles are what practitioners already use.
- **Ensemble size.** The naive covering argument over value functions gives $B$ linear in $S$; the logarithmic $B$ requires substituting $V^*$ for the empirical value when proving optimism. Whether that substitution survives function approximation is the technical form of the previous question.
- **Theory for the ensembles people run.** Bootstrapped DQN, UCB Q-ensembles and SUNRISE are named by the source as the practical methods this line justifies in spirit — but none is analyzed, and the tabular method is not one of them. The gap between provable and deployed is the same one [[offline-reinforcement-learning]] records for its own side.
- **The vault's own gap.** No online-RL bonus-based paper is ingested — the count-based line the ensemble result is measured against is cited only through it. Zanette & Brunskill (2019) is in Zotero and would make the comparison checkable.

## Provenance

*Sourced.* The optimism decomposition, the variance-dependent rate and its definition of $\mathcal Q^*$, the QoM mechanism and the two open problems on ensembles come from Cassel & Rosenberg (2026), arXiv 2606.20107, read in full on 2026-09-22; not yet ingested, so no paper page. [[Song2019Revisiting]] and [[deep-q-network]] from their vault pages.

*Cited through a source, not checked directly.* Azar et al. (2017), Jin et al. (2018), Domingues et al. (2021) for the minimax rate; Zanette & Brunskill (2019) and Zhou et al. (2023) for the variance-dependent bound and its lower bound; Bootstrapped DQN, UCB Q-ensembles and SUNRISE as the practical ensemble methods. All taken from how Cassel & Rosenberg position themselves.

*This page's judgment, not a citation.* The framing of optimism and pessimism as one machinery with opposite signs, the reading of the ensemble quantile as a smooth aggregator over estimates, and the decision not to write a survey.
