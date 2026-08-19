---
title: "Power Mean Estimation in Stochastic Monte-Carlo Tree Search"
authors: [Tuan Dam, Odalric-Ambrym Maillard, Emilie Kaufmann]
year: 2024
venue: arXiv
arxiv: "2406.02235"
tags: [mcts, planning, power-mean, reinforcement-learning, convergence]
source: raw/papers/Dam2024Power.pdf
---

# Power Mean Estimation in Stochastic Monte-Carlo Tree Search

**TL;DR:** Introduces Stochastic-Power-UCT, an MCTS algorithm for stochastic MDPs using power mean as the value backup operator, with a provable $\mathcal{O}(n^{-1/2})$ convergence rate for root-node value estimation — extending Fixed-Depth-MCTS's guarantees from deterministic to stochastic environments.

## Problem

UCT's logarithmic exploration bonus $B(t, s, a) = C\sqrt{\log(T_s(t)) / T_{s,a}(t)}$ rests on a flawed assumption: it assumes regret at each node concentrates exponentially, but it actually concentrates polynomially (Audibert et al. 2009). This breaks the theoretical foundation of UCT in stochastic settings. Fixed-Depth-MCTS (Shah et al. 2022) fixes this with a polynomial bonus, but only for deterministic environments. Power-UCT (Dam et al. 2019) uses power mean for better value estimates but lacks a complete convergence proof in stochastic MDPs. No prior work gave a theoretically sound MCTS algorithm with convergence guarantees for stochastic settings using general power mean estimators.

## Method

**Stochastic-Power-UCT** combines a polynomial exploration bonus with power mean value backup:

**Value backup (power mean):**

$$\hat{V}_t(s_h) = \left( \sum_{a \in \mathcal{A}_{s_h}} \frac{T_{s_h,a}(t)}{t} \left(\hat{Q}_{T_{s_h,a}(t)}(s_h, a)\right)^p \right)^{1/p}, \quad p \in [1, +\infty)$$

With $p = 1$: reduces to the average mean (Fixed-Depth-MCTS). With $p \to \infty$: approaches max. $p = 2$ performs best empirically.

**Polynomial exploration bonus:**

$$B(t, s_h, a) = C \frac{T_{s_h}(t)^{b_{h+1}/\alpha_{h+1}}}{T_{s_h,a}(t)^{\alpha_{h+1}/\alpha_{h+1}}}, \quad h = 0, 1, \ldots, H-1$$

**Optimal parameter choice** (from Remark 2): setting $\alpha_i / \beta_i = 1/2$ and $b_i / \beta_i = 1/4$ gives:

$$B_h(n, s, a) = C \frac{n^{1/4}}{T_{s,a}(n)^{1/2}}$$

achieving the best possible $\mathcal{O}(n^{-1/2})$ convergence rate.

**Q-value update:**

$$\hat{Q}_t(s_h, a) = \frac{1}{t} \sum_{i=1}^t \left[ r^i(s_h, a) + \gamma \hat{V}_{T^{s_{h+1}}_{s_h,a}(i)}(s_{h+1}) \right]$$

## Results

**Theorem 1 (Power mean concentration, non-stationary MAB):** Under the optimistic action selection strategy (Eq. 5), the power mean backup $\hat{\mu}_n(p)$ concentrates at rate $(\alpha', \beta')$ towards $\mu_*$, where $\alpha' = (b-1)(1 - b/\alpha)$ and $\beta' = (b-1)$.

**Theorem 2 (Full tree convergence):** For any node $s_h$ at depth $h \in [0, H]$:
- Value estimates: $\hat{V}_n(s_h) \xrightarrow{\alpha_h, \beta_h}_{n \to \infty} \tilde{V}(s_h)$
- Q-value estimates: $\hat{Q}_n(s_h, a) \xrightarrow{\alpha_{h+1}, \beta_{h+1}}_{n \to \infty} \tilde{Q}(s_h, a)$ for all $a \in \mathcal{A}_{s_h}$

**Theorem 3 (Expected payoff):**

$$\left| \mathbb{E}[\hat{V}_n(s_0)] - \tilde{V}(s_0) \right| \leq \mathcal{O}(n^{-1/2})$$

**Empirical results** (SyntheticTree, FrozenLake $4\times4$, $8\times8$, Taxi):
- Stochastic-Power-UCT ($p = 2$) consistently outperforms UCT, Fixed-Depth-MCTS ($p = 1$), and other $p$ values
- Validated across branching factors $k \in \{2,4,6,8,10,16\}$ and depths $d \in \{1,2,3,4\}$

## Strengths & Limitations

**Strengths:** First complete convergence proof for power-mean MCTS in stochastic environments. Optimal exploration bonus derived from theory. General: $p=1$ recovers Fixed-Depth-MCTS as a special case. Empirically robust across environments.

**Limitations:** Convergence rate $\mathcal{O}(n^{-1/2})$ is shared with Fixed-Depth-MCTS — power mean does not improve the rate, only the constant. Hyperparameter conditions (Table 1) are complex. No adversarial MDP extension.

## Connections

- [[monte-carlo-tree-search]] — the algorithmic framework this paper operates within
- [[power-mean-mcts]] — the novel algorithm introduced here
- [[Kaufmann-Emilie]] — third author; also known for TAS (BAI), connecting to [[best-arm-identification]]
- [[Dam-Tuan]] — first author
- [[Maillard-Odalric-Ambrym]] — second author
- **Extends:** Fixed-Depth-MCTS (Shah et al. 2022) — deterministic predecessor
- **Extends:** Power-UCT (Dam et al. 2019) — adds theoretical guarantees
- **Thematic parallel:** [[Song2019Revisiting]] — same core idea (replace max backup with a smooth aggregator to reduce estimation noise) applied to deep Q-learning targets rather than MCTS; both papers argue greedy max is suboptimal under value estimation error

## Open Questions

- Can Stochastic-Power-UCT be extended to adversarial MDP settings?
- Is $\mathcal{O}(n^{-1/2})$ the optimal convergence rate for stochastic MCTS, or can it be improved?
- Can power mean be combined with deep learning value estimators (e.g., AlphaZero-style)?
- What is the optimal $p$ value as a function of the environment structure?
