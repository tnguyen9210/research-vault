---
title: "Monte-Carlo Tree Search (MCTS)"
tags: [planning, reinforcement-learning, tree-search, bandits]
introduced_by: [[Dam2024Power]]
---

# Monte-Carlo Tree Search (MCTS)

**Definition:** A family of online planning algorithms that build a search tree incrementally by running Monte-Carlo simulations (trajectories) from the root, using bandit-style exploration bonuses to guide action selection at each node.

## Intuition
MCTS treats each internal tree node as a multi-armed bandit problem: which action (child) is worth exploring next? By running many simulated trajectories and updating value estimates along the path, the tree progressively focuses computation on the most promising regions of the state space. At the end, the action with the best estimated value at the root is returned.

## Formal Description
MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma \rangle$. Starting from root $s_0$, each trajectory of length $\leq H$ is collected by a generic UCT-like algorithm that selects actions via:

$$a_h = \arg\max_{a \in \mathcal{A}_{s_h}} \left\{ \hat{Q}_{T_{s_h,a}(t)}(s_h, a) + B(t, s_h, a) \right\}$$

After $n$ trajectories, the guess for best action is:

$$\hat{a}_n = \arg\max_{a \in \mathcal{A}_{s_0}} \hat{Q}_{T_{s_0,a}(n)}(s_0, a)$$

Quality measured by convergence rate $r(t)$: $|\mathbb{E}[\hat{V}_n(s_0)] - V^*(s_0)| \leq r(n)$.

**Key algorithmic choices:**
- **Bonus function** $B(t, s, a)$: logarithmic (UCT), polynomial (Fixed-Depth-MCTS, Stochastic-Power-UCT)
- **Value backup operator**: average mean, max, or power mean $p \in [1, \infty)$
- **Playout policy** $\pi_0$: random rollout or learned value network

## Key Papers
- [[Dam2024Power]] — introduces Stochastic-Power-UCT: power mean backup + polynomial bonus; $\mathcal{O}(n^{-1/2})$ convergence in stochastic MDPs

## Variants & Related Concepts
- [[power-mean-mcts]] — power mean backup operator; generalizes average and max
- **UCT** (Kocsis & Szepesvári 2006) — logarithmic bonus; flawed theory in stochastic settings
- **Fixed-Depth-MCTS** (Shah et al. 2022) — polynomial bonus; deterministic environments only

## Current State
UCT remains dominant in practice (AlphaGo, AlphaZero). Theoretical foundations for stochastic settings are incomplete; [[Dam2024Power]] makes progress. Combining MCTS with deep learning value functions is an active frontier.
