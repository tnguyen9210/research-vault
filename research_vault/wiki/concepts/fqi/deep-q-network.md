---
title: "Deep Q-Network (DQN)"
tags: [deep-reinforcement-learning, q-learning, bellman-operator, deep-q-network]
introduced_by: [[Song2019Revisiting]]
---

# Deep Q-Network (DQN)

**Definition:** A Q-learning algorithm that parameterizes the Q-function with a deep neural network $Q_\theta(s,a)$, trained via stochastic gradient descent on the Bellman TD error using experience replay and a periodically-frozen target network.

## Intuition

Tabular Q-learning is intractable for large state spaces. DQN (Mnih et al. 2015) scales Q-learning to high-dimensional inputs (e.g. Atari pixels) using a convolutional neural network. Two key stabilization tricks: (1) **experience replay** — sample random minibatches from a buffer, breaking correlations; (2) **target network** $\theta^-$ — frozen for $C$ steps, used only in the Bellman target, preventing moving-target instability.

## Formal Description

The DQN training objective minimizes:

$$
\min_\theta \frac{1}{2}\left\| Q_\theta(s,a) - \left[R(s,a) + \gamma \max_{a'} Q_{\theta^-}(s',a')\right]\right\|^2,
$$

optimized by RMSProp with minibatches from a replay buffer. $\theta^-$ is updated to $\theta$ every $C$ steps.

**Double DQN (DDQN)** (van Hasselt et al. 2016a): replaces the target with $R(s,a) + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s',a'))$, decoupling action selection from evaluation to reduce [[overestimation-bias]].

**S-DQN / S-DDQN** ([[Song2019Revisiting]]): replace $\max_{a'}$ in the target with the [[softmax-bellman-operator]] at inverse temperature $\tau$. Exploration ($\varepsilon$-greedy) is unchanged.

## Key Papers

- Mnih et al. (2015) — original DQN; human-level Atari performance
- van Hasselt et al. (2016a) — DDQN; addresses overestimation
- [[Song2019Revisiting]] — S-DQN/S-DDQN; softmax target reduces overestimation and gradient noise, outperforms DDQN on Atari

## Variants & Related Concepts

- [[fitted-q-iteration]] — DQN is an incremental, stochastic FQI: the target network plays the role of the previous iterate $Q_k$, and SGD steps replace the full refit
- [[overestimation-bias]] — core failure mode of the max operator in DQN targets
- [[softmax-bellman-operator]] — drop-in replacement for max in the DQN target network
- Distributional DQN (C51, QR-DQN) — models full return distribution; addresses overestimation differently
- Rainbow (Hessel et al. 2018) — combines 6 DQN improvements; current strong baseline
- [[implicit-q-learning]] — the offline counterpart: same TD machinery, but the target's max is replaced by an in-sample [[expectile-regression]] estimate so no unseen action is ever evaluated

## Current State

DQN and DDQN are the foundational baselines for value-based deep RL. Modern practice uses Rainbow or distributional variants. The [[softmax-bellman-operator]] offers a simple single-hyperparameter improvement over vanilla DQN/DDQN targets, though it has not been widely adopted in post-2019 architectures.
