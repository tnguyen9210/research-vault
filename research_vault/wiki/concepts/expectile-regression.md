---
title: "Expectile Regression"
tags: [statistics, regression, offline-reinforcement-learning, value-estimation]
introduced_by: [[Kostrikov2022Offline]]
---

# Expectile Regression

**Definition:** Estimation of the $\tau$-expectile of a (conditional) distribution by minimizing an asymmetric squared loss $L_2^\tau(u) = |\tau - \mathbb{1}(u<0)|\,u^2$, which upweights positive residuals when $\tau > 0.5$ and so tracks an upper statistic of the distribution rather than its mean.

## Intuition

Ordinary least squares fits the conditional mean because the squared loss penalizes over- and under-prediction symmetrically. Break the symmetry — charge $\tau$ for being too low and $1-\tau$ for being too high — and the minimizer moves up the distribution. At $\tau = 0.5$ you recover the mean; as $\tau \to 1$ the fit is dragged toward the supremum of the support.

This gives a way to **approximate a maximum by regression**. That matters in offline RL: you can estimate $\max_a Q(s,a)$ over the actions the behavior policy actually takes, without ever evaluating $Q$ at an action you have not seen. The maximization becomes a property of the loss function rather than an explicit `argmax` over a queried set. See [[implicit-q-learning]].

## Formal Description

For a real-valued random variable $X$, the $\tau \in (0,1)$ expectile is

$$
m_\tau = \arg\min_{m} \; \mathbb{E}_{x\sim X}\big[L_2^\tau(x - m)\big],
\qquad L_2^\tau(u) = |\tau - \mathbb{1}(u < 0)|\,u^2 .
$$

Conditionally, $m_\tau(\cdot) = \arg\min_{m(\cdot)} \mathbb{E}_{(x,y)\sim\mathcal{D}}[L_2^\tau(y - m(x))]$, which is directly optimizable by SGD with unbiased gradients — a one-line change to an MSE objective.

Two properties do the work in [[Kostrikov2022Offline]]:

- **Monotonicity:** $\tau_1 < \tau_2 \implies m_{\tau_1} \le m_{\tau_2}$.
- **Limit (Lemma 1):** for bounded support with supremum $x^*$, $\lim_{\tau\to 1} m_\tau = x^*$.

### Contrast with quantile regression

Quantile regression uses the asymmetric $\ell_1$ loss $|\tau - \mathbb{1}(u<0)|\,|u|$ and estimates quantiles; expectile regression uses the asymmetric $\ell_2$ loss and estimates expectiles. Expectiles are *not* quantiles: the $\tau$-expectile is defined by a balance of weighted mean deviations rather than a probability mass split, so the two coincide only at $\tau = 0.5$ for symmetric distributions.

[[Kostrikov2022Offline]] prefers expectiles for a pragmatic reason — the $\ell_2$ form is a minimal modification to the MSE loss already present in TD learning — and reports it worked "somewhat better" than the $\ell_1$ alternative. No analysis is offered for why, which is an open question.

### The statistic being taken matters

A subtlety that is easy to miss. Applying expectile regression to a full TD target $r + \gamma Q(s',a')$ takes an upper statistic over **both** action randomness and transition randomness, so a high value may reflect a lucky transition rather than a good action — optimism about the environment, which is unsound. [[Kostrikov2022Offline]] avoids this by taking the expectile only over actions (fitting $V_\psi$) and using plain MSE for the backup over dynamics.

## Key Papers

- Newey & Powell (1987) — introduces expectile regression in econometrics (asymmetric least squares)
- Koenker & Hallock (2001) — quantile regression, the $\ell_1$ counterpart
- [[Kostrikov2022Offline]] — uses upper expectiles of $Q(s,\cdot)$ to perform in-sample maximization in offline RL; Theorem 3 shows $\tau \to 1$ recovers the support-constrained optimal value
- Dabney et al. (2018a,b) — QR-DQN / IQN use *quantile* regression in RL, but over the return distribution induced by stochastic transitions, a different statistic with a different purpose

## Variants & Related Concepts

- [[implicit-q-learning]] — the algorithm built on this primitive
- [[fitted-q-iteration]] — the template IQL modifies; the expectile replaces the $\max_{a'}$ in its Bellman target
- [[softmax-bellman-operator]] — a different smooth surrogate for the max, with an explicit finite-$\tau$ approximation bound ([[Song2019Revisiting]]) that the expectile treatment lacks
- [[power-mean-mcts]] — a third smooth-aggregator-in-place-of-max instance, in tree search ([[Dam2024Power]])
- [[overestimation-bias]] — the reason smooth surrogates are attractive in the first place

## Current State

Standard and widely adopted in offline RL after IQL — the expectile-$V$ / MSE-$Q$ pattern now appears across offline and offline-to-online algorithms, and IQL is a routine baseline. Theoretical understanding lags practice: the finite-$\tau$ approximation error, the finite-sample cost of estimating high expectiles (upper expectiles are effectively supported by fewer samples), and the expectile-vs.-quantile choice are all open. There is no instance-dependent characterization of how large $\tau$ must be for a given behavior-policy density — arguably the central missing result.
