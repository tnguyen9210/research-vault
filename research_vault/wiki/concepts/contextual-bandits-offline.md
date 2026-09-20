---
title: "Offline Contextual Bandits"
tags: [contextual-bandits, offline-contextual-bandits, pessimism, learning-theory]
aliases: [offline CB, batch contextual bandits, offline bandit policy learning, offline-contextual-bandits]
---

# Offline Contextual Bandits

**Definition:** Learning a decision rule $\mathcal X\to\mathcal A$ from a fixed dataset of context–action–reward triples logged by a behavior policy $\mu$, with no further interaction, so that the learner can never sample the actions it wishes it had seen.

> **Terminology guard.** "Offline" here means *learning from a fixed logged dataset*. The vault's [[offline-oracle-efficient-bandits]] topic is a different use of the word — *online* algorithms that call an *offline regression oracle*. Do not conflate. The pessimism literature writes $\mathrm{SubOpt}(\hat\pi)$, and the policy-learning literature $\mathrm{Regret}$, for what this page calls $\Delta(\hat\pi)$.

> **Scope.** This page is the setting and the taxonomy: what the
> problem is, the notation the family pages share, the coverage
> coefficients, and the two routes with the constructions that
> straddle them (§5) and their comparison (§6). **Left to other
> pages:** the value-based family in full — estimators, algorithms,
> assumptions, guarantees — on [[contextual-bandits-offline-value-based]]; the
> policy-based family, which has no page yet and is covered here only
> by §4.2 and §6; the horizon-$H$ analysis the bandit case
> specializes, on [[fqi-finite-sample-analysis]]. §7 records what was
> checked against sources and what was not.

## 1. Intuition

Online, an algorithm that is unsure about an action can play it. Offline it cannot, and that single difference drives the whole theory. The data distribution is whatever $\mu$ happened to produce, the learner's errors are never self-correcting, and an action that was rarely logged is both badly estimated and — if the learner acts greedily on its estimates — disproportionately likely to be chosen, because the noise that inflates an estimate is exactly what makes it win an $\arg\max$. Everything below is organized around that asymmetry: how much the logged distribution has to overlap the policy one wants to deploy, and what an algorithm can do to need less overlap.

## 2. Formal Description

Notation follows the Overleaf research log (`02_offline_contextual_bandits.tex`, §1) and is shared with [[contextual-bandits-offline-value-based]].

### 2.1 Offline data

Let $\mathcal X$ be a context space and $\mathcal A$ a finite action set of size $K:=|\mathcal A|$. A **policy** is a map $\pi:\mathcal X\to\Delta(\mathcal A)$ taking a context to a distribution over the actions. In an offline setting the policy that collects the data has to be fixed in advance, and such a policy is called a **behavior policy**. A problem instance is specified by a context distribution $\nu\in\Delta(\mathcal X)$, a reward kernel $\rho$ mapping a context–action pair to a distribution supported on $[0,1]$, and a behavior policy $\mu$. The learner observes only a fixed dataset
$$
\mathcal D:=\{(x_t,a_t,r_t)\}_{t=1}^{T},
$$
collected by running $\mu$ for $T$ rounds: $x_t\sim\nu$, $a_t\sim\mu(\cdot\mid x_t)$, $r_t\sim\rho(\cdot\mid x_t,a_t)$, independently across $t$. Both $\nu$ and $\rho$ are unknown to the learner. The behavior policy is fixed before the data is collected and cannot be adapted in response to what has been observed, and the learner takes no actions of its own. This is what makes the problem offline, and it is what separates the objective below from online cumulative regret and from active best-arm identification, where the sampling distribution can be steered toward the informative pairs.

Whether $\mu$ itself is available to the learner is a modelling choice, and it is what separates the two algorithm families below. The value-based methods never use it. The policy-level methods need the logged propensities $\mu(a_t\mid x_t)$, which we assume are recorded whenever such a method is considered. For the value-based methods $\mu$ enters only through the coverage coefficients below. Where a result uses a sample split, $\mathcal D^{\mathrm{reg}}$ and $\mathcal D^{\mathrm{eval}}$ denote the first $T_{\mathrm{reg}}=\lceil T/2\rceil$ and the last $m=T-T_{\mathrm{reg}}$ triples; this is stated explicitly wherever it happens.

### 2.2 Mean reward, value, and optimal policy

Define the conditional mean reward, its variance, and the optimal value
$$
q^*(x,a):=\mathbb E[r\mid x,a],\qquad \sigma^2(x,a):=\mathrm{Var}(r\mid x,a)\le\tfrac14,\qquad V^*(x):=\max_{a\in\mathcal A}q^*(x,a).
$$
For any $f:\mathcal X\times\mathcal A\to\mathbb R$ and any policy $\pi$ write $f(x,\pi):=\sum_a\pi(a\mid x)f(x,a)$, which is $f(x,\pi(x))$ for deterministic $\pi$. The **value** of a policy is
$$
J(\pi):=\mathbb E_{x\sim\nu,\ a\sim\pi(\cdot\mid x)}\big[q^*(x,a)\big]=\mathbb E_{x\sim\nu}\big[q^*(x,\pi)\big],
$$
and $\pi^*$ is greedy with respect to $q^*$, $\ \pi^*(x)\in\arg\max_aq^*(x,a)$. Because the maximization decouples across contexts, $J(\pi^*)=\mathbb E_{x\sim\nu}[V^*(x)]$ and $\pi^*$ maximizes $J$ over *all* policies. We may therefore take $\pi^*$ deterministic, and no policy class has to be fixed as part of the problem definition. A policy class $\Pi$ enters only when an algorithm restricts its search to $\Pi$, in which case the comparator becomes $\pi^*_\Pi:=\arg\max_{\pi\in\Pi}J(\pi)$ ([[contextual-bandits-offline-value-based]], the comparison below).

### 2.3 Learning objective

A learning rule maps the dataset to a policy, $\mathcal D\mapsto\hat\pi$, and is measured by its simple regret, also called the **suboptimality gap**,
$$
\Delta(\hat\pi):=J(\pi^*)-J(\hat\pi).
$$
Since $\hat\pi$ depends on $\mathcal D$, $\Delta(\hat\pi)$ is a random variable. A high-probability guarantee has the form $\mathbb P_{\mathcal D}(\Delta(\hat\pi)\le B_T(\delta))\ge1-\delta$ for some bound $B_T(\delta)$, whereas an expected guarantee controls $\mathbb E_{\mathcal D}[\Delta(\hat\pi)]$ instead. All statements "with probability at least $1-\delta$" refer to the draw of $\mathcal D$; union bounds are made explicit.

### 2.4 Function approximation and realizability

Let $\mathcal F\subseteq\{f:\mathcal X\times\mathcal A\to[0,1]\}$ be a function class used to model the mean reward, and assume **realizability**, $q^*\in\mathcal F$. The learner fits the class by least squares,
$$
\hat q:=\arg\min_{f\in\mathcal F}\ \frac1T\sum_{t=1}^{T}\big(f(x_t,a_t)-r_t\big)^2. \tag{1}
$$
Write $\pi_f(x):=\arg\max_af(x,a)$ for the greedy policy of $f$ and $\Pi_{\mathcal F}:=\{\pi_f:f\in\mathcal F\}$ for the induced policy class. For the empirical and population squared losses and the norms under the data and target distributions write
$$
\mathcal L_{\mathcal D}(f):=\frac1{|\mathcal D|}\sum_{(x,a,r)\in\mathcal D}\big(f(x,a)-r\big)^2,\qquad
\mathcal L(f):=\mathbb E_{d^\mu\times\rho}\big[(f(x,a)-r)^2\big],
$$
$$
\|g\|^2_{\nu\times\pi}:=\mathbb E_{x\sim\nu,\ a\sim\pi(\cdot\mid x)}\big[g(x,a)^2\big],\qquad
\|g\|_{L_1(\nu\times\pi)}:=\mathbb E_{(x,a)\sim d^\pi}\big|g(x,a)\big| ,
$$
so that $\|\cdot\|_{\nu\times\mu}$ is the norm under the data distribution. Since $q^*$ is the Bayes regressor, $\mathcal L(f)-\mathcal L(q^*)=\|f-q^*\|^2_{\nu\times\mu}$ for every $f$ (the Pythagorean identity in [[contextual-bandits-offline-value-based]]) — an identity needing only that $q^*$ is the conditional mean. What realizability adds is that $q^*$ is the population minimizer *within* $\mathcal F$, so the excess loss of the empirical minimizer over its own class is exactly its squared estimation error. This is the assumption that at $H=1$ plays the role Bellman completeness plays in fitted $Q$-iteration, and it is what converts a statement about the empirical loss into one about the estimation error of $\hat q$.

In the **tabular model**, $|\mathcal X|=S<\infty$ and $\mathcal F$ is unrestricted, so there are $SK$ unknown means. Writing $N(x,a):=\sum_t\mathbb 1\{(x_t,a_t)=(x,a)\}$ for the number of times the pair appears in $\mathcal D$, the least-squares fit reduces to the per-cell empirical mean $\hat q(x,a)=N(x,a)^{-1}\sum_{t:(x_t,a_t)=(x,a)}r_t$ at every pair with $N(x,a)\ge1$. A pair with $N(x,a)=0$ is left undetermined by the data, since any value in $[0,1]$ minimizes the empirical loss there; how such pairs are treated is what separates a greedy rule from a pessimistic one, and it is why the coverage coefficients below are needed.

In the **linear model**, a known feature map $\phi:\mathcal X\times\mathcal A\to\mathbb R^d$ with $\|\phi\|_2\le1$ satisfies $q^*(x,a)=\phi(x,a)^\top\theta^*$ for an unknown $\theta^*$ with $\|\theta^*\|_2\le B$, and the least-squares fit is the ordinary or ridge estimate of $\theta^*$, with regularized Gram matrix $\Lambda:=\lambda I+\sum_t\phi_t\phi_t^\top$, population Gram matrix $\Sigma_\mu:=\mathbb E_{d^\mu}[\phi\phi^\top]$, and confidence radius $\beta_\delta$ (both fixed in [[contextual-bandits-offline-value-based]]). The tabular model is recovered by $\phi(x,a)=e_{(x,a)}$, with $d=SK$.

### 2.5 Coverage

For a policy $\pi$ let $d^\pi(x,a):=\nu(x)\pi(a\mid x)$ be the induced distribution over context–action pairs, so $d^\mu$ is the distribution of the logged pairs. Adopt the convention that a ratio with positive numerator and vanishing denominator is $+\infty$, and write $\mu_{\min}:=\inf_{x,a}\mu(a\mid x)$, which may be $0$. Two coefficients are central. The **single-policy concentrability** coefficient is
$$
C^*:=\max_{(x,a):\,d^{\pi^*}(x,a)>0}\frac{d^{\pi^*}(x,a)}{d^\mu(x,a)}=\max_{x:\,\nu(x)>0}\frac1{\mu(\pi^*(x)\mid x)},
$$
the second equality because $\pi^*$ is deterministic, and the **uniform** coefficient is
$$
C_{\mathrm{unif}}:=\max_\pi\max_{(x,a):\,d^\pi(x,a)>0}\frac{d^\pi(x,a)}{d^\mu(x,a)}=\max_{x:\,\nu(x)>0}\max_{a\in\mathcal A}\frac1{\mu(a\mid x)}=\frac1{\mu_{\min}} .
$$
Restricting each maximum to the pairs its numerator charges is not a formality: without it, a context of probability zero at which $\mu$ never plays the optimal action would send both coefficients to infinity, although such a context never occurs and no algorithm can be penalized for it. Always $C^*\le C_{\mathrm{unif}}$, and $C^*$ can be finite while $C_{\mathrm{unif}}$ is infinite — the behavior policy only has to cover the actions $\pi^*$ actually takes. Exploiting that gap is the purpose of pessimism. The coverage section of [[contextual-bandits-offline-value-based]] gives the full family ($C^\pi$, the second-moment $C^\pi_2$ with $\bar C^*:=C^{\pi^*}_2$ and $\bar C_{\mathrm{unif}}$, and the class-dependent $C_{\mathcal F}(\pi)$) together with the change of measure that uses them.


### 2.6 Standing assumptions

**Standing assumptions**, invoked by name below. (A1) *Data:* $(x_t,a_t,r_t)$ i.i.d. as in the offline-data paragraph above. (A2) *Boundedness:* rewards in $[0,1]$, every $f\in\mathcal F$ maps into $[0,1]$, $K<\infty$. (A3) *Realizability:* $q^*\in\mathcal F$. (A4) *Finite class:* $|\mathcal F|<\infty$; for infinite classes $\ln|\mathcal F|$ becomes a covering number or pseudo-dimension and the rates are unchanged. (A5) *Comparator:* value-based methods place no restriction on the policy, so the comparator is the global optimum $\pi^*$; where a class $\Pi$ is used (below, and in [[contextual-bandits-offline-value-based]]) it is $\pi^*_\Pi$.

## 3. Relation to offline RL

A contextual bandit is the one-step, $H=1$ case of offline reinforcement learning. Its action value is the immediate conditional mean reward, so there is no estimated next-state value, no Bellman-error propagation across iterations, and no need for a transition model. See [[fqi-finite-sample-analysis]] for the horizon-$H$ statement this specializes.


## 4. The two families of methods

Two families are considered. **Value-based** algorithms fit an action-value model $\hat q$, possibly subtract an uncertainty penalty $\Gamma$ to obtain a lower confidence bound $\underline q$, and return the greedy policy $\hat\pi(x)\in\arg\max_a\underline q(x,a)$. **Policy-level** algorithms instead optimize a criterion directly over a policy class $\Pi$, possibly using importance weights or a worst-case policy value over a confidence set of reward models. Each family has its own page: [[contextual-bandits-offline-value-based]] and (forthcoming) `policy-based-offline-bandits`.




### 4.1 The value-based route, in one paragraph

Fit a reward model $\hat q$ by least-squares regression on the logged triples and act greedily on it, $\hat\pi(x)\in\arg\max_a\hat q(x,a)$; in the modern form, subtract a computable uncertainty penalty first, $\hat\pi(x)\in\arg\max_a\hat q(x,a)-\Gamma(x,a)$. The propensities are never used, the behavior policy may be deterministic, and no policy class is specified, so the comparator is the global optimum $\pi^*$. Coverage enters only through the analysis. Full treatment: [[contextual-bandits-offline-value-based]].

### 4.2 The policy-based route, in one paragraph

The policy-based route estimates $J(\pi)$ for $\pi\in\Pi$ by importance weighting, $\widehat J^{\mathrm{IPW}}(\pi):=\frac1T\sum_t\frac{\pi(a_t\mid x_t)}{\mu(a_t\mid x_t)}r_t$, unbiased by the change of measure $\mathbb E_{a\sim\mu(x)}[\frac{\pi(a|x)}{\mu(a|x)}q^*(x,a)]=\mathbb E_{a\sim\pi(x)}[q^*(x,a)]$, and selects by MaxIPW ($\arg\max_\pi\widehat J^{\mathrm{IPW}}$) or PES ($\arg\max_\pi\widehat J^{\mathrm{IPW}}(\pi)-W^U_\pi$); implicit exploration (IX, weights $\pi/(\mu+\gamma)$, bias $\gamma C_\gamma(\pi)$ with $C_\gamma(\pi):=\mathbb E_x\sum_a\frac{\pi(a|x)q^*(x,a)}{\mu(a|x)+\gamma}$) and logarithmic smoothing (LS) tame unbounded weights. It needs an explicit $\Pi$, the propensities, a stochastic $\mu$ with $\mu(a\mid x)>0$ wherever $\pi$ puts mass, and nothing about $\rho$ beyond $r\in[0,1]$. the comparison below compares the two routes; the only fact about this route used before the comparison below is the Hoeffding width $|\widehat J^{\mathrm{IPW}}(\pi)-J(\pi)|\le\frac1{\mu_{\min}}\sqrt{\ln(2|\Pi|/\delta)/(2T)}$, valid for all $\pi\in\Pi$ simultaneously with probability at least $1-\delta$ (Hoeffding (see [[contextual-bandits-offline-value-based]])).


## 5. Where the families meet

Two constructions sit on the boundary: both start from a fitted reward model and both need the propensities, which is what makes them neither purely value-based nor purely policy-based.

### 5.1 The doubly robust estimator

Given an estimate $\hat\mu$ of the behavior policy (exact when propensities were logged), the **doubly robust** estimator corrects (DM) by the importance-weighted residual:
$$
\widehat J^{\mathrm{DR}}(\pi):=\frac1m\sum_{(x,a,r)\in \mathcal D^{\mathrm{eval}}}\Big[\hat q(x,\pi)+\frac{\pi(a\mid x)}{\hat\mu(a\mid x)}\big(r-\hat q(x,a)\big)\Big]. \tag{DR}
$$

**Proposition 5.1 (Dudík, Langford & Li 2011, Theorems 1–2, bandit form).** Let $g:=\hat q-q^*$, $\delta_\mu:=1-\mu/\hat\mu$, $g(x,\pi):=\sum_a\pi(a\mid x)g(x,a)$. Conditional on $\hat q$ and $\hat\mu$,
$$
\mathbb E\big[\widehat J^{\mathrm{DR}}(\pi)\big]-J(\pi)=\mathbb E_{d^\pi}\big[g\,\delta_\mu\big],
$$
so the bias is the product of the two model errors and vanishes if either is zero; and when $\hat\mu=\mu$,
$$
m\,\mathrm{Var}\big(\widehat J^{\mathrm{DR}}(\pi)\big)=\mathrm{Var}_x\big(q^*(x,\pi)\big)+\mathbb E_x\sum_a\frac{\pi(a\mid x)^2}{\mu(a\mid x)}\Big(\sigma^2(x,a)+g(x,a)^2\Big)-\mathbb E_x\big[g(x,\pi)^2\big].
$$
*Proof.* Condition on $x$. The summand has conditional mean $\hat q(x,\pi)+\sum_a\pi(a|x)\frac{\mu(a|x)}{\hat\mu(a|x)}(q^*-\hat q)(x,a)=q^*(x,\pi)+\sum_a\pi(a|x)g(x,a)\delta_\mu(x,a)$; average over $x$. For the variance take $\hat\mu=\mu$ and use the law of total variance: the conditional mean is $q^*(x,\pi)$ and the conditional variance is that of $\frac{\pi(a|x)}{\mu(a|x)}(r-\hat q(x,a))$ with $a\sim\mu(x)$, $r\sim\rho(x,a)$, namely $\sum_a\frac{\pi^2}{\mu}(\sigma^2+g^2)-g(x,\pi)^2$. $\square$

Setting $\hat q\equiv0$ recovers IPW with $(q^*)^2$ in place of $g^2$: DR keeps unbiasedness and pays the importance-weighting penalty only on the residual, small wherever the reward model is good; dropping the correction recovers (DM), whose variance has no $1/\mu$ term. Refinements keep the template: MRDR (Farajtabar et al. 2018) fits $\hat q$ to minimize the DR variance; SWITCH (Wang, Agarwal & Dudík 2017) uses (DM) exactly on the pairs whose importance weight exceeds a threshold; shrinkage (Su et al. 2020) shrinks the weights. All require $\mu$ or a good $\hat\mu$, which the value-based learner of the value-based route above does not.


### 5.2 Class-restricted learning with imputed rewards

When the deployed policy must lie in a given class $\Pi\subsetneq\mathcal A^{\mathcal X}$, the value-based learner becomes
$$
\hat\pi_\Pi:=\arg\max_{\pi\in\Pi}\widehat J^{\mathrm{DM}}(\pi)=\arg\max_{\pi\in\Pi}\frac1m\sum_{x\in \mathcal D^{\mathrm{eval}}}\hat q(x,\pi), \tag{DM-Π}
$$
a cost-sensitive classification problem with the imputed reward vector $\hat q(x,\cdot)$ at each context (Dudík, Langford & Li 2011, §5.1.3; Beygelzimer & Langford 2009). Its guarantee is a uniform-deviation bound.

**the class-restricted bound below.** Let $\Pi$ be finite and $\hat q$ fitted on $\mathcal D^{\mathrm{reg}}$. With probability at least $1-\delta$ over $\mathcal D^{\mathrm{eval}}$,
$$
\Delta_\Pi(\hat\pi_\Pi)\ \le\ 2\max_{\pi\in\Pi}\big|\widehat J^{\mathrm{DM}}(\pi)-J(\pi)\big|\ \le\ 2\sqrt{C_\Pi}\;\|\hat q-q^*\|_{\nu\times\mu}+2\sqrt{\frac{\ln(2|\Pi|/\delta)}{2m}},\qquad C_\Pi:=\max_{\pi\in\Pi}\min\{C^\pi_2,C_{\mathcal F}(\pi)\}.
$$
*Proof.* $J(\pi^*_\Pi)-J(\hat\pi_\Pi)\le[J(\pi^*_\Pi)-\widehat J^{\mathrm{DM}}(\pi^*_\Pi)]+[\widehat J^{\mathrm{DM}}(\hat\pi_\Pi)-J(\hat\pi_\Pi)]$ since $\widehat J^{\mathrm{DM}}(\hat\pi_\Pi)\ge\widehat J^{\mathrm{DM}}(\pi^*_\Pi)$; bound both brackets by the uniform version of the direct-method bound in [[contextual-bandits-offline-value-based]]. $\square$

The coverage is now that of the *whole class* — both $\pi^*_\Pi$ and the selected policy appear, exactly as in the $L_2$-form greedy bound in [[contextual-bandits-offline-value-based]] — so the class-restricted plug-in learner inherits the greedy rule's need for coverage of everything it might select. Pessimism restores single-policy coverage at the level of policies: the rule $\arg\max_{\pi\in\Pi}\min_{f\in\mathcal F_\varepsilon}\widehat J_f(\pi)$ satisfies, by the proof of the version-space theorem in [[contextual-bandits-offline-value-based]] with $\Pi$ in place of $\Pi_{\mathcal F}$,
$$
\Delta_\Pi\ \le\ \sqrt{\frac{20\,C_{\mathcal F}(\pi^*_\Pi)\ln(|\mathcal F|/\delta)}{T_{\mathrm{reg}}}}+2\sqrt{\frac{\ln(2N|\Pi|/\delta)}{2m}},
$$
which is the shape of Hoeffding (see [[contextual-bandits-offline-value-based]]) of Xie et al. (2021), with $\log(|\mathcal F||\Pi|/\delta)$ and the coverage of the comparator only. With doubly robust scores in place of $\hat q(x,\cdot)$, (DM-Π) becomes the policy learning of Athey & Wager (2021) and Zhou, Athey & Wager (2023): the same argmax over $\Pi$, a uniform-deviation bound driven by the DR variance of Proposition 1 and the complexity of $\Pi$, and cross-fitting for the nuisance estimates — a hybrid that uses the reward model as a control variate and the propensities for unbiasedness.


## 6. Comparing the two routes

The policy-based route sketched above is estimate-then-select with importance-weighted scores. Its guarantees are the same two inequalities as the two suboptimality lemmas in [[contextual-bandits-offline-value-based]], read over policies.

**Proposition 6.1 (MaxIPW and PES).** Let $\Pi$ be finite and $W_\pi:=\frac1{\mu_{\min}}\sqrt{\ln(2|\Pi|/\delta)/(2T)}$, so that $|\widehat J^{\mathrm{IPW}}(\pi)-J(\pi)|\le W_\pi$ for all $\pi\in\Pi$ with probability at least $1-\delta$ (Hoeffding, as in [[contextual-bandits-offline-value-based]]; summands in $[0,1/\mu_{\min}]$). Then MaxIPW satisfies $\Delta_\Pi\le W_{\pi^*_\Pi}+W_{\hat\pi}$ and PES with any valid per-policy widths satisfies $\Delta_\Pi\le W^L_{\pi^*_\Pi}+W^U_{\pi^*_\Pi}$.
*Proof.* the plug-in decomposition in [[contextual-bandits-offline-value-based]]'s first inequality with $f\to\widehat J^{\mathrm{IPW}}$, $q^*\to J$, actions $\to$ policies; and the pessimism lemma in [[contextual-bandits-offline-value-based]] likewise. $\square$

With the Hoeffding width the two bounds coincide because the width is policy-independent; with variance-adaptive widths — the IPW summands have variance at most $\mathbb E_{d^\mu}[(\pi/\mu)^2r^2]\le C^\pi_2$, so Bernstein (see [[contextual-bandits-offline-value-based]]) gives $W_\pi\asymp\sqrt{2C^\pi_2\ln(4|\Pi|/\delta)/T}+2\ln(4|\Pi|/\delta)/(3\mu_{\min}T)$ — PES is governed by $C^{\pi^*_\Pi}_2$ alone while MaxIPW is governed by $\max_\pi C^\pi_2$, the policy-level version of uniform versus single-policy coverage. (Making such widths data-driven is the hyperparameter-adaptation problem of `jun2026CS703Q10`, proposed in [[Ryu2025Improved]]. The two smoothed estimators are not interchangeable: IX, $\pi/(\mu+\gamma)$, bounds the weights by $1/\gamma$ and pays a bias of exactly $\gamma C_\gamma(\pi)$; LS, $b^{-1}\ln(1+b\hat r)$, is *not* bounded for fixed $b$, and pays $bD_b(\pi^*)$ with $D_b(\pi)\le C_b(\pi)$ — hence the better of the two bounds.)

| | Policy-based (IPW → MaxIPW / PES → IX / LS) | Value-based (regression → greedy / (LCB)) |
|---|---|---|
| Object estimated | $J(\pi)$ for each $\pi\in\Pi$ | the function $q^*$ on $\mathcal X\times\mathcal A$ |
| What must be well-specified | propensities $\mu$ (stochastic, known) | $\mathcal F\ni q^*$ (realizability, (A3)) |
| Source of bias | none (IPW); chosen, $\gamma C_\gamma(\pi)$ (IX) | misspecification, $\mathrm{dist}(q^*,\mathcal F)$; does not vanish with $T$ (Prop. 5.4) |
| Source of variance | ratios $\pi/\mu$, possibly unbounded | bounded regression; none from $\mu$ (the direct-method bias/variance identities in [[contextual-bandits-offline-value-based]]) |
| Deficient support | fatal: weights undefined | tolerated through $C_{\mathcal F}$ (Lemma 2.9(c)) |
| Comparator | best in $\Pi$ | global optimum $\pi^*$ |
| Uncertainty quantified | per policy, $W_\pi$ | per pair, $b(x,a)$ / $\Gamma(x,a)$ (Def. 2.10) |
| Pessimism | $\arg\max_\pi\widehat J(\pi)-W^U_\pi$ | $\arg\max_a\hat q(x,a)-\Gamma(x,a)$; equals per-policy pessimism on $\mathcal A^{\mathcal X}$ (Prop. 6.1) |
| Coverage in the bound | $C^{\pi^*_\Pi}_2$ (Bernstein widths); $C_\gamma(\pi^*_\Pi)$ (IX) | $C^*$, $\bar C^*$, $C_{\mathcal F}(\pi^*)$, $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ (§4) |
| Complexity factor | $\ln\lvert\Pi\rvert$ | $\ln N$, $S$, $d$, $\tilde d$ — larger: modeling rewards is harder than ranking actions |
| Learning (large class) | $\arg\max$ over $\Pi$: non-convex, NP-hard in general | regression (convex for linear $\mathcal F$) + per-context $\arg\max$ |
| Overparameterized models | objective not action-stable (Brandfonbrener et al. 2021) | action-stable; DR collapses to value-based |
| Selection (finite class) | native: one width per candidate | validation loss $\ne$ best policy (Qian & Murphy 2011, §3) |
| Fast rates | $1/T$ only under a gap in policy values | $1/T$ under per-context action gaps (Thm. 7.2) |
| Minimax OPE | IPS/DR match the lower bound | DM alone does not; DR combines both (§5.1 above) |

**Rule of thumb.** Trust the value-based route when a good reward model is plausible (rich features, dense actions, deterministic or unlogged $\mu$, large models); trust the policy-based route when the policy class is simple, $\mu$ is logged and stochastic, and rewards are hard to model. DR and SWITCH exist because the honest answer is usually "both, partially".


## 7. Provenance

*Notation.* §2 follows the Overleaf research log
(`02_offline_contextual_bandits.tex`, §1) and is shared verbatim with
[[contextual-bandits-offline-value-based]]; the two pages are meant to be read
with one set of symbols.

*Quoted.* Proposition 5.1 is Dudík, Langford & Li (2011), Theorems
1–2, written in bandit form; the cost-sensitive-classification reading
of §5.2 is their §5.1.3, and the uniform-deviation guarantee is
standard.

Proposition 6.1 is Lemmas 1 and 2 of `jun2026CS703Q10` (K.-S. Jun,
*CS703Q10: Offline contextual bandits*, CSED703Q, Spring 2026), which
states them in
exactly this abstracted form: $N$ variables $X_i$ with means $\mu_i$
and two-sided widths $W^L_i,W^U_i$, with MaxIPW giving
$\mu_1-\mu_J\le W^L_1+W^U_J$ and PES giving $\mu_1-\mu_J\le
W^L_1+W^U_1$. Read over policies rather than indices these are the
same two inequalities as the suboptimality lemmas of
[[contextual-bandits-offline-value-based]] §2.5, which is why the proof below is
a substitution.

*Derived here rather than quoted.* The route comparison in §6 is this
page's synthesis, not a table from the literature.

*Verify before citing.* The per-claim record for everything cited in
§5 and §6 is kept in one place, [[contextual-bandits-offline-value-based]] §8.3,
rather than duplicated here. Two entries there bear directly on this
page: the Singh & Yee (1994) attribution and the Athey & Wager /
Zhou–Athey–Wager theorem forms are both still recorded from memory.

## Key Papers

- Rashidinejad, Zhu, Ma, Jiao & Russell (NeurIPS 2021) — single-policy concentrability; the $(C^*-1)$ regimes; the lower bound
- Jin, Yang & Wang (ICML 2021) — uncertainty quantifiers and pessimistic value iteration, at horizon one
- Brandfonbrener, Whitney, Ranganath & Bruna (ICML 2021) — the value-based / policy-based distinction itself, and action-stability
- Dudík, Langford & Li (2011) — the direct method, IPS and doubly robust estimators, and the taxonomy the field still uses
- Xiao, Wu, Lattimore et al. (ICML 2021) — what pessimism does and does not buy in the batch setting
- [[Ryu2025Improved]] — the policy-side second-order route

## Variants & Related Concepts

- [[contextual-bandits-offline-value-based]] — the reward-model family: estimators, algorithms, assumptions, guarantees
- [[contextual-bandits]] — the online problem this is the batch version of
- [[pessimism-principle]] — the mechanism that converts uniform coverage into single-policy coverage
- [[coverage-coefficient]] — the quantity the whole theory is graded on
- [[importance-weighting]] — the policy route's estimator
- [[extrapolation-error]] — the greedy rule's failure mode, in RL language
- [[offline-reinforcement-learning]] — the $H>1$ generalization
- [[fqi-finite-sample-analysis]] — the horizon-$H$ analysis this specializes
- [[offline-oracle-efficient-bandits]] — a different sense of "offline"; see the terminology guard

## Current State

Active. The tabular case is settled up to constants — matching upper and lower bounds in $S$, $C^*$ and $T$ — and the linear case is settled for the policy-level rules, where the $\ell_\infty$ confidence set is adaptively minimax optimal. What is open is mostly on the value-based side and is recorded there: pointwise quantifiers for general classes, whether pessimism at the level of policies rather than contexts buys anything, and instance-dependent rates that are not driven by a worst-case coverage constant.
