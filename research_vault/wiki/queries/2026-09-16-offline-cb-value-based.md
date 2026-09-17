---
date: 2026-09-16
question: "Offline contextual bandits via the value-based (regression) approach, in the notation of Jun's CSED703Q note — what is the standard formulation, what is known, and how does it compare to the policy-based (IPW / IX / LS) route?"
tags: [contextual-bandits, offline-contextual-bandits, pessimism, learning-theory]
---

# Value-Based Offline Contextual Bandits — a self-contained account of the standard approach

> **Scope.** This page presents the *value-based* approach to offline contextual bandits as it is formulated and analyzed in the literature (Beygelzimer & Langford 2009; Dudík et al. 2011; Rashidinejad et al. 2021; Jin, Yang & Wang 2021; Xie et al. 2021; Brandfonbrener et al. 2021; Xiao et al. 2021; Li, Ma & Srebro 2022; Nguyen-Tang et al. 2022). K.-S. Jun's CSED703Q note *Offline contextual bandits* (Spring 2026; `raw/papers/Jun2026Offline.pdf` PENDING) supplies the notation and the problem definition (§1.1) and serves as the reference point for the policy-based route in §9; nothing else is taken from it. The page is self-contained: every definition, assumption, inequality and intermediate result is stated here, and tools shared by several results are collected in §3.
>
> **Terminology guard.** "Offline" here means *learning from a fixed logged dataset*. The vault's [[offline-oracle-efficient-bandits]] topic is a different use of the word — *online* algorithms that call an *offline regression oracle*. Do not conflate. The pessimism literature writes $\mathrm{SubOpt}(\hat\pi)$ for what the note calls $\mathrm{Regret}_n$; the page keeps the note's symbol.

## 0. Summary of results

1. **What a value-based method is.** Fit a reward model $\hat f$ by least-squares regression on the logged triples and act greedily on it, $\hat\pi(x)\in\arg\max_a\hat f(x,a)$; in the modern form, subtract an *uncertainty quantifier* $b(x,a)$ first, $\hat\pi(x)\in\arg\max_a\hat f(x,a)-b(x,a)$ (§2.1). This — not "estimate $\hat v(\pi)$ for every candidate policy and pick the largest" — is the representative formulation (§2.3). The estimate-then-select formulation is the policy-optimization view; with plug-in value estimates it *coincides* with greedy when the policy class is unrestricted (Proposition 7.1), and otherwise it is cost-sensitive classification with imputed rewards (§7).
2. **Two lemmas carry the analysis.** The plug-in decomposition, $\mathrm{Regret}\le\mathbb E_x[(r-\hat f)(x,\pi^*(x))+(\hat f-r)(x,\hat\pi(x))]$ (Lemma 3.13), charges the greedy policy for the error at the action *it* picks, so greedy needs *uniform* coverage of all actions (Theorem 4.2, Proposition 4.4). The pessimism lemma, $\mathrm{Regret}(\hat\pi)\le2\,\mathbb E_x[b(x,\pi^*(x))]$ (Lemma 3.14, Theorem 5.1), charges only the optimal policy's uncertainty, so pessimism needs *single-policy* coverage (Definition 3.8). Replacing uniform by single-policy coverage is what pessimism buys, and it is not a matter of minimax rates (§5.6).
3. **Rates.** Under realizability: tabular LCB $\tilde O(\sqrt{S\,C^*/n})$ with a matching lower bound (Theorem 5.2); linear $2\beta\,\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}=\tilde O(\sqrt{d/(\kappa n)})$ (Theorem 5.3); general finite $\mathcal F$ via the version space, $\tilde O(\sqrt{C_{\mathcal F}(\pi^*)\ln|\mathcal F|/n})$ with class-dependent coverage (Theorem 5.4); greedy $O(\sqrt{\ln|\mathcal F|/(\mu_{\min}n)})$ (Corollary 4.3). Action gaps give $1/n$ for the plug-in rule (Theorem 8.2).
4. **What must be assumed.** Value-based methods need realizability $r\in\mathcal F$ and never use the propensities $\mu$; coverage enters only the analysis, and through $C_{\mathcal F}$ it can be finite where importance weights are unbounded. The policy-based route (§9) needs the propensities and pays $\ln|\Pi|$ instead of the complexity of $\mathcal F$; the two meet in the doubly robust estimator (§6.3).

## 1. Problem setup

### 1.1 Problem definition (as in the note)

We consider an **offline setting**: the policy that collects data has to be fixed — such a policy is called a **behavior policy**.

Formally, a **policy** $\pi$ is a mapping from context $x$ to a distribution over the actions: $\pi:\mathcal X\to\Delta(\mathcal A)$, where $\mathcal X$ is the context space, $\mathcal A$ is the action space, and $\Delta(\mathcal A)$ is a probability distribution over actions.

This problem assumes that the data was collected as follows:

- **Input:** behavior policy $\mu$
- For $t=1,\dots,n$,
  - Observe context $x_t\sim\nu$ &nbsp;&nbsp; // $\nu$: context distribution
  - Choose action $a_t\sim\mu(x_t)$
  - Receive reward $r_t\sim\rho(x_t,a_t)$ &nbsp;&nbsp; // $\rho$: joint probability mapping from context–action pairs to rewards in $[0,1]$

After this, we are given the dataset $D_n=\{(x_t,a_t,r_t)\}_{t=1}^n$ and a policy class $\Pi$. We wish to find the policy $\pi\in\Pi$ with the maximum value using the dataset $D_n$, but without knowledge of $\nu$ and $\rho$.

Formally, we are interested in minimizing the offline regret defined below. Define the expected reward of a policy $\pi$ as
$$
v(\pi):=\mathbb E_{x\sim\nu,\ a\sim\pi(x),\ r\sim\rho(x,a)}[r],
$$
which we also call the **value** of policy $\pi$. Let $\hat\pi$ be the policy chosen by the algorithm. The **(offline) regret** is
$$
\mathrm{Regret}_n = v(\pi^*)-v(\hat\pi),\qquad \pi^*:=\arg\max_{\pi\in\Pi}v(\pi).
$$
This is a very natural target to minimize.

### 1.2 Notation, conventions and standing assumptions

| Symbol | Meaning |
|---|---|
| $\mathcal X$, $\mathcal A$, $K:=\lvert\mathcal A\rvert$ | context space; finite action set; number of actions. Tabular case: $S:=\lvert\mathcal X\rvert$ |
| $\nu$, $\mu$, $\rho$ | context distribution; behavior policy; reward distribution, supported on $[0,1]$ |
| $r(x,a):=\mathbb E_{r\sim\rho(x,a)}[r]$, $\sigma^2(x,a)$ | mean reward and reward variance ($\le1/4$) |
| $f(x,\pi):=\sum_a\pi(a\mid x)f(x,a)$ | a function's prediction for acting by $\pi$ at $x$; $=f(x,\pi(x))$ for deterministic $\pi$ |
| $v(\pi)=\mathbb E_{x\sim\nu}[r(x,\pi)]$ | value, as in §1.1 |
| $d^\pi(x,a):=\nu(x)\pi(a\mid x)$ | occupancy of $\pi$; the data distribution is $d^\mu$ |
| $\pi^*$, $v^*$ | $\pi^*(x)\in\arg\max_ar(x,a)$, deterministic; $v^*=v(\pi^*)=\mathbb E_x\max_ar(x,a)$ |
| $\mathrm{Regret}(\hat\pi):=v^*-v(\hat\pi)$ | regret (suboptimality) against the global optimum. For a policy class $\Pi$: $\mathrm{Regret}_\Pi(\hat\pi):=\max_{\pi\in\Pi}v(\pi)-v(\hat\pi)$, maximizer $\pi^*_\Pi$ |
| $D_n$; $D^{\mathrm{reg}}$, $D^{\mathrm{eval}}$ | the logged data; its first $n_1=\lceil n/2\rceil$ and last $m=n-n_1$ triples (used only where a sample split is stated) |
| $n(x,a)$ | number of logged triples with $(x_t,a_t)=(x,a)$ |
| $\mathcal F$, $N:=\lvert\mathcal F\rvert$ | reward class $\subset[0,1]^{\mathcal X\times\mathcal A}$ and its cardinality |
| $\mathcal L_D(f):=\frac1{\lvert D\rvert}\sum_{(x,a,r)\in D}(f(x,a)-r)^2$; $\mathcal L(f):=\mathbb E_{d^\mu\times\rho}[(f(x,a)-r)^2]$ | empirical and population squared loss |
| $\hat f:=\arg\min_{f\in\mathcal F}\mathcal L_D(f)$ | least-squares fit, on $D=D_n$ unless a split is stated |
| $\pi_f(x):=\arg\max_af(x,a)$; $\Pi_{\mathcal F}:=\{\pi_f:f\in\mathcal F\}$ | greedy policy of $f$; induced policy class |
| $\|g\|_{\nu\times\mu}^2:=\mathbb E_{d^\mu}[g^2]$, likewise $\|g\|_{\nu\times\pi}$; $\|g\|_{L_1(\nu\times\pi)}:=\mathbb E_{d^\pi}\lvert g\rvert$ | norms under the data and target distributions |
| $\mu_{\min}:=\inf_{x,a}\mu(a\mid x)$ | may be $0$ |
| $C^\pi$, $C^\pi_2$, $C_{\mathcal F}(\pi)$; $C^*:=C^{\pi^*}$, $\bar C^*:=C^{\pi^*}_2$ | coverage coefficients, Definition 3.8 |
| $b(x,a)$ | uncertainty quantifier for $\hat f$, Definition 3.10 |
| $\phi$, $d$, $\Lambda$, $\Sigma_\mu$, $\beta_\delta$ | linear case: features, dimension, regularized Gram matrix, population Gram matrix, confidence radius (§3.1, §3.5) |

**Standing assumptions.** (A1) *Data:* $(x_t,a_t,r_t)$ are i.i.d. as in §1.1. (A2) *Boundedness:* rewards lie in $[0,1]$, every $f\in\mathcal F$ maps into $[0,1]$, $K<\infty$. (A3) *Realizability:* $r\in\mathcal F$ — invoked where stated. (A4) *Finite class:* $N<\infty$ — invoked where stated; for infinite classes $\ln N$ becomes a covering number or pseudo-dimension and the rates are unchanged. (A5) *Comparator:* value-based methods place no restriction on the policy, so the comparator is the global optimum $\pi^*$, which may be taken deterministic because $\max_\pi\mathbb E_x[r(x,\pi)]=\mathbb E_x\max_ar(x,a)$; where a class $\Pi$ is used (§7, §9) the comparator is $\pi^*_\Pi$. All statements "with probability at least $1-\delta$" refer to the draw of $D_n$; union bounds are made explicit.

## 2. The value-based approach

### 2.1 The representative formulation: regression, then act greedily

A value-based method never estimates the value of a candidate policy. It estimates the **mean-reward function** $r$ by supervised regression on the logged triples — inputs $(x_t,a_t)$, labels $r_t$ — within a class $\mathcal F$,
$$
\hat f:=\arg\min_{f\in\mathcal F}\ \frac1n\sum_{t=1}^n\big(f(x_t,a_t)-r_t\big)^2, \tag{1}
$$
and then *derives* a policy from $\hat f$ by choosing, in each context, the action the model rates highest. Two versions exist.

- **Greedy (plug-in).** $\hat\pi(x)\in\arg\max_a\hat f(x,a)$, i.e. $\hat\pi=\pi_{\hat f}$. This is "the regression approach" of Beygelzimer & Langford (2009), the plug-in individualized-treatment rule of Murphy (2005) and Qian & Murphy (2011), and "value-based learning" in Brandfonbrener et al. (2021): "first learn the $Q$ function and then use a greedy policy with respect to this estimated $Q$ function."
- **Pessimistic (lower confidence bound).** With an *uncertainty quantifier* $b(x,a)\ge0$ such that $|\hat f(x,a)-r(x,a)|\le b(x,a)$ for all $(x,a)$ with high probability (Definition 3.10),
$$
\hat\pi(x)\in\arg\max_{a\in\mathcal A}\ \hat f(x,a)-b(x,a). \tag{LCB}
$$
This is LCB in Rashidinejad et al. (2021), pessimistic value iteration at horizon one in Jin, Yang & Wang (2021), the confidence-adjusted index rule with $\alpha=-\beta_\delta$ in Xiao et al. (2021), NeuraLCB in Nguyen-Tang et al. (2022), and, in the tabular case, the $\hat\pi_\infty$ (PUNC) rule of Li, Ma & Srebro (2022).

Both versions use the data through $\hat f$ (and $b$) alone: the propensities $\mu(a_t\mid x_t)$ are never used, the behavior policy may be deterministic, and no policy class is specified — the policy ranges over all of $\mathcal A^{\mathcal X}$, so the comparator is the global optimum.

### 2.2 Intuition

*Why regression.* The reward of the action a policy would have taken is unobserved whenever $\mu$ took another. A reward model fitted on the pairs $\mu$ did visit predicts the pairs it did not — a linear model predicts $r(x,2)$ from contexts where only $a=1$ was played — so the counterfactual is *interpolated through $\mathcal F$* rather than re-weighted through the randomness of $\mu$. The price is that the prediction is only as good as $\mathcal F$: the regression is fit where the data are, not where the derived policy acts, so a model accurate on average under $d^\mu$ may be wrong at the pairs the policy selects (Beygelzimer & Langford 2009; Dudík et al. 2011).

*Why pessimism.* The greedy policy chooses the action the model rates highest, which is systematically the action the model *over*-rates; among poorly covered actions there is always one whose estimate is high by chance. Subtracting an uncertainty quantifier makes an action attractive only if the data support its value, and the analysis then charges the learner only for uncertainty at the optimal policy's actions rather than at whatever it chose (Lemma 3.14). This turns a *uniform* coverage requirement into a *single-policy* one, which is the central theme of the offline literature (Rashidinejad et al. 2021; Jin, Yang & Wang 2021).

### 2.3 The alternative formulation, and how it relates

The other way to write an offline learner is *estimate, then select*: form an estimate $\hat v(\pi)$ of every candidate's value and output $\arg\max_{\pi\in\Pi}\hat v(\pi)$ — or, pessimistically, $\arg\max_\pi\hat v(\pi)-W_\pi$ with per-policy widths. This is the **policy-optimization** view. It is the natural home of the policy-based route (importance-weighted estimates: §2.4 and §9), but it can also be run with value-based estimates: the **direct method** plugs the reward model into every policy's value, $\hat v^{\mathrm{DM}}(\pi)=\frac1n\sum_t\hat f(x_t,\pi)$ (§6), and the doubly robust estimator combines the two (§6.3). Its relation to §2.1 is exact:

- **Unrestricted class.** If $\Pi=\mathcal A^{\mathcal X}$, the maximizer of $\hat v^{\mathrm{DM}}$ over $\Pi$ is the greedy policy $\pi_{\hat f}$ on the logged contexts, and the maximizer of the pessimistic plug-in value $\frac1n\sum_t[\hat f-b](x_t,\pi)$ is (LCB): the maximization decouples across contexts (Proposition 7.1). Estimate-then-select with plug-in values *is* regression-then-greedy.
- **Restricted class.** If $\Pi\subsetneq\mathcal A^{\mathcal X}$, maximizing $\hat v^{\mathrm{DM}}$ over $\Pi$ is cost-sensitive classification with the imputed reward vector $\hat f(x_t,\cdot)$ — the "direct method for policy optimization" of Dudík et al. (2011), the Offset Tree's regression baseline, and, with doubly robust scores in place of $\hat f$, the policy learning of Athey & Wager (2021) and Zhou, Athey & Wager (2023). Its guarantee is a uniform-deviation bound over $\Pi$ (Theorem 7.2), and it is what one uses when the deployed policy must lie in a given class.
- **Policy-level pessimism.** Pessimism can also be applied to policy values rather than to action values: choose $\arg\max_{\pi}\min_{f\in\mathcal F_\varepsilon}\hat v_f(\pi)$ over a version space $\mathcal F_\varepsilon$ of reward models consistent with the data. This is Bellman-consistent pessimism at horizon one (Xie et al. 2021; Theorem 5.4) and the $\hat\pi_2$ rule of Li, Ma & Srebro (2022). Their framework — a confidence set $\Theta\ni\theta^*$ induces the pessimistic value $\hat V(\pi):=\inf_{\theta\in\Theta}\mathbb E_x[\phi(x,\pi(x))^\top\theta]$, then $\hat\pi:=\arg\max_\pi\hat V(\pi)$ — is exactly estimate-then-select with pessimistic plug-in values, and it contains both forms: the $\ell_2$ set gives $\hat\pi_2=$ BCP, the $\ell_\infty$ set gives $\hat\pi_\infty$ (PUNC), which reduces to tabular LCB and is adaptively minimax optimal, strictly dominating $\hat\pi_2$ (§5.4). It is the right tool when no pointwise uncertainty quantifier is available (general $\mathcal F$).

So the literature's value-based methods are organized around (1)+(LCB); the estimate-then-select formulation is either the same object written differently (unrestricted class), the class-restricted variant, or the policy-level form of pessimism. The page follows that organization: §3 tools, §4 greedy, §5 pessimistic, §6 evaluation with the reward model, §7 the class-restricted and policy-level variants, §8 fast rates, §9 comparison with the policy-based route.

### 2.4 The policy-based route, for reference

In the note's notation the policy-based route estimates $v(\pi)$ for $\pi\in\Pi$ by importance weighting, $\hat v^{\mathrm{IPW}}(\pi):=\frac1n\sum_t\frac{\pi(a_t\mid x_t)}{\mu(a_t\mid x_t)}r_t$, unbiased by the change of measure $\mathbb E_{a\sim\mu(x)}[\frac{\pi(a|x)}{\mu(a|x)}r(x,a)]=\mathbb E_{a\sim\pi(x)}[r(x,a)]$, and selects by MaxIPW ($\arg\max_\pi\hat v^{\mathrm{IPW}}$) or PES ($\arg\max_\pi\hat v^{\mathrm{IPW}}(\pi)-W^U_\pi$); implicit exploration (IX, weights $\pi/(\mu+\gamma)$, bias $\gamma C_\gamma(\pi)$ with $C_\gamma(\pi):=\mathbb E_x\sum_a\frac{\pi(a|x)r(x,a)}{\mu(a|x)+\gamma}$) and logarithmic smoothing (LS) tame unbounded weights. It needs an explicit $\Pi$, the propensities, a stochastic $\mu$ with $\mu(a\mid x)>0$ wherever $\pi$ puts mass, and nothing about $\rho$ beyond $r\in[0,1]$. §9 compares the two routes; the only fact about this route used before §9 is the Hoeffding width $|\hat v^{\mathrm{IPW}}(\pi)-v(\pi)|\le\frac1{\mu_{\min}}\sqrt{\ln(2|\Pi|/\delta)/(2n)}$, valid for all $\pi\in\Pi$ simultaneously with probability at least $1-\delta$ (Theorem 3.1).

## 3. Technical toolkit

Everything in §4–§8 is assembled from this section: concentration inequalities (§3.1); the least-squares guarantee under realizability (§3.2); coverage coefficients and the change of measure that carries an error under the data distribution to an error under a target policy (§3.3); uncertainty quantifiers for $\hat f$ (§3.4); and the two suboptimality lemmas on which every regret bound rests (§3.5).

### 3.1 Concentration inequalities

**Theorem 3.1 (Hoeffding).** Let $Z_1,\dots,Z_n$ be i.i.d. with values in $[a,b]$. With probability at least $1-\delta$, $\big|\frac1n\sum_iZ_i-\mathbb EZ\big|\le(b-a)\sqrt{\ln(2/\delta)/(2n)}$. For $M$ such sample means simultaneously, replace $\delta$ by $\delta/M$ (union bound).

**Theorem 3.2 (Bernstein, one-sided).** Let $Y_1,\dots,Y_n$ be i.i.d. with $\mathrm{Var}(Y)\le s^2$ and $|Y-\mathbb EY|\le b$ almost surely. With probability at least $1-\delta$, $\ \mathbb EY-\frac1n\sum_iY_i\le\sqrt{2s^2\ln(1/\delta)/n}+2b\ln(1/\delta)/(3n)$.

**Theorem 3.3 (self-normalized bound; Abbasi-Yadkori, Pál & Szepesvári 2011, Thm 2).** Let $\phi_t:=\phi(x_t,a_t)\in\mathbb R^d$ with $\|\phi_t\|_2\le1$, and $r_t=\langle\theta^*,\phi_t\rangle+\eta_t$ with $\mathbb E[\eta_t\mid x_t,a_t]=0$ and $\eta_t$ supported on an interval of length $1$ (hence $\tfrac12$-sub-Gaussian), $\|\theta^*\|_2\le B$. Let $\Lambda:=\lambda I+\sum_t\phi_t\phi_t^\top$ and $\hat\theta:=\Lambda^{-1}\sum_t\phi_tr_t$. With probability at least $1-\delta$,
$$
\|\hat\theta-\theta^*\|_{\Lambda}\ \le\ \beta_\delta:=\tfrac12\sqrt{2\ln(1/\delta)+d\ln\big(1+\tfrac{n}{\lambda d}\big)}+\sqrt\lambda\,B .
$$

### 3.2 Least-squares regression under realizability

**Lemma 3.4 (Pythagorean identity).** For any $f:\mathcal X\times\mathcal A\to[0,1]$, $\ \mathcal L(f)-\mathcal L(r)=\|f-r\|^2_{\nu\times\mu}$.
*Proof.* $\mathcal L(f)=\mathbb E[(f-r+r-r_t)^2]=\|f-r\|^2_{\nu\times\mu}+\mathcal L(r)+2\,\mathbb E[(f-r)(x,a)(r(x,a)-r_t)]$, and the cross term vanishes because $\mathbb E[r(x,a)-r_t\mid x,a]=0$. $\square$

Minimizing the population squared loss over $\mathcal F$ is therefore projecting $r$ onto $\mathcal F$ in $L_2(d^\mu)$, and the excess loss is exactly the squared $L_2$ error.

**Lemma 3.5 (excess-loss variables).** For $f\in\mathcal F$ and one triple $(x,a,r_t)$ let $Y(f):=(f(x,a)-r_t)^2-(r(x,a)-r_t)^2$. Then $\mathbb E[Y(f)]=\|f-r\|^2_{\nu\times\mu}$, $Y(f)\in[-2,2]$, $|Y(f)-\mathbb EY(f)|\le3$, and $\mathrm{Var}(Y(f))\le4\,\mathbb E[Y(f)]$.
*Proof.* $Y(f)=(f-r)(f+r-2r_t)$ with $|f-r|\le1$ and $|f+r-2r_t|\le2$; the mean is Lemma 3.4; $\mathbb E[Y^2]\le4\,\mathbb E[(f-r)^2]$; and $\mathbb EY\in[0,1]$ gives the range of $Y-\mathbb EY$. $\square$

The variance bound is a *Bernstein condition* — the fluctuation of the excess loss shrinks with the excess loss itself — and it is what yields a $1/n$ rather than $1/\sqrt n$ rate for the squared error.

**Theorem 3.6 (realizable least squares, finite class).** Under (A1)–(A4), let $\hat f$ minimize $\mathcal L_D$ over $\mathcal F$ on $n$ i.i.d. triples and set $L:=\ln(N/\delta)/n$. With probability at least $1-\delta$, simultaneously for all $f\in\mathcal F$,
$$
\mathbb E[Y(f)]-\frac1n\sum_{i=1}^nY_i(f)\ \le\ \sqrt{8\,\mathbb E[Y(f)]\,L}+2L, \tag{2}
$$
and consequently
$$
\|\hat f-r\|^2_{\nu\times\mu}\ \le\ 12\,L\ =\ \frac{12\ln(N/\delta)}{n}. \tag{3}
$$
*Proof.* (2) is Theorem 3.2 with $s^2=4\,\mathbb E[Y(f)]$ and $b=3$ (Lemma 3.5) at confidence $\delta/N$, union-bounded over $\mathcal F$. Because $r\in\mathcal F$ and $\hat f$ minimizes the empirical loss, $\frac1n\sum_iY_i(\hat f)=\mathcal L_D(\hat f)-\mathcal L_D(r)\le0$. Applying (2) at $f=\hat f$ and writing $x:=\|\hat f-r\|^2_{\nu\times\mu}$: $x\le\sqrt{8xL}+2L\le x/2+4L+2L$, i.e. $x\le12L$. $\square$

**Corollary 3.7 (version space).** On the event of Theorem 3.6, with $\varepsilon:=4L$, the set $\mathcal F_\varepsilon:=\{f\in\mathcal F:\mathcal L_D(f)\le\mathcal L_D(\hat f)+\varepsilon\}$ satisfies (i) $r\in\mathcal F_\varepsilon$ and (ii) $\|f-r\|^2_{\nu\times\mu}\le20L$ for every $f\in\mathcal F_\varepsilon$.
*Proof.* (i) $\mathcal L_D(r)-\mathcal L_D(\hat f)=-\frac1n\sum_iY_i(\hat f)\le(\sqrt{8xL}-x)+2L\le2L+2L$, since $\max_{x\ge0}(\sqrt{8xL}-x)=2L$. (ii) For $f\in\mathcal F_\varepsilon$, $\frac1n\sum_iY_i(f)=\mathcal L_D(f)-\mathcal L_D(r)\le\mathcal L_D(\hat f)+\varepsilon-\mathcal L_D(r)\le\varepsilon$; by (2), $x_f:=\mathbb E[Y(f)]\le\varepsilon+2L+\sqrt{8x_fL}\le6L+x_f/2+4L$, so $x_f\le20L$. $\square$

Theorem 3.6 controls an $L_2(d^\mu)$ norm, never a supremum: it says nothing about $|\hat f(x,a)-r(x,a)|$ at an individual pair. Pointwise control needs structure (§3.4); without it, pessimism is implemented at the level of policies through the version space (Theorem 5.4).

### 3.3 Coverage and the change of measure

**Definition 3.8 (coverage coefficients).** For a policy $\pi$,
$$
C^\pi:=\sup_{x,a}\frac{d^\pi(x,a)}{d^\mu(x,a)}=\sup_{x,a:\ \pi(a|x)>0}\frac{\pi(a\mid x)}{\mu(a\mid x)},\qquad
C^\pi_2:=\mathbb E_{d^\mu}\Big[\Big(\frac{\pi}{\mu}\Big)^2\Big]=\mathbb E_{x\sim\nu}\sum_a\frac{\pi(a\mid x)^2}{\mu(a\mid x)},\qquad
C_{\mathcal F}(\pi):=\sup_{f\in\mathcal F,\ f\ne r}\frac{\|f-r\|^2_{\nu\times\pi}}{\|f-r\|^2_{\nu\times\mu}},
$$
with $c/0:=\infty$ for $c>0$. $C^\pi$ is the **single-policy concentrability** coefficient (Rashidinejad et al. 2021, Definition 1), $C^*:=C^{\pi^*}$; the **uniform** coefficient is $C_{\mathrm{unif}}:=\sup_\pi C^\pi=1/\mu_{\min}$. $C^\pi_2$ is the second moment of the importance weights, and $\bar C^*:=C^{\pi^*}_2$. $C_{\mathcal F}(\pi)$ is the class-dependent coefficient of Xie et al. (2021, Definition 1) at horizon one, where the Bellman residual $f-\mathcal T^\pi f$ is $f-r$. For deterministic $\pi$: $C^\pi=\sup_x1/\mu(\pi(x)\mid x)$ and $C^\pi_2=\mathbb E_x[1/\mu(\pi(x)\mid x)]$. Always $C^\pi_2\le C^\pi\le C_{\mathrm{unif}}$ and $C_{\mathcal F}(\pi)\le C^\pi$; $C^\pi_2$ and $C_{\mathcal F}(\pi)$ are not ordered in general.

**Lemma 3.9 (change of measure).** (a) For any $g:\mathcal X\times\mathcal A\to\mathbb R$ and any $\pi$, $\ \big|\mathbb E_{d^\pi}[g]\big|\le\|g\|_{L_1(\nu\times\pi)}\le\sqrt{C^\pi_2}\,\|g\|_{\nu\times\mu}$.
(b) If $g=f-r$ with $f\in\mathcal F$, $\ \|g\|_{L_1(\nu\times\pi)}\le\|g\|_{\nu\times\pi}\le\sqrt{C_{\mathcal F}(\pi)}\,\|g\|_{\nu\times\mu}\le\sqrt{C^\pi}\,\|g\|_{\nu\times\mu}$.
(c) If $\mathcal F=\{\langle\theta,\phi\rangle\}$ is linear and realizable, then with $\Sigma_\pi:=\mathbb E_{d^\pi}[\phi\phi^\top]$ and $\Sigma_\mu:=\mathbb E_{d^\mu}[\phi\phi^\top]$ (pseudo-inverse if singular, all $\phi$ in its range), $\ C_{\mathcal F}(\pi)\le\lambda_{\max}\big(\Sigma_\mu^{-1/2}\Sigma_\pi\Sigma_\mu^{-1/2}\big)\le\mathbb E_{d^\pi}\|\phi\|^2_{\Sigma_\mu^{-1}}$.
*Proof.* (a) $\mathbb E_x\sum_a\pi(a|x)|g|=\mathbb E_x\sum_a\mu(a|x)\frac{\pi(a|x)}{\mu(a|x)}|g|\le\big(\mathbb E_{d^\mu}[(\pi/\mu)^2]\big)^{1/2}\|g\|_{\nu\times\mu}$ (Cauchy–Schwarz). (b) Jensen, the definition of $C_{\mathcal F}$, and $\|g\|^2_{\nu\times\pi}=\mathbb E_{d^\mu}[(\pi/\mu)g^2]$. (c) $f-r=\langle\theta-\theta^*,\phi\rangle$, so the ratio is $\frac{u^\top\Sigma_\pi u}{u^\top\Sigma_\mu u}\le\lambda_{\max}(\cdot)\le\mathrm{tr}(\Sigma_\mu^{-1}\Sigma_\pi)$, which is the stated expectation. $\square$

Part (c) is the mechanism behind "extrapolation": class-dependent coverage is measured in feature space and can be finite when $\mu(a\mid x)=0$ for some of $\pi$'s actions, where $C^\pi=C^\pi_2=\infty$ and every importance-weighted estimator is undefined.

### 3.4 Uncertainty quantifiers

**Definition 3.10 ($\delta$-uncertainty quantifier).** Given $\hat f$, a function $b:\mathcal X\times\mathcal A\to[0,\infty)$ is a $\delta$-uncertainty quantifier if $\ \mathbb P\big(\forall(x,a):\ |\hat f(x,a)-r(x,a)|\le b(x,a)\big)\ge1-\delta$. (This is the $\xi$-uncertainty quantifier of Jin, Yang & Wang (2021) at horizon one and the penalty function of Rashidinejad et al. (2021).)

**Proposition 3.11 (tabular quantifiers).** Let $S,K<\infty$, $\mathcal F=[0,1]^{\mathcal X\times\mathcal A}$, and let $\hat f(x,a)$ be the empirical mean of the rewards logged at $(x,a)$ (any value in $[0,1]$ if $n(x,a)=0$); this is the least-squares fit (1). Conditional on the design $\{(x_t,a_t)\}_{t\le n}$, each of
$$
b^{\mathrm H}(x,a):=\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,n(x,a)}}\Big\},\qquad
b^{\mathrm B}(x,a):=\min\Big\{1,\sqrt{\tfrac{2\sigma^2(x,a)\ln(2SK/\delta)}{n(x,a)}}+\tfrac{2\ln(2SK/\delta)}{3\,n(x,a)}\Big\}
$$
is a $\delta$-uncertainty quantifier.
*Proof.* Given the design, the rewards in cell $(x,a)$ are $n(x,a)$ i.i.d. draws from $\rho(x,a)$. $b^{\mathrm H}$: Theorem 3.1 at confidence $\delta/(SK)$, union bound over cells. $b^{\mathrm B}$: Theorem 3.2 on both sides with $s^2=\sigma^2(x,a)$, $b=1$, at confidence $\delta/(2SK)$, union bound over cells and sides. Empty cells use $|\hat f-r|\le1$. $\square$

**Proposition 3.12 (linear quantifier).** Let $\mathcal F=\{\langle\theta,\phi(\cdot,\cdot)\rangle:\|\theta\|_2\le B\}$ with $\|\phi\|_2\le1$, realizable with $r=\langle\theta^*,\phi\rangle$, and let $\hat f:=\langle\hat\theta,\phi\rangle$ with $\hat\theta$ the ridge estimate of Theorem 3.3. Then $b(x,a):=\beta_\delta\|\phi(x,a)\|_{\Lambda^{-1}}$ is a $\delta$-uncertainty quantifier.
*Proof.* On the event of Theorem 3.3, $|\hat f(x,a)-r(x,a)|=|\langle\hat\theta-\theta^*,\phi(x,a)\rangle|\le\|\hat\theta-\theta^*\|_\Lambda\|\phi(x,a)\|_{\Lambda^{-1}}\le\beta_\delta\|\phi(x,a)\|_{\Lambda^{-1}}$. $\square$

Here $\beta_\delta=\tilde O(\sqrt d)$, and if $\Lambda\succeq\kappa nI$ then $b\le\beta_\delta/\sqrt{\kappa n}$ uniformly. For a general finite $\mathcal F$ no pointwise quantifier follows from Theorem 3.6; Theorem 5.4 works with the version space instead.

### 3.5 The two suboptimality lemmas

Both lemmas are pointwise in $x$; regret bounds follow by averaging over $x\sim\nu$. Write $g:=f-r$.

**Lemma 3.13 (plug-in decomposition).** For any $f:\mathcal X\times\mathcal A\to[0,1]$ and every $x$, with $a^*=\pi^*(x)$ and $\hat a=\pi_f(x)$,
$$
r(x,a^*)-r(x,\hat a)\ \le\ (r-f)(x,a^*)+(f-r)(x,\hat a)\ \le\ |g(x,a^*)|+|g(x,\hat a)|\ \le\ 2\max_a|g(x,a)| .
$$
*Proof.* $r(x,a^*)-r(x,\hat a)=(r-f)(x,a^*)+\big(f(x,a^*)-f(x,\hat a)\big)+(f-r)(x,\hat a)$, and the middle term is $\le0$ by the definition of $\pi_f$. $\square$

**Lemma 3.14 (pessimism).** Let $b$ satisfy $|\hat f(x,a)-r(x,a)|\le b(x,a)$ for all $(x,a)$ and let $\hat\pi$ be the rule (LCB). Then for every $x$,
$$
r(x,\pi^*(x))-r(x,\hat\pi(x))\ \le\ 2\,b(x,\pi^*(x)) .
$$
*Proof.* $r(x,\hat\pi(x))\ge\hat f(x,\hat\pi(x))-b(x,\hat\pi(x))\ge\hat f(x,\pi^*(x))-b(x,\pi^*(x))\ge r(x,\pi^*(x))-2b(x,\pi^*(x))$, using the quantifier, the definition of (LCB), and the quantifier again. $\square$

The difference between the two is the whole story. The plug-in rule is charged for the error at the action *it* chose, which is random and is systematically the over-estimated one; the pessimistic rule is charged only at the action the *optimal* policy chooses. Lemma 3.14 is Theorem 4.2 of Jin, Yang & Wang (2021) at horizon one and the core of the LCB analysis in Rashidinejad et al. (2021). (Read with policies in place of actions and $\hat v(\pi)$ in place of $\hat f(x,a)$, the same two inequalities give the MaxIPW and PES guarantees of the policy-based route; §9.)

## 4. The greedy policy

### 4.1 Method

Fit $\hat f$ by (1) on all $n$ triples and output $\hat\pi=\pi_{\hat f}$. No propensities, no sample split, no policy class.

### 4.2 Analysis

**Theorem 4.1 (greedy, pointwise form).** Let $b$ be a $\delta$-uncertainty quantifier for $\hat f$. With probability at least $1-\delta$,
$$
\mathrm{Regret}(\pi_{\hat f})\ \le\ \mathbb E_x\big[b(x,\pi^*(x))\big]+\mathbb E_x\big[b(x,\pi_{\hat f}(x))\big]\ \le\ \mathbb E_x\big[b(x,\pi^*(x))\big]+\mathbb E_x\Big[\max_ab(x,a)\Big].
$$
*Proof.* Lemma 3.13 with $|g|\le b$, averaged over $x\sim\nu$. $\square$

**Theorem 4.2 (greedy, $L_2$ form; uniform coverage).** For any $f:\mathcal X\times\mathcal A\to[0,1]$,
$$
\mathrm{Regret}(\pi_f)\ \le\ \Big(\sqrt{C^{\pi^*}_2}+\sqrt{C^{\pi_f}_2}\Big)\,\|f-r\|_{\nu\times\mu}\ \le\ 2\sqrt{C_{\mathrm{unif}}}\;\|f-r\|_{\nu\times\mu}.
$$
*Proof.* Average the middle expression of Lemma 3.13 over $x$ and apply Lemma 3.9(a) to each term, with $\pi=\pi^*$ and $\pi=\pi_f$ (both deterministic); then $C^{\pi_f}_2\le C^{\pi_f}\le C_{\mathrm{unif}}$. $\square$

**What the two theorems say.** Theorem 4.1 says the greedy rule can be wrong at a context only when the model error there is at least as large as the true gap it has to resolve, so its loss is controlled by the accuracy of $\hat f$ at two actions only: the one $\pi^*$ takes, and the one the rule itself takes. The first term is unavoidable — up to a factor of two it is the *only* term the pessimistic rule pays (Lemma 3.14) — so the second term is the entire difference between the two approaches. That term is the price of letting the data choose the action: the rule commits to whichever action the regression rates highest, so accuracy at the remaining actions does not help it, and relaxing it to $\mathbb E_x[\max_ab(x,a)]$ is the most that can be said in general, because the rule has to be accurate wherever it might land. Theorem 4.2 answers a different question. Theorem 3.6 controls an average squared error under $d^\mu$ and nothing more, and Theorem 4.2 converts that average into a statement about the decision loss. The conversion is the Cauchy–Schwarz step of Lemma 3.9(a), which is why the coverage coefficient enters under a square root, and why a squared error of order $1/n$ can only become a regret of order $1/\sqrt n$. The two results are therefore two currencies for the same fact: Theorem 4.1 is the sharper of them but needs pointwise accuracy, which Theorem 3.6 does not deliver without further structure (§3.4), while Theorem 4.2 needs only what Theorem 3.6 does deliver, and pays for it with a coefficient that ranges over all actions rather than the optimal one.

By Lemma 3.4, $\|f-r\|^2_{\nu\times\mu}=\mathcal L(f)-\mathcal L(r)$, so Theorem 4.2 reads $\mathrm{Regret}(\pi_f)\le2\mu_{\min}^{-1/2}[\mathcal L(f)-\mathcal L(r)]^{1/2}$: this is Murphy's (2005) generalization-error bound for the plug-in rule, restated as (3.1) in Qian & Murphy (2011), and, for $K$ actions logged uniformly, Theorem 6.1 of Beygelzimer & Langford (2009), $\mathrm{reg}(\pi_f)\le2\sqrt{K\,\mathrm{reg}_r(f)}$, which they show is tight. The proof shows where the uniform coverage comes from: the $\pi^*$ term costs $\sqrt{\bar C^*}$, the coverage of the *optimal* policy, while the chosen-action term costs the coverage of the *learned* policy, which can only be bounded by the least-covered action.

**Provenance.** Theorem 4.2 is not new. Its content is Murphy's (2005) generalization-error bound and Theorem 6.1 of Beygelzimer & Langford (2009), both verified in §10.1; what is written here is that bound restated in terms of the second-moment coefficients of Definition 3.8 — the change of measure standard in the offline RL literature — in place of the original $\mu_{\min}^{-1/2}$ constant. Theorem 4.1 is an assembly rather than a quotation. Both of its ingredients are classical: Lemma 3.13, whose sup-norm version is the horizon-one case of the classical bound on the loss of a policy greedy with respect to an approximate value function (Singh & Yee 1994 — *from memory, not verified against the PDF; §10.3*), and the $\xi$-uncertainty quantifier of Jin, Yang & Wang (2021), whose Theorem 4.2 is its pessimistic counterpart, recorded here as Lemma 3.14. The greedy statement in this quantifier form does not appear to be a named result in any single paper, and the proofs above are the standard two-line arguments, written out rather than cited. The same holds for Theorem 3.6, and hence for Corollary 4.3.

**Corollary 4.3 (rate under realizability).** Under (A3)–(A4), with probability at least $1-\delta$, $\ \mathrm{Regret}(\pi_{\hat f})\le2\sqrt{12\,C_{\mathrm{unif}}\ln(N/\delta)/n}$.
*Proof.* Theorem 4.2 with (3). $\square$

### 4.3 Why uniform coverage cannot be dropped

**Proposition 4.4 (two actions).** Let $\mathcal X$ be a single context and $\mathcal A=\{1,2\}$. Action 1 has the deterministic reward $1/2$; action 2 has a Bernoulli reward with mean $1/2-\Delta$, $\Delta\in(0,1/4]$; the behavior policy logs action 2 with a small probability $p$, so that $n(2)\approx pn=:k$. Then: (i) $b(1)=0$ is a valid quantifier, action 1 being noiseless, while for action 2 the Hoeffding quantifier $b^{\mathrm H}$ of Proposition 3.11 is $b(2)=\min\{1,\sqrt{\ln(4/\delta)/(2k)}\}$; (ii) if $\Delta\le1/(2\sqrt k)$, the event $\hat f(2)>1/2$ has probability at least an absolute constant $c_0>0$ (binomial anti-concentration), on which the greedy policy chooses action 2 and incurs regret $\Delta$; with $\Delta=1/(2\sqrt k)$, $\ \mathbb E[\mathrm{Regret}(\pi_{\hat f})]\ge c_0/(2\sqrt k)\asymp b(2)$; (iii) on the event of Proposition 3.11, which has probability at least $1-\delta$, the rule (LCB) chooses action 1 and incurs no regret, because $\hat f(2)-b(2)\le r(2)<1/2=\hat f(1)-b(1)$.

Here $C^*=(1-p)^{-1}\approx1$ — the optimal action is covered as well as it could be — and the greedy rule still fails at the rate of the *un*covered action's uncertainty: the term $\mathbb E_x[\max_ab(x,a)]$ in Theorem 4.1 is not an artifact of the proof. Rashidinejad et al. (2021, Proposition 1) make the same point for the empirical best arm: it fails even when $C^*\approx1$.

### 4.4 Discussion and limitations

The greedy policy is governed by the least-covered action: Lemma 3.13 charges the error at whatever it *chooses*, and it chooses whatever the regression happened to overvalue. With tabular $\mathcal F$ and $n(x,a)=0$ its choice at $x$ is arbitrary; more data does not help unless coverage improves. This is the one-step form of the extrapolation error of fitted Q-iteration and the reason the offline literature does not use the plain plug-in rule without either uniform coverage (Theorem 4.2), a margin condition (§8.2), or pessimism (§5).

## 5. The pessimistic plug-in policy

### 5.1 Method

Fit $\hat f$ by (1) on all $n$ triples, take a $\delta$-uncertainty quantifier $b$ for it (Proposition 3.11 or 3.12), and output the rule (LCB), $\hat\pi(x)\in\arg\max_a\hat f(x,a)-b(x,a)$. Computable quantifiers are where the structure of $\mathcal F$ enters; propensities are still not used.

### 5.2 The main theorem

**Theorem 5.1 (pessimism: single-policy coverage suffices).** Let $b$ be a $\delta$-uncertainty quantifier for $\hat f$ and $\hat\pi$ the rule (LCB). With probability at least $1-\delta$,
$$
\boxed{\ \mathrm{Regret}(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\big[b(x,\pi^*(x))\big]\ } \tag{4}
$$
*Proof.* Lemma 3.14 averaged over $x\sim\nu$. $\square$

Only the optimal policy's uncertainty appears: a poorly covered action costs nothing unless $\pi^*$ uses it. This is the horizon-one case of Theorem 4.2 of Jin, Yang & Wang (2021), $\mathrm{SubOpt}(\hat\pi)\le2\sum_h\mathbb E_{\pi^*}[\Gamma_h]$. Explicit rates follow by inserting the quantifiers of §3.4.

### 5.3 Tabular classes

**Theorem 5.2 (tabular LCB).** Let $S,K<\infty$, $\hat f$ the cell-wise empirical means, and $\hat\pi$ the rule (LCB) with $b^{\mathrm H}$. Conditional on the design, with probability at least $1-\delta$,
$$
\mathrm{Regret}(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\Big[\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,n(x,\pi^*(x))}}\Big\}\Big].
$$
Moreover, writing $\mu_x:=\mu(\pi^*(x)\mid x)$ and using $n(x,a)\sim\mathrm{Bin}(n,\nu(x)\mu(a\mid x))$ with the multiplicative Chernoff bound $\mathbb P(\mathrm{Bin}(n,q)\le nq/2)\le e^{-nq/8}$: if $n\,\nu(x)\mu_x\ge8\ln(S/\delta')$ for all $x$, then with probability at least $1-\delta-\delta'$,
$$
\mathrm{Regret}(\hat\pi)\ \le\ 2\sum_x\nu(x)\sqrt{\frac{\ln(2SK/\delta)}{n\,\nu(x)\mu_x}}\ \le\ 2\sqrt{\frac{S\,\bar C^*\ln(2SK/\delta)}{n}}\ \le\ 2\sqrt{\frac{S\,C^*\ln(2SK/\delta)}{n}},
$$
using $\sum_x\sqrt{\nu(x)/\mu_x}\le\sqrt{S\sum_x\nu(x)/\mu_x}=\sqrt{S\bar C^*}$ (Cauchy–Schwarz) and $\bar C^*\le C^*$.
*Proof.* Theorem 5.1 with Proposition 3.11; the count condition and union bound over $\mathcal X$ give $n(x,\pi^*(x))\ge n\nu(x)\mu_x/2$ for all $x$. $\square$

Rashidinejad et al. (2021) remove the count condition and prove the sharper $\tilde O\big(\sqrt{S(C^*-1)/n}+S/n\big)$ (their Theorem 4, with $C^*=\max_x1/\mu_x$), matched by an information-theoretic lower bound of the same form (their Theorem 5): when $C^*=1$ the data are "expert" and the rate is $1/n$, and for $C^*>1$ it is $\sqrt{SC^*/n}$ up to the $-1$. Their LCB is (LCB) with $b\asymp\sqrt{\ln(S/\delta)/n(x,a)}$.

### 5.4 Linear classes

**Theorem 5.3 (linear LCB).** Let $\mathcal F$ be the realizable linear class of Proposition 3.12 and $\hat\pi$ the rule (LCB) with its quantifier. With probability at least $1-\delta$,
$$
\mathrm{Regret}(\hat\pi)\ \le\ 2\beta_\delta\,\mathbb E_{x\sim\nu}\big\|\phi(x,\pi^*(x))\big\|_{\Lambda^{-1}},\qquad\text{and}\qquad\mathrm{Regret}(\hat\pi)\le\frac{2\beta_\delta}{\sqrt{\kappa n}}=\tilde O\Big(\sqrt{\tfrac d{\kappa n}}\Big)\ \text{ if }\Lambda\succeq\kappa nI.
$$
*Proof.* Theorem 5.1 with Proposition 3.12. $\square$

This is the pessimistic value iteration of Jin, Yang & Wang (2021) at horizon one, whose radius $\beta=c\,dH\sqrt{\zeta}$ carries factors of $d$ and $H$ from covering the next-state value class that are absent for bandits; they show the $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ form is minimax optimal for their instance class. Li, Ma & Srebro (2022) organize the linear case differently, at the level of policies (fixed design, OLS estimate $\hat\theta$, whitening matrix $\Sigma_D:=\frac1n\Phi^\top\Phi$): a confidence set $\Theta\ni\theta^*$ induces the pessimistic value $\hat V(\pi):=\inf_{\theta\in\Theta}\mathbb E_x[\phi(x,\pi(x))^\top\theta]$ and the rule $\hat\pi_\Theta:=\arg\max_\pi\hat V(\pi)$; if $\theta^*\in\Theta$ and $\sup_{\theta\in\Theta}\|\theta-\theta^*\|\le\beta$ with probability $1-\delta$ ("pessimism-validity"), then $\mathrm{Regret}(\hat\pi_\Theta)\le\beta\,\|\mathbb E_x\phi(x,\pi^*(x))\|_*$ in the dual norm (their Proposition 1 — the decomposition (4)–(5) of §7.1 with a policy-level width). With the $\ell_p$ sets $\Theta_p:=\{\theta:\|\Sigma_D^{1/2}(\theta-\hat\theta)\|_p\le\beta\}$, $\beta=d^{1/p}\sqrt{8\ln(d/\delta)/n}$, their Theorem 1 gives $\mathrm{Regret}(\hat\pi_p)\le d^{1/p}\sqrt{8\ln(d/\delta)/n}\,\big\|\Sigma_D^{-1/2}\mathbb E_x\phi(x,\pi^*(x))\big\|_q$, $1/p+1/q=1$. Three consequences for this page. (i) $\hat\pi_2$ *is* the version-space rule of §5.5: in a linear bandit the empirical Bellman error is $\|\Sigma_D^{1/2}(\theta-\hat\theta)\|_2^2$, so BCP's version space is an $\ell_2$ ball. (ii) $\hat\pi_\infty$ (PUNC) reduces to the tabular LCB of Theorem 5.2 when $\phi(x,a)=e_{xa}$ (their Corollary 1 recovers $\sqrt{SC^*\ln(SK/\delta)/n}$), but it is *not* the pointwise linear rule of Theorem 5.3: that rule (PEVI) subtracts $\beta\|\Sigma_D^{-1/2}\phi\|_2$ at every context, which amounts to an $\ell_2$ set enlarged context by context; its guarantee $\sqrt{d^2/n}\,\mathbb E_x\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_2$ is looser by a factor $d$ and by Jensen ($\|\Sigma_D^{-1/2}\mathbb E_x\phi\|_2\le\mathbb E_x\|\Sigma_D^{-1/2}\phi\|_2$), but holds for every test distribution at once, whereas the policy-level rules are tuned to one $\nu$. (iii) Their Theorem 2 is a minimax lower bound $\Omega(d^{1/p}\Lambda/\sqrt n)$ over the classes $\mathrm{CB}_q(\Lambda)=\{\|\Sigma_D^{-1/2}\mathbb E_x\phi(x,\pi^*(x))\|_q\le\Lambda\}$, so $\hat\pi_p$ is minimax over $\mathrm{CB}_q$, and since $\|v\|_1\le d^{1-1/q}\|v\|_q$ the $\ell_\infty$ rule is optimal over *every* class simultaneously — "adaptively minimax optimal" — and strictly dominates $\hat\pi_2$. The coverage quantity is thus the norm of the *averaged* whitened feature of $\pi^*$, a refinement of $C^*$ and of $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ that can be much smaller than either.

### 5.5 General finite classes: pessimism through the version space

Theorem 3.6 provides no pointwise quantifier, so pessimism is applied to policy values. **Method:** split the data; on $D^{\mathrm{reg}}$ fit $\hat f$ and form the version space $\mathcal F_\varepsilon$ of Corollary 3.7 with $\varepsilon=4L_1$, $L_1:=\ln(N/\delta)/n_1$; on $D^{\mathrm{eval}}$ define empirical values $\hat v_f(\pi):=\frac1m\sum_{x\in D^{\mathrm{eval}}}f(x,\pi(x))$ for $f\in\mathcal F$ and $\pi\in\Pi_{\mathcal F}$ (at most $N$ policies, containing $\pi^*=\pi_r$ under realizability); output
$$
\hat\pi:=\arg\max_{\pi\in\Pi_{\mathcal F}}\ \min_{f\in\mathcal F_\varepsilon}\hat v_f(\pi). \tag{VS}
$$

**Theorem 5.4 (version-space pessimism).** Under (A3)–(A4), with probability at least $1-2\delta$,
$$
\mathrm{Regret}(\hat\pi)\ \le\ \sqrt{\frac{20\,C_{\mathcal F}(\pi^*)\ln(N/\delta)}{n_1}}+2\sqrt{\frac{\ln(2N^2/\delta)}{2m}} .
$$
*Proof.* Let $v_f(\pi):=\mathbb E_{x\sim\nu}[f(x,\pi(x))]$, so $v_r=v$, and $\varepsilon_m:=\sqrt{\ln(2N^2/\delta)/(2m)}$. By Theorem 3.1 with a union bound over the at most $N^2$ pairs $(f,\pi)\in\mathcal F\times\Pi_{\mathcal F}$, with probability at least $1-\delta$ over $D^{\mathrm{eval}}$, $|\hat v_f(\pi)-v_f(\pi)|\le\varepsilon_m$ for all pairs. On this event and that of Corollary 3.7,
$$
v(\hat\pi)=v_r(\hat\pi)\ \ge\ \hat v_r(\hat\pi)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}\hat v_f(\hat\pi)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}\hat v_f(\pi^*)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}v_f(\pi^*)-2\varepsilon_m ,
$$
using $r\in\mathcal F_\varepsilon$, then the choice of $\hat\pi$ with $\pi^*\in\Pi_{\mathcal F}$. Hence $\mathrm{Regret}(\hat\pi)\le\max_{f\in\mathcal F_\varepsilon}[v_r(\pi^*)-v_f(\pi^*)]+2\varepsilon_m\le\max_{f\in\mathcal F_\varepsilon}\|f-r\|_{L_1(\nu\times\pi^*)}+2\varepsilon_m\le\sqrt{C_{\mathcal F}(\pi^*)}\max_{f\in\mathcal F_\varepsilon}\|f-r\|_{\nu\times\mu}+2\varepsilon_m$ by Lemma 3.9(b), and Corollary 3.7(ii) bounds the maximum by $\sqrt{20L_1}$. $\square$

(VS) is Eq. (3.2) of Xie et al. (2021) at horizon one — $\arg\max_\pi\min_{f\in\mathcal F_{\pi,\varepsilon}}f(s_0,\pi)$ over the version space of functions with small Bellman loss — where the Bellman loss is the squared loss, their completeness assumption is vacuous, and their coverage coefficient is $C_{\mathcal F}$; the coverage is class-dependent and can be far below $C^*$ for structured classes. The restriction to $\Pi_{\mathcal F}$ is what makes the empirical values uniformly controllable; over all of $\mathcal A^{\mathcal X}$ the supremum of an empirical average aligns with the sample and does not concentrate.

### 5.6 Neural classes, and what pessimism buys

NeuraLCB (Nguyen-Tang et al. 2022; `nguyen-tang2022Offline`) uses (LCB) with $b(x,a)=\beta\|\nabla_\theta f(x,a;\hat\theta)\|_{\Lambda^{-1}}$ for a neural-tangent-kernel Gram matrix, trains $\hat f$ by stochastic gradient descent in an online manner, and obtains regret $\tilde O(\sqrt{\tilde d/n})$ up to a coverage constant for $\pi^*$, $\tilde d$ the effective dimension, under a condition milder than single-policy concentrability (cited, not re-derived). Jeunen & Goethals (2021) report the practical version — regression reward models with a lower confidence bound — for recommendation.

*What pessimism buys.* Comparing Theorems 4.2 and 5.1: greedy needs uniform coverage, $C_{\mathrm{unif}}<\infty$, while pessimism needs single-policy coverage, $C^*<\infty$ (or its average or class-dependent forms). This is not a matter of worst-case rates. In the context-free case with counts $n(a)$, Xiao et al. (2021; `xiao2021Optimality`) define confidence-adjusted index rules $\arg\max_a\hat f(a)+\alpha/\sqrt{n(a)}$ — optimistic, greedy or pessimistic according to the sign of $\alpha$ — and prove that *every* such rule is minimax optimal, matching an $\Omega(1/\sqrt{\min_an(a)})$ lower bound on simple regret; that instance-dependent optimality in the sense of online bandits cannot be achieved by any batch algorithm; and that under a *weighted-minimax* criterion, which weights each instance by the inherent difficulty of predicting its optimal value, the pessimistic rule is the one that is optimal. Proposition 4.4 is the instance behind the criterion: pessimism's bound depends only on the optimal action's uncertainty, the others' on the largest.

## 6. Evaluating policies with the reward model: the direct method

The reward model also yields a value estimate for *any* policy. This is not how a value-based learner chooses its policy (§2.3), but it is what the class-restricted variants of §7 maximize, it is the value-based half of the doubly robust estimator, and its error bound exhibits the coverage coefficients in their simplest form.

### 6.1 Method

Split $D_n$ into $D^{\mathrm{reg}}$ ($n_1$ triples) and $D^{\mathrm{eval}}$ ($m$ triples); fit $\hat f$ on $D^{\mathrm{reg}}$ by (1); for a target policy $\pi$,
$$
\hat v^{\mathrm{DM}}(\pi):=\frac1m\sum_{x\in D^{\mathrm{eval}}}\hat f(x,\pi). \tag{DM}
$$
Only the evaluation contexts are consumed. The split makes $\hat f$ independent of them; $K$-fold cross-fitting recovers the full sample at the price of a constant (Chernozhukov et al. 2018; Athey & Wager 2021). The name is due to Dudík, Langford & Li (2011): "the first, which we call the *direct method* (DM), estimates the reward function from given data and uses this estimate in place of actual reward to evaluate the policy value."

### 6.2 Analysis

Throughout, $\hat f$ is fixed (we condition on $D^{\mathrm{reg}}$); $g:=\hat f-r$.

**Lemma 6.1 (bias–variance decomposition).** Conditional on $\hat f$,
$$
\mathbb E\big[\hat v^{\mathrm{DM}}(\pi)\big]-v(\pi)=\mathbb E_{d^\pi}[g],\qquad
\mathrm{Var}\big(\hat v^{\mathrm{DM}}(\pi)\big)=\frac1m\,\mathrm{Var}_{x\sim\nu}\big(\hat f(x,\pi)\big)\le\frac1{4m}.
$$
*Proof.* The summands $\hat f(x,\pi)$ are i.i.d. with mean $\mathbb E_{d^\pi}[\hat f]$, while $v(\pi)=\mathbb E_{d^\pi}[r]$; each summand lies in $[0,1]$. $\square$

The bias is the model error averaged under the *target* policy; the variance is that of a bounded sample mean, with no dependence on $\mu$, $\pi$ or the reward noise — the mirror image of importance weighting (zero bias, variance driven by $\pi/\mu$). These are Sections 3–4 of Dudík, Langford & Li (2011).

**Theorem 6.2 (direct-method error).** Fix $\pi$ and $\hat f$, and let $C(\pi):=\min\{C^\pi_2,C_{\mathcal F}(\pi)\}$. With probability at least $1-\delta$ over $D^{\mathrm{eval}}$,
$$
\big|\hat v^{\mathrm{DM}}(\pi)-v(\pi)\big|\ \le\ \sqrt{C(\pi)}\;\|\hat f-r\|_{\nu\times\mu}+\sqrt{\frac{\ln(2/\delta)}{2m}} .
$$
For a finite class $\Pi$ the statement holds simultaneously for all $\pi\in\Pi$ with $\ln(2|\Pi|/\delta)$ in the second term only.
*Proof.* Triangle inequality around $\mathbb E[\hat v^{\mathrm{DM}}(\pi)]$: the sampling term is Theorem 3.1 for $m$ summands in $[0,1]$; the bias term is Lemma 6.1 followed by Lemma 3.9(a) and (b). The bias term is a deterministic function of $\hat f$, so a union bound over $\Pi$ touches only the sampling term. $\square$

**Corollary 6.3 (rate under realizability).** Under (A3)–(A4), for even $n$ ($n_1=m=n/2$), with probability at least $1-2\delta$, $\ \big|\hat v^{\mathrm{DM}}(\pi)-v(\pi)\big|\le\sqrt{24\,C(\pi)\ln(N/\delta)/n}+\sqrt{\ln(2/\delta)/n}$.
*Proof.* Theorem 6.2 with (3) on $D^{\mathrm{reg}}$. $\square$

**Proposition 6.4 (misspecification floor).** If $r\notin\mathcal F$, then $\|\hat f-r\|_{\nu\times\mu}\ge\mathrm{dist}(r,\mathcal F):=\inf_{f\in\mathcal F}\|f-r\|_{\nu\times\mu}>0$ for every $n$, and the bias of (DM) does not vanish: its limit is $\mathbb E_{d^\pi}[f_{\mathcal F}-r]$ for the $L_2(d^\mu)$-projection $f_{\mathcal F}$ of $r$ onto $\mathcal F$ (unique when $\mathcal F$ is convex) — a quantity that depends on $\pi$, is bounded only by $\sqrt{C(\pi)}\,\mathrm{dist}(r,\mathcal F)$, and is not identifiable from $D_n$ without propensities.
*Proof.* The lower bound is the definition of the distance; by Lemma 3.4 the population minimizer is the projection and the empirical minimizer converges to it over the finite class; Lemma 6.1 identifies the limiting bias and Lemma 3.9 bounds it. $\square$

Three consequences. (i) Under realizability the direct method matches the $\sqrt{\text{coverage}/n}$ form of importance weighting with $\ln N$ in place of the weight range, and its bias term is uniform over *all* target policies at no extra cost — $\ln N$ is paid once, in the regression. (ii) Through $C_{\mathcal F}$ the coverage is class-dependent and can be finite where importance weights are unbounded (Lemma 3.9(c)). (iii) Without realizability there is no guarantee, and the bias is invisible to the regression loss; Wang, Agarwal & Dudík (2017) show that the minimax mean-squared-error lower bound for off-policy evaluation is matched by IPS and DR while DM carries no distribution-free guarantee, so in the assumption-free regime DM is a *component*, not an estimator of choice.

### 6.3 The doubly robust combination

Given an estimate $\hat\mu$ of the behavior policy (exact when propensities were logged), the **doubly robust** estimator corrects (DM) by the importance-weighted residual:
$$
\hat v^{\mathrm{DR}}(\pi):=\frac1m\sum_{(x,a,r)\in D^{\mathrm{eval}}}\Big[\hat f(x,\pi)+\frac{\pi(a\mid x)}{\hat\mu(a\mid x)}\big(r-\hat f(x,a)\big)\Big]. \tag{DR}
$$

**Proposition 6.5 (Dudík, Langford & Li 2011, Theorems 1–2, bandit form).** Let $\Delta:=\hat f-r$, $\delta_\mu:=1-\mu/\hat\mu$, $\Delta(x,\pi):=\sum_a\pi(a\mid x)\Delta(x,a)$. Conditional on $\hat f$ and $\hat\mu$,
$$
\mathbb E\big[\hat v^{\mathrm{DR}}(\pi)\big]-v(\pi)=\mathbb E_{d^\pi}\big[\Delta\,\delta_\mu\big],
$$
so the bias is the product of the two model errors and vanishes if either is zero; and when $\hat\mu=\mu$,
$$
m\,\mathrm{Var}\big(\hat v^{\mathrm{DR}}(\pi)\big)=\mathrm{Var}_x\big(r(x,\pi)\big)+\mathbb E_x\sum_a\frac{\pi(a\mid x)^2}{\mu(a\mid x)}\Big(\sigma^2(x,a)+\Delta(x,a)^2\Big)-\mathbb E_x\big[\Delta(x,\pi)^2\big].
$$
*Proof.* Condition on $x$. The summand has conditional mean $\hat f(x,\pi)+\sum_a\pi(a|x)\frac{\mu(a|x)}{\hat\mu(a|x)}(r-\hat f)(x,a)=r(x,\pi)+\sum_a\pi(a|x)\Delta(x,a)\delta_\mu(x,a)$; average over $x$. For the variance take $\hat\mu=\mu$ and use the law of total variance: the conditional mean is $r(x,\pi)$ and the conditional variance is that of $\frac{\pi(a|x)}{\mu(a|x)}(r-\hat f(x,a))$ with $a\sim\mu(x)$, namely $\sum_a\frac{\pi^2}{\mu}(\sigma^2+\Delta^2)-\Delta(x,\pi)^2$. $\square$

Setting $\hat f\equiv0$ recovers IPW with $r^2$ in place of $\Delta^2$: DR keeps unbiasedness and pays the importance-weighting penalty only on the residual, small wherever the reward model is good; dropping the correction recovers (DM), whose variance has no $1/\mu$ term. Refinements keep the template: MRDR (Farajtabar et al. 2018) fits $\hat f$ to minimize the DR variance; SWITCH (Wang, Agarwal & Dudík 2017) uses (DM) exactly on the pairs whose importance weight exceeds a threshold; shrinkage (Su et al. 2020) shrinks the weights. All require $\mu$ or a good $\hat\mu$, which the value-based learner of §2.1 does not.

## 7. The alternative formulation: estimate, then select

### 7.1 Unrestricted class: the same object

**Proposition 7.1.** Let $\Pi=\mathcal A^{\mathcal X}$. (a) The greedy policy $\pi_{\hat f}$ maximizes $\hat v^{\mathrm{DM}}$ over $\Pi$, and every maximizer agrees with $\pi_{\hat f}$ on the evaluation contexts; the population plug-in value $v_{\hat f}(\pi):=\mathbb E_{x\sim\nu}[\hat f(x,\pi)]$ is maximized by $\pi_{\hat f}$ for every $\nu$. (b) With a quantifier $b$, the rule (LCB) maximizes the pessimistic plug-in values $\frac1m\sum_{x\in D^{\mathrm{eval}}}[\hat f-b](x,\pi)$ and $\mathbb E_{x\sim\nu}[(\hat f-b)(x,\pi)]$ over $\Pi$ in the same sense. (c) Define the per-policy score $X_\pi:=v_{\hat f}(\pi)$ and width $W_\pi:=\mathbb E_{x\sim\nu}[b(x,\pi(x))]$ for deterministic $\pi$; on the event of the quantifier, $|X_\pi-v(\pi)|\le W_\pi$ for all $\pi$, the pessimistic selection rule $\arg\max_\pi X_\pi-W_\pi$ is (LCB), and its guarantee $W_{\pi^*}+W_{\pi^*}$ is (4).
*Proof.* For any $h:\mathcal X\times\mathcal A\to\mathbb R$ and any distribution over contexts, $\max_{\pi}\mathbb E[h(x,\pi)]=\mathbb E[\max_ah(x,a)]$, attained by the pointwise maximizer independently of the distribution; apply with $h=\hat f$ and $h=\hat f-b$. (c) integrates the quantifier along $a=\pi(x)$; then $v(\hat\pi)\ge X_{\hat\pi}-W_{\hat\pi}\ge X_{\pi^*}-W_{\pi^*}\ge v(\pi^*)-2W_{\pi^*}$. $\square$

So estimate-then-select with plug-in values is regression-then-greedy written differently: the selection objective is not computable (it involves $\nu$), but its maximizer is, because the maximization decouples across contexts. The per-policy widths do not scale with $|\Pi|=K^{|\mathcal X|}$ because uniformity was obtained over $\mathcal F$.

### 7.2 Restricted class: cost-sensitive classification with imputed rewards

When the deployed policy must lie in a given class $\Pi\subsetneq\mathcal A^{\mathcal X}$, the value-based learner becomes
$$
\hat\pi_\Pi:=\arg\max_{\pi\in\Pi}\hat v^{\mathrm{DM}}(\pi)=\arg\max_{\pi\in\Pi}\frac1m\sum_{x\in D^{\mathrm{eval}}}\hat f(x,\pi), \tag{DM-Π}
$$
a cost-sensitive classification problem with the imputed reward vector $\hat f(x,\cdot)$ at each context (Dudík, Langford & Li 2011, §5.1.3; Beygelzimer & Langford 2009). Its guarantee is a uniform-deviation bound.

**Theorem 7.2.** Let $\Pi$ be finite and $\hat f$ fitted on $D^{\mathrm{reg}}$. With probability at least $1-\delta$ over $D^{\mathrm{eval}}$,
$$
\mathrm{Regret}_\Pi(\hat\pi_\Pi)\ \le\ 2\max_{\pi\in\Pi}\big|\hat v^{\mathrm{DM}}(\pi)-v(\pi)\big|\ \le\ 2\sqrt{C_\Pi}\;\|\hat f-r\|_{\nu\times\mu}+2\sqrt{\frac{\ln(2|\Pi|/\delta)}{2m}},\qquad C_\Pi:=\max_{\pi\in\Pi}\min\{C^\pi_2,C_{\mathcal F}(\pi)\}.
$$
*Proof.* $v(\pi^*_\Pi)-v(\hat\pi_\Pi)\le[v(\pi^*_\Pi)-\hat v^{\mathrm{DM}}(\pi^*_\Pi)]+[\hat v^{\mathrm{DM}}(\hat\pi_\Pi)-v(\hat\pi_\Pi)]$ since $\hat v^{\mathrm{DM}}(\hat\pi_\Pi)\ge\hat v^{\mathrm{DM}}(\pi^*_\Pi)$; bound both brackets by the uniform version of Theorem 6.2. $\square$

The coverage is now that of the *whole class* — both $\pi^*_\Pi$ and the selected policy appear, exactly as in Theorem 4.2 — so the class-restricted plug-in learner inherits the greedy rule's need for coverage of everything it might select. Pessimism restores single-policy coverage at the level of policies: the rule $\arg\max_{\pi\in\Pi}\min_{f\in\mathcal F_\varepsilon}\hat v_f(\pi)$ satisfies, by the proof of Theorem 5.4 with $\Pi$ in place of $\Pi_{\mathcal F}$,
$$
\mathrm{Regret}_\Pi\ \le\ \sqrt{\frac{20\,C_{\mathcal F}(\pi^*_\Pi)\ln(N/\delta)}{n_1}}+2\sqrt{\frac{\ln(2N|\Pi|/\delta)}{2m}},
$$
which is the shape of Theorem 3.1 of Xie et al. (2021), with $\log(|\mathcal F||\Pi|/\delta)$ and the coverage of the comparator only. With doubly robust scores in place of $\hat f(x,\cdot)$, (DM-Π) becomes the policy learning of Athey & Wager (2021) and Zhou, Athey & Wager (2023): the same argmax over $\Pi$, a uniform-deviation bound driven by the DR variance of Proposition 6.5 and the complexity of $\Pi$, and cross-fitting for the nuisance estimates — a hybrid that uses the reward model as a control variate and the propensities for unbiasedness.

### 7.3 Summary of the relation

| Formulation | Policy | Needs | Coverage in the bound |
|---|---|---|---|
| regression → greedy (§4) | $\pi_{\hat f}$ | $r\in\mathcal F$ | uniform, $C_{\mathrm{unif}}$ |
| regression → (LCB) (§5) | $\arg\max_a\hat f-b$ | $r\in\mathcal F$, a quantifier $b$ | single-policy, $C^*$ / $\bar C^*$ / $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ |
| estimate-then-select, plug-in values, $\Pi=\mathcal A^{\mathcal X}$ (§7.1) | $=\pi_{\hat f}$, resp. $=$ (LCB) | same | same |
| estimate-then-select, plug-in values, $\Pi$ restricted (§7.2) | $\arg\max_{\pi\in\Pi}\hat v^{\mathrm{DM}}$ | $r\in\mathcal F$ | whole class, $C_\Pi$ |
| policy-level pessimism, version space (§5.5, §7.2) | $\arg\max_\pi\min_{f\in\mathcal F_\varepsilon}\hat v_f(\pi)$ | $r\in\mathcal F$ | class-dependent, $C_{\mathcal F}(\pi^*_\Pi)$ |
| estimate-then-select, DR scores (§7.2) | $\arg\max_{\pi\in\Pi}\hat v^{\mathrm{DR}}$ | $\mu$ or $\hat\mu$, and $\hat f$ | whole class, via the DR variance |
| estimate-then-select, IPW scores (§9) | $\arg\max_{\pi\in\Pi}\hat v^{\mathrm{IPW}}$ (MaxIPW / PES) | $\mu$ | $C^{\pi}_2$ over the class, resp. $C^{\pi^*_\Pi}_2$ |

## 8. Instance dependence and fast rates

### 8.1 Variance-aware quantifiers

**Proposition 8.1.** In the tabular setting, the rule (LCB) with the Bernstein quantifier $b^{\mathrm B}$ of Proposition 3.11 satisfies, conditional on the design and with probability at least $1-\delta$,
$$
\mathrm{Regret}(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\Big[\min\Big\{1,\sqrt{\tfrac{2\sigma^2(x,\pi^*(x))\ln(2SK/\delta)}{n(x,\pi^*(x))}}+\tfrac{2\ln(2SK/\delta)}{3\,n(x,\pi^*(x))}\Big\}\Big].
$$
*Proof.* Theorem 5.1 with Proposition 3.11. $\square$

The reward *variance* at $\pi^*$'s actions replaces the worst-case $1/4$; the unknown $\sigma^2$ is replaced by the sample variance through the empirical Bernstein inequality (Maurer & Pontil 2009) with the same form, and Yin & Wang (2021) give the rigorous tabular statement. This is the value-side counterpart of replacing second moments of importance weights by variances on the policy side (betting or freezing; [[Ryu2025Improved]]).

### 8.2 Gap and margin conditions

Define the **action gap** at $x$ as $\mathrm{gap}(x):=r(x,\pi^*(x))-\max_{a\ne\pi^*(x)}r(x,a)\ge0$.

**Theorem 8.2 (hard gap: the plug-in rule at rate $1/n$).** If $\mathrm{gap}(x)\ge\epsilon>0$ for $\nu$-almost every $x$, then for any $f:\mathcal X\times\mathcal A\to[0,1]$, with $g:=f-r$,
$$
\mathrm{Regret}(\pi_f)\ \le\ \frac4\epsilon\,\mathbb E_x\Big[\max_ag(x,a)^2\Big]\ \le\ \frac{4\,C_{\mathrm{unif}}}{\epsilon}\,\|f-r\|^2_{\nu\times\mu}.
$$
*Proof.* Fix $x$, $a^*=\pi^*(x)$, $\hat a=\pi_f(x)$. If $\hat a=a^*$ the regret at $x$ is $0$. Otherwise it is at least $\epsilon$ and, by Lemma 3.13, at most $2\max_a|g(x,a)|$; hence $2\max_a|g(x,a)|\ge\epsilon$ and the regret at $x$ is at most $2\max_a|g(x,a)|\cdot\frac{2\max_a|g(x,a)|}{\epsilon}=\frac4\epsilon\max_ag(x,a)^2$. Average over $x$, then $\max_ag^2\le\sum_ag^2=\sum_a\mu(a|x)\frac{g(x,a)^2}{\mu(a|x)}\le C_{\mathrm{unif}}\sum_a\mu(a|x)g(x,a)^2$. $\square$

**Corollary 8.3.** Under (A3)–(A4) and a hard gap $\epsilon$, with probability at least $1-\delta$, $\ \mathrm{Regret}(\pi_{\hat f})\le48\,C_{\mathrm{unif}}\ln(N/\delta)/(\epsilon\,n)$ — a $1/n$ rate for the *greedy* rule, without pessimism.
*Proof.* Theorem 8.2 with (3). $\square$

Theorem 8.2 is the hard-gap remark after Theorem 3.1 of Qian & Murphy (2011), $V(d_0)-V(d)\le4S[L(Q)-L(Q_0)]/\epsilon$ with $S=C_{\mathrm{unif}}$. Their Theorem 3.1 interpolates: under the margin condition $\mathbb P_x(\mathrm{gap}(x)\le\epsilon)\le C\epsilon^\alpha$ for all $\epsilon>0$, $\ \mathrm{Regret}(\pi_f)\le C'[\mathcal L(f)-\mathcal L(r)]^{(1+\alpha)/(2+\alpha)}$ with $C'=(2^{2+3\alpha}S^{1+\alpha}C)^{1/(2+\alpha)}$, so that $\alpha=0$ recovers Theorem 4.2 and $\alpha\to\infty$ recovers Theorem 8.2. Hu, Kallus & Uehara (2021) carry the mechanism — misranking requires errors larger than the gap — to infinite-horizon Markov decision processes and plug-in policies from fitted Q-iteration or Bellman-residual minimization: the regret rate is the exponentiation of the pointwise convergence rate of the $Q^*$ estimate, giving $O(1/n)$ in linear and $e^{-\Omega(n)}$ in tabular cases, the exponent depending on the noise level of the decision problem. On the policy route the analogous condition is a gap in *policy values*, $v(\pi^*_\Pi)-\max_{\pi\ne\pi^*_\Pi}v(\pi)$, which the same two-line argument converts into $O(W^2/\mathrm{gap})=O(1/n)$; but that gap is a property of the class, tiny or zero for large $\Pi$, whereas action gaps are a property of the problem. Fast rates are therefore natural on the value side and exceptional on the policy side.

## 9. Comparison with the policy-based route

The policy-based route of §2.4 is estimate-then-select with importance-weighted scores. Its guarantees are the same two inequalities as §3.5, read over policies.

**Proposition 9.1 (MaxIPW and PES).** Let $\Pi$ be finite and $W_\pi:=\frac1{\mu_{\min}}\sqrt{\ln(2|\Pi|/\delta)/(2n)}$, so that $|\hat v^{\mathrm{IPW}}(\pi)-v(\pi)|\le W_\pi$ for all $\pi\in\Pi$ with probability at least $1-\delta$ (Theorem 3.1, summands in $[0,1/\mu_{\min}]$). Then MaxIPW satisfies $\mathrm{Regret}_\Pi\le W_{\pi^*_\Pi}+W_{\hat\pi}$ and PES with any valid per-policy widths satisfies $\mathrm{Regret}_\Pi\le W^L_{\pi^*_\Pi}+W^U_{\pi^*_\Pi}$.
*Proof.* Lemma 3.13's first inequality with $f\to\hat v^{\mathrm{IPW}}$, $r\to v$, actions $\to$ policies; and Lemma 3.14 likewise. $\square$

With the Hoeffding width the two bounds coincide because the width is policy-independent; with variance-adaptive widths — the IPW summands have variance at most $\mathbb E_{d^\mu}[(\pi/\mu)^2r^2]\le C^\pi_2$, so Theorem 3.2 gives $W_\pi\asymp\sqrt{2C^\pi_2\ln(4|\Pi|/\delta)/n}+2\ln(4|\Pi|/\delta)/(3\mu_{\min}n)$ — PES is governed by $C^{\pi^*_\Pi}_2$ alone while MaxIPW is governed by $\max_\pi C^\pi_2$, the policy-level version of uniform versus single-policy coverage. (Making such widths data-driven is the note's hyperparameter-adaptation problem; IX and LS trade a chosen bias $\gamma C_\gamma(\pi)$ for bounded weights.)

| | Policy-based (IPW → MaxIPW / PES → IX / LS) | Value-based (regression → greedy / (LCB)) |
|---|---|---|
| Object estimated | $v(\pi)$ for each $\pi\in\Pi$ | the function $r$ on $\mathcal X\times\mathcal A$ |
| What must be well-specified | propensities $\mu$ (stochastic, known) | $\mathcal F\ni r$ (realizability, (A3)) |
| Source of bias | none (IPW); chosen, $\gamma C_\gamma(\pi)$ (IX) | misspecification, $\mathrm{dist}(r,\mathcal F)$; does not vanish with $n$ (Prop. 6.4) |
| Source of variance | ratios $\pi/\mu$, possibly unbounded | bounded regression; none from $\mu$ (Lemma 6.1) |
| Deficient support | fatal: weights undefined | tolerated through $C_{\mathcal F}$ (Lemma 3.9(c)) |
| Comparator | best in $\Pi$ | global optimum $\pi^*$ |
| Uncertainty quantified | per policy, $W_\pi$ | per pair, $b(x,a)$ (Def. 3.10) |
| Pessimism | $\arg\max_\pi\hat v(\pi)-W^U_\pi$ | $\arg\max_a\hat f(x,a)-b(x,a)$; equals per-policy pessimism on $\mathcal A^{\mathcal X}$ (Prop. 7.1) |
| Coverage in the bound | $C^{\pi^*_\Pi}_2$ (Bernstein widths); $C_\gamma(\pi^*_\Pi)$ (IX) | $C^*$, $\bar C^*$, $C_{\mathcal F}(\pi^*)$, $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ (§5) |
| Complexity factor | $\ln\lvert\Pi\rvert$ | $\ln N$, $S$, $d$, $\tilde d$ — larger: modeling rewards is harder than ranking actions |
| Learning (large class) | $\arg\max$ over $\Pi$: non-convex, NP-hard in general | regression (convex for linear $\mathcal F$) + per-context $\arg\max$ |
| Overparameterized models | objective not action-stable (Brandfonbrener et al. 2021) | action-stable; DR collapses to value-based |
| Selection (finite class) | native: one width per candidate | validation loss $\ne$ best policy (Qian & Murphy 2011, §3) |
| Fast rates | $1/n$ only under a gap in policy values | $1/n$ under per-context action gaps (Thm. 8.2) |
| Minimax OPE | IPS/DR match the lower bound | DM alone does not; DR combines both (§6.3) |

**Rule of thumb.** Trust the value-based route when a good reward model is plausible (rich features, dense actions, deterministic or unlogged $\mu$, large models); trust the policy-based route when the policy class is simple, $\mu$ is logged and stochastic, and rewards are hard to model. DR and SWITCH exist because the honest answer is usually "both, partially".

## 10. Bibliographic notes, reading order and vault actions

### 10.1 Provenance (verified against the PDFs, 2026-09-17)

*Formulation.* "Value-based learning. Another simple algorithm is to first learn the $Q$ function and then use a greedy policy with respect to this estimated $Q$ function," with $\hat Q:=\arg\min_{f\in\mathcal Q}\sum_i(f(x_i,a_i)-r_i(a_i))^2$ — Brandfonbrener, Whitney, Ranganath & Bruna (ICML 2021), who contrast it with "policy-based learning. Importance weighted or 'inverse propensity weighted' policy optimization directly optimizes the policy to maximize an estimate of its value," define action-stability, show value-based objectives are action-stable and policy-based ones are not, and note that the doubly robust approach collapses to the value-based one with overparameterized models. The same regression-then-greedy rule is "the regression approach" of Beygelzimer & Langford (KDD 2009, §6.1: "probably the simplest approach is to regress on the reward $r_a$ given $x$ and $a$, and then choose according to the largest predicted reward"; Theorem 6.1, $\mathrm{reg}(\pi_f)\le2\sqrt{k\,\mathrm{reg}_r(f)}$, tight), the plug-in individualized-treatment rule of Murphy (JMLR 2005; bound restated as (3.1) in Qian & Murphy, Ann. Statist. 2011, whose Theorem 3.1 and remarks contain §8.2 and whose §3 toy example is the selection mismatch), and the algorithm of Rashidinejad, Zhou, Ma, Jiao & Russell (NeurIPS 2021): LCB with penalty $\sqrt{L/n(x,a)}$, single-policy concentrability $C^\pi$ (their Definition 1), Theorem 4 $\tilde O(\sqrt{S(C^*-1)/N}+S/N)$, Theorem 5 lower bound, Proposition 1 on the failure of the empirical best arm. Jin, Yang & Wang (ICML 2021) supply the $\xi$-uncertainty quantifier, PEVI ($\hat Q_h=\hat{\mathbb B}_h\hat V_{h+1}-\Gamma_h$, greedy), Theorem 4.2 ($\mathrm{SubOpt}\le2\sum_h\mathbb E_{\pi^*}[\Gamma_h]$), the linear bonus $\beta\|\phi\|_{\Lambda^{-1}}$ and minimax optimality. Xie, Cheng, Jiang, Mineiro & Agarwal (NeurIPS 2021) supply Eq. (3.2), $\arg\max_\pi\min_{f\in\mathcal F_{\pi,\varepsilon}}f(s_0,\pi)$, and Definition 1, $C(\nu;\mu,\mathcal F,\pi)=\max_f\|f-\mathcal T^\pi f\|^2_{2,\nu}/\|f-\mathcal T^\pi f\|^2_{2,\mu}\le\|\nu/\mu\|_\infty$. Xiao, Wu, Lattimore, Dai, Mei, Li, Szepesvári & Schuurmans (2021; `xiao2021Optimality`): confidence-adjusted index rules $\arg\max_i\hat\mu_i+\alpha/\sqrt{n_i}$, all minimax optimal, $\Omega(1/\sqrt{\min_in_i})$ lower bound, no instance-dependent optimality in the batch setting, weighted-minimax criterion. Li, Ma & Srebro (NeurIPS 2022; full text read): pessimism-validity and Proposition 1 ($\mathrm{Regret}\le\beta\|\mathbb E\phi(\pi^*)\|_*$), the $\ell_p$ family with Theorem 1, $\hat\pi_2=$ BCP exactly, $\hat\pi_\infty=$ PUNC reducing to tabular LCB (Corollary 1), PEVI as a context-wise enlarged $\ell_2$ set, Theorem 2 lower bounds over $\mathrm{CB}_q(\Lambda)$, adaptive minimax optimality of PUNC; fixed design; the plug-in rule analyzed in their Appendix E. Hu, Kallus & Uehara (COLT 2021): plug-in policies from FQI or Bellman-residual minimization in infinite-horizon MDPs, regret rate $=$ exponentiation of the pointwise $Q^*$ rate, $O(1/n)$ linear, $e^{-\Omega(n)}$ tabular.

*Estimators.* "Direct method (DM)" is coined in `dudik2011Doubly`, §1 and §2.1, with the bias and variance identities of Lemma 6.1 as its Sections 3–4 (journal version: Dudík, Erhan, Langford & Li, *Statistical Science* 29(4):485–511, 2014); IPS is credited there to Horvitz & Thompson (1952), the DR form to Cassel, Särndal & Wretman (1976), and double robustness to Robins, Rotnitzky & Zhao (1994), Robins & Rotnitzky (1995), Lunceford & Davidian (2004), Kang & Schafer (2007); DM-based policy optimization is their §5.1.3 (cost-sensitive classification with imputed losses). Wang, Agarwal & Dudík (ICML 2017): minimax lower bound matched by IPS and DR; SWITCH. Voloshin, Le, Jiang & Yue (2019) adopt IPS / Direct Methods / Hybrid as the RL taxonomy, the DR extension to RL being Jiang & Li (ICML 2016).

### 10.2 Reading order

1. **Rashidinejad et al. (NeurIPS 2021)** — tabular LCB, single-policy concentrability, the $(C^*-1)$ regimes; Theorem 5.2's source.
2. **Jin, Yang, Wang (ICML 2021)** — uncertainty quantifiers, PEVI, Theorem 5.1's source; read the horizon-one case.
3. **Brandfonbrener et al. (ICML 2021)** — the value-based / policy-based distinction, action-stability, overparameterization.
4. **Xie et al. (NeurIPS 2021)** and **Li, Ma, Srebro (NeurIPS 2022)** — version-space (policy-level) pessimism, class-dependent coverage, and the pointwise-versus-policy-level comparison. *(Not to be confused with the vault's `li2022InstanceOptimal`, an online PAC paper by the same authors.)*
5. **`xiao2021Optimality`** (ICML 2021) — what pessimism does and does not buy (§5.6).
6. **`dudik2011Doubly`** + Wang, Agarwal, Dudík (ICML 2017) — direct method, DR, SWITCH (§6).
7. **Murphy (JMLR 2005)**, **Beygelzimer & Langford (KDD 2009, §6.1)**, **Qian & Murphy (Ann. Statist. 2011)** — the pre-pessimism origins of Theorems 4.2 and 8.2.
8. **`nguyen-tang2022Offline`** (ICLR 2022) — neural instantiation.
9. **Yin & Wang (NeurIPS 2021)**, **Hu, Kallus, Uehara (COLT 2021)** — §8.
10. Athey & Wager (Econometrica 2021), Zhou, Athey & Wager (Oper. Res. 2023) — class-restricted policy learning with DR scores (§7.2).
11. Policy-side context: `swaminathan2015Batch` (clipped IPW with an empirical-Bernstein penalty), `gabbianelli2023ImportanceWeighted` (IX), Sakhi et al. (NeurIPS 2024, LS), [[Ryu2025Improved]] (betting/freezing). Foster, Krishnamurthy, Simchi-Levi & Xu (COLT 2022) explain why the bandit case is clean and reinforcement learning is not.

### 10.3 Verify-before-citing flags

Verified on 2026-09-17 (see §10.1): everything attributed to Dudík et al., Beygelzimer & Langford, Murphy, Qian & Murphy, Rashidinejad et al. (Definition 1, Theorems 4–5, Proposition 1 as stated in their overview), Jin et al. (Theorem 4.2, bonus form), Xie et al. (Eq. 3.2, Definition 1), Xiao et al. (index rules, minimax results, weighted-minimax), Li–Ma–Srebro (full text), Hu–Kallus–Uehara (abstract-level rates), Wang–Agarwal–Dudík (abstract-level), Brandfonbrener et al. (definitions and action-stability). Still from memory: the Singh & Yee (1994) attribution for the classical greedy-policy loss bound (§4.2); Rashidinejad's exact penalty constant $L$; Jin's lower-bound instance class; Xie's exact Theorem 3.1 constants; the empirical-Bernstein statement of Maurer & Pontil; Athey & Wager / Zhou, Athey & Wager theorem forms; NeuraLCB's exact coverage condition.

### 10.4 Vault follow-ups

(a) Ingest items 1–4 (Zotero has 5, 6 (Dudík), 8, 11; the rest are cited author–year). (b) A concept page `offline-contextual-bandit` (the problem setting, §1.1–1.2) is warranted now; a topic page is *not yet* earned — revisit after ingesting items 1–4. (c) Drop the note into `raw/papers/Jun2026Offline.pdf`.

## Connections
- [[pessimism-principle]] — the shared mechanism; Lemma 3.14 is its two-line core
- [[importance-weighting]] — the policy route's estimator; DR (§6.3) is where the two routes meet
- [[fqi-finite-sample-analysis]] / [[fitted-q-iteration]] — the bandit is the horizon-one case; Theorem 3.6 is its "part C", Lemma 3.13 its "part A" with no error propagation
- [[extrapolation-error]] — the greedy rule's failure mode (§4.4)
- [[instance-dependent-bounds]] — §8
- [[Ryu2025Improved]] — the policy-route second-order bound that §8.1 mirrors
- [[contextual-bandits]] — the online problem this is the batch version of; distinct from [[offline-oracle-efficient-bandits]]
