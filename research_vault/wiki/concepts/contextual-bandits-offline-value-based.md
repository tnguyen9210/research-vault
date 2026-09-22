---
title: "Value-Based Offline Contextual Bandits"
tags: [contextual-bandits, offline-contextual-bandits, pessimism, learning-theory]
aliases: [value-based learning, regression-then-greedy, plug-in policy, LCB for contextual bandits, value-based-offline-bandits]
---

# Value-Based Offline Contextual Bandits

**Definition:** The family of offline contextual-bandit methods that fit a model $\hat q$ of the mean reward by least-squares regression on the logged triples and derive a policy from it — greedily, $\hat\pi=\pi_{\hat q}$, or pessimistically, $\hat\pi(x)\in\arg\max_a\hat q(x,a)-\Gamma(x,a)$ — without ever using the logged propensities.

> **Scope.** The problem setup, notation and the value-vs-policy taxonomy live in [[contextual-bandits-offline]]; this page assumes them and treats the value-based family alone: its estimators, algorithms, assumptions and guarantees. The two constructions that straddle the families — the doubly robust estimator and class-restricted learning with imputed rewards — are on the setting page, as is the comparison of the two routes. Notation follows the Overleaf research log, `02_offline_contextual_bandits.tex`.
>
> **What is derived here rather than quoted** is stated per result; §8.1 records what has been verified against the PDFs and §8.3 what has not.

## 0. Summary

1. **The representative formulation** is regression-then-act, not "estimate every policy's value and pick the largest" (§1.1, §1.3). Two versions: greedy (§3) and pessimistic (§4).
2. **Two lemmas carry every bound** (§2.5). The plug-in decomposition charges the greedy rule for the error at the action *it* picks, so greedy needs *uniform* coverage; the pessimism lemma charges only $\pi^*$'s uncertainty, so pessimism needs *single-policy* coverage. Replacing one by the other is what pessimism buys.
3. **Rates under realizability.** Tabular LCB $\tilde O(\sqrt{S\,C^*/T})$ with a matching lower bound (Theorem 4.2); linear $2\beta_\delta\,\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}=\tilde O(\sqrt{d/(\kappa T)})$ (Theorem 4.3); general finite $\mathcal F$ through the version space (Theorem 4.4); greedy $2\sqrt{12\,C_{\mathrm{unif}}\ln(|\mathcal F|/\delta)/T}$ (Corollary 3.3). Action gaps give $1/T$ (Theorem 7.2).
4. **Greedy against pessimism, one symbol apart.** In the tabular model greedy pays $2\sqrt{S\,C_{\mathrm{unif}}\ln(2SK/\delta)/T}$ and pessimism $2\sqrt{S\,C^*\ln(2SK/\delta)/T}$ (Theorem 3.5 vs Theorem 4.2), and the count conditions differ the same way. Since $\bar C_{\mathrm{unif}}\ge K$ always, the action count — absent from Theorem 3.1 — enters greedy at rate $\sqrt K$; and the regime where pessimism is fastest ($C^*\to1$) is where the greedy bound is vacuous.
5. **The sharpest tabular rate** is Rashidinejad et al.'s $\tilde O(\sqrt{S(C^*-1)/T}+S/T)$, proved in full in §4.7, whose $-1$ comes from one inequality bounding the mass of the ambiguous contexts by $10(C^*-1)$.

## 1. The value-based approach

### 1.1 The representative formulation: regression, then act greedily

A value-based method never estimates the value of a candidate policy. It estimates the **mean-reward function** $q^*$ by supervised regression on the logged triples — inputs $(x_t,a_t)$, labels $r_t$ — within a class $\mathcal F$,

$$
\hat q:=\arg\min_{f\in\mathcal F}\ \frac1T\sum_{t=1}^T\big(f(x_t,a_t)-r_t\big)^2, \tag{1}
$$

and then *derives* a policy from $\hat q$ by choosing, in each context, the action the model rates highest. Two versions exist.

- **Greedy (plug-in).** $\hat\pi(x)\in\arg\max_a\hat q(x,a)$, i.e. $\hat\pi=\pi_{\hat q}$. This is "the regression approach" of Beygelzimer & Langford (2009), the plug-in individualized-treatment rule of Murphy (2005) and Qian & Murphy (2011), and "value-based learning" in Brandfonbrener et al. (2021): "first learn the $Q$ function and then use a greedy policy with respect to this estimated $Q$ function."
- **Pessimistic (lower confidence bound).** With a penalty $\Gamma(x,a)\ge0$ computed from the data that is a valid *uncertainty quantifier*, i.e. $|\hat q(x,a)-q^*(x,a)|\le\Gamma(x,a)$ for all $(x,a)$ with high probability (Definition 2.10),

$$
\hat\pi(x)\in\arg\max_{a\in\mathcal A}\ \hat q(x,a)-\Gamma(x,a). \tag{LCB}
$$

This is LCB in Rashidinejad et al. (2021), pessimistic value iteration at horizon one in Jin, Yang & Wang (2021), the confidence-adjusted index rule with $\alpha=-\beta_\delta$ in Xiao et al. (2021), NeuraLCB in Nguyen-Tang et al. (2022), and, in the tabular case, the $\hat\pi_\infty$ (PUNC) rule of Li, Ma & Srebro (2022).

Both versions use the data through $\hat q$ (and $\Gamma$) alone: the propensities $\mu(a_t\mid x_t)$ are never used, the behavior policy may be deterministic, and no policy class is specified — the policy ranges over all of $\mathcal A^{\mathcal X}$, so the comparator is the global optimum.

### 1.2 Intuition

*Why regression.* The reward of the action a policy would have taken is unobserved whenever $\mu$ took another. A reward model fitted on the pairs $\mu$ did visit predicts the pairs it did not — a linear model predicts $q^*(x,2)$ from contexts where only $a=1$ was played — so the counterfactual is *interpolated through $\mathcal F$* rather than re-weighted through the randomness of $\mu$. The price is that the prediction is only as good as $\mathcal F$: the regression is fit where the data are, not where the derived policy acts, so a model accurate on average under $d^\mu$ may be wrong at the pairs the policy selects (Beygelzimer & Langford 2009; Dudík et al. 2011).

*Why pessimism.* The greedy policy chooses the action the model rates highest, which is systematically the action the model *over*-rates; among poorly covered actions there is always one whose estimate is high by chance. Subtracting an uncertainty quantifier makes an action attractive only if the data support its value, and the analysis then charges the learner only for uncertainty at the optimal policy's actions rather than at whatever it chose (Lemma 2.14). This turns a *uniform* coverage requirement into a *single-policy* one, which is the central theme of the offline literature (Rashidinejad et al. 2021; Jin, Yang & Wang 2021).

### 1.3 The alternative formulation, and how it relates

The other way to write an offline learner is *estimate, then select*: form an estimate $\widehat J(\pi)$ of every candidate's value and output $\arg\max_{\pi\in\Pi}\widehat J(\pi)$ — or, pessimistically, $\arg\max_\pi\widehat J(\pi)-W_\pi$ with per-policy widths. This is the **policy-optimization** view. It is the natural home of the policy-based route (importance-weighted estimates: see [[contextual-bandits-offline]]), but it can also be run with value-based estimates: the **direct method** plugs the reward model into every policy's value, $\widehat J^{\mathrm{DM}}(\pi)=\frac1T\sum_t\hat q(x_t,\pi)$ (§5), and the doubly robust estimator combines the two (in [[contextual-bandits-offline]]). Its relation to §1.1 is exact:

- **Unrestricted class.** If $\Pi=\mathcal A^{\mathcal X}$, the maximizer of $\widehat J^{\mathrm{DM}}$ over $\Pi$ is the greedy policy $\pi_{\hat q}$ on the logged contexts, and the maximizer of the pessimistic plug-in value $\frac1T\sum_t[\hat q-\Gamma](x_t,\pi)$ is (LCB): the maximization decouples across contexts (Proposition 6.1). Estimate-then-select with plug-in values *is* regression-then-greedy.
- **Restricted class.** If $\Pi\subsetneq\mathcal A^{\mathcal X}$, maximizing $\widehat J^{\mathrm{DM}}$ over $\Pi$ is cost-sensitive classification with the imputed reward vector $\hat q(x_t,\cdot)$ — the "direct method for policy optimization" of Dudík et al. (2011), the Offset Tree's regression baseline, and, with doubly robust scores in place of $\hat q$, the policy learning of Athey & Wager (2021) and Zhou, Athey & Wager (2023). Its guarantee is a uniform-deviation bound over $\Pi$ (stated in [[contextual-bandits-offline]]), and it is what one uses when the deployed policy must lie in a given class.
- **Policy-level pessimism.** Pessimism can also be applied to policy values rather than to action values: choose $\arg\max_{\pi}\min_{f\in\mathcal F_\varepsilon}\widehat J_f(\pi)$ over a version space $\mathcal F_\varepsilon$ of reward models consistent with the data. This is Bellman-consistent pessimism at horizon one (Xie et al. 2021; Theorem 4.4) and the $\hat\pi_2$ rule of Li, Ma & Srebro (2022). Their framework — a confidence set $\Theta\ni\theta^*$ induces the pessimistic value $\hat V(\pi):=\inf_{\theta\in\Theta}\mathbb E_x[\phi(x,\pi(x))^\top\theta]$, then $\hat\pi:=\arg\max_\pi\hat V(\pi)$ — is exactly estimate-then-select with pessimistic plug-in values, and it contains both forms: the $\ell_2$ set gives $\hat\pi_2=$ BCP, the $\ell_\infty$ set gives $\hat\pi_\infty$ (PUNC), which reduces to tabular LCB and is adaptively minimax optimal, strictly dominating $\hat\pi_2$ (§4.4). It is the right tool when no pointwise uncertainty quantifier is available (general $\mathcal F$).

So the literature's value-based methods are organized around (1)+(LCB); the estimate-then-select formulation is either the same object written differently (unrestricted class), the class-restricted variant, or the policy-level form of pessimism. The page follows that organization: §2 tools, §3 greedy, §4 pessimistic, §5 evaluation with the reward model, §6 the class-restricted and policy-level variants, §7 fast rates, the route comparison in [[contextual-bandits-offline]] comparison with the policy-based route.

## 2. Technical toolkit

Everything in §3–§7 is assembled from this section: concentration inequalities (§2.1); the least-squares guarantee under realizability (§2.2); coverage coefficients and the change of measure that carries an error under the data distribution to an error under a target policy (§2.3); uncertainty quantifiers for $\hat q$ (§2.4); and the two suboptimality lemmas on which every regret bound rests (§2.5).

### 2.1 Concentration inequalities

**Theorem 2.1 (Hoeffding).** Let $Z_1,\dots,Z_T$ be i.i.d. with values in $[a,b]$. With probability at least $1-\delta$, $\big|\frac1T\sum_iZ_i-\mathbb EZ\big|\le(b-a)\sqrt{\ln(2/\delta)/(2T)}$. For $M$ such sample means simultaneously, replace $\delta$ by $\delta/M$ (union bound).

**Theorem 2.2 (Bernstein, one-sided).** Let $Y_1,\dots,Y_T$ be i.i.d. with $\mathrm{Var}(Y)\le s^2$ and $|Y-\mathbb EY|\le c$ almost surely. With probability at least $1-\delta$, $\ \mathbb EY-\frac1T\sum_iY_i\le\sqrt{2s^2\ln(1/\delta)/T}+2c\ln(1/\delta)/(3T)$.

**Theorem 2.3 (self-normalized bound; Abbasi-Yadkori, Pál & Szepesvári 2011, Thm 2).** Let $\phi_t:=\phi(x_t,a_t)\in\mathbb R^d$ with $\|\phi_t\|_2\le1$, and $r_t=\langle\theta^*,\phi_t\rangle+\eta_t$ with $\mathbb E[\eta_t\mid x_t,a_t]=0$ and $\eta_t$ supported on an interval of length $1$ (hence $\tfrac12$-sub-Gaussian), $\|\theta^*\|_2\le B$. Let $\Lambda:=\lambda I+\sum_t\phi_t\phi_t^\top$ and $\hat\theta:=\Lambda^{-1}\sum_t\phi_tr_t$. With probability at least $1-\delta$,

$$
\|\hat\theta-\theta^*\|_{\Lambda}\ \le\ \beta_\delta:=\tfrac12\sqrt{2\ln(1/\delta)+d\ln\big(1+\tfrac{T}{\lambda d}\big)}+\sqrt\lambda\,B .
$$

### 2.2 Least-squares regression under realizability

**Lemma 2.4 (Pythagorean identity).** For any $f:\mathcal X\times\mathcal A\to[0,1]$, $\ \mathcal L(f)-\mathcal L(q^*)=\|f-q^*\|^2_{\nu\times\mu}$.
*Proof.* $\mathcal L(f)=\mathbb E[(f-q^*+r-r_t)^2]=\|f-q^*\|^2_{\nu\times\mu}+\mathcal L(q^*)+2\,\mathbb E[(f-q^*)(x,a)(q^*(x,a)-r_t)]$, and the cross term vanishes because $\mathbb E[q^*(x,a)-r_t\mid x,a]=0$. $\square$

Minimizing the population squared loss over $\mathcal F$ is therefore projecting $q^*$ onto $\mathcal F$ in $L_2(d^\mu)$, and the excess loss is exactly the squared $L_2$ error.

**Lemma 2.5 (excess-loss variables).** For $f\in\mathcal F$ and one triple $(x,a,r_t)$ let $Y(f):=(f(x,a)-r_t)^2-(q^*(x,a)-r_t)^2$. Then $\mathbb E[Y(f)]=\|f-q^*\|^2_{\nu\times\mu}$, $Y(f)\in[-2,2]$, $|Y(f)-\mathbb EY(f)|\le3$, and $\mathrm{Var}(Y(f))\le4\,\mathbb E[Y(f)]$.
*Proof.* $Y(f)=(f-q^*)(f+q^*-2r_t)$ with $|f-q^*|\le1$ and $|f+q^*-2r_t|\le2$; the mean is Lemma 2.4; $\mathbb E[Y^2]\le4\,\mathbb E[(f-q^*)^2]$; and $\mathbb EY\in[0,1]$ gives the range of $Y-\mathbb EY$. $\square$

The variance bound is a *Bernstein condition* — the fluctuation of the excess loss shrinks with the excess loss itself — and it is what yields a $1/T$ rather than $1/\sqrt T$ rate for the squared error.

**Theorem 2.6 (realizable least squares, finite class).** Under (A1)–(A4), let $\hat q$ minimize $\mathcal L_{\mathcal D}$ over $\mathcal F$ on $T$ i.i.d. triples and set $L:=\ln(|\mathcal F|/\delta)/T$. With probability at least $1-\delta$, simultaneously for all $f\in\mathcal F$,

$$
\mathbb E[Y(f)]-\frac1T\sum_{i=1}^nY_i(f)\ \le\ \sqrt{8\,\mathbb E[Y(f)]\,L}+2L, \tag{2}
$$

and consequently

$$
\|\hat q-q^*\|^2_{\nu\times\mu}\ \le\ 12\,L\ =\ \frac{12\ln(|\mathcal F|/\delta)}{T}. \tag{3}
$$

*Proof.* (2) is Theorem 2.2 with $s^2=4\,\mathbb E[Y(f)]$ and $c=3$ (Lemma 2.5) at confidence $\delta/N$, union-bounded over $\mathcal F$. Because $q^*\in\mathcal F$ and $\hat q$ minimizes the empirical loss, $\frac1T\sum_iY_i(\hat q)=\mathcal L_{\mathcal D}(\hat q)-\mathcal L_{\mathcal D}(q^*)\le0$. Applying (2) at $f=\hat q$ and writing $x:=\|\hat q-q^*\|^2_{\nu\times\mu}$: $x\le\sqrt{8xL}+2L\le x/2+4L+2L$, i.e. $x\le12L$. $\square$

**Corollary 2.7 (version space).** On the event of Theorem 2.6, with $\varepsilon:=4L$, the set $\mathcal F_\varepsilon:=\{f\in\mathcal F:\mathcal L_{\mathcal D}(f)\le\mathcal L_{\mathcal D}(\hat q)+\varepsilon\}$ satisfies (i) $q^*\in\mathcal F_\varepsilon$ and (ii) $\|f-q^*\|^2_{\nu\times\mu}\le20L$ for every $f\in\mathcal F_\varepsilon$.
*Proof.* (i) $\mathcal L_{\mathcal D}(q^*)-\mathcal L_{\mathcal D}(\hat q)=-\frac1T\sum_iY_i(\hat q)\le(\sqrt{8xL}-x)+2L\le2L+2L$, since $\max_{x\ge0}(\sqrt{8xL}-x)=2L$. (ii) For $f\in\mathcal F_\varepsilon$, $\frac1T\sum_iY_i(f)=\mathcal L_{\mathcal D}(f)-\mathcal L_{\mathcal D}(q^*)\le\mathcal L_{\mathcal D}(\hat q)+\varepsilon-\mathcal L_{\mathcal D}(q^*)\le\varepsilon$; by (2), $x_f:=\mathbb E[Y(f)]\le\varepsilon+2L+\sqrt{8x_fL}\le6L+x_f/2+4L$, so $x_f\le20L$. $\square$

Theorem 2.6 controls an $L_2(d^\mu)$ norm, never a supremum: it says nothing about $|\hat q(x,a)-q^*(x,a)|$ at an individual pair. Pointwise control needs structure (§2.4); without it, pessimism is implemented at the level of policies through the version space (Theorem 4.4).

### 2.3 Coverage and the change of measure

**Definition 2.8 (coverage coefficients).** For a policy $\pi$,

$$
C^\pi:=\sup_{x,a}\frac{d^\pi(x,a)}{d^\mu(x,a)}=\sup_{x,a:\ \pi(a|x)>0}\frac{\pi(a\mid x)}{\mu(a\mid x)},\qquad
C^\pi_2:=\mathbb E_{d^\mu}\Big[\Big(\frac{\pi}{\mu}\Big)^2\Big]=\mathbb E_{x\sim\nu}\sum_a\frac{\pi(a\mid x)^2}{\mu(a\mid x)},\qquad
C_{\mathcal F}(\pi):=\sup_{f\in\mathcal F,\ f\ne q^*}\frac{\|f-q^*\|^2_{\nu\times\pi}}{\|f-q^*\|^2_{\nu\times\mu}},
$$

with $c/0:=\infty$ for $c>0$. $C^\pi$ is the **single-policy concentrability** coefficient (Rashidinejad et al. 2021, Definition 1), $C^*:=C^{\pi^*}$; the **uniform** coefficient is $C_{\mathrm{unif}}:=\sup_\pi C^\pi=1/\mu_{\min}$. $C^\pi_2$ is the second moment of the importance weights, and $\bar C^*:=C^{\pi^*}_2$. $C_{\mathcal F}(\pi)$ is the class-dependent coefficient of Xie et al. (2021, Definition 1) at horizon one, where the Bellman residual $f-\mathcal T^\pi f$ is $f-q^*$. For deterministic $\pi$: $C^\pi=\sup_x1/\mu(\pi(x)\mid x)$ and $C^\pi_2=\mathbb E_x[1/\mu(\pi(x)\mid x)]$. Always $C^\pi_2\le C^\pi\le C_{\mathrm{unif}}$ and $C_{\mathcal F}(\pi)\le C^\pi$; $C^\pi_2$ and $C_{\mathcal F}(\pi)$ are not ordered in general.

**Lemma 2.9 (change of measure).** (a) For any $g:\mathcal X\times\mathcal A\to\mathbb R$ and any $\pi$, $\ \big|\mathbb E_{d^\pi}[g]\big|\le\|g\|_{L_1(\nu\times\pi)}\le\sqrt{C^\pi_2}\,\|g\|_{\nu\times\mu}$.
(b) If $g=f-q^*$ with $f\in\mathcal F$, $\ \|g\|_{L_1(\nu\times\pi)}\le\|g\|_{\nu\times\pi}\le\sqrt{C_{\mathcal F}(\pi)}\,\|g\|_{\nu\times\mu}\le\sqrt{C^\pi}\,\|g\|_{\nu\times\mu}$.
(c) If $\mathcal F=\{\langle\theta,\phi\rangle\}$ is linear and realizable, then with $\Sigma_\pi:=\mathbb E_{d^\pi}[\phi\phi^\top]$ and $\Sigma_\mu:=\mathbb E_{d^\mu}[\phi\phi^\top]$ (pseudo-inverse if singular, all $\phi$ in its range), $\ C_{\mathcal F}(\pi)\le\lambda_{\max}\big(\Sigma_\mu^{-1/2}\Sigma_\pi\Sigma_\mu^{-1/2}\big)\le\mathbb E_{d^\pi}\|\phi\|^2_{\Sigma_\mu^{-1}}$.
*Proof.* (a) $\mathbb E_x\sum_a\pi(a|x)|g|=\mathbb E_x\sum_a\mu(a|x)\frac{\pi(a|x)}{\mu(a|x)}|g|\le\big(\mathbb E_{d^\mu}[(\pi/\mu)^2]\big)^{1/2}\|g\|_{\nu\times\mu}$ (Cauchy–Schwarz). (b) Jensen, the definition of $C_{\mathcal F}$, and $\|g\|^2_{\nu\times\pi}=\mathbb E_{d^\mu}[(\pi/\mu)g^2]$. (c) $f-q^*=\langle\theta-\theta^*,\phi\rangle$, so the ratio is $\frac{u^\top\Sigma_\pi u}{u^\top\Sigma_\mu u}\le\lambda_{\max}(\cdot)\le\mathrm{tr}(\Sigma_\mu^{-1}\Sigma_\pi)$, which is the stated expectation. $\square$

Part (c) is the mechanism behind "extrapolation": class-dependent coverage is measured in feature space and can be finite when $\mu(a\mid x)=0$ for some of $\pi$'s actions, where $C^\pi=C^\pi_2=\infty$ and every importance-weighted estimator is undefined.

### 2.4 Uncertainty quantifiers

**Definition 2.10 ($\delta$-uncertainty quantifier).** Given $\hat q$, a function $b:\mathcal X\times\mathcal A\to[0,\infty)$ is a $\delta$-uncertainty quantifier if $\ \mathbb P\big(\forall(x,a):\ |\hat q(x,a)-q^*(x,a)|\le b(x,a)\big)\ge1-\delta$. (This is the $\xi$-uncertainty quantifier of Jin, Yang & Wang (2021) at horizon one and the penalty function of Rashidinejad et al. (2021).)

The condition is used in two roles, and the two get separate symbols. A quantifier $b$ appearing in an *analysis* need not be computable from the data: any valid bound may be used, and the sharpest available one gives the sharpest conclusion. A penalty $\Gamma$ that an *algorithm* subtracts, as in (LCB), is evaluated by that algorithm and so must be computable, and the algorithm's guarantee holds only if $\Gamma$ is in addition a valid quantifier — pessimism-validity in the sense of Li, Ma & Srebro (2022). Hence Theorem 3.1 is stated with $b$, which the greedy rule never touches, and Lemma 2.14 with $\Gamma$. Of the two tabular quantifiers below, $b^{\mathrm H}$ is computable and serves in either role, while $b^{\mathrm B}$ involves the unknown variance $\sigma^2(x,a)$ and is an analysis object until that variance is replaced by an empirical estimate.

**Proposition 2.11 (tabular quantifiers).** Let $S,K<\infty$, $\mathcal F=[0,1]^{\mathcal X\times\mathcal A}$, and let $\hat q(x,a)$ be the empirical mean of the rewards logged at $(x,a)$ (any value in $[0,1]$ if $N(x,a)=0$); this is the least-squares fit (1). Conditional on the design $\{(x_t,a_t)\}_{t\le T}$, each of

$$
b^{\mathrm H}(x,a):=\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,N(x,a)}}\Big\},\qquad
b^{\mathrm B}(x,a):=\min\Big\{1,\sqrt{\tfrac{2\sigma^2(x,a)\ln(2SK/\delta)}{N(x,a)}}+\tfrac{2\ln(2SK/\delta)}{3\,N(x,a)}\Big\}
$$

is a $\delta$-uncertainty quantifier.
*Proof.* Given the design, the rewards in cell $(x,a)$ are $N(x,a)$ i.i.d. draws from $\rho(x,a)$. $b^{\mathrm H}$: Theorem 2.1 at confidence $\delta/(SK)$, union bound over cells. $b^{\mathrm B}$: Theorem 2.2 on both sides with $s^2=\sigma^2(x,a)$, $c=1$, at confidence $\delta/(2SK)$, union bound over cells and sides. Empty cells use $|\hat q-q^*|\le1$. $\square$

**Proposition 2.12 (linear quantifier).** Let $\mathcal F=\{\langle\theta,\phi(\cdot,\cdot)\rangle:\|\theta\|_2\le B\}$ with $\|\phi\|_2\le1$, realizable with $q^*=\langle\theta^*,\phi\rangle$, and let $\hat q:=\langle\hat\theta,\phi\rangle$ with $\hat\theta$ the ridge estimate of Theorem 2.3. Then $b(x,a):=\beta_\delta\|\phi(x,a)\|_{\Lambda^{-1}}$ is a $\delta$-uncertainty quantifier.
*Proof.* On the event of Theorem 2.3, $|\hat q(x,a)-q^*(x,a)|=|\langle\hat\theta-\theta^*,\phi(x,a)\rangle|\le\|\hat\theta-\theta^*\|_\Lambda\|\phi(x,a)\|_{\Lambda^{-1}}\le\beta_\delta\|\phi(x,a)\|_{\Lambda^{-1}}$. $\square$

Here $\beta_\delta=\tilde O(\sqrt d)$, and if $\Lambda\succeq\kappa nI$ then $b\le\beta_\delta/\sqrt{\kappa T}$ uniformly. This quantifier is computable, so it also serves as $\Gamma$. For a general finite $\mathcal F$ no pointwise quantifier follows from Theorem 2.6; Theorem 4.4 works with the version space instead.

### 2.5 The two suboptimality lemmas

Both lemmas are pointwise in $x$; regret bounds follow by averaging over $x\sim\nu$. Write $g:=f-q^*$.

**Lemma 2.13 (plug-in decomposition).** For any $f:\mathcal X\times\mathcal A\to[0,1]$ and every $x$, with $a^*=\pi^*(x)$ and $\hat a=\pi_f(x)$,

$$
q^*(x,a^*)-q^*(x,\hat a)\ \le\ (q^*-f)(x,a^*)+(f-q^*)(x,\hat a)\ \le\ |g(x,a^*)|+|g(x,\hat a)|\ \le\ 2\max_a|g(x,a)| .
$$

*Proof.* $q^*(x,a^*)-q^*(x,\hat a)=(q^*-f)(x,a^*)+\big(f(x,a^*)-f(x,\hat a)\big)+(f-q^*)(x,\hat a)$, and the middle term is $\le0$ by the definition of $\pi_f$. $\square$

**Lemma 2.14 (pessimism).** Let $\Gamma$ satisfy $|\hat q(x,a)-q^*(x,a)|\le\Gamma(x,a)$ for all $(x,a)$ and let $\hat\pi$ be the rule (LCB). Then for every $x$,

$$
q^*(x,\pi^*(x))-q^*(x,\hat\pi(x))\ \le\ 2\,\Gamma(x,\pi^*(x)) .
$$

*Proof.* $q^*(x,\hat\pi(x))\ge\hat q(x,\hat\pi(x))-\Gamma(x,\hat\pi(x))\ge\hat q(x,\pi^*(x))-\Gamma(x,\pi^*(x))\ge q^*(x,\pi^*(x))-2\Gamma(x,\pi^*(x))$, using the quantifier, the definition of (LCB), and the quantifier again. $\square$

The difference between the two is the whole story. The plug-in rule is charged for the error at the action *it* chose, which is random and is systematically the over-estimated one; the pessimistic rule is charged only at the action the *optimal* policy chooses. Lemma 2.14 is Theorem 4.2 of Jin, Yang & Wang (2021) at horizon one and the core of the LCB analysis in Rashidinejad et al. (2021). (Read with policies in place of actions and $\widehat J(\pi)$ in place of $\hat q(x,a)$, the same two inequalities give the MaxIPW and PES guarantees of the policy-based route; the route comparison in [[contextual-bandits-offline]].)

## 3. The greedy policy

### 3.1 Method

Fit $\hat q$ by (1) on all $T$ triples and output $\hat\pi=\pi_{\hat q}$. No propensities, no sample split, no policy class.

### 3.2 Analysis

**Theorem 3.1 (greedy, pointwise form).** Let $b:\mathcal X\times\mathcal A\to[0,\infty)$ be any function bounding the error of $\hat q$ pointwise, in the sense that $|\hat q(x,a)-q^*(x,a)|\le b(x,a)$ for all $(x,a)$ simultaneously with probability at least $1-\delta$ over $\mathcal D$. Then on that same event,

$$
\Delta(\pi_{\hat q})\ \le\ \mathbb E_x\big[b(x,\pi^*(x))\big]+\mathbb E_x\big[b(x,\pi_{\hat q}(x))\big]\ \le\ \mathbb E_x\big[b(x,\pi^*(x))\big]+\mathbb E_x\Big[\max_ab(x,a)\Big].
$$

*Proof.* Lemma 2.13 with $|g|\le b$, averaged over $x\sim\nu$. $\square$

**Reading Theorem 3.1.** The greedy rule can be wrong at a context only when the model error there is at least as large as the true gap it has to resolve, so its loss is controlled by the accuracy of $\hat q$ at two actions only: the one $\pi^*$ takes, and the one the rule itself takes. The first term is unavoidable — up to a factor of two it is the *only* term the pessimistic rule pays (Lemma 2.14) — so the second term is the entire difference between the two approaches. That term is the price of letting the data choose the action: the rule commits to whichever action the regression rates highest, so accuracy at the remaining actions does not help it, and relaxing it to $\mathbb E_x[\max_ab(x,a)]$ is the most that can be said in general, because the rule has to be accurate wherever it might land.

**Theorem 3.1 in words.** Read $b(x,a)$ as the error bar of the fitted model at $(x,a)$. The theorem says: you lose at most the amount by which you underrated the truly best action, plus the amount by which you overrated the action you took. The proof is three facts chained. The true value of $a^*$ is at most its predicted value plus the error bar at $a^*$; the predicted value of $a^*$ is at most that of $\hat a$, which is why $\hat a$ was chosen; and the predicted value of $\hat a$ is at most its true value plus the error bar at $\hat a$. So the rule can be fooled only by being too pessimistic about the good action or too optimistic about the one it chose, and the loss is capped by how much — nothing else about the instance matters, and the number of actions does not enter. Since *which* action gets chosen depends on the data, a bound stated in advance has to allow for the worst error bar, which is the second inequality. The first term is usually harmless: a sensible behavior policy plays good actions often, so $\pi^*$'s actions are well estimated. The second is the problem, and not merely as bookkeeping — a rarely played action has a wide error bar, which is exactly the condition under which its estimate can come out too high, and the greedy rule takes whatever comes out highest, so it is drawn to the actions it knows least about. Proposition 3.4 makes that concrete.

**Provenance of Theorem 3.1.** This is an assembly rather than a quotation. Both of its ingredients are classical: Lemma 2.13, whose sup-norm version is the horizon-one case of the classical bound on the loss of a policy greedy with respect to an approximate value function (Singh & Yee 1994 — *from memory, not verified against the PDF; §8.3*), and the $\xi$-uncertainty quantifier of Jin, Yang & Wang (2021), whose Theorem 3.2 is its pessimistic counterpart, recorded here as Lemma 2.14. The greedy statement in this quantifier form does not appear to be a named result in any single paper, and the proof above is the standard two-line argument, written out rather than cited.

**Theorem 3.2 (greedy, $L_2$ form; uniform coverage).** For any $f:\mathcal X\times\mathcal A\to[0,1]$,

$$
\Delta(\pi_f)\ \le\ \Big(\sqrt{C^{\pi^*}_2}+\sqrt{C^{\pi_f}_2}\Big)\,\|f-q^*\|_{\nu\times\mu}\ \le\ 2\sqrt{C_{\mathrm{unif}}}\;\|f-q^*\|_{\nu\times\mu}.
$$

*Proof.* Average the middle expression of Lemma 2.13 over $x$ and apply Lemma 2.9(a) to each term, with $\pi=\pi^*$ and $\pi=\pi_f$ (both deterministic); then $C^{\pi_f}_2\le C^{\pi_f}\le C_{\mathrm{unif}}$. $\square$

**Reading Theorem 3.2.** This answers a different question. Theorem 2.6 controls an average squared error under $d^\mu$ and nothing more, and Theorem 3.2 converts that average into a statement about the decision loss. The conversion is the Cauchy–Schwarz step of Lemma 2.9(a), which is why the coverage coefficient enters under a square root, and why a squared error of order $1/T$ can only become a regret of order $1/\sqrt T$. The two results are therefore two currencies for the same fact: Theorem 3.1 is the sharper of them but needs pointwise accuracy, which Theorem 2.6 does not deliver without further structure (§2.4), while Theorem 3.2 needs only what Theorem 2.6 does deliver, and pays for it with a coefficient that ranges over all actions rather than the optimal one.

By Lemma 2.4, $\|f-q^*\|^2_{\nu\times\mu}=\mathcal L(f)-\mathcal L(q^*)$, so Theorem 3.2 reads $\Delta(\pi_f)\le2\mu_{\min}^{-1/2}[\mathcal L(f)-\mathcal L(q^*)]^{1/2}$: this is Murphy's (2005) generalization-error bound for the plug-in rule, restated as (3.1) in Qian & Murphy (2011), and, for $K$ actions logged uniformly, Theorem 6.1 of Beygelzimer & Langford (2009), $\mathrm{reg}(\pi_f)\le2\sqrt{K\,\mathrm{reg}_r(f)}$, which they show is tight. The proof shows where the uniform coverage comes from: the $\pi^*$ term costs $\sqrt{\bar C^*}$, the coverage of the *optimal* policy, while the chosen-action term costs the coverage of the *learned* policy, which can only be bounded by the least-covered action.

**Provenance of Theorem 3.2.** This bound is not new. Its content is Murphy's (2005) generalization-error bound and Theorem 6.1 of Beygelzimer & Langford (2009), both verified in §8.1; what is written here is that bound restated in terms of the second-moment coefficients of Definition 2.8 — the change of measure standard in the offline RL literature — in place of the original $\mu_{\min}^{-1/2}$ constant. The same holds for Theorem 2.6, and hence for Corollary 3.3: standard realizable analysis, not a result of ours.

**Corollary 3.3 (rate under realizability).** Under (A3)–(A4), with probability at least $1-\delta$, $\ \Delta(\pi_{\hat q})\le2\sqrt{12\,C_{\mathrm{unif}}\ln(|\mathcal F|/\delta)/T}$.
*Proof.* Theorem 3.2 with (3). $\square$

Corollary 3.3 is the class-complexity currency; §3.5 instantiates Theorem 3.1 instead in the tabular and linear models, which is the form that lines up term by term with §5. For an infinite class $\ln N$ is read as a metric entropy, $\lesssim SK\ln T$ for the tabular class $[0,1]^{\mathcal X\times\mathcal A}$ and $\lesssim d\ln(Bn)$ for $\{\langle\theta,\phi\rangle:\|\theta\|_2\le B\}$, with $\ln(1/\delta)$ entering additively beside it and the constant $12$ holding only in the exactly realizable finite case. In the tabular model that route is looser than Theorem 3.5 by $\sqrt K$, since it charges $SK\,C_{\mathrm{unif}}$ where the pointwise route charges $S\,C_{\mathrm{unif}}$.

### 3.3 Why uniform coverage cannot be dropped

**Proposition 3.4 (two actions).** Let $\mathcal X$ be a single context and $\mathcal A=\{1,2\}$. Action 1 has the deterministic reward $1/2$; action 2 has a Bernoulli reward with mean $1/2-\gamma$, $\gamma\in(0,1/4]$; the behavior policy logs action 2 with a small probability $p$, so that $N(2)\approx pn=:k$. Then: (i) in the analysis of the greedy rule one may take $b(1)=0$, valid because action 1 is noiseless, together with the Hoeffding bound $b(2)=\min\{1,\sqrt{\ln(4/\delta)/(2k)}\}$ at action 2; the pessimistic rule cannot use $b(1)=0$, since it must evaluate its penalty, and instead subtracts the computable $\Gamma=b^{\mathrm H}$ of Proposition 2.11, so that $\Gamma(2)=b(2)$ and $\Gamma(1)=\min\{1,\sqrt{\ln(4/\delta)/(2N(1))}\}$; (ii) if $\gamma\le1/(2\sqrt k)$, the event $\hat q(2)>1/2$ has probability at least an absolute constant $c_0>0$ (binomial anti-concentration), on which the greedy policy chooses action 2 and incurs regret $\gamma$; with $\gamma=1/(2\sqrt k)$, $\ \mathbb E[\Delta(\pi_{\hat q})]\ge c_0/(2\sqrt k)\asymp b(2)$; (iii) on the event of Proposition 2.11, which has probability at least $1-\delta$, the rule (LCB) chooses action 1 and incurs no regret whenever $\Gamma(1)<\gamma$, since then $\hat q(2)-\Gamma(2)\le q^*(2)=\tfrac12-\Delta<\tfrac12-\Gamma(1)=\hat q(1)-\Gamma(1)$; with $\gamma=1/(2\sqrt k)$ and $N(1)=T-k$ this holds as soon as $p<1/(1+2\ln(4/\delta))$.

Here $C^*=(1-p)^{-1}\approx1$ — the optimal action is covered as well as it could be — and the greedy rule still fails at the rate of the *un*covered action's uncertainty: the term $\mathbb E_x[\max_ab(x,a)]$ in Theorem 3.1 is not an artifact of the proof. Rashidinejad et al. (2021, Proposition 1) make the same point for the empirical best arm: it fails even when $C^*\approx1$.

### 3.4 Discussion and limitations

The greedy policy is governed by the least-covered action: Lemma 2.13 charges the error at whatever it *chooses*, and it chooses whatever the regression happened to overvalue. With tabular $\mathcal F$ and $N(x,a)=0$ its choice at $x$ is arbitrary; more data does not help unless coverage improves. This is the one-step form of the extrapolation error of fitted Q-iteration and the reason the offline literature does not use the plain plug-in rule without either uniform coverage (Theorem 3.2), a margin condition (§7.2), or pessimism (§4).

### 3.5 The greedy bound in the tabular and linear models

Theorem 3.1 is stated for an abstract quantifier; inserting the quantifiers of §2.4 turns it into rates, in the same two models §4.3–§4.4 use for pessimism. Write $\bar C_{\mathrm{unif}}:=\mathbb E_{x\sim\nu}\big[\max_a1/\mu(a\mid x)\big]$ — the largest $C^\pi_2$ over deterministic $\pi$, and the uniform analogue of $\bar C^*$ — and $N_{\min}(x):=\min_aN(x,a)$.

**Theorem 3.5 (tabular greedy).** Let $S,K<\infty$ and let $\hat q$ be the cell-wise empirical means, with an arbitrary value in $[0,1]$ where $N(x,a)=0$. Conditional on the design, with probability at least $1-\delta$,

$$
\Delta(\pi_{\hat q})\ \le\ \mathbb E_x\Big[\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,N(x,\pi^*(x))}}\Big\}\Big]+\mathbb E_x\Big[\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,N_{\min}(x)}}\Big\}\Big].
$$

If in addition $T\,\nu(x)\mu(a\mid x)\ge8\ln(SK/\delta')$ at every pair with $\nu(x)>0$, then with probability at least $1-\delta-\delta'$,

$$
\Delta(\pi_{\hat q})\ \le\ \sqrt{\frac{S\ln(2SK/\delta)}{T}}\Big(\sqrt{\bar C^*}+\sqrt{\bar C_{\mathrm{unif}}}\Big)\ \le\ 2\sqrt{\frac{S\,C_{\mathrm{unif}}\ln(2SK/\delta)}{T}} .
$$

*Proof.* Theorem 3.1 with $b=b^{\mathrm H}$ (Proposition 2.11), valid conditional on the design. The map $t\mapsto\min\{1,\sqrt{c/2t}\}$ is nonincreasing, so $\max_ab^{\mathrm H}(x,a)$ is attained at the smallest count and equals the second term, an empty cell giving the value $1$. For the second display the counts are binomial and the multiplicative Chernoff bound, union-bounded over the at most $SK$ pairs with $\nu(x)>0$, gives $N(x,a)\ge T\nu(x)\mu(a\mid x)/2$ at all of them with probability at least $1-\delta'$; discard the truncations ($\min\{1,u\}\le u$) and apply Cauchy–Schwarz over the at most $S$ contexts to each term separately, exactly as in Theorem 4.2, which yields $\sqrt{S\bar C^*}$ from the first and $\sqrt{S\bar C_{\mathrm{unif}}}$ from the second. Finally $\bar C^*\le\bar C_{\mathrm{unif}}\le C_{\mathrm{unif}}$. $\square$

**Greedy against pessimism, in one symbol.** Theorem 3.5 and Theorem 4.2 differ only in $C_{\mathrm{unif}}$ versus $C^*$, and their count conditions differ in the same way: greedy needs all $SK$ cells filled, pessimism only the $S$ cells of $\pi^*$. Before either coefficient is relaxed the comparison is sharper still — pessimism pays $\sqrt{\bar C^*}+\sqrt{\bar C^*}$ where greedy pays $\sqrt{\bar C^*}+\sqrt{\bar C_{\mathrm{unif}}}$ — so the penalty in (LCB) buys a second copy of the first term in place of a charge at the worst-covered action. The ratio of the relaxed bounds is exactly $\sqrt{C_{\mathrm{unif}}/C^*}$, that of the unrelaxed ones $\tfrac12(1+\sqrt{\bar C_{\mathrm{unif}}/\bar C^*})$.

Two consequences. First, $\bar C_{\mathrm{unif}}\ge K$ always, since the $K$ probabilities $\mu(\cdot\mid x)$ sum to one and their smallest is therefore at most $1/K$, with equality exactly when $\mu$ is uniform at every context: the number of actions never appears in Theorem 3.1, but it enters here at rate $\sqrt K$ whatever the behavior policy, and the bound is never below $\sqrt{SK\ln(2SK/\delta)/T}$. Second, the truncation at $1$ makes the second term contribute $\nu(x)$ outright at any context with an unvisited cell, so the bound is trivial until all $SK$ cells have been visited — $T\gtrsim SK\ln(SK)$ for uniform $\nu$ and $\mu$, by the coupon-collector estimate $\mathbb P(N(x,a)=0)=(1-\tfrac1{SK})^T$ — against the $S$ cells pessimism needs. The two rules are thus optimized by opposite logging policies: uniform logging gives $\bar C^*=\bar C_{\mathrm{unif}}=K$ and makes *both* bounds $2\sqrt{SK\ln(2SK/\delta)/T}$, the best case for greedy, while logging concentrated on $\pi^*$ drives $\bar C^*\to1$ and $C_{\mathrm{unif}}\to\infty$. The regime in which pessimism attains its fastest rate is the one in which the greedy bound says nothing at all — which is Proposition 3.4 again, in rate form.

**Theorem 3.6 (linear greedy).** Let $\mathcal F$ be the realizable linear class of Proposition 2.12 and $\hat q=\langle\hat\theta,\phi\rangle$ its ridge estimate. With probability at least $1-\delta$,

$$
\Delta(\pi_{\hat q})\ \le\ \beta_\delta\Big(\mathbb E_x\big\|\phi(x,\pi^*(x))\big\|_{\Lambda^{-1}}+\mathbb E_x\Big[\max_a\big\|\phi(x,a)\big\|_{\Lambda^{-1}}\Big]\Big),
$$

and $\Delta(\pi_{\hat q})\le2\beta_\delta/\sqrt{\kappa T}=\tilde O\big(\sqrt{d/(\kappa T)}\big)$ if $\Lambda\succeq\kappa nI$.
*Proof.* Theorem 3.1 with Proposition 2.12; then $\|\phi(x,a)\|_{\Lambda^{-1}}\le\|\phi(x,a)\|_2/\sqrt{\kappa T}\le1/\sqrt{\kappa T}$ at every pair, and $\beta_\delta=\tilde O(\sqrt d)$. $\square$

Against Theorem 4.3 the structure is the same: pessimism pays $2\beta_\delta\mathbb E_x\|\phi(x,\pi^*(x))\|_{\Lambda^{-1}}$ and greedy the same quantity with one of the two copies replaced by the maximum over actions. Neither $S$ nor $K$ appears — $d$ has taken over from $SK$, and $\beta_\delta$ carries no union bound over pairs — but the requirement has not gone away: $\kappa>0$ asks the logged features to span every direction some $\phi(x,a)$ points in, the linear analogue of $\mu_{\min}>0$. Under the uniform spectral condition the two bounds *coincide* at $\tilde O(\sqrt{d/(\kappa T)})$, so stating the linear guarantees in that form erases the very distinction the tabular pair makes visible; $1/\kappa$ is a uniform quantity, and unlike the scale-free ratio $\lambda_{\max}(\Sigma_\mu^{-1/2}\Sigma_\pi\Sigma_\mu^{-1/2})$ of Lemma 2.9(c) it compares $\Sigma_\mu$ to the identity and so inherits the context distribution: under $\phi=e_{xa}$, $1/\kappa=\max_{x,a}1/(\nu(x)\mu(a\mid x))\ge C_{\mathrm{unif}}$, and Theorem 3.6 returns $\tilde O(SK/\sqrt T)$, a factor $\sqrt{SK}$ worse than Theorem 3.5. The linear form earns its keep only when $d\ll SK$.

**Provenance of Theorems 4.5 and 4.6.** Both are routine instantiations, derived here rather than quoted. Theorem 3.5 is Theorem 3.1 with the Hoeffding quantifier and the Chernoff-plus-Cauchy–Schwarz step of Theorem 4.2, and its $\sqrt{C_{\mathrm{unif}}}$ shape is the coverage-language form of Murphy's (2005) $\mu_{\min}^{-1/2}$ constant; Theorem 3.6 is the greedy counterpart of Theorem 4.3, with the same proof. That $\bar C_{\mathrm{unif}}\ge K$ is immediate, but worth recording: it is what turns the action-free statement of Theorem 3.1 into a $\sqrt K$ cost.

## 4. The pessimistic plug-in policy

### 4.1 Method

Fit $\hat q$ by (1) on all $T$ triples, take a computable penalty $\Gamma$ that is a $\delta$-uncertainty quantifier for it (Proposition 2.11 or 3.12), and output the rule (LCB), $\hat\pi(x)\in\arg\max_a\hat q(x,a)-\Gamma(x,a)$. Computable quantifiers are where the structure of $\mathcal F$ enters; propensities are still not used.

### 4.2 The main theorem

**Theorem 4.1 (pessimism: single-policy coverage suffices).** Let $\Gamma$ be a $\delta$-uncertainty quantifier for $\hat q$ and $\hat\pi$ the rule (LCB). With probability at least $1-\delta$,

$$
\boxed{\ \Delta(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\big[\Gamma(x,\pi^*(x))\big]\ } \tag{4}
$$

*Proof.* Lemma 2.14 averaged over $x\sim\nu$. $\square$

Only the optimal policy's uncertainty appears: a poorly covered action costs nothing unless $\pi^*$ uses it. This is the horizon-one case of Theorem 4.2 of Jin, Yang & Wang (2021), $\Delta(\hat\pi)\le2\sum_h\mathbb E_{\pi^*}[\Gamma_h]$. Explicit rates follow by inserting the quantifiers of §2.4.

### 4.3 Tabular classes

**Theorem 4.2 (tabular LCB).** Let $S,K<\infty$, $\hat q$ the cell-wise empirical means, and $\hat\pi$ the rule (LCB) with $\Gamma=b^{\mathrm H}$. Conditional on the design, with probability at least $1-\delta$,

$$
\Delta(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\Big[\min\Big\{1,\sqrt{\tfrac{\ln(2SK/\delta)}{2\,N(x,\pi^*(x))}}\Big\}\Big].
$$

Moreover, writing $\mu_x:=\mu(\pi^*(x)\mid x)$ and using $N(x,a)\sim\mathrm{Bin}(T,\nu(x)\mu(a\mid x))$ with the multiplicative Chernoff bound $\mathbb P(\mathrm{Bin}(T,q)\le nq/2)\le e^{-nq/8}$: if $T\,\nu(x)\mu_x\ge8\ln(S/\delta')$ for all $x$, then with probability at least $1-\delta-\delta'$,

$$
\Delta(\hat\pi)\ \le\ 2\sum_x\nu(x)\sqrt{\frac{\ln(2SK/\delta)}{T\,\nu(x)\mu_x}}\ \le\ 2\sqrt{\frac{S\,\bar C^*\ln(2SK/\delta)}{T}}\ \le\ 2\sqrt{\frac{S\,C^*\ln(2SK/\delta)}{T}},
$$

using $\sum_x\sqrt{\nu(x)/\mu_x}\le\sqrt{S\sum_x\nu(x)/\mu_x}=\sqrt{S\bar C^*}$ (Cauchy–Schwarz) and $\bar C^*\le C^*$.
*Proof.* Theorem 4.1 with Proposition 2.11; the count condition and union bound over $\mathcal X$ give $N(x,\pi^*(x))\ge T\nu(x)\mu_x/2$ for all $x$. $\square$

Rashidinejad et al. (2021) remove the count condition and prove the sharper $\tilde O\big(\sqrt{S(C^*-1)/T}+S/T\big)$ (their Theorem 4, with $C^*=\max_x1/\mu_x$; stated and proved in full in §4.7), matched by an information-theoretic lower bound of the same form (their Theorem 5): when $C^*=1$ the data are "expert" and the rate is $1/T$, and for $C^*>1$ it is $\sqrt{SC^*/T}$ up to the $-1$. Their LCB is (LCB) with $b\asymp\sqrt{\ln(S/\delta)/N(x,a)}$.

### 4.4 Linear classes

**Theorem 4.3 (linear LCB).** Let $\mathcal F$ be the realizable linear class of Proposition 2.12 and $\hat\pi$ the rule (LCB) with its quantifier. With probability at least $1-\delta$,

$$
\Delta(\hat\pi)\ \le\ 2\beta_\delta\,\mathbb E_{x\sim\nu}\big\|\phi(x,\pi^*(x))\big\|_{\Lambda^{-1}},\qquad\text{and}\qquad\Delta(\hat\pi)\le\frac{2\beta_\delta}{\sqrt{\kappa T}}=\tilde O\Big(\sqrt{\tfrac d{\kappa T}}\Big)\ \text{ if }\Lambda\succeq\kappa nI.
$$

*Proof.* Theorem 4.1 with Proposition 2.12. $\square$

This is the pessimistic value iteration of Jin, Yang & Wang (2021) at horizon one, whose radius $\beta=c\,dH\sqrt{\zeta}$ carries factors of $d$ and $H$ from covering the next-state value class that are absent for bandits; they show the $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ form is minimax optimal for their instance class. Li, Ma & Srebro (2022) organize the linear case differently, at the level of policies (fixed design, OLS estimate $\hat\theta$, whitening matrix $\Sigma_D:=\frac1T\Phi^\top\Phi$): a confidence set $\Theta\ni\theta^*$ induces the pessimistic value $\hat V(\pi):=\inf_{\theta\in\Theta}\mathbb E_x[\phi(x,\pi(x))^\top\theta]$ and the rule $\hat\pi_\Theta:=\arg\max_\pi\hat V(\pi)$; if $\theta^*\in\Theta$ and $\sup_{\theta\in\Theta}\|\theta-\theta^*\|\le\beta$ with probability $1-\delta$ ("pessimism-validity"), then $\Delta(\hat\pi_\Theta)\le\beta\,\|\mathbb E_x\phi(x,\pi^*(x))\|_*$ in the dual norm (their Proposition 1 — the decomposition (4)–(5) of §6.1 with a policy-level width). With the $\ell_p$ sets $\Theta_p:=\{\theta:\|\Sigma_D^{1/2}(\theta-\hat\theta)\|_p\le\beta\}$, $\beta=d^{1/p}\sqrt{8\ln(d/\delta)/T}$, their Theorem 1 gives $\Delta(\hat\pi_p)\le d^{1/p}\sqrt{8\ln(d/\delta)/T}\,\big\|\Sigma_D^{-1/2}\mathbb E_x\phi(x,\pi^*(x))\big\|_q$, $1/p+1/q=1$. Three consequences for this page. (i) $\hat\pi_2$ *is* the version-space rule of §4.5: in a linear bandit the empirical Bellman error is $\|\Sigma_D^{1/2}(\theta-\hat\theta)\|_2^2$, so BCP's version space is an $\ell_2$ ball. (ii) $\hat\pi_\infty$ (PUNC) reduces to the tabular LCB of Theorem 4.2 when $\phi(x,a)=e_{xa}$ (their Corollary 1 recovers $\sqrt{SC^*\ln(SK/\delta)/T}$), but it is *not* the pointwise linear rule of Theorem 4.3: that rule (PEVI) subtracts $\beta\|\Sigma_D^{-1/2}\phi\|_2$ at every context, which amounts to an $\ell_2$ set enlarged context by context; its guarantee $\sqrt{d^2/T}\,\mathbb E_x\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_2$ is looser by a factor $d$ and by Jensen ($\|\Sigma_D^{-1/2}\mathbb E_x\phi\|_2\le\mathbb E_x\|\Sigma_D^{-1/2}\phi\|_2$), but holds for every test distribution at once, whereas the policy-level rules are tuned to one $\nu$. (iii) Their Theorem 2 is a minimax lower bound $\Omega(d^{1/p}\Lambda/\sqrt T)$ over the classes $\mathrm{CB}_q(\Lambda)=\{\|\Sigma_D^{-1/2}\mathbb E_x\phi(x,\pi^*(x))\|_q\le\Lambda\}$, so $\hat\pi_p$ is minimax over $\mathrm{CB}_q$, and since $\|v\|_1\le d^{1-1/q}\|v\|_q$ the $\ell_\infty$ rule is optimal over *every* class simultaneously — "adaptively minimax optimal" — and strictly dominates $\hat\pi_2$. The coverage quantity is thus the norm of the *averaged* whitened feature of $\pi^*$, a refinement of $C^*$ and of $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ that can be much smaller than either.

### 4.5 General finite classes: pessimism through the version space

Theorem 2.6 provides no pointwise quantifier, so pessimism is applied to policy values. **Method:** split the data; on $\mathcal D^{\mathrm{reg}}$ fit $\hat q$ and form the version space $\mathcal F_\varepsilon$ of Corollary 2.7 with $\varepsilon=4L_{\mathrm{reg}}$, $L_{\mathrm{reg}}:=\ln(|\mathcal F|/\delta)/T_{\mathrm{reg}}$; on $\mathcal D^{\mathrm{eval}}$ define empirical values $\widehat J_f(\pi):=\frac1m\sum_{x\in \mathcal D^{\mathrm{eval}}}f(x,\pi(x))$ for $f\in\mathcal F$ and $\pi\in\Pi_{\mathcal F}$ (at most $|\mathcal F|$ policies, containing $\pi^*=\pi_{q^*}$ under realizability); output

$$
\hat\pi:=\arg\max_{\pi\in\Pi_{\mathcal F}}\ \min_{f\in\mathcal F_\varepsilon}\widehat J_f(\pi). \tag{VS}
$$

**Theorem 4.4 (version-space pessimism).** Under (A3)–(A4), with probability at least $1-2\delta$,

$$
\Delta(\hat\pi)\ \le\ \sqrt{\frac{20\,C_{\mathcal F}(\pi^*)\ln(|\mathcal F|/\delta)}{T_{\mathrm{reg}}}}+2\sqrt{\frac{\ln(2|\mathcal F|^2/\delta)}{2m}} .
$$

*Proof.* Let $J_f(\pi):=\mathbb E_{x\sim\nu}[f(x,\pi(x))]$, so $J_{q^*}=J$, and $\varepsilon_m:=\sqrt{\ln(2|\mathcal F|^2/\delta)/(2m)}$. By Theorem 2.1 with a union bound over the at most $|\mathcal F|^2$ pairs $(f,\pi)\in\mathcal F\times\Pi_{\mathcal F}$, with probability at least $1-\delta$ over $\mathcal D^{\mathrm{eval}}$, $|\widehat J_f(\pi)-J_f(\pi)|\le\varepsilon_m$ for all pairs. On this event and that of Corollary 2.7,

$$
J(\hat\pi)=J_{q^*}(\hat\pi)\ \ge\ \widehat J_{q^*}(\hat\pi)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}\widehat J_f(\hat\pi)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}\widehat J_f(\pi^*)-\varepsilon_m\ \ge\ \min_{f\in\mathcal F_\varepsilon}J_f(\pi^*)-2\varepsilon_m ,
$$

using $q^*\in\mathcal F_\varepsilon$, then the choice of $\hat\pi$ with $\pi^*\in\Pi_{\mathcal F}$. Hence $\Delta(\hat\pi)\le\max_{f\in\mathcal F_\varepsilon}[J_{q^*}(\pi^*)-J_f(\pi^*)]+2\varepsilon_m\le\max_{f\in\mathcal F_\varepsilon}\|f-q^*\|_{L_1(\nu\times\pi^*)}+2\varepsilon_m\le\sqrt{C_{\mathcal F}(\pi^*)}\max_{f\in\mathcal F_\varepsilon}\|f-q^*\|_{\nu\times\mu}+2\varepsilon_m$ by Lemma 2.9(b), and Corollary 2.7(ii) bounds the maximum by $\sqrt{20L_{\mathrm{reg}}}$. $\square$

(VS) is Eq. (3.2) of Xie et al. (2021) at horizon one — $\arg\max_\pi\min_{f\in\mathcal F_{\pi,\varepsilon}}f(s_0,\pi)$ over the version space of functions with small Bellman loss — where the Bellman loss is the squared loss, their completeness assumption is vacuous, and their coverage coefficient is $C_{\mathcal F}$; the coverage is class-dependent and can be far below $C^*$ for structured classes. The restriction to $\Pi_{\mathcal F}$ is what makes the empirical values uniformly controllable; over all of $\mathcal A^{\mathcal X}$ the supremum of an empirical average aligns with the sample and does not concentrate.

### 4.6 Neural classes, and what pessimism buys

NeuraLCB (Nguyen-Tang et al. 2022; `nguyen-tang2022Offline`) uses (LCB) with $\Gamma(x,a)=\beta\|\nabla_\theta f(x,a;\hat\theta)\|_{\Lambda^{-1}}$ for a neural-tangent-kernel Gram matrix, trains $\hat q$ by stochastic gradient descent in an online manner, and obtains regret $\tilde O(\sqrt{\tilde d/T})$ up to a coverage constant for $\pi^*$, $\tilde d$ the effective dimension, under a condition milder than single-policy concentrability (cited, not re-derived). Jeunen & Goethals (2021) report the practical version — regression reward models with a lower confidence bound — for recommendation.

*What pessimism buys.* Comparing Theorems 4.2 and 5.1: greedy needs uniform coverage, $C_{\mathrm{unif}}<\infty$, while pessimism needs single-policy coverage, $C^*<\infty$ (or its average or class-dependent forms). This is not a matter of worst-case rates. In the context-free case with counts $N(a)$, Xiao et al. (2021; `xiao2021Optimality`) define confidence-adjusted index rules $\arg\max_a\hat q(a)+\alpha/\sqrt{N(a)}$ — optimistic, greedy or pessimistic according to the sign of $\alpha$ — and prove that *every* such rule is minimax optimal, matching an $\Omega(1/\sqrt{\min_aN(a)})$ lower bound on simple regret; that instance-dependent optimality in the sense of online bandits cannot be achieved by any batch algorithm; and that under a *weighted-minimax* criterion, which weights each instance by the inherent difficulty of predicting its optimal value, the pessimistic rule is the one that is optimal. Proposition 3.4 is the instance behind the criterion: pessimism's bound depends only on the optimal action's uncertainty, the others' on the largest.

### 4.7 Rashidinejad et al. (2021), Theorem 4, in detail

Theorem 4.2 needs a count condition and pays $C^*$. Their Theorem 4 removes the condition and replaces $C^*$ by $C^*-1$, so that near-expert data give a $1/T$ rate rather than a $1/\sqrt T$ one; it is the sharpest tabular guarantee for (LCB). *Notation:* their $s$, $\rho$, $N$, $N(s,a)$, $\mu(s,a)$, $r$, $\hat r$, $b$ are our $x$, $\nu$, $T$, $N(x,a)$, $d^\mu(x,a)$, $q^*$, $\hat q$, $\Gamma$; their coverage $\max_s\rho(s)/\mu(s,\pi^*(s))\le C^*$ is ours exactly, since $\nu(x)/d^\mu(x,\pi^*(x))=1/\mu(\pi^*(x)\mid x)$.

**The algorithm.** (LCB) with the conventions $\hat q(x,a)=0,\ \Gamma(x,a)=1$ where $N(x,a)=0$, and $\hat q$ the cell mean with $\Gamma(x,a)=\sqrt{2000\ln(2SK/\delta)/N(x,a)}$ elsewhere. Two features carry the proof: an unvisited pair is assigned lower confidence value $0-1=-1$, so it is never selected unless every action at that context is unvisited; and the constant $2000$ is deliberately generous, which is what makes the failure probabilities polynomially small in $T$ rather than merely below $\delta$. Throughout, $\delta=1/T$ and $L:=2000\ln(2SKn)$, so $\Gamma=\sqrt{L/N(x,a)}$ on visited cells.

**Theorem 4.5 (Rashidinejad et al. 2021, Theorem 4).** For a tabular contextual bandit with $S\ge2$ and $\max_{x:\nu(x)>0}1/\mu(\pi^*(x)\mid x)\le C^*$, the rule above with $\delta=1/T$ satisfies

$$
\mathbb E_{\mathcal D}\big[\Delta(\hat\pi)\big]\ \lesssim\ \min\Big\{1,\ \tilde O\Big(\sqrt{\tfrac{S(C^*-1)}{T}}+\tfrac Sn\Big)\Big\} .
$$

The guarantee is on *expected* suboptimality, not high-probability as in Theorem 4.1. What the proof establishes is the same bound for an arbitrary deterministic comparator $\pi$ with $C^\pi$ in place of $C^*$; Theorem 4.5 is the case $\pi=\pi^*$.

**Lemma 4.6 (two-sample Hoeffding; their Lemma 13).** For independent i.i.d. samples $X_1,\dots,X_p$ and $Y_1,\dots,Y_q$ in $[0,1]$ and any $\varepsilon$ with $\varepsilon+\mathbb EY-\mathbb EX\ge0$,
$\ \mathbb P\big(\frac1p\sum_iX_i-\frac1q\sum_jY_j>\varepsilon\big)\le\exp\big(-\frac{2pq(\varepsilon+\mathbb EY-\mathbb EX)^2}{p+q}\big)$.

**Lemma 4.7 (binomial inverse moments; their Lemma 14).** For $m\sim\mathrm{Bin}(N,p)$ and every $k\ge0$ there is $c_k$ with $\mathbb E[(m\vee1)^{-k}]\le c_k(Np)^{-k}$.

Lemma 4.7 at $k=1/2$ is the step that converts a bound involving the *random* $1/\sqrt{N(x,a)}$ into one involving $1/\sqrt{T\,d^\mu(x,a)}$; it is what makes Theorem 4.2's count condition unnecessary.

**Proof.** On the good event $\mathcal E:=\{|q^*(x,a)-\hat q(x,a)|\le\Gamma(x,a)\ \forall(x,a)\}$, write $\ell(x):=q^*(x,\pi(x))-q^*(x,\hat\pi(x))$ and split on whether the comparator's action was logged:

$$
\mathbb E\big[J(\pi)-J(\hat\pi)\big]=T_1+T_2+T_3,\qquad
T_i=\mathbb E\Big[\textstyle\sum_x\nu(x)\ell(x)\cdot\mathbb 1_i\Big],
$$

with $\mathbb 1_1=\mathbb 1\{N(x,\pi(x))=0\}$, $\mathbb 1_2=\mathbb 1\{N(x,\pi(x))\ge1\}\mathbb 1\{\mathcal E\}$, $\mathbb 1_3=\mathbb 1\{N(x,\pi(x))\ge1\}\mathbb 1\{\mathcal E^c\}$. The claim is

$$
T_1\le\frac{4SC^\pi}{9T},\qquad T_2\lesssim\frac{SC^\pi L}{T}+\sqrt{\frac{S(C^\pi-1)L}{T}}+\frac1{T^9},\qquad T_3\le\frac1T .
$$

*Missing mass, $T_1$.* Bound $\ell\le1$; $N(x,\pi(x))\sim\mathrm{Bin}(T,d^\mu(x,\pi(x)))$, so $T_1\le\sum_x\nu(x)(1-d^\mu(x,\pi(x)))^T$. Coverage gives $\nu(x)\le C^\pi d^\mu(x,\pi(x))$, and $\max_{z\in[0,1]}z(1-z)^T\le4/(9T)$, whence $T_1\le4SC^\pi/(9T)$. This is the price of contexts where the comparator's action is never logged — no algorithm can do anything there — and it is the source of the $S/T$ term.

*Bad event, $T_3$.* $T_3\le\mathbb P(\mathcal E^c)$. At an unvisited pair $\mathcal E$ holds by construction ($\hat q-\Gamma=-1\le q^*\le1=\hat q+\Gamma$). At a visited one, Hoeffding conditional on the count gives width $\sqrt{\ln(2SK/\delta)/2N(x,a)}$, which $\Gamma$ exceeds by $\sqrt{4000}$; the bound holds for every value of the count, hence unconditionally, and a union bound over $SK$ pairs gives $\mathbb P(\mathcal E^c)\le\delta=1/T$.

*Partition for $T_2$.* With $\bar\mu(x):=\sum_{a\ne\pi(x)}d^\mu(x,a)$ the logged mass off the comparator's action, split $\mathcal X_1:=\{\nu(x)<2C^\pi L/T\}$; $\mathcal X_2:=\{\nu(x)\ge2C^\pi L/T,\ d^\mu(x,\pi(x))\ge10\bar\mu(x)\}$; $\mathcal X_3$ the rest. Light contexts carry too little mass ($T_{2,1}\le\sum_{\mathcal X_1}\nu(x)<2SC^\pi L/T$); at the well-covered contexts of $\mathcal X_2$ the comparator's action is logged far more often than all others together, and two-sample Hoeffding (Lemma 4.6) plus $d^\mu(x,\pi(x))\ge\nu(x)/C^\pi\ge2L/T$ off $\mathcal X_1$ gives $T_{2,2}\lesssim T^{-9}$ — the rule essentially never errs there. The heavy, genuinely ambiguous contexts of $\mathcal X_3$ produce the rate.

*The ambiguous contexts, $T_{2,3}$.* On $\mathcal E$ the pessimism argument of Lemma 2.14 applies verbatim, $\ell(x)\le2\Gamma(x,\pi(x))$, so $T_{2,3}\le2\sqrt L\sum_{\mathcal X_3}\nu(x)\,\mathbb E[(N(x,\pi(x))\vee1)^{-1/2}]$. Lemma 4.7 bounds each expectation by $c/\sqrt{T\,d^\mu(x,\pi(x))}$, and coverage in the form $\nu(x)/\sqrt{d^\mu(x,\pi(x))}\le\sqrt{C^\pi\nu(x)}$ followed by Cauchy–Schwarz over the at most $S$ terms gives $T_{2,3}\lesssim\sqrt{C^\pi L/T}\cdot\sqrt{S\sum_{\mathcal X_3}\nu(x)}$. **The $-1$ enters in one inequality:**

$$
\sum_{x\in\mathcal X_3}\nu(x)\ \le\ \min\{1,\ 10(C^\pi-1)\} .
$$

Indeed coverage and the definition of $\mathcal X_3$ give $\sum_{\mathcal X_3}\nu(x)\le C^\pi\sum_{\mathcal X_3}d^\mu(x,\pi(x))\le10C^\pi\sum_{\mathcal X_3}\bar\mu(x)$, while $\sum_xd^\mu(x,\pi(x))\ge1/C^\pi$ forces $\sum_x\bar\mu(x)\le1-1/C^\pi$; multiplying gives $10C^\pi(1-1/C^\pi)=10(C^\pi-1)$. Substituting and splitting on $C^\pi\ge2$ or not yields $T_{2,3}\lesssim\sqrt{S(C^\pi-1)L/T}$. Adding the three parts and using $J(\pi)-J(\hat\pi)\le1$ gives the theorem. $\square$

**What the proof turns on.** The $C^*-1$ is not a refinement of the concentration; it is the mass bound above. A behavior policy with $C^\pi$ close to $1$ puts almost all of its mass on the comparator's action, so the set of contexts where any confusion is possible is itself small, and $\sqrt{S\cdot\text{(that mass)}/T}$ is what appears. At $C^*=1$ the data are expert, $\mathcal X_3$ is empty, and only the $S/T$ missing-mass term survives.

**A constant that does not follow.** At the $\mathcal X_2$ step the paper asserts $N(x,\pi(x))\ge4\,N(x,a)$ for the competing action; the properties they invoke give only $5/2$. The argument is written here with $5/2$, hence $\varepsilon\ge\tfrac13\sqrt{L/N(x,a)}$ in place of $\tfrac12\sqrt{L/N(x,a)}$ and an exponent $L/9$ in place of $L/4$. Nothing downstream changes: $\exp(-L/9)=(2SKn)^{-2000/9}$ is still far below $1/T^{10}$. Flagged in §8.3.

## 5. Evaluating policies with the reward model: the direct method

The reward model also yields a value estimate for *any* policy. This is not how a value-based learner chooses its policy (§1.3), but it is what the class-restricted variants of §6 maximize, it is the value-based half of the doubly robust estimator, and its error bound exhibits the coverage coefficients in their simplest form.

### 5.1 Method

Split $\mathcal D$ into $\mathcal D^{\mathrm{reg}}$ ($T_{\mathrm{reg}}$ triples) and $\mathcal D^{\mathrm{eval}}$ ($m$ triples); fit $\hat q$ on $\mathcal D^{\mathrm{reg}}$ by (1); for a target policy $\pi$,

$$
\widehat J^{\mathrm{DM}}(\pi):=\frac1m\sum_{x\in \mathcal D^{\mathrm{eval}}}\hat q(x,\pi). \tag{DM}
$$

Only the evaluation contexts are consumed. The split makes $\hat q$ independent of them; $K$-fold cross-fitting recovers the full sample at the price of a constant (Chernozhukov et al. 2018; Athey & Wager 2021). The name is due to Dudík, Langford & Li (2011): "the first, which we call the *direct method* (DM), estimates the reward function from given data and uses this estimate in place of actual reward to evaluate the policy value."

### 5.2 Analysis

Throughout, $\hat q$ is fixed (we condition on $\mathcal D^{\mathrm{reg}}$); $g:=\hat q-q^*$.

**Lemma 5.1 (bias–variance decomposition).** Conditional on $\hat q$,

$$
\mathbb E\big[\widehat J^{\mathrm{DM}}(\pi)\big]-J(\pi)=\mathbb E_{d^\pi}[g],\qquad
\mathrm{Var}\big(\widehat J^{\mathrm{DM}}(\pi)\big)=\frac1m\,\mathrm{Var}_{x\sim\nu}\big(\hat q(x,\pi)\big)\le\frac1{4m}.
$$

*Proof.* The summands $\hat q(x,\pi)$ are i.i.d. with mean $\mathbb E_{d^\pi}[\hat q]$, while $J(\pi)=\mathbb E_{d^\pi}[q^*]$; each summand lies in $[0,1]$. $\square$

The bias is the model error averaged under the *target* policy; the variance is that of a bounded sample mean, with no dependence on $\mu$, $\pi$ or the reward noise — the mirror image of importance weighting (zero bias, variance driven by $\pi/\mu$). These are Sections 3–4 of Dudík, Langford & Li (2011).

**Theorem 5.2 (direct-method error).** Fix $\pi$ and $\hat q$, and let $C(\pi):=\min\{C^\pi_2,C_{\mathcal F}(\pi)\}$. With probability at least $1-\delta$ over $\mathcal D^{\mathrm{eval}}$,

$$
\big|\widehat J^{\mathrm{DM}}(\pi)-J(\pi)\big|\ \le\ \sqrt{C(\pi)}\;\|\hat q-q^*\|_{\nu\times\mu}+\sqrt{\frac{\ln(2/\delta)}{2m}} .
$$

For a finite class $\Pi$ the statement holds simultaneously for all $\pi\in\Pi$ with $\ln(2|\Pi|/\delta)$ in the second term only.
*Proof.* Triangle inequality around $\mathbb E[\widehat J^{\mathrm{DM}}(\pi)]$: the sampling term is Theorem 2.1 for $m$ summands in $[0,1]$; the bias term is Lemma 5.1 followed by Lemma 2.9(a) and (b). The bias term is a deterministic function of $\hat q$, so a union bound over $\Pi$ touches only the sampling term. $\square$

**Corollary 5.3 (rate under realizability).** Under (A3)–(A4), for even $T$ ($T_{\mathrm{reg}}=m=T/2$), with probability at least $1-2\delta$, $\ \big|\widehat J^{\mathrm{DM}}(\pi)-J(\pi)\big|\le\sqrt{24\,C(\pi)\ln(|\mathcal F|/\delta)/T}+\sqrt{\ln(2/\delta)/T}$.
*Proof.* Theorem 5.2 with (3) on $\mathcal D^{\mathrm{reg}}$. $\square$

**Proposition 5.4 (misspecification floor).** If $q^*\notin\mathcal F$, then $\|\hat q-q^*\|_{\nu\times\mu}\ge\mathrm{dist}(q^*,\mathcal F):=\inf_{f\in\mathcal F}\|f-q^*\|_{\nu\times\mu}>0$ for every $T$, and the bias of (DM) does not vanish: its limit is $\mathbb E_{d^\pi}[f_{\mathcal F}-q^*]$ for the $L_2(d^\mu)$-projection $f_{\mathcal F}$ of $q^*$ onto $\mathcal F$ (unique when $\mathcal F$ is convex) — a quantity that depends on $\pi$, is bounded only by $\sqrt{C(\pi)}\,\mathrm{dist}(q^*,\mathcal F)$, and is not identifiable from $\mathcal D$ without propensities.
*Proof.* The lower bound is the definition of the distance; by Lemma 2.4 the population minimizer is the projection and the empirical minimizer converges to it over the finite class; Lemma 5.1 identifies the limiting bias and Lemma 2.9 bounds it. $\square$

Three consequences. (i) Under realizability the direct method matches the $\sqrt{\text{coverage}/T}$ form of importance weighting with $\ln N$ in place of the weight range, and its bias term is uniform over *all* target policies at no extra cost — $\ln N$ is paid once, in the regression. (ii) Through $C_{\mathcal F}$ the coverage is class-dependent and can be finite where importance weights are unbounded (Lemma 2.9(c)). (iii) Without realizability there is no guarantee, and the bias is invisible to the regression loss; Wang, Agarwal & Dudík (2017) show that the minimax mean-squared-error lower bound for off-policy evaluation is matched by IPS and DR while DM carries no distribution-free guarantee, so in the assumption-free regime DM is a *component*, not an estimator of choice.

## 6. The alternative formulation: estimate, then select

### 6.1 Unrestricted class: the same object

**Proposition 6.1.** Let $\Pi=\mathcal A^{\mathcal X}$. (a) The greedy policy $\pi_{\hat q}$ maximizes $\widehat J^{\mathrm{DM}}$ over $\Pi$, and every maximizer agrees with $\pi_{\hat q}$ on the evaluation contexts; the population plug-in value $v_{\hat q}(\pi):=\mathbb E_{x\sim\nu}[\hat q(x,\pi)]$ is maximized by $\pi_{\hat q}$ for every $\nu$. (b) With a penalty $\Gamma$, the rule (LCB) maximizes the pessimistic plug-in values $\frac1m\sum_{x\in \mathcal D^{\mathrm{eval}}}[\hat q-\Gamma](x,\pi)$ and $\mathbb E_{x\sim\nu}[(\hat q-\Gamma)(x,\pi)]$ over $\Pi$ in the same sense. (c) Define the per-policy score $X_\pi:=v_{\hat q}(\pi)$ and width $W_\pi:=\mathbb E_{x\sim\nu}[\Gamma(x,\pi(x))]$ for deterministic $\pi$; on the event of the quantifier, $|X_\pi-J(\pi)|\le W_\pi$ for all $\pi$, the pessimistic selection rule $\arg\max_\pi X_\pi-W_\pi$ is (LCB), and its guarantee $W_{\pi^*}+W_{\pi^*}$ is (4).
*Proof.* For any $h:\mathcal X\times\mathcal A\to\mathbb R$ and any distribution over contexts, $\max_{\pi}\mathbb E[h(x,\pi)]=\mathbb E[\max_ah(x,a)]$, attained by the pointwise maximizer independently of the distribution; apply with $h=\hat q$ and $h=\hat q-\Gamma$. (c) integrates the quantifier along $a=\pi(x)$; then $J(\hat\pi)\ge X_{\hat\pi}-W_{\hat\pi}\ge X_{\pi^*}-W_{\pi^*}\ge J(\pi^*)-2W_{\pi^*}$. $\square$

So estimate-then-select with plug-in values is regression-then-greedy written differently: the selection objective is not computable (it involves $\nu$), but its maximizer is, because the maximization decouples across contexts. The per-policy widths do not scale with $|\Pi|=K^{|\mathcal X|}$ because uniformity was obtained over $\mathcal F$.

### 6.2 Summary of the relation

| Formulation | Policy | Needs | Coverage in the bound |
|---|---|---|---|
| regression → greedy (§3) | $\pi_{\hat q}$ | $q^*\in\mathcal F$ | uniform, $C_{\mathrm{unif}}$ |
| regression → (LCB) (§4) | $\arg\max_a\hat q-\Gamma$ | $q^*\in\mathcal F$, a computable quantifier $\Gamma$ | single-policy, $C^*$ / $\bar C^*$ / $\mathbb E_{\pi^*}\|\phi\|_{\Lambda^{-1}}$ |
| estimate-then-select, plug-in values, $\Pi=\mathcal A^{\mathcal X}$ (§6.1) | $=\pi_{\hat q}$, resp. $=$ (LCB) | same | same |
| estimate-then-select, plug-in values, $\Pi$ restricted (§6.2) | $\arg\max_{\pi\in\Pi}\widehat J^{\mathrm{DM}}$ | $q^*\in\mathcal F$ | whole class, $C_\Pi$ |
| policy-level pessimism, version space (§4.5, §6.2) | $\arg\max_\pi\min_{f\in\mathcal F_\varepsilon}\widehat J_f(\pi)$ | $q^*\in\mathcal F$ | class-dependent, $C_{\mathcal F}(\pi^*_\Pi)$ |
| estimate-then-select, DR scores (§6.2) | $\arg\max_{\pi\in\Pi}\widehat J^{\mathrm{DR}}$ | $\mu$ or $\hat\mu$, and $\hat q$ | whole class, via the DR variance |
| estimate-then-select, IPW scores (the route comparison in [[contextual-bandits-offline]]) | $\arg\max_{\pi\in\Pi}\widehat J^{\mathrm{IPW}}$ (MaxIPW / PES) | $\mu$ | $C^{\pi}_2$ over the class, resp. $C^{\pi^*_\Pi}_2$ |

## 7. Instance dependence and fast rates

### 7.1 Variance-aware quantifiers

**Proposition 7.1.** In the tabular setting, the rule (LCB) with $\Gamma=b^{\mathrm B}$, the Bernstein quantifier of Proposition 2.11 satisfies, conditional on the design and with probability at least $1-\delta$,

$$
\Delta(\hat\pi)\ \le\ 2\,\mathbb E_{x\sim\nu}\Big[\min\Big\{1,\sqrt{\tfrac{2\sigma^2(x,\pi^*(x))\ln(2SK/\delta)}{N(x,\pi^*(x))}}+\tfrac{2\ln(2SK/\delta)}{3\,N(x,\pi^*(x))}\Big\}\Big].
$$

*Proof.* Theorem 4.1 with Proposition 2.11. $\square$

The reward *variance* at $\pi^*$'s actions replaces the worst-case $1/4$; the unknown $\sigma^2$ is replaced by the sample variance through the empirical Bernstein inequality (Maurer & Pontil 2009) with the same form, and Yin & Wang (2021) give the rigorous tabular statement. This is the value-side counterpart of replacing second moments of importance weights by variances on the policy side (betting or freezing; [[ryu2025Improved]]).

### 7.2 Gap and margin conditions

Define the **action gap** at $x$ as $\mathrm{gap}(x):=q^*(x,\pi^*(x))-\max_{a\ne\pi^*(x)}q^*(x,a)\ge0$.

**Theorem 7.2 (hard gap: the plug-in rule at rate $1/T$).** If $\mathrm{gap}(x)\ge\epsilon>0$ for $\nu$-almost every $x$, then for any $f:\mathcal X\times\mathcal A\to[0,1]$, with $g:=f-q^*$,

$$
\Delta(\pi_f)\ \le\ \frac4\epsilon\,\mathbb E_x\Big[\max_ag(x,a)^2\Big]\ \le\ \frac{4\,C_{\mathrm{unif}}}{\epsilon}\,\|f-q^*\|^2_{\nu\times\mu}.
$$

*Proof.* Fix $x$, $a^*=\pi^*(x)$, $\hat a=\pi_f(x)$. If $\hat a=a^*$ the regret at $x$ is $0$. Otherwise it is at least $\epsilon$ and, by Lemma 2.13, at most $2\max_a|g(x,a)|$; hence $2\max_a|g(x,a)|\ge\epsilon$ and the regret at $x$ is at most $2\max_a|g(x,a)|\cdot\frac{2\max_a|g(x,a)|}{\epsilon}=\frac4\epsilon\max_ag(x,a)^2$. Average over $x$, then $\max_ag^2\le\sum_ag^2=\sum_a\mu(a|x)\frac{g(x,a)^2}{\mu(a|x)}\le C_{\mathrm{unif}}\sum_a\mu(a|x)g(x,a)^2$. $\square$

**Corollary 7.3.** Under (A3)–(A4) and a hard gap $\epsilon$, with probability at least $1-\delta$, $\ \Delta(\pi_{\hat q})\le48\,C_{\mathrm{unif}}\ln(|\mathcal F|/\delta)/(\epsilon\,T)$ — a $1/T$ rate for the *greedy* rule, without pessimism.
*Proof.* Theorem 7.2 with (3). $\square$

Theorem 7.2 is the hard-gap remark after Theorem 2.1 of Qian & Murphy (2011), $V(d_0)-V(d)\le4S[L(Q)-L(Q_0)]/\epsilon$ with $S=C_{\mathrm{unif}}$. Their Theorem 2.1 interpolates: under the margin condition $\mathbb P_x(\mathrm{gap}(x)\le\epsilon)\le C\epsilon^\alpha$ for all $\epsilon>0$, $\ \Delta(\pi_f)\le C'[\mathcal L(f)-\mathcal L(q^*)]^{(1+\alpha)/(2+\alpha)}$ with $C'=(2^{2+3\alpha}S^{1+\alpha}C)^{1/(2+\alpha)}$, so that $\alpha=0$ recovers Theorem 3.2 and $\alpha\to\infty$ recovers Theorem 7.2. Hu, Kallus & Uehara (2021) carry the mechanism — misranking requires errors larger than the gap — to infinite-horizon Markov decision processes and plug-in policies from fitted Q-iteration or Bellman-residual minimization: the regret rate is the exponentiation of the pointwise convergence rate of the $Q^*$ estimate, giving $O(1/T)$ in linear and $e^{-\Omega(T)}$ in tabular cases, the exponent depending on the noise level of the decision problem. On the policy route the analogous condition is a gap in *policy values*, $J(\pi^*_\Pi)-\max_{\pi\ne\pi^*_\Pi}J(\pi)$, which the same two-line argument converts into $O(W^2/\mathrm{gap})=O(1/T)$; but that gap is a property of the class, tiny or zero for large $\Pi$, whereas action gaps are a property of the problem. Fast rates are therefore natural on the value side and exceptional on the policy side.

## 8. Bibliographic notes, reading order and vault actions

### 8.1 Provenance (verified against the PDFs, 2026-09-17; Rashidinejad's Theorem 4 re-verified line by line 2026-09-19)

*Formulation.* "Value-based learning. Another simple algorithm is to first learn the $Q$ function and then use a greedy policy with respect to this estimated $Q$ function," with $\hat Q:=\arg\min_{f\in\mathcal Q}\sum_i(f(x_i,a_i)-r_i(a_i))^2$ — Brandfonbrener, Whitney, Ranganath & Bruna (ICML 2021), who contrast it with "policy-based learning. Importance weighted or 'inverse propensity weighted' policy optimization directly optimizes the policy to maximize an estimate of its value," define action-stability, show value-based objectives are action-stable and policy-based ones are not, and note that the doubly robust approach collapses to the value-based one with overparameterized models. The same regression-then-greedy rule is "the regression approach" of Beygelzimer & Langford (KDD 2009, §6.1: "probably the simplest approach is to regress on the reward $r_a$ given $x$ and $a$, and then choose according to the largest predicted reward"; Theorem 6.1, $\mathrm{reg}(\pi_f)\le2\sqrt{k\,\mathrm{reg}_r(f)}$, tight), the plug-in individualized-treatment rule of Murphy (JMLR 2005; bound restated as (3.1) in Qian & Murphy, Ann. Statist. 2011, whose Theorem 3.1 and remarks contain §7.2 and whose §3 toy example is the selection mismatch), and the algorithm of Rashidinejad, Zhu, Ma, Jiao & Russell (NeurIPS 2021): LCB with penalty $\sqrt{L/N(x,a)}$, single-policy concentrability $C^\pi$ (their Definition 1), Theorem 4 $\tilde O(\sqrt{S(C^*-1)/N}+S/N)$, Theorem 5 lower bound, Proposition 1 on the failure of the empirical best arm. Jin, Yang & Wang (ICML 2021) supply the $\xi$-uncertainty quantifier, PEVI ($\hat Q_h=\hat{\mathbb B}_h\hat V_{h+1}-\Gamma_h$, greedy), Theorem 3.2 ($\mathrm{SubOpt}\le2\sum_h\mathbb E_{\pi^*}[\Gamma_h]$), the linear bonus $\beta\|\phi\|_{\Lambda^{-1}}$ and minimax optimality. Xie, Cheng, Jiang, Mineiro & Agarwal (NeurIPS 2021) supply Eq. (3.2), $\arg\max_\pi\min_{f\in\mathcal F_{\pi,\varepsilon}}f(s_0,\pi)$, and Definition 1, $C(\nu;\mu,\mathcal F,\pi)=\max_f\|f-\mathcal T^\pi f\|^2_{2,\nu}/\|f-\mathcal T^\pi f\|^2_{2,\mu}\le\|\nu/\mu\|_\infty$. Xiao, Wu, Lattimore, Dai, Mei, Li, Szepesvári & Schuurmans (2021; `xiao2021Optimality`): confidence-adjusted index rules $\arg\max_i\hat\mu_i+\alpha/\sqrt{N_i}$, all minimax optimal, $\Omega(1/\sqrt{\min_i N_i})$ lower bound, no instance-dependent optimality in the batch setting, weighted-minimax criterion. Li, Ma & Srebro (NeurIPS 2022; full text read): pessimism-validity and Proposition 1 ($\Delta\le\beta\|\mathbb E\phi(\pi^*)\|_*$), the $\ell_p$ family with Theorem 1, $\hat\pi_2=$ BCP exactly, $\hat\pi_\infty=$ PUNC reducing to tabular LCB (Corollary 1), PEVI as a context-wise enlarged $\ell_2$ set, Theorem 2 lower bounds over $\mathrm{CB}_q(\Lambda)$, adaptive minimax optimality of PUNC; fixed design; the plug-in rule analyzed in their Appendix E. Hu, Kallus & Uehara (COLT 2021): plug-in policies from FQI or Bellman-residual minimization in infinite-horizon MDPs, regret rate $=$ exponentiation of the pointwise $Q^*$ rate, $O(1/T)$ linear, $e^{-\Omega(T)}$ tabular.

*Estimators.* "Direct method (DM)" is coined in `dudik2011Doubly`, §1 and §2.1, with the bias and variance identities of Lemma 5.1 as its Sections 3–4 (journal version: Dudík, Erhan, Langford & Li, *Statistical Science* 29(4):485–511, 2014); IPS is credited there to Horvitz & Thompson (1952), the DR form to Cassel, Särndal & Wretman (1976), and double robustness to Robins, Rotnitzky & Zhao (1994), Robins & Rotnitzky (1995), Lunceford & Davidian (2004), Kang & Schafer (2007); DM-based policy optimization is their §5.1.3 (cost-sensitive classification with imputed losses). Wang, Agarwal & Dudík (ICML 2017): minimax lower bound matched by IPS and DR; SWITCH. Voloshin, Le, Jiang & Yue (2019) adopt IPS / Direct Methods / Hybrid as the RL taxonomy, the DR extension to RL being Jiang & Li (ICML 2016).

### 8.2 Reading order

1. **Rashidinejad et al. (NeurIPS 2021)** — tabular LCB, single-policy concentrability, the $(C^*-1)$ regimes; Theorem 4.2's source.
2. **Jin, Yang, Wang (ICML 2021)** — uncertainty quantifiers, PEVI, Theorem 4.1's source; read the horizon-one case.
3. **Brandfonbrener et al. (ICML 2021)** — the value-based / policy-based distinction, action-stability, overparameterization.
4. **Xie et al. (NeurIPS 2021)** and **Li, Ma, Srebro (NeurIPS 2022)** — version-space (policy-level) pessimism, class-dependent coverage, and the pointwise-versus-policy-level comparison. *(Not to be confused with the vault's `li2022InstanceOptimal`, an online PAC paper by the same authors.)*
5. **`xiao2021Optimality`** (ICML 2021) — what pessimism does and does not buy (§4.6).
6. **`dudik2011Doubly`** + Wang, Agarwal, Dudík (ICML 2017) — direct method, DR, SWITCH (§5).
7. **Murphy (JMLR 2005)**, **Beygelzimer & Langford (KDD 2009, §6.1)**, **Qian & Murphy (Ann. Statist. 2011)** — the pre-pessimism origins of Theorems 4.2 and 8.2.
8. **`nguyen-tang2022Offline`** (ICLR 2022) — neural instantiation.
9. **Yin & Wang (NeurIPS 2021)**, **Hu, Kallus, Uehara (COLT 2021)** — §8.
10. Athey & Wager (Econometrica 2021), Zhou, Athey & Wager (Oper. Res. 2023) — class-restricted policy learning with DR scores (§6.2).
11. Policy-side context: `swaminathan2015Batch` (clipped IPW with an empirical-Bernstein penalty), `gabbianelli2023ImportanceWeighted` (IX), Sakhi et al. (NeurIPS 2024, LS), [[ryu2025Improved]] (betting/freezing). Foster, Krishnamurthy, Simchi-Levi & Xu (COLT 2022) explain why the bandit case is clean and reinforcement learning is not.

### 8.3 Verify-before-citing flags

Verified on 2026-09-17 (see §8.1): everything attributed to Dudík et al., Beygelzimer & Langford, Murphy, Qian & Murphy, Rashidinejad et al. (Definition 1, Theorems 4–5, Proposition 1 as stated in their overview), Jin et al. (Theorem 3.2, bonus form), Xie et al. (Eq. 3.2, Definition 1), Xiao et al. (index rules, minimax results, weighted-minimax), Li–Ma–Srebro (full text), Hu–Kallus–Uehara (abstract-level rates), Wang–Agarwal–Dudík (abstract-level), Brandfonbrener et al. (definitions and action-stability). Still from memory: the Singh & Yee (1994) attribution for the classical greedy-policy loss bound (§3.2); Jin's lower-bound instance class; Xie's exact Theorem 2.1 constants; the empirical-Bernstein statement of Maurer & Pontil; Athey & Wager / Zhou, Athey & Wager theorem forms; NeuraLCB's exact coverage condition.

Resolved 2026-09-19: Rashidinejad's penalty constant is verified — $\Gamma=\sqrt{2000\ln(2SK/\delta)/N(x,a)}$, $L=2000\ln(2SKn)$ at $\delta=1/T$ (§4.7). **One discrepancy found in their proof:** at the well-covered-contexts step they assert $N(x,\pi(x))\ge4\,N(x,a)$, where the properties invoked give only $5/2$; §4.7 is written with $5/2$ and the resulting $L/9$ exponent, and nothing downstream changes. Theorems 3.5 and 3.6 are ours, derived not quoted (§3.5).


## Key Papers

- Rashidinejad, Zhu, Ma, Jiao & Russell (NeurIPS 2021) — tabular LCB, single-policy concentrability, the $(C^*-1)$ rate and its lower bound; proved in §4.7
- Jin, Yang & Wang (ICML 2021) — uncertainty quantifiers and pessimistic value iteration; the source of Theorem 4.1 at horizon one
- Xie, Cheng, Jiang, Mineiro & Agarwal (NeurIPS 2021) — version-space pessimism and class-dependent coverage (§4.5)
- Li, Ma & Srebro (NeurIPS 2022) — the linear case at the level of policies; adaptive minimax optimality of the $\ell_\infty$ rule (§4.4)
- Xiao, Wu, Lattimore et al. (ICML 2021) — what pessimism does and does not buy (§4.6)
- Beygelzimer & Langford (KDD 2009), Murphy (JMLR 2005), Qian & Murphy (2011) — the pre-pessimism origins of the greedy bounds
- Dudík, Langford & Li (2011) — the direct method (§5)

## Related Concepts

- [[contextual-bandits-offline]] — the setting, the notation, and the comparison with the policy route
- [[pessimism-principle]] — the mechanism of §4, in one page
- [[coverage-coefficient]] — the coefficients of §2.3
- [[realizability]] — assumption (A3), on which every rate here depends
- [[extrapolation-error]] — the greedy failure mode of §3.4, in RL language
- [[fqi-finite-sample-analysis]] — the horizon-$H$ analysis; Theorem 2.6 is its "part C", Lemma 2.13 its "part A" without error propagation
- [[instance-dependent-bounds]] — §7
- [[importance-weighting]] — the other route's estimator
- [[ryu2025Improved]] — the policy-side second-order bound that §7.1 mirrors

## Current State and Open Problems

The tabular case is closed up to constants. The linear case is closed for policy-level rules. What is open sits in three places: pointwise uncertainty quantifiers for general function classes, where least squares delivers only an $L_2$ guarantee and pessimism has to fall back on the version space (§4.5); whether pessimism applied at the level of policies rather than contexts buys anything beyond what Li–Ma–Srebro already establish; and instance-dependent rates not driven by a worst-case coverage constant (§7).
