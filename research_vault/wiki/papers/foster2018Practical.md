---
title: "Practical Contextual Bandits with Regression Oracles"
authors: [Dylan J. Foster, Alekh Agarwal, Miroslav Dudík, Haipeng Luo, Robert E. Schapire]
year: 2018
venue: ICML
arxiv: "1803.01088"
citekey: foster2018Practical
tags: [contextual-bandits, oracle-efficiency, regression, confidence-bounds, ucb, realizability, regret-bounds, learning-theory]
---

# RegCB: Confidence Bounds from a Regression Oracle

**TL;DR:** RegCB turns any class with a weighted least-squares oracle into a UCB-style contextual bandit. It computes the two ends of a version space at a pair by refitting with one extra example whose target is out of range: $`2`$ for the upper end, $`-1`$ for the lower. It then binary-searches that example's weight until the excess loss reaches the tolerance. For convex classes this takes $`O(\log(1/\alpha))`$ oracle calls per bound (Theorem 1). The regret is not minimax: a class of local perturbations forces $`\lvert\mathcal F\rvert`$ mistakes (Proposition 1). Under distributional coefficients it is $`\tilde O(\sqrt{T L_{2,0}\log\lvert\mathcal F\rvert})`$ (Theorem 3), or logarithmic under a margin. With gradient-boosted trees it beats ε-greedy and ILTCB on 18 datasets.

## Intuition

The paper reacts to a split in the field. *Agnostic* methods (ε-greedy, ILOVETOCONBANDITS) are provably good for any policy class, but they need a cost-sensitive classification oracle, which is NP-hard in general. *Realizability-based* methods (LinUCB, Thompson sampling) work better in practice, but they need closed-form confidence bounds or posterior sampling, which exist only for special families. The move: keep realizability and the UCB logic, and compute the confidence bounds with the same regression oracle that fits the model.

The mechanism is a Lagrangian. The version space is the set of members whose squared loss is within $`\beta`$ of the best. Its upper end at a pair $`z`$, $`\max_f f(z)`$ over that set, is a constrained maximization that the oracle cannot solve. Its penalized form is a regression the oracle can solve: append the example $`(z,2)`$ with weight $`w/2`$ and refit. A larger $`w`$ pulls the prediction at $`z`$ toward the impossible target $`2`$, and the fit pays for it in loss on the data.

For a convex class both the prediction and the loss move monotonically in $`w`$. So a binary search on $`w`$ finds the weight at which the excess loss equals $`\beta`$, and the prediction there is the upper bound up to $`\alpha`$. The lower bound uses the target $`-1`$. In one line: *pull the prediction toward an unreachable target until the data object, and read off where it stopped.*

With the bounds in hand the algorithms are classical. **Elimination** plays uniformly among the actions whose upper bound beats the best lower bound. **Optimism** plays the action with the largest upper bound, after a warm start.

The analysis needs a distributional quantity, because Proposition 1 shows that no confidence-based rule of this kind is minimax. The paper offers the disagreement coefficient from active learning and two new moment coefficients, the *surprise bound* and the *implicit exploration coefficient*.

## Formal Problem Definition

**Notation.** Notation follows [[contextual-bandits-online]], which itself builds on the anchor in [[contextual-bandits-offline]] §2, not the paper's. The following are translated throughout:

- the paper's $`f^\star`$ is $`q^*`$, its $`D_{\mathcal X}`$ is $`\nu`$, and $`\pi^\star=\pi_{f^\star}`$ is $`\pi^*`$;
- its reward vector $`r\in[0,1]^K`$, drawn jointly with $`x`$, is kept only through the observed entry: its $`r_t(a_t)`$ is $`r_t`$, with mean $`q^*(x_t,a_t)`$;
- $`\star`$ is $`*`$.

Kept as in the paper:

- the greedy policy $`\pi_f(x)\in\arg\max_af(x,a)`$, and the oracle $`\mathrm{ORACLE}(H)`$;
- the epochs $`\tau_m=2^{m-1}`$, $`m=1,\dots,M`$, with $`M=O(\log T)`$;
- the empirical loss $`\hat R_m`$, the version space $`\hat{\mathcal F}_m(\beta)`$, the tolerance $`\beta_m`$ and the precision $`\alpha`$;
- $`\mathrm{High}`$, $`\mathrm{Low}`$ and the width $`W`$;
- the coefficients $`\theta_0`$, $`L_1`$ and $`L_{2,\lambda}`$, the margin sets $`U_\lambda(a)`$, the Massart margin $`\gamma`$, and the constants $`C_\delta`$, $`C'_\delta`$;
- the product class $`\mathcal F=\mathcal G^{\mathcal A}`$ with base class $`\mathcal G`$, whose members are $`f=(g_a)_{a\in\mathcal A}`$ with $`f(x,a)=g_a(x)`$.

Several symbols mean something else here than elsewhere in the vault: $`\alpha`$ is the search precision, not a QoM level; $`\beta`$ is a loss tolerance; $`M`$ is an epoch count; and $`\lambda`$ indexes a margin.

**One difference is substantive, and is restated rather than translated.** The paper's $`\mathrm{Reg}_T`$ is the *realized* regret $`\sum_t\big(r_t(\pi^*(x_t))-r_t(a_t)\big)`$. It needs the full reward vector, including the unplayed entry. The vault's $`\mathrm{Reg}(T)`$ on [[contextual-bandits-online]] is in means, $`\sum_t\big(V^*(x_t)-q^*(x_t,p_t)\big)`$. The two differ by a martingale. The proofs bound the mean form and then absorb that martingale by Azuma–Hoeffding, at $`O(\sqrt{T\log(1/\delta)})`$. Under the Massart condition that term would dominate, so those results are stated in expectation only (Remark 2).

### Setting and learning protocol

A known class $`\mathcal F\subseteq\{f:\mathcal X\times\mathcal A\to[0,1]\}`$ that contains $`q^*`$, accessed only through a weighted least-squares oracle. The oracle takes any set $`H`$ of weighted examples $`(w,x,a,y)`$ and returns

```math
\mathrm{ORACLE}(H)=\arg\min_{f\in\mathcal F}\ \sum_{(w,x,a,y)\in H} w\,\big(f(x,a)-y\big)^2 .
```

```math
\begin{aligned}
&\textbf{Given: } \text{class } \mathcal F \text{ through } \mathrm{ORACLE},\ \text{actions } \mathcal A \text{ with } K:=\lvert\mathcal A\rvert,\ \text{horizon } T \\
&\textbf{for } t=1,\dots,T: \\
&\qquad \textbf{observe } x_t\sim\nu \\
&\qquad \textbf{choose } a_t \text{ from } \{(x_s,a_s,r_s)\}_{s<t} \qquad \triangleright\ \text{computed by oracle calls only} \\
&\qquad \textbf{observe } r_t\sim\rho(\cdot\mid x_t,a_t)\in[0,1] \qquad \triangleright\ \text{the other } K-1 \text{ rewards are never seen} \\
&\textbf{scored by } \mathrm{Reg}_T=\textstyle\sum_{t=1}^T\big(r_t(\pi^*(x_t))-r_t(a_t)\big)
\end{aligned}
```

Three features of this loop carry the difficulty:

- **The oracle is an exact weighted ERM, fed synthetic data.** The algorithm builds the weighted datasets itself, including examples that never occurred. The paper's signature restricts the targets to $`[0,1]`$, but Algorithm 3 feeds it the targets $`2`$ and $`-1`$, as do the unbounded variants of Appendix A.1 (Algorithms 4–5). So the oracle must accept out-of-range targets.
- **The data are adaptive.** The algorithm chose the logged actions, so the concentration is a martingale (Freedman) bound, with $`C'_\delta=16\log(2\lvert\mathcal F\rvert T^2/\delta)`$ for a finite class.
- **Speed needs convexity.** Theorem 1 assumes $`\mathcal F`$ convex. For a nonconvex class the same bounds cost $`O(1/\alpha)`$ calls, by the procedure of Krishnamurthy et al. (2017).

### Learning objective

Standard regret against $`\pi^*`$, in the realized form above. The benchmark the paper measures itself against is the optimal $`\tilde O(\sqrt{KT\log\lvert\mathcal F\rvert})`$. Exp4 attains it at computation linear in $`\lvert\mathcal F\rvert`$, and Regressor Elimination (Agarwal et al. 2012) attains it without an oracle.

### How this differs from the surrounding literature

- **Against agnostic oracle methods** (ε-greedy, Langford & Zhang 2008; ILTCB, Agarwal et al. 2014). A regression oracle replaces cost-sensitive classification, and realizability is assumed. What is lost is minimax optimality.
- **Against LinUCB.** The rule extends to any class with an oracle. With the ridge oracle, Algorithm 2 coincides with OFUL (Abbasi-Yadkori et al. 2011) on an epoch schedule (footnote 6). For the standard LinUCB setting, its moment-based bound is worse by at least a factor $`K`$.
- **Against Krishnamurthy et al. (2017)**, cost-sensitive active learning. The version-space bounds are the same, but the search drops from $`O(1/\alpha)`$ calls to $`O(\log(1/\alpha))`$ when $`\mathcal F`$ is convex.
- **Against eluder dimension** (Russo & Van Roy 2013), which is a worst case over histories and hard to bound beyond linear classes. It is replaced by moment coefficients of the i.i.d. distribution.
- **Against Bastani & Bayati (2015).** Their linear exploration conditions are generalized to arbitrary classes, and weakened for Algorithm 1, which needs no warm start.

## Assumptions

1. **Realizability (Assumption 1):** $`q^*\in\mathcal F`$, that is, $`\mathbb E[r(a)\mid x,a]=q^*(x,a)`$ at every pair. See [[realizability]].
2. **Stochastic i.i.d. data, bounded rewards:** $`(x_t,r_t)`$ are drawn i.i.d., with $`r\in[0,1]^K`$.
3. **For Theorem 1:** $`\mathcal F`$ is convex and closed under pointwise convergence.
4. **For Theorem 2:** a product class $`\mathcal F=\mathcal G^{\mathcal A}`$. Only then does the set $`A_t`$ that the algorithm computes equal the disagreement set (Lemma 6).
5. **The coefficients.** $`\theta_0`$, $`L_1`$ and $`L_{2,\lambda}`$ are properties of $`(\nu,\mathcal F,q^*)`$. The bounds are informative only when they are small, and nothing in the algorithm checks this.
6. **Massart margin (Definition 8, optional):** $`q^*(x,\pi^*(x))\ge q^*(x,a)+\gamma`$ for all $`x`$ and $`a\ne\pi^*(x)`$. It is used only for the logarithmic-in-$`T`$ rates.

**Not assumed:**

- a cost-sensitive classification oracle;
- closed-form confidence sets, or posterior sampling;
- linear or parametric structure;
- knowledge of $`\theta_0`$ or $`L_{2,\lambda}`$. Algorithm 1 needs neither. Algorithm 2's warm start length $`M_0`$ does depend on $`L_1`$ and the margin $`\lambda`$;
- an eluder-dimension bound.

**Relaxation.** Infinite classes enter through empirical $`L_1`$ covering numbers (Lemma 4, built on Krishnamurthy et al. 2017). Then $`C_\delta`$ becomes $`\inf_{\varepsilon>0}\{100\varepsilon T+320\log(\dots)\}`$, with the covering number inside the logarithm.

## Method

### Version spaces and the two strategies (§3)

For $`m\ge2`$, with $`\hat{\mathcal F}_1(\beta):=\mathcal F`$,

```math
\hat R_m(f):=\frac1{\tau_m-1}\sum_{s<\tau_m}\big(f(x_s,a_s)-r_s\big)^2,\qquad
\hat{\mathcal F}_m(\beta):=\Big\{f\in\mathcal F:\ \hat R_m(f)-\min_{f'\in\mathcal F}\hat R_m(f')\le\beta\Big\},
```

```math
\mathrm{High}_{\mathcal F'}(x,a):=\max_{f\in\mathcal F'}f(x,a),\qquad
\mathrm{Low}_{\mathcal F'}(x,a):=\min_{f\in\mathcal F'}f(x,a),\qquad
W_{\mathcal F'}(x,a):=\mathrm{High}_{\mathcal F'}(x,a)-\mathrm{Low}_{\mathcal F'}(x,a).
```

For a product class, $`\hat{\mathcal G}_m(\beta,a)`$ is the same construction for one action, with loss $`\frac1{\tau_m-1}\sum_{s<\tau_m}(g(x_s)-r_s)^2\mathbf 1\{a_s=a\}`$.

```math
\begin{aligned}
&\textbf{Algorithm 1 (RegCB.Elimination). Input: } \text{tolerances } \beta_m \\
&\textbf{for } m=1,\dots,M: \\
&\qquad \mathcal F_m\leftarrow\textstyle\prod_{a}\hat{\mathcal G}_m(\beta_m,a)\ \text{(Option I)}\quad\text{or}\quad \hat{\mathcal F}_m(\beta_m)\ \text{(Option II)} \\
&\qquad \textbf{for } t=\tau_m,\dots,\tau_{m+1}-1: \\
&\qquad\qquad A_t\leftarrow\big\{a:\ \mathrm{High}_{\mathcal F_m}(x_t,a)\ge\max_{a'}\mathrm{Low}_{\mathcal F_m}(x_t,a')\big\} \qquad \triangleright\ \text{plausibly optimal actions} \\
&\qquad\qquad a_t\sim\mathrm{Unif}(A_t) \\[4pt]
&\textbf{Algorithm 2 (RegCB.Optimistic). Input: } \beta_m,\ \text{warm-start epochs } M_0 \\
&\textbf{for } t<\tau_{M_0}:\ a_t\sim\mathrm{Unif}(\mathcal A) \qquad \triangleright\ \text{pure exploration} \\
&\textbf{for } m=M_0,\dots,M:\ \mathcal F_m\leftarrow\hat{\mathcal F}_m(\beta_m);\quad a_t=\arg\max_a\mathrm{High}_{\mathcal F_m}(x_t,a)\ \text{ for } t\in[\tau_m,\tau_{m+1})
\end{aligned}
```

The theory uses $`\beta_m=(M-m+1)C_\delta/(\tau_m-1)`$. The factor $`M-m+1`$ makes the version spaces nested, $`\mathcal F_m\subseteq\mathcal F_{m-1}`$, while every one of them still contains $`q^*`$ (Lemma 5, claim 4).

### Confidence bounds by binary search (§3.1, Theorem 1)

```math
\begin{aligned}
&\textbf{Algorithm 3 (BinSearch). Input: } \text{type}\in\{\mathrm{Low},\mathrm{High}\},\ \text{pair } (x,a),\ \text{history } H,\ \text{radius } \beta,\ \text{precision } \alpha \\
&r\leftarrow2 \text{ if High},\quad r\leftarrow-1 \text{ if Low} \qquad \triangleright\ \text{a target outside } [0,1] \\
&R(f):=\textstyle\sum_{(x',a',r')\in H}\big(f(x',a')-r'\big)^2,\qquad \tilde R(f,w):=R(f)+\tfrac w2\big(f(x,a)-r\big)^2 \\
&w_L\leftarrow0,\ w_H\leftarrow\beta/\alpha;\qquad f_L\leftarrow\arg\min_f\tilde R(f,w_L),\ z_L\leftarrow f_L(x,a);\qquad f_H\leftarrow\arg\min_f\tilde R(f,w_H),\ z_H\leftarrow f_H(x,a) \\
&R_{\min}\leftarrow R(f_L),\qquad \Delta\leftarrow\alpha\beta/(r-z_L)^3 \\
&\textbf{while } \lvert z_H-z_L\rvert>\alpha \textbf{ and } \lvert w_H-w_L\rvert>\Delta: \\
&\qquad w\leftarrow(w_H+w_L)/2;\quad f\leftarrow\arg\min_{\tilde f}\tilde R(\tilde f,w);\quad z\leftarrow f(x,a) \qquad \triangleright\ \text{one oracle call} \\
&\qquad \textbf{if } R(f)\ge R_{\min}+\beta:\ (w_H,z_H)\leftarrow(w,z)\quad\textbf{else}:\ (w_L,z_L)\leftarrow(w,z) \\
&\textbf{return } z_H
\end{aligned}
```

**Theorem 1.** Let $`H_m=\{(x_s,a_s,r_s)\}_{s=1}^{\tau_m-1}`$. If $`\mathcal F`$ is convex and closed under pointwise convergence, then $`\mathrm{BinSearch}(\mathrm{High},(x,a),H_m,\beta,\alpha)`$ and $`\mathrm{BinSearch}(\mathrm{Low},(x,a),H_m,\beta,\alpha)`$ terminate after $`O(\log(1/\alpha))`$ oracle invocations, and

```math
\big\lvert\mathrm{High}_{\hat{\mathcal F}_m(\beta)}(x,a)-z_{\mathrm{High}}\big\rvert\le\alpha,\qquad
\big\lvert\mathrm{Low}_{\hat{\mathcal F}_m(\beta)}(x,a)-z_{\mathrm{Low}}\big\rvert\le\alpha .
```

*Technique (Appendix A.1, Lemma 1).* The proof is for the unbounded variant. It reduces the search to one variable through the loss profile at the query,

```math
\varphi(z):=\min\{R(f):\ f\in\mathcal F,\ f(x,a)=z\}\qquad(+\infty \text{ if no member reaches } z).
```

- **The profile is convex.** Embed each member as its vector of values on $`H\cup\{(x,a)\}`$. Convexity and closure of $`\mathcal F`$ then make $`\varphi`$ convex and lower semicontinuous (Rockafellar 1970, Theorem 9.2).
- **The search moves monotonically.** The fit's prediction minimizes $`\varphi(z)+\frac w2(2-z)^2`$. Its first-order condition $`w=\varphi'(z_w)/(2-z_w)`$ makes $`z_w`$ nondecreasing in $`w`$ and below $`2`$. So the search over $`w`$ is a search over $`z`$ for $`z^*=\max\{z:\varphi(z)\le R_{\min}+\beta\}`$.
- **The two constants are calibrated.** The initial $`w_H=\beta/\alpha`$ is large enough that the bracket contains $`z^*`$ (Step 1). The stopping width $`\Delta`$ caps the overshoot at $`\alpha`$ (Step 2).
- **Iteration count.** It is $`O(\log(1/\alpha)+\log\max\{2-z_0,1\})`$, where $`z_0`$ is the unweighted fit's prediction.

**Cost per round.** This is this page's count; the paper states only the cost of one search.

- Algorithm 2 needs $`K`$ upper bounds per round, and Algorithm 1 needs $`2K`$ bounds.
- So, for a convex class, both make $`O(TK\log(1/\alpha))`$ oracle calls over $`T`$ rounds.
- [[simchi-levi2022Bypassing]] quotes RegCB at $`O(T^{3/2})`$ calls. That matches the nonconvex $`O(1/\alpha)`$ route at $`\alpha\approx T^{-1/2}`$. This reconciliation is this page's reading and has not been checked against FALCON's source.

**Three slips in the statement, found in this page's check.**

- **The stopping width is negative for Low.** As printed, Algorithm 3's $`\Delta=\alpha\beta/(r-z_L)^3`$ is negative when $`r=-1`$. The unbounded Low variant (Algorithm 5) uses $`\alpha\beta/(1+z_L)^3`$, that is, $`\lvert r-z_L\rvert^3`$.
- **The two tolerances are on different scales.** $`R`$ in Algorithm 3 is a *sum* over $`\tau_m-1`$ examples, while $`\hat{\mathcal F}_m(\beta)`$ thresholds the *average* $`\hat R_m`$. For Theorem 1's version space, the radius passed to the search must therefore be $`(\tau_m-1)\beta`$.
- **The targets leave the declared range,** as noted under the protocol.

### Not minimax: the local-perturbation instance (Proposition 1)

**Proposition 1.** For every $`\epsilon\in(0,1]`$ and $`N\in\mathbb N`$, there are a class with $`\lvert\mathcal F\rvert=N+1`$ and a distribution on which both algorithms have regret at least $`(1-\epsilon)\cdot\min\{N,\tilde\Omega(T)\}`$ over $`T`$ rounds.

The instance (Appendix A.2):

- two actions $`a_g,a_b`$ with noiseless rewards $`1-\epsilon`$ and $`0`$, and $`N`$ contexts, $`\nu`$ uniform;
- $`\mathcal F=\{q^*\}\cup\{f_i\}_{i\le N}`$, where $`f_i`$ agrees with $`q^*`$ except at $`x_i`$, with $`f_i(x_i,a_g)=0`$ and $`f_i(x_i,a_b)=1`$;
- until $`x_i`$ is observed, $`f_i`$ survives in every version space, even at $`\beta=0`$;
- so at a first visit $`\mathrm{High}(x_i,a_b)=1>1-\epsilon=\mathrm{High}(x_i,a_g)`$ and both lower bounds are $`0`$, and the algorithms err at each first visit.

The paper calls the instance pathological and lets the experiments argue it is not met in practice.

### Disagreement-based analysis (§4.1, Theorem 2)

**Definitions 1–5.**

- **Policy-regret ball:** $`\mathcal F(\varepsilon):=\{f:\ \mathbb E[r(\pi_f(x))]\ge\mathbb E[r(\pi^*(x))]-\varepsilon\}`$.
- **Width:** $`W_{\mathcal F}(x,a)`$, defined above.
- **Disagreement region:** $`\mathrm{Dis}(\mathcal F):=\{x:\ \exists f,f'\in\mathcal F \text{ with different greedy actions at } x\}`$.
- **Disagreement set:** $`A_{\mathcal F}(x):=\bigcup_{f\in\mathcal F}\arg\max_af(x,a)`$.
- **Disagreement coefficient:**

```math
\theta_0:=\sup_{\delta>0,\varepsilon>0}\ \frac\delta\varepsilon\,\Pr_{x\sim\nu}\Big[x\in\mathrm{Dis}(\mathcal F(\varepsilon))\ \text{and}\ \exists a\in A_{\mathcal F(\varepsilon)}(x):\ W_{\mathcal F(\varepsilon)}(x,a)>\delta\Big].
```

**Theorem 2.** For a product class $`\mathcal F=\mathcal G^{\mathcal A}`$, with $`\beta_m=(M-m+1)C_\delta/(\tau_m-1)`$ and $`C_\delta=16\log(2\lvert\mathcal G\rvert KT^2/\delta)`$, Algorithm 1 with Option I has, with probability at least $`1-\delta`$,

```math
\mathrm{Reg}_T=\tilde O\Big(T^{3/4}(\log\lvert\mathcal G\rvert)^{1/4}\sqrt{\theta_0K}\Big).
```

Under the Massart condition, the full version (Theorem 5, with $`\delta=1/T`$) gives $`\mathbb E[\mathrm{Reg}_T]=\tilde O\big(\theta_0K^2\log(\lvert\mathcal G\rvert T)/\gamma^2\big)`$.

*Technique.*

- **Why the product class is needed.** For a product class, $`A_t`$ is exactly the disagreement set $`A_{\mathcal F_m}(x_t)`$ (Lemma 6). Computing that set for a non-product class is left open.
- **From the version space to the ball.** The version space sits inside a policy-regret ball $`\mathcal F(\varepsilon_m)`$ (Lemma 7). The uniform draw from $`A_t`$ makes the squared error along both $`\pi^*`$ and $`\pi_f`$ at most $`K`$ times the error on the logged actions.
- **Charging each round.** A round's regret is charged to the probability of landing in the disagreement region with a wide action, which is at most $`2\theta_0\varepsilon_m/\eta'`$.

How to analyze Algorithm 2 with $`\theta_0`$ is left open.

### Moment-based analysis (§4.2, Theorems 3–4)

**Definition 6 (surprise bound).** $`L_1`$ is the smallest constant with

```math
\big(f(x,a)-q^*(x,a)\big)^2\le L_1\,\mathbb E_{x'\sim\nu}\mathbb E_{a'\sim\mathrm{Unif}(\mathcal A)}\big[(f(x',a')-q^*(x',a'))^2\big]\quad\text{for all } f\in\mathcal F,\ x,\ a.
```

**Definition 7 (implicit exploration coefficient, IEC).** Let $`U_\lambda(a):=\{x:\ q^*(x,a)\ge q^*(x,a')+\lambda\ \text{for all } a'\ne a\}`$, the contexts where $`a`$ is best by a margin $`\lambda`$. Then $`L_{2,\lambda}`$ is the smallest constant with

```math
\big(f(x,a)-q^*(x,a)\big)^2\le L_{2,\lambda}\,\mathbb E_{x'\sim\nu}\mathbb E_{a'\sim\mathrm{Unif}(\mathcal A)}\big[\mathbf 1\{x'\in U_\lambda(a')\}\,(f(x',a')-q^*(x',a'))^2\big]\quad\text{for all } f,\ x,\ a.
```

- **At $`\lambda=0`$,** it suffices that $`\sup_{x,a}(f-q^*)^2\le\frac{L_{2,0}}K\,\mathbb E_x[(f-q^*)^2(x,\pi^*(x))]`$. That is, the error on the optimal policy's pairs controls the error everywhere.
- **Ordering.** $`L_{2,\lambda}`$ is nondecreasing in $`\lambda`$, and $`L_1\le L_{2,0}`$.

**Theorem 3.** For a general class, with $`\beta_m=(M-m+1)C'_\delta/(\tau_m-1)`$ and $`C'_\delta=16\log(2\lvert\mathcal F\rvert T^2/\delta)`$, Algorithm 1 with Option II has, with probability at least $`1-\delta`$,

```math
\mathrm{Reg}_T=\tilde O\Big(\sqrt{T\,L_{2,0}\log\lvert\mathcal F\rvert}\Big).
```

Under Massart (Theorem 6), $`\mathbb E[\mathrm{Reg}_T]=\tilde O(L_{2,0}\log\lvert\mathcal F\rvert/\gamma)`$.

**Theorem 4.** With the same $`\beta_m`$, take any $`\lambda\in(0,1)`$ and $`M_0=2+\lfloor\log_2(1+(2M+3)L_1C'_\delta/\lambda^2)\rfloor`$. Then Algorithm 2 has, with probability at least $`1-\delta`$,

```math
\mathrm{Reg}_T=\tilde O\Big(\frac{L_1\log\lvert\mathcal F\rvert}{\lambda^2}+\sqrt{T\,L_{2,\lambda}\log\lvert\mathcal F\rvert}\Big).
```

Under Massart (Theorem 7), the second term becomes $`L_{2,\lambda}\log\lvert\mathcal F\rvert/\gamma`$ times $`\mathbf 1\{\lambda>\gamma\}`$. So with $`\lambda\le\gamma`$, Algorithm 2 incurs no regret after the warm start. Since $`L_{2,\lambda}\ge L_{2,0}`$ and the warm start is extra, Theorem 4 is always weaker than Theorem 3.

*Technique (Theorem 6).*

- **Per-round regret is a sup-norm width.** For any $`\eta>0`$ it is at most $`\eta P_\eta+\frac4\eta\sup_{f\in\mathcal F_m}\sup_{x,a}(f-q^*)^2`$, where $`P_\eta`$ is the probability of a gap below $`\eta`$.
- **The IEC removes the sup.** Every $`a`$ with $`x\in U_0(a)`$ lies in $`A_t`$, so the IEC turns the sup into the squared error on actions the algorithm actually plays, which the version space controls (Lemma 10).

**Proposition 2 (linear and sparse cases).**

- **Linear,** with $`\lVert\phi\rVert_2\le1`$ and $`\lVert w\rVert_2\le1`$:

```math
L_{2,\lambda}\le\frac{K}{\lambda_{\min}\big(\sum_a\mathbb E_x[\mathbf 1\{x\in U_\lambda(a)\}\phi(x,a)\phi(x,a)^\top]\big)},\qquad
L_1\le\frac{K}{\lambda_{\min}\big(\sum_a\mathbb E_x[\phi(x,a)\phi(x,a)^\top]\big)} .
```

- **Sparse,** with $`\lVert\phi\rVert_\infty\le1`$, $`\lVert w\rVert_\infty\le1`$ and $`\lVert w\rVert_0\le s`$: the same bounds with $`2Ks`$ in the numerator, and the minimum restricted eigenvalue over $`2s`$-sparse vectors, $`\psi_{\min}`$, in place of $`\lambda_{\min}`$.
- **The sparse corollary.** Theorem 3 with a cover of size $`\log\lvert\mathcal F\rvert=O(s\log d)`$ gives $`\tilde O\big(s\sqrt{KT\log d/\psi_{\min}(\mathbb E_x[\phi(x,\pi^*(x))\phi(x,\pi^*(x))^\top])}\big)`$. This is nearly dimension-free. But the sparse class is not convex, so it is oracle-efficient only with the slower $`O(1/\alpha)`$ search (footnote 3).

### Experiments (§5, Appendix B)

**Setup.**

- **Datasets: 18.** Two learning-to-rank sets: MSLR-WEB30k ($`K=10`$, $`d=136`$) and Yahoo C14B ($`K=6`$, $`d=415`$). Eight UCI multiclass sets: letter, optdigits, adult, page-blocks, pendigits, satimage, vehicle and yeast. Eight "noisy" UCI variants, whose rewards come from Bernoulli reward matrices.
- **Oracles.** scikit-learn `Ridge(alpha=1)` ("Linear") and `GradientBoostingRegressor(max_depth=5, n_estimators=100)` ("GB5").
- **The GB5 heuristic.** RegCB fixes the tree structure for each epoch and refits only the leaf values. That refit is in closed form, so Algorithm 3 is not run.
- **Baselines.** ε-greedy, ILTCB and Bootstrap (Dimakopoulou et al. 2017). LinUCB is RegCB-Opt with the Linear oracle.
- **Tuning.** Each method tries 8 log-spaced parameter values. RegCB uses a constant $`\beta_m=\beta`$ and a warm start of $`0`$, so the theory's schedule and warm start are not what was run.
  - The baselines are plotted as the pointwise maximum over their parameters, an upper envelope.
  - RegCB is plotted as an actual run at its best fixed parameter.

**Results.**

- **With GB5,** RegCB beats ε-greedy and ILTCB, sometimes loses to Bootstrap, and is almost always within $`20\%`$ of the best method. Elimination is slightly better than optimism in aggregate.
- **With Linear oracles,** RegCB dominates every baseline, Bootstrap included.
- **Confidence widths shrink.** RegCB's width decays with fitted log-log slopes of $`-0.34`$ (letter), $`-0.51`$ (letter+N) and $`-0.52`$ (yahoo). Bootstrap's best setting on letter has slope $`-0.05`$, an essentially constant width that coincides with poor validation reward. The paper reads this as Bootstrap's intervals not containing $`q^*`$.

## Connections

- [[oracle-efficiency]] — RegCB is the earliest regression-oracle algorithm in the vault's oracle-efficiency line. It reaches general classes with a weighted least-squares oracle and gives up minimax optimality, which is the gap that FALCON and SquareCB later close.
- [[simchi-levi2022Bypassing]] — FALCON names RegCB as the tractable-but-suboptimal baseline, and answers the paper's first open question for the offline oracle, with $`O(\log T)`$ calls.
- [[offline-regression-oracle]] — the oracle here is an *exact weighted ERM*, called on synthetic weighted data with out-of-range targets and on adaptively collected data. That is a different abstraction from the generalization-error oracle defined on that page.
- [[upper-confidence-bound]] — Algorithm 2 is UCB with version-space upper bounds, and OFUL on an epoch schedule when the oracle is ridge regression.
- [[realizability]] — Assumption 1. Every guarantee rests on it, and the bounds are version-space ends, which mean nothing without it.
- [[eluder-dimension]] — the paper's foil. Moment coefficients of the i.i.d. distribution replace this worst-case-over-histories measure.
- [[coverage-coefficient]] — a contrast, and this page's reading. Offline coverage bounds the error along $`\pi^*`$'s pairs by the error on the data. The IEC runs the other way: the error along $`\pi^*`$'s pairs, which greedy play generates, must control the error at every pair.
- [[contextual-bandits-online]] — the setting.
- [[pessimism-principle]] and [[contextual-bandits-offline-value-based]] — $`\mathrm{Low}`$ is an oracle-computed pessimistic estimate. See the last part of the Open Questions.
- **Concepts this paper introduces:** the binary-search confidence bounds, the surprise bound $`L_1`$, the implicit exploration coefficient $`L_{2,\lambda}`$, and the policy-regret ball. The disagreement coefficient is imported from active learning (Hanneke et al. 2014). None has a concept page: each is fully explained here and used elsewhere only by reference.
- **Extends:** Krishnamurthy et al. (2017), cost-sensitive active learning, whose version-space bounds RegCB computes exponentially faster for convex classes. Also LinUCB/OFUL (Chu et al. 2011; Abbasi-Yadkori et al. 2011) and Bastani & Bayati (2015). **Superseded on regret by:** FALCON ([[simchi-levi2022Bypassing]]) for the offline oracle, and SquareCB (Foster & Rakhlin 2020) for the online one.
- Dylan J. Foster, Alekh Agarwal, Miroslav Dudík, Haipeng Luo, Robert E. Schapire. Foster is also the first author of [[foster2025Good]].
- Krishnamurthy et al. (2017); Hanneke et al. (2014); Russo & Van Roy (2013); Bastani & Bayati (2015); Agarwal et al. (2012, 2014); Langford & Zhang (2008); Dimakopoulou et al. (2017); Chu et al. (2011); Abbasi-Yadkori et al. (2011); Foster & Rakhlin (2020) — cited author–year, no pages here.

## Open Questions

**The paper's own:**

- **Optimal regret with a regression oracle,** $`\tilde O(\sqrt{KT\log\lvert\mathcal F\rvert})`$. Can Regressor Elimination be oraclized? FALCON ([[simchi-levi2022Bypassing]]) has since answered this for the offline oracle, and SquareCB (Foster & Rakhlin 2020, not paged) for the online one.
- **Bootstrap.** Analyze it under distributional assumptions like those of §4.
- **Randomization.** Find a better scheme among the plausibly optimal actions than uniform play or all mass on the optimistic one.
- **Non-product classes.** Compute the disagreement set efficiently for them (Discussion after Theorem 2).
- **Algorithm 2.** Analyze it with the disagreement coefficient.

**Untested or unclaimed:**

- **Algorithm 3 is never run on a nonconvex class.** Theorem 1 needs convexity. GB5 is not convex, and the experiments sidestep the search with the fixed-structure leaf refit.
- **The theory's parameters are not what was run.** The experiments use a tuned constant $`\beta`$ and no warm start. Each baseline is plotted at its best parameter value for each sample size, while RegCB is one run at a single fixed value. The paper does this to be "as generous as possible" to the baselines.
- **Exact oracle.** All bounds assume the oracle's argmin is exact. An approximate oracle is not analyzed.

**For the offline value-based line.** This part is this page's reading, not the paper's. It concerns [[contextual-bandits-offline-value-based]] and the Overleaf note `02_offline_cb.tex`, whose general-function-approximation section has a local LCB with a regression oracle (`sec:ocb-lcb-regcb`) and QoM estimators built on a weighted extra data point (`sec:ocb-qom-wridge` and its general versions).

- **$`\mathrm{Low}`$ is a weighted extra data point.** BinSearch.Low refits with one extra example $`(z,-1)`$ of weight $`w/2`$. The note's weighted-extra-data-point estimators append the same example. What differs is how the weight is chosen:
  - **RegCB targets the tolerance.** It searches $`w`$ until the excess loss equals $`\beta`$. For a convex class that returns the version-space minimum, and with it the version space's $`\ln\lvert\mathcal F\rvert`$ price.
  - **The QoM estimators use geometry.** They fix the weight from a batch's Gram or kernel matrix, with no search and no version space, and get weak pessimism from Feige's inequality instead.
  - **The phantom round** of [[qom-value-based-pessimism]] is the weight-1, reward-0 member of the same family.
- **Offline, BinSearch.Low run once on all the data is the note's local LCB with a regression oracle.** The note uses a finite class, which Theorem 1 does not cover. For a nonconvex class $`\varphi`$ is not convex.
  - The penalized fits minimize $`R(f)+\frac w2(1+f(z))^2`$, which is linear in $`u:=(1+f(z))^2`$. So they reach only the points on the lower convex hull of $`\{((1+f(z))^2,R(f))\}`$. A point that is extreme in $`f(z)`$ but above that hull is never returned.
  - The search therefore returns a Lagrangian relaxation of the version-space minimum, not the minimum itself.
  - The note quantifies this relaxation: it can sit below the minimum by a constant, yet the coverage-form bound is unchanged up to the precision term.
  - That is an $`O(\log(1/\alpha))`$ route for finite classes, where this paper points to $`O(1/\alpha)`$ calls.
- **Proposition 1's class is the note's local-vs-global instance.** Both are built from local perturbations: each $`f_i`$ differs from $`q^*`$ at one context. The note's counterexample (`sec:tmp-gen-pess-cex` in `02_offline_temp.tex`) perturbs the single pair $`(x_i,2)`$.
  - **Online,** this costs RegCB one mistake per unseen context.
  - **Offline,** it costs pointwise pessimism a constant loss for $`T\le S/2`$, while pessimism over policies loses at most $`4/(5S)`$.
  - **The common cause:** a per-pair bound combines different members at different contexts. This is the per-pair versus per-policy gap that [[li2022Pessimism]] measures as a Jensen gap in the linear case.
