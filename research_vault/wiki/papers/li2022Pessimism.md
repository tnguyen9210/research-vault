---
title: "Pessimism for Offline Linear Contextual Bandits using ℓp Confidence Sets"
authors: [Gene Li, Cong Ma, Nathan Srebro]
year: 2022
venue: NeurIPS
arxiv: "2205.10671"
citekey: li2022Pessimism
tags: [contextual-bandits, offline-contextual-bandits, pessimism, confidence-bounds, complexity-measure, learning-theory]
---

# Pessimism via $`\ell_p`$ Confidence Sets

**TL;DR:** Any confidence set for $`\theta^*`$ induces a pessimistic policy: maximize each policy's worst value over the set. Taking $`\ell_p`$ balls around least squares gives a family $`\{\hat\pi_p\}_{p\ge1}`$ that contains Bellman-consistent pessimism ($`p=2`$) and a linear generalization of tabular LCB ($`p=\infty`$, called PUNC). PUNC's guarantee dominates every other member's, and PUNC is minimax optimal up to $`\log d`$ over every $`\ell_q`$-constrained instance class at once, which no finite $`p`$ achieves.

## Intuition

The paper asks which of the many pessimistic learning rules one should use, and answers it for linear offline bandits by putting them in one frame. A pessimistic rule builds a set $`\Theta`$ of parameters the data cannot rule out and scores each policy by its worst value over that set. The decomposition behind every pessimism bound then charges the learner only for underrating the optimal policy. So the whole guarantee is the width of $`\Theta`$ along a single direction, $`\bar\phi_{\pi^*}`$, the optimal policy's mean feature. "Which rule?" becomes "which set is narrowest along a direction the learner does not know?"

The default answer in the linear literature is an $`\ell_2`$ ball in the data's whitened geometry: BCP and PACLE use one, and PEVI uses one per context. The paper's move is to change the norm. The least-squares error has whitened coordinates that are each $`T^{-1/2}`$-sub-Gaussian. A box that bounds every coordinate separately therefore costs only $`\sqrt{\ln(d/\delta)}`$ per coordinate, where an $`\ell_2`$ ball built from the same bound costs $`\sqrt{d\ln(d/\delta)}`$ in every direction. Along a whitened direction $`w`$, with Lemma 1's widths, the box's half-width is $`\|w\|_1\sqrt{2\ln(d/\delta)/T}`$ and the ball's is $`\|w\|_2\sqrt{2d\ln(d/\delta)/T}`$. Since $`\|w\|_1\le\sqrt d\,\|w\|_2`$, the box is never wider. It is narrower by up to $`\sqrt d`$ when $`w`$ sits on a few coordinates. In the paper's words, the $`\ell_p`$ sets capture an error "averaged" over all directions, while the $`\ell_\infty`$ set separately estimates "the error in each direction".

The lower bounds make this more than a better constant. Measuring an instance by the dual norm along that one direction, $`\mathfrak C_q=\|\Sigma_D^{-1/2}\bar\phi_{\pi^*}\|_q`$, gives a nested family of instance classes. Each $`\hat\pi_p`$ is minimax on its own dual class, and the box is minimax on all of them. PUNC is the only member of the family that never needs to be told which class it is in.

## Formal Problem Definition

**Notation.** Notation follows [[contextual-bandits-offline]] §2 and the linear model of [[contextual-bandits-offline-value-based]] §2.4, not the paper's. The following are translated throughout:

- the paper's state $`s\in\mathcal S`$ is a context $`x\in\mathcal X`$;
- its test distribution $`\rho`$ is $`\nu`$ (the vault's $`\rho`$ is the reward kernel);
- its reward distribution $`R(s,a)`$ is $`\rho(\cdot\mid x,a)`$, and its mean $`r(s,a)`$ is $`q^*(x,a)`$;
- its value $`V(\pi)`$ is $`J(\pi)`$, and $`V(\pi^\star)-V(\hat\pi)`$ is $`\Delta(\hat\pi)`$;
- its sample size $`n`$ is $`T`$, $`A:=\lvert\mathcal A\rvert`$ is $`K`$, $`n(s,a)`$ is $`N(x,a)`$, and $`\star`$ is $`*`$;
- $`\hat\theta_{\mathrm{ols}}`$ is $`\hat\theta`$;
- the behavior distribution $`\mu\in\Delta(\mathcal S\times\mathcal A)`$ of its random-design corollaries is $`d^\mu`$;
- its instance-wise value $`V_{\mathcal Q}`$ is $`J_{\mathcal Q}`$, with $`\Delta_{\mathcal Q}`$ the suboptimality in instance $`\mathcal Q`$;
- its class radius $`\Lambda`$ is $`M`$, because $`\Lambda`$ is the vault's Gram matrix.

Kept as in the paper:

- $`\Sigma_D:=\frac1T\Phi^\top\Phi`$, the normalized, unregularized Gram matrix. It is the vault's $`\Lambda/T`$ at $`\lambda=0`$, so $`T^{-1/2}\|\Sigma_D^{-1/2}v\|_2=\|v\|_{(T\Sigma_D)^{-1}}`$.
- The width $`\beta`$ and the complexity $`\mathfrak C_q`$.
- Conjugate exponents, $`1/p+1/q=1`$ with $`1/\infty=0`$.

One shorthand is added: $`\bar\phi_\pi:=\mathbb E_{x\sim\nu}[\phi(x,\pi(x))]`$, the mean feature of a deterministic policy, which the paper writes out in full.

**One difference is substantive, and is restated rather than translated.** The design is *fixed*: the pairs $`(x_t,a_t)`$ are arbitrary and only the rewards are random. And $`\nu`$ is a *known* evaluation distribution with no link to the logged contexts. So $`\nu`$ here is only the distribution $`J`$ averages over, not the anchor's data-generating context law. Every guarantee is conditional on the design. Only the tabular corollaries return to the anchor's i.i.d. data, with $`d^\mu`$ the law of the logged pairs.

### Setting and learning protocol

A known feature map $`\phi:\mathcal X\times\mathcal A\to\mathbb R^d`$, with $`q^*(x,a)=\phi(x,a)^\top\theta^*`$ for an unknown $`\theta^*\in\mathbb R^d`$. Write $`\phi_t:=\phi(x_t,a_t)`$, let $`\Phi\in\mathbb R^{T\times d}`$ have rows $`\phi_t^\top`$, and let $`r:=(r_1,\dots,r_T)^\top`$. Policies are deterministic, $`\Pi=\mathcal A^{\mathcal X}`$.

```math
\begin{aligned}
&\textbf{Given: } \text{features } \phi,\ \text{test distribution } \nu,\ \text{a fixed design } \{(x_t,a_t)\}_{t=1}^T \text{ with } \Sigma_D:=\tfrac1T\textstyle\sum_t\phi_t\phi_t^\top\succ0 \\
&\textbf{for } t=1,\dots,T: \\
&\qquad \textbf{observe } r_t\sim\rho(\cdot\mid x_t,a_t) \qquad \triangleright\ \text{independent; mean } \phi_t^\top\theta^*,\ 1\text{-sub-Gaussian} \\
&\hat\pi\leftarrow\mathsf{Alg}(\mathcal D,\phi,\nu),\quad \mathcal D:=\{(x_t,a_t,r_t)\}_{t=1}^T \qquad \triangleright\ \text{no interaction} \\
&\textbf{return } \hat\pi:\mathcal X\to\mathcal A,\ \text{scored by } \Delta(\hat\pi)
\end{aligned}
```

Three features of this loop carry the difficulty:

- **Only the noise is random.** Coverage is a property of the fixed design and enters only through $`\Sigma_D`$. There is no behavior policy to estimate, and no propensity is used.
- **The rule sees $`\nu`$.** $`\hat\pi_p`$ maximizes a $`\nu`$-average, so its choice at one context depends on all the others. The per-pair rules it is compared with, tabular LCB and PEVI, never use $`\nu`$.
- **Nothing is scaled.** There is no bound on $`\|\theta^*\|`$ or $`\|\phi\|`$. Everything is measured in the whitened geometry of $`\Sigma_D`$.

### Learning objective

$`J(\pi)=\mathbb E_{x\sim\nu}[q^*(x,\pi(x))]=\bar\phi_\pi^\top\theta^*`$ and $`\pi^*(x)\in\arg\max_a\phi(x,a)^\top\theta^*`$. The objective is the suboptimality gap of [[contextual-bandits-offline]] §2.3,

```math
\Delta(\hat\pi)=J(\pi^*)-J(\hat\pi)=\big(\bar\phi_{\pi^*}-\bar\phi_{\hat\pi}\big)^\top\theta^* ,
```

random through the reward noise given the design. Upper bounds hold with high probability. Lower bounds are minimax in expectation, over classes indexed by the **complexity** $`\mathfrak C_q`$:

```math
\mathfrak C_q:=\big\|\Sigma_D^{-1/2}\bar\phi_{\pi^*}\big\|_q,\qquad
\mathrm{CB}_q(M):=\Big\{\big(\nu,\{(x_t,a_t)\}_{t\le T},\theta^*,\rho\big):\ \mathfrak C_q\le M,\ \rho(\cdot\mid x,a)\ 1\text{-sub-Gaussian}\Big\},\qquad q\in[1,\infty).
```

Since $`\|v\|_q`$ is non-increasing in $`q`$, $`\mathrm{CB}_1(M)\subseteq\mathrm{CB}_q(M)`$ for every $`q`$.

### How this differs from the surrounding literature

- **Against BCP (Xie et al. 2021).** $`\hat\pi_2`$ *is* BCP at horizon one. What improves is the bound: the norm of the mean whitened feature replaces the mean of the norms, a Jensen gap.
- **Against PEVI (Jin, Yang & Wang 2021).** PEVI is pessimistic per pair, not per policy, and lies outside the family. Its quoted guarantee is looser by $`\sqrt d`$ and by the same Jensen gap, but it holds for every test distribution at once.
- **Against tabular LCB (Rashidinejad et al. 2021).** PUNC contains LCB. Corollary 1 is stated in the interaction of $`\nu`$ and $`d^\mu`$, where LCB's bound is stated in $`C^*`$ alone.
- **Against earlier lower bounds.** Those of Zanette, Wainwright & Brunskill (2021, Theorem 2) and Jin et al. (2021, Theorem 4.7) hold for $`p=q=2`$ and one fixed radius. Theorem 2 below is a nested family that exposes the dependence on $`M`$, and Jin et al.'s bound is loose by $`\sqrt d`$ (two-point testing).
- **New:** the $`\ell_\infty`$ rule, its adaptive minimax optimality, and the separation showing that no finite $`p`$ has it.

## Assumptions

The paper states its assumptions in §2–§3 without numbering them. In order:

1. **Linear realizability with known features:** $`q^*(x,a)=\phi(x,a)^\top\theta^*`$ at every pair.
2. **Sub-Gaussian rewards:** each $`\rho(\cdot\mid x,a)`$ is 1-sub-Gaussian ($`\mathbb P[\lvert X-\mathbb EX\rvert\ge t]\le2e^{-t^2/2}`$), and the rewards are independent given the design.
3. **Fixed design.** The random-design tabular results (Corollaries 1–2, Propositions 2 and 5) instead draw $`(x_t,a_t)`$ i.i.d. from $`d^\mu`$ and add a count condition.
4. **$`\Sigma_D`$ invertible.** The paper says a ridge $`\Sigma_D+\lambda I`$ would accommodate the singular case, but does not carry it out.
5. **A known test distribution $`\nu`$, entering the rule,** and $`\Pi=\mathcal A^{\mathcal X}`$, so $`\pi^*\in\Pi`$.

**Not assumed:**

- bounded rewards;
- any norm bound on $`\theta^*`$ or $`\phi`$ (only the plug-in analysis of Appendix E uses $`\|\phi\|_2\le B`$);
- i.i.d. contexts, or any relation between the logged contexts and $`\nu`$;
- propensities;
- any coverage assumption: $`\mathfrak C_q`$ appears only in the bound, and PUNC is not told which class the instance lies in;
- gaps or margins;
- sample splitting.

**Relaxation.** Corollary 1 moves to random design by a Chernoff bound. The price is a count condition, $`T\gtrsim\ln(S/\delta)/\min_xd^\mu(x,\pi^*(x))`$.

## Method

### The confidence-set template (§3.1)

For a set $`\Theta\subseteq\mathbb R^d`$,

```math
\underline J_\Theta(\pi):=\inf_{\theta\in\Theta}\bar\phi_\pi^\top\theta,\qquad \hat\pi_\Theta\in\arg\max_{\pi\in\mathcal A^{\mathcal X}}\underline J_\Theta(\pi).
```

**Definition 1 (pessimism-validity).** $`\Theta`$ is $`(\beta,\delta)`$ pessimism-valid under a norm $`\|\cdot\|`$ on $`\mathbb R^d`$ if, with probability at least $`1-\delta`$, both (1) $`\theta^*\in\Theta`$ and (2) $`\sup_{\theta\in\Theta}\|\theta^*-\theta\|\le\beta`$.

**Proposition 1.** If $`\Theta`$ is $`(\beta,\delta)`$ pessimism-valid under $`\|\cdot\|`$, then with probability at least $`1-\delta`$, $`\ \Delta(\hat\pi_\Theta)\le\beta\,\|\bar\phi_{\pi^*}\|_*`$, where $`\|\cdot\|_*`$ is the dual norm.

*Proof idea.* The proof uses the paper's decomposition (4):

```math
\Delta(\hat\pi)=\big[J(\pi^*)-\underline J(\pi^*)\big]+\big[\underline J(\pi^*)-\underline J(\hat\pi)\big]+\big[\underline J(\hat\pi)-J(\hat\pi)\big].
```

The middle term is at most $`0`$ by the choice of $`\hat\pi`$, and the last is at most $`0`$ because $`\theta^*\in\Theta`$. The first is $`\sup_{\theta\in\Theta}\bar\phi_{\pi^*}^\top(\theta^*-\theta)\le\beta\|\bar\phi_{\pi^*}\|_*`$. This is Lemma 2.14 of [[contextual-bandits-offline-value-based]] read over policies. The difference is the width: here it is a norm of the mean feature, not an average of per-pair widths.

### The $`\ell_p`$ family and PUNC (§3.2)

```math
\hat\theta:=(\Phi^\top\Phi)^{-1}\Phi^\top r,\qquad
\Theta_p:=\Big\{\theta\in\mathbb R^d:\ \big\|\Sigma_D^{1/2}(\theta-\hat\theta)\big\|_p\le\beta/2\Big\}.
```

**Lemma 1.** Fix $`\delta\in(0,1)`$ and set $`\beta=\beta_p:=d^{1/p}\sqrt{8\ln(d/\delta)/T}`$. Then $`\Theta_p`$ is $`(\beta_p,\delta)`$ pessimism-valid under $`\|v\|:=\|\Sigma_D^{1/2}v\|_p`$.

*Technique.* Write $`\eta`$ for the noise vector. Then

```math
\Sigma_D^{1/2}(\hat\theta-\theta^*)=T^{-1/2}(\Phi^\top\Phi)^{-1/2}\Phi^\top\eta ,
```

and each row of $`(\Phi^\top\Phi)^{-1/2}\Phi^\top`$ is a unit vector. So every whitened coordinate is $`T^{-1/2}`$ times a 1-sub-Gaussian variable. A union bound over the $`d`$ coordinates bounds the $`\ell_\infty`$ norm by $`\sqrt{2\ln(d/\delta)/T}=\beta_\infty/2`$. Every other $`p`$ follows from $`\|v\|_p\le d^{1/p}\|v\|_\infty`$, and the diameter bound follows from the triangle inequality.

The whole family is therefore one $`\ell_\infty`$ concentration, loosened by $`d^{1/p}`$. With these widths the box $`\Theta_\infty`$ sits inside every $`\Theta_p`$, so $`\underline J_\infty\ge\underline J_p`$ policy by policy. That inclusion is this page's observation, not the paper's.

**Theorem 1.** For any $`p\ge1`$, with probability at least $`1-\delta`$,

```math
\Delta(\hat\pi_p)\ \le\ d^{1/p}\sqrt{\frac{8\ln(d/\delta)}{T}}\ \big\|\Sigma_D^{-1/2}\bar\phi_{\pi^*}\big\|_q,\qquad\frac1p+\frac1q=1 .
```

Since $`\|v\|_1\le d^{1-1/q}\|v\|_q=d^{1/p}\|v\|_q`$, the $`p=\infty`$ bound, $`\sqrt{8\ln(d/\delta)/T}\,\mathfrak C_1`$, is never larger than the bound for any finite $`p`$. The logarithm is the same for every $`p`$, so this domination of the bounds is exact. It is a statement about upper bounds; Theorem 3 is what shows the ordering is real.

**The max-only form (Eq. 10).** The infimum over $`\Theta_p`$ has a closed form, which turns the rule into a penalized maximization:

```math
\hat\pi_p\in\arg\max_{\pi\in\mathcal A^{\mathcal X}}\Big\{\bar\phi_\pi^\top\hat\theta-\frac{\beta_p}2\,\big\|\Sigma_D^{-1/2}\bar\phi_\pi\big\|_q\Big\}.
```

At $`q=1`$, write $`u:=\Sigma_D^{1/2}\theta`$, $`\hat u:=\Sigma_D^{1/2}\hat\theta`$ and $`w:=\Sigma_D^{-1/2}\bar\phi_\pi`$, so that $`\bar\phi_\pi^\top\theta=w^\top u`$. The infimum over the box is then $`\sum_j\big(w_j\hat u_j-\lvert w_j\rvert\beta_\infty/2\big)`$: each coordinate sits at the lower end of its interval where $`w_j>0`$ and at the upper end where $`w_j<0`$.

```math
\begin{aligned}
&\textbf{Input: } \mathcal D=\{(x_t,a_t,r_t)\}_{t=1}^T,\ \text{features } \phi,\ \text{test distribution } \nu,\ p\in[1,\infty],\ \delta \\
&\Sigma_D\leftarrow\tfrac1T\textstyle\sum_t\phi_t\phi_t^\top,\qquad \hat\theta\leftarrow(T\Sigma_D)^{-1}\textstyle\sum_t\phi_tr_t \qquad \triangleright\ \text{OLS; } \Sigma_D\succ0 \text{ assumed} \\
&\beta_p\leftarrow d^{1/p}\sqrt{8\ln(d/\delta)/T},\qquad q\leftarrow\text{the conjugate of } p \\
&\textbf{for each } \pi\in\mathcal A^{\mathcal X}: \\
&\qquad \bar\phi_\pi\leftarrow\mathbb E_{x\sim\nu}[\phi(x,\pi(x))] \qquad \triangleright\ \nu \text{ is known} \\
&\qquad \underline J_p(\pi)\leftarrow\bar\phi_\pi^\top\hat\theta-\tfrac{\beta_p}2\big\|\Sigma_D^{-1/2}\bar\phi_\pi\big\|_q \qquad \triangleright\ =\inf_{\theta\in\Theta_p}\bar\phi_\pi^\top\theta \\
&\textbf{return } \hat\pi_p\in\arg\max_\pi\underline J_p(\pi) \qquad \triangleright\ p=\infty \text{ is PUNC}
\end{aligned}
```

**The loop over $`\mathcal A^{\mathcal X}`$ is a definition, not an algorithm.** The penalty is a norm of a $`\nu`$-average, so the maximization does not decouple across contexts, and the paper treats the family statistically. Computation comes up in only two places. The paper reads Zanette, Wainwright & Brunskill's actor-critic PACLE as an efficient solver for $`p=2`$, and believes it extends to every $`p`$. And in the tabular case $`q=1`$ decouples.

PUNC stands for Pessimism via Uniform Norm Confidence.

### The special cases (§3.3)

**$`p=2`$ is BCP.** In a linear bandit, the empirical Bellman error of Xie et al. (their Eq. 3.1) is the excess squared loss,

```math
\frac1T\sum_{t=1}^T\big(\phi_t^\top\theta-r_t\big)^2-\min_{\theta'\in\mathbb R^d}\frac1T\sum_{t=1}^T\big(\phi_t^\top\theta'-r_t\big)^2=\big\|\Sigma_D^{1/2}(\theta-\hat\theta)\big\|_2^2 .
```

So BCP's version space is an $`\ell_2`$ ball around $`\hat\theta`$, and "BCP exactly matches the learning rule $`\hat\pi_2`$". The paper quotes Xie et al.'s guarantee, up to logarithms, as $`\sqrt{d/T}\,\mathbb E_{x\sim\nu}\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_2`$. Theorem 1 gives $`\sqrt{d/T}\,\|\Sigma_D^{-1/2}\bar\phi_{\pi^*}\|_2`$, which is never larger, by Jensen.

**$`p=\infty`$ contains tabular LCB.** With $`\phi(x,a)=e_{(x,a)}`$ and $`d=SK`$, $`\Sigma_D`$ is diagonal with entries $`N(x,a)/T`$. The $`q=1`$ penalty becomes $`\sum_x\nu(x)\sqrt{2\ln(SK/\delta)/N(x,\pi(x))}`$. That is a sum over contexts, so the maximization decouples into

```math
\hat\pi_\infty(x)\in\arg\max_{a\in\mathcal A}\ \hat q(x,a)-\sqrt{\frac{2\ln(SK/\delta)}{N(x,a)}} ,
```

which is the LCB rule of Rashidinejad et al. (2021), up to the choice of width.

**Corollary 1 (tabular, random design).** If the pairs are drawn i.i.d. from $`d^\mu`$, then with probability at least $`1-\delta`$,

```math
\Delta(\hat\pi_\infty)\ \lesssim\ \sqrt{\frac{\ln(SK/\delta)}{T}}\ \sum_x\frac{\nu(x)}{\sqrt{d^\mu(x,\pi^*(x))}}\qquad\text{if}\quad T\gtrsim\frac{\ln(S/\delta)}{\min_xd^\mu(x,\pi^*(x))} .
```

Let $`C^*:=\max_x\nu(x)/d^\mu(x,\pi^*(x))`$. When the logged contexts are drawn from $`\nu`$, this is the vault's $`C^*=\max_x1/\mu(\pi^*(x)\mid x)`$.

- **Recovering LCB.** By Cauchy–Schwarz the sum is at most $`\sqrt{SC^*}`$. This recovers Rashidinejad et al.'s $`\sqrt{SC^*\ln(SK/\delta)/T}`$, which is optimal when $`C^*\ge2`$.
- **The sum itself is finer.** It lets $`\nu`$ and $`d^\mu`$ interact, where $`C^*`$ keeps only their worst ratio.
- **Every $`p`$.** Corollary 2 is the same statement for every $`\hat\pi_p`$, with a factor $`(SK)^{1/p}`$ and the $`\ell_q`$ aggregate. Appendix B.3 shows that all $`\hat\pi_p`$ reach $`\sqrt{SC^*/T}`$ up to $`K^{1/p}`$.

**PEVI is per pair, and outside the family.** Jin, Yang & Wang (2021) subtract a bonus at every pair:

```math
\hat\pi_{\mathrm{PEVI}}(x)\in\arg\max_{a\in\mathcal A}\ \phi(x,a)^\top\hat\theta-\beta\,\big\|\Sigma_D^{-1/2}\phi(x,a)\big\|_2 .
```

The confidence-set template recovers PEVI only if the infimum runs over functions:

```math
\hat\pi_{\mathrm{PEVI}}\in\arg\max_{\pi}\ \inf_{\theta(\cdot)}\ \mathbb E_{x\sim\nu}\big[\phi(x,\pi(x))^\top\theta(x)\big],\qquad \big\|\Sigma_D^{1/2}(\theta(x)-\hat\theta)\big\|_2\le\beta\ \text{ for all } x .
```

In the paper's words, "PEVI enlarges the $`\ell_2`$ confidence set by separately picking a pessimistic parameter $`\theta(s)`$ for each state". The paper quotes Jin et al.'s guarantee as $`\sqrt{d^2/T}\,\mathbb E_{x\sim\nu}\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_2`$. That is looser than Theorem 1 at $`p=2`$ by $`\sqrt d`$ and by the exchange of expectation and norm. "However, their guarantee holds for all test distributions", which the paper calls "a consequence of being pessimistic for every state".

**At horizon one the $`\sqrt d`$ is not intrinsic.** This is this page's reading, not the paper's. Theorem 4.3 of [[contextual-bandits-offline-value-based]] gives the ridge version of this rule

```math
\Delta\le2\beta_\delta T^{-1/2}\,\mathbb E_{x\sim\nu}\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_2 ,
```

with a radius $`\beta_\delta=\tilde O(\sqrt d)`$, because $`\Lambda\succeq T\Sigma_D`$ gives $`\|\phi\|_{\Lambda^{-1}}\le T^{-1/2}\|\Sigma_D^{-1/2}\phi\|_2`$. The $`\sqrt d`$ in the comparison comes from Jin et al.'s horizon-$`H`$ radius $`c\,dH\sqrt\zeta`$, which pays for covering the next-state value class. At horizon one only the Jensen gap remains.

### The rules in one frame

| Rule | Pessimistic about | Uses $`\nu`$ | Guarantee, up to logs |
|---|---|---|---|
| $`\hat\pi_2`$ = BCP | policy values | yes | $`\sqrt{d/T}\,\Vert\Sigma_D^{-1/2}\bar\phi_{\pi^*}\Vert_2`$ (Thm. 1) |
| $`\hat\pi_\infty`$ = PUNC | policy values | yes | $`T^{-1/2}\,\Vert\Sigma_D^{-1/2}\bar\phi_{\pi^*}\Vert_1`$ (Thm. 1) |
| PEVI | each pair | no | $`\sqrt{d^2/T}\,\mathbb E_\nu\Vert\Sigma_D^{-1/2}\phi(x,\pi^*(x))\Vert_2`$ (Jin et al., as quoted) |
| tabular LCB = tabular PUNC | each pair, which here is the same | no | $`\sqrt{\ln(SK/\delta)/T}\,\sum_x\nu(x)/\sqrt{d^\mu(x,\pi^*(x))}`$ (Cor. 1) |

### Lower bounds and adaptive optimality (§4)

Over $`\mathrm{CB}_q(M)`$, Theorem 1 gives $`\Delta_{\mathcal Q}(\hat\pi_p)\lesssim d^{1/p}\sqrt{\ln(d/\delta)/T}\,M`$ for every instance (Eq. 14).

**Theorem 2.** For every $`d\ge2`$ there is a feature map $`\phi`$ with the following property. For all conjugate $`p,q\ge1`$, whenever $`M\ge\sqrt8\,d^{1/q-1/2}`$ and $`T\ge d^{2/p}M^2`$,

```math
\inf_{\hat\pi}\ \sup_{\mathcal Q\in\mathrm{CB}_q(M)}\mathbb E\big[\Delta_{\mathcal Q}(\hat\pi)\big]\ \ge\ c\,d^{1/p}\,\frac{M}{\sqrt T},
```

with $`c>0`$ universal. At $`p=\infty,q=1`$ the bound holds for all $`M\ge2`$.

So $`\hat\pi_p`$ is minimax over its dual class $`\mathrm{CB}_q(M)`$ up to a $`\log d`$. The rate is $`\tilde\Theta(\sqrt{d/T}\,M)`$ over $`\mathrm{CB}_2`$ and $`\tilde\Theta(M/\sqrt T)`$ over $`\mathrm{CB}_1`$.

*Technique.* The hard instances are tabular, with $`S=d/2`$ contexts and $`K=2`$.

- The minimax suboptimality tensorizes over contexts into a sum of per-context Bayes lower bounds from Xiao et al. (2021, Theorem 1). This is Lemma 3: $`c\sum_x\nu(x)\max_aN(x,a)^{-1/2}`$.
- Rashidinejad et al.'s count construction is then tuned so that the instance satisfies the $`\ell_q`$ constraint.
- This tabular route cannot reach below $`M\gtrsim d^{-1/p}`$ (Appendix C.1).

**Adaptive minimax optimality (§4.2).** Theorem 1 at $`p=\infty`$ and $`\|v\|_1\le d^{1/p}\|v\|_q`$ give, for every conjugate pair,

```math
\Delta(\hat\pi_\infty)\ \lesssim\ \sqrt{\frac{\ln(d/\delta)}{T}}\ \mathfrak C_1\ \le\ d^{1/p}\sqrt{\frac{\ln(d/\delta)}{T}}\ \mathfrak C_q .
```

So PUNC is minimax over every $`\mathrm{CB}_q(M)`$ simultaneously, up to $`\log d`$, without being told $`q`$. The paper did not investigate removing the $`\log d`$. Its footnote 5 allows that $`\hat\pi_2`$ may beat $`\hat\pi_\infty`$ over $`\mathrm{CB}_2`$ by $`\sqrt{\log d}`$.

**Theorem 3 (informal), stated formally as Theorem 4: the adaptivity is unique to $`p=\infty`$.** Fix any $`p\ge1`$, $`d\ge20`$, $`T\ge9d^3`$, and a width multiplier $`\xi(d,T)\ge K_\xi`$ for an absolute constant $`K_\xi>0`$. There is an instance $`\mathcal Q\in\mathrm{CB}_1(\sqrt{8d})`$ on which:

- with probability at least $`1/4`$, $`\hat\pi_p`$ with $`\beta=\xi\,d^{1/p}/\sqrt T`$ has $`\Delta(\hat\pi_p)\ge(K_\xi/\sqrt8)\,d^{1/p+1/2}/\sqrt T`$, so its expected suboptimality is $`\Omega(K_\xi d^{1/p+1/2}/\sqrt T)`$;
- $`\hat\pi_\infty`$ with $`\beta=\sqrt{8\ln(K_\xi d^{5/2}/\sqrt8)/T}`$ has $`\mathbb E_{\mathcal D}[\Delta(\hat\pi_\infty)]\le c\sqrt{d\ln(K_\xi d)/T}`$.

With $`M=\sqrt{8d}`$, the first is $`d^{1/p}M/\sqrt T`$ up to constants and the second is $`\tilde O(M/\sqrt T)`$. So every finite $`p`$ loses $`d^{1/p}`$ over $`\mathrm{CB}_1`$.

The instance is tabular, with $`S=d/2`$ contexts, two actions and uniform $`\nu`$. At every context but one, only the optimal action is logged. At that one context, the optimal action has gap $`\gamma\propto S^{1/p+3/2}/\sqrt T`$ and is logged $`T/(9S^3)`$ times, against a mean-zero decoy logged about $`T/S`$ times. In the paper's words, "only one direction determines the difficulty of the offline learning problem".

**What this means for $`\hat\pi_2`$.** Theorem 1 at $`p=2`$, with norm inequalities, gives

```math
\Delta(\hat\pi_2)\ \lesssim\ \begin{cases} d^{1/p}\sqrt{\ln(d/\delta)/T}\ \mathfrak C_q, & q\ge2,\\[2pt] \sqrt{d\ln(d/\delta)/T}\ \mathfrak C_q, & q\in[1,2],\end{cases}
```

where $`p`$ is the conjugate of the class index $`q`$. Theorems 2–3 make both cases tight up to logarithms.

- $`\hat\pi_2`$ is adaptively optimal over the classes with $`q\ge2`$.
- It may need $`\Omega(dM^2)`$ samples over $`\mathrm{CB}_1(M)`$, where $`M^2`$ suffice.
- In general, $`\hat\pi_{\tilde p}`$ is adaptively optimal only for $`q\ge\tilde p/(\tilde p-1)`$ (Figure 1).

**A caveat from this page's check.** Appendix D rewrites $`\hat\pi_p`$'s tabular penalty with an $`\ell_p`$ norm of the whitened mean feature, where Eq. (10) has an $`\ell_q`$ norm. The separation's scaling is set by the one hard context's coordinate, which dominates either norm. So the reading here does not depend on which norm is meant.

### Where the $`\ell_1`$ complexity stands (§4.3–§4.4)

- **Single-radius lower bounds.** Zanette et al. (2021, Theorem 2) give $`d^{3/2}/\sqrt T`$ over $`\mathrm{CB}_2(M=d)`$. Jin et al. (2021, Theorem 4.7) give $`1/\sqrt T`$ over $`\mathrm{CB}_2(\Theta(1))`$, loose by $`\sqrt d`$. Theorem 2's nested classes expose the dependence on $`M`$.
- **Against $`C^*`$.** $`\mathrm{CB}_{\mathrm{conc}}(C^*)\subseteq\mathrm{CB}_1(\sqrt{SC^*})`$.
  - For $`C^*\ge2`$ the two minimax rates coincide. For $`C^*\in[1,2)`$ they do not. At $`C^*=1`$ the rate over $`\mathrm{CB}_{\mathrm{conc}}`$ is $`S/T`$, while over $`\mathrm{CB}_1(\sqrt S)`$ it is $`\sqrt{S/T}`$. So $`\mathfrak C_1`$ loses the fast rates of near-expert data.
  - In the other direction, take uniform $`\nu`$, $`d^\mu(1,\pi^*(1))=1/S^3`$ and $`d^\mu(x,\pi^*(x))=1/S`$ for $`x\ge2`$. This instance has $`C^*=S^2`$, which gives a guarantee of $`S^{3/2}/\sqrt T`$. But $`\mathfrak C_1=O(\sqrt S)`$, which gives $`\sqrt{S/T}`$.
- **Rotation.** $`\mathfrak C_1`$ and PUNC are not rotation invariant, and "$`\mathfrak C_2`$ is the only rotational invariant complexity". Rotating the features by $`U`$ changes $`\mathfrak C_1(U):=\|U\Sigma_D^{-1/2}\bar\phi_{\pi^*}\|_1`$ by up to $`\sqrt d`$. An infimum over $`U`$ inside the $`\ell_1`$ set returns $`\Theta_2`$.
- **Local minimax.** Yin & Wang (2021)'s local-minimax Theorem 4.3, stated in $`\mathfrak C_1`$, "seems incorrect" to the authors. Their Appendix F gives a two-armed instance on which a likelihood-ratio test achieves $`e^{-cT}`$: with a large reward gap, policy learning is easier than optimal-value estimation.

### Two further results

- **The plug-in rule (Appendix E).** For $`\|\phi\|_2\le B`$, the greedy rule on $`\hat\theta`$ has $`\Delta\le\sqrt{8d\ln(d/\delta)/T}\,B\,\lambda_{\min}(\Sigma_D)^{-1/2}`$ (Proposition 4, "folklore"). That dependence on $`B\,\lambda_{\min}(\Sigma_D)^{-1/2}`$ is always worse than $`\hat\pi_2`$'s, whose complexity is $`\mathfrak C_2`$. Proposition 5 is a $`K`$-armed bandit, $`K\ge8`$, on which the plug-in rule has $`\mathbb E[\Delta]\ge c_1`$ for $`T\le2^K`$, while PUNC has $`\mathbb E[\Delta]\le c_2\sqrt{\ln(KT)/T}`$ for $`T\ge200`$.
- **Validity is not necessary (Appendix B.3).** In the tabular case, run $`\hat\pi_2`$ with $`\beta:=\sqrt{16S\ln(SK/\delta)/T}`$. For $`\delta\in(0,1/2)`$ and $`T\gtrsim\ln(S/\delta)/\min_xd^\mu(x,\pi^*(x))`$, it attains $`\Delta(\hat\pi_2)\lesssim\sqrt{SC^*\ln(SK/\delta)/T}`$ (Proposition 2). That removes the $`K^{1/2}`$ which Corollary 2 leaves at $`p=2`$.
  - Yet by the paper's remark, this rule's set does not contain $`\theta^*`$ with high probability.
  - The proof bounds two terms of the decomposition directly, $`J(\pi^*)-\underline J(\pi^*)`$ (15a) and $`\underline J(\hat\pi_2)-J(\hat\pi_2)\le0`$ (15b), by Hoeffding at the pairs and Cauchy–Schwarz.
  - Hence, in the paper's words, "proof techniques relying on assuming $`\theta^\star\in\Theta`$ … can be fundamentally loose".

### Experiments

$`d=100`$, one context, and the unit ball as the feature set, so a policy is a unit vector with $`J(\pi)=\pi^\top\theta^*`$. There are 100 trials, with $`T`$ from $`10^3`$ to $`10^5`$.

- **Random rotation.** With $`\phi_t\sim\mathcal N(0,QDQ^\top)`$ for a random rotation $`Q`$, $`\theta^*=Qe_{20}`$ and $`D_{ii}\propto1/i`$, $`\hat\pi_2`$ and $`\hat\pi_\infty`$ are close, and $`\mathfrak C_1\approx\sqrt d\,\mathfrak C_2`$.
- **Basis-aligned.** With $`\phi_t\sim\mathcal N(0,D)`$ and $`\theta^*=e_{20}`$, $`\hat\pi_\infty`$'s suboptimality falls orders of magnitude below $`\hat\pi_2`$'s, and $`\mathfrak C_1\ll\sqrt d\,\mathfrak C_2`$.
- **Plug-in.** The plug-in rule is the worst in both.

## Connections

- [[contextual-bandits-offline-value-based]] — §4.4 carries this paper's linear results, and §1.3 files the confidence-set template as policy-level pessimism. That page's Theorem 4.3 is the per-pair (PEVI) side of the comparison above, and the discussion after its Definition 2.10 cites pessimism-validity.
- [[contextual-bandits-offline]] — the setting. This paper changes it in two ways: a fixed design, and a known evaluation distribution that enters the rule.
- [[pessimism-principle]] — the template $`\arg\max_\pi[\hat v(\pi)-\Gamma(\pi)]`$. Here $`\Gamma(\pi)=\frac{\beta_p}2\|\Sigma_D^{-1/2}\bar\phi_\pi\|_q`$ is a norm of the policy's mean feature, not an average of per-pair widths.
- [[coverage-coefficient]] — $`\mathfrak C_q`$ is a single-policy coverage quantity in feature space; §4.3 above relates $`\mathfrak C_1`$ to $`C^*`$.
- [[qom-value-based-pessimism]] and [[qom-value-based-pessimism-subgaussian]] — the quantile-of-means route to value-based pessimism. Its linear case meets this paper in the Open Questions below.
- **Concepts this paper introduces:** pessimism-validity (Definition 1), PUNC, the $`\ell_q`$-constrained classes $`\mathrm{CB}_q(M)`$ with their complexity $`\mathfrak C_q`$, and adaptive minimax optimality. None has a concept page. Each is fully explained here and used elsewhere only by reference.
- **Extends:** Rashidinejad et al. (2021)'s tabular LCB, which PUNC contains, and Xie et al. (2021)'s BCP at horizon one, which is $`\hat\pi_2`$, with a sharper bound. **Challenges:** Yin & Wang (2021), Theorem 4.3 (Appendix F).
- Gene Li, Cong Ma, Nathan Srebro. Cong Ma is also a coauthor of Rashidinejad et al. (2021), the LCB paper this one generalizes. Not to be confused with `li2022InstanceOptimal` (Zhaoqi Li et al.), an online PAC paper.
- Xie, Cheng, Jiang, Mineiro & Agarwal (2021); Jin, Yang & Wang (2021); Rashidinejad, Zhu, Ma, Jiao & Russell (2021); Zanette, Wainwright & Brunskill (2021); Xiao et al. (2021); Yin & Wang (2021) — cited author–year, no pages here.

## Open Questions

**The paper's own:**

- **MDPs.** Extend the family beyond $`H=1`$, possibly by modifying PACLE to solve any $`\ell_p`$ rule.
- **Gap-dependent bounds** offline.
- **General function approximation.** $`\hat\pi_2`$ is a version space of small squared error; for PUNC, "no such interpretation exists".
- **Small radii.** Does Theorem 2 extend to $`M=O(d^{1/q-1/2})`$, and are faster rates possible there, as they are for tabular LCB? Proposition 3 extends the range for $`q\in(1,2]`$, at the price of $`T\ge d^{2/q}M^2`$, which the paper conjectures is an artifact.
- **The $`\log d`$** in Theorem 1, and whether $`\hat\pi_2`$ beats $`\hat\pi_\infty`$ over $`\mathrm{CB}_2`$ by $`\sqrt{\log d}`$.
- **Rotation.** Learn a rotation $`\hat U`$ for the data, then solve the $`\ell_\infty`$ problem.
- **Instance-dependent optimality** for offline learning.
- **Analyses that do not need $`\theta^*\in\Theta`$,** for general function classes (Appendix B.3).

**Untested or unclaimed:**

- **Computation.** Nothing is claimed for $`p\ne2`$ outside the tabular case.
- **An unknown $`\nu`$.** The rule needs $`\nu`$. Replacing it by the logged contexts is not analyzed, whereas the per-pair rules need no $`\nu`$.
- **Experiments.** They are one-context and synthetic, and PEVI is not run.

**For the QoM question.** This concerns the linear Approach 1 of [[qom-value-based-pessimism]]: per-batch least squares with a phantom round, then a per-pair quantile of the batch predictions, where the least-squares weights can be negative. What this paper supports, and where it stops:

- **Validity is needed only along the functionals the rule compares.** Proposition 1's proof uses two facts: $`\underline J(\hat\pi)\le J(\hat\pi)`$, and a bound on $`J(\pi^*)-\underline J(\pi^*)`$. $`\theta^*\in\Theta`$ is one way to get the first uniformly, and Proposition 2 attains the optimal tabular rate without it.
  - So a per-pair quantile of predictions, $`\underline q(x,a)`$, is the right kind of object: pessimism defined through predicted values, not through a parameter set.
  - An analysis of Approach 1 then needs $`\underline q\le q^*`$ only at the pairs the greedy step can pick, plus a width at $`\pi^*`$'s pairs. That is the proof of Lemma 2.14 of [[contextual-bandits-offline-value-based]], which uses only these two one-sided facts.
- **Per pair versus per policy.** Approach 1 is per pair, the PEVI side of the paper's comparison. By that comparison it pays the width averaged over contexts, $`\mathbb E_\nu[w(x,\pi^*(x))]`$, where a per-policy rule pays the width of the averaged feature. In exchange it needs no $`\nu`$ and holds for every test distribution.
  - **The gap, made explicit.** The following is derived here, not in the paper. On Lemma 1's event, $`\Gamma(x,a):=\frac{\beta_\infty}2\|\Sigma_D^{-1/2}\phi(x,a)\|_1`$ is a valid uncertainty quantifier at every pair at once. So (LCB) with it has $`\Delta\le\beta_\infty\,\mathbb E_\nu\|\Sigma_D^{-1/2}\phi(x,\pi^*(x))\|_1`$, against PUNC's $`\beta_\infty\|\Sigma_D^{-1/2}\bar\phi_{\pi^*}\|_1`$. The two differ by exactly the Jensen gap.
  - **The gap vanishes in the tabular case,** where the whitened features of different contexts have disjoint supports. That is where QoM-LCB lives, so it loses nothing to the per-policy rules. Under the count condition, the leading term of its Theorem 1 bound has Corollary 1's per-context form, with $`\sigma(x,\pi^*(x))\sqrt B`$ in place of $`\sqrt{\ln(SK/\delta)}`$.
  - **A policy-level QoM,** a quantile over batches of $`\bar\phi_\pi^\top\hat\theta^b`$, would work with each policy's own value functional rather than an average of per-pair widths. But it needs validity uniformly over the policies it compares. Lemma 1 gets that uniformity for free from one parameter set; a quantile would have to earn it by a union bound over policies. What width it would then pay is not settled here.
- **Signs are handled coordinate by coordinate.** The box certifies each whitened coordinate of $`\theta^*`$, $`(\Sigma_D^{1/2}\theta^*)_j`$, separately. That is a union over $`d`$ coordinates, not over pairs, so validity holds at a continuum of contexts at once. A signed functional $`w^\top u`$ is then lower-bounded by taking each coordinate's lower end where $`w_j>0`$ and its upper end where $`w_j<0`$ (the max-only form at $`q=1`$). Negative weights never enter validity.
  - **A QoM analogue** would replace each coordinate's interval by a lower and an upper batch quantile of $`(\Sigma_D^{1/2}\hat\theta^b)_j`$. The paper does not go there.
  - **The idea relocates the sign problem rather than removing it.** Each whitened coordinate is itself a signed combination of rewards, so Feige's non-negative route fails there too. A two-sided undershoot property would be needed, such as the symmetric-noise (U) of [[qom-value-based-pessimism-subgaussian]]. The analogue would also inherit PUNC's dependence on rotation.
