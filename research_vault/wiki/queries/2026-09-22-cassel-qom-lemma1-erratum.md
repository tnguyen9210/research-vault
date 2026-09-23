---
date: 2026-09-22
question: "Are Lemma 1 and Corollary 2 of Cassel & Rosenberg (2026) correct as stated? They are the two results the quantile-of-means estimator rests on, and the offline adaptation would cite them rather than reprove them."
---

# Cassel & Rosenberg's Lemma 1 and Corollary 2, Checked

**Scope.** §2 of [[cassel2026Quantile]] — the definition of the quantile-of-means (QoM) estimator and the two guarantees stated for it. The occasion was a decision to build the tabular offline section of the Overleaf note on those results directly rather than on the versions reproved in [[2026-09-22-qom-value-based-pessimism]] and [[2026-09-22-qom-value-based-pessimism-subgaussian]]. Statements below were read from the rendered PDF pages, not from a text extraction, because the extraction garbles the radicals — and a radical is exactly where the error is.

## Short answer

- **Corollary 2 is correct**, with one corner case: the proof divides by $`\mu`$, so the strict inequality it states fails when $`\mu=0`$. The non-strict form holds always and is the one to use offline, where an action with mean reward $`0`$ is an ordinary cell.
- **Lemma 1(1), optimism, is correct.** This is the interesting half — it is what replaces an uncertainty bonus — and it survives intact.
- **Lemma 1(2), the bias bound, is false as stated.** Its constant $`1.7`$ comes from a corollary to Freedman's inequality that drops a $`\log(1/\delta)`$ from under a square root. The constant the same route supports is $`4.69`$. At $`\delta=0.01`$ the lemma claims probability $`\ge0.99`$ where the truth is $`\le0.21`$, so this is not slack in a proof but a false statement.
- **The paper's own related work corroborates it.** It criticises Cassel et al. (2025) because "their bias bound incurs additional logarithmic factors" — and those factors are precisely the $`\sqrt{\log(1/\delta)}`$ that Lemma 1(2) is missing. The earlier paper was most likely right.

## 1. What the paper states

Let $`X_1,\dots,X_n`$ be i.i.d. non-negative with mean $`\mu`$. For a sequence $`\hat\mu^1,\dots,\hat\mu^B\in\mathbb R`$ the quantile at level $`\alpha\in[0,1]`$ is

```math
q_\alpha\big(\hat\mu^b,\ b\in[B]\big)\ \overset{\text{def}}{=}\ \hat\mu^{(\lceil\alpha B\rceil)},
```

where $`\hat\mu^{(1)},\dots,\hat\mu^{(B)}`$ is the sequence sorted in **ascending** order (their Eq. 1). Given a partition of the $`n`$ samples into $`B`$ fixed disjoint subsets $`\mathcal D^1,\dots,\mathcal D^B`$, the **quantile-of-means estimator** is

```math
\hat\mu_\alpha\ \overset{\text{def}}{=}\ q_\alpha\Big(\sum_{X\in\mathcal D^b}\frac{X}{\lvert\mathcal D^b\rvert+1},\ b\in[B]\Big),
```

their Eq. (2). Note the $`+1`$ in the denominator: the batch statistic is a **shrunk** mean, not a mean.

**Lemma 1.** Let $`X_1,\dots,X_n`$ be i.i.d. non-negative, bounded almost surely by $`R`$, with mean $`\mu`$ and variance $`\sigma^2`$. Let $`\hat\mu_\alpha`$ be the estimator above with $`B\ge26\log\delta^{-1}`$ batches and $`\alpha=1/65`$. Then each of the following holds individually with probability at least $`1-\delta`$:

1. *(Optimism)* $`\hat\mu_\alpha\le\mu`$.
2. *(Bias)* If $`\lvert\mathcal D^b\rvert\ge\lfloor n/B\rfloor`$ for all $`b\in[B]`$, then $`\hat\mu_\alpha\ge\mu-1.7\sqrt{\frac{\sigma^2B}{\max\lbrace1,n\rbrace}}-\frac{9RB}{\max\lbrace1,n\rbrace}`$.

**Corollary 2.** Let $`X_1,\dots,X_n`$ be i.i.d. non-negative with mean $`\mu`$. Then for any $`c\ge1/12`$, $`\Pr\big[\sum_{i=1}^{n}\frac{X_i}{n+c}<\mu\big]\ge1/13`$. *Proof:* apply their Lemma 13 (Feige 2004, Theorem 1) with $`X_i/\mu`$ and $`\delta=c`$.

## 2. Corollary 2: correct, with one corner case

Feige's Theorem 1 states: for independent non-negative $`X_i`$ with $`\mathbb E X_i\le1`$ and $`X=\sum_iX_i`$ of mean $`\mu_X`$, $`\Pr[X<\mu_X+\delta]\ge\min\lbrace\delta/(1+\delta),\,1/13\rbrace`$. Put $`Y_i:=X_i/\mu`$, so $`\mathbb E Y_i=1`$ and $`\mathbb E\sum_iY_i=n`$, and apply it with $`\delta=c`$:

```math
\Pr\Big[\sum_iY_i<n+c\Big]\ \ge\ \min\Big\lbrace\frac{c}{1+c},\ \frac1{13}\Big\rbrace\ =\ \frac1{13}\quad\text{for }c\ge\tfrac1{12},
```

since $`c\mapsto c/(1+c)`$ is increasing and equals $`1/13`$ at $`c=1/12`$. Multiplying through by $`\mu/(n+c)`$ gives the corollary. Checked against Feige's original statement.

**The corner case.** The proof divides by $`\mu`$, so it says nothing at $`\mu=0`$ — and there the conclusion is actually false: $`\mu=0`$ with $`X_i\ge0`$ forces $`X_i=0`$ almost surely, so $`\sum_iX_i/(n+c)=0=\mu`$ and the strict event $`\lbrace\cdot<\mu\rbrace`$ has probability $`0`$, not $`1/13`$. The non-strict version $`\Pr[\,\cdot\le\mu\,]\ge1/13`$ holds at every $`\mu\ge0`$. Online this is invisible, since a loss that is identically zero is not an interesting arm. Offline it is not: an action that never pays is an ordinary cell, and the validity event has to hold there too. Use the non-strict form.

## 3. Lemma 1(1): correct

Each batch statistic is $`\sum_{X\in\mathcal D^b}X/(\lvert\mathcal D^b\rvert+1)`$, which is Corollary 2 with $`c=1\ge1/12`$, so it is at most $`\mu`$ with probability at least $`1/13`$. The batches are disjoint and the samples independent, so those $`B`$ events are independent. The estimator is the $`\lceil\alpha B\rceil`$-th smallest, so $`\hat\mu_\alpha\le\mu`$ iff at least $`\lceil\alpha B\rceil`$ of the $`B`$ batches undershoot, and the Chernoff–Hoeffding lower tail gives

```math
\Pr[\hat\mu_\alpha>\mu]\ \le\ \Pr\Big[\mathrm{Bin}\big(B,\tfrac1{13}\big)\le\tfrac{B}{65}\Big]\ \le\ e^{-B\cdot\mathrm{kl}(1/65,\,1/13)}\ \le\ e^{-B/26}\ \le\ \delta
```

for $`B\ge26\log(1/\delta)`$, using $`\mathrm{kl}(1/65,1/13)\ge1/26`$ (the exact value is $`0.038786`$ against $`1/26=0.038462`$). Correct as stated.

## 4. Lemma 1(2): false as stated

### 4.1 Where the constant comes from

The proof in their §A.1 applies Freedman's inequality — their Lemma 14, which is Theorem 1 of Beygelzimer, Langford, Li, Reyzin & Schapire (2011) — to a single batch, through the corollary stated immediately under that lemma. Freedman's bound has the shape

```math
\sum_tX_t\ \le\ (e-2)\,\lambda\,V+\frac{\log(1/\delta)}{\lambda},\qquad V=\sigma^2T,
```

and the corollary instantiates it at $`\lambda=\min\big\lbrace R^{-1},\ \sqrt{\log(1/\delta)/((e-2)\sigma^2T)}\big\rbrace`$. That $`\lambda`$ is the minimizer of the right-hand side, and substituting it gives

```math
(e-2)\sigma^2T\sqrt{\frac{\log(1/\delta)}{(e-2)\sigma^2T}}+\log(1/\delta)\sqrt{\frac{(e-2)\sigma^2T}{\log(1/\delta)}}
\ =\ 2\sqrt{(e-2)\,\sigma^2T\,\log(1/\delta)} .
```

**The corollary states it as $`2\sqrt{(e-2)\sigma^2T}+R\log(1/\delta)`$** — the $`\log(1/\delta)`$ has been dropped from under the radical. Beygelzimer et al.'s Theorem 1 carries the logarithm inside the root as well, so the paper's own source disagrees with the corollary drawn from it.

### 4.2 The arithmetic says it is a transcription slip

Two numbers line up exactly:

- $`2\sqrt{e-2}=1.69503`$, which is their $`1.7`$;
- the per-batch failure probability their Binomial step needs is the $`\delta_0`$ with $`\mathrm{kl}(1/65,\delta_0)=1/26`$, namely $`\delta_0=4.83\times10^{-4}`$, so $`\log(1/\delta_0)=7.64`$ — and their **second** term is $`9R`$, consistent with $`R\log(1/\delta_0)=7.64R\le9R`$.

So $`\log(1/\delta)`$ was in hand and used correctly in the linear term; it simply did not make it inside the square root of the variance term. The constant the same route supports is

```math
2\sqrt{(e-2)\log(1/\delta_0)}\ =\ 1.695\times\sqrt{7.64}\ =\ 4.68 ,
```

so $`1.7`$ should be about $`4.69`$. Only the first term is affected; the $`9RB/\max\lbrace1,n\rbrace`$ term is fine and slightly conservative.

### 4.3 It is false, not merely loose

Take Bernoulli$`(1/2)`$ rewards, so $`\sigma=1/2`$, with $`m=n/B`$ samples in each batch, and let $`n\to\infty`$ with $`B`$ fixed. The $`R`$ term is $`O(1/n)`$ and the variance term is $`\Theta(1/\sqrt n)`$, so the former is negligible. Each batch statistic is asymptotically $`\mathcal N(\mu,\sigma^2/m)`$, so the probability that one batch falls below the claimed threshold $`\mu-1.7\sigma/\sqrt m`$ tends to

```math
\Phi(-1.7)\ =\ 0.044565 .
```

The estimator is the $`\lceil B/65\rceil`$-th smallest, so it violates the claimed bound as soon as at least a $`1/65=0.015385`$ fraction of the batches fall that low. Since $`0.0446>0.0154`$, that is the **typical** case rather than a rare one: the expected fraction of low batches is nearly three times the quantile level. Quantitatively, with $`\mathrm{kl}(1/65,\,0.044565)=0.013259`$, the Chernoff lower tail gives

```math
\Pr\big[\text{Lemma 1(2) holds}\big]\ \longrightarrow\ \Pr\Big[\mathrm{Bin}\big(B,\Phi(-1.7)\big)\le\lceil B/65\rceil-1\Big]\ \le\ e^{-0.013259\,B}\ =\ \delta^{0.3447}
```

at $`B=26\ln(1/\delta)`$. Against the claimed $`1-\delta`$:

| $`\delta`$ | $`B=26\ln(1/\delta)`$ | claimed | actual |
|---|---|---|---|
| $`0.05`$ | $`78`$ | $`\ge0.950`$ | $`\le0.356`$ |
| $`0.01`$ | $`120`$ | $`\ge0.990`$ | $`\le0.204`$ |
| $`0.001`$ | $`180`$ | $`\ge0.999`$ | $`\le0.092`$ |

The claimed and actual probabilities move in opposite directions as $`\delta`$ shrinks, which is the signature of a false statement rather than a loose one. With the corrected constant the problem disappears: $`\Phi(-4.69)=1.4\times10^{-6}`$, far below $`1/65`$, and the Binomial step goes through with room to spare.

### 4.4 The paper's own related work points the same way

§2 closes with: *"Cassel et al. [2025] recently considered the Minimum of Means estimator, which is equivalent to QoM with $`\alpha=1/B`$. They provide similar guarantees in their Lemmas 3 and 4; however, their optimism claim is applicable only for Bernoulli or symmetric random variables, and their bias bound incurs additional logarithmic factors."*

Those "additional logarithmic factors" are exactly the $`\sqrt{\log(1/\delta)}`$ that §4.1 shows belongs inside the radical. The natural reading is that the 2025 paper carried it correctly and the improvement claimed here is the error.

## 5. What is safe to cite

| result | status | use |
|---|---|---|
| Eq. (1), Eq. (2), the estimator | fine | adopt verbatim, including the $`+1`$ |
| Corollary 2 | correct for $`\mu>0`$ | use the **non-strict** form, valid at $`\mu=0`$ too |
| Lemma 1(1), optimism | correct | cite as stated |
| Lemma 1(2), bias, constant $`1.7`$ | **false** | replace $`1.7`$ by $`4.69`$, or use a width proved directly |
| Lemma 1(2), term $`9RB/n`$ | fine | cite as stated |

The half that matters is intact. Optimism is what replaces an uncertainty bonus and is the reason the estimator is interesting; it rests on Corollary 2 and Feige's theorem, both of which check out. The damage is confined to one constant in the width.

*Rates are unaffected.* Their Theorem 4 and the downstream regret bounds keep their shape, since $`1.7\to4.69`$ is a constant factor in one term. Their Theorem 4's own constants ($`22`$ and $`1924`$) were not re-derived here.

## 6. Verification

Statements read from the rendered PDF pages of [[cassel2026Quantile]] on 2026-09-22 — pages 3 and 4 for §2, Lemma 1 and Corollary 2 — because `pdftotext` garbles the radicals, which is where the error lives. Feige (2004) Theorem 1 and Beygelzimer et al. (2011) Theorem 1 were read from their originals. All numerical values above are evaluations of closed-form expressions: the $`\mathrm{kl}`$ constants, $`\Phi(-1.7)`$, $`2\sqrt{e-2}`$, and the table.

## 7. Open

1. **Tell the authors.** It is a two-line fix in a preprint and the paper's main claims survive it. Worth doing before building on the result publicly.
2. **Check Cassel et al. (2025) directly.** Their Lemmas 3 and 4 are the same statements for the minimum of means. If their bias bound has the $`\sqrt{\log(1/\delta)}`$ inside the root, that settles §4.4 and also bears on the novelty question for [[2026-09-22-qom-value-based-pessimism-subgaussian]], since their optimism claim covers exactly the symmetric case that page assumes.
3. **Whether $`4.69`$ is tight.** Proposition 2 of [[2026-09-22-qom-value-based-pessimism]] gives a floor of $`3.30`$ for any constant valid at this calibration, so the truth lies between $`3.30`$ and $`4.69`$.

## Sources

- [[cassel2026Quantile]], §2: Eq. (1), Eq. (2), Lemma 1, Corollary 2, and the closing paragraph on Cassel et al. (2025); §A.1 for the proof of Lemma 1; §A.2 for Lemmas 13 and 14.
- Feige (STOC 2004), Theorem 1. Beygelzimer, Langford, Li, Reyzin & Schapire (AISTATS 2011; arXiv:1002.4058), Theorem 1.
- [[2026-09-22-qom-value-based-pessimism]] §7.1, where this was first recorded, and its Proposition 2 for the floor of $`3.30`$.
