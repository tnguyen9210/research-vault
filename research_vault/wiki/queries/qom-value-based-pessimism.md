---
date: 2026-09-22
question: "Can the Quantile of Means idea of Cassel & Rosenberg (2026) be adapted into value-based pessimism for offline contextual bandits? Propose an algorithm and write a Theorem 4.2-style analysis step by step, verifying every step mathematically."
aliases: [2026-09-22-qom-value-based-pessimism]
---

# Quantile of Means as Value-Based Pessimism for Offline Contextual Bandits

**Scope.** This page mirrors §3 of the Overleaf note `02_offline_contextual_bandits.tex`, *The quantile-of-means policy* (`sec:qom`), as of Overleaf commit `06b0730` (2026-09-29).

- §3.1, the tabular setting, is mirrored in full.
- §3.2, the first linear approach, is mirrored as far as it goes: a setting and an algorithm, with no guarantee yet.
- Notes staged in the project's `temp.tex` are summarized in §6, with the status the note gives them.

The note is the source of truth: this page restates its results in the vault's notation and adds none of its own. It was rewritten on 2026-09-29, replacing an earlier, independent analysis of the same question (§9). The sub-Gaussian simplification, [[qom-value-based-pessimism-subgaussian]], still builds on that earlier analysis.

## Short answer

- **Yes, in the tabular model, and with no penalty.**
  - At each pair, the logged rewards are split uniformly at random into $`B`$ batches.
  - Each batch gives a shrunk mean $`\sum r/(\lvert\mathcal D^b\rvert+1)`$. By Feige's inequality, it lands at or below $`q^*`$ with probability at least $`1/13`$.
  - Once $`B\ge26\ln(2SK/\delta)`$, the $`\lceil B/65\rceil`$-th smallest of the $`B`$ batch values is at most $`q^*`$ with probability at least $`1-\delta/(2SK)`$.
  - The policy is greedy in that order statistic (QoM-LCB).
- **Pointwise guarantee (Theorem 1).** With probability at least $`1-\delta`$, simultaneously for every deterministic $`\pi`$,
  - $`J(\pi)-J(\hat\pi)\le\mathbb E_{x\sim\nu}[w(x,\pi(x))]`$, with $`w=4.69\sqrt{\sigma^2/(m+1)}+9/(m+1)`$;
  - $`m=\lfloor N/B\rfloor`$ is the smallest batch size.

  The width adapts to the variance, is charged once, and needs no condition on the counts.
- **Coverage form (Theorem 2).** Assume the count condition $`T\nu(x)\mu(\pi^*(x)\mid x)\ge8\ln(S/\delta')`$. With probability at least $`1-\delta-\delta'`$,
  - $`\Delta(\hat\pi)\le6.64\sqrt{S\bar\sigma^2_*B/T}+18SC^*B/T\le3.32\sqrt{S\bar C^*B/T}+18SC^*B/T`$;
  - the average coverage $`\bar C^*`$ sits in the leading term, and the worst case $`C^*`$ only in the $`1/T`$ term.
- **The constants are Cassel & Rosenberg's Lemma 1, corrected.**
  - $`4.69\ge2\sqrt{(e-2)L}`$ and $`9\ge L+1`$, with $`L=\ln(1/\delta_0)=7.642`$ and $`\delta_0=4.8\times10^{-4}`$.
  - Their printed $`1.7`$ is $`2\sqrt{e-2}`$ with the $`\sqrt L`$ dropped ([[cassel-qom-lemma1-erratum]]).
- **Staged in `temp.tex`** (§6):
  - the count condition can be dropped, at an additive $`8SC^*\ln(S/\delta')/T`$;
  - $`\bar C^*`$ can replace $`C^*`$ in the second term only at the $`1/\sqrt T`$ rate, which gives $`\Delta(\hat\pi)\le7.57\sqrt{S\bar C^*B/T}`$;
  - a bound in $`\bar C^*`$ alone, with no count condition, is derived but parked.
- **Linear, Approach 1: an algorithm, no guarantee yet** (§7).
  - Each batch is fit by least squares, with a phantom round $`(\phi(x,a),0)`$ in place of the $`+1`$.
  - When $`\phi(x,a)=e_{(x,a)}`$, this is exactly QoM-LCB.
  - For general features, weak pessimism is not automatic: the fit can put negative weight on some rewards.

## 1. Setting and notation

Notation follows [[contextual-bandits-offline]] §2, which the note shares. The learner observes only a fixed dataset $`\mathcal D:=\lbrace(x_t,a_t,r_t)\rbrace_{t=1}^T`$, collected by running a behavior policy $`\mu`$ for $`T`$ rounds:

```math
x_t\sim\nu,\qquad a_t\sim\mu(\cdot\mid x_t),\qquad r_t\sim\rho(\cdot\mid x_t,a_t).
```

- **The model.** The context law $`\nu`$ and the reward kernel $`\rho`$ are unknown. The model is tabular, with $`S:=\lvert\mathcal X\rvert`$ and $`K:=\lvert\mathcal A\rvert`$ finite. Rewards lie in $`[0,1]`$.
- **Means and variances.** At a pair, the mean reward is $`q^*(x,a)`$ and the variance $`\sigma^2(x,a)`$. The variance is not an input to the algorithm; it appears only in the analysis.
- **Values.** $`J(\pi)`$ is a policy's value and $`\pi^*`$ a deterministic optimal policy. The suboptimality is $`\Delta(\hat\pi):=J(\pi^*)-J(\hat\pi)`$.
- **Counts.** $`N(x,a)`$ is the number of rounds at which the pair $`(x,a)`$ is logged.

**Coverage** (the note's §2.1). A ratio with a positive numerator and a vanishing denominator is $`+\infty`$. Every maximum and sum over contexts runs over the $`x`$ with $`\nu(x)>0`$.

```math
C^*:=\max_{x}\frac1{\mu(\pi^*(x)\mid x)},\qquad
\bar C^*:=\mathbb E_{x\sim\nu}\Big[\frac1{\mu(\pi^*(x)\mid x)}\Big]\le C^*,\qquad
\bar\sigma^2_*:=\sum_x\frac{\nu(x)\,\sigma^2(x,\pi^*(x))}{\mu(\pi^*(x)\mid x)}\le\frac{\bar C^*}4 .
```

- **Why $`\bar\sigma^2_*\le\bar C^*/4`$.** For a reward $`r\in[0,1]`$ with mean $`q`$, the variance is $`\mathbb E[r^2]-q^2\le\mathbb E[r]-q^2=q(1-q)\le1/4`$.
- **The uniform coefficient** $`C_{\mathrm{unif}}:=\max_{x}\max_a1/\mu(a\mid x)\ge C^*`$ enters only the greedy rule.

**Average versus worst-case coverage** (the note's §2.1):

- **One ratio, two aggregations.** $`C^*`$ and $`\bar C^*`$ aggregate the same per-context ratio $`1/\mu(\pi^*(x)\mid x)`$. $`C^*`$ takes its maximum; $`\bar C^*`$ takes its average under $`\nu`$.
- **The gap is between a rare failure and a common one.** Take one context with $`\nu(x)=\varepsilon`$ and $`\mu(\pi^*(x)\mid x)=\varepsilon`$, and cover every other context perfectly. Then $`C^*=1/\varepsilon`$, while that context adds only $`1`$ to $`\bar C^*`$.
- **Both are norms of one density ratio.** Let $`d^\pi(x,a):=\nu(x)\pi(a\mid x)`$ be a policy's occupancy. Then $`C^*`$ is the supremum of $`d^{\pi^*}/d^\mu`$, and $`\bar C^*=\lVert d^{\pi^*}/d^\mu\rVert^2_{2,d^\mu}`$ is its squared $`L^2(d^\mu)`$ norm. Hence $`\bar C^*\le C^*`$, because $`\mathbb E_{d^\mu}[d^{\pi^*}/d^\mu]=1`$.
- **Neither aggregation is new.**
  - The supremum form is Rashidinejad et al. (2021), Definition 1.
  - The squared $`L^2`$ form is the usual all-policy coefficient; an example is $`\sup_\pi\sup_h\lVert d^\pi_h/d^\mu_h\rVert^2_{2,d^\mu_h}`$ in Assumption 2.2 of [[yin2023Offline]].
  - The bounds below use the averaged form at the single policy $`\pi^*`$. See also [[coverage-coefficient]].

## 2. Algorithm: QoM-LCB

**Definitions.**

- **The $`\alpha`$-quantile.** For a sequence $`\hat\mu^1,\dots,\hat\mu^B\in\mathbb R`$, it is $`q_\alpha(\hat\mu^b,\ b\in[B]):=\hat\mu^{(\lceil\alpha B\rceil)}`$, the $`\lceil\alpha B\rceil`$-th smallest value (Cassel & Rosenberg's Eq. (1)).
- **The pessimistic estimator** below is their Eq. (2).

```math
\begin{aligned}
&\textbf{Input: } \mathcal D,\ \text{number of batches } B,\ \text{level } \alpha \\
&\textbf{for } (x,a)\in\mathcal X\times\mathcal A: \\
&\qquad \mathcal D(x,a)\leftarrow\lbrace r_t: t\in[T],\ (x_t,a_t)=(x,a)\rbrace \qquad \triangleright\ \text{repeated values kept; } N(x,a)=\lvert\mathcal D(x,a)\rvert \\
&\qquad \text{split } \mathcal D(x,a) \text{ uniformly at random into } \mathcal D^1(x,a),\dots,\mathcal D^B(x,a),\ \text{sizes as equal as possible} \\
&\qquad \hat\mu^b(x,a)\leftarrow\sum_{r\in\mathcal D^b(x,a)}\frac{r}{\lvert\mathcal D^b(x,a)\rvert+1},\quad b\in[B] \qquad \triangleright\ \text{weakly pessimistic; an empty batch gives } 0 \\
&\qquad \underline q(x,a)\leftarrow q_\alpha\big(\hat\mu^b(x,a),\ b\in[B]\big) \qquad \triangleright\ \text{pessimistic estimator} \\
&\textbf{return } \hat\pi(x)\in\arg\max_{a\in\mathcal A}\underline q(x,a)
\end{aligned}
```

**Calibration.** Throughout, $`\alpha=1/65`$ and $`B\ge26\ln(2SK/\delta)`$, the values Lemma 4 fixes.

**The note's two terms.**

- $`\hat\mu^b`$ is a **weakly pessimistic** estimator: by Corollary 3, $`\Pr(\hat\mu^b(x,a)\le q^*(x,a))\ge1/13`$.
- $`\underline q`$ is the **pessimistic** estimator.

Notes:

- $`\underline q\ge0`$ always, since the rewards are non-negative.
- **The split must be independent of the reward values.** This is what makes the $`B`$ weakly pessimistic estimators independent.
- **The split must be uniform.** This is what gives $`\lvert\mathcal D^b(x,a)\rvert\ge m(x,a):=\lfloor N(x,a)/B\rfloor`$, which Lemma 4(2) requires.
- **Relation to the plug-in rule.** $`\underline q`$ plays the role of $`\hat q-\Gamma`$ in the pessimistic plug-in rule of [[contextual-bandits-offline-value-based]]. Both are pessimistic estimators of $`q^*`$ that the rule maximizes, and only their construction differs.

## 3. Assumptions

Besides the i.i.d. data of §1, the analysis uses one assumption.

**Assumption 1 (non-negative bounded rewards).** $`0\le r\le1`$ almost surely at every pair.

- **What each half is for.** Non-negativity is the only distributional assumption behind the validity of $`\underline q`$, and it is what Corollary 3 needs. The upper bound $`1`$ is used only for the width.
- **Not assumed:** the variances, the propensities, the reward family, or the value of $`C^*`$.
- **No count condition.** At a pair with $`N(x,a)<B`$, some batches are empty and their statistics are $`0`$.
  - $`\underline q(x,a)`$ is still defined, and Lemma 4 still applies.
  - Only the width of Theorem 1 becomes vacuous there.

## 4. Main results

**Theorem 1 (QoM-LCB, pointwise form; `thm:ocb-qom`).** Let $`S,K<\infty`$ and let Assumption 1 hold. Run QoM-LCB with $`\alpha=1/65`$ and $`B\ge26\ln(2SK/\delta)`$, where $`\delta\in(0,1)`$. Define

```math
w(x,a):=4.69\sqrt{\frac{\sigma^2(x,a)}{m(x,a)+1}}+\frac9{m(x,a)+1},\qquad m(x,a):=\Big\lfloor\frac{N(x,a)}B\Big\rfloor .
```

Here $`m(x,a)`$ is the number of rewards in the smallest batch at $`(x,a)`$, which is $`0`$ when $`N(x,a)<B`$. Then with probability at least $`1-\delta`$, simultaneously for every deterministic policy $`\pi:\mathcal X\to\mathcal A`$,

```math
J(\pi)-J(\hat\pi)\ \le\ \mathbb E_{x\sim\nu}\big[w(x,\pi(x))\big].
```

*Proof.*

**The two events.** Define $`\mathcal E_p:=\lbrace\underline q(x,a)\le q^*(x,a)\ \forall(x,a)\rbrace`$ and $`\mathcal E_w:=\lbrace\underline q(x,a)\ge q^*(x,a)-w(x,a)\ \forall(x,a)\rbrace`$.

**They hold with probability $`1-\delta`$.** At each pair, the rewards are $`N(x,a)`$ i.i.d. draws from $`\rho(\cdot\mid x,a)`$. The batches are disjoint, chosen independently of the rewards, and each holds at least $`m(x,a)`$ of them. So Lemma 4 applies at each pair at level $`\delta/(2SK)`$, and each of its two parts fails with probability at most $`\delta/(2SK)`$. A union bound over the $`2SK`$ events gives $`\Pr(\mathcal E_p^c\cup\mathcal E_w^c)\le\delta`$.

**On the events.** On $`\mathcal E_p\cap\mathcal E_w`$, for every $`\pi`$ and every $`x`$,

```math
q^*(x,\hat\pi(x))\ \ge\ \underline q(x,\hat\pi(x))\ \ge\ \underline q(x,\pi(x))\ \ge\ q^*(x,\pi(x))-w(x,\pi(x)).
```

The three steps use $`\mathcal E_p`$, then the definition of $`\hat\pi`$, then $`\mathcal E_w`$.

**Conclusion.** Both $`\pi`$ and $`\hat\pi`$ are deterministic, so averaging over $`x\sim\nu`$ gives the bound. It holds for every $`\pi`$ at once, since the event does not depend on $`\pi`$. $`\square`$

Notes (the note's *Batch size, not count*):

- **Why the width is in $`m+1`$.** It is stated in the smallest batch size because the proof looks at one batch at a time. It carries $`m+1`$ rather than $`m`$ because Lemma 4(2) divides the batch sum by $`m_b+1`$, the statistic's own denominator.
- **Reading it in the count.** Under a uniform split, $`m+1>n/B`$, so $`1/(m+1)\le B/\max\lbrace1,n\rbrace`$. The width therefore reads in the count at no cost.
- **Small counts.** When $`n<B`$ the width is at least $`9`$: vacuous but valid.
- **Comparison with the tabular LCB.**
  - The shape is that of Theorem 4.2 of [[contextual-bandits-offline-value-based]] (the note's Tabular LCB theorem), $`\Delta(\hat\pi)\le2\,\mathbb E_\nu[\min\lbrace1,\sqrt{\ln(2SK/\delta)/(2N(x,\pi^*(x)))}\rbrace]`$.
  - That theorem charges the Hoeffding width twice. Here $`w`$ replaces it and is charged once.

**Theorem 2 (QoM-LCB, coverage form; `thm:ocb-qom-coverage`).** Take the setting and assumptions of Theorem 1, and fix a second confidence level $`\delta'\in(0,1)`$. Suppose in addition that $`C^*<\infty`$ and that the counts satisfy the condition of the note's Tabular LCB theorem:

```math
T\,\nu(x)\,\mu(\pi^*(x)\mid x)\ \ge\ 8\ln(S/\delta')\qquad\text{for every } x \text{ with } \nu(x)>0 .
```

Then with probability at least $`1-\delta-\delta'`$,

```math
\Delta(\hat\pi)\ \le\ 6.64\sqrt{\frac{S\,\bar\sigma^2_*B}T}+\frac{18\,S\,C^*B}T\ \le\ 3.32\sqrt{\frac{S\,\bar C^*B}T}+\frac{18\,S\,C^*B}T .
```

*Proof.* The note's sketch has two steps.

**Step 1: control the counts.** Let $`q_x:=\nu(x)\mu(\pi^*(x)\mid x)`$.

- A round contributes to $`N(x,\pi^*(x))`$ exactly when $`x_t=x`$ and $`a_t=\pi^*(x)`$. This happens independently across rounds, so $`N(x,\pi^*(x))\sim\mathrm{Bin}(T,q_x)`$.
- Define the event $`\mathcal E_N:=\lbrace N(x,\pi^*(x))\ge Tq_x/2\ \forall x\rbrace`$.

A union bound, the multiplicative Chernoff bound and the count condition give

```math
\Pr(\mathcal E_N^c)\ \le\ \sum_x\Pr\big(\mathrm{Bin}(T,q_x)\le Tq_x/2\big)\ \le\ \sum_xe^{-Tq_x/8}\ \le\ \delta' .
```

On $`\mathcal E_N`$, we get $`1/(m(x,\pi^*(x))+1)\le B/N(x,\pi^*(x))\le2B/(Tq_x)`$. The first inequality holds because $`\lfloor n/B\rfloor+1>n/B`$.

**Step 2: apply Theorem 1 at $`\pi=\pi^*`$.** Its event and $`\mathcal E_N`$ hold together with probability at least $`1-\delta-\delta'`$. On both, using $`4.69\sqrt2\le6.64`$,

```math
\Delta(\hat\pi)\ \le\ \sum_x\nu(x)\Big[6.64\sqrt{\frac{\sigma^2(x,\pi^*(x))B}{Tq_x}}+\frac{18B}{Tq_x}\Big]
=6.64\sqrt{\frac BT}\sum_x\sqrt{\frac{\nu(x)\sigma^2(x,\pi^*(x))}{\mu(\pi^*(x)\mid x)}}+\frac{18B}T\sum_x\frac1{\mu(\pi^*(x)\mid x)} .
```

**Bounding the two sums.**

- Cauchy–Schwarz over at most $`S`$ contexts bounds the first sum by $`\sqrt{S\bar\sigma^2_*}`$.
- The second sum has at most $`S`$ terms, and each is at most $`C^*`$.

This gives the first inequality. The second follows from $`\bar\sigma^2_*\le\bar C^*/4`$ and $`6.64/2=3.32`$. $`\square`$

Notes:

- **Two confidence levels.** $`\delta'`$ pays for the randomness of the design, separately from the $`\delta`$ that pays for the reward noise. Taking $`\delta'=\delta`$ gives probability at least $`1-2\delta`$.
- **What the bar marks.** $`\bar\sigma^2_*`$ is a coverage-weighted variance, not a variance. It is at most $`\bar C^*/4`$, not $`1/4`$.
- **Why $`\bar C^*`$ leads and $`C^*`$ trails.**
  - In the leading term, Cauchy–Schwarz keeps the weight $`\nu(x)`$, so a context is charged in proportion to how often it occurs.
  - In the additive term, the width is of order $`1/N(x,\pi^*(x))`$. The $`\nu(x)`$ cancels against it, and the sum is bounded only by $`SC^*`$.
- **The rate in $`T`$.**
  - It is governed by the average coverage. The worst case only decides how large $`T`$ must be before the leading term dominates.
  - This holds for a fixed instance as $`T`$ grows, but not uniformly over instances (§6.2).
- **The only constant lost** is the Chernoff factor $`2`$ on $`\mathcal E_N`$: $`\sqrt2`$ in the first term of $`w`$ and $`2`$ in the second. The floor in $`m`$ costs nothing, because the width is stated in $`m+1`$.

## 5. The two supporting results

Both results concern i.i.d. non-negative $`X_1,\dots,X_n`$. Following the note, in this section $`\mu`$ denotes their mean, not the behavior policy. They are Cassel & Rosenberg's Corollary 2 and Lemma 1 ([[cassel2026Quantile]]). The note proves both in full, because the constants of §4 come out of them.

**Corollary 3 (a shrunk sample mean lands below its mean; `cor:ocb-qom-feige`).** For any $`c\ge1/12`$,

```math
\Pr\Big(\sum_{i=1}^n\frac{X_i}{n+c}\le\mu\Big)\ \ge\ \frac1{13}.
```

*Proof.* The tool is Feige's inequality (Feige 2004, Theorem 1). Let $`Y_1,\dots,Y_n`$ be independent and non-negative, with $`\mathbb EY_i=\mu_i\le1`$ and $`\bar\mu:=\sum_i\mu_i`$. Then for every $`\delta>0`$,

```math
\Pr\Big(\sum_iY_i<\bar\mu+\delta\Big)\ \ge\ \min\Big\lbrace\frac\delta{1+\delta},\ \frac1{13}\Big\rbrace .
```

- **If $`\mu=0`$,** every $`X_i`$ is $`0`$ almost surely, so the event holds surely.
- **Otherwise,** let $`Y_i:=X_i/\mu`$, so that $`\bar\mu=n`$, and take $`\delta:=c`$. The map $`t\mapsto t/(1+t)`$ is increasing and equals $`1/13`$ at $`t=1/12`$, so the minimum is $`1/13`$. $`\square`$

Notes:

- **Feige's hypothesis $`\mu_i\le1`$ is not decorative.**
  - Without it the bound fails: a single $`Y`$ equal to $`100`$ with probability $`0.99`$, and $`0`$ otherwise, violates it at $`\delta=1`$.
  - Normalizing the means to $`1`$ is what makes the hypothesis hold. It also turns the deviation $`c\mu`$, which is not bounded away from $`0`$, into $`c`$, which is.
- **It is not a concentration inequality.** It uses nothing beyond the first moment. The estimator of §2 uses $`c=1`$.
- **Only the non-strict form is available**, because of the case $`\mu=0`$.
  - Cassel & Rosenberg state the strict form, which fails there ([[cassel-qom-lemma1-erratum]] §2).
  - Offline, an action that never pays is an ordinary case, not one to exclude.

**Lemma 4 (quantile of means: pessimism and bias; `lem:ocb-qom-cassel`).** Let $`X_1,\dots,X_n`$ be i.i.d. in $`[0,1]`$, with mean $`\mu`$ and variance $`\sigma^2`$. Let $`\hat\mu_\alpha`$ be the quantile-of-means estimator formed from $`B\ge26\ln\delta^{-1}`$ batches at level $`\alpha=1/65`$, with the batches disjoint and chosen independently of the values. Each of the following holds with probability at least $`1-\delta`$:

1. *(Pessimism)* $`\hat\mu_\alpha\le\mu`$.
2. *(Bias)* If every batch holds at least $`m\ge0`$ of the observations, then $`\hat\mu_\alpha\ge\mu-4.69\sqrt{\sigma^2/(m+1)}-9/(m+1)`$.

The two parts share one shape. A single batch does the right thing with constant probability, and the order statistic converts that constant into $`1-\delta`$ through a binomial tail.

*Proof of (1).* Let $`S^b:=\sum_{i\in\mathcal D^b}X_i/(\lvert\mathcal D^b\rvert+1)`$ be the statistic of batch $`b`$, and let $`Z:=\lvert\lbrace b\in[B]:S^b\le\mu\rbrace\rvert`$.

- **One batch.** Corollary 3 with $`c=1`$ gives $`\Pr(S^b\le\mu)\ge1/13`$. An empty batch has $`S^b=0\le\mu`$ outright.
- **Independence.** The batches are disjoint and chosen without reference to the values, so the $`S^b`$ are independent. Hence $`Z`$ stochastically dominates $`\mathrm{Bin}(B,1/13)`$.
- **The order statistic.** The estimator is the $`\lceil B/65\rceil`$-th smallest $`S^b`$, so $`\lbrace\hat\mu_\alpha>\mu\rbrace\subseteq\lbrace Z<\lceil B/65\rceil\rbrace`$.

Since $`\lceil B/65\rceil-1<B/65`$,

```math
\Pr(\hat\mu_\alpha>\mu)\ \le\ \Pr\Big(\mathrm{Bin}\big(B,\tfrac1{13}\big)\le\tfrac B{65}\Big)\ \le\ e^{-B\,\mathrm{kl}(1/65,\,1/13)}\ \le\ e^{-B/26}\ \le\ \delta .
```

Here $`\mathrm{kl}(a,p):=a\ln\frac ap+(1-a)\ln\frac{1-a}{1-p}`$. The chain uses three facts:

- the Chernoff–Hoeffding lower tail, which applies since $`1/65<1/13`$;
- $`\mathrm{kl}(1/65,1/13)=0.03879\ge1/26`$;
- $`B\ge26\ln\delta^{-1}`$. $`\square`$

*Proof of (2).* Fix a per-batch failure level $`\delta_0:=4.8\times10^{-4}`$, and let $`L:=\ln\delta_0^{-1}=7.642`$. Take a batch with $`m_b:=\lvert\mathcal D^b\rvert\ge m`$.

**Freedman's inequality on one batch.** Apply it, in the form of Beygelzimer et al. (2011), Theorem 1, to the differences $`d_i:=\mu-X_i`$. These are i.i.d. with mean zero, satisfy $`d_i\le\mu\le1`$, and have $`\sum_i\mathbb Ed_i^2=m_b\sigma^2`$. For each fixed $`\lambda\in(0,1]`$, with probability at least $`1-\delta_0`$,

```math
m_b\mu-\sum_{i\in\mathcal D^b}X_i\ \le\ (e-2)\lambda m_b\sigma^2+\frac L\lambda .
```

**Choosing $`\lambda`$.** Take $`\lambda:=\min\lbrace1,\sqrt{L/((e-2)m_b\sigma^2)}\rbrace`$.

- If the second entry is the minimum, the right side is exactly $`2\sqrt{(e-2)Lm_b\sigma^2}`$.
- Otherwise $`(e-2)m_b\sigma^2<L`$, and the right side is at most $`\sqrt{(e-2)Lm_b\sigma^2}+L`$.

In both cases $`\sum_iX_i\ge m_b\mu-2\sqrt{(e-2)L}\sqrt{m_b\sigma^2}-L`$. An empty batch satisfies this surely.

**Dividing by $`m_b+1`$.** This gives

```math
S^b\ \ge\ \mu-\frac{\mu+2\sqrt{(e-2)L}\sqrt{m_b\sigma^2}+L}{m_b+1}
\ \ge\ \mu-2\sqrt{(e-2)L}\sqrt{\frac{\sigma^2}{m_b+1}}-\frac{L+1}{m_b+1}
\ \ge\ \mu-4.69\sqrt{\frac{\sigma^2}{m+1}}-\frac9{m+1} .
```

- The second step uses $`\mu\le1`$ and $`\sqrt{m_b}\le\sqrt{m_b+1}`$.
- The third uses $`2\sqrt{(e-2)L}=4.686`$, $`L+1=8.642`$ and $`m_b\ge m`$.

**From batches to the estimator.** Call a batch *bad* if this display fails for it. The number of bad batches is dominated by $`\mathrm{Bin}(B,\delta_0)`$, and the estimator falls below the threshold only if at least $`\lceil B/65\rceil`$ batches are bad. Hence

```math
\Pr\Big(\hat\mu_\alpha<\mu-4.69\sqrt{\tfrac{\sigma^2}{m+1}}-\tfrac9{m+1}\Big)\ \le\ \Pr\big(\mathrm{Bin}(B,\delta_0)\ge\lceil B/65\rceil\big)\ \le\ e^{-B\,\mathrm{kl}(1/65,\,\delta_0)}\ \le\ e^{-B/26}\ \le\ \delta .
```

The steps are the Chernoff–Hoeffding upper tail, which applies since $`\delta_0<1/65`$, and then $`\mathrm{kl}(1/65,\delta_0)=0.03855\ge1/26`$. $`\square`$

Notes:

- **What each part needs.** Part (1) uses no assumption beyond non-negativity, and none on the batch sizes. Only part (2) gives a single batch a width.
- **Variance adaptivity** comes from the Freedman step: its deviation scales with $`\sigma`$ rather than with the range.
- **The shrinkage costs one unit.** Dividing by $`m_b+1`$, the statistic's own denominator, keeps that cost to a constant: the $`1`$ in $`L+1`$.
- **What pins $`\delta_0`$.** The step $`\mathrm{kl}(1/65,\delta_0)\ge1/26`$ holds only up to $`\delta_0\approx4.829\times10^{-4}`$, and $`4.8\times10^{-4}`$ is that level rounded down. A smaller $`\delta_0`$ would be legitimate, but it enlarges $`L`$ and both constants.
- **Against the source.**
  - Cassel & Rosenberg state part (2) in the count: $`\hat\mu_\alpha\ge\mu-1.7\sqrt{\sigma^2B/\max\lbrace1,n\rbrace}-9RB/\max\lbrace1,n\rbrace`$. Their $`1.7=2\sqrt{e-2}`$ omits the $`\sqrt L`$.
  - With $`R=1`$ and a uniform split, $`1/(m+1)\le B/\max\lbrace1,n\rbrace`$. So part (2) implies their statement with $`4.69`$ in place of $`1.7`$ ([[cassel-qom-lemma1-erratum]] §4).

**Staged detail on part (1)** (`temp.tex`, `sec:tmp-domination`):

- **Domination.** Independent coins with success probabilities $`p_b\ge p`$ sum to a variable that stochastically dominates $`\mathrm{Bin}(B,p)`$. There are two proofs:
  - by monotonicity: $`\Pr(Z\ge j)=\Pr(Z_{-b}\ge j)+p_b\Pr(Z_{-b}=j-1)`$ is nondecreasing in each $`p_b`$;
  - by coupling: $`\mathbf 1\lbrace U_b\le p\rbrace\le\mathbf 1\lbrace U_b\le p_b\rbrace`$ for i.i.d. uniform $`U_b`$.
- **Independence cannot be dropped.** If all batches succeed or fail together, then $`\Pr(Z=0)=1-p_1`$ for every $`B`$.
- **The order statistic and the count.** $`S^{(k)}\le\mu`$ if and only if $`Z\ge k`$, so the inclusion in the proof of (1) is in fact an equality.
- **The Chernoff–Hoeffding lower tail.** For $`0\le a<p`$, $`\Pr(\mathrm{Bin}(B,p)\le aB)\le e^{-B\,\mathrm{kl}(a,p)}`$.
  - It follows from Markov's inequality on $`e^{-\lambda W}`$, at the optimal $`e^{-\lambda^*}=a(1-p)/(p(1-a))`$.
  - It is Hoeffding (1963), Theorem 1, applied to $`1-J_b`$.
- **Why the $`\mathrm{kl}`$ form and not Hoeffding's additive one.**
  - At $`a=p/5`$, Pinsker gives only $`2(p-a)^2=\tfrac{32}{25}p^2`$, which is of order $`p^2`$.
  - By contrast, $`\mathrm{kl}(a,p)\ge p-a-a\ln(p/a)=\tfrac{4-\ln5}5p`$ is of order $`p`$.
  - So the kl form needs $`B`$ of order $`p^{-1}\ln\delta^{-1}`$, against $`p^{-2}\ln\delta^{-1}`$. The constant $`1/26`$ comes from evaluating $`\mathrm{kl}(1/65,1/13)`$ itself, since $`(4-\ln5)/65=0.0368<1/26`$.
- **A shortcut.** Bounding $`\mathbb Ee^{-\lambda I_b}\le1-p+pe^{-\lambda}`$ directly gives the same tail with no coupling; independence is still needed for the product. Domination is the more general statement, since it covers every nonincreasing test function.

## 6. Staged in `temp.tex`: the coverage form revisited

These notes sit in the project's `temp.tex`, which `main.tex` includes after §3. They are not yet in the main text.

### 6.1 The count condition (`sec:tmp-count`)

**Meaning.** $`q_x=\nu(x)\mu(\pi^*(x)\mid x)=d^\mu(x,\pi^*(x))`$ is the probability that one round logs $`x`$ together with $`\pi^*(x)`$. So $`N(x,\pi^*(x))\sim\mathrm{Bin}(T,q_x)`$, with mean $`Tq_x`$.

- **What it says.** At every context that can occur, the optimal action is expected to be logged at least $`8\ln(S/\delta')`$ times.
- **Its role is concentration.** It lets the proofs replace the random $`N(x,\pi^*(x))`$ by $`Tq_x/2`$.
- **Where the 8 comes from.**
  - For $`0<a\le p<1`$, $`\mathrm{kl}(a,p)\ge(p-a)^2/(2p)`$, by Taylor's theorem at $`p`$, since $`f''(t)=1/(t(1-t))\ge1/p`$ for $`t\le p`$.
  - So $`\mathrm{kl}(p/2,p)\ge p/8`$, and the lower tail gives $`e^{-Tq_x/8}`$.
  - Asking for a fraction $`1-\eta`$ of the mean instead needs $`Tq_x\ge(2/\eta^2)\ln(S/\delta')`$, and costs $`1/(1-\eta)`$ in place of the factor $`2`$.
  - The $`S`$ is the union bound over contexts.
- **Single-policy coverage.** Only $`\pi^*`$'s actions need to be logged often. The greedy rule instead needs $`T\nu(x)\mu(a\mid x)\ge8\ln(SK/\delta')`$ at every pair.
- **Some condition is necessary.** If $`q_x\le1/2`$, then $`\Pr(N(x,\pi^*(x))=0)=(1-q_x)^T\ge e^{-2Tq_x}`$, which is not small when $`Tq_x`$ is of constant order.
- **As a sample size.** The condition holds whenever $`T\ge8C^*\ln(S/\delta')/\nu_{\min}`$.
  - The $`1/\nu_{\min}`$ is the weak spot.
  - A context of tiny probability forces a large $`T`$, although it contributes at most $`\nu(x)`$ to the regret.

**Proposition 6 (no count condition; `prop:tmp-no-count`).** Let $`C^*<\infty`$ and $`\delta'\in(0,1)`$. With probability at least $`1-\delta-\delta'`$:

1. for the tabular LCB rule, $`\Delta(\hat\pi)\le2\sqrt{S\bar C^*\ln(2SK/\delta)/T}+16SC^*\ln(S/\delta')/T`$;
2. for QoM-LCB, $`\Delta(\hat\pi)\le6.64\sqrt{S\bar\sigma^2_*B/T}+(18B+8\ln(S/\delta'))SC^*/T`$.

*Proof idea.*

- **Heavy and light contexts.** Split the contexts into heavy ones, $`H:=\lbrace x:Tq_x\ge8\ln(S/\delta')\rbrace`$, and light ones, $`L`$. The count event holds on $`H`$ with no condition.
- **The light mass is small.** Each light context has $`\nu(x)\le C^*q_x<8C^*\ln(S/\delta')/T`$, so $`\nu(L)\le8SC^*\ln(S/\delta')/T`$.
- **The light contexts' cost.** Under QoM-LCB, the regret at a light context is at most $`1`$. The LCB bound charges it at most $`2`$, because that bound's per-context term is $`2\min\lbrace1,\cdot\rbrace`$. $`\square`$

Note: if $`\delta'\ge\delta`$, then $`\ln(S/\delta')\le B/26`$, so $`8\ln(S/\delta')\le4B/13`$. The constant $`18`$ becomes at most $`18+4/13`$, and the rate is unchanged.

### 6.2 The second term (`sec:tmp-second-term`)

**Can $`\bar C^*`$ replace $`C^*`$ in the $`1/T`$ term?** Not through the per-context bound. With $`C_x:=18B/(Tq_x)`$, the second term of Theorem 2 comes from

```math
\sum_x\nu(x)\,C_x\ =\ \frac{18B}T\sum_x\frac1{\mu(\pi^*(x)\mid x)} ,
```

in which $`\nu(x)`$ cancels. A rarely seen, badly covered context therefore costs its full $`1/\mu(\pi^*(x)\mid x)`$, while $`S\bar C^*`$ would discount it by $`\nu(x)`$.

**An instance, depending on $`T`$, shows the gap is real.** Take $`S=2`$, $`\delta'\ge\delta`$ and $`T\ge1296B`$, with noiseless rewards at $`\pi^*`$'s actions, so that $`\bar\sigma^2_*=0`$. Let

```math
\nu(x_2):=18\sqrt{B/T},\qquad\mu(\pi^*(x_2)\mid x_2):=\sqrt{B/T},\qquad\nu(x_1):=1-\nu(x_2),\qquad\mu(\pi^*(x_1)\mid x_1):=1 .
```

- **The count condition holds.** $`Tq_{x_2}=18B`$ and $`Tq_{x_1}\ge T/2\ge18B`$, while $`18B\ge8\ln(S/\delta')`$.
- **The coefficients separate.** $`\bar C^*=\nu(x_1)+18\le19`$ stays bounded, while $`C^*=\sqrt{T/B}`$ grows.
- **The per-context bound charges $`x_2`$ its full mass.** $`C_{x_2}=1`$, so the charge at $`x_2`$ is $`\nu(x_2)=18\sqrt{B/T}`$. That is not $`O(B/T)`$, whereas $`S\bar C^*B/T\le38B/T`$.
- **The $`C^*`$ term is not of lower order.** Theorem 2's second term equals $`36\sqrt{B/T}`$ here.
- **Open.** Whether the algorithm's own regret at $`x_2`$ is that large is a lower-bound question, and the note leaves it open.

**Proposition 7 (coverage bound in $`\bar C^*`$; `prop:tmp-cbar-rate`).** In the setting of Theorem 2 and with its assumptions, with probability at least $`1-\delta-\delta'`$,

```math
\Delta(\hat\pi)\ \le\ 6.64\sqrt{\frac{S\,\bar\sigma^2_*B}T}+\min\Big\lbrace\frac{18\,S\,C^*B}T,\ 4.25\sqrt{\frac{S\,\bar C^*B}T}\Big\rbrace\ \le\ 7.57\sqrt{\frac{S\,\bar C^*B}T} .
```

*Proof idea.* Let $`A_x:=6.64\sqrt{\sigma^2(x,\pi^*(x))B/(Tq_x)}`$. The regret at $`x`$ is at most $`\min\lbrace1,A_x+C_x\rbrace`$, which is at most $`A_x+\min\lbrace1,C_x\rbrace`$.

- $`\min\lbrace1,C_x\rbrace\le C_x`$ gives the $`C^*`$ entry.
- $`\min\lbrace1,C_x\rbrace\le\sqrt{C_x}`$, with Cauchy–Schwarz, gives $`\sqrt{18S\bar C^*B/T}`$. Here $`\sqrt{18}=4.243`$.
- The last inequality uses $`\bar\sigma^2_*\le\bar C^*/4`$ and $`6.64/2+4.25=7.57`$. $`\square`$

Notes:

- **Theorem 2 alone needs a burn-in.** It gives the rate $`\sqrt{S\bar C^*B/T}`$ only once $`T\ge(18/3.32)^2SB(C^*)^2/\bar C^*`$, where $`(18/3.32)^2\le29.4`$. The instance above never reaches this.
- **The min keeps both regimes.** With noiseless optimal actions, the first entry still gives $`18SC^*B/T`$.
- **A gap in how the proof cites Theorem 1.** The proof uses the per-context bound $`\min\lbrace1,w(x,\pi^*(x))\rbrace`$. That bound comes from a display inside the proof of Theorem 1, not from its statement, and Proposition 6 has the same gap.
- **Plan for the main text.** The note moves this result into the main text in three steps:
  1. state Theorem 1 with $`\mathbb E_\nu[\min\lbrace1,w(x,\pi(x))\rbrace]`$;
  2. replace Theorem 2's second term by the min;
  3. qualify the Notes item that confines the worst case to the lower-order term.

### 6.3 Parked: a bound in $`\bar C^*`$ alone (`sec:tmp-cbar-only`)

The note marks this *parked on 2026-09-28*: the bound is derived but not yet checked against the literature.

**Proposition 8 (no count condition and no $`C^*`$; `prop:tmp-cbar-no-count`, parked).** In the setting of Theorem 1 and with its assumptions, let $`\delta'\in(0,1)`$. With probability at least $`1-\delta-\delta'`$,

```math
\Delta(\hat\pi)\ \le\ 6.64\sqrt{\frac{S\,\bar\sigma^2_*B}T}+\sqrt{\frac{18\,S\,\bar C^*B}T}+\sqrt{\frac{8\,S\,\bar C^*\ln(S/\delta')}T} .
```

If $`\delta'\ge\delta`$, the right side is at most $`8.13\sqrt{S\bar C^*B/T}`$.

*Proof idea.* The heavy contexts are handled as in Proposition 7. A light context satisfies

```math
\nu(x)=\sqrt{q_x}\,\sqrt{\frac{\nu(x)}{\mu(\pi^*(x)\mid x)}}<\sqrt{\frac{8\ln(S/\delta')}T}\,\sqrt{\frac{\nu(x)}{\mu(\pi^*(x)\mid x)}} ,
```

and Cauchy–Schwarz over at most $`S`$ contexts bounds $`\nu(L)`$ by the last term. For the constant, $`\sqrt{8/26}\le0.56`$ and $`3.32+4.25+0.56=8.13`$. $`\square`$

- **The price is variance adaptivity.** With noiseless optimal actions, this bound is still of order $`1/\sqrt T`$, where Theorem 2 gives $`1/T`$.
- **A combined form.** The note also records a min form that contains both Proposition 6(2) and Proposition 8.
- **Open: why prior work states $`C^*`$ rather than $`\bar C^*`$.** The note's hypothesis, flagged as unchecked, is the minimax framing.
  - Rashidinejad et al. (2021) match their rate with a lower bound over the instances with $`C^*\le c`$.
  - If those hard instances have $`\bar C^*`$ of order $`c`$, as uniform $`\nu`$ with $`\mu(\pi^*(x)\mid x)=1/c`$ does, then stating $`C^*`$ loses nothing.
  - Their $`(C^*-1)`$ refinement is also a statement about the supremum.

## 7. Linear setting, Approach 1 (`sec:ocb-qom-lin1`)

The note carries §2 to the linear model with two replacements:

- the batch mean at a pair becomes a least-squares fit on a batch of rounds;
- the $`+1`$ in its denominator becomes a phantom round with reward $`0`$ at the pair being scored.

One split serves every pair, and the features share information across pairs. So far the subsection has a setting, an algorithm and the canonical-case lemma, but no guarantee.

**Setting.** As in §1, $`S`$ and $`K`$ are finite and rewards lie in $`[0,1]`$. A known feature map $`\phi:\mathcal X\times\mathcal A\to\mathbb R^d`$ satisfies $`q^*(x,a)=\phi(x,a)^\top\theta^*`$ for an unknown $`\theta^*\in\mathbb R^d`$. Write $`\phi_t:=\phi(x_t,a_t)`$. The tabular model is the canonical case $`\phi(x,a)=e_{(x,a)}`$, with $`d=SK`$.

**Algorithm (Linear QoM-LCB, Approach 1).**

```math
\begin{aligned}
&\textbf{Input: } \mathcal D,\ \text{features } \phi,\ \text{number of batches } B,\ \text{level } \alpha \\
&\textbf{for } (x,a)\in\mathcal X\times\mathcal A: \\
&\qquad \text{split the rounds } \lbrace t\in[T]:(x_t,a_t)=(x,a)\rbrace \text{ uniformly at random into } \mathcal D^1(x,a),\dots,\mathcal D^B(x,a),\ \text{sizes as equal as possible} \\
&\mathcal D^b\leftarrow\textstyle\bigcup_{(x,a)}\mathcal D^b(x,a)\subseteq[T],\quad b\in[B] \qquad \triangleright\ \text{one split, stratified by pair} \\
&\textbf{for } (x,a)\in\mathcal X\times\mathcal A: \\
&\qquad \textbf{for } b\in[B]: \\
&\qquad\qquad \hat\theta^b(x,a)\in\arg\min_{\theta\in\mathbb R^d}\ \sum_{t\in\mathcal D^b}\big(r_t-\phi_t^\top\theta\big)^2+\big(0-\phi(x,a)^\top\theta\big)^2 \qquad \triangleright\ \text{phantom round } (\phi(x,a),0) \\
&\qquad\qquad \hat\mu^b(x,a)\leftarrow\phi(x,a)^\top\hat\theta^b(x,a) \qquad \triangleright\ \text{weakly pessimistic estimator} \\
&\qquad \underline q(x,a)\leftarrow q_\alpha\big(\hat\mu^b(x,a),\ b\in[B]\big) \qquad \triangleright\ \text{pessimistic estimator} \\
&\textbf{return } \hat\pi(x)\in\arg\max_{a\in\mathcal A}\underline q(x,a)
\end{aligned}
```

**Calibration.** The same as in §2: $`\alpha=1/65`$ and $`B\ge26\ln(2SK/\delta)`$.

**The value is well defined.** Let $`\Lambda^b:=\sum_{t\in\mathcal D^b}\phi_t\phi_t^\top`$, with pseudo-inverse $`(\Lambda^b)^+`$. Let $`c^b_t(x,a):=\phi(x,a)^\top(\Lambda^b)^+\phi_t`$ and $`v^b(x,a):=\phi(x,a)^\top(\Lambda^b)^+\phi(x,a)`$.

Suppose first that $`\phi(x,a)`$ lies in the span of the batch's features. Then the normal equations fix the component of $`\theta`$ in that span, and the Sherman–Morrison formula on that span gives

```math
\hat\mu^b(x,a)=\frac{\sum_{t\in\mathcal D^b}c^b_t(x,a)\,r_t}{1+v^b(x,a)} ,
```

which is the batch's least-squares prediction divided by $`1+v^b(x,a)`$. If instead $`\phi(x,a)`$ lies outside the span, every minimizer has $`\phi(x,a)^\top\theta=0`$, like an empty batch in §2.

Notes:

- **In mean, the phantom shrinks toward $`0`$, as the $`+1`$ does.**
  - Condition on the design and the split, and let $`\phi(x,a)`$ lie in the span. Then $`\sum_tc^b_t(x,a)q^*(x_t,a_t)=q^*(x,a)`$, so $`\mathbb E[\hat\mu^b(x,a)]=q^*(x,a)/(1+v^b(x,a))\le q^*(x,a)`$.
  - In the canonical case $`v^b(x,a)=1/\lvert\mathcal D^b(x,a)\rvert`$, and the shrinkage factor is that of §2.
- **Weak pessimism is not automatic.**
  - The coefficients $`c^b_t(x,a)`$ can be negative, while Feige's inequality, which is behind Corollary 3, needs a sum of non-negative terms.
  - For the same reason $`\hat\mu^b`$ and $`\underline q`$ can be negative, whereas $`\underline q\ge0`$ always holds in §2.
  - For general features, the note says, the guarantee $`\Pr(\hat\mu^b\le q^*)\ge1/13`$ needs a condition on the design.
  - In the canonical case, $`c^b_t(x,a)=1/\lvert\mathcal D^b(x,a)\rvert`$ at the pair's rounds and $`0`$ elsewhere.
- **The split.**
  - It depends on the pairs and on independent randomness, not on the rewards. So, given the design and the split, the $`B`$ estimators at a pair are independent.
  - Stratifying by pair makes the split that of §2 in the canonical case.
  - It also gives $`\Lambda^b\succeq\sum_{(x,a)}m(x,a)\phi(x,a)\phi(x,a)^\top`$.
- **Why not ridge.**
  - Ridge with penalty $`\lVert\theta\rVert_2^2`$ also reduces to §2 in the canonical case, since there it appends a phantom round at every coordinate.
  - However, its bias $`-\phi(x,a)^\top(\Lambda^b+I)^{-1}\theta^*`$ can have either sign.
  - The phantom at $`\phi(x,a)`$, by contrast, shrinks exactly the quantity being estimated.
- **Relation to the plug-in rule.** As in §2, $`\underline q`$ plays the role of $`\hat q-\Gamma`$.

**Lemma 5 (the canonical case; `lem:ocb-qom-lin1-canonical`).** Let $`\phi(x,a)=e_{(x,a)}`$ for every pair, with $`d=SK`$. Then at every pair, the split of the rewards $`\lbrace r_t:t\in\mathcal D^b(x,a)\rbrace`$, $`b\in[B]`$, is that of $`\mathcal D(x,a)`$ in §2, and for every batch

```math
\hat\mu^b(x,a)=\sum_{t\in\mathcal D^b(x,a)}\frac{r_t}{\lvert\mathcal D^b(x,a)\rvert+1} .
```

Hence Approach 1 is QoM-LCB.

*Proof.* With $`\phi_t=e_{(x_t,a_t)}`$, the criterion separates across coordinates. The coordinate $`\theta_{(x,a)}`$ enters only through the strictly convex quadratic $`f(u):=\sum_{t\in\mathcal D^b(x,a)}(r_t-u)^2+u^2`$.

- Its derivative is $`f'(u)=2(\lvert\mathcal D^b(x,a)\rvert+1)u-2\sum_tr_t`$, whose zero is the display; for an empty batch it is $`0`$.
- The quantile and greedy steps are the same in both algorithms. $`\square`$

The $`+1`$ is the phantom's contribution to the normal equation of the coordinate $`(x,a)`$: it turns the count $`\lvert\mathcal D^b(x,a)\rvert`$ into $`\lvert\mathcal D^b(x,a)\rvert+1`$.

## 8. Open questions, in priority order

1. **A linear guarantee.** This blocks everything in §7.
   - First, a condition on the design under which the weakly pessimistic property survives signed least-squares weights.
   - Then, a width in place of Lemma 4(2).
   - Two choices are under discussion (2026-09-29) and not yet settled in the note:
     - the split: stratified by pair, as in the note, or a plain uniform split into $`B`$ equal batches;
     - the mechanism behind weak pessimism: non-negative coefficients with comparable means, symmetric noise (the route of [[qom-value-based-pessimism-subgaussian]]), or injected noise.
2. **Move the staged results into the main text**, in the order §6.2 records.
3. **A lower bound for the second term.** Does QoM-LCB's own regret at a rare, badly covered context match the per-context charge (§6.2)?
4. **The parked $`\bar C^*`$ questions** (§6.3).
   - Check Proposition 8 against the literature.
   - Test the minimax hypothesis for why prior work states $`C^*`$, against Rashidinejad et al. (2021), Theorems 4 and 5.

## 9. The earlier version of this page

Until 2026-09-29, this page held an independent analysis of the same question. It was written before the Overleaf note and is superseded by it. It differed from the note in three ways:

- **its width** came from a Bernstein route, with constants $`4`$ and $`\tfrac{11}3`$ in place of the corrected Freedman route's $`4.69`$ and $`9`$;
- **its batches** were round-robin in time order, in place of a uniform random split;
- **its coverage** was a variance-weighted $`V^*`$, which is the note's $`\bar\sigma^2_*`$.

It also held results the note does not contain, which other pages still cite:

- **Proposition 2.** At this calibration the estimator's typical gap is about $`7.8`$ Bernstein widths. As $`B`$ grows, no leading constant valid at this calibration can fall below $`3.30`$; at $`B=149`$ the exact floor is $`2.89`$. [[cassel-qom-lemma1-erratum]] cites the floor.
- **Proposition 3.** A family of instances on which QoM-LCB's expected suboptimality is $`\Omega(\sqrt{SB/T})`$, while Rashidinejad et al.'s LCB attains $`\tilde O(T^{-3/4})`$. So QoM-LCB loses the $`(C^*-1)`$ refinement. [[qom-value-based-pessimism-subgaussian]] uses its construction.
- **Two corollaries and a table.**
  - An expert-data corollary: $`\mathbb E\,\Delta(\hat\pi)\le S/(eT)`$ at $`C^*=1`$, with ties broken toward the larger count.
  - A heavy-tail corollary, with finite second moments in place of $`[0,1]`$.
  - A comparison table against the Hoeffding, Bernstein and Rashidinejad LCBs.

**Recovering it.** From the vault repository, run `git show a1caf4e:research_vault/wiki/queries/2026-09-22-qom-value-based-pessimism.md`. [[qom-value-based-pessimism-subgaussian]] calls that version "the general page" throughout.

## Sources

- **The Overleaf note**, `02_offline_contextual_bandits.tex` at Overleaf commit `06b0730`:
  - §2.1 (*Coverage*; *Average versus worst-case coverage*);
  - the Tabular LCB theorem (`thm:ocb-tabular-lcb`);
  - §3.1–3.2 (`sec:ocb-qom`, `sec:ocb-qom-lin1`).
- **`temp.tex`**, at the same commit: `sec:tmp-domination`, `sec:tmp-count`, `sec:tmp-second-term` and `sec:tmp-cbar-only`.
- **How the results were checked.** They are transcribed, not re-derived. The numerical constants were re-evaluated in closed form when this page was written: $`\mathrm{kl}(1/65,1/13)`$, $`\mathrm{kl}(1/65,\delta_0)`$, $`L`$, $`2\sqrt{(e-2)L}`$, the threshold on $`\delta_0`$, $`(18/3.32)^2`$, $`\sqrt{18}`$ and $`\sqrt{8/26}`$.
- **[[cassel2026Quantile]]**: Eq. (1), Eq. (2), Corollary 2 and Lemma 1, the last as corrected in [[cassel-qom-lemma1-erratum]].
- **The black boxes, as the note cites them.**
  - Feige (STOC 2004), Theorem 1.
  - Beygelzimer, Langford, Li, Reyzin & Schapire (AISTATS 2011), Theorem 1.
  - Hoeffding (1963), Theorem 1.

  The first two were checked against their originals on 2026-09-22 ([[cassel-qom-lemma1-erratum]] §6).
- **Coverage.**
  - Rashidinejad, Zhu, Ma, Jiao & Russell (2021), Definition 1 and Theorems 4–5 (no paper page yet).
  - [[yin2023Offline]], Assumption 2.2.
- **Notation and context.** [[contextual-bandits-offline]] §2; [[contextual-bandits-offline-value-based]], Theorem 4.2 and the pessimistic plug-in rule; [[pessimism-principle]].
