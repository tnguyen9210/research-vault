---
date: 2026-09-22
question: "Can the Quantile of Means idea of Cassel & Rosenberg (2026) be adapted into value-based pessimism for offline contextual bandits? Propose an algorithm and write a Theorem 4.2-style analysis step by step, verifying every step mathematically."
---

# Quantile of Means as Value-Based Pessimism for Offline Contextual Bandits

**Scope.** The tabular value-based route of [[contextual-bandits-offline-value-based]] §4, with the explicit penalty of the rule (LCB) replaced by the quantile-of-means (QoM) estimator of [[cassel2026Quantile]]. Policy-based (importance-weighted) variants are out of scope. Every step below is proved from results whose statements were checked against their sources (§8); no step rests on simulation.

## Short answer

- **The mechanism transfers in the right direction.** QoM makes an estimate fall *below* its mean with high probability. Online, with losses, that is optimism; offline, with rewards, it is pessimism. Validity (Lemma 3) needs only non-negative rewards with finite means.
- **Guarantee in the form of Theorem 4.2.** Conditional on the design, with probability at least $`1-\delta`$ (Theorem 1),

```math
\Delta(\hat\pi) \le \mathbb E_{x\sim\nu}\Big[\min\Big\lbrace 1,\ 4\sqrt{\frac{\sigma^2(x,\pi^{\ast}(x))B}{\max\lbrace 1,N(x,\pi^{\ast}(x))\rbrace}}+\frac{11}{3}\frac{B}{\max\lbrace 1,N(x,\pi^{\ast}(x))\rbrace}\Big\rbrace\Big],\qquad B=\big\lceil 26\ln(2SK/\delta)\big\rceil,
```

and under a count condition (Theorem 2), $`\Delta(\hat\pi)\le 4\sqrt{2SV^{\ast}B/T}+\tfrac{22}{3}SC^{\ast}B/T`$, where $`V^{\ast}\le C^{\ast}/4`$ is a variance-weighted single-policy coverage.
- **What it buys.** No penalty to design, no variance to estimate, and no reward range needed for validity; heavy-tailed rewards are allowed (Corollary 4); a $`1/T`$ rate when the optimal actions' rewards are noiseless (Corollary 2).
- **What it costs.** A constant: at the calibration that makes Lemma 3 valid for every non-negative law, the estimator itself sits about $`7.8`$ Bernstein widths below the mean (Proposition 2), and this is intrinsic to the estimator, not slack in the proof. And it loses Rashidinejad's $`(C^{\ast}-1)`$ refinement (Proposition 3), a loss shared by every variance-adaptive penalty.
- **Found while verifying.** Cassel & Rosenberg's Lemma 1(2), the bias bound, is false as stated. Its constant $`1.7`$ comes from a Freedman corollary that dropped a $`\log(1/\delta)`$ from under a square root (§7.1). Lemma 4 below replaces it.

## 1. Setting and notation

Notation follows [[contextual-bandits-offline]] §2. The model is tabular, $`S:=\lvert\mathcal X\rvert`$ and $`K:=\lvert\mathcal A\rvert`$ finite, with context law $`\nu`$, behavior policy $`\mu`$, reward kernel $`\rho`$, mean $`q^{\ast}(x,a)`$, variance $`\sigma^2(x,a)`$, value $`J`$, a deterministic optimal policy $`\pi^{\ast}`$, and suboptimality $`\Delta(\hat\pi):=J(\pi^{\ast})-J(\hat\pi)`$. The data are $`T`$ i.i.d. triples $`(x_t,a_t,r_t)`$.

The **design** is the sequence of pairs $`(x_1,a_1),\dots,(x_T,a_T)`$, and $`N(x,a)`$ counts the rounds with $`(x_t,a_t)=(x,a)`$.

**Fact 0 (conditional independence).** Conditional on the design, the rewards are independent with $`r_t\sim\rho(\cdot\mid x_t,a_t)`$. So the rewards logged at a pair $`(x,a)`$ are $`N(x,a)`$ i.i.d. draws from $`\rho(\cdot\mid x,a)`$.
*Proof.* The joint law of the data is the product over $`t`$ of $`\nu(x_t)\mu(a_t\mid x_t)\rho(r_t\mid x_t,a_t)`$. Conditioning on all pairs leaves the product of the factors $`\rho(r_t\mid x_t,a_t)`$. $`\square`$

Shorthand: $`N^{\ast}(x):=N(x,\pi^{\ast}(x))`$, $`\sigma^2_{\ast}(x):=\sigma^2(x,\pi^{\ast}(x))`$ and $`\mu_x:=\mu(\pi^{\ast}(x)\mid x)`$, so that $`C^{\ast}=\max_{x:\nu(x)>0}1/\mu_x`$ and $`\bar C^{\ast}:=\sum_x\nu(x)/\mu_x\le C^{\ast}`$ as on the value-based page. New here is the **variance-weighted coverage**

```math
V^{\ast}:=\sum_{x}\frac{\nu(x)\sigma^2_{\ast}(x)}{\mu_x}\ \le\ \frac{\bar C^{\ast}}{4},
```

where the bound uses $`\sigma^2\le 1/4`$ for rewards in $`[0,1]`$.

## 2. Why the direction transfers

QoM rests on one undershoot fact (Lemma 1 below; Corollary 2 of Cassel & Rosenberg). A sum of i.i.d. non-negative variables, divided by one more than the number of terms, is at most its mean with probability at least $`1/13`$. It only ever gives *under*-estimation.

- **Online, losses.** An under-estimate of a loss is optimism. This is Cassel & Rosenberg's use.
- **Offline, rewards.** An under-estimate of a reward is pessimism. This is the use here, so no sign flip is needed.
- **Offline, losses.** Pessimism would need over-estimation, which the fact does not provide. One would pass to rewards $`L_{\max}-L`$, which needs a known upper bound.
- **Rewards of either sign.** Shift by a known lower bound first.

## 3. Algorithm: QoM-LCB

```math
\begin{aligned}
&\textbf{Input: } \mathcal D=\lbrace (x_t,a_t,r_t) : t=1,\dots,T\rbrace,\ \text{number of batches } B,\ \text{quantile level } \alpha,\ k:=\lceil\alpha B\rceil \\
&\textbf{for } (x,a)\in\mathcal X\times\mathcal A: \\
&\qquad t_1 < t_2 < \dots < t_{N(x,a)} \leftarrow \text{the rounds with } (x_t,a_t)=(x,a) \\
&\qquad D^{(b)}(x,a) \leftarrow \text{the rewards } r_{t_j} \text{ with } j\equiv b \pmod B,\quad b=1,\dots,B \qquad \triangleright\ \text{round-robin} \\
&\qquad \hat q^{(b)}(x,a) \leftarrow \frac{1}{n_b(x,a)+1}\sum_{r\in D^{(b)}(x,a)} r,\qquad n_b(x,a):=\big\lvert D^{(b)}(x,a)\big\rvert \qquad \triangleright\ \text{empty batch gives } 0 \\
&\qquad \underline q(x,a) \leftarrow \text{the } k\text{-th smallest of } \hat q^{(1)}(x,a),\dots,\hat q^{(B)}(x,a) \\
&\textbf{return } \hat\pi(x)\in\arg\max_{a\in\mathcal A}\underline q(x,a) \qquad \triangleright\ \text{ties toward the larger } N(x,a)
\end{aligned}
```

Set $`\alpha=1/65`$ and $`B=\lceil 26\ln(2SK/\delta)\rceil`$. Four remarks, each used below:

- The batch of a reward depends only on the design (its rank in time within its cell), never on reward values. Lemma 3's independence needs exactly this.
- Round-robin makes the batch sizes of a cell differ by at most one, so $`n_b\ge\lfloor n/B\rfloor`$ for every $`b`$. Lemma 4 needs this.
- Since rewards are non-negative, every $`\hat q^{(b)}\ge 0`$ and hence $`\underline q\ge 0`$.
- The tie-break is used only in Corollary 3. Nothing is tuned: $`\alpha`$ and $`B`$ are fixed by the analysis.

## 4. Assumptions

- **(A1) Data.** I.i.d. triples, as in [[contextual-bandits-offline]] §2.1.
- **(N) Non-negativity.** $`r\ge 0`$ almost surely. Together with finite means, this is the only distributional assumption behind validity (Lemma 3).
- **(A2) Boundedness.** $`r\in[0,1]`$, used for the width (Lemma 4). Lemma 4' replaces it by finite second moments.
- **Tabular.** $`S,K`$ finite.
- **For Theorem 2 only.** $`C^{\ast}<\infty`$, and the count condition $`T\nu(x)\mu_x\ge 8\ln(S/\delta')`$ for every $`x`$ with $`\nu(x)>0`$.
- **Not assumed.** The variances; the reward range, for validity; the reward family; the propensities $`\mu`$; the value of $`C^{\ast}`$; uniform coverage.

## 5. Core lemmas

Throughout this section, fix one cell, condition on the design, and write $`n`$ for its count, $`q`$ for its mean and $`\hat q^{(b)},\underline q`$ for its estimates. By Fact 0 its rewards are $`n`$ i.i.d. draws, and its batches are fixed disjoint sets of them.

**Lemma 1 (undershoot).** Let $`X_1,\dots,X_m\ge 0`$ be i.i.d. with mean $`q<\infty`$, and $`m\ge 0`$. Then

```math
\Pr\Big[\sum_{i=1}^{m}X_i\le(m+1)q\Big]\ge\frac1{13}.
```

*Proof.* If $`m=0`$ the sum is $`0\le q`$. If $`q=0`$ then $`X_i=0`$ almost surely and equality holds. Otherwise apply Feige's Theorem 1 (§8) to $`Y_i:=X_i/q`$. These are independent and non-negative with $`\mathbb E Y_i=1\le 1`$, so with $`\delta=1`$ it gives $`\Pr[\sum_iY_i < m+1]\ge\min\lbrace 1/2,1/13\rbrace=1/13`$. Multiply through by $`q`$. $`\square`$

*Verification note.* Cassel & Rosenberg's Corollary 2 states the strict version and divides by $`\mu`$. At $`\mu=0`$ the strict event is empty and the division undefined. The non-strict form is what Lemma 3 needs, and $`q=0`$ is no corner case offline: an action that never pays is an ordinary cell.

**Lemma 2 (binomial tails).** Let $`\mathrm{kl}(a,p):=a\ln\frac ap+(1-a)\ln\frac{1-a}{1-p}`$ and $`Z\sim\mathrm{Bin}(B,p)`$. Then $`\Pr[Z\le aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\le p`$, and $`\Pr[Z\ge aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\ge p`$ (the Chernoff–Hoeffding bound).

For a sum $`Z'=\sum_bZ_b`$ of independent Bernoulli variables with means $`p_b`$:

- if $`p_b\ge p`$ for all $`b`$, then $`\Pr[Z'\le z]\le\Pr[\mathrm{Bin}(B,p)\le z]`$;
- if $`p_b\le p`$ for all $`b`$, then $`\Pr[Z'\ge z]\le\Pr[\mathrm{Bin}(B,p)\ge z]`$.

Both follow by coupling $`Z_b=\mathbb 1\lbrace U_b\le p_b\rbrace`$ with i.i.d. uniform $`U_b`$. Two constants are used below, each with a proof checkable by hand:

- (i) $`\mathrm{kl}(1/65,1/13)=\big(64\ln\tfrac{16}{15}-\ln 5\big)/65\ge 1/26`$. Truncating the alternating series of $`\ln(1+x)`$ after its fourth term gives a lower bound on $`(0,1)`$, so $`\ln\tfrac{16}{15}\ge 0.0645382`$. Hence $`64\ln\tfrac{16}{15}-\ln5\ge 4.13044-1.60944=2.52100\ge\tfrac52`$.
- (ii) $`\mathrm{kl}(1/65,e^{-8})\ge\big(8-\ln65+64\ln\tfrac{64}{65}\big)/65\ge 1/26`$. The first step drops the positive term $`-\tfrac{64}{65}\ln(1-e^{-8})`$. Then $`\ln 65\le 4.1744`$ and $`64\ln\tfrac{64}{65}\ge -1`$ (as $`\ln(1+\tfrac1{64})\le\tfrac1{64}`$) give $`8-4.1744-1=2.8256\ge\tfrac52`$.

**Lemma 3 (QoM is pessimistic).** With $`\alpha=1/65`$, $`k=\lceil\alpha B\rceil`$ and any $`B\ge1`$, under (N) with finite mean, $`\Pr[\underline q\le q]\ge 1-e^{-B/26}`$.

*Proof.* The $`k`$-th smallest of $`B`$ numbers is at most $`q`$ iff at least $`k`$ of them are. Let $`Z_b:=\mathbb 1\lbrace\hat q^{(b)}\le q\rbrace`$. Batch $`b`$ holds $`n_b`$ rewards, and $`\hat q^{(b)}\le q`$ iff their sum is at most $`(n_b+1)q`$, so $`\Pr[Z_b=1]\ge 1/13`$ by Lemma 1. The $`Z_b`$ are independent: they are functions of disjoint sets of rewards, the sets are fixed by the design, and the rewards are independent given the design (Fact 0). Since $`k-1 < B/65`$, Lemma 2 and constant (i) give

```math
\Pr\Big[\sum_bZ_b\le k-1\Big]\le\Pr\big[\mathrm{Bin}(B,\tfrac1{13})\le\tfrac{B}{65}\big]\le e^{-B\cdot\mathrm{kl}(1/65,1/13)}\le e^{-B/26}.\qquad\square
```

**Lemma 4 (width of QoM, bounded rewards).** Under (A2), with variance $`\sigma^2`$ and round-robin batches, with probability at least $`1-e^{-B/26}`$,

```math
\underline q\ \ge\ q-4\sqrt{\frac{\sigma^2B}{\max\lbrace1,n\rbrace}}-\frac{11}{3}\frac{B}{\max\lbrace1,n\rbrace}.
```

*Proof.* **Step 1 (one batch).** Let $`m:=n_b`$ and $`Y_i:=q-X_i`$ over the batch's rewards $`X_i`$. Then $`\mathbb EY_i=0`$, $`Y_i\le q\le 1`$ and $`\mathbb EY_i^2=\sigma^2`$. Bernstein's inequality (§8) with $`v=m\sigma^2`$, $`b=1`$ and $`t=8`$ gives $`\Pr\big[\sum_iY_i\ge\sqrt{16m\sigma^2}+\tfrac83\big]\le e^{-8}`$. So with probability at least $`1-e^{-8}`$, $`\sum_iX_i\ge mq-4\sqrt{m\sigma^2}-\tfrac83`$. This is trivial when $`m=0`$.

**Step 2 (divide).** On that event,

```math
\hat q^{(b)}=\frac{\sum_iX_i}{m+1}\ \ge\ q-\frac{q+4\sqrt{m\sigma^2}+8/3}{m+1}.
```

Round-robin gives $`m\ge\lfloor n/B\rfloor`$, hence $`m+1\ge\max\lbrace1,n\rbrace/B`$. Also $`\sqrt m/(m+1)\le 1/\sqrt{m+1}`$, since $`\sqrt m\le\sqrt{m+1}`$. With $`q\le1`$ these give $`\hat q^{(b)}\ge\tau`$, where $`\tau:=q-4\sqrt{\sigma^2B/\max\lbrace1,n\rbrace}-\tfrac{11}{3}B/\max\lbrace1,n\rbrace`$.

**Step 3 (quantile).** $`\underline q < \tau`$ iff at least $`k`$ batches have $`\hat q^{(b)} < \tau`$. The threshold $`\tau`$ is fixed given the design, and each batch falls below it with probability at most $`e^{-8}`$, independently across batches. Since $`k\ge B/65`$, Lemma 2 and constant (ii) give $`\Pr[\underline q < \tau]\le\Pr\big[\mathrm{Bin}(B,e^{-8})\ge\tfrac B{65}\big]\le e^{-B\cdot\mathrm{kl}(1/65,e^{-8})}\le e^{-B/26}`$. $`\square`$

*Verification note.* This replaces Lemma 1(2) of Cassel & Rosenberg, whose constants $`1.7`$ and $`9`$ are not valid (§7.1). Its leading constant $`4`$ cannot fall below $`3.30`$ for this estimator at this confidence (Proposition 2(b)).

**Lemma 4′ (width of QoM, heavy tails).** Under (N) with second moment $`s^2:=\mathbb E r^2<\infty`$ and no upper bound, with probability at least $`1-e^{-B/26}`$,

```math
\underline q\ \ge\ q-4\sqrt{\frac{s^2B}{\max\lbrace1,n\rbrace}}-\frac{qB}{\max\lbrace1,n\rbrace}.
```

*Proof.* The function $`f(u):=1-u+u^2/2-e^{-u}`$ has $`f(0)=0`$ and $`f'(u)=u-1+e^{-u}\ge0`$, so $`e^{-u}\le 1-u+u^2/2`$ for $`u\ge0`$. Hence for $`\lambda\ge0`$, $`\mathbb E e^{-\lambda X}\le 1-\lambda q+\lambda^2s^2/2\le e^{-\lambda q+\lambda^2s^2/2}`$. By Markov's inequality applied to $`e^{-\lambda\sum X_i}`$ and independence, $`\Pr[\sum_iX_i\le mq-t]\le e^{-\lambda t+\lambda^2ms^2/2}`$. Choosing $`\lambda=t/(ms^2)`$ gives $`e^{-t^2/(2ms^2)}`$, which is $`e^{-8}`$ at $`t=4\sqrt{ms^2}`$. Steps 2 and 3 of Lemma 4 then apply with $`8/3`$ replaced by $`0`$; the remaining shrinkage term is $`q/(m+1)\le qB/\max\lbrace1,n\rbrace`$. $`\square`$

**Lemma 5 (one-sided pessimism reduction).** On the event that $`\underline q\le q^{\ast}`$ at every cell, for every deterministic policy $`\pi`$ and every $`x`$,

```math
q^{\ast}(x,\pi(x))-q^{\ast}(x,\hat\pi(x))\ \le\ q^{\ast}(x,\pi(x))-\underline q(x,\pi(x)).
```

*Proof.* $`q^{\ast}(x,\hat\pi(x))\ge\underline q(x,\hat\pi(x))\ge\underline q(x,\pi(x))`$, by the event and then by the definition of $`\hat\pi`$. $`\square`$

The difference from Lemma 2.14 of the value-based page: there a two-sided width $`\Gamma\ge\lvert\hat q-q^{\ast}\rvert`$ is charged twice, while here the charge is $`q^{\ast}-\underline q`$ itself, once.

**Lemma 6 (counts).** $`N^{\ast}(x)\sim\mathrm{Bin}(T,\nu(x)\mu_x)`$, and by the multiplicative Chernoff bound $`\Pr[N^{\ast}(x)\le T\nu(x)\mu_x/2]\le e^{-T\nu(x)\mu_x/8}`$. This is the step used in the proof of Theorem 4.2.

## 6. Main results

**Theorem 1 (QoM-LCB, conditional on the design).** Assume (A1) and (A2) in the tabular model, and run QoM-LCB with $`\alpha=1/65`$ and $`B\ge 26\ln(2SK/\delta)`$. Conditional on the design, with probability at least $`1-\delta`$, simultaneously for every deterministic $`\pi`$,

```math
J(\pi)-J(\hat\pi)\ \le\ \mathbb E_{x\sim\nu}\Big[\min\Big\lbrace q^{\ast}(x,\pi(x)),\ 4\sqrt{\frac{\sigma^2(x,\pi(x))B}{\max\lbrace1,N(x,\pi(x))\rbrace}}+\frac{11}{3}\frac{B}{\max\lbrace1,N(x,\pi(x))\rbrace}\Big\rbrace\Big].
```

In particular, with $`\pi=\pi^{\ast}`$ and $`q^{\ast}\le1`$, this gives the display in the short answer.

*Proof.* Let $`E_p`$ be the event that $`\underline q\le q^{\ast}`$ at all $`SK`$ cells, and $`E_w`$ the event that Lemma 4's bound holds at all $`SK`$ cells. Conditioning on the design fixes each cell's count and batches, so Lemmas 3 and 4 apply cell by cell. Each of the $`2SK`$ events fails with probability at most $`e^{-B/26}\le\delta/(2SK)`$, and a union bound gives $`\Pr[E_p\cap E_w]\ge1-\delta`$. On $`E_p\cap E_w`$, for every $`x`$:

- Lemma 5 bounds $`q^{\ast}(x,\pi(x))-q^{\ast}(x,\hat\pi(x))`$ by $`q^{\ast}(x,\pi(x))-\underline q(x,\pi(x))`$;
- Lemma 4 bounds that by the width, and $`\underline q\ge0`$ bounds it by $`q^{\ast}(x,\pi(x))`$.

Average over $`x\sim\nu`$. $`\square`$

Against Theorem 4.2 the structure is the same: a minimum of $`1`$ and a per-context width at the comparator's action, conditional on the design, with a union bound over $`SK`$ cells. Only the width differs. Here it is $`4\sqrt{\sigma^2B/N}+\tfrac{11}{3}B/N`$, charged once; there it is the Hoeffding width $`\sqrt{\ln(2SK/\delta)/(2N)}`$, charged twice.

**Theorem 2 (coverage form).** If in addition $`T\nu(x)\mu_x\ge 8\ln(S/\delta')`$ for every $`x`$ with $`\nu(x)>0`$, then with probability at least $`1-\delta-\delta'`$,

```math
\Delta(\hat\pi)\ \le\ 4\sqrt{\frac{2SV^{\ast}B}{T}}+\frac{22}{3}\frac{SC^{\ast}B}{T}.
```

*Proof.* Lemma 6 and a union bound over at most $`S`$ contexts give $`N^{\ast}(x)\ge T\nu(x)\mu_x/2`$ for all such $`x`$, with probability at least $`1-\delta'`$. Theorem 1's event has probability at least $`1-\delta`$ unconditionally too, by the tower property. On both events,

```math
\Delta(\hat\pi)\le\sum_x\nu(x)\Big[4\sqrt{\frac{2\sigma^2_{\ast}(x)B}{T\nu(x)\mu_x}}+\frac{11}{3}\frac{2B}{T\nu(x)\mu_x}\Big]=4\sqrt{\frac{2B}{T}}\sum_x\sqrt{\frac{\nu(x)\sigma^2_{\ast}(x)}{\mu_x}}+\frac{22B}{3T}\sum_x\frac1{\mu_x}.
```

Cauchy–Schwarz over at most $`S`$ terms bounds the first sum by $`\sqrt{SV^{\ast}}`$. Each term of the second sum is at most $`C^{\ast}`$. $`\square`$

*Remark (dropping the count condition).* Let $`\mathcal L:=\lbrace x:T\nu(x)\mu_x < 8\ln(S/\delta')\rbrace`$. On $`\mathcal L`$, $`\nu(x) < 8\ln(S/\delta')/(T\mu_x)\le 8C^{\ast}\ln(S/\delta')/T`$, and the regret at each such $`x`$ is at most $`1`$. So these contexts cost at most $`8SC^{\ast}\ln(S/\delta')/T`$ in total, and Theorem 2 applies to the rest. This is the same device as Rashidinejad's missing-mass term.

**Corollary 1 (worst case).** From $`\sigma^2\le1/4`$, $`V^{\ast}\le\bar C^{\ast}/4\le C^{\ast}/4`$, so $`\Delta(\hat\pi)\le\sqrt{8S\bar C^{\ast}B/T}+\tfrac{22}{3}SC^{\ast}B/T`$. This is Theorem 4.2's rate $`\tilde O(\sqrt{SC^{\ast}/T})`$. At $`B=\lceil26\ln(2SK/\delta)\rceil`$ the leading constant is $`\sqrt{208}\approx14.4`$, against Theorem 4.2's $`2`$.

**Corollary 2 (noiseless optimal actions).** If $`\sigma^2_{\ast}\equiv0`$:

- Theorem 1 gives $`\Delta(\hat\pi)\le\mathbb E_\nu\big[\min\lbrace1,\tfrac{11}{3}B/\max\lbrace1,N^{\ast}(x)\rbrace\rbrace\big]`$;
- under the count condition, Theorem 2 gives $`\Delta(\hat\pi)\le\tfrac{22}{3}SC^{\ast}B/T`$.

This is a $`1/T`$ rate without a gap condition. The Bernstein rule (Proposition 7.1) has the same form but needs $`\sigma^2`$ or its empirical estimate.

**Corollary 3 (expert data).** If $`C^{\ast}=1`$ and ties go to the larger count, then $`\mathbb E_{\mathcal D}[\Delta(\hat\pi)]\le\sum_x\nu(x)(1-\nu(x))^T\le S/(eT)`$.

*Proof.*

1. $`C^{\ast}=1`$ forces $`\mu_x=1`$. No other action is ever logged at $`x`$, so every other action has all batches empty and $`\underline q=0\le\underline q(x,\pi^{\ast}(x))`$.
2. If $`N^{\ast}(x)\ge1`$, the tie-break selects $`\pi^{\ast}(x)`$. So the regret at $`x`$ is at most $`\mathbb 1\lbrace N^{\ast}(x)=0\rbrace`$, whose probability is $`(1-\nu(x))^T`$.
3. Finally, $`\max_{z\in[0,1]}z(1-z)^T`$ is attained at $`z=1/(T+1)`$ and equals $`\tfrac1T\big(1-\tfrac1{T+1}\big)^{T+1}\le\tfrac1{eT}`$. $`\square`$

This matches the missing-mass term of Rashidinejad's Theorem 4 at $`C^{\ast}=1`$, which the value-based page writes as $`4S/(9T)`$. No failure probability enters.

**Corollary 4 (heavy tails).** Replace (A2) by (N) with finite second moments $`s^2(x,a):=\mathbb E[r^2\mid x,a]`$. Then Theorem 1 holds with $`\sigma^2`$ replaced by $`s^2`$ and $`\tfrac{11}{3}`$ by $`q^{\ast}(x,\pi(x))`$. The proof uses Lemma 4′ in place of Lemma 4, and validity (Lemma 3) needs only finite means.

**Proposition 2 (the width is the estimator's, not the proof's).** Let a cell's rewards be Bernoulli with parameter $`1/2`$, so $`\sigma=1/2`$. Let each batch hold $`m`$ rewards, so $`n=mB`$, and keep $`B`$ fixed. Then for every $`c>0`$,

```math
\lim_{m\to\infty}\Pr\Big[q-\underline q\ge c\sqrt{\sigma^2B/n}\Big]=\Pr\big[\mathrm{Bin}(B,\Phi(-c))\ge k\big].
```

*Proof.* $`q-\underline q\ge c\sqrt{\sigma^2B/n}=c\sigma/\sqrt m`$ iff at least $`k`$ batches satisfy $`\hat q^{(b)}\le q-c\sigma/\sqrt m`$. For each batch, with $`S_b\sim\mathrm{Bin}(m,1/2)`$, this is $`S_b\le(m+1)(1/2-c/(2\sqrt m))`$. In standardized form the threshold is $`1/\sqrt m-c(m+1)/m\to-c`$. By the de Moivre–Laplace theorem, and a sandwich between the limits at $`-c\pm\varepsilon`$ with $`\varepsilon\to0`$, the per-batch probability tends to $`\Phi(-c)`$. The batches are i.i.d., and $`\Pr[\mathrm{Bin}(B,p)\ge k]`$ is continuous in $`p`$. $`\square`$

Consequences:

- **(a) Typical gap.** As $`B\to\infty`$ with $`k/B\to1/65`$, the median of $`(q-\underline q)/\sqrt{\sigma^2B/n}`$ tends to $`\lvert\Phi^{-1}(1/65)\rvert=2.160`$. With $`B=26\ln(1/\delta')`$ the estimate therefore typically sits $`2.160\sqrt{26}\cdot\sigma\sqrt{\ln(1/\delta')/n}`$ below the mean. Bernstein's penalty is $`\sqrt{2\ln(1/\delta')}\cdot\sigma/\sqrt n`$, so the ratio is $`2.160\sqrt{13}=7.79`$.
- **(b) Floor on any valid constant.** Suppose a bound $`q-\underline q\le c\sqrt{\sigma^2B/n}+c'B/n`$ held with probability at least $`1-e^{-B/26}`$ for all $`m`$. Letting $`m\to\infty`$ forces $`\Pr[\mathrm{Bin}(B,\Phi(-c))\ge k]\le e^{-B/26}`$. With the lower bound $`\Pr[\mathrm{Bin}(B,p)\ge k]\ge e^{-B\cdot\mathrm{kl}(k/B,p)}/(B+1)`$, which follows from $`\binom Bk\ge e^{B\cdot h(k/B)}/(B+1)`$, the limit $`B\to\infty`$ forces two things. First $`\Phi(-c) < 1/65`$: otherwise the probability stays bounded away from $`0`$, tending to $`1`$ when $`\Phi(-c) > 1/65`$ by the law of large numbers and to $`1/2`$ at equality by the central limit theorem, while $`e^{-B/26}\to0`$. Then $`\mathrm{kl}(1/65,\Phi(-c))\ge1/26`$, which, as $`\mathrm{kl}(1/65,p)`$ decreases on $`p < 1/65`$, means $`\Phi(-c)\le4.83\times10^{-4}`$, so $`c\ge3.30`$. Lemma 4's $`4`$ is within about 21 percent of this floor. At $`B=149`$ the exact floor for that finite $`B`$ is $`2.89`$.
- **(c) Cassel & Rosenberg's constant fails the floor.** $`\Phi(-1.7)=0.0446 > 1/65`$, so $`\Pr[\mathrm{Bin}(B,0.0446)\ge k]\ge1-e^{-B\cdot\mathrm{kl}(1/65,0.0446)}=1-e^{-0.0133B}`$. This exceeds $`e^{-B/26}`$ for every $`B\ge 60`$ (§7.1).

**Proposition 3 (no $`(C^{\ast}-1)`$ adaptivity).** There are instances on which QoM-LCB's expected suboptimality is $`\Omega(\sqrt{SB/T})`$ while Rashidinejad's LCB achieves $`\tilde O(T^{-3/4})`$.

*Instance.* Take $`S\ge2`$ contexts with $`\nu`$ uniform, and two actions $`a^{\ast},a^{\circ}`$ with $`\mu(a^{\ast}\mid x)=1-\eta`$ and $`\mu(a^{\circ}\mid x)=\eta`$. Rewards at $`a^{\ast}`$ are Bernoulli with parameter $`1/2`$; the reward at $`a^{\circ}`$ is the constant $`1/2-\varepsilon`$. So $`\pi^{\ast}\equiv a^{\ast}`$ and $`C^{\ast}=1/(1-\eta)`$. Fix $`c=1`$, write $`p_c:=\Pr[\mathrm{Bin}(B,\Phi(-c))\ge k] > 0`$, and choose

```math
\bar m:=\frac{(1-\eta)T}{SB},\qquad\varepsilon:=\frac{c}{4\sqrt{\bar m}},\qquad\eta:=\frac8c\sqrt{\frac{BS}{T}}.
```

*Proof.* Fix $`x`$. Let $`N^{\circ}:=N(x,a^{\circ})`$, and let $`m^{\ast}:=\lfloor N^{\ast}(x)/B\rfloor`$ be the smaller batch size at $`a^{\ast}`$.

1. **The decoy's estimate.** Batch sizes at $`a^{\circ}`$ are at least $`\lfloor N^{\circ}/B\rfloor`$, and $`\lfloor N^{\circ}/B\rfloor+1\ge N^{\circ}/B`$. So $`\underline q(x,a^{\circ})\ge(1/2-\varepsilon)(1-B/N^{\circ})\ge1/2-\varepsilon-B/(2N^{\circ})`$.
2. **The optimal action's estimate.** Let $`A_x:=\lbrace\underline q(x,a^{\ast})\le1/2-c/(2\sqrt{m^{\ast}})\rbrace`$. Its conditional probability given the design tends to $`p_c`$ as $`m^{\ast}\to\infty`$ (Proposition 2's argument, with batch sizes $`m^{\ast}`$ or $`m^{\ast}+1`$).
3. **When the decoy wins.** On $`A_x`$, $`a^{\circ}`$ is selected whenever $`\varepsilon+B/(2N^{\circ}) < c/(2\sqrt{m^{\ast}})`$.
4. **The counts cooperate.** On $`\lbrace N^{\circ}\ge T\eta/(2S)\rbrace`$, the choice of $`\eta`$ gives $`B/(2N^{\circ})\le(c/8)\sqrt{BS/T}\le c/(8\sqrt{\bar m})`$. On $`\lbrace m^{\ast} < \tfrac{16}9\bar m\rbrace`$, $`3c/(8\sqrt{\bar m}) < c/(2\sqrt{m^{\ast}})`$. Both events have probability tending to $`1`$ by Chernoff, since $`T\eta/S\to\infty`$ and $`m^{\ast}/\bar m\to1`$.
5. **Conclusion.** So $`\Pr[\hat\pi(x)=a^{\circ}]\ge p_c-o(1)`$, and each such selection costs $`\varepsilon`$. Hence $`\mathbb E[\Delta(\hat\pi)]\ge(p_c-o(1))\varepsilon\ge\big(\tfrac{cp_c}4-o(1)\big)\sqrt{SB/T}`$.

Meanwhile $`C^{\ast}-1\le2\eta`$, so Rashidinejad's Theorem 4 bound on the same instances is $`\tilde O\big(\sqrt{S\eta/T}+S/T\big)=\tilde O\big(S^{3/4}B^{1/4}T^{-3/4}\big)`$. $`\square`$

*Why.* Rashidinejad's proof keeps the rule from choosing a rarely logged action at well-covered contexts, because its penalty $`\sqrt{L/N}`$ grows as the count falls. QoM's implicit penalty at a noiseless cell is only the shrinkage $`q/(n_b+1)\approx qB/N`$. The same construction defeats any penalty that vanishes like $`1/N`$ at a noiseless cell. That includes the Bernstein rule of Proposition 7.1, whose penalty there is $`2\ln(2SK/\delta)/(3N)`$, and its empirical-Bernstein version, since the sample variance at $`a^{\circ}`$ is $`0`$. Variance adaptivity at $`\pi^{\ast}`$'s actions (Corollary 2) and $`(C^{\ast}-1)`$ adaptivity come from opposite features of the penalty. Whether one rule can have both is open.

## 7. What did not carry over from Cassel & Rosenberg

### 7.1 The constants of Lemma 1(2)

Their proof in §A.1 applies their Lemma 14, a Freedman-type inequality of Beygelzimer et al. (2011), to one batch. It uses the corollary stated under Lemma 14: with $`\lambda=\min\lbrace R^{-1},\sqrt{\log(1/\delta)/((e-2)\sigma^2T)}\rbrace`$, $`\sum_tX_t\le2\sqrt{(e-2)\sigma^2T}+R\log(1/\delta)`$.

The stated $`\lambda`$ actually gives $`(e-2)\lambda\sigma^2T+\lambda^{-1}\log(1/\delta)=2\sqrt{(e-2)\sigma^2T\log(1/\delta)}`$ when $`\lambda < R^{-1}`$. When $`\lambda=R^{-1}`$ it gives at most $`\sqrt{(e-2)\sigma^2T\log(1/\delta)}+R\log(1/\delta)`$. So the correct corollary is

```math
\sum_tX_t\le2\sqrt{(e-2)\sigma^2T\log(1/\delta)}+R\log(1/\delta).
```

Beygelzimer et al.'s Theorem 1 has the logarithm under the root too: $`S\le\sqrt{(e-2)\ln(1/\delta)}\big(V/\sqrt{V'}+\sqrt{V'}\big)`$ for $`V'\ge R^2\ln(1/\delta)/(e-2)`$.

Their per-batch confidence is $`0.99953`$, i.e. $`\delta=4.7\times10^{-4}`$. At that value $`2\sqrt{e-2}=1.695`$ is exactly their $`1.7`$, and $`\log(1/\delta)=7.66`$ is their $`8R`$. The coefficient their route actually supports is $`1.695\sqrt{7.66}\approx4.69`$.

Proposition 2 shows the stated lemma is false, not merely loosely proved. For Bernoulli rewards with parameter $`1/2`$ and $`n\to\infty`$, the event of Lemma 1(2) holds with probability tending to $`\Pr[\mathrm{Bin}(B,\Phi(-1.7))\le k-1]\le e^{-0.0133B}\le\delta^{0.3447}`$ when $`B\ge26\ln(1/\delta)`$. That is below the claimed $`1-\delta`$ for every $`\delta < 0.32`$. Rates, including their Theorem 4's, are unaffected; its constants $`22`$ and $`1924`$ were not re-derived here. The error is flagged on [[cassel2026Quantile]].

### 7.2 The other points

- **Corollary 2's strict inequality.** It fails at $`\mu=0`$; Lemma 1 uses the non-strict form.
- **Independence.** Offline it is exact given the design (Fact 0). Cassel & Rosenberg need a union over dataset-size configurations and the $`V^{\ast}`$ substitution because their batches are built online and backups couple steps. At horizon one neither is needed.
- **Batch assignment.** It must be design-measurable; the round-robin must be by time rank, not by reward.
- **One-sided charge.** Lemma 5 charges the gap once, where Lemma 2.14 charges twice.
- **The $`(C^{\ast}-1)`$ step.** Rashidinejad's step does not transfer (Proposition 3).

## 8. Verification of the techniques used

Each black box, where it was checked, and the hypotheses verified at the point of use:

| result | checked against | used in | hypotheses verified |
|---|---|---|---|
| Feige's Theorem 1: independent $`X_i\ge0`$, $`\mathbb EX_i\le1`$, $`\Pr[X < \mu+\delta]\ge\min\lbrace\delta/(1+\delta),1/13\rbrace`$ | Feige, STOC 2004, p. 594 (PDF read 2026-09-22); agrees with Cassel & Rosenberg's Lemma 13 | Lemma 1 | $`Y_i=X_i/q`$ independent, non-negative, mean $`1`$; $`\delta=1`$ |
| Chernoff–Hoeffding bound $`\Pr[\mathrm{Bin}(B,p)\ge aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\ge p`$, and the lower-tail mirror | standard (Hoeffding 1963); the same bound Cassel & Rosenberg use | Lemmas 3, 4 | $`a=1/65`$ below $`p=1/13`$; above $`p=e^{-8}`$ |
| Bernstein: independent $`X_i\le b`$, $`v=\sum\mathbb EX_i^2`$, $`\Pr[S\ge\sqrt{2vt}+bt/3]\le e^{-t}`$ | standard (Boucheron, Lugosi & Massart 2013, Theorem 2.10); cited, not re-derived | Lemma 4 | $`Y_i=q-X_i\le1`$, centered, $`v=m\sigma^2`$ |
| multiplicative Chernoff $`\Pr[X\le\mu/2]\le e^{-\mu/8}`$ | standard; as in the proof of Theorem 4.2 | Lemma 6 | $`N^{\ast}(x)`$ binomial |
| Freedman form of Beygelzimer et al. (2011), Theorem 1 | arXiv:1002.4058, p. 3 (read 2026-09-22) | §7.1 only | $`X_t\le R`$, zero conditional mean |
| de Moivre–Laplace; $`\binom Bk\ge e^{B\cdot h(k/B)}/(B+1)`$ | standard | Propositions 2, 3 | Bernoulli batches with parameter $`1/2`$; $`0 < k < B`$ |

Bernstein's inequality is used only in its standard one-sided form for independent, centered summands bounded above by $`b`$, with $`v`$ the sum of their variances. At its point of use (Lemma 4, Step 1) the summands $`q-X_i`$ are independent given the design, centered, bounded above by $`q\le1`$, and have variance $`\sigma^2`$ each.

Feige's paper states the conjecture that $`1/13`$ can be replaced by $`1/e`$. A claimed general proof (arXiv:2508.07316) was withdrawn by its author as flawed. A note on the identically distributed case, the case that matters here, is an unrefereed preprint (arXiv:2509.19949). Neither is used.

## 9. How it compares

| rule | computable | validity needs | penalty at a cell with $`n`$ samples | coverage-form bound |
|---|---|---|---|---|
| Hoeffding LCB (Theorem 4.2) | yes | range $`[0,1]`$ | $`\sqrt{\ln(2SK/\delta)/(2n)}`$, charged twice | $`2\sqrt{S\bar C^{\ast}\ln(2SK/\delta)/T}`$ |
| Bernstein LCB (Proposition 7.1) | no, needs $`\sigma^2`$ | range and $`\sigma^2`$ | $`\sqrt{2\sigma^2\ln(2SK/\delta)/n}+2\ln(2SK/\delta)/(3n)`$, charged twice | variance form |
| empirical-Bernstein LCB | yes | range | as Bernstein, up to lower-order terms (Maurer & Pontil 2009; from memory) | variance form |
| Rashidinejad LCB (Theorem 4.5) | yes | range | $`\sqrt{2000\ln(2SK/\delta)/n}`$ | $`\tilde O(\sqrt{S(C^{\ast}-1)/T}+S/T)`$, in expectation |
| **QoM-LCB (Theorem 1)** | yes | non-negativity | none explicit; the gap is at most $`4\sqrt{\sigma^2B/n}+\tfrac{11}3B/n`$, charged once | $`4\sqrt{2SV^{\ast}B/T}+\tfrac{22}3SC^{\ast}B/T`$ |

**Reading the table.**

- **Constants.** Proposition 2(a): the estimator's typical gap is about $`7.8`$ Bernstein penalties. The source is the $`1/13`$ in Feige's theorem, a worst case over all non-negative laws, which fixes both $`\alpha`$ and $`B`$. If a larger undershoot probability $`p_F`$ could be certified for a narrower class, then $`B=\ln(1/\delta')/\mathrm{kl}(\alpha,p_F)`$, and the Gaussian-regime ratio becomes $`\lvert\Phi^{-1}(\alpha)\rvert/\sqrt{2\mathrm{kl}(\alpha,p_F)}`$. Evaluated with the best $`\alpha`$, this is $`7.4`$ at $`p_F=1/13`$, $`5.3`$ at $`1/8`$, $`3.1`$ at $`1/4`$ and $`2.1`$ at $`0.37`$. At $`p_F=1/2`$ it tends to $`\sqrt{\pi/2}\approx1.25`$ as $`\alpha\to1/2`$, the median's efficiency loss, by a Taylor expansion of both terms.
- **Where QoM-LCB is worth it.** Rewards whose range is unknown or unbounded (Corollary 4). Noiseless optimal actions (Corollary 2), which Bernstein-type rules also handle but only with a variance estimate. And, the real motivation, settings where no computable penalty exists. For bounded tabular problems, an empirical-Bernstein LCB is the better choice on constants.
- **Scaling.** The penalty is variance-scaled rather than count-scaled, which is both the source of Corollary 2 and the reason for Proposition 3.

## 10. Open problems, in priority order

1. **Novelty.** Not yet checked: whether median-of-means or quantile-of-means pessimism for offline bandits exists, or whether Cassel et al. (2025) has an offline version. Do this before claiming anything.
2. **Variance adaptivity together with $`(C^{\ast}-1)`$ adaptivity.** Proposition 3 separates the two mechanisms. Is there a rule, or a lower bound, for both at once?
3. **Better undershoot constants.** Feige's conjecture ($`1/e`$) for i.i.d. summands would cut the factor from about $`7.4`$ to about $`2.1`$ at the best $`\alpha`$ (§9). A constant proved only for $`[0,1]`$-valued rewards would already help.
4. **Function approximation.** Feige's theorem allows non-identically distributed summands. So a design-measurable weighted average with non-negative weights undershoots its own mean with probability at least $`1/13`$, after a shrinkage of the order of its largest weight. That covers histogram, $`k`$-NN and kernel smoothers, leaving only their approximation bias. Least squares has signed weights and would need a new undershoot lemma. Pessimism over a continuum of pairs needs covering, which likely forces $`B`$ proportional to the dimension: the horizon-one analogue of Cassel & Rosenberg's "ensemble size linear in $`S`$".
5. **An empirical check** against Hoeffding and empirical-Bernstein LCBs, with both the analysis constants and a tuned $`(\alpha,B)`$. This is the only item that is not theoretical.

## 11. Next steps

1. Novelty search, about 30 minutes; it gates the rest.
2. Write Theorems 1–2 and Lemmas 1–6 into `02_offline_contextual_bandits.tex` on Overleaf, where this analysis belongs as LaTeX.
3. Write Proposition 3 out in full: it is the new negative result, and its limits are argued here only at the level of Chernoff and the CLT.
4. Decide whether to tell Cassel & Rosenberg about §7.1. It is a constant-level error with a two-line fix.

## Sources

- [[cassel2026Quantile]]: Lemma 1, Corollary 2, the proof in §A.1, Lemmas 13 and 14 with the corollary. Read from the PDF on 2026-09-22; §A.1 and Lemma 14 were checked on rendered pages, since the text extraction garbles the radicals.
- [[contextual-bandits-offline]] §2; [[contextual-bandits-offline-value-based]] Lemma 2.14, Theorem 4.2 and its Chernoff step, Proposition 7.1, and Theorem 4.5 with its missing-mass term.
- Feige (STOC 2004), Theorem 1. Beygelzimer, Langford, Li, Reyzin & Schapire (AISTATS 2011; arXiv:1002.4058), Theorem 1. Boucheron, Lugosi & Massart (2013), Theorem 2.10.
- Numerical values (the $`\mathrm{kl}`$ constants, $`\Phi^{-1}`$ values, the floor at finite $`B`$) are evaluations of closed-form expressions. The simulations run while drafting were sanity checks only, and nothing above depends on them.
