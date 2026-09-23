---
date: 2026-09-22
question: "Can the Quantile of Means pessimism analysis be simplified by assuming sigma^2-sub-Gaussian rewards, the standard bandit setting? Propose the algorithm and write a Theorem 4.2-style analysis step by step, verifying every step."
---

# Quantile of Means as Value-Based Pessimism: the Sub-Gaussian Case

**Scope.** The simplified case of [[2026-09-22-qom-value-based-pessimism]] (below, *the general page*): the same tabular value-based route of [[contextual-bandits-offline-value-based]] §4 and the same estimator, with the reward noise assumed $`\sigma^2`$-sub-Gaussian with one common $`\sigma`$, as in the standard stochastic bandit of Lattimore & Szepesvári (2020, §5.3). Rewards may be signed and unbounded, and $`S=1`$ is the $`K`$-armed bandit. Throughout, every pair is assumed logged at least $`B`$ times, assumption (C) of §1; it removes the empty-batch convention and the gap terms from the statements, and §7 records what it costs. The page reads on its own; where a step is identical to the general page it says so instead of repeating the proof. Every black box is cited and its hypotheses checked at the point of use (§8); nothing rests on simulation.

## Short answer

- **Sub-Gaussianity is half of what the estimator needs.** A tail bound controls how far a batch mean can fall *below* its mean, which gives the width (Lemma 4). It says nothing about how *often* a batch mean falls below its mean at all, which is what makes the low quantile pessimistic (Lemma 3). Proposition 3 gives a $`\sigma^2`$-sub-Gaussian law under which the estimate exceeds the mean by $`\sigma/8`$ with probability at least $`1/2`$, for every $`B`$, and an instance on which no bound of Theorem 4.2's form can hold. The general page got the missing half from Feige's theorem, non-negativity and the $`+1`$ shrink. Here it is an assumption, (U): a batch mean is at most its mean with probability at least $`1/2`$. Symmetric noise gives it at once, so the Gaussian bandit is covered, with unknown and cell-dependent variances.
- **Guarantee in the form of Theorem 4.2.** Under (SG), (U) and (C), conditional on the design, with probability at least $`1-\delta`$ (Theorem 1),

```math
\Delta(\hat\pi)\ \le\ 4\sigma\,\mathbb E_{x\sim\nu}\Big[\sqrt{\frac{\ln 2}{\lfloor N^{\ast}(x)/B\rfloor}}\Big],\qquad B=\big\lceil 3.2\ln(2SK/\delta)\big\rceil,
```

  and under a count condition (Theorem 2) $`\Delta(\hat\pi)\le 8\sigma\sqrt{\ln 2\cdot S\bar C^{\ast}B/T}`$, with $`\bar C^{\ast}=\sum_x\nu(x)/\mu_x`$ the average single-policy coverage of Theorem 4.2. Nothing in the proofs uses one common $`\sigma`$: with cell-dependent proxies $`\sigma(x,a)`$ the same bounds hold with $`\sigma_{\ast}(x)`$ in place of $`\sigma`$, and $`\sigma^2\bar C^{\ast}`$ becomes a variance-weighted coverage (Remark after Theorem 2).
- **What it buys over the general page.** $`B`$ is $`3.2\ln(2SK/\delta)`$ instead of $`26\ln(2SK/\delta)`$; the estimate typically sits $`1.46`$ exact-$`\sigma`$ LCB widths below the mean instead of $`7.8`$ Bernstein widths (Proposition 2); the proven constant is within $`26`$ percent of the floor, and at the floor for Gaussian noise; there is no $`B/N`$ term, so noiseless optimal actions are recovered exactly (Corollary 2); signed and unbounded rewards need no shift; and the algorithm never sees $`\sigma`$. Under symmetry the undershoot is two-sided, so the same estimator also gives an optimistic value (§2).
- **What it costs.** (U). Without it (Theorem 3) the bound carries a term at the action the rule *chooses*, which only uniform coverage controls, the kind of bound the greedy rule has. The $`(C^{\ast}-1)`$ refinement is still lost (Proposition 4).
- **Standing simplification (C).** Every pair is logged at least $`B`$ times, so no batch is empty. It makes the algorithm branch-free and the statements clean, but it is a uniform-coverage condition on the design: it excludes expert data and the missing-mass term, and the proofs need it only at $`\pi^{\ast}`$'s actions; the one-line convention that replaces it is in §7.

## 1. Setting and notation

As on the general page and [[contextual-bandits-offline]] §2: tabular with $`S`$ contexts and $`K`$ actions, context law $`\nu`$, behavior policy $`\mu`$, mean reward $`q^{\ast}(x,a)`$, a deterministic optimal policy $`\pi^{\ast}`$, suboptimality $`\Delta(\hat\pi):=J(\pi^{\ast})-J(\hat\pi)`$, $`T`$ i.i.d. triples, the **design** $`(x_1,a_1),\dots,(x_T,a_T)`$ with counts $`N(x,a)`$, and **Fact 0**: conditional on the design, the rewards at a pair are $`N(x,a)`$ i.i.d. draws from $`\rho(\cdot\mid x,a)`$. Shorthand as before: $`N^{\ast}(x):=N(x,\pi^{\ast}(x))`$, $`\mu_x:=\mu(\pi^{\ast}(x)\mid x)`$, $`C^{\ast}=\max_{x:\nu(x)>0}1/\mu_x`$ and $`\bar C^{\ast}:=\sum_x\nu(x)/\mu_x`$.

New here:

- **(SG) $`\sigma`$-sub-Gaussian noise.** There is one $`\sigma>0`$ such that at **every** pair the noise $`\eta:=r-q^{\ast}(x,a)`$ satisfies $`\mathbb E[e^{\lambda\eta}\mid x,a]\le e^{\lambda^2\sigma^2/2}`$ for all $`\lambda\in\mathbb R`$: Definition 5.2 of Lattimore & Szepesvári, applied to the centered reward, with the homoscedastic proxy of the standard stochastic bandit. Two things to keep in mind. $`\sigma^2`$ is a *variance proxy*, an upper bound: the variance at a pair is at most $`\sigma^2`$ (their Lemma 5.4(a)), with equality for Gaussian noise, and a pair may be far quieter, a deterministic reward being $`0`$-sub-Gaussian and hence $`\sigma`$-sub-Gaussian for every $`\sigma`$. And a common proxy is an assumption of convenience only: no proof below uses it, so the cell-dependent forms come free (Remark after Theorem 2).
- **(C) Counts.** $`N(x,a)\ge B`$ for every pair $`(x,a)`$ with $`\nu(x)>0`$: a condition on the design, under which every batch of every pair is non-empty. The strict form $`N(x,a) > B`$ changes nothing below. By Lemma 6 and a union bound over the $`SK`$ pairs, $`\Pr[\text{(C) fails}]\le\varepsilon_C:=\sum_{x,a}2B/(T\nu(x)\mu(a\mid x))`$, a uniform-coverage quantity; §7 discusses the price. Corollary 3 is the one statement below that (C) excludes rather than simplifies, and it is stated without it.
- **Gaps.** $`\Delta_x:=q^{\ast}(x,\pi^{\ast}(x))-\min_aq^{\ast}(x,a)`$ and $`\Delta_{\max}:=\max_x\Delta_x`$, used where a context must be written off: the trivial bound on the regret at $`x`$. With means in $`[0,1]`$, $`\Delta_{\max}\le1`$, which is Theorem 4.2's $`1`$. Under (C) no such context arises in Theorem 1.
- **Coverage.** Under (SG) the coverage quantity is Theorem 4.2's own $`\bar C^{\ast}=\sum_x\nu(x)/\mu_x\le C^{\ast}`$; with cell-dependent proxies it refines to $`V^{\ast}_{\sigma}:=\sum_x\nu(x)\sigma^2(x,\pi^{\ast}(x))/\mu_x\le\sigma^2\bar C^{\ast}`$, the general page's $`V^{\ast}`$ for Gaussian noise.

## 2. What sub-Gaussianity gives, and what it does not

The estimator rests on two facts about the mean $`\bar X`$ of $`m`$ i.i.d. rewards with mean $`q`$:

- **Width (concentration).** $`\Pr[\bar X\le q-t]\le e^{-mt^2/(2\sigma^2)}`$, and the same for the upper tail. This is what (SG) gives (Lemma 1), in both directions.
- **Undershoot (anti-concentration).** $`\Pr[\bar X\le q]\ge p_0`$ for a constant $`p_0 > \alpha`$. Nothing in (SG) implies this: a $`\sigma^2`$-sub-Gaussian law may put mass $`15/16`$ just above its mean and the rest far below it, and then a batch of one reward lies above the mean fifteen times in sixteen (Proposition 3).

On the general page the undershoot came from Feige's theorem, for non-negative rewards and the shrunk estimate $`\sum X_i/(m+1)`$, with $`p_0=1/13`$. Here it is assumed:

**(U) undershoot.** For every pair $`(x,a)\in\mathcal X\times\mathcal A`$ with $`\nu(x)>0`$ and every integer $`m\ge1`$: if $`r_1,\dots,r_m`$ are i.i.d. with law $`\rho(\cdot\mid x,a)`$, then

```math
\Pr\Big[\frac1m\sum_{i=1}^{m}r_i\ \le\ q^{\ast}(x,a)\Big]\ \ge\ \frac12.
```

Equivalently, in terms of the centered noise $`\eta_i:=r_i-q^{\ast}(x,a)`$, $`\Pr\big[\sum_{i=1}^{m}\eta_i\le0\big]\ge1/2`$ for every $`m\ge1`$: the mean of the law is at or above a median of every batch mean drawn from it. Only $`m\ge1`$ matters, and $`m`$ ranges over the batch sizes the algorithm can produce.

**Fact 1 (symmetric noise satisfies (U)).** If the noise law at $`(x,a)`$ is symmetric about $`0`$, that is $`\eta`$ and $`-\eta`$ have the same distribution under $`\rho(\cdot\mid x,a)`$, then (U) holds at $`(x,a)`$.

*Proof.* Fix $`m\ge1`$ and put $`Z:=\sum_{i=1}^{m}\eta_i`$. The $`\eta_i`$ are i.i.d. and each is distributed as $`-\eta_i`$, so $`Z`$ and $`-Z`$ have the same law, whence $`\Pr[Z\le0]=\Pr[-Z\le0]=\Pr[Z\ge0]`$. Since $`\Pr[Z\le0]+\Pr[Z\ge0]=1+\Pr[Z=0]\ge1`$, each is at least $`1/2`$. $`\square`$

Gaussian, uniform, Rademacher and every symmetric bounded law qualify, as does a deterministic reward ($`Z=0`$ almost surely). Skewed laws need not: Bernoulli$`(0.4)`$ rewards in a batch of two have mean at most $`0.4`$ only when both are $`0`$, probability $`0.36`$. The general page's shrink and Feige's theorem are the device for such laws, at the price $`p_0=1/13`$. The Berry–Esseen theorem gives (U) up to a correction of order $`\mathbb E\lvert\eta\rvert^3/(\mathrm{Var}(\eta)^{3/2}\sqrt m)`$, so for large batches any law of bounded skewness is close to symmetric; but the small batches are exactly the poorly covered cells where pessimism matters, so this does not replace (U).

Two consequences of symmetry. First, the direction is the general page's: an estimate that undershoots a reward is pessimistic, and no sign convention or shift is needed even for signed rewards. Second, unlike Feige's inequality, (U) under symmetry is two-sided: the $`k`$-th *largest* batch mean is at least $`q`$ with the same probability, so the same estimator gives an optimistic value for online use. The general page's §2 said under-estimation was all the undershoot fact provided; with symmetric noise both directions are available.

## 3. Algorithm: QoM-LCB

In the form the offline contextual-bandit literature uses for LCB rules: the rule (LCB) of the value-based page, or Rashidinejad et al.'s Algorithm 1, with steps 2–4 in place of the penalty.

**Algorithm 1 (QoM-LCB).**

*Input:* the dataset $`\mathcal D=\lbrace(x_t,a_t,r_t)\rbrace_{t=1}^{T}`$; the number of batches $`B`$; the quantile level $`\alpha`$. Set $`k:=\lceil\alpha B\rceil`$.

*For every pair* $`(x,a)\in\mathcal X\times\mathcal A`$:

1. **Count.** Let

```math
\mathcal T(x,a):=\lbrace t\in\lbrace1,\dots,T\rbrace\ :\ (x_t,a_t)=(x,a)\rbrace,\qquad N(x,a):=\lvert\mathcal T(x,a)\rvert,
```

and write $`\hat r_1,\dots,\hat r_{N(x,a)}`$ for the rewards $`(r_t)_{t\in\mathcal T(x,a)}`$, indexed in increasing order of $`t`$. By (C), $`N(x,a)\ge B`$.
2. **Split.** Partition them uniformly at random into $`B`$ batches $`\mathcal B^{1}(x,a),\dots,\mathcal B^{B}(x,a)`$ of sizes as equal as possible.
3. **Batch means.** For $`b=1,\dots,B`$,

```math
\hat q^{b}(x,a):=\frac{1}{\lvert\mathcal B^{b}(x,a)\rvert}\sum_{\hat r_i\in\mathcal B^{b}(x,a)}\hat r_i.
```

4. **Pessimistic value.** Let $`\hat q_{(1)}(x,a)\le\hat q_{(2)}(x,a)\le\dots\le\hat q_{(B)}(x,a)`$ denote the batch means $`\hat q^{1}(x,a),\dots,\hat q^{B}(x,a)`$ arranged in non-decreasing order, and set

```math
\underline q(x,a)\ :=\ \hat q_{(k)}(x,a)\ =\ \min\Big\lbrace u\in\mathbb R\ :\ \big\lvert\lbrace b\in\lbrace1,\dots,B\rbrace\ :\ \hat q^{b}(x,a)\le u\rbrace\big\rvert\ \ge\ k\Big\rbrace,
```

the $`k`$-th order statistic of the $`B`$ batch means, that is their empirical $`\alpha`$-quantile.

*Output:* $`\hat\pi(x)\in\arg\max_a\underline q(x,a)`$ for every context $`x`$.

Against the tabular LCB of Theorem 4.2, $`\hat\pi(x)\in\arg\max_a\hat q(x,a)-\Gamma(x,a)`$ with the cell mean $`\hat q`$ and the penalty $`\Gamma(x,a)=\sigma\sqrt{2\ln(2SK/\delta)/N(x,a)}`$, steps 2–4 replace the penalty and nothing else changes. $`\sigma`$ is never used.

Set $`\alpha=1/8`$ and $`B=\lceil 3.2\ln(2SK/\delta)\rceil`$. Against the general page's algorithm, two changes. The batch means are plain averages, not $`\sum r/(n_b+1)`$: the shrink pulls toward $`0`$, and $`0`$ is no longer a lower bound on anything. And $`\alpha,B`$ are set by $`\mathrm{kl}(1/8,1/2)`$ rather than by Feige's $`1/13`$. The general page's "empty batch gives $`0`$" has no counterpart, because (C) rules out empty batches; without (C), a pair with fewer than $`B`$ rewards would need the convention $`\underline q(x,a):=-\infty`$, the analogue of Rashidinejad's value $`-1`$ for an unvisited pair (§7).

Three properties of the split are used below, and both the random partition of step 2 and the general page's round-robin by time rank have all three:

- **Independent of the reward values.** Lemma 3 and Lemma 4 need the batches to be fixed before the rewards are looked at, so that the per-batch events are independent. A partition drawn from external randomness qualifies, as does one read off the design; a split that used the rewards, say low values first, would not.
- **Balanced.** $`\lvert\mathcal B^{b}\rvert\ge\lfloor N/B\rfloor`$ for every $`b`$, which is the batch size in Lemma 4. Assigning each reward to a uniformly random batch independently would *not* do: batch sizes would be multinomial, with an empty batch possible even under (C), and Lemma 4 would have to pay for the smallest one.
- **Non-empty**, by (C) together with balance, so that every batch mean is defined and Lemma 3's (U) applies to it.

Under (A1) the two splits also have the same law: the rewards at a pair are i.i.d. (Fact 0), hence exchangeable, so dealing them by time rank and partitioning them at random give the same joint distribution of $`(\hat q^{1},\dots,\hat q^{B})`$. Randomizing matters when that exchangeability fails — adaptively collected data, as in Cassel & Rosenberg's online batches — and is stated here because it is the form that keeps the analysis valid in that case. The level $`1/8`$ is within $`2`$ percent in width of the numerically best level, about $`1/6`$ (§9).

## 4. Assumptions

- **(A1) Data.** I.i.d. triples, as in [[contextual-bandits-offline]] §2.1.
- **(SG) $`\sigma`$-sub-Gaussian noise**, §1, one common proxy. Used for the widths (Lemmas 4 and 5). $`\sigma`$ is not an input of the algorithm, and the proofs never use that the proxy is common (Remark after Theorem 2).
- **(U) Undershoot**, §2. Used for validity (Lemma 3). Implied by symmetric noise.
- **(C) Counts**, §1. Every pair is logged at least $`B`$ times. A simplification of the algorithm's statement; §7 records its price and the convention that replaces it.
- **Tabular.** $`S,K`$ finite.
- **For Theorem 2 only.** $`C^{\ast}<\infty`$, and the count condition $`T\nu(x)\mu_x\ge8\ln(S/\delta')`$ for every $`x`$ with $`\nu(x)>0`$, as in Theorem 4.2.
- **Not assumed.** The proxy $`\sigma`$, or any bound on it, for running the algorithm; a reward range or sign; the variances; the propensities $`\mu`$; the value of $`C^{\ast}`$. Theorem 3 drops (U) and shows what remains.

## 5. Main results

The results are stated first and the technical facts they call on are proved in §6. Three of those are genuinely per-cell statements, applied at each of the $`SK`$ cells under a union bound, and so are stated separately: **Lemma 3** (the estimate is at most the cell's mean), **Lemma 4** (it is not much less) and **Lemma 5** (it is not much more). The rest of each argument is in the proof.

Throughout, $`\underline q`$ denotes the values Algorithm 1 computes, $`\hat\pi`$ the policy it returns, and

```math
w(x,a)\ :=\ 4\sigma\sqrt{\frac{\ln2}{\lfloor N(x,a)/B\rfloor}}
```

the per-cell width of Lemma 4, finite at every cell by (C).

### 5.1 The guarantee

**Theorem 1 (QoM-LCB, conditional on the design).** Let $`\mathcal X`$ and $`\mathcal A`$ be finite, $`S=\lvert\mathcal X\rvert`$ and $`K=\lvert\mathcal A\rvert`$. Let the data be $`T`$ i.i.d. triples $`(x_t,a_t,r_t)`$ with $`x_t\sim\nu`$, $`a_t\sim\mu(\cdot\mid x_t)`$ and $`r_t\sim\rho(\cdot\mid x_t,a_t)`$ of mean $`q^{\ast}(x_t,a_t)`$ (A1), and let $`N(x,a)`$ count the rounds with $`(x_t,a_t)=(x,a)`$. Assume:

- **(SG)** for one $`\sigma>0`$ and every pair, the noise $`r-q^{\ast}(x,a)`$ is $`\sigma^2`$-sub-Gaussian, that is $`\mathbb E[e^{\lambda(r-q^{\ast}(x,a))}\mid x,a]\le e^{\lambda^2\sigma^2/2}`$ for all $`\lambda\in\mathbb R`$;
- **(U)** for every pair $`(x,a)`$ with $`\nu(x)>0`$ and every integer $`m\ge1`$, i.i.d. draws $`r_1,\dots,r_m\sim\rho(\cdot\mid x,a)`$ satisfy $`\Pr\big[\tfrac1m\sum_{i\le m}r_i\le q^{\ast}(x,a)\big]\ge\tfrac12`$ — which holds whenever the noise is symmetric about $`0`$ (Fact 1);
- **(C)** $`N(x,a)\ge B`$ for every $`(x,a)`$ with $`\nu(x)>0`$.

Run Algorithm 1 with $`\alpha=1/8`$, so $`k=\lceil B/8\rceil`$, and with $`B\ge3.2\ln(2SK/\delta)`$ batches, $`\delta\in(0,1)`$. Then, conditional on the design $`(x_1,a_1),\dots,(x_T,a_T)`$ and on the split randomness, with probability at least $`1-\delta`$ over the rewards, simultaneously for every deterministic policy $`\pi:\mathcal X\to\mathcal A`$,

```math
J(\pi)-J(\hat\pi)\ \le\ \mathbb E_{x\sim\nu}\big[w(x,\pi(x))\big]\ =\ 4\sigma\,\mathbb E_{x\sim\nu}\Big[\sqrt{\frac{\ln2}{\lfloor N(x,\pi(x))/B\rfloor}}\Big],
```

where $`J(\pi):=\mathbb E_{x\sim\nu}[q^{\ast}(x,\pi(x))]`$. Taking $`\pi=\pi^{\ast}`$, an optimal policy, bounds $`\Delta(\hat\pi):=J(\pi^{\ast})-J(\hat\pi)`$ and gives the display in the short answer.

*Proof.* Condition on the design and on the split randomness, and hold both fixed for the whole argument; $`\hat\pi`$ is then a function of the rewards alone, and the expectations below are over $`x\sim\nu`$ only. Under this conditioning the rewards at a cell are $`N(x,a)`$ i.i.d. draws from $`\rho(\cdot\mid x,a)`$: they are i.i.d. given the design by Fact 0, and the split is drawn independently of $`\mathcal D`$, so conditioning on it in addition changes nothing. Each cell's batches are therefore fixed disjoint sets of i.i.d. rewards, which is what Lemmas 3 and 4 require, and by (C) each holds at least one reward.

**Step 1: the two one-sided events.** Let

```math
E_p:=\big\lbrace\underline q(x,a)\le q^{\ast}(x,a)\ \text{ for all }(x,a)\big\rbrace,\qquad
E_w:=\big\lbrace\underline q(x,a)\ge q^{\ast}(x,a)-w(x,a)\ \text{ for all }(x,a)\big\rbrace.
```

At a single cell, Lemma 3 bounds the failure probability of the first by $`e^{-5B/16}`$ and Lemma 4 that of the second by the same quantity. The choice $`\alpha=1/8`$ enters through $`\mathrm{kl}(1/8,1/2)\ge5/16`$, and the constant $`3.2=16/5`$ is chosen exactly so that $`B\ge3.2\ln(2SK/\delta)`$ is $`5B/16\ge\ln(2SK/\delta)`$, i.e. $`e^{-5B/16}\le\delta/(2SK)`$. A union bound over the $`2SK`$ events, $`SK`$ cells and two per cell, gives $`\Pr[E_p\cap E_w]\ge1-\delta`$. Argue on $`E_p\cap E_w`$ from here.

**Step 2: the regret at one context.** Both $`\pi`$ and $`\hat\pi`$ are deterministic, so

```math
J(\pi)-J(\hat\pi)=\mathbb E_{x\sim\nu}\big[q^{\ast}(x,\pi(x))-q^{\ast}(x,\hat\pi(x))\big].
```

Fix a context $`x`$. Applying $`E_p`$ at the pair $`(x,\hat\pi(x))`$, then the definition of $`\hat\pi`$ as a maximizer of $`\underline q`$ at $`x`$, then $`E_w`$ at the pair $`(x,\pi(x))`$,

```math
q^{\ast}(x,\hat\pi(x))\ \ge\ \underline q(x,\hat\pi(x))\ \ge\ \underline q(x,\pi(x))\ \ge\ q^{\ast}(x,\pi(x))-w(x,\pi(x)),
```

so that $`q^{\ast}(x,\pi(x))-q^{\ast}(x,\hat\pi(x))\le w(x,\pi(x))`$.

**Step 3: average.** Averaging the last inequality over $`x\sim\nu`$ gives the claim, for every deterministic $`\pi`$ at once, since the event of Step 1 does not depend on $`\pi`$. $`\square`$

**Explanation.** The proof uses each of the two events once, and at *different* actions. Validity ($`E_p`$) is used at the action the rule chose, where it says that a selected action's true value is at least the score that got it selected: an action cannot be picked by being overestimated. The width ($`E_w`$) is used at the comparator's action, where it says the comparator's score was not much below its true value. In between sits the only property of $`\hat\pi`$ that is used, that its score is the largest.

Two consequences. First, the bound involves the counts at $`\pi^{\ast}`$'s actions only: a badly covered action costs nothing unless $`\pi^{\ast}`$ wants to play it, which is the single-policy coverage that separates pessimism from the greedy rule. Second, the width is charged **once**, where the LCB of Theorem 4.2 charges its penalty twice — once for the comparator and once to undo the penalty it subtracted at the chosen action. QoM-LCB subtracts nothing, so there is nothing to undo; what replaces the penalty is the one-sided validity $`E_p`$, and (U) is exactly the assumption that buys it. Drop (U) and the first inequality of the chain fails by an amount that must then be paid at the chosen action, which is Theorem 3.

### 5.2 The coverage form and its consequences

**Theorem 2 (coverage form).** In the setting of Theorem 1, and with its assumptions, suppose in addition that $`C^{\ast}<\infty`$ and that $`T\nu(x)\mu_x\ge8\ln(S/\delta')`$ for every $`x`$ with $`\nu(x)>0`$, the count condition of Theorem 4.2, where $`\mu_x:=\mu(\pi^{\ast}(x)\mid x)`$. Then with probability at least $`1-\delta-\delta'`$,

```math
\Delta(\hat\pi)\ \le\ 8\sigma\sqrt{\frac{\ln2\cdot S\bar C^{\ast}B}{T}}\ \approx\ 6.66\,\sigma\sqrt{\frac{S\bar C^{\ast}B}{T}},\qquad \bar C^{\ast}=\sum_x\frac{\nu(x)}{\mu_x}\le C^{\ast}.
```

*Proof.* Theorem 1 is conditional on the design, and its event has probability at least $`1-\delta`$ for every design, so by the tower property it has probability at least $`1-\delta`$ unconditionally as well. It remains to replace the random counts $`N^{\ast}(x):=N(x,\pi^{\ast}(x))`$ by their scale $`T\nu(x)\mu_x`$, which is a statement about the design alone.

**Step 1: the counts.** A round contributes to $`N^{\ast}(x)`$ exactly when $`x_t=x`$ and $`a_t=\pi^{\ast}(x)`$, which are independent across rounds and have probability $`\nu(x)\mu_x`$, so $`N^{\ast}(x)\sim\mathrm{Bin}(T,\nu(x)\mu_x)`$. The multiplicative Chernoff bound $`\Pr[\mathrm{Bin}(T,p)\le Tp/2]\le e^{-Tp/8}`$ at $`p=\nu(x)\mu_x`$ is at most $`\delta'/S`$ by the count condition, so a union bound over the at most $`S`$ contexts with $`\nu(x)>0`$ gives

```math
N^{\ast}(x)\ \ge\ \tfrac12T\nu(x)\mu_x\qquad\text{for all such }x,
```

with probability at least $`1-\delta'`$ over the design. Argue on this event and on Theorem 1's.

**Step 2: the floor.** By (C), $`N^{\ast}(x)/B\ge1`$, and $`\lfloor y\rfloor\ge y/2`$ for $`y\ge1`$, so $`\lfloor N^{\ast}(x)/B\rfloor\ge N^{\ast}(x)/(2B)\ge T\nu(x)\mu_x/(4B)`$. This is the only place the floor is paid, and it costs a factor $`\sqrt2`$ in the final constant.

**Step 3: substitute and average.** Theorem 1 at $`\pi=\pi^{\ast}`$, with Step 2 in the denominator, gives

```math
\Delta(\hat\pi)\ \le\ \sum_x\nu(x)\cdot4\sigma\sqrt{\frac{4\ln2\cdot B}{T\nu(x)\mu_x}}\ =\ 8\sigma\sqrt{\frac{\ln2\cdot B}{T}}\sum_x\sqrt{\frac{\nu(x)}{\mu_x}}.
```

**Step 4: Cauchy–Schwarz.** Over the at most $`S`$ contexts, $`\sum_x\sqrt{\nu(x)/\mu_x}\le\sqrt{S\sum_x\nu(x)/\mu_x}=\sqrt{S\bar C^{\ast}}`$, exactly as in Theorem 4.2, which gives the claim. $`\square`$

**Explanation.** Theorem 2 is a consequence of Theorem 1 and a Chernoff bound on the design, not an independent result; the value-based page states the two together as the one Theorem 4.2, its second display being this computation. Nothing about the estimator enters after Step 1: the same four steps take any per-context width of the form $`c/\sqrt{N^{\ast}(x)}`$ to $`c\sqrt{S\bar C^{\ast}/T}`$, which is why the comparison with Theorem 4.2 reduces to comparing constants (Corollary 1).

*Remark (dropping the count condition).* Contexts with $`T\nu(x)\mu_x < M:=8\ln(S/\delta')`$ have $`\nu(x) < MC^{\ast}/T`$ and cost at most $`\Delta_x`$ each, so at most $`\Delta_{\max}SC^{\ast}M/T`$ in total, and Theorem 2 applies to the rest: Rashidinejad's missing-mass device, as on the general page. This is a statement about $`\nu`$, not about the counts, so it survives under (C).

**Corollary 1 (the constant against Theorem 4.2).** Substituting $`B=3.2\ln(2SK/\delta)`$ into Theorem 2, the ceiling ignored,

```math
\Delta(\hat\pi)\ \le\ 11.9\,\sigma\sqrt{\frac{S\bar C^{\ast}\ln(2SK/\delta)}{T}},
```

against Theorem 4.2's $`2\sqrt{S\bar C^{\ast}\ln(2SK/\delta)/T}`$ for rewards in $`[0,1]`$. The two are now the same expression in the same quantities, so the comparison is exact: with $`\sigma\le1/2`$, which Hoeffding's lemma gives for rewards in $`[0,1]`$, the constant is $`5.96`$ against $`2`$, a factor $`2.98`$ — where the general page's Corollary 1 had $`7.2`$. The rate is Theorem 4.2's $`\tilde O(\sqrt{S\bar C^{\ast}/T})`$, and $`\bar C^{\ast}\le C^{\ast}`$ gives the $`C^{\ast}`$ form. Theorem 4.2 needs the range $`[0,1]`$ and QoM-LCB does not, so for unbounded noise the comparison is with the exact-$`\sigma`$ Hoeffding LCB, whose constant in this form is $`2\sqrt2\,\sigma=2.83\sigma`$: a factor $`4.2`$. Of that factor, $`\sqrt2`$ is Step 2's floor and the rest is the quantile (Proposition 2).

*Remark (cell-dependent proxies; the variance-adaptive forms).* No step above uses one common $`\sigma`$. Lemma 4 is proved for a single cell and needs only a valid proxy **for that cell**, so if $`\sigma(x,a)^2`$ is any family of valid proxies — for instance the smallest one at each pair — then Theorem 1 holds with $`w(x,a)=4\sigma(x,a)\sqrt{\ln2/\lfloor N(x,a)/B\rfloor}`$, and Theorem 2 with $`\sigma\sqrt{\bar C^{\ast}}`$ replaced by $`\sqrt{V^{\ast}_{\sigma}}`$, $`V^{\ast}_{\sigma}=\sum_x\nu(x)\sigma^2(x,\pi^{\ast}(x))/\mu_x`$, the only change being that $`\sigma`$ stays inside the sum until Step 4. This is the substantive gain over an LCB, and it is worth separating from the constants: the bound then depends on the noise at $`\pi^{\ast}`$'s actions only, a loud action that $`\pi^{\ast}`$ never takes costing nothing, and the algorithm still needs no proxy as an input, where the LCB penalty needs one and pays the largest proxy over the pairs unless it is given the family. Under (SG) as stated this refinement is invisible, since $`V^{\ast}_{\sigma}=\sigma^2\bar C^{\ast}`$ exactly. Corollary 2 is its extreme case and is stated in those terms.

**Corollary 2 (noiseless optimal actions).** Suppose the reward at $`\pi^{\ast}(x)`$ is deterministic for every $`x`$, that is $`\sigma(x,\pi^{\ast}(x))=0`$ is a valid proxy there, the other actions being arbitrary $`\sigma`$-sub-Gaussian. Then on the validity event $`E_p`$ alone, which has probability at least $`1-\delta/2`$, $`\Delta(\hat\pi)=0`$.

*Proof.* A $`0`$-sub-Gaussian variable has variance $`0`$ (Lattimore & Szepesvári, Lemma 5.4(a)), so the rewards at $`\pi^{\ast}(x)`$ are almost surely equal to $`q^{\ast}(x,\pi^{\ast}(x))`$. By (C) every batch at that pair is non-empty, so every batch mean there equals $`q^{\ast}(x,\pi^{\ast}(x))`$, and so does their $`k`$-th order statistic $`\underline q(x,\pi^{\ast}(x))`$. Now run Step 2 of Theorem 1's proof with $`\pi=\pi^{\ast}`$, using only $`E_p`$: $`q^{\ast}(x,\hat\pi(x))\ge\underline q(x,\hat\pi(x))\ge\underline q(x,\pi^{\ast}(x))=q^{\ast}(x,\pi^{\ast}(x))`$, so $`\hat\pi(x)`$ is optimal at every $`x`$. $`\square`$

The general page's Corollary 2 had $`\tfrac{22}3SC^{\ast}B/T`$ from the shrink and Bernstein's second-order term, and without (C) a remainder $`2\Delta_{\max}SC^{\ast}B/T`$ would appear here for the contexts with $`N^{\ast}(x) < B`$ (§7). Under (C) the recovery is exact. Note that the conclusion is about the algorithm, not about the bound of Theorem 1, which under a common $`\sigma`$ is strictly positive here: QoM-LCB reads the quiet cell off the data without being told it is quiet.

**Corollary 3 (expert data; not under (C)).** Let $`C^{\ast}=1`$ and run the algorithm with the convention $`\underline q(x,a):=-\infty`$ where $`N(x,a) < B`$. Then with no failure probability,

```math
\mathbb E_{\mathcal D}[\Delta(\hat\pi)]\ \le\ \sum_x\nu(x)\Delta_x\Pr\big[\mathrm{Bin}(T,\nu(x)) < B\big]\ \le\ 2\Delta_{\max}SB/T.
```

*Proof.* $`C^{\ast}=1`$ forces $`\mu_x=1`$ at every context with $`\nu(x)>0`$, so no action other than $`\pi^{\ast}(x)`$ is ever logged at $`x`$ and every other action has $`\underline q=-\infty`$. If $`N^{\ast}(x)\ge B`$ then $`\pi^{\ast}(x)`$ carries the only finite score at $`x`$ and is selected, costing nothing; otherwise the regret at $`x`$ is at most $`\Delta_x`$. Since $`\mu_x=1`$, $`N^{\ast}(x)\sim\mathrm{Bin}(T,\nu(x))`$, which gives the first inequality, and Lemma 6 bounds each $`\Pr[N^{\ast}(x) < B]`$ by $`2B/(T\nu(x))`$, whence $`\sum_x\nu(x)\Delta_x\cdot2B/(T\nu(x))\le2\Delta_{\max}SB/T`$. $`\square`$

(C) is stated without it because expert data is exactly what (C) excludes: with $`\mu_x=1`$ the other actions are never logged, so $`N(x,a)=0 < B`$ and (C) fails at every context. This is the sharpest form of the price in §7. The general page has $`S/(eT)`$ here; with unbounded rewards an action needs $`B`$ rewards before it has a finite score, where "empty batch gives $`0`$" made a single reward enough.

**Corollary 4 (K-armed bandit).** With $`S=1`$, arm counts $`N(a)\sim\mathrm{Bin}(T,\mu(a))`$ satisfying (C), and $`B\ge3.2\ln(2K/\delta)`$: conditional on the counts and the split, with probability at least $`1-\delta`$, $`\Delta(\hat\pi)\le4\sigma\sqrt{\ln2/\lfloor N(a^{\ast})/B\rfloor}`$; and if $`T\mu(a^{\ast})\ge8\ln(1/\delta')`$, then with probability at least $`1-\delta-\delta'`$, $`\Delta(\hat\pi)\le8\sigma\sqrt{\ln2\cdot B/(T\mu(a^{\ast}))}`$. Here $`\bar C^{\ast}=1/\mu(a^{\ast})`$, so only the optimal arm's count enters, and with cell-dependent proxies only its noise.

### 5.3 What the assumptions are doing

**Proposition 2 (the width is the estimator's, not the proof's).** Let a cell have Gaussian noise $`\mathcal N(0,s^2)`$, so $`\sigma=s`$ is the smallest proxy, and $`m`$ rewards per batch, $`n=mB`$. Then for every $`\gamma>0`$ and every $`m`$,

```math
\Pr\Big[q-\underline q\ge\gamma\frac{s}{\sqrt m}\Big]=\Pr\big[\mathrm{Bin}(B,\Phi(-\gamma))\ge k\big].
```

*Proof.* Each batch mean is $`\mathcal N(q,s^2/m)`$, so $`\hat q^{b}\le q-\gamma s/\sqrt m`$ has probability exactly $`\Phi(-\gamma)`$, independently across batches, and the $`k`$-th smallest is at most $`q-\gamma s/\sqrt m`$ iff at least $`k`$ batches are. $`\square`$

For other symmetric laws of variance $`s^2`$ the identity holds as $`m\to\infty`$ by the central limit theorem, as in the general page's Proposition 2. Consequences:

- **(a) Typical gap.** As $`B\to\infty`$ with $`k/B\to1/8`$, the median of $`(q-\underline q)\sqrt m/s`$ tends to $`\lvert\Phi^{-1}(1/8)\rvert=1.150`$. With $`B=3.2\ln(1/\delta')`$, the value at which one cell's two events each fail with probability $`\delta'`$, the estimate therefore typically sits $`1.150\sqrt{3.2}\,s\sqrt{\ln(1/\delta')/n}=2.06\,s\sqrt{\ln(1/\delta')/n}`$ below the mean. The exact-$`\sigma`$ Hoeffding penalty at the same confidence is $`\sqrt2\,\sigma\sqrt{\ln(1/\delta')/n}`$, so the ratio is $`1.46`$. The general page's ratio to Bernstein's penalty was $`7.79`$; the difference is $`p_0=1/2`$ against $`1/13`$, which is what (U) buys. Lemma 4's constant is $`3.33`$ against the typical $`1.150`$, a factor $`2.9`$; against the Hoeffding penalty it is $`4.2`$ for $`n\gg B`$, and $`6.0`$ with the floor loss.
- **(b) Floor on any valid constant.** Suppose $`q-\underline q\le\gamma s/\sqrt m`$ held with probability at least $`1-e^{-5B/16}`$ for all $`B`$, as Lemma 4's bound does. Then $`\Pr[\mathrm{Bin}(B,\Phi(-\gamma))\ge k]\le e^{-5B/16}`$, and with the lower bound $`\Pr[\mathrm{Bin}(B,p)\ge k]\ge e^{-B\cdot\mathrm{kl}(k/B,p)}/(B+1)`$ for $`p\le k/B`$, the limit $`B\to\infty`$ forces $`\Phi(-\gamma) < 1/8`$ and then $`\mathrm{kl}(1/8,\Phi(-\gamma))\ge5/16`$, i.e. $`\Phi(-\gamma)\le0.00415`$ and $`\gamma\ge2.639`$. Lemma 4's $`3.33`$ is within $`26`$ percent of this. For Gaussian noise, Step 1 of Lemma 4 with the exact tail $`\Phi(-t\sqrt m/s)`$ in place of the sub-Gaussian bound gives $`\gamma=\lvert\Phi^{-1}(1/256)\rvert=2.66`$ with Step 2 unchanged: at the floor.

**Theorem 3 (without (U)).** Assume (A1), (SG) and (C) in the tabular model, with $`\alpha=1/8`$ and $`B\ge3.2\ln(2SK/\delta)`$. Conditional on the design and the split, with probability at least $`1-\delta`$, for every $`x`$,

```math
q^{\ast}(x,\pi^{\ast}(x))-q^{\ast}(x,\hat\pi(x))\ \le\ 4\sigma\sqrt{\frac{\ln2}{\lfloor N^{\ast}(x)/B\rfloor}}+\sigma\sqrt{\frac{2\ln2}{\lfloor N(x,\hat\pi(x))/B\rfloor}}.
```

*Proof.* Without (U) the validity event $`E_p`$ is unavailable, and Lemma 5 replaces it: at each cell, with probability at least $`1-e^{-5B/16}`$, $`\underline q(x,a)\le q^{\ast}(x,a)+u(x,a)`$ with $`u(x,a):=\sigma\sqrt{2\ln2/\lfloor N(x,a)/B\rfloor}`$. Let $`E_u`$ be the event that this holds at all $`SK`$ cells and $`E_w`$ be as in Theorem 1; the union bound of Step 1 there applies verbatim, so $`\Pr[E_u\cap E_w]\ge1-\delta`$. Argue on it and fix $`x`$. Splitting the regret through the two scores,

```math
q^{\ast}(x,\pi^{\ast}(x))-q^{\ast}(x,\hat\pi(x))
=\underbrace{\big[q^{\ast}(x,\pi^{\ast}(x))-\underline q(x,\pi^{\ast}(x))\big]}_{\le\,w(x,\pi^{\ast}(x))\ \text{on }E_w}
+\underbrace{\big[\underline q(x,\pi^{\ast}(x))-\underline q(x,\hat\pi(x))\big]}_{\le\,0\ \text{by the definition of }\hat\pi}
+\underbrace{\big[\underline q(x,\hat\pi(x))-q^{\ast}(x,\hat\pi(x))\big]}_{\le\,u(x,\hat\pi(x))\ \text{on }E_u},
```

which is the claim. $`\square`$

**Explanation.** Compare the chains. In Theorem 1 the third bracket is at most $`0`$, because $`E_p`$ says the score never exceeds the truth; here it is only at most a batch width, and that width sits at the action the rule *chose*. Averaging it over $`x`$ therefore needs a count for every action the rule might choose — uniform coverage, which is the greedy rule's requirement (Theorem 5.1 against Theorem 4.2 on the value-based page). So under (SG) alone QoM-LCB is a greedy rule with a slightly biased estimate, not a pessimistic one. The next proposition shows this is not slack in the proof.

**Proposition 3 ((U) cannot be dropped).**

- **(a) The estimator.** Fix $`\sigma>0`$ and $`B\ge1`$. Let $`X=q+\sigma/8`$ with probability $`15/16`$ and $`X=q-15\sigma/8`$ with probability $`1/16`$. Then $`\mathbb EX=q`$, and $`X-q`$ takes values in an interval of length $`2\sigma`$, so $`X`$ is $`\sigma^2`$-sub-Gaussian by Hoeffding's lemma (§8). For a cell with $`n=B`$ rewards each batch holds one reward, and $`\underline q=q+\sigma/8`$ unless at least $`k`$ rewards are low. By Markov's inequality and $`k\ge B/8`$, $`\Pr[\mathrm{Bin}(B,\tfrac1{16})\ge k]\le(B/16)/k\le1/2`$. So $`\Pr[\underline q\ge q+\sigma/8]\ge1/2`$ for every $`B`$, and Lemma 3's conclusion fails.
- **(b) The rule.** Take $`S=1`$ and two actions: $`a^{\ast}`$ with reward identically $`1`$, and $`a^{\circ}`$ with the law of (a) at $`q=1-\sigma/16`$, so the gap is $`\sigma/16`$. Both laws are $`\sigma`$-sub-Gaussian with the one proxy of (SG): the two-point law by Hoeffding's lemma, the constant reward because a deterministic variable is $`0`$-sub-Gaussian and hence $`\sigma`$-sub-Gaussian for every $`\sigma`$. On every design with $`N(a^{\circ})=B`$ and $`N(a^{\ast})\ge B`$: $`\underline q(a^{\ast})=1`$, and with probability at least $`1/2`$, $`\underline q(a^{\circ})=1+\sigma/16 > 1`$, so $`\hat\pi=a^{\circ}`$ and $`\Delta(\hat\pi)=\sigma/16`$. Hence $`\mathbb E[\Delta(\hat\pi)\mid\text{design}]\ge\sigma/32`$ however large $`N(a^{\ast})`$ is, while Theorem 1's bound at $`\pi=\pi^{\ast}=a^{\ast}`$ is $`4\sigma\sqrt{\ln2/\lfloor N(a^{\ast})/B\rfloor}\to0`$ as $`N(a^{\ast})\to\infty`$. So no bound of that form can hold without (U). Theorem 3's extra term at $`a^{\circ}`$ is $`\sigma\sqrt{2\ln2}=1.18\sigma`$, above the realized $`\sigma/16`$, as it must be.

**Proposition 4 (no $`(C^{\ast}-1)`$ adaptivity).** The general page's Proposition 3 holds here verbatim: its instance, Bernoulli$`(1/2)`$ rewards at $`a^{\ast}`$ and the constant $`1/2-\varepsilon`$ at $`a^{\circ}`$, satisfies (SG) with $`\sigma=1/2`$ (Hoeffding's lemma) and (U), both laws being symmetric. The only change is that the decoy's estimate is exactly $`1/2-\varepsilon`$ once $`N(x,a^{\circ})\ge B`$, without the shrinkage term, which simplifies its Step 1 and leaves Step 4 needing only $`N(x,a^{\circ})\ge B`$. So QoM-LCB's expected suboptimality is $`\Omega(\sqrt{SB/T})`$ on instances where Rashidinejad's LCB achieves $`\tilde O(T^{-3/4})`$, for the reason given there: an implicit penalty that vanishes at a noiseless cell cannot keep the rule off a rarely logged decoy.

## 6. Supporting lemmas

Lemmas 1 and 2 are the two external tools, quoted and checked in §8. Lemmas 3, 4 and 5 are the per-cell facts about the estimator, each applied at all $`SK`$ cells under the union bound of Theorem 1, Step 1. Lemma 6 is the small count bound used in Corollary 3 and in §1.

For Lemmas 3 to 5, **fix one cell and condition on the design and on the split randomness**, as in the proof of Theorem 1: write $`n`$ for its count, which is at least $`B`$ by (C), $`q`$ for its mean, and $`\hat q^{1},\dots,\hat q^{B},\underline q`$ for its estimates. Under this conditioning the cell's rewards are $`n`$ i.i.d. draws and its batches $`\mathcal B^{1},\dots,\mathcal B^{B}`$ are fixed disjoint sets of them, with $`\lvert\mathcal B^{b}\rvert\ge m:=\lfloor n/B\rfloor\ge1`$. Only two properties of $`\sigma`$ are used, that it is a valid proxy **for this cell** and that the batch means are independent across $`b`$; the first is what makes the cell-dependent forms of §5.2 free.

**Lemma 1 (sub-Gaussian batch mean).** Let $`X_1,\dots,X_m`$ be i.i.d. with mean $`q`$ and $`\sigma^2`$-sub-Gaussian noise, and $`\bar X`$ their mean. For every $`t\ge0`$,

```math
\Pr[\bar X\le q-t]\le e^{-mt^2/(2\sigma^2)}\qquad\text{and}\qquad\Pr[\bar X\ge q+t]\le e^{-mt^2/(2\sigma^2)}.
```

*Proof.* Corollary 5.5 of Lattimore & Szepesvári, which states both tails: by their Lemma 5.4(b),(c) the centered mean is $`(\sigma/\sqrt m)`$-sub-Gaussian, and their Theorem 5.3 is the tail bound. Hypotheses at the point of use: the batch's rewards are independent under the conditioning above, identically distributed, and their noise has proxy $`\sigma`$ by (SG). $`\square`$

**Lemma 2 (binomial tails).** Let $`\mathrm{kl}(a,p):=a\ln\frac ap+(1-a)\ln\frac{1-a}{1-p}`$ and $`Z\sim\mathrm{Bin}(B,p)`$. Then $`\Pr[Z\le aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\le p`$, and $`\Pr[Z\ge aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\ge p`$ (the Chernoff–Hoeffding bound). For a sum of independent Bernoulli variables whose means are all at least $`p`$ (at most $`p`$), the lower (upper) tail is at most that of $`\mathrm{Bin}(B,p)`$, by the coupling $`Z_b=\mathbb 1\lbrace U_b\le p_b\rbrace`$ with i.i.d. uniform $`U_b`$. Three constants are used below, each with a check by hand:

- (i) $`\mathrm{kl}(1/8,1/2)=\tfrac78\ln\tfrac74-\tfrac14\ln2\ge\tfrac5{16}`$. Since $`\ln\tfrac74=\ln\tfrac32+\ln\tfrac76=2\operatorname{artanh}\tfrac15+2\operatorname{artanh}\tfrac1{13}`$ and every term of the artanh series is positive, $`\ln\tfrac74\ge2(\tfrac15+\tfrac1{375})+2(\tfrac1{13}+\tfrac1{6591})=0.55948`$, so $`\mathrm{kl}\ge\tfrac78(0.55948)-\tfrac14(0.69315)=0.31626\ge0.3125`$. The exact value is $`0.31638`$.
- (ii) $`\mathrm{kl}(1/8,1/256)=\tfrac58\ln2-\tfrac78\ln(1+\tfrac{31}{224})\ge\tfrac5{16}`$. On $`(0,1)`$, $`\ln(1+x)\le x-x^2/2+x^3/3`$ (the alternating series cut after a positive term), which at $`x=31/224`$ gives at most $`0.12971`$; then $`\tfrac58(0.69314)-\tfrac78(0.12971)=0.31972\ge0.3125`$. The exact value is $`0.31980`$.
- (iii) $`\mathrm{kl}(7/8,1/2)=\mathrm{kl}(1/8,1/2)`$, since $`\mathrm{kl}(1-a,1-p)=\mathrm{kl}(a,p)`$.

The three enter as follows: (i) bounds the validity failure in Lemma 3, (ii) the width failure in Lemma 4, and (iii) the upper-deviation failure in Lemma 5. All three are at least $`5/16`$, which is why one $`B`$ serves all of them and why $`3.2=16/5`$ appears in the theorems.

**Lemma 3 (the estimate is at most the cell's mean).** Under (U), with $`\alpha=1/8`$, $`k=\lceil B/8\rceil`$ and any $`B\ge1`$, $`\Pr[\underline q\le q]\ge1-e^{-5B/16}`$.

*Proof.* By step 4 of Algorithm 1, $`\underline q\le q`$ holds exactly when at least $`k`$ of the $`B`$ batch means are at most $`q`$, so put $`Z_b:=\mathbb 1\lbrace\hat q^{b}\le q\rbrace`$ and bound $`\Pr[\sum_bZ_b\le k-1]`$. Batch $`b`$ is the mean of $`\lvert\mathcal B^{b}\rvert\ge1`$ i.i.d. rewards from the cell, so $`\Pr[Z_b=1]\ge1/2`$ by (U) — this is the only place (U) is used, and it is used once per batch. The $`Z_b`$ are independent, being functions of disjoint sets of rewards that the conditioning has fixed. Since $`k-1 < B/8\le B/2`$, Lemma 2 applies with $`a=1/8`$ below $`p=1/2`$, and constant (i) gives

```math
\Pr\Big[\sum_bZ_b\le k-1\Big]\le\Pr\big[\mathrm{Bin}(B,\tfrac12)\le\tfrac B8\big]\le e^{-B\cdot\mathrm{kl}(1/8,1/2)}\le e^{-5B/16}.\qquad\square
```

**Lemma 4 (the estimate is not much less).** Under (SG) and (C), with probability at least $`1-e^{-5B/16}`$,

```math
\underline q\ \ge\ q-4\sigma\sqrt{\frac{\ln2}{\lfloor n/B\rfloor}}.
```

*Proof.* **Step 1 (one batch).** Put $`t:=4\sigma\sqrt{\ln2/m}`$ with $`m=\lfloor n/B\rfloor`$. Batch $`b`$ holds $`\lvert\mathcal B^{b}\rvert\ge m`$ rewards, so $`\lvert\mathcal B^{b}\rvert t^2/(2\sigma^2)\ge8\ln2`$ and the lower tail of Lemma 1 gives $`\Pr[\hat q^{b} < q-t]\le2^{-8}=1/256`$. The batch sizes enter only here, and only through the smallest of them, which is where the balance of the split is needed.

**Step 2 (quantile).** By step 4 of Algorithm 1, $`\underline q < q-t`$ holds exactly when at least $`k`$ batch means are below $`q-t`$. The threshold is fixed by the conditioning, and the corresponding indicators are independent for the reason given in Lemma 3, with means at most $`1/256`$. Since $`k\ge B/8\ge B/256`$, Lemma 2 applies with $`a=1/8`$ above $`p=1/256`$, and constant (ii) gives $`\Pr[\underline q < q-t]\le\Pr[\mathrm{Bin}(B,\tfrac1{256})\ge\tfrac B8]\le e^{-B\cdot\mathrm{kl}(1/8,1/256)}\le e^{-5B/16}`$. $`\square`$

*Verification note.* On the general page, Bernstein's inequality, the $`+1`$ shrink and $`q\le1`$ gave $`4\sqrt{\sigma^2B/n}+\tfrac{11}3B/n`$. Here the tail is sub-Gaussian and there is no shrink, so the second term is gone and $`B`$ enters only through the batch size. The constant $`4\sqrt{\ln2}=3.33`$ is within $`26`$ percent of the floor for this estimator (Proposition 2(b)).

**Lemma 5 (the estimate is not much more, from (SG) alone).** Under (SG) and (C), with probability at least $`1-e^{-5B/16}`$, $`\underline q\le q+\sigma\sqrt{2\ln2/\lfloor n/B\rfloor}`$.

*Proof.* $`\underline q > q+t`$ holds exactly when fewer than $`k`$ batch means are at most $`q+t`$, that is when more than $`B-k`$ of them exceed $`q+t`$. With $`t:=\sigma\sqrt{2\ln2/m}`$, the upper tail of Lemma 1 gives each batch probability at most $`e^{-\ln2}=1/2`$. Since $`k < B/8+1`$, we have $`B-k+1 > 7B/8`$, so Lemma 2 applies with $`a=7/8`$ above $`p=1/2`$, and constant (iii) gives $`\Pr[\underline q > q+t]\le\Pr[\mathrm{Bin}(B,\tfrac12)\ge\tfrac{7B}8]\le e^{-B\cdot\mathrm{kl}(7/8,1/2)}\le e^{-5B/16}`$. $`\square`$

Lemma 5 is the most (SG) says in the direction of validity: without (U) the low quantile can exceed the cell's mean, but only by a batch width. It carries Theorem 3, and Proposition 3 shows the excess is real rather than an artifact of the bound.

**Lemma 6 (counts).** For $`N\sim\mathrm{Bin}(T,p)`$ and $`B\ge2`$, $`\Pr[N < B]\le2B/(Tp)`$.

*Proof.* If $`Tp < 2B`$ the right side exceeds $`1`$ and there is nothing to prove. Otherwise $`B\le Tp/2`$, so $`\Pr[N < B]\le\Pr[N\le Tp/2]\le e^{-Tp/8}`$ by the multiplicative Chernoff bound, and $`e^{-u}\le1/(eu)`$ for $`u>0`$ (Lattimore & Szepesvári, p. 77) with $`u=Tp/8`$ turns this into $`8/(eTp)\le2B/(Tp)`$, since $`8/e < 4\le2B`$. $`\square`$

## 7. What changed from the general page

- **Anti-concentration is assumed, not derived.** (U) replaces non-negativity, the $`+1`$ shrink and Feige's theorem. Symmetric noise gives it with $`p_0=1/2`$; nothing else standard does (Proposition 3, §2).
- **The estimator is the plain batch mean**, which follows from rewards having no sign: the general page's shrink toward $`0`$ is not a shrink toward a lower bound any more.
- **The split is written as a uniformly random balanced partition** rather than round-robin by time rank. Under (A1) the two have the same law, the rewards at a pair being exchangeable; the random form is the one that stays valid when they are not (§3).
- **Bernstein gives way to the sub-Gaussian tail.** The $`B/N`$ term disappears; noiseless optimal actions are recovered exactly (Corollary 2), and Corollary 3 gains a factor $`B`$ against the general page's $`S/(eT)`$ because $`B`$ rewards are needed before any estimate is finite.
- **Constants.** $`B=3.2\ln(2SK/\delta)`$ against $`26\ln(2SK/\delta)`$; a typical gap of $`1.46`$ exact-$`\sigma`$ widths against $`7.8`$ Bernstein widths; a proven constant within $`26`$ percent of the floor, and at the floor for Gaussian noise (Proposition 2).
- **Both directions.** Under symmetry the same estimator gives an optimistic value (§2), which Feige's one-sided inequality could not.
- **What did not change.** The one-sided reduction (Theorem 1, Step 2), the design conditioning (Fact 0), the union bound, the coverage step (Theorem 2, Step 1), and the loss of $`(C^{\ast}-1)`$ adaptivity (Proposition 4). Nothing from Cassel & Rosenberg's Lemma 1(2) is used, so the erratum of the general page's §7.1 plays no role here.

### 7.1 The price of (C), and how to drop it

(C) buys clean statements: no $`\min\lbrace\Delta_x,\cdot\rbrace`$, no $`\pm\infty`$ convention, no $`2B`$ in the count condition, and exact recovery in Corollary 2. It is not free, and it is worth keeping the two apart when writing this up.

- **It is a uniform-coverage condition.** It asks every action at every context to be logged $`B`$ times, so it holds with the probability $`1-\varepsilon_C`$ of §1, which needs $`T\gtrsim2B\sum_{x,a}1/(\nu(x)\mu(a\mid x))`$ — a sum over *all* pairs. Pessimism's point is to need only $`T\gtrsim2BSC^{\ast}`$, a sum over $`\pi^{\ast}`$'s pairs. Assuming (C) therefore assumes away the regime the method exists for, and it is strictly stronger than what the proofs use.
- **It excludes expert data.** At $`C^{\ast}=1`$ the non-optimal actions are never logged, so (C) fails everywhere (Corollary 3).
- **What the proofs actually need is (C\*):** $`N(x,\pi^{\ast}(x))\ge B`$ for every $`x`$ with $`\nu(x)>0`$, only at the comparator's actions. Lemma 4 is applied there and nowhere else; at every other cell only validity is needed, and $`\underline q=-\infty\le q^{\ast}`$ is validity for free. Theorem 2's count condition already forces (C\*) with probability $`1-\delta'`$, so under it Theorem 2, Corollary 1 and Corollary 4 hold verbatim with (C) deleted. Theorem 3 is the exception: its second width sits at $`\hat\pi(x)`$, so it needs (C) or a bound on the chosen action's count, which is the uniform-coverage remark that follows it.
- **The one-line replacement.** Keep the convention $`\underline q(x,a):=-\infty`$ where $`N(x,a) < B`$, the analogue of Rashidinejad's value $`-1`$ at an unvisited pair, and restore $`\min\lbrace\Delta_x,\cdot\rbrace`$ in Theorems 1 and 3, $`\max\lbrace8\ln(S/\delta'),2B\rbrace`$ in Theorem 2, and the remainder $`2\Delta_{\max}SC^{\ast}B/T`$ in Corollary 2. Nothing else in §6 changes, since every lemma is already stated for one cell with $`n\ge B`$.

So (C) is an exposition device, (C\*) is the assumption, and a write-up should state the theorems under (C\*) with the convention in the algorithm.

## 8. Verification of the techniques used

| result | checked against | used in | hypotheses verified |
|---|---|---|---|
| Sub-Gaussian mean: $`\Pr[\hat\mu\ge\mu+\varepsilon]`$ and $`\Pr[\hat\mu\le\mu-\varepsilon]`$ at most $`e^{-n\varepsilon^2/(2\sigma^2)}`$ for independent $`\sigma`$-sub-Gaussian $`X_i-\mu`$ | Lattimore & Szepesvári (2020), Definition 5.2, Theorem 5.3, Lemma 5.4, Corollary 5.5, pp. 76–77; PDF in the mirror, read 2026-09-22 | Lemmas 1, 4, 5 | batch rewards independent given the design and the split (Fact 0, the split being drawn independently of $`\mathcal D`$), identically distributed, noise with proxy $`\sigma`$ by (SG) |
| Chernoff–Hoeffding bound $`\Pr[\mathrm{Bin}(B,p)\ge aB]\le e^{-B\cdot\mathrm{kl}(a,p)}`$ for $`a\ge p`$, and the lower-tail mirror | standard (Hoeffding 1963); as on the general page | Lemmas 3, 4, 5 | independent indicators; $`1/8\le1/2`$; $`1/8\ge1/256`$; $`7/8\ge1/2`$ |
| Hoeffding's lemma: a centered variable in $`[a,b]`$ is $`((b-a)/2)`$-sub-Gaussian | Boucheron, Lugosi & Massart (2013), Lemma 2.2; Lattimore & Szepesvári, Example 5.6(c); both PDFs in the mirror, read 2026-09-22 | Corollary 1, Propositions 3, 4 | the two-point law has range $`2\sigma`$; rewards in $`[0,1]`$ have range $`1`$ |
| Symmetric summands give a symmetric sum, so $`\Pr[\text{sum}\le0]\ge1/2`$ | elementary | (U) for symmetric noise, §2 | i.i.d. noise symmetric about $`0`$ |
| A $`0`$-sub-Gaussian variable has variance $`0`$; and a $`\sigma'`$-sub-Gaussian variable is $`\sigma`$-sub-Gaussian for every $`\sigma\ge\sigma'`$ | Lattimore & Szepesvári, Lemma 5.4(a) and Definition 5.2 | Corollary 2; Propositions 3, 4 | $`\sigma(x,\pi^{\ast}(x))=0`$ a valid proxy there; the deterministic arm of Proposition 3(b) and the constant arm of Proposition 4 under a common $`\sigma`$ |
| multiplicative Chernoff $`\Pr[X\le\mu/2]\le e^{-\mu/8}`$; $`e^{-u}\le1/(eu)`$ | standard; the latter is Lattimore & Szepesvári, p. 77 | Theorem 2 Step 1, Lemma 6, Corollaries 2, 3 | binomial counts; $`B\ge2`$ |
| Markov's inequality | standard | Proposition 3 | non-negative count |
| Gaussian batch means are Gaussian; the binomial lower bound $`\binom Bk\ge e^{B\cdot h(k/B)}/(B+1)`$ | standard; as on the general page | Proposition 2 | Gaussian noise; $`0 < k < B`$ |
| Fact 0, the one-sided reduction and the Proposition 3 construction of the general page | [[2026-09-22-qom-value-based-pessimism]] | Theorem 1 Step 2, Proposition 4 | the instance is symmetric and $`1/2`$-sub-Gaussian |

The Berry–Esseen remark of §2 is used for nothing; its constant is not quoted.

## 9. How it compares

| rule | needs $`\sigma`$ | validity needs | penalty or gap at a cell with $`n`$ rewards, $`m=\lfloor n/B\rfloor`$ | coverage-form bound |
|---|---|---|---|---|
| Hoeffding LCB, exact $`\sigma`$ | yes | (SG) with that $`\sigma`$ | $`\sigma\sqrt{2\ln(2SK/\delta)/n}`$, charged twice | $`2\sqrt2\,\sigma\sqrt{S\bar C^{\ast}\ln(2SK/\delta)/T}`$ |
| greedy, cell means | no | nothing, but the bound needs every action | $`\sigma\sqrt{2\ln(2SK/\delta)/n}`$ at the two actions compared | uniform coverage |
| **QoM-LCB under (SG), (U), (C)** (Theorem 1) | no | (U), e.g. symmetric noise | none explicit; the gap is at most $`4\sigma\sqrt{\ln2/m}`$, charged once | $`8\sigma\sqrt{\ln2\cdot S\bar C^{\ast}B/T}`$, $`B=3.2\ln(2SK/\delta)`$; $`\sqrt{V^{\ast}_{\sigma}}`$ for $`\sigma\sqrt{\bar C^{\ast}}`$ with cell-dependent proxies |
| QoM-LCB under (SG) and (C) alone (Theorem 3) | no | not valid (Proposition 3) | $`4\sigma\sqrt{\ln2/m}`$ at $`\pi^{\ast}`$'s action plus $`\sigma\sqrt{2\ln2/m'}`$ at the chosen action | uniform coverage |
| QoM-LCB of the general page | no | non-negativity | $`4\sqrt{\sigma^2B/n}+\tfrac{11}3B/n`$, $`B=26\ln(2SK/\delta)`$ | $`4\sqrt{2SV^{\ast}B/T}+\tfrac{22}3SC^{\ast}B/T`$ |
| Rashidinejad LCB (Theorem 4.5) | no, needs a range | rewards in $`[0,1]`$ | $`\sqrt{2000\ln(2SK/\delta)/n}`$ | $`\tilde O(\sqrt{S(C^{\ast}-1)/T}+S/T)`$ |

**Reading the table.**

- **Constants.** With $`B=3.2\ln(2SK/\delta)`$, Theorem 2's leading constant is $`11.9\sigma`$ against the exact-$`\sigma`$ Hoeffding LCB's $`2\sqrt2\,\sigma=2.83\sigma`$ in the same form, a factor $`4.2`$, and against Theorem 4.2's $`2`$ for rewards in $`[0,1]`$ a factor $`2.98`$ (Corollary 1). Of the $`4.2`$, $`\sqrt2`$ is the floor at small counts and the rest is the quantile (Proposition 2). Over the level $`\alpha`$, the proven width is smallest near $`\alpha=1/6`$; $`1/8`$ is within $`2`$ percent of it and gives the constants above in closed form.
- **Where QoM-LCB is worth it.** Symmetric noise of unknown, cell-dependent scale, the Gaussian bandit with heteroscedastic unknown variances being the model case: the LCB needs $`\sigma`$ or a bound on it, the sample standard deviation is not a valid substitute without a range or a distributional assumption, and QoM-LCB needs nothing and adapts to the actual spread. For bounded rewards with a known range, the value-based page's rules are better on constants; for skewed unbounded noise, neither this page nor the general page applies.
- **Scaling.** The gap is proxy-scaled rather than count-scaled, as on the general page: the source of Corollary 2 and of Proposition 4.

## 10. The linear case

Written into the Overleaf note as §2.6 on 2026-09-22. The tabular algorithm splits each cell's rewards, which makes a cell a repeated experiment and is tabular by construction. The linear version splits the **rounds** instead, fits one least-squares estimate per batch, and takes the order statistic of the $`B`$ predictions at the pair in question, so one split serves every pair and the features share information across contexts.

**Algorithm.** With $`q^{\ast}(x,a)=\phi(x,a)^{\top}\theta^{\ast}`$, $`\lVert\phi\rVert_2\le1`$, and $`\phi_t:=\phi(x_t,a_t)`$: partition the rounds into $`B`$ balanced batches $`\mathcal B^{1},\dots,\mathcal B^{B}`$; set $`\Lambda^{b}:=\sum_{t\in\mathcal B^{b}}\phi_t\phi_t^{\top}`$ and $`\hat\theta^{b}:=(\Lambda^{b})^{-1}\sum_{t\in\mathcal B^{b}}\phi_tr_t`$, the **ordinary** least-squares fit; put $`\hat q^{b}(x,a):=\phi(x,a)^{\top}\hat\theta^{b}`$; let $`\underline q(x,a)`$ be the $`k`$-th smallest of these; act greedily on $`\underline q`$.

**Why it transfers, and this is the point.** Writing $`\phi:=\phi(x,a)`$,

```math
\hat q^{b}(x,a)-q^{\ast}(x,a)=\phi^{\top}\big(\hat\theta^{b}-\theta^{\ast}\big)=\sum_{t\in\mathcal B^{b}}c_t\eta_t,\qquad c_t:=\phi^{\top}(\Lambda^{b})^{-1}\phi_t,
```

a noise average with **signed** weights, the $`c_t`$ being fixed once the design and the split are fixed. This is exactly where the general page's route dies: an order statistic of batch *means* has non-negative weights, so undershoot follows from non-negative rewards through Feige's theorem, whereas a least-squares prediction has no such structure and no assumption on the sign of the rewards controls it. Symmetry does, and without reference to the weights at all — each $`c_t\eta_t`$ is symmetric, a sum of independent symmetric variables is symmetric, so $`\Pr[\hat q^{b}(x,a)\le q^{\ast}(x,a)]\ge1/2`$ and Lemma 3 goes through unchanged. **This supplies the "new undershoot lemma" that the general page's open problem 4 asks for.**

**Width.** The same sum is sub-Gaussian with proxy $`\sigma^2\sum_tc_t^2`$, and the cross terms telescope against $`\Lambda^{b}`$:

```math
\sum_{t\in\mathcal B^{b}}c_t^2=\phi^{\top}(\Lambda^{b})^{-1}\Big(\sum_{t\in\mathcal B^{b}}\phi_t\phi_t^{\top}\Big)(\Lambda^{b})^{-1}\phi=\phi^{\top}(\Lambda^{b})^{-1}\phi=\lVert\phi\rVert^2_{(\Lambda^{b})^{-1}},
```

so each batch falls $`4\sigma\sqrt{\ln2}\,\lVert\phi\rVert_{(\Lambda^{b})^{-1}}`$ below the mean with probability at most $`1/256`$, the same constant as Lemma 4.

**Theorem (finite context set).** With $`\alpha=1/8`$, $`B\ge3.2\ln(2SK/\delta)`$ and $`\Lambda^{b}\succ0`$ for every $`b`$, conditional on the design and the split, with probability at least $`1-\delta`$, simultaneously for every deterministic $`\pi`$,

```math
J(\pi)-J(\hat\pi)\ \le\ 4\sigma\sqrt{\ln2}\ \mathbb E_{x\sim\nu}\Big[\max_{b\le B}\lVert\phi(x,\pi(x))\rVert_{(\Lambda^{b})^{-1}}\Big].
```

Steps 2 and 3 of Theorem 1 carry over word for word: they use only the two events and the maximization defining $`\hat\pi`$, never the tabular structure.

**Tabular recovery.** At $`\phi=e_{(x,a)}`$, $`\Lambda^{b}`$ is diagonal with entry $`\lvert\mathcal B^{b}(x,a)\rvert`$, so $`\max_b\lVert\phi\rVert_{(\Lambda^{b})^{-1}}=(\min_b\lvert\mathcal B^{b}(x,a)\rvert)^{-1/2}=\lfloor N(x,a)/B\rfloor^{-1/2}`$ and the bound is Theorem 1's exactly. The condition $`\Lambda^{b}\succ0`$ becomes $`N(x,a)>B`$.

**Two limitations, both real.**

- **Ordinary least squares, not ridge.** Ridge shrinks toward the origin, and that shrinkage is a deterministic bias sitting inside the order statistic, which breaks the symmetry the undershoot rests on. What replaces the regularizer is $`\Lambda^{b}\succ0`$, each batch spanning $`\mathbb R^{d}`$ on its own — the analogue of $`N(x,a)>B`$.
- **Infinite context sets are not covered.** The union bound runs over the $`SK`$ pairs, and that is the only place finiteness enters; both lemmas hold at a fixed pair whatever $`\mathcal X`$ is. For a continuum, validity is a statement about infinitely many linear functionals. A covering argument is the natural route: on an $`\varepsilon`$-net of the feature set there are at most $`(3/\varepsilon)^{d}`$ pairs, and between net points $`\underline q`$ moves by at most $`\varepsilon\max_b\lVert\hat\theta^{b}\rVert_2`$, an order statistic of linear functions being Lipschitz in $`\phi`$ with that constant. Validity then holds up to an additive $`\varepsilon(\max_b\lVert\hat\theta^{b}\rVert_2+\lVert\theta^{\ast}\rVert_2)`$, negligible at $`\varepsilon`$ polynomially small in $`T`$, at the cost of $`B\gtrsim d\ln T`$. Not carried out, and the $`B\gtrsim d`$ it forces has no counterpart in the linear LCB, whose quantifier is already uniform.

**Verification.** The two load-bearing identities were checked numerically on random designs before the section was written: $`\sum_tc_t^2=\lVert\phi\rVert^2_{(\Lambda^{b})^{-1}}`$ to machine precision, the deviation identity, the tabular collapse, and that $`t=4\sigma\sqrt{\ln2}\lVert\phi\rVert`$ gives exactly $`2^{-8}`$.

## 11. Open problems, in priority order

1. **Novelty.** As on the general page, not yet checked, and it gates the rest.
2. **Characterize (U) inside the sub-Gaussian class.** Symmetry is sufficient; Proposition 3 shows something is needed. Which skewed laws satisfy $`\Pr[\bar X\le q]\ge p_0`$ at every batch size, and is there a shrinkage device for signed rewards, in the way $`\sum X_i/(m+1)`$ serves non-negative ones, that restores (U) from (SG) alone?
3. **Constants.** Lemma 4 with the exact Gaussian tail reaches the floor; whether a sub-Gaussian-only proof can close the remaining $`26`$ percent, and the level $`\alpha\approx1/6`$.
4. **Optimism online.** Under symmetric noise the high quantile is a valid optimistic index (§2). Its regret analysis for the $`K`$-armed bandit would be the online counterpart of Theorem 1, with no confidence width to compute.
5. **Function approximation beyond the linear, finite-context case.** §10 settles linear classes with finitely many contexts and supplies the undershoot lemma for signed weights that the general page's open problem 4 asks for. What remains: the covering argument for a continuum of contexts, and whether $`B\gtrsim d`$ is really necessary or an artifact of the net; and whether ridge can be used after all, by carrying its bias through the order statistic rather than assuming it away.
6. **An empirical check**, as on the general page.

## 12. Next steps

1. Novelty search, shared with the general page.
2. Decide which assumption the paper leads with: (SG) with symmetric noise, this page, with the non-negative case of the general page as the extension; or the reverse. This page has the cleaner statements and constants, the general page the weaker assumption.
3. ~~Write Theorems 1–2 and the core lemmas into `02_offline_contextual_bandits.tex`~~ — done 2026-09-22 as Overleaf §2.5, with the linear case of §10 as §2.6. Still unwritten there: Theorem 3, Propositions 2–4, and §7.1 on what the count condition costs.

## Sources

- [[2026-09-22-qom-value-based-pessimism]]: Fact 0, the algorithm, the one-sided reduction, the count step, Proposition 2's argument, Proposition 3's instance, and the comparison tables.
- [[contextual-bandits-offline]] §2; [[contextual-bandits-offline-value-based]] Theorem 4.2 and its proof, Theorem 5.1, Theorem 4.5.
- [[cassel2026Quantile]]: the estimator. Nothing else of the paper is used here.
- Lattimore & Szepesvári, *Bandit Algorithms* (2020), §5.3: Definition 5.2, Theorem 5.3, Lemma 5.4, Corollary 5.5, Example 5.6, and the remark $`e^{-x}\le1/(ex)`$ on p. 77. Read from the mirrored PDF on 2026-09-22.
- Boucheron, Lugosi & Massart, *Concentration Inequalities* (2013), Lemma 2.2. Read from the mirrored PDF on 2026-09-22.
- Hoeffding (1963) for the Chernoff–Hoeffding bound.
- Numerical values ($`\mathrm{kl}`$ constants, $`\Phi^{-1}`$ values, the optimal level) are evaluations of closed-form expressions.
