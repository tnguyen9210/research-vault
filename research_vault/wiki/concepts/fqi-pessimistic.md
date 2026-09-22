---
title: "Pessimistic Fitted Q-Learning (PFQL)"
aliases: [PFQL, VAFQL, pessimistic-fitted-q-learning, pfql-algorithm-1, fitted-q-iteration-pessimistic]
tags: [offline-reinforcement-learning, pessimism, fitted-q-iteration, differentiable-function-approximation, variance-awareness, algorithm-walkthrough]
introduced_by: [[Yin2023Offline]]
---

# Pessimistic Fitted Q-Learning (PFQL)

**Definition:** Backward fitted Q-iteration over a [[differentiable-function-approximation]] class in which each Q-estimate is penalized by a data-driven uncertainty bonus $\Gamma_h(s,a) = \beta\sqrt{\nabla_\theta f(\hat\theta_h,\phi)^\top\Sigma_h^{-1}\nabla_\theta f(\hat\theta_h,\phi)}$, where $\Sigma_h$ is the Gram matrix of parameter gradients on the offline data.

> **Scope.** The concept and the line-by-line reading of Algorithm 1
> of [[Yin2023Offline]], on one page. **Assumed, not restated here:**
> the FQI template and its notation, on [[fqi]]; the
> model class, on [[differentiable-function-approximation]].
> **Left to other pages:** the failure pessimism corrects,
> [[overestimation-bias]] and [[extrapolation-error]]; the general
> principle, [[pessimism-principle]]; the horizon-$H$ analysis of
> unpenalized FQI, [[fqi-finite-sample-analysis]]. §11 records what
> was checked against the paper and what was not.

**The one-sentence version:** Algorithm 1 is FQI with pessimism
inserted *inside the backup* rather than at policy extraction — lines
4 and 9 alone are plain FQI, and lines 5–8 are the entire
modification.

## 1. Intuition

PFQL fuses two well-worn ingredients and gets something neither gives alone.

**[[fqi]]** supplies computational realism: a squared-error regression at each step, solvable by SGD, matching what neural FQI and DQN-style critics actually do. Compare the maxmin objectives that general-function-approximation theory produces, which are not runnable.

**Pessimism** supplies safety under distributional shift. Where the data is thin, the fitted model extrapolates and $Q$ is unreliable — usually upward, since downstream maximization selects large errors (see [[overestimation-bias]]). Subtracting an uncertainty penalty makes the algorithm prefer actions it has evidence for, the offline mirror of optimism-under-uncertainty in online RL.

The bonus has a reading that makes the rest of the page easier: it is
the familiar $1/\sqrt{n}$ confidence width with $n$ replaced by a
local, direction-aware effective sample size. §4.3 derives it, and §8
explains the one wrinkle that has no linear-case counterpart — the
width is built from the fit it is meant to protect against.

## 2. The algorithm

```
Input: D = {(s^k_h, a^k_h, r^k_h, s^k_{h+1})}_{k=1..K, h=1..H},  β,  λ > 0
       φ_{h,k} := φ(s^k_h, a^k_h)
Init:  V̂_{H+1}(·) ← 0

for h = H, H−1, …, 1:
  4:  θ̂_h ← argmin_{θ∈Θ}  Σ_k [ f(θ, φ_{h,k}) − r_{h,k} − V̂_{h+1}(s^k_{h+1}) ]²  +  λ‖θ‖²₂
  5:  Σ_h ← Σ_k ∇_θ f(θ̂_h, φ_{h,k}) ∇_θᵀ f(θ̂_h, φ_{h,k})  +  λ I_d
  6:  Γ_h(·,·) ← β √( ∇_θ f(θ̂_h, φ(·,·))ᵀ Σ_h⁻¹ ∇_θ f(θ̂_h, φ(·,·)) )  +  Õ(1/K)
  7:  Q̄_h(·,·) ← f(θ̂_h, φ(·,·)) − Γ_h(·,·)
  8:  Q̂_h(·,·) ← min{ Q̄_h(·,·), H − h + 1 }⁺
  9:  π̂_h(·|·) ← argmax_{π_h} ⟨Q̂_h(·,·), π_h(·|·)⟩_A
      V̂_h(·)   ← max_{π_h}    ⟨Q̂_h(·,·), π_h(·|·)⟩_A

Output: {π̂_h}_{h=1}^H
```

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| $K$ | number of episodes in the offline dataset |
| $H$ | horizon; rewards in $[0,1]$, so values lie in $[0, H-h+1]$ at stage $h$ |
| $\phi(s,a)\in\Psi\subset\mathbb{R}^m$ | feature map |
| $\theta\in\Theta\subset\mathbb{R}^d$ | parameter; $d$, not $m$, drives the bound |
| $f(\theta,\phi)$ | the model, three-times differentiable in $\theta$ — see [[differentiable-function-approximation]] |
| $\kappa_1,\kappa_2,\kappa_3$ | bounds on $\|\nabla_\theta f\|, \|\nabla^2 f\|, \|\nabla^3 f\|$ |
| $\kappa$ | uniform-coverage constant (Assumption 2.3) |
| $C_\Theta$ | bound on $\|\theta\|_2$ |
| $\iota$ | polylog factor |

## 3. Structure before content

Three things about the shape of the loop, since they differ from the classical FQI presentation:

**Backward over $h$, once.** This is the finite-horizon episodic setting, so the loop indexes *stages*, not iterations. Each stage gets its own parameter $\hat\theta_h$, and one sweep suffices — $H$ steps of lookahead is all there is. Classical FQI's $Q_0\to Q_1\to\cdots\to Q_K$ over a shared $Q$ is the infinite-horizon discounted analogue.

**No $\gamma$.** Undiscounted; the finite horizon does the work that discounting does elsewhere.

**$\hat V_{H+1}\leftarrow 0$** is the terminal condition, not an initialization heuristic. There is genuinely no value after the last stage.

## 4. Line by line

### 4.1 Line 4 — the regression

$$
\hat\theta_h \leftarrow \arg\min_{\theta\in\Theta}\ \sum_{k=1}^K\Big[f(\theta,\phi_{h,k}) - \underbrace{r_{h,k} - \hat V_{h+1}(s^k_{h+1})}_{\text{the FQI label}}\Big]^2 + \lambda\|\theta\|_2^2
$$

This is the FQI step verbatim: inputs $(s^k_h,a^k_h)$, labels "immediate reward + next-stage value," ordinary least squares.

**What the ridge term does.** Two jobs. It guarantees $\Sigma_h \succeq \lambda I_d$ at line 5, so the inverse in line 6 exists unconditionally. And it keeps $\hat\theta$ inside the compact $\Theta$, which the smoothness constants depend on — hence the condition $\lambda \le 1/2C_\Theta^2$ in Theorem 3.2, small enough that regularization bias stays lower-order.

**The part that makes the paper hard.** Because $f$ is nonlinear in $\theta$, this argmin has **no closed form**. Under LSVI (linear MDPs) $\hat w_h$ is an explicit least-squares solution you can substitute into the analysis; here you know only that $\hat\theta_h$ is a *stationary point* of

$$
Z_h(\theta) := \sum_k\big[f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]\nabla f(\theta,\phi_{h,k}) + \lambda\theta,
\qquad Z_h(\hat\theta_h) = 0 .
$$

Expanding $Z_h$ around $\hat\theta_h$ produces a curvature term $\Delta\Sigma_h^s = \sum_k[\text{residual}_k]\nabla^2_{\theta\theta}f$ that destroys positive definiteness — under linearity $\nabla^2_{\theta\theta}f\equiv 0$ and it vanishes. Working around it is the technical content of [[Yin2023Offline]].

Practically, the square loss is the *reason* FQI was chosen here: SGD applies directly, unlike the maxmin objectives that general-function-approximation theory produces.

### 4.2 Line 5 — the information matrix

$$
\Sigma_h \leftarrow \sum_{k=1}^K \nabla_\theta f(\hat\theta_h,\phi_{h,k})\,\nabla_\theta^\top f(\hat\theta_h,\phi_{h,k}) + \lambda I_d
$$

The Gram matrix of **gradients**, evaluated at the freshly fitted $\hat\theta_h$ — i.e. the empirical Fisher information of the local linearization. The intuition: near $\hat\theta_h$, $f$ behaves like $\nabla_\theta f$, so all the linear-model machinery transfers with $\phi \mapsto \nabla_\theta f$.

Setting $f = \langle\theta,\phi\rangle$ gives $\nabla_\theta f = \phi$ and $\Sigma_h = \sum_k\phi\phi^\top + \lambda I$, the standard design matrix.

### 4.3 Line 6 — the uncertainty width

$$
\Gamma_h(s,a) \leftarrow \beta\sqrt{\nabla_\theta f(\hat\theta_h,\phi(s,a))^\top\Sigma_h^{-1}\nabla_\theta f(\hat\theta_h,\phi(s,a))} + \tilde{O}(1/K)
$$

Define

$$
m(s,a) := \Big(\nabla_\theta f^\top\Sigma_h^{-1}\nabla_\theta f\Big)^{-1}
\qquad\Longrightarrow\qquad
\Gamma_h(s,a) = \frac{\beta}{\sqrt{m(s,a)}} .
$$

So the bonus is the familiar $1/\sqrt{n}$ confidence width, with $n$ replaced by the **effective sample size at $(s,a)$ along the gradient direction** — how much the data constrains the model in the specific direction that changing $f(\cdot,\phi(s,a))$ requires. Well-covered pairs: large $m$, small penalty. Pairs the data says nothing about: small $m$, large penalty.

Under linearity this is the elliptical bonus $\beta\|\phi\|_{\Sigma_h^{-1}}$ from linear-bandit and linear-MDP analyses.

The $\tilde{O}(1/K)$ tail is stated in the paper to be "for theoretical reason only" — a higher-order correction to make the confidence bound valid, negligible at any practical $K$.

### 4.4 Lines 7–8 — subtract, then clip

$$
\bar Q_h \leftarrow f(\hat\theta_h,\phi) - \Gamma_h,
\qquad
\hat Q_h \leftarrow \min\{\bar Q_h,\ H-h+1\}^+
$$

Line 7 is the lower confidence bound. Line 8 is free accuracy from known structure: with rewards in $[0,1]$ and $H-h+1$ stages remaining, no true value can exceed $H-h+1$ or fall below $0$, so any estimate outside that interval is provably wrong and clipping strictly improves it. Standard in this literature and it tightens the analysis.

### 4.5 Line 9 — greedy extraction

$$
\hat\pi_h \leftarrow \arg\max_{\pi_h}\langle \hat Q_h(\cdot,\cdot),\pi_h(\cdot|\cdot)\rangle_\mathcal{A},
\qquad
\hat V_h \leftarrow \max_{\pi_h}\langle \hat Q_h(\cdot,\cdot),\pi_h(\cdot|\cdot)\rangle_\mathcal{A}
$$

The inner product is over the action space, so this is $\arg\max_a$ and $\max_a$ written to permit stochastic $\pi_h$. Nothing exotic — but note *which* $Q$ it maximizes, which is the next section.

## 5. Why the pessimism goes inside the backup

The single most important structural point, and easy to miss because it lives in the *ordering* rather than in any one line.

$\hat V_h$ produced at line 9 becomes line 4's **target at the next iteration** ($h-1$). So the value being backed up is $\max_a \hat Q_h$, where $\hat Q_h$ already has $\Gamma_h$ subtracted.

That is what neutralizes [[extrapolation-error]]. In vanilla FQI the target's $\max_{a'}$ ranges over the whole action space and preferentially selects whichever unsupported action the approximator overvalued; the contaminated value then propagates backward through every remaining stage. Here an unsupported action has small $m(s,a)$, hence large $\Gamma_h$, hence loses the max *before* it can enter any target:

| $a$ | $f(\hat\theta_h,\phi)$ | $\Gamma_h$ | $\hat Q_h$ |
|---|---|---|---|
| $A$ (well covered) | 10 | 1 | **9** |
| $B$ (unsupported) | 20 | 15 | 5 |
| $C$ (unsupported) | 8 | 10 | 0 (clipped) |

Vanilla FQI backs up $20$; PFQL backs up $9$.

**Contrast with penalizing at extraction only.** Running plain FQI to completion and subtracting a penalty just before choosing actions gives a cautious final policy computed from values that were already contaminated during the backups. Different, and weaker. The penalty has to be in the loop.

## 6. Hyperparameters and conditions

| Quantity | Setting | Why |
|---|---|---|
| $\beta$ | $8dH\iota$ (Thm 3.2) | scales the width so $\Gamma_h$ is a valid high-probability bound; the $H$ is what VAFQL removes |
| $\lambda$ | $0 < \lambda \le 1/2C_\Theta^2$ | invertibility of $\Sigma_h$ + bounded parameter, with bias kept lower-order |
| $K$ | burn-in threshold | roughly $K \gtrsim \frac{\kappa_1^6}{\kappa^2}\big(\log\frac{2Hd}{\delta} + d\log(1 + \tfrac{4\kappa_1^4\kappa_2C_\Theta K^3}{\lambda^2})\big)$ and $K \ge 4\lambda/\kappa$ — read exact constants from Theorem 3.2 |

The burn-in is a covering-argument artifact: the guarantee needs $\hat\theta_h$ already close enough to $\theta^*_h$ for the local linearization to be meaningful, and that requires enough data before anything can be said at all.

## 7. Specializations

**Linear MDPs.** $f = \langle\theta,\phi\rangle$, so $\nabla_\theta f = \phi$, $\Sigma_h = \sum_k\phi\phi^\top+\lambda I$, $\Gamma_h = \beta\|\phi\|_{\Sigma_h^{-1}}$. Algorithm 1 becomes PEVI (Jin et al. 2021b) exactly, and Theorem 3.2 collapses to its bound. PFQL is a strict generalization, not a parallel construction.

**Tabular.** One-hot $\phi$ makes $m(s,a)$ the literal visitation count $N(s,a)$ and $\Gamma_h \propto \beta/\sqrt{N(s,a)}$ — count-based pessimism, recovering VPVI (Yin & Wang 2021).

**GLM (Corollary 3.3).** With $f(\theta,\phi) = g(\langle\theta,\phi\rangle)$, the chain rule puts the link derivative into the width: $\Gamma_h \propto \sqrt{g'(\langle\hat\theta_h,\phi\rangle)^2\,\phi^\top\Sigma_h^{-1}\phi}$. Where the link is flat, the model is insensitive to $\theta$ there and the effective uncertainty shrinks accordingly.

## 8. The guarantee

**Guarantee (Thm 3.2).** Under realizability + Bellman completeness and uniform coverage, with $\epsilon_\mathcal{F}=0$ and $K$ large enough, with probability $1-\delta$:

$$
v^{\pi^*} - v^{\hat\pi} \;\le\; 16dH\sum_{h=1}^H \mathbb{E}_{\pi^*}\Big[\sqrt{\nabla_\theta^\top f(\theta^*_h,\phi)\,\Sigma_h^{\star-1}\,\nabla_\theta f(\theta^*_h,\phi)}\Big]\cdot\iota + \tilde{O}(C'_\text{hot}/K).
$$

Following Jin et al. (2021b), the suboptimality decomposes as

$$
v^{\pi} - v^{\hat\pi} \;\le\; \sum_{h=1}^H 2\,\mathbb{E}_{\pi}\big[\Gamma_h(s_h,a_h)\big]
\qquad\text{provided}\qquad
\big|(\mathcal{P}_h\hat V_{h+1} - f(\hat\theta_h,\phi))(s,a)\big| \le \Gamma_h(s,a) .
$$

Two consequences worth extracting.

**What pessimism actually costs you.** The penalty evaluated *along the comparator policy's trajectory* — not the model's uncertainty in general. What matters is how uncertain the model is exactly where $\pi^*$ would go, which is why the bound is instance-dependent (see [[instance-dependent-bounds]]) and why coverage of $\pi^*$ specifically is the operative condition rather than coverage everywhere.

**Where the nonlinearity bites.** The whole guarantee rests on that validity condition, and there is a circularity in establishing it: $\Gamma_h$ is computed from $\hat\theta_h$, so the penalty meant to protect against a bad fit is itself built from the fit. Plug in an arbitrary $\theta$ and $\Gamma_h$ is meaningless, possibly harmful. In a linear MDP no such issue arises — $\phi^\top(\Sigma_h^\text{linear})^{-1}\phi$ has no parameter dependence. [[Yin2023Offline]] breaks the circle with a non-asymptotic $\|\theta_{\mathcal{T}\hat V_{h+1}} - \hat\theta_h\|_2 = \tilde{O}(\sqrt{dH}/(\kappa\sqrt{K}))$, obtained by reducing to Chen & Jiang's GFA analysis plus covering — the OPE predecessor (Zhang et al. 2022a) had only an asymptotic $B(\delta)/\sqrt{K}$, which could hide an $e^H$ and destroy sample efficiency.

## 9. VAFQL — the variance-aware variant

Algorithm 3 changes **only line 4**, dividing each squared residual by an estimated conditional variance:

$$
\hat\theta_h \leftarrow \arg\min_{\theta\in\Theta}\ \sum_{k=1}^K \frac{\big[f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]^2}{\hat\sigma^2_h(s^k_h,a^k_h)} + \lambda\|\theta\|_2^2
$$

with $\sigma^{\star2}_h := \max\{1,\text{Var}_{P_h}V^*_{h+1}\}$. High-noise transitions carry less weight. $\Sigma_h$ becomes the variance-weighted $\Lambda_h$, and $\beta$ drops from $8dH\iota$ to $8d\iota$ — a full factor of $H$, since $\text{Var}_{P_h}V^*_{h+1}\le H^2$. On deterministic transitions $\sigma^\star_h\approx 0$, $\Lambda_h^{\star-1}\to 0$, and the rate improves to $O(1/K)$.

Theorem 4.1 is the resulting guarantee, with $\Sigma^\star_h$ replaced by $\Lambda^\star_h = \sum_k \nabla f\nabla f^\top/\sigma^{\star2}_h + \lambda I_d$, and Theorem 4.2 gives a matching lower bound up to $\sqrt{d}$.

## 10. What Algorithm 1 does not do

- **No computational guarantee.** Line 4 is a nonconvex argmin, analyzed as exactly solved. "Provably efficient" in this paper means *statistically* efficient.
- **No misspecification.** Theorems 3.2 and 4.1 assume $\epsilon_\mathcal{F} = 0$ (exact Bellman completeness); the $\epsilon_\mathcal{F} > 0$ case is deferred to Appendix H.
- **No single-policy coverage.** Assumption 2.3 is uniform over $\Theta$, stronger than the single-policy concentrability Xie et al. (2021a) achieve for GFA.
- **No experiments.** The paper is purely theoretical; PFQL is not run on a benchmark.

## 11. Provenance

*Source.* Everything in §2–§10 follows Algorithm 1 of
[[Yin2023Offline]] (Yin, Duan, Wang & Wang, *Offline RL with
Differentiable Function Approximation is Provably Efficient*, ICLR
2023), read against the paper on 2026-08-19. The line numbering is
theirs; the section titles, the line-by-line commentary and the
reasons given for each step are this page's.

*Quoted.* The algorithm itself, Assumption 2.3, the hyperparameter
conditions of §6, the guarantee of §8 (their Theorem 3.2) and the
VAFQL variant of §9 (their Theorems 4.1 and 4.2).

*This page's own, not the paper's.* The framing of §5 — that the
whole modification is pessimism inside the backup rather than at
policy extraction, so lines 4 and 9 alone are plain FQI — is how this
page organises the algorithm, not a claim the paper makes in those
words. The same goes for the four limitations in §10, which are
inferred from what the theorems assume rather than stated as
limitations by the authors.

*Not checked.* The comparison in §10 to the single-policy
concentrability of Xie et al. (2021a) is taken from the paper's own
related-work framing and has not been verified against Xie et al.
directly; that paper is `xie2021Bellmanconsistent`, uningested.

*This page is a merge.* It was assembled on 2026-09-19 from the
concept page `pessimistic-fitted-q-learning` and the walkthrough
`pfql-algorithm-1`, which are the same subject at two levels of
detail. Nothing was dropped except duplicated statements of the
algorithm, the bonus reading and VAFQL.

## Key Papers

- [[Yin2023Offline]] — introduces PFQL and VAFQL; first instance-dependent offline RL guarantee under nonlinear function approximation
- Jin et al. (2021b) — PEVI, the linear-MDP special case PFQL strictly generalizes
- Yin & Wang (2021) — VPVI, the tabular predecessor
- Yin et al. (2022) — variance-aware linear offline RL, recovered by VAFQL up to $\sqrt{d}$
- Ernst et al. (2005), Riedmiller (2005) — fitted Q-iteration and neural FQI, the practical lineage
- Xie et al. (2021a) — PSPI: weaker (single-policy) coverage but a slower $O(n^{-1/3})$ rate and a maxmin objective

## Variants

- **VAFQL** — the variance-aware variant of §9. Changes only line 4, reweighting each squared residual by an estimated conditional variance; saves a factor $H$ and is minimax-optimal up to $\sqrt{d}$
- **PEVI / linear-MDP pessimism** — the specializations of §7, recovered by taking $f$ linear

## Related Concepts

- [[fqi]] — the base template; PFQL is FQI with an uncertainty penalty subtracted before the greedy step, and lines 4 and 9 are its two steps
- [[fqi-finite-sample-analysis]] — the finite-sample analysis of unpenalized FQI that this line descends from
- [[differentiable-function-approximation]] — the class $f$ lives in, and the source of the $\nabla^2_{\theta\theta}f$ obstacle
- [[instance-dependent-bounds]] — what the $\mathbb{E}_{\pi^*}[\Gamma_h]$ form delivers
- [[pessimism-principle]] — the general idea; $\Gamma_h$ is one computable instance of the width $b(s,a)$
- [[extrapolation-error]] — the offline-specific failure the penalty is aimed at, and what the ordering of lines 6–9 is designed to stop
- [[overestimation-bias]] — the failure mode pessimism corrects
- [[implicit-q-learning]] — the opposing design in offline RL: rather than penalize uncertainty at out-of-sample actions, never evaluate them. Empirically strong, theoretically much weaker
- [[importance-weighting]] / [[Ryu2025Improved]] — pessimism via betting-based LCBs, applied to offline policy *selection* rather than learning
- [[2026-08-19-offline-fqi-walkthrough]] — the saved query that arrives here from the FQI template
- **Model-based/model-free bridge:** FQI is a batch Q-learning update but also an instantiation of approximate value iteration, so PFQL unifies the value-iteration (PEVI, VPVI) and fitted-Q lineages

## Current State and Open Problems

The reference theoretical algorithm for offline RL beyond linear models. Its guarantee is the sharpest available in that setting and comes with a near-matching lower bound. What it does not offer: any computational claim (the fitted-Q step is a nonconvex argmin, analyzed as exactly solved), tolerance of overparameterized classes (uniform coverage fails under non-identifiability), or a $\sqrt{d}$ rate. Empirically PFQL is not run — the practical offline RL field uses [[implicit-q-learning]], CQL, and TD3+BC, none of which carry instance-dependent guarantees. Closing that theory/practice gap is the standing challenge in this line.
