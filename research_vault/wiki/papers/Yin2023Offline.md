---
title: "Offline Reinforcement Learning with Differentiable Function Approximation is Provably Efficient"
authors: [Ming Yin, Mengdi Wang, Yu-Xiang Wang]
year: 2023
venue: ICLR
arxiv: "2210.00750"
tags: [offline-reinforcement-learning, differentiable-function-approximation, instance-dependent-bounds, pessimism, function-approximation, sample-complexity, realizability]
source: raw/papers/Yin2023Offline.pdf
---

# Offline Reinforcement Learning with Differentiable Function Approximation is Provably Efficient

**TL;DR:** The first **instance-dependent** offline RL guarantee under *nonlinear* function approximation. Over the [[differentiable-function-approximation]] class $\mathcal{F} = \{f(\theta,\phi(\cdot,\cdot)) : \theta \in \Theta \subset \mathbb{R}^d\}$, [[pessimistic-fitted-q-learning]] (PFQL) achieves a suboptimality bound driven by a Fisher-information-style quantity $\sqrt{\nabla_\theta f^\top \Sigma_h^{\star-1} \nabla_\theta f}$; a variance-reweighted variant (VAFQL) removes a factor of $H$ and is minimax-optimal up to $\sqrt{d}$.

## Problem

Offline RL theory has been split between two unsatisfying regimes.

**Linear models** (linear MDPs, linear mixture MDPs) admit sharp, instance-dependent analysis because least-squares value iteration has a closed form — but they cannot describe the nonlinear approximators practitioners actually deploy.

**General function approximation** (GFA; Chen & Jiang 2019, Xie et al. 2021a) imposes only realizability and completeness, so it covers everything — but precisely because the class is structureless, the bounds come out worst-case, of the form $O(V_\text{max}^2\sqrt{C/n})$. As Zanette & Brunskill (2019) observed, real algorithms routinely beat such problem-independent bounds, so these say little about which instances are easy.

The gap: **a class expressive enough to include nonlinear models, yet structured enough to support [[instance-dependent-bounds]].** Differentiability is the structure this paper proposes — gradients and higher-order derivatives are exactly the local information a worst-case analysis throws away.

## Method

### The function class (Definition 1.1)

With feature map $\phi : \mathcal{S}\times\mathcal{A} \to \Psi \subset \mathbb{R}^m$ and compact parameter space $\Theta \subset \mathbb{R}^d$,

$$
\mathcal{F} := \{f(\theta, \phi(\cdot,\cdot)) : \mathcal{S}\times\mathcal{A}\to\mathbb{R},\ \theta\in\Theta\},
$$

where $f$ is three-times differentiable in $\theta$ with $f, \partial_\theta f, \partial^2_{\theta\theta} f, \partial^3_{\theta\theta\theta} f$ jointly continuous, and $\|\theta\|_2 \le C_\Theta$, $|f| \le B_\mathcal{F}$, $\|\nabla_\theta f\|_2 \le \kappa_1$, $\|\nabla^2 f\| \le \kappa_2$, $\|\nabla^3 f\| \le \kappa_3$. Setting $f(\theta,\phi) = \langle\theta,\phi\rangle$ recovers linear; one-hot $\phi$ recovers tabular; a link function recovers GLM.

### Assumptions

- **2.1 Realizability + Bellman completeness.** $Q_h^* = f(\theta_h^*,\phi)$ for some $\theta_h^*$, and $\sup_{V}\inf_{f\in\mathcal{F}}\|f - \mathcal{P}_h(V)\|_\infty \le \epsilon_\mathcal{F}$. See [[realizability]].
- **2.2 Concentrability.** $C_\text{eff} := \sup_\pi \sup_h \|d_h^\pi/d_h^\mu\|_{2,d^\mu_h}^2 < \infty$. Used only for the warm-up.
- **2.3 Uniform coverage.** For all $h$, some $\kappa > 0$:
  $\mathbb{E}_{\mu,h}[(f(\theta_1,\phi) - f(\theta_2,\phi))^2] \ge \kappa\|\theta_1-\theta_2\|_2^2$ $(\star)$ and $\mathbb{E}_{\mu,h}[\nabla f \nabla f^\top] \succ \kappa I$ $(\star\star)$.
  Under linearity, $(\star)$ and $(\star\star)$ coincide and reduce to the usual $\Sigma_h^\text{feature} \succ \kappa I$.

### Algorithms

**VFQL** (warm-up, Alg. 2) — vanilla fitted Q-iteration, needs only 2.2.

**PFQL** (Alg. 1) — fitted Q-iteration plus pessimism, run backward over $h$:

1. $\hat\theta_h \leftarrow \arg\min_\theta \sum_{k=1}^K [f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})]^2 + \lambda\|\theta\|_2^2$
2. $\Sigma_h \leftarrow \sum_k \nabla_\theta f(\hat\theta_h,\phi_{h,k})\nabla_\theta^\top f(\hat\theta_h,\phi_{h,k}) + \lambda I_d$
3. $\Gamma_h(\cdot,\cdot) \leftarrow \beta\sqrt{\nabla_\theta f(\hat\theta_h,\phi)^\top \Sigma_h^{-1}\nabla_\theta f(\hat\theta_h,\phi)} + \tilde{O}(1/K)$
4. $\bar Q_h \leftarrow f(\hat\theta_h,\phi) - \Gamma_h$, then clip and greedify

The penalty has a clean reading: $m(s,a) := (\nabla_\theta f^\top\Sigma_h^{-1}\nabla_\theta f)^{-1}$ is the **effective sample size at $(s,a)$ along the gradient direction** $\nabla_\theta f$, and the bonus is $\beta/\sqrt{m(s,a)}$. Uncertainty is measured in parameter space, projected through the local gradient.

**VAFQL** (Alg. 3) — reweights each squared residual by an estimated conditional variance $\hat\sigma_h^2(s,a) \approx \max\{1, \text{Var}_{P_h} V^*_{h+1}\}$, so high-noise transitions count less.

## Results

All bounds are on the suboptimality gap $v^{\pi^*} - v^{\hat\pi}$, omitting higher-order terms.

| Algorithm | Assumption | Main term |
|---|---|---|
| VFQL (Thm 3.1) | Concentrability 2.2 | $\sqrt{C_\text{eff}}H\cdot\tilde{O}\big(\sqrt{(H^2d+\lambda C_\Theta^2)/K}\big)$ |
| PFQL (Thm 3.2) | Uniform coverage 2.3 | $16dH\sum_{h}\mathbb{E}_{\pi^*}\big[\sqrt{\nabla_\theta^\top f(\theta_h^*,\phi)\,\Sigma_h^{\star-1}\,\nabla_\theta f(\theta_h^*,\phi)}\big]$ |
| VAFQL (Thm 4.1) | Uniform coverage 2.3 | $16d\sum_{h}\mathbb{E}_{\pi^*}\big[\sqrt{\nabla_\theta^\top f(\theta_h^*,\phi)\,\Lambda_h^{\star-1}\,\nabla_\theta f(\theta_h^*,\phi)}\big]$ |

with $\Sigma_h^\star = \sum_k \nabla_\theta f(\theta_h^*,\phi_{h,k})\nabla_\theta^\top f(\theta_h^*,\phi_{h,k}) + \lambda I_d$ and $\Lambda_h^\star$ its variance-weighted analogue $\sum_k \nabla f \nabla f^\top/\sigma_h^{\star 2} + \lambda I_d$.

**What the instance measure buys.** The GFA bound $O(V^2_\text{max}\sqrt{C/n})$ is identical for every MDP with the same concentrability. The PFQL quantity is not: two instances $\mathcal{M}_1, \mathcal{M}_2$ induce different $\theta^*_{h,\mathcal{M}_i}$ and therefore different gradient geometry, so the bound separates them explicitly.

**Rate recovery.** $\|\nabla_\theta f(\theta_h^*,\phi)\|_{\Sigma_h^{-1}} \lesssim \kappa_1/\sqrt{\kappa K}$, so the main term is $O(dH/\sqrt{\kappa K})$ — the standard $1/\sqrt{K}$ rate.

**Strict generalization.** With $f = \langle\theta,\phi\rangle$, $\nabla_\theta f = \phi$ and Thm 3.2 collapses to $O(dH\sum_h\mathbb{E}_{\pi^*}[\sqrt{\phi^\top(\Sigma_h^\text{linear})^{-1}\phi}])$ — exactly Jin et al. (2021b)'s PEVI bound for linear MDPs. Corollary 3.3 instantiates GLM, where the measure picks up the link derivative: $\sqrt{f'(\langle\hat\theta_h,\phi\rangle)^2\,\phi^\top\Sigma_h^{-1}\phi}$.

**Variance awareness.** VAFQL saves a full factor of $H$ (since $\text{Var}_{P_h}V^*_{h+1}\le H^2$), and on deterministic transitions $\sigma_h^\star \approx 0$ so $\Lambda_h^{\star-1}\to 0$, giving a fast $O(1/K)$ rate. Reduced to linear MDPs it recovers Yin et al. (2022) up to $\sqrt{d}$.

**Lower bound (Thm 4.2).** Via reduction to Zanette et al. (2021) / Yin et al. (2022): for $K > c'd^3$ there is a family of instances on which any algorithm suffers $\ge c\sqrt{d}\sum_h \mathbb{E}_{\pi^*}[\sqrt{\nabla_\theta f^\top(\Lambda_h^{\star,p})^{-1}\nabla_\theta f}]$. So VAFQL is optimal up to $\sqrt{d}$.

**A curious observation.** The complexity depends on the *parameter* dimension $d$ and never on the *feature* dimension $m$ — not even in higher-order terms. Linear analysis cannot see this because $d = m$ there. The authors suggest it as a hint about what actually governs hardness in representation learning; they do not pursue it.

## Assumptions & Theorems

### The technical crux

The whole difficulty is that fitted-Q has **no closed-form solution**. Under LSVI (Jin et al. 2020b, 2021b) $\hat w_h$ is an explicit least-squares solution; here $\hat\theta_h$ is only known to be a stationary point of

$$
Z_h(\theta) := \sum_{k=1}^K \big[f(\theta,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]\nabla f(\theta,\phi_{h,k}) + \lambda\theta,
\qquad Z_h(\hat\theta_h) = 0 .
$$

Vector Taylor expansion at $\hat\theta_h$ gives $Z_h(\theta) - Z_h(\hat\theta_h) = \Sigma_h^s(\theta - \hat\theta_h) + R_K(\theta)$ with

$$
\Sigma_h^s = \underbrace{\sum_k\big[f(\hat\theta_h,\phi_{h,k}) - r_{h,k} - \hat V_{h+1}(s^k_{h+1})\big]\nabla^2_{\theta\theta}f(\hat\theta_h,\phi_{h,k})}_{=: \Delta\Sigma_h^s} + \underbrace{\sum_k \nabla_\theta f\nabla_\theta^\top f + \lambda I_d}_{=: \Sigma_h}.
$$

$\Delta\Sigma_h^s$ carries the curvature $\nabla^2_{\theta\theta}f$ and **destroys positive definiteness**, so $\Sigma_h^s$ cannot be inverted. Under linearity $\nabla^2_{\theta\theta}f \equiv 0$ and the problem evaporates; this term is the entire price of nonlinearity. The fix is to move $\Delta\Sigma_h^s$ into the residual and invert the well-conditioned $\Sigma_h$ instead.

That still requires a finite-sample bound on $\|\theta_{\mathcal{T}\hat V_{h+1}} - \hat\theta_h\|_2$. The OPE predecessor (Zhang et al. 2022a) obtains only an *asymptotic* $B(\delta)/\sqrt{K}$ via Prohorov's theorem — useless here, since an unquantified $B(\delta)$ could hide $e^H\log(1/\delta)$ and destroy sample efficiency. The paper instead reduces to Chen & Jiang (2019)'s GFA analysis, bounding the loss gap by an orthogonal decomposition and a quadratic inequality, then extends the finite-hypothesis argument to all of $\mathcal{F}$ by covering. Result: $\|\theta_{\mathcal{T}\hat V_{h+1}} - \hat\theta_h\|_2 = \tilde{O}(\sqrt{dH}/(\kappa\sqrt{K}))$, non-asymptotic.

### The honest gaps

1. **Uniform coverage 2.3 excludes overparameterized networks.** Condition $(\star)$ is an identifiability requirement: distinct parameters must induce functions that differ measurably under $\mu$. Overparameterized neural networks violate it by construction — permutation and scaling symmetries make $\theta_1 \ne \theta_2$ realize identical functions, so $\mathbb{E}_\mu[(f(\theta_1,\phi)-f(\theta_2,\phi))^2] = 0$. The paper says as much in a footnote-level remark ("2.3 can be violated for function class $\mathcal{F}$ that is *not identifiable*") and offers GLM as the representative nonlinear example. But Section 1 motivates DFA with "when $f$ is specified to be neural networks, $\theta$ corresponds to the weights of each network layer," which reads as broader reach than the theorems support.
2. **$d$ is the parameter count, and the rate carries $d$, not $\sqrt{d}$.** For the deep models invoked in the motivation, $d$ is enormous and the bound is uninformative in exactly that regime. The authors flag this in the conclusion, attribute it to the covering argument, and ask whether $d$ is essential — a genuinely open question, honestly stated.
3. **"Provably efficient" = statistically efficient.** $\hat\theta_h$ is the argmin of a nonconvex objective, analyzed as if exactly obtained. Nothing is claimed about finding it. This is a sharp contrast with [[Foster2025Foundation]], whose entire subject is the separation between statistical and computational efficiency.
4. **Coverage is all-policy, not single-policy.** Xie et al. (2021a) achieve GFA guarantees under *single-policy* concentrability; 2.3 is stronger. The authors believe the DFA analogue is derivable but expect it to yield a computationally intractable algorithm, and leave it open.
5. Main theorems assume $\epsilon_\mathcal{F} = 0$ (exact Bellman completeness); the misspecified case is deferred to Appendix H.

## Strengths & Limitations

**Strengths.** The choice of structure is well-judged — differentiability is weak enough to include GLMs and nonconvex parametric models, strong enough to yield gradient geometry, and it degrades gracefully to every previously studied special case rather than sitting beside them. The effective-sample-size reading of the pessimism bonus is genuinely illuminating and is the right generalization of the linear-MDP elliptical bonus. Including a matching lower bound is more than most papers in this line do. The technical obstacle ($\Delta\Sigma_h^s$ breaking invertibility) is identified precisely and the workaround is not a patch but a real argument. Limitations are stated plainly in the conclusion rather than buried.

**Limitations.** The framing-to-theorem gap on neural networks (see honest gap 1) is the main overclaim. $d$-dependence limits practical bite. No computational guarantee. Uniform coverage is strong and, unlike concentrability, depends jointly on the MDP *and* the function class, which makes it harder to interpret or check. Third-order smoothness is required for technical reasons with no argument that it is necessary. Purely theoretical — no experiments, which is defensible for this contribution but leaves the instance measure unvalidated as a *predictor* of empirical difficulty.

## Connections

- [[differentiable-function-approximation]] — the function class introduced here for policy learning
- [[pessimistic-fitted-q-learning]] — PFQL and its variance-aware variant VAFQL
- [[instance-dependent-bounds]] — the guarantee type this paper delivers for nonlinear offline RL
- [[offline-reinforcement-learning]] — the setting
- [[realizability]] — Assumption 2.1, alongside Bellman completeness
- [[coverage-coefficient]] — concentrability (2.2) and uniform coverage (2.3) are the offline-RL members of the coverage-condition family that [[Foster2025Foundation]] formalizes for LM alignment
- **Theory counterpart:** [[Kostrikov2022Offline]] — IQL is the empirical face of the same problem. IQL avoids out-of-sample queries and offers only an asymptotic $\tau\to1$ guarantee with a binary support condition; PFQL queries freely but subtracts a computable uncertainty penalty and delivers the instance-dependent bound IQL lacks. Reading them together: pessimism and in-sample learning are two answers to one question, and only pessimism currently has a sharp theory.
- **Shares the pessimism principle:** [[Ryu2025Improved]] — PUB applies betting-based LCBs to offline *policy selection*; PFQL applies a gradient-geometry LCB to offline *policy learning*
- **Generalizes:** Jin et al. (2021b) PEVI (linear MDPs), Yin & Wang (2021) VPVI (tabular), Yin et al. (2022) (variance-aware linear)
- **Contrasts with:** Chen & Jiang (2019), Xie et al. (2021a) — GFA, worst-case bounds, weaker coverage
- [[Yin-Ming]], [[Wang-Mengdi]], [[Wang-Yu-Xiang]] — authors
- [[Foster-Dylan-J]] — co-author of the "fundamental barriers" result (Foster et al. 2021) that motivates Assumption 2.1

## Open Questions

- **Is the $d$ dependence essential, or an artifact of covering?** The lower bound (Thm 4.2) leaves a $\sqrt{d}$ gap. Closing it would say whether nonlinearity genuinely costs a $\sqrt{d}$ over the linear case.
- **Can uniform coverage be relaxed to single-policy concentrability?** The authors conjecture yes at the cost of computational tractability. A version that keeps both would be the real result.
- **What replaces identifiability for overparameterized models?** Since $(\star)$ fails for neural networks, the interesting question is whether a quotient formulation — coverage on the function space or on an equivalence class of parameters, rather than on $\theta$ — restores the guarantee. This seems like the most valuable follow-up and the paper does not raise it.
- **Is third-order smoothness necessary?** The authors ask what survives with only first- or second-order information.
- **Parameters vs. features.** Why does $m$ never appear? If real, this is a statement about representation learning that deserves its own treatment.
- **Does the Fisher-information measure predict empirical difficulty?** The bound separates instances in theory; nobody has checked whether $\sum_h\mathbb{E}_{\pi^*}[\|\nabla_\theta f\|_{\Sigma_h^{\star-1}}]$ correlates with observed hardness on, say, D4RL. That is a cheap and informative experiment, and would connect this line directly to [[Kostrikov2022Offline]].
