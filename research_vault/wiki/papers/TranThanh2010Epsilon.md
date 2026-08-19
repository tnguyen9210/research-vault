---
title: "Epsilon-First Policies for Budget-Limited Multi-Armed Bandits"
authors: [Long Tran-Thanh, Archie Chapman, Enrique Munoz de Cote, Alex Rogers, Nicholas R. Jennings]
year: 2010
venue: AAAI
tags: [budget-limited-mab, multi-armed-bandits, epsilon-first, knapsack, regret, bandits]
source: raw/papers/TranThanh2010Epsilon.pdf
---

# Epsilon-First Policies for Budget-Limited Multi-Armed Bandits

**TL;DR:** The paper that **introduces the [[budget-limited-mab]]** — arms have heterogeneous pull costs $c_i$ and a single budget $B$ caps exploration *and* exploitation — and identifies its full-information optimum as an **unbounded knapsack** problem on reward densities $\mu_i/c_i$. It solves the problem with an [[epsilon-first]] policy (uniform exploration on $\varepsilon B$, density-ordered greedy exploitation on $(1-\varepsilon)B$) and proves the first loss bound for it.

## Problem

Prior cost-aware bandit work budgeted only the *exploration* phase: MAB with pure exploration (Bubeck et al. 2009), budgeted MAB (Guha & Munagala 2007), max-loss value estimation (Antos et al. 2008). In all of these, once the best arm is identified, exploitation is free and unbounded, so the answer is still "find the best arm, pull it forever."

This paper removes that assumption. With one budget covering both phases, **the total number of pulls is finite**, and the optimal policy is no longer to repeatedly pull one arm but to pull the best *combination* of arms that fully consumes $B$. The motivating example is online ad placement: each banner site charges a different rental price, the click-through rates are unknown, and one budget covers both learning and spending.

The consequence the paper draws out is that standard MAB machinery does not transfer. UCB and $\varepsilon_n$-greedy concentrate on learning the single highest-reward arm, but here *every* arm's mean matters, because any of them may enter the optimal combination. Their worked example: arms with (reward, cost) pairs $\langle 9,4\rangle, \langle 1.5,1\rangle, \langle 2,1\rangle$ and $B = 15$. The optimum is three pulls each of $X$ and $Z$ for expected reward 33; slightly sloppy estimates of $Y$ and $Z$ (1.76 vs. 1.74) flip the choice to $X$ and $Y$ for 31.5. Precision on the *non-best* arms is what decides the combination.

## Method

### The knapsack reduction

With known means, the problem *is* an unbounded knapsack: item $i$ has value $\mu_i$ and weight $c_i$, capacity is $B$, and one chooses non-negative integer pull counts

$$
\max_{x_1,\dots,x_k}\ \sum_{i=1}^k x_i\mu_i \quad\text{s.t.}\quad \sum_{i=1}^k x_i c_i \le B .
$$

This is NP-hard, so the paper uses the **density-ordered greedy** approximation (Kohli, Krishnamurti & Mirchandani 2004), $O(k\log k)$: sort by density $\hat\mu_i/c_i$, take as many units of the densest feasible item as fit, repeat over at most $k$ rounds. The density $\mu_i/c_i$ is reward per unit of budget — the quantity that replaces the mean in this setting.

### The $\varepsilon$-first algorithm

Split $B$ into $\varepsilon B$ for exploration and $(1-\varepsilon)B$ for exploitation.

- **Exploration:** pull arms uniformly in sequence until $\varepsilon B$ is exhausted, so $n_i \ge \big\lfloor \varepsilon B / \sum_{j=1}^k c_j \big\rfloor$ for every arm.
- **Exploitation:** run density-ordered greedy on the estimates $\hat\mu_i$.

The stated reason for uniform exploration is analytic rather than empirical: because every arm is sampled equally often, estimate accuracy is a clean function of $\varepsilon$, which makes the loss bound tractable and gives a principled way to choose $\varepsilon$.

## Results

Loss is $L(A) = \mathbb{E}[R(A^*)] - \mathbb{E}[R(A)]$ against the full-information optimum $A^*$.

**Theorem 1 (any exploration policy).** For an arbitrary exploration sequence $A^\text{arb}$ that exhausts the exploration budget, with probability $(1-\delta)^k$,

$$
L(A) \le 2\varepsilon B\, D_{\max} + B\left(\sqrt{\frac{-\ln\delta}{n_{i^{\max}}}} + \sqrt{\frac{-\ln\delta}{n_{I^{\max}}}}\right),
$$

where $D_{\max} = \frac{\mu_{i^{\max}}}{c_{i^{\max}}} - \frac{\mu_{i^{\min}}}{c_{i^{\min}}}$ is the spread of true reward densities, $i^{\max}/i^{\min}$ are the true best/worst-density arms, and $I^{\max}$ is the estimated best-density arm. Proof by Hoeffding per arm, hence the $(1-\delta)^k$ confidence.

The authors are candid that this "is of no practical use, since it assumes knowledge of $i^{\max}$ and $i^{\min}$" — it exists to be specialized.

**Corollary 2 (uniform pull exploration).** Substituting $n_i \ge \varepsilon B/\sum_j c_j$:

$$
L(A_{\varepsilon\text{-first}}) \le 2\varepsilon B\, D_{\max} + 2B\sqrt{\frac{(-\ln\delta)\sum_{j=1}^k c_j}{\varepsilon B}} .
$$

**On the $O(B^{2/3})$ figure.** This paper never states it. Balancing the two terms of Corollary 2 gives $\varepsilon^* \propto B^{-1/3}$ and hence loss $O(B^{2/3})$, but that optimization — and the claim that $\varepsilon$-first is *provably stuck* at that rate — comes from [[TranThanh2012Knapsack]] in retrospect. What is here is the bound above, parameterized by $\varepsilon$ and $\delta$.

**Experiments.** Six-armed machines, Gaussian rewards (variance 0.1, support $[0,20]$), budgets 400–4000, three regimes: homogeneous costs, moderately diverse, extremely diverse. Compared against a cost-modified $\varepsilon_n$-greedy (pull the highest reward-cost-ratio arm w.p. $1-\varepsilon_n$).

| Regime | $\varepsilon$-first vs. $\varepsilon_n$-greedy |
|---|---|
| homogeneous | roughly tied |
| diverse | ~50% lower total loss |
| extremely diverse | ~80% lower total loss |

The tie in the homogeneous case is expected and honestly explained: when costs are similar, density-ordered greedy degenerates to pulling a single arm, so the algorithm reduces to something $\varepsilon_n$-greedy-like.

**A negative result worth keeping.** $\varepsilon$-first with *UCB* exploration performs about the same as with uniform exploration, despite UCB being the better pure explorer. Their reading: both identify the arms the greedy step actually needs. A single-run trace at $B=3500$ shows both tracking the optimal pull profile (arm 2 twelve times, arm 4 twice, arm 6 twenty-three times), while $\varepsilon_n$-greedy fixates on arms 2 and 5 and never samples arm 6 at all — because if it starts on the wrong arm it stays there, and the budget is too short to recover.

## Strengths & Limitations

**Strengths.** The contribution that lasted is the **problem formulation**, and it is a good one: a small change to the standard MAB (one budget, both phases) that genuinely breaks the standard solution concept and connects bandits to combinatorial optimization. The knapsack reduction is exactly right and is still the frame the whole line uses. The worked three-arm counterexample earns its space — it shows in four lines why single-best-arm exploration is the wrong objective. Theorem 1 is stated for *arbitrary* exploration, which is more than the algorithm needs. The experimental design isolates the mechanism (cost diversity) rather than just reporting wins, and reports the tie in the homogeneous case rather than hiding it.

**Limitations.** No lower bound, so nothing here says whether $\varepsilon$-first is close to optimal — as it turns out, it is not. The headline theorem is admitted to be unusable and only Corollary 2 is actionable. Confidence degrades as $(1-\delta)^k$ in the number of arms. The $\varepsilon$-first structure is the real limitation: separating the phases means the exploration budget yields no exploitation value and the estimates never improve during exploitation, which is precisely the slack [[kube]] later exploits by interleaving. Experiments are small (6 arms, one reward distribution family, three cost configurations). Finally, the paper only explores one exploration/exploitation pairing and says so, deferring other combinations to future work.

## Connections

- [[budget-limited-mab]] — **introduced here**, along with the unbounded-knapsack characterization of its optimum and the reward-density $\mu_i/c_i$ statistic
- [[epsilon-first]] — the policy family this paper proposes and bounds
- **Superseded by:** [[TranThanh2012Knapsack]] — same problem, same group; [[kube]] interleaves exploration and exploitation through a UCB-augmented knapsack and reaches the optimal $O(\ln B)$ with a matching lower bound
- [[upper-confidence-bound]] — used here only as an *exploration* baseline inside the $\varepsilon$-first shell, and shown not to help; the 2012 paper instead puts UCB inside the knapsack objective itself, which does help
- [[kube]] — the successor algorithm
- [[Tran-Thanh-Long]], [[Chapman-Archie]], [[Munoz-de-Cote-Enrique]], [[Rogers-Alex]], [[Jennings-Nicholas-R]] — authors

## Open Questions

- **The one the paper states:** refining the generic bound of Theorem 1 for non-uniform exploration (UCB, Boltzmann) is hard because $n_i$ cannot be estimated for those policies. New technique needed, or the analysis stays restricted to uniform sampling.
- Why does better exploration (UCB) *not* improve $\varepsilon$-first? The paper's explanation is plausible but tested on 6 arms in one setting. If it holds generally, it says something about how coarse the greedy step's information requirement is.
- What is the right $\varepsilon$ without knowing $D_{\max}$? Corollary 2's optimizer depends on the density spread, which is exactly what has not been learned yet.
- The $(1-\delta)^k$ confidence is loose for large $k$; a union-bound-free or self-normalized argument should improve it.
