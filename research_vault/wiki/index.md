# Wiki Index

Master catalog of all pages. Updated on every ingest and query-save.  
Format: `- [[slug]] — one-line description`

---

## Papers

- [[Dam2024Power]] — Stochastic-Power-UCT: power mean MCTS with $\mathcal{O}(n^{-1/2})$ convergence in stochastic MDPs; fixes UCT's flawed logarithmic bonus
- [[Ryu2025Improved]] — PUB: parameter-free variance-adaptive off-policy selection via betting-based LCB; freezing score function wins in small-data regimes (COLT 2025)
- [[Li2026Predicting]] — SLG Search: tail-guided BoN scaling law prediction + adaptive two-stage compute allocation; polynomial amplification over BoN
- [[Kanarios2024Cost]] — CABAI: cost-aware BAI with heterogeneous arm costs; optimal proportions scale $\sqrt{c_a}$; CTAS (optimal) and CO (fast)
- [[Lardy2025Constrained]] — CBAI: BAI with cost-threshold constraint on bivariate arms; handles dependent reward-cost; asymptotically optimal TaS (NeurIPS 2025)
- [[Yang2025Stochastically]] — BFAI-TS: fixed-budget constrained BAI with $m$ constraints via Thompson sampling; asymptotically optimal exponential PFS decay (arXiv 2025)
- [[Qin2026Taming]] — OE2D: first offline-oracle-efficient contextual bandit algorithm for general function classes with O(log T) calls; introduces DOEC
- [[SimchiLevi2022Bypassing]] — FALCON: first optimal offline-oracle-efficient contextual bandit algorithm; O(log T) calls for discrete actions under realizability
- [[TranThanh2012Knapsack]] — KUBE / fractional KUBE: first $O(\ln B)$ algorithms for the budget-limited MAB; full-info optimum = unbounded knapsack; matching lower bound (AAAI 2012)

## Concepts

- [[best-arm-identification]] — fixed-confidence pure exploration; identify best arm with prob $\geq 1-\delta$ at minimum sample cost
- [[budget-limited-mab]] — bandit with per-arm cost $c_i$ and a single shared budget $B$; full-info optimum is an unbounded knapsack on densities $\mu_i/c_i$; introduced in [[TranThanh2012Knapsack]]
- [[constrained-bai]] — CBAI: BAI with cost-threshold constraint $\mathbb{E}[C_k] \leq \gamma$; handles dependent reward-cost; introduced in [[Lardy2025Constrained]]
- [[importance-weighting]] — IW estimator for offline policy evaluation; variance control via pessimism and score functions; core primitive in off-policy learning
- [[kube]] — Knapsack-based UCB Exploration; solves a UCB-augmented knapsack each step and samples by multiplicity; fractional variant = budget-limited UCB; introduced in [[TranThanh2012Knapsack]]
- [[cabai]] — Cost Aware BAI; minimize cumulative testing cost; optimal arm proportions $\propto \sqrt{c_a}$; introduced in [[Kanarios2024Cost]]
- [[contextual-bandits]] — sequential decision-making with side information; minimize regret vs. best per-context action
- [[dec]] — Decision Estimation Coefficient; complexity measure for online-oracle-efficient contextual bandits (Foster et al. 2021a)
- [[doec]] — Decision-Offline Estimation Coefficient; complexity measure for offline-oracle-efficient bandits; introduced in [[Qin2026Taming]]
- [[exploitative-f-design]] — per-context minimax optimization simultaneously satisfying Low Regret and Good Coverage; core primitive of OE2D
- [[offline-regression-oracle]] — batch supervised learner used as oracle; standard ERM qualifies
- [[monte-carlo-tree-search]] — online planning via bandit-guided tree simulation; UCT and successors
- [[slg-search]] — Scaling-Law Guided Search; two-stage adaptive test-time compute; polynomial amplification over BoN; introduced in [[Li2026Predicting]]
- [[test-time-scaling]] — LLM inference-time compute scaling; Best-of-N and adaptive alternatives — reducing bandit learning to few calls to a regression oracle; enables practical implementation
- [[power-mean-mcts]] — power mean backup operator for MCTS; $p=2$ optimal; $\mathcal{O}(n^{-1/2})$ convergence; introduced in [[Dam2024Power]]
- [[realizability]] — assumption f* ∈ F enabling FALCON's optimal offline-oracle-efficient guarantees; introduced in [[SimchiLevi2022Bypassing]]

## Authors

- [[Dam-Tuan]] — first author of Stochastic-Power-UCT; Univ. Lille / Inria
- [[Jennings-Nicholas-R]] — senior author of KUBE / budget-limited MAB; Loughborough (then Southampton); multi-agent systems
- [[Jun-Kwang-Sung]] — senior author of PUB/freezing paper; U. Arizona; betting-based confidence bounds
- [[Koolen-Wouter-M]] — senior author of CBAI; CWI/Twente; mixture martingales, pure exploration
- [[Lardy-Tyron]] — first author of CBAI; CWI/Leiden; BAI with cost-threshold constraints
- [[Ryu-J-Jon]] — first author of PUB/freezing paper; MIT; betting-based confidence bounds
- [[Li-Muheng]] — first author of SLG Search; U. Toronto
- [[Mou-Wenlong]] — senior author of SLG Search; U. Toronto
- [[Qian-Jian]] — second author of SLG Search; U. Hong Kong
- [[Kanarios-Kellen]] — first author of CABAI; U. Michigan
- [[Kaufmann-Emilie]] — third author of Stochastic-Power-UCT; also co-developer of TAS (BAI); Univ. Lille / Inria
- [[Maillard-Odalric-Ambrym]] — second author of Stochastic-Power-UCT; Univ. Lille / Inria
- [[Qin-Hao]] — first author of OE2D; U. Arizona
- [[Rogers-Alex]] — co-author of KUBE / budget-limited MAB; Oxford (then Southampton); MAS, sensor networks
- [[Simchi-Levi-David]] — first author of FALCON; MIT
- [[Tran-Thanh-Long]] — first author of KUBE / budget-limited MAB; Warwick (then Southampton); bandits under resource budgets
- [[Xu-Yunzong]] — second author of FALCON; MIT
- [[Yang-Le]] — first author of BFAI-TS; City University of Hong Kong; constrained BAI, Thompson sampling
- [[Wang-Yi]] — senior author of BFAI-TS; University of Hong Kong; simulation optimization
- [[Ying-Lei]] — senior author of CABAI; U. Michigan
- [[Zhang-Chicheng]] — second author of OE2D; U. Arizona; vault owner's advisor
- [[Zhang-Qining]] — second author of CABAI; U. Michigan

## Topics

- [[best-arm-identification]] — synthesis of BAI and cost-aware BAI; CABAI open problems
- [[budget-limited-bandits]] — synthesis of budget/cost-constrained bandits; cumulative-reward (KUBE) vs. cost-aware BAI
- [[monte-carlo-tree-search]] — synthesis of MCTS theory; Stochastic-Power-UCT and open problems
- [[offline-oracle-efficient-bandits]] — synthesis of the research line on offline-oracle contextual bandits; FALCON → OE2D
- [[test-time-scaling]] — synthesis of LLM test-time compute scaling; SLG Search and open problems — synthesis of the research line on offline-oracle contextual bandits; FALCON → OE2D

## Queries

*(empty — saved analyses filed here)*
