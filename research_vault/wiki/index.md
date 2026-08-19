# Wiki Index

Master catalog of all pages. Updated on every ingest and query-save.  
Format: `- [[slug]] — one-line description`

---

## Papers

- [[Dam2024Power]] — Stochastic-Power-UCT: power mean MCTS with $\mathcal{O}(n^{-1/2})$ convergence in stochastic MDPs; fixes UCT's flawed logarithmic bonus
- [[Foster2025Foundation]] — coverage is necessary and sufficient for computationally efficient LM alignment; SpannerSampling matches $C_\text{cov}$ lower bound; ETH-hardness of training-time interventions
- [[Ryu2025Improved]] — PUB: parameter-free variance-adaptive off-policy selection via betting-based LCB; freezing score function wins in small-data regimes (COLT 2025)
- [[Li2026Predicting]] — SLG Search: tail-guided BoN scaling law prediction + adaptive two-stage compute allocation; polynomial amplification over BoN
- [[Kanarios2024Cost]] — CABAI: cost-aware BAI with heterogeneous arm costs; optimal proportions scale $\sqrt{c_a}$; CTAS (optimal) and CO (fast)
- [[Kostrikov2022Offline]] — IQL: offline RL with strictly in-sample value evaluation via upper-expectile regression; $\tau\to1$ recovers the support-constrained optimum; SOTA on D4RL antmaze (ICLR 2022)
- [[Lardy2025Constrained]] — CBAI: BAI with cost-threshold constraint on bivariate arms; handles dependent reward-cost; asymptotically optimal TaS (NeurIPS 2025)
- [[Yang2025Stochastically]] — BFAI-TS: fixed-budget constrained BAI with $m$ constraints via Thompson sampling; asymptotically optimal exponential PFS decay (arXiv 2025)
- [[Qin2026Taming]] — OE2D: first offline-oracle-efficient contextual bandit algorithm for general function classes with O(log T) calls; introduces DOEC
- [[SimchiLevi2022Bypassing]] — FALCON: first optimal offline-oracle-efficient contextual bandit algorithm; O(log T) calls for discrete actions under realizability
- [[TranThanh2012Knapsack]] — KUBE / fractional KUBE: first $O(\ln B)$ algorithms for the budget-limited MAB; full-info optimum = unbounded knapsack; matching lower bound (AAAI 2012)
- [[TranThanh2010Epsilon]] — introduces the budget-limited MAB + its unbounded-knapsack optimum; $\varepsilon$-first policy with the first loss bound (AAAI 2010)
- [[Song2019Revisiting]] — S-DQN/S-DDQN: softmax Bellman operator reduces overestimation + gradient noise in DQNs; exponential convergence in $\tau$; outperforms DDQN on Atari (ICML 2019)
- [[Yin2023Offline]] — PFQL/VAFQL: first instance-dependent offline RL bound under nonlinear (differentiable) function approximation; minimax-optimal up to $\sqrt{d}$ (ICLR 2023)

## Concepts

*FQI-family pages live in `concepts/fqi/`; links are still plain `[[slug]]`.*

- [[best-arm-identification]] — fixed-confidence pure exploration; identify best arm with prob $\geq 1-\delta$ at minimum sample cost
- [[budget-limited-mab]] — bandit with per-arm cost $c_i$ and a single shared budget $B$; full-info optimum is an unbounded knapsack on densities $\mu_i/c_i$; introduced in [[TranThanh2010Epsilon]]
- [[constrained-bai]] — CBAI: BAI with cost-threshold constraint $\mathbb{E}[C_k] \leq \gamma$; handles dependent reward-cost; introduced in [[Lardy2025Constrained]]
- [[deep-q-network]] — DQN/DDQN: Q-learning with neural network, experience replay, target network; DDQN decouples action selection from evaluation to reduce overestimation
- [[differentiable-function-approximation]] — $\mathcal{F}=\{f(\theta,\phi(\cdot,\cdot))\}$ with $f$ thrice differentiable in $\theta$; generalizes tabular/linear/GLM; gradient geometry enables instance-dependent analysis
- [[expectile-regression]] — asymmetric-$\ell_2$ regression estimating the $\tau$-expectile; $\tau\to1$ approaches the supremum, enabling in-sample maximization; used by [[Kostrikov2022Offline]]
- [[implicit-q-learning]] — IQL: upper-expectile $V$ + MSE $Q$ backup + AWR extraction; multi-step DP without out-of-sample queries; $\tau$ interpolates SARSA to Q-learning
- [[instance-dependent-bounds]] — hub: guarantees driven by the individual instance (gaps, variance, gradient geometry, coverage) rather than a worst case over the class
- [[importance-weighting]] — IW estimator for offline policy evaluation; variance control via pessimism and score functions; core primitive in off-policy learning
- [[kube]] — Knapsack-based UCB Exploration; solves a UCB-augmented knapsack each step and samples by multiplicity; fractional variant = budget-limited UCB; introduced in [[TranThanh2012Knapsack]]
- [[overestimation-bias]] — systematic upward bias in Q-learning from the max operator; mitigated by DDQN, distributional RL, and the [[softmax-bellman-operator]]
- [[pessimism-principle]] — act on a lower confidence bound offline; same confidence machinery as UCB with the opposite sign, because offline errors are not self-correcting
- [[pessimistic-fitted-q-learning]] — PFQL/VAFQL: fitted Q-iteration plus a gradient-geometry uncertainty penalty $\beta\|\nabla_\theta f\|_{\Sigma_h^{-1}}$; introduced in [[Yin2023Offline]]
- [[softmax-bellman-operator]] — replaces max in Bellman backup with softmax-weighted average at inverse temperature $\tau$; reduces overestimation; exponential convergence to $\mathcal{T}$; introduced in [[Song2019Revisiting]]
- [[cabai]] — Cost Aware BAI; minimize cumulative testing cost; optimal arm proportions $\propto \sqrt{c_a}$; introduced in [[Kanarios2024Cost]]
- [[contextual-bandits]] — sequential decision-making with side information; minimize regret vs. best per-context action
- [[coverage-coefficient]] — $C_\text{cov}(\pi^*_\beta)$: measures how well $\pi_\text{ref}$ covers the optimal policy; lower bounds sampling oracle calls in [[Foster2025Foundation]]
- [[linear-softmax-policy]] — $\pi_\theta(y|x) \propto \pi_\text{ref}(y|x)\exp(\beta^{-1}\langle\theta,\phi(x,y)\rangle)$; natural RLHF parameterization studied in [[Foster2025Foundation]]
- [[spanner-sampling]] — two-phase improper exploration algorithm achieving optimal $T_\text{comp} = \tilde{O}(C_\text{cov})$; introduced in [[Foster2025Foundation]]
- [[dec]] — Decision Estimation Coefficient; complexity measure for online-oracle-efficient contextual bandits (Foster et al. 2021a)
- [[doec]] — Decision-Offline Estimation Coefficient; complexity measure for offline-oracle-efficient bandits; introduced in [[Qin2026Taming]]
- [[eluder-dimension]] — how long a point can elude being determined by prior queries; bounds [[epsilon-sec]] and hence [[doec]]
- [[epsilon-first]] — split the budget into $\varepsilon$ explore / $1-\varepsilon$ commit; the phase split alone caps performance at $O(B^{2/3})$
- [[epsilon-sec]] — passive coverage measure upper-bounding [[doec]] (Thm 3 of [[Qin2026Taming]]); can be exponentially loose vs. active design
- [[exploitative-f-design]] — per-context minimax optimization simultaneously satisfying Low Regret and Good Coverage; core primitive of OE2D
- [[fitted-q-iteration]] — the offline RL template: relabel a fixed batch with Bellman targets, refit by least squares, repeat; $Q_{k+1}\approx\Pi_\mathcal{F}\mathcal{T}Q_k$
- [[offline-regression-oracle]] — batch supervised learner used as oracle; standard ERM qualifies; reduces bandit learning to few oracle calls, enabling practical implementation
- [[monte-carlo-tree-search]] — online planning via bandit-guided tree simulation; UCT and successors
- [[slg-search]] — Scaling-Law Guided Search; two-stage adaptive test-time compute; polynomial amplification over BoN; introduced in [[Li2026Predicting]]
- [[test-time-scaling]] — LLM inference-time compute scaling; Best-of-$N$ and adaptive alternatives
- [[upper-confidence-bound]] — optimism in the face of uncertainty; gap-dependent $O(\sum_i \log T/\Delta_i)$ regret; the online mirror of [[pessimism-principle]]
- [[power-mean-mcts]] — power mean backup operator for MCTS; $p=2$ optimal; $\mathcal{O}(n^{-1/2})$ convergence; introduced in [[Dam2024Power]]
- [[realizability]] — assumption f* ∈ F enabling FALCON's optimal offline-oracle-efficient guarantees; introduced in [[SimchiLevi2022Bypassing]]

## Authors

- [[Dam-Tuan]] — first author of Stochastic-Power-UCT; Univ. Lille / Inria
- [[Carin-Lawrence]] — third author of [[Song2019Revisiting]]; Duke; Bayesian ML (peripheral to the RL content)
- [[Chapman-Archie]] — co-author of both budget-limited MAB papers; Southampton; multi-agent systems, distributed optimization
- [[Foster-Dylan-J]] — first author of [[Foster2025Foundation]]; Microsoft Research; online learning, LM alignment
- [[Mhammedi-Zakaria]] — co-author of [[Foster2025Foundation]]; Google Research; RL theory, sampling oracle framework
- [[Rohatgi-Dhruv]] — co-author of [[Foster2025Foundation]]; MIT; computational hardness of proper exploration
- [[Jennings-Nicholas-R]] — senior author of KUBE / budget-limited MAB; Loughborough (then Southampton); multi-agent systems
- [[Jun-Kwang-Sung]] — senior author of PUB/freezing paper; U. Arizona; betting-based confidence bounds
- [[Koolen-Wouter-M]] — senior author of CBAI; CWI/Twente; mixture martingales, pure exploration
- [[Kostrikov-Ilya]] — first author of [[Kostrikov2022Offline]] (IQL); UC Berkeley; offline RL, efficient RL implementations
- [[Lardy-Tyron]] — first author of CBAI; CWI/Leiden; BAI with cost-threshold constraints
- [[Levine-Sergey]] — senior author of [[Kostrikov2022Offline]]; UC Berkeley; offline RL, robot learning
- [[Ryu-J-Jon]] — first author of PUB/freezing paper; MIT; betting-based confidence bounds
- [[Li-Muheng]] — first author of SLG Search; U. Toronto
- [[Mou-Wenlong]] — senior author of SLG Search; U. Toronto
- [[Qian-Jian]] — second author of SLG Search; U. Hong Kong
- [[Kanarios-Kellen]] — first author of CABAI; U. Michigan
- [[Kaufmann-Emilie]] — third author of Stochastic-Power-UCT; also co-developer of TAS (BAI); Univ. Lille / Inria
- [[Maillard-Odalric-Ambrym]] — second author of Stochastic-Power-UCT; Univ. Lille / Inria
- [[Munoz-de-Cote-Enrique]] — co-author of [[TranThanh2010Epsilon]]; Southampton; multi-agent learning, RL
- [[Wang-Mengdi]] — co-author of [[Yin2023Offline]]; Princeton; RL theory, sample complexity with function approximation
- [[Nair-Ashvin]] — co-author of [[Kostrikov2022Offline]]; UC Berkeley; offline-to-online finetuning, AWAC
- [[Parr-Ronald-E]] — co-author of softmax Bellman operator paper; Duke; RL theory, MDPs
- [[Qin-Hao]] — first author of OE2D; U. Arizona
- [[Rogers-Alex]] — co-author of KUBE / budget-limited MAB; Oxford (then Southampton); MAS, sensor networks
- [[Song-Zhao]] — first author of softmax Bellman operator paper; Baidu Research (Duke PhD); deep RL theory
- [[Simchi-Levi-David]] — first author of FALCON; MIT
- [[Tran-Thanh-Long]] — first author of KUBE / budget-limited MAB; Warwick (then Southampton); bandits under resource budgets
- [[Xu-Yunzong]] — second author of FALCON; MIT
- [[Yang-Le]] — first author of BFAI-TS; City University of Hong Kong; constrained BAI, Thompson sampling
- [[Wang-Yi]] — senior author of BFAI-TS; University of Hong Kong; simulation optimization
- [[Wang-Yu-Xiang]] — senior author of [[Yin2023Offline]]; UC Santa Barbara; offline RL theory, statistical learning
- [[Yin-Ming]] — first author of [[Yin2023Offline]]; UC Santa Barbara; instance-dependent offline RL sample complexity
- [[Ying-Lei]] — senior author of CABAI; U. Michigan
- [[Zhang-Chicheng]] — second author of OE2D; U. Arizona; vault owner's advisor
- [[Zhang-Qining]] — second author of CABAI; U. Michigan

## Topics

- [[best-arm-identification]] — synthesis of BAI and cost-aware BAI; CABAI open problems
- [[budget-limited-bandits]] — synthesis of budget/cost-constrained bandits; cumulative-reward (KUBE) vs. cost-aware BAI
- [[monte-carlo-tree-search]] — synthesis of MCTS theory; Stochastic-Power-UCT and open problems
- [[offline-oracle-efficient-bandits]] — synthesis of the research line on offline-oracle contextual bandits; FALCON → OE2D
- [[offline-reinforcement-learning]] — synthesis of offline RL; in-sample vs. constrained vs. regularized, single-step vs. multi-step stitching
- [[test-time-scaling]] — synthesis of LLM test-time compute scaling; SLG Search and open problems

## Queries

- [[2026-06-16-foster2025-sections-1-4]] — section-by-section summary of Foster2025Foundation Sections 1–4: setup, coverage lower bound, SpannerSampling, hardness of proper exploration
