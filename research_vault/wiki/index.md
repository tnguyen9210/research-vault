# Wiki Index

Master catalog of all pages. Updated on every ingest and query-save.  
Format: `- [[slug]] — one-line description`

---

## Papers

- [[Dam2024Power]] — Stochastic-Power-UCT: power mean MCTS with $\mathcal{O}(n^{-1/2})$ convergence in stochastic MDPs; fixes UCT's flawed logarithmic bonus
- [[Foster2025Foundation]] — coverage is necessary and sufficient for computationally efficient LM alignment; SpannerSampling matches $C_\text{cov}$ lower bound; ETH-hardness of training-time interventions
- [[Kanarios2024Cost]] — CABAI: cost-aware BAI with heterogeneous arm costs; optimal proportions scale $\sqrt{c_a}$; CTAS (optimal) and CO (fast)
- [[Kostrikov2022Offline]] — IQL: offline RL with strictly in-sample value evaluation via upper-expectile regression; $\tau\to1$ recovers the support-constrained optimum; SOTA on D4RL antmaze (ICLR 2022)
- [[Lardy2025Constrained]] — CBAI: BAI with cost-threshold constraint on bivariate arms; handles dependent reward-cost; asymptotically optimal TaS (NeurIPS 2025)
- [[Li2026Predicting]] — SLG Search: tail-guided BoN scaling law prediction + adaptive two-stage compute allocation; polynomial amplification over BoN
- [[Qin2026Taming]] — OE2D: first offline-oracle-efficient contextual bandit algorithm for general function classes with O(log T) calls; introduces DOEC
- [[Ryu2025Improved]] — PUB: parameter-free variance-adaptive off-policy selection via betting-based LCB; freezing score function wins in small-data regimes (COLT 2025)
- [[SimchiLevi2022Bypassing]] — FALCON: first optimal offline-oracle-efficient contextual bandit algorithm; O(log T) calls for discrete actions under realizability
- [[Song2019Revisiting]] — S-DQN/S-DDQN: softmax Bellman operator reduces overestimation + gradient noise in DQNs; exponential convergence in $\tau$; outperforms DDQN on Atari (ICML 2019)
- [[TranThanh2010Epsilon]] — introduces the budget-limited MAB + its unbounded-knapsack optimum; $\varepsilon$-first policy with the first loss bound (AAAI 2010)
- [[TranThanh2012Knapsack]] — KUBE / fractional KUBE: first $O(\ln B)$ algorithms for the budget-limited MAB; full-info optimum = unbounded knapsack; matching lower bound (AAAI 2012)
- [[Yang2025Stochastically]] — BFAI-TS: fixed-budget constrained BAI with $m$ constraints via Thompson sampling; asymptotically optimal exponential PFS decay (arXiv 2025)
- [[Yin2023Offline]] — PFQL/VAFQL: first instance-dependent offline RL bound under nonlinear (differentiable) function approximation; minimax-optimal up to $\sqrt{d}$ (ICLR 2023)

## Concepts

- [[best-arm-identification]] — fixed-confidence pure exploration; identify best arm with prob $\geq 1-\delta$ at minimum sample cost
- [[budget-limited-mab]] — bandit with per-arm cost $c_i$ and a single shared budget $B$; full-info optimum is an unbounded knapsack on densities $\mu_i/c_i$; introduced in [[TranThanh2010Epsilon]]
- [[cabai]] — Cost Aware BAI; minimize cumulative testing cost; optimal arm proportions $\propto \sqrt{c_a}$; introduced in [[Kanarios2024Cost]]
- [[constrained-bai]] — CBAI: BAI with cost-threshold constraint $\mathbb{E}[C_k] \leq \gamma$; handles dependent reward-cost; introduced in [[Lardy2025Constrained]]
- [[contextual-bandits]] — sequential decision-making with side information; minimize regret vs. best per-context action
- [[coverage-coefficient]] — $C_\text{cov}(\pi^*_\beta)$: measures how well $\pi_\text{ref}$ covers the optimal policy; lower bounds sampling oracle calls in [[Foster2025Foundation]]
- [[decision-estimation-coefficient]] — DEC; complexity measure for online-oracle-efficient contextual bandits (Foster et al. 2021a)
- [[decision-offline-estimation-coefficient]] — DOEC; complexity measure for offline-oracle-efficient bandits; introduced in [[Qin2026Taming]]
- [[deep-q-network]] — DQN/DDQN: Q-learning with neural network, experience replay, target network; DDQN decouples action selection from evaluation to reduce overestimation
- [[differentiable-function-approximation]] — $\mathcal{F}=\{f(\theta,\phi(\cdot,\cdot))\}$ with $f$ thrice differentiable in $\theta$; generalizes tabular/linear/GLM; gradient geometry enables instance-dependent analysis
- [[eluder-dimension]] — how long a point can elude being determined by prior queries; bounds [[epsilon-sec]] and hence [[decision-offline-estimation-coefficient]]
- [[epsilon-first]] — split the budget into $\varepsilon$ explore / $1-\varepsilon$ commit; the phase split alone caps performance at $O(B^{2/3})$
- [[epsilon-sec]] — passive coverage measure upper-bounding [[decision-offline-estimation-coefficient]] (Thm 3 of [[Qin2026Taming]]); can be exponentially loose vs. active design
- [[expectile-regression]] — asymmetric-$\ell_2$ regression estimating the $\tau$-expectile; $\tau\to1$ approaches the supremum, enabling in-sample maximization; used by [[Kostrikov2022Offline]]
- [[exploitative-f-design]] — per-context minimax optimization simultaneously satisfying Low Regret and Good Coverage; core primitive of OE2D
- [[extrapolation-error]] — FQI trains $Q$ on the data distribution but queries it outside; the $\max$ then selects whichever unsupported action was overvalued
- [[fitted-q-iteration]] — the offline RL template: relabel a fixed batch with Bellman targets, refit by least squares, repeat; $Q_{k+1}\approx\Pi_\mathcal{F}\mathcal{T}Q_k$
- [[fqi-finite-sample-analysis]] — Jiang (2020) note, line by line: $J(\pi^*)-J(\hat\pi)$ bound via Bellman-error propagation + Bernstein fast rate $O(n^{-1/2})$
- [[implicit-q-learning]] — IQL: upper-expectile $V$ + MSE $Q$ backup + AWR extraction; multi-step DP without out-of-sample queries; $\tau$ interpolates SARSA to Q-learning
- [[importance-weighting]] — IW estimator for offline policy evaluation; variance control via pessimism and score functions; core primitive in off-policy learning
- [[instance-dependent-bounds]] — hub: guarantees driven by the individual instance (gaps, variance, gradient geometry, coverage) rather than a worst case over the class
- [[kube]] — Knapsack-based UCB Exploration; solves a UCB-augmented knapsack each step and samples by multiplicity; fractional variant = budget-limited UCB; introduced in [[TranThanh2012Knapsack]]
- [[linear-softmax-policy]] — $\pi_\theta(y|x) \propto \pi_\text{ref}(y|x)\exp(\beta^{-1}\langle\theta,\phi(x,y)\rangle)$; natural RLHF parameterization studied in [[Foster2025Foundation]]
- [[monte-carlo-tree-search]] — online planning via bandit-guided tree simulation; UCT and successors
- [[offline-contextual-bandits]] — the batch contextual bandit: fixed logged data, no interaction; setup, coverage coefficients, and the value-based / policy-based taxonomy
- [[offline-regression-oracle]] — batch supervised learner used as oracle; standard ERM qualifies; reduces bandit learning to few oracle calls, enabling practical implementation
- [[oracle-efficiency]] — reducing bandit learning to a few calls to a regression oracle; contrasts the online and [[offline-regression-oracle]] variants
- [[overestimation-bias]] — systematic upward bias in Q-learning from the max operator; mitigated by DDQN, distributional RL, and the [[softmax-bellman-operator]]
- [[pessimism-principle]] — act on a lower confidence bound offline; same confidence machinery as UCB with the opposite sign, because offline errors are not self-correcting
- [[pessimistic-fitted-q-learning]] — PFQL/VAFQL: fitted Q-iteration plus a gradient-geometry uncertainty penalty $\beta\|\nabla_\theta f\|_{\Sigma_h^{-1}}$; introduced in [[Yin2023Offline]]
- [[pfql-algorithm-1]] — line-by-line walkthrough of Algorithm 1 of [[Yin2023Offline]]: each line's role, hyperparameter conditions, specializations, and why pessimism sits inside the backup
- [[power-mean-mcts]] — power mean backup operator for MCTS; $p=2$ optimal; $\mathcal{O}(n^{-1/2})$ convergence; introduced in [[Dam2024Power]]
- [[realizability]] — assumption f* ∈ F enabling FALCON's optimal offline-oracle-efficient guarantees; introduced in [[SimchiLevi2022Bypassing]]
- [[slg-search]] — Scaling-Law Guided Search; two-stage adaptive test-time compute; polynomial amplification over BoN; introduced in [[Li2026Predicting]]
- [[softmax-bellman-operator]] — replaces max in Bellman backup with softmax-weighted average at inverse temperature $\tau$; reduces overestimation; exponential convergence to $\mathcal{T}$; introduced in [[Song2019Revisiting]]
- [[spanner-sampling]] — two-phase improper exploration algorithm achieving optimal $T_\text{comp} = \tilde{O}(C_\text{cov})$; introduced in [[Foster2025Foundation]]
- [[test-time-scaling]] — LLM inference-time compute scaling; Best-of-$N$ and adaptive alternatives
- [[upper-confidence-bound]] — optimism in the face of uncertainty; gap-dependent $O(\sum_i \log T/\Delta_i)$ regret; the online mirror of [[pessimism-principle]]
- [[value-based-offline-bandits]] — fit $\hat q$ by regression, then act greedily or pessimistically; toolkit, greedy and LCB guarantees, Rashidinejad's Theorem 4 with proof

## Topics

- [[budget-limited-bandits]] — synthesis of budget/cost-constrained bandits; cumulative-reward (KUBE) vs. cost-aware BAI
- [[cost-aware-bai]] — the cost-aware BAI literature: cost as objective (CABAI), constraint (CBAI), fixed budget (BFAI); CABAI/CBAI positioning table
- [[offline-oracle-efficient-bandits]] — synthesis of the research line on offline-oracle contextual bandits; FALCON → OE2D
- [[offline-reinforcement-learning]] — synthesis of offline RL; in-sample vs. constrained vs. regularized, single-step vs. multi-step stitching
- [[smooth-aggregators]] — cross-cutting pattern: power-mean MCTS, softmax DQN, expectile IQL as one parameterized avg→max family; unified analysis open

## Queries

- [[2026-06-16-foster2025-sections-1-4]] — section-by-section summary of Foster2025Foundation Sections 1–4: setup, coverage lower bound, SpannerSampling, hardness of proper exploration
- [[2026-08-19-offline-fqi-walkthrough]] — reading path from the FQI template through offline-vs-online, extrapolation error, and Algorithm 1 of [[Yin2023Offline]]
- [[2026-09-16-offline-cb-value-based]] — reading path for the value-based offline contextual bandit account; the answer now lives in [[offline-contextual-bandits]] and [[value-based-offline-bandits]]
