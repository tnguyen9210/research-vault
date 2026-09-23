# Wiki Index

Master catalog of all pages. Updated on every ingest and query-save.  
Format: `- [[slug]] — one-line description`

---

## Papers

- [[cassel2026Quantile]] — VIBE: optimism from a quantile of an ensemble on disjoint batches, no bonuses or counts; variance-dependent minimax regret in tabular MDPs
- [[dam2024Power]] — Stochastic-Power-UCT: power mean MCTS with $`\mathcal{O}(n^{-1/2})`$ convergence in stochastic MDPs; fixes UCT's flawed logarithmic bonus
- [[foster2025Good]] — coverage is necessary and sufficient for computationally efficient LM alignment; SpannerSampling matches $`C_\text{cov}`$ lower bound; ETH-hardness of training-time interventions
- [[kanarios2024Cost]] — CABAI: cost-aware BAI with heterogeneous arm costs; optimal proportions scale $`\sqrt{c_a}`$; CTAS (optimal) and CO (fast)
- [[kostrikov2021Offline]] — IQL: offline RL with strictly in-sample value evaluation via upper-expectile regression; $`\tau\to1`$ recovers the support-constrained optimum; SOTA on D4RL antmaze (ICLR 2022)
- [[lardy2025Constrained]] — CBAI: BAI with cost-threshold constraint on bivariate arms; handles dependent reward-cost; asymptotically optimal TaS (NeurIPS 2025)
- [[li2026Predicting]] — SLG Search: tail-guided BoN scaling law prediction + adaptive two-stage compute allocation; polynomial amplification over BoN
- [[qin2026Taming]] — OE2D: first offline-oracle-efficient contextual bandit algorithm for general function classes with O(log T) calls; introduces DOEC
- [[ryu2025Improved]] — PUB: parameter-free variance-adaptive off-policy selection via betting-based LCB; freezing score function wins in small-data regimes (COLT 2025)
- [[simchi-levi2022Bypassing]] — FALCON: first optimal offline-oracle-efficient contextual bandit algorithm; O(log T) calls for discrete actions under realizability
- [[song2019Revisiting]] — S-DQN/S-DDQN: softmax Bellman operator reduces overestimation + gradient noise in DQNs; exponential convergence in $`\tau`$; outperforms DDQN on Atari (ICML 2019)
- [[tran-thanh2010Epsilon]] — introduces the budget-limited MAB + its unbounded-knapsack optimum; $`\varepsilon`$-first policy with the first loss bound (AAAI 2010)
- [[tran-thanh2012Knapsack]] — KUBE / fractional KUBE: first $`O(\ln B)`$ algorithms for the budget-limited MAB; full-info optimum = unbounded knapsack; matching lower bound (AAAI 2012)
- [[yang2025Stochastically]] — BFAI-TS: fixed-budget constrained BAI with $`m`$ constraints via Thompson sampling; asymptotically optimal exponential PFS decay (arXiv 2025)
- [[yin2023Offline]] — PFQL/VAFQL: first instance-dependent offline RL bound under nonlinear (differentiable) function approximation; minimax-optimal up to $`\sqrt{d}`$ (ICLR 2023)
- [[zanette2019Tighter]] — EULER: problem-dependent regret scaling with the environmental norm, with no domain knowledge; answers the Jiang–Agarwal horizon question

## Concepts
- [[contextual-bandits-offline]] — the batch contextual bandit: fixed logged data, no interaction; setup, coverage coefficients, and the value-based / policy-based taxonomy
- [[contextual-bandits-offline-value-based]] — fit $`\hat q`$ by regression, then act greedily or pessimistically; toolkit, greedy and LCB guarantees, Rashidinejad's Theorem 4 with proof
- [[contextual-bandits-online]] — the interactive protocol: side information each round, exploration against exploitation; sibling of [[contextual-bandits-offline]]
- [[coverage-coefficient]] — $`C_\text{cov}(\pi^*_\beta)`$: measures how well $`\pi_\text{ref}`$ covers the optimal policy; lower bounds sampling oracle calls in [[foster2025Good]]
- [[decision-estimation-coefficient]] — DEC; complexity measure for online-oracle-efficient contextual bandits (Foster et al. 2021a)
- [[decision-offline-estimation-coefficient]] — DOEC; complexity measure for offline-oracle-efficient bandits; introduced in [[qin2026Taming]]
- [[deep-q-network]] — DQN/DDQN: Q-learning with neural network, experience replay, target network; DDQN decouples action selection from evaluation to reduce overestimation
- [[differentiable-function-approximation]] — $`\mathcal{F}=\{f(\theta,\phi(\cdot,\cdot))\}`$ with $`f`$ thrice differentiable in $`\theta`$; generalizes tabular/linear/GLM; gradient geometry enables instance-dependent analysis
- [[eluder-dimension]] — how long a point can elude being determined by prior queries; bounds [[epsilon-sec]] and hence [[decision-offline-estimation-coefficient]]
- [[epsilon-sec]] — passive coverage measure upper-bounding [[decision-offline-estimation-coefficient]] (Thm 3 of [[qin2026Taming]]); can be exponentially loose vs. active design
- [[expectile-regression]] — asymmetric-$`\ell_2`$ regression estimating the $`\tau`$-expectile; $`\tau\to1`$ approaches the supremum, enabling in-sample maximization; used by [[kostrikov2021Offline]]
- [[exploitative-f-design]] — per-context minimax optimization simultaneously satisfying Low Regret and Good Coverage; core primitive of OE2D
- [[extrapolation-error]] — FQI trains $`Q`$ on the data distribution but queries it outside; the $`\max`$ then selects whichever unsupported action was overvalued
- [[fqi]] — the offline RL template: relabel a fixed batch with Bellman targets, refit by least squares, repeat; $`Q_{k+1}\approx\Pi_\mathcal{F}\mathcal{T}Q_k`$
- [[fqi-finite-sample-analysis]] — Jiang (2020) note, line by line: $`J(\pi^*)-J(\hat\pi)`$ bound via Bellman-error propagation + Bernstein fast rate $`O(n^{-1/2})`$
- [[fqi-pessimistic]] — PFQL/VAFQL: fitted Q-iteration plus a gradient-geometry uncertainty penalty, with Algorithm 1 of [[yin2023Offline]] read line by line
- [[implicit-q-learning]] — IQL: upper-expectile $`V`$ + MSE $`Q`$ backup + AWR extraction; multi-step DP without out-of-sample queries; $`\tau`$ interpolates SARSA to Q-learning
- [[importance-weighting]] — IW estimator for offline policy evaluation; variance control via pessimism and score functions; core primitive in off-policy learning
- [[instance-dependent-bounds]] — hub: guarantees driven by the individual instance (gaps, variance, gradient geometry, coverage) rather than a worst case over the class
- [[linear-softmax-policy]] — $`\pi_\theta(y|x) \propto \pi_\text{ref}(y|x)\exp(\beta^{-1}\langle\theta,\phi(x,y)\rangle)`$; natural RLHF parameterization studied in [[foster2025Good]]
- [[offline-regression-oracle]] — batch supervised learner used as oracle; standard ERM qualifies; reduces bandit learning to few oracle calls, enabling practical implementation
- [[offline-reinforcement-learning]] — policy learning from a fixed dataset, no interaction; distributional shift is the binding constraint; four algorithm families
- [[online-reinforcement-learning]] — interaction with an unknown MDP, judged by cumulative regret $`\mathrm{Reg}(T)=\sum_t\Delta(\pi_t)`$; optimism by bonus or by ensemble quantile; the mirror of the offline page
- [[oracle-efficiency]] — reducing bandit learning to few regression-oracle calls; the $`O(\log T)`$ call-count line from FALCON to OE2D, online vs. offline oracle
- [[overestimation-bias]] — systematic upward bias in Q-learning from the max operator; mitigated by DDQN, distributional RL, and the [[softmax-bellman-operator]]
- [[pessimism-principle]] — act on a lower confidence bound offline; same confidence machinery as UCB with the opposite sign, because offline errors are not self-correcting
- [[realizability]] — assumption f* ∈ F enabling FALCON's optimal offline-oracle-efficient guarantees; introduced in [[simchi-levi2022Bypassing]]
- [[smooth-aggregators]] — power-mean MCTS, softmax DQN and expectile IQL as one parameterized average$`\to`$max family; no unified analysis exists
- [[softmax-bellman-operator]] — replaces max in Bellman backup with softmax-weighted average at inverse temperature $`\tau`$; reduces overestimation; exponential convergence to $`\mathcal{T}`$; introduced in [[song2019Revisiting]]
- [[spanner-sampling]] — two-phase improper exploration algorithm achieving optimal $`T_\text{comp} = \tilde{O}(C_\text{cov})`$; introduced in [[foster2025Good]]
- [[upper-confidence-bound]] — optimism in the face of uncertainty; gap-dependent $`O(\sum_i \log T/\Delta_i)`$ regret; the online mirror of [[pessimism-principle]]

### Inactive

> Parked while the focus is contextual bandits and RL. Files live in
> `concepts/inactive/`; links resolve by basename, so every `[[slug]]`
> still works and nothing needed relinking.

- [[bai]] — fixed-confidence pure exploration; identify best arm with prob $`\geq 1-\delta`$ at minimum sample cost
- [[bai-constrained]] — CBAI: BAI with cost-threshold constraint $`\mathbb{E}[C_k] \leq \gamma`$; handles dependent reward-cost; introduced in [[lardy2025Constrained]]
- [[bai-cost-aware]] — cost-aware pure exploration; cost as objective (CABAI, proportions $`\propto \sqrt{c_a}`$), as constraint (CBAI), or under a fixed budget
- [[budget-limited-mab]] — bandit with per-arm cost $`c_i`$ and a single shared budget $`B`$; full-info optimum is an unbounded knapsack on densities $`\mu_i/c_i`$; introduced in [[tran-thanh2010Epsilon]]
- [[budget-limited-mab-epsilon-first]] — split the budget into $`\varepsilon`$ explore / $`1-\varepsilon`$ commit; the phase split alone caps performance at $`O(B^{2/3})`$
- [[budget-limited-mab-kube]] — Knapsack-based UCB Exploration; solves a UCB-augmented knapsack each step and samples by multiplicity; fractional variant = budget-limited UCB; introduced in [[tran-thanh2012Knapsack]]
- [[mcts]] — online planning via bandit-guided tree simulation; UCT and successors
- [[mcts-power-mean]] — power mean backup operator for MCTS; $`p=2`$ optimal; $`\mathcal{O}(n^{-1/2})`$ convergence; introduced in [[dam2024Power]]
- [[test-time-scaling]] — LLM inference-time compute scaling; Best-of-$`N`$ and adaptive alternatives
- [[test-time-scaling-slg-search]] — Scaling-Law Guided Search; two-stage adaptive test-time compute; polynomial amplification over BoN; introduced in [[li2026Predicting]]
## Queries

- [[2026-06-16-foster2025-sections-1-4]] — section-by-section summary of Foster2025Foundation Sections 1–4: setup, coverage lower bound, SpannerSampling, hardness of proper exploration
- [[2026-08-19-offline-fqi-walkthrough]] — reading path from the FQI template through offline-vs-online, extrapolation error, and Algorithm 1 of [[yin2023Offline]]
- [[2026-09-16-offline-cb-value-based]] — reading path for the value-based offline contextual bandit account; the answer now lives in [[contextual-bandits-offline]] and [[contextual-bandits-offline-value-based]]

- [[2026-09-22-qom-value-based-pessimism]] — Quantile of Means as tabular value-based pessimism: Theorem 4.2-style variance-adaptive bound without a penalty, its constant price, lost $`(C^{\ast}-1)`$ adaptivity, and a [[cassel2026Quantile]] erratum
- [[2026-09-22-qom-value-based-pessimism-subgaussian]] — QoM-LCB for $`\sigma^2`$-sub-Gaussian rewards: Theorem 4.2-style bound, $`B=3.2\ln(2SK/\delta)`$ under symmetric noise; sub-Gaussianity alone is not enough; extends to linear classes via signed-weight symmetry
