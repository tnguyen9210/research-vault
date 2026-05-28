---
title: "Best Arm Identification"
tags: [bandits, pure-exploration]
---

# Best Arm Identification

## Overview
Best arm identification (BAI) is the fixed-confidence pure exploration paradigm: run sequential trials on K arms and stop when confident the best arm has been identified (with probability ≥ 1−δ), at minimum expected sample cost. Unlike regret minimization, all pulls serve exploration and the algorithm terminates. The theoretical foundation is mature: Kaufmann et al. (2016) established matching lower bounds, and TAS achieves them asymptotically.

The vault currently covers a cost-aware generalization ([[cabai]]) where each arm has a heterogeneous testing cost, motivating a fundamentally different optimal allocation rule (√c_a vs. uniform).

## Key Papers

- **[[Kanarios2024Cost]]** (2024) — introduces CABAI: BAI where each arm has a cost distribution; lower bound shows optimal arm proportions scale with √c_a; proposes CTAS (asymptotically optimal via bilevel optimization) and CO (low-complexity, model-free, near-optimal empirically)

## Open Problems

- Can CO's asymptotic optimality be established for K > 2 arms or general exponential families?
- Cost-aware BAI in the regret (non-stopping) setting
- ETC (Explore-Then-Commit) analogues for CABAI
- First-order CABAI algorithms

## Related Topics
- [[offline-oracle-efficient-bandits]] — regret minimization setting; distinct goal (cumulative regret over T rounds rather than fixed-confidence identification)
