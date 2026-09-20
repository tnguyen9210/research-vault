---
date: 2026-09-16
question: "Offline contextual bandits via the value-based (regression) approach — what is the standard formulation, what is known, and how does it compare to the policy-based (IPW / IX / LS) route?"
tags: [contextual-bandits, offline-contextual-bandits, pessimism, learning-theory]
---

# Value-Based Offline Contextual Bandits — reading path

The self-contained account this query produced has been split into two maintained pages. This page is the entry point and records what was asked, what was settled, and what was left open.

## Where the answer lives

- **[[offline-contextual-bandits]]** — the setting. Offline data and the learning objective, mean reward / value / optimal policy, function approximation with the tabular and linear models, coverage coefficients, standing assumptions, the relation to offline RL, and the taxonomy of the two families. It also holds the material that straddles them: the doubly robust estimator, class-restricted learning with imputed rewards, and the route comparison.
- **[[value-based-offline-bandits]]** — the reward-model family in full. The representative formulation, the technical toolkit, the greedy rule and its tabular and linear instantiations, the pessimistic rule with the tabular, linear, version-space and neural cases, Rashidinejad et al.'s Theorem 4 with a complete proof, the direct method, estimate-then-select, fast rates, and per-result provenance.
- A third page on policy-based methods (IPW / IX / LS, MaxIPW, PES) is still to be written; until it exists, the policy route is covered only by the one-paragraph sketch and the comparison on the setting page.

## What the query settled

1. The representative value-based formulation is regression-then-act, not estimate-then-select; the two coincide exactly when the policy class is unrestricted, and otherwise the second is cost-sensitive classification with imputed rewards.
2. Two lemmas carry every bound. The plug-in decomposition charges the rule for the action *it* picks, so greedy needs uniform coverage; the pessimism lemma charges only $\pi^*$'s uncertainty, so pessimism needs single-policy coverage.
3. In the tabular model the two families' bounds differ in exactly one symbol, $C_{\mathrm{unif}}$ against $C^*$ — in the rate and in the sample-size precondition alike.
4. Value-based methods need realizability and never use the propensities; coverage enters only the analysis, and through $C_{\mathcal F}$ it can be finite where importance weights are unbounded.

## Reading order

Kept with the technical account, in [[value-based-offline-bandits]] §8.2.

## Open follow-ups

- Ingest Rashidinejad et al. (2021), Jin–Yang–Wang (2021), Brandfonbrener et al. (2021), Xie et al. (2021) — all four are cited author–year here because no paper page exists yet.
- Write the policy-based page; move the route comparison to it, or keep it on the setting page as the cross-family home.
- A topic page is still not earned: revisit once the four papers above have pages.
- Drop K.-S. Jun's CSED703Q note into `raw/papers/Jun2026Offline.pdf`; it is the reference point for the policy route.

## Connections

- [[offline-contextual-bandits]] / [[value-based-offline-bandits]] — the two pages this split into
- [[pessimism-principle]] — the shared mechanism
- [[fqi-finite-sample-analysis]] — the horizon-$H$ analysis the bandit case specializes
- [[contextual-bandits]] — the online problem this is the batch version of
