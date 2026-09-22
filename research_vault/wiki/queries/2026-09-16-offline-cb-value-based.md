---
date: 2026-09-16
question: "Offline contextual bandits via the value-based (regression) approach — what is the standard formulation, what is known, and how does it compare to the policy-based (IPW / IX / LS) route?"
tags: [contextual-bandits, offline-contextual-bandits, pessimism, learning-theory]
---

# Value-Based Offline Contextual Bandits — reading path

The self-contained account this query produced has been split into two maintained pages. This page is the entry point and records what was asked, what was settled, and what was left open.

## Where the answer lives

- **[[contextual-bandits-offline]]** — the setting. Offline data and the learning objective, mean reward / value / optimal policy, function approximation with the tabular and linear models, coverage coefficients, standing assumptions, the relation to offline RL, and the taxonomy of the two families. It also holds the material that straddles them: the doubly robust estimator, class-restricted learning with imputed rewards, and the route comparison.
- **[[contextual-bandits-offline-value-based]]** — the reward-model family in full. The representative formulation, the technical toolkit, the greedy rule and its tabular and linear instantiations, the pessimistic rule with the tabular, linear, version-space and neural cases, Rashidinejad et al.'s Theorem 4 with a complete proof, the direct method, estimate-then-select, fast rates, and per-result provenance.
- A third page on policy-based methods (IPW / IX / LS, MaxIPW, PES) is still to be written; until it exists, the policy route is covered only by the one-paragraph sketch and the comparison on the setting page.

## What the query settled

1. The representative value-based formulation is regression-then-act, not estimate-then-select; the two coincide exactly when the policy class is unrestricted, and otherwise the second is cost-sensitive classification with imputed rewards.
2. Two lemmas carry every bound. The plug-in decomposition charges the rule for the action *it* picks, so greedy needs uniform coverage; the pessimism lemma charges only $`\pi^*`$'s uncertainty, so pessimism needs single-policy coverage.
3. In the tabular model the two families' bounds differ in exactly one symbol, $`C_{\mathrm{unif}}`$ against $`C^*`$ — in the rate and in the sample-size precondition alike.
4. Value-based methods need realizability and never use the propensities; coverage enters only the analysis, and through $`C_{\mathcal F}`$ it can be finite where importance weights are unbounded.

## Reading order

Kept with the technical account, in [[contextual-bandits-offline-value-based]] §8.2.

## Open follow-ups

- Ingest the four papers cited author–year throughout [[contextual-bandits-offline-value-based]], which have no paper pages yet. Citekeys resolved 2026-09-19 (postponed, not yet ingested):
  - `rashidinejad2021Bridging` — NeurIPS 2021. PDF already mirrored and md5-matched; the mirrored copy is arXiv v2 (Jul 2023), whose own note says "part of the paper has been published at Neurips 2021". The numbering §4.7 was verified against — Definition 1, Proposition 1, Theorems 4 and 5, Lemmas 13 and 14 — is v2's and checks out.
  - `jin2021Pessimism` — *Is Pessimism Provably Efficient for Offline RL?*, ICML 2021.
  - `xie2021Bellmanconsistent` — *Bellman-consistent Pessimism for Offline RL*, NeurIPS 2021.
  - `brandfonbrener2021OfflineCB` — *Offline Contextual Bandits with Overparameterized Models*, ICML 2021 (arXiv 2006.15368). **Not** `brandfonbrener2021Offline`, which is *Offline RL Without Off-Policy Evaluation*, NeurIPS 2021, by the same four authors in the same year. Checked: the NeurIPS PDF contains "action-stab" and "overparameteriz" zero times, so none of what this vault attributes to Brandfonbrener is in it.
  - All four were keyed to arXiv revision years until 2026-09-19; the keys above are the corrected ones. Confirm they are pinned before ingesting, since the citekey is the page filename, the link target and the mirror filename.
- Write the policy-based page; move the route comparison to it, or keep it on the setting page as the cross-family home.
- A topic page is still not earned: revisit once the four papers above have pages.
- `jun2026CS703Q10` (K.-S. Jun, *CS703Q10: Offline contextual bandits*, CSED703Q, Spring 2026) is the reference point for the policy route. Read 2026-09-19 and mirrored; no paper page yet. What it settles: MaxIPW and PES are its Lemmas 1–2 (now credited in [[contextual-bandits-offline]] §7), IX is its Theorem 3 with bias exactly $`\gamma C_\gamma(\pi)`$ and weights bounded by $`1/\gamma`$, LS its Theorem 6 with $`bD_b(\pi^*)`$ and $`D_b\le C_b`$ but *unbounded* weights, and the hyperparameter-adaptation sketch is [[ryu2025Improved]]. Its own open ends are offpolicy learning and the loss-vs-reward asymmetry, both marked TODO in the note.

## Connections

- [[contextual-bandits-offline]] / [[contextual-bandits-offline-value-based]] — the two pages this split into
- [[pessimism-principle]] — the shared mechanism
- [[fqi-finite-sample-analysis]] — the horizon-$`H`$ analysis the bandit case specializes
- [[contextual-bandits-online]] — the online problem this is the batch version of
