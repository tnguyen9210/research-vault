---
title: "Epsilon-First Policies for Budget-Limited Multi-Armed Bandits"
authors: [Long Tran-Thanh, Archie Chapman, Enrique Munoz de Cote, Alex Rogers, Nicholas R. Jennings]
year: 2010
venue: AAAI
tags: [budget-limited-mab, bandits, epsilon-first, regret]
source: not ingested — no source file in raw/papers/
---

> **STUB — not ingested.** No PDF for this paper is present in `raw/papers/`, so this page has not been written from the source. Everything below is sourced from the description given by [[TranThanh2012Knapsack]], which is ingested, and is limited to what that paper states about its predecessor. Do not cite specifics from this page; ingest the paper to replace it.

# Epsilon-First Policies for Budget-Limited Multi-Armed Bandits

**TL;DR:** The earlier $\varepsilon$-first treatment of the [[budget-limited-mab]], superseded by [[kube]]. It splits the budget $B$ into an exploration phase of $\varepsilon B$ and an exploitation phase of $(1-\varepsilon)B$, and is provably stuck at $O(B^{2/3})$ regret.

## What the vault records about it

As described in [[TranThanh2012Knapsack]]:

- Introduces the $\varepsilon$-first policy family for budget-limited multi-armed bandits, where each arm $i$ has a pull cost $c_i$ and a single shared budget $B$ constrains both exploration and exploitation.
- Uniformly explores during the first $\varepsilon B$ of budget, then exploits for the remaining $(1-\varepsilon)B$.
- Achieves $O(B^{2/3})$ regret, and this rate is not improvable within the $\varepsilon$-first family.
- Performance is sensitive to the choice of $\varepsilon$.
- [[TranThanh2012Knapsack]] supersedes it: [[kube]] and fractional KUBE attain the optimal $O(\ln B)$ regret, with a matching lower bound, by interleaving exploration and exploitation through a UCB-augmented knapsack rather than separating them into phases.

## Connections

- **Superseded by:** [[TranThanh2012Knapsack]] — same authors (in part); $O(\ln B)$ replaces $O(B^{2/3})$
- [[budget-limited-mab]] — the problem setting this paper introduced policies for
- [[kube]] — the successor algorithm
- [[upper-confidence-bound]] — what the $\varepsilon$-first split fails to exploit and KUBE does
- [[Tran-Thanh-Long]], [[Rogers-Alex]], [[Jennings-Nicholas-R]] — authors also on [[TranThanh2012Knapsack]]

## To do

Obtain the AAAI 2010 paper and run a proper ingest. Until then, the author list, venue, and the $O(B^{2/3})$ claim above are secondhand.
