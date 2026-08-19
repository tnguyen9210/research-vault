---
name: "Ilya Kostrikov"
affiliation: "UC Berkeley (at time of publication)"
areas: [offline-reinforcement-learning, deep-reinforcement-learning, imitation-learning]
---

# Ilya Kostrikov

**Affiliation:** Department of EECS, UC Berkeley (at time of publication)
**Research areas:** Offline and off-policy deep RL; imitation learning; efficient RL implementations (author of JAXRL).

## Papers in this Vault

- [[Kostrikov2022Offline]] (2022) — IQL: in-sample multi-step dynamic programming for offline RL via upper-[[expectile-regression]] value learning; state of the art on D4RL antmaze

## Research Themes

Practical offline RL methods that reduce algorithmic machinery rather than add to it — IQL replaces explicit constraints and value penalties with a single loss-function change. A parallel thread of work on fast, minimal RL implementations (JAXRL), reflected in IQL's emphasis on compute cost as a first-class evaluation axis.
