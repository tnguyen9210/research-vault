---
name: "Ashvin Nair"
affiliation: "UC Berkeley (at time of publication)"
areas: [offline-reinforcement-learning, robot-learning, online-finetuning]
---

# Ashvin Nair

**Affiliation:** Department of EECS, UC Berkeley (at time of publication)
**Research areas:** Offline RL and offline-to-online finetuning; robotic manipulation; goal-conditioned RL.

## Papers in this Vault

- [[Kostrikov2022Offline]] (2022) — IQL; co-author, and author of the AWAC baseline it compares against

## Research Themes

Getting offline-pretrained policies to keep improving with online interaction. AWAC (Nair et al. 2020) introduced advantage-weighted policy updates for exactly this purpose; [[implicit-q-learning]] inherits that extraction step, which is the stated reason IQL finetunes well (antmaze $370.1 \to 473.7$ after 1M online steps).
