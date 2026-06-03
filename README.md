# RyuDojo-RL: Street Fighter II PPO Self-Play AI

[简体中文](README_CN.md) | [English](README.md)

An AI agent trained via PPO (Proximal Policy Optimization) to play Street Fighter II: Special Champion Edition (SEGA Genesis) through self-play. The agent learns entirely from RGB pixels, facing a diverse pool of rule-based opponents with distinct fighting styles and frozen historical checkpoints of itself.

**Key contribution:** A composite reward function incorporating combo bonuses, okizeme (wake-up pressure) rewards, comeback multipliers, and a ramping idle penalty, combined with four stylized rule-based opponents to mitigate policy collapse and cowardice.

## Setup

| Component | Detail |
|-----------|--------|
| Game | Street Fighter II: Special Champion Edition (Genesis) |
| Matchup | Ryu vs Ryu (mirror self-play) |
| Action space | 10-dim (model) → 12-dim (game), 6-frame repeat |
| Observation | NatureCNN, 9-frame stack (100×128×3) |
| Algorithm | PPO (stable-baselines3 CnnPolicy) |
| GPU | NVIDIA RTX 4060 Laptop |

## Reward System (v6)

```
base          = 3.0 × damage_dealt - damage_taken
comeback      = 1.0 + 0.5 × (1.0 - hp_ratio)     [up to 1.5x]
combo_bonus   = +1.5  [damage ≥ 15 = knockdown]
oki_bonus     = +2.0  [re-hit within 20-step pressure window]
idle_penalty  = -0.5 → -3.5/step  [ramping after 2s no combat]
proximity     = +0.3 × delta_dist + 0.2  [within 60px]
timeout       = -50   [both survive without KO]
```

Round transition guard: HP increase detection prevents mid-step KO miscounting.

## P2 Opponent Pool

| Opponent | Style |
|----------|-------|
| RuleBasedP2Strong | Anti-air, whiff punish, corner pressure, hadouken zoning |
| JumpInBullyP2 | High-frequency jump-in + throw/overhead mixup |
| TurtleGodP2 | Crouch-block everything, punish only on confirm |
| HadoukenSpammerP2 | Fireball at all ranges + meaty hadouken oki |
| Frozen historical models | Mirror self-play |

P2 noise: 45% → 25% over training (decayed linearly).

## Quick Start

```bash
conda create -n StreetFighterAI python=3.10
conda activate StreetFighterAI
cd main && pip install -r requirements.txt
```

Place legally obtained ROM and state/config files in the gym-retro data directory.

### Training

```bash
cd main
python train.py
```

Hyperparams: lr=5e-5→5e-6, gamma=0.94, ent_coef=0.02, batch=512, n_steps=512, 8 envs.

### Evaluation

```bash
cd main
python evaluate.py   # 30 episodes vs RuleBasedP2Strong
```

### Human vs AI

```bash
cd main
python human_play.py   # W/A/S/D move, J/K/L punch, U/I/O kick, Z/X/C specials
```

### Agent vs Agent Spectator

```bash
cd main
python agent_play.py
```

## Training History

| Version | Steps | vs StrongP2 | Notes |
|---------|-------|-------------|-------|
| v4 | 10M | 100% perfect HP | Baseline self-play |
| v5 | +5M | — | combo/oki reward bug (never fired) |
| v6 | +5M (15M total) | 100% perfect HP | **Current best**, fixed reward system |

## Key Technical Decisions

1. **P2 direction fix:** P2 faces LEFT on right side; dx>0 → LEFT (a[6]=1) to approach, counterintuitively
2. **6-frame action repeat:** Damage checked once per 6-frame step to avoid mid-step miscounting
3. **HP-increase guard:** HP never rises during SF2 combat — catching mid-step KOs prevents massive reward errors
4. **Self-play value net:** 1P-pretrained value network collapses in 2P self-play; must train natively during warmup
5. **Opponent diversity:** Fusion of stylized rule-based AI + frozen checkpoints prevents cyclic policy exploitation

## Acknowledgements

- [linyiLYi/street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai) — 提供初始框架与预训练基座模型
- [OpenAI Gym Retro](https://retro.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [DIAMBRA Arena (arXiv:2210.10595)](https://arxiv.org/abs/2210.10595)
- [Mitigating Cowardice for RL (IEEE CoG 2022)](https://ieee-cog.org/2022/assets/papers/paper_111.pdf)

## License

Apache License 2.0
