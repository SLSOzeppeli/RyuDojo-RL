# RyuDojo-RL：基于PPO与多样化对手池的街霸II自对弈策略学习

本项目使用 PPO（Proximal Policy Optimization）深度强化学习算法，在 SEGA Genesis 版《街头霸王 II：特别冠军版》上训练一个自对弈格斗智能体。智能体仅通过游戏画面的 RGB 像素值进行决策，通过与规则对手及历史版本自身的镜像对战，逐步习得人类的格斗策略。

**核心创新：** 设计了包含连击奖励、压起身奖励、血量落后加权的复合奖励函数，以及四种风格化的规则对手构成多样化的训练对手池，有效缓解了格斗 AI 中"懦夫行为"和策略单一的问题。

## 环境与模型

| 配置 | 说明 |
|------|------|
| 游戏 | Street Fighter II: Special Champion Edition (Genesis) |
| 角色 | Ryu vs Ryu 镜像对局 |
| 动作空间 | 10 维（模型输出）→ 12 维（游戏动作，经映射与清理） |
| 决策帧步 | 每 6 帧执行一次动作（~100ms/决策） |
| 观测 | NatureCNN，9 帧堆叠 (100×128×3) |
| 算法 | PPO (stable-baselines3 CnnPolicy) |
| GPU | NVIDIA RTX 4060 Laptop |

## 奖励系统 (v6)

```
基础奖励       = 3.0 × 造成伤害 - 受到伤害
血量落后加成   = 1.0 + 0.5 × (1.0 - 血量比)   [HP落后时最高 1.5x]
连击奖励       = +1.5   [单次伤害 ≥ 15，即扫倒/重击/投技]
压起身奖励     = +2.0   [20步窗口内再次命中，激励 knockdown → 压起身节奏]
空闲惩罚       = -0.5 → -3.5/步  [2秒无战斗后递增，杜绝消极回避]
距离奖励       = +0.3 × 距离变化量 + 0.2   [60px内额外加点，鼓励接近交火]
超时惩罚       = -50    [双方存活且无 KO，惩罚不敢进攻]
```

**关键设计：**
- **连击检测**：用伤害阈值（≥15）而非连续命中步数，适配 6 帧步长下无法区分单次命中与连续技的问题
- **压起身窗口**：命中后启动 20 步倒计时，窗口内再次命中即奖励，无延时递减后清零
- **回合过渡保护**：通过 HP 回升检测中途 KO，避免将新回合 HP 重置误读为巨额负奖励
- **血量落后加成**：越落后伤害权重越高，"永不言弃"，防止面对人类玩家时因不适应前期劣势而崩盘

## P2 对手体系

训练时 P2 随机选择以下对手，噪声率从 45% 线性衰减至 25%：

| 对手 | 风格 | 核心策略 |
|------|------|----------|
| RuleBasedP2Strong | 综合强敌 | 升龙对空、防御预读、挥空惩罚、版边压制、波动拳牵制 |
| JumpInBullyP2 | 跳入压制 | 高频跳入 + 投/中段择，惩罚蹲防龟缩 |
| TurtleGodP2 | 极致龟缩 | 全程蹲防，只在确反时出招，极度消耗耐心 |
| HadoukenSpammerP2 | 波升压起身 | 全距离波动拳 + 倒地起身压波动，还原人类 Ryu 玩家打法 |
| 冻结历史模型 | 镜像对战 | 自对弈同角色，画面翻转 + 方向镜像 |

## 项目结构

```
├── main/
│   ├── train.py              # 自对弈 PPO 训练脚本
│   ├── evaluate.py           # 模型评估（vs RuleBasedP2Strong，30局）
│   ├── human_play.py         # 人机对战（玩家自选角色 vs AI Ryu）
│   ├── agent_play.py         # AI vs AI 对战观战
│   ├── sf_wrapper.py         # 奖励包装器（v6 奖励系统）
│   ├── p2_strong.py          # 综合强P2 规则对手
│   ├── p2_styles.py          # 三种风格化P2
│   ├── requirements.txt      # Python 依赖
│   ├── models/               # 里程碑模型
│   │   ├── model_10M_v4.zip  # v4 10M 基座
│   │   └── model_15M_v6.zip  # v6 15M 当前最优 ★
│   ├── logs/                 # TensorBoard 训练日志
│   ├── opponents/pool/       # 对手池快照
│   └── models/checkpoints/   # 训练检查点
├── chat/                     # 工作日志
└── used/                     # 历史文件归档
```

## 快速开始

### 环境配置

```bash
# 创建 conda 环境
conda create -n StreetFighterAI python=3.10
conda activate StreetFighterAI

# 安装依赖
cd main
pip install -r requirements.txt
```

**游戏 ROM 配置：**

需自行合法获取 `Street Fighter II: Special Champion Edition` (Genesis) 的 ROM 文件。将 ROM 放入 gym-retro 游戏数据目录并重命名为 `rom.md`（通常位于 `<conda_env>/Lib/site-packages/retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis/`）。

将本项目的 `.state` 快照文件和 `data.json`、`metadata.json`、`scenario.json` 配置文件一并放入该目录。

### 训练

从 v4 基座模型继续训练：

```bash
cd main
python train.py
```

默认超参：lr=5e-5→5e-6，gamma=0.94，ent_coef=0.02，batch=512，n_steps=512，8 并发环境，每轮 5M 步。

### 评估

```bash
cd main
python evaluate.py
```

默认使用 `models/model_15M_v6.zip`，30 局 vs RuleBasedP2Strong (deterministic)，输出胜率及平均剩余 HP。

### 人机对战

```bash
cd main
python human_play.py
```

- WASD 移动，J/K/L 拳，U/I/O 脚
- Z/X/C 一键必杀（波动拳/升龙拳/龙卷旋风腿）
- F1-F3 调速，ESC 退出
- 支持全 12 角色（玩家任选 vs AI Ryu）

### Agent 对战观战

```bash
cd main
python agent_play.py
```

### TensorBoard

```bash
tensorboard --logdir=main/logs/
```

## 训练历程

| 版本 | 总步数 | vs StrongP2 胜率 | 说明 |
|------|--------|------------------|------|
| v4 | 10M | 100% 完美HP | 基础自对弈，奖励信号较简 |
| v5 | +5M | — | combo/oki 奖励有 bug，实际等价于 v4 |
| v6 | +5M (15M总) | 100% 完美HP | **当前最优**：修复 combo/oki，学会 knockdown→压起身 |

## 关键技术决策

1. **方向修复**：P2 在屏幕右侧面向左，dx>0 需按 LEFT（a[6]=1）才能走近，而非直觉的 RIGHT
2. **6 帧步**：每个 agent 决策执行 6 帧，伤害在 6 帧结束后统一检查，避免帧间误判
3. **回合过渡检测**：SF2 中 HP 不可逆涨，检测到 HP 上升 = 中途 KO → 回合切换，避免误奖/误罚
4. **自对弈价值网络**：预训练 1P 价值网络在 2P 自对弈中会立即崩溃，必须在热身阶段原生训练
5. **对手多样化**：四种风格化规则 P2 + 冻结历史模型池，防止策略单一化和循环作弊

## 致谢

- [linyiLYi/street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai) — 提供初始框架与预训练基座模型
- [OpenAI Gym Retro](https://retro.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [DIAMBRA Arena: A New RL Platform for Research](https://arxiv.org/abs/2210.10595)
- [Mitigating Cowardice for Reinforcement Learning](https://ieee-cog.org/2022/assets/papers/paper_111.pdf)

## 许可证

Apache License 2.0
