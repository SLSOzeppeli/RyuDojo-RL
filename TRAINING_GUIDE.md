# RyuDojo-RL 训练指南

## 目录

1. [项目脚本](#项目脚本)
2. [自对弈训练系统](#自对弈训练系统)
3. [创建游戏快照](#创建游戏快照)
4. [使用方法](#使用方法)
5. [参数调优](#参数调优)
6. [常见问题](#常见问题)

---

## 项目脚本

| 脚本 | 用途 |
|------|------|
| `train.py` | **主训练脚本** — v6 奖励系统 + 4风格P2 + 冻结对手池自对弈 |
| `evaluate.py` | 模型评估 — 30 局 vs RuleBasedP2Strong |
| `human_play.py` | 人机对战 — 玩家自选角色 vs AI Ryu |
| `agent_play.py` | Agent 对战观战 — 两个模型对打 |
| `sf_wrapper.py` | 奖励包装器 — v6 奖励系统实现 |
| `p2_strong.py` | 综合强P2 — 升龙对空/防御预读/挥空惩罚/版边压制/波动牵制 |
| `p2_styles.py` | 风格化P2 — JumpInBully / TurtleGod / HadoukenSpammer |

---

## 自对弈训练系统

### 架构

```
训练模型 (12维, 仅控制P1)
     │
     ▼
┌─────────────────────────────────┐
│         P2 对手选择              │
│                                 │
│  噪声概率 45%→25% (线性衰减):   │
│  ┌───────────────────────────┐  │
│  │ RuleBasedP2Strong         │  │
│  │ JumpInBullyP2             │  │
│  │ TurtleGodP2               │  │
│  │ HadoukenSpammerP2         │  │
│  └───────────────────────────┘  │
│                                 │
│  其余: 冻结历史模型池 (5个)     │
│  ├─ opponent_3000000.zip       │
│  ├─ opponent_5000000.zip       │
│  ├─ opponent_7000000.zip       │
│  ├─ opponent_9000000.zip       │
│  └─ opponent_10000000.zip      │
└─────────────────────────────────┘
```

### 训练流程

**阶段 1: 热身 (前几个 episode)**
- 价值网络在 2P 环境中从头训练
- 预训练的 1P 价值网络在 2P 自对弈中会立即崩溃

**阶段 2: 规则P2主导期 (高噪声)**
- 噪声 45% — 智能体主要面对规则对手
- 学会基本格斗策略：接近、攻击、防御

**阶段 3: 自对弈主导期 (低噪声)**
- 噪声降至 25% — 智能体主要面对自己的历史版本
- 通过与不同时期的自己对抗实现持续进步

### 对手池管理

每 500K 步保存当前模型到 `opponents/pool/`，FIFO 淘汰（保留最近 5 个）。对手池在环境 reset 时随机选择对手。

---

## 创建游戏快照

借助 gym-retro 的 Integration UI，可以为任意关卡/对手创建自定义训练快照。

### 1. 启动 Integration UI

将 `data/Gym Retro Integration.exe` 复制到 gym-retro 的 `retro/` 目录（ROM 所在目录的上两级），双击运行。或者通过 Python 启动：

```python
import retro
retro.IntegrationGUI()
```

### 2. 捕捉快照

1. 选择 `StreetFighterIISpecialChampionEdition-Genesis` 并启动
2. 在游戏中进入目标关卡（例如修改存档到特定对手）
3. 按 `Ctrl+H` 调出快照管理器
4. 点击 **"Take Snapshot"** 捕捉当前游戏状态
5. 修改快照名称为有意义的名字，例如 `ryu_vs_guile`、`ryu_vs_ken`
6. `.state` 文件会保存到游戏数据目录

### 3. 创建不同对手的快照

要训练 Agent 对抗不同角色，需要捕捉对应关卡的游戏状态：

**方法 A: 使用游戏存档**
- 在模拟器中正常游玩到目标关卡
- 在 "ROUND 1 FIGHT!" 出现时按 `Ctrl+H` 捕捉快照

**方法 B: 修改 data.json 定位内存地址**
- `data.json` 中定义了 `enemy_character` 等内存地址
- 可以通过修改这些地址的值来切换对手角色（需要了解 Genesis 内存布局）

**方法 C: 使用已有快照**
- 项目中已包含的快照：
  - `ryu vs ryu self_play` — 镜像对局（默认训练用）
  - `Champion.Level12.RyuVsBison` — 对战 Boss
  - `ryu vs zahgief` — 对战桑吉尔夫
  - `choose P2` — 角色选择画面

### 4. 为新对手配置训练

在 `train.py` 中修改 `STATE` 变量即可切换对战快照：

```python
STATE = "ryu vs ryu self_play"    # 默认
# STATE = "ryu_vs_guile"          # 切换到对战古烈
# STATE = "Champion.Level12.RyuVsBison"  # 切换到对战Boss
```

对于自对弈训练（Ryu vs Ryu），需确保快照中包含两个由 AI 控制的角色。

### 5. 添加新的游戏内对手

如果想训练对抗特定的游戏 AI（如 Level 8 Balrog），步骤为：

```bash
# 1. 在游戏中打到对应关卡
# 2. Ctrl+H 捕捉快照
# 3. 将 .state 文件复制到 data/ 目录
# 4. 在 train.py 中设置 STATE = "你的快照名"
```

注意：游戏内 AI 的行为是固定的，不会像规则 P2 那样产生风格多样性。推荐使用规则 P2 系统来模拟不同对手风格。

---

## 使用方法

### 环境配置

```bash
conda create -n StreetFighterAI python=3.10
conda activate StreetFighterAI
cd main
pip install -r requirements.txt
```

### ROM 与数据配置

1. 合法获取 ROM 文件，放入 gym-retro 数据目录
2. 重命名为 `rom.md`
3. 将 `data/` 中的 `.state`、`.json` 文件复制到同一目录

查找数据目录：
```bash
python -c "import retro; print(retro.data.GAME_DATA_DIR)"
```

### 训练

```bash
cd main
python train.py
```

### 评估

```bash
python evaluate.py
```

### 人机对战

```bash
python human_play.py
```

### Agent 对战观战

```bash
python agent_play.py
```

### TensorBoard

```bash
tensorboard --logdir=main/logs/
```

---

## 参数调优

### 核心超参 (train.py)

```python
LEARNING_RATE = 5e-5      # 初始学习率
LR_FINAL = 5e-6           # 最终学习率
GAMMA = 0.94              # 折扣因子
ENT_COEF = 0.02           # 熵系数 (防止策略单一化)
BATCH_SIZE = 512
N_STEPS = 512             # 每轮采样步数
N_EPOCHS = 4
CLIP_RANGE = 0.10         # PPO clip 范围
MAX_GRAD_NORM = 0.5
NUM_ENV = 8               # 并行环境数
```

### 奖励权重 (sf_wrapper.py)

| 组件 | 值 | 作用 |
|------|-----|------|
| `reward_coeff` | 3.0 | 伤害奖励系数 |
| `combo_bonus` | 1.5 | 击倒奖励 |
| `oki_bonus` | 2.0 | 压起身奖励 |
| `idle_penalty` | -0.5→-3.5 | 空闲惩罚 |
| `proximity` | 0.3×delta + 0.2 | 距离奖励 |

### 常见调优场景

**智能体太被动 (龟缩)：**
- 增大 `idle_penalty` 起始值（-0.5 → -1.0）
- 增大 `proximity` 奖励权重
- 降低 `ent_coef`

**智能体策略单一：**
- 增大 `ent_coef`（0.02 → 0.05）
- 增大 `P2_NOISE_START`（0.45 → 0.60）

**训练不稳定：**
- 降低 `LEARNING_RATE`（5e-5 → 1e-5）
- 减小 `CLIP_RANGE`（0.10 → 0.05）
- 增大 `BATCH_SIZE`（512 → 1024）

---

## 常见问题

### Q1: 模型无法加载 ROM

确保 ROM 文件名为 `rom.md`，且位于正确的 gym-retro 数据目录。使用 `utils/print_game_lib_folder.py` 确认路径。

### Q2: 训练速度太慢

- 减少 `NUM_ENV` 到 4
- 降低 `N_STEPS` 到 256
- 确认 CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`

### Q3: 训练到后期奖励不再增长

这是正常现象 — P2 也在变强（对手池中的历史模型越来越强）。关注 vs StrongP2 的评估胜率而非绝对奖励值。

### Q4: 如何从 checkpoint 继续训练

```python
# 修改 train.py 中的 BASE_MODEL
BASE_MODEL = os.path.join(SCRIPT_DIR, 'models', 'checkpoints', 'model_005000000.zip')
```

### Q5: 如何添加新的 P2 风格

在 `p2_styles.py` 中创建新类，实现 `act(self, info)` 方法，返回 12 维动作。然后在 `train.py` 的 `SelfPlayEnvV4.__init__` 中注册。

---

## 训练指标解读

### 平均奖励

- **> 150**: 压倒性优势
- **80-150**: 稳定优势
- **30-80**: 均势对战
- **< 30**: 处于下风

### explained_variance

- **> 0.5**: 价值函数良好
- **0-0.5**: 可接受
- **< 0**: 价值函数崩溃，需降低 lr 或增加 vf_coef

### entropy

- 应从初始值稳定下降但不归零。归零表示策略坍缩，增大 `ent_coef`。

---

## 致谢

- [linyiLYi/street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai) — 初始框架与预训练基座
