"""RyuDojo-RL 自对弈训练：v6奖励系统，4风格P2 + 冻结对手池，从10M基座继续训练。"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import glob
import random
import numpy as np

import retro
import gym
from gym import spaces

import torch
torch.backends.cudnn.benchmark = True

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from sf_wrapper import StreetFighterCustomWrapperV4
from p2_strong import RuleBasedP2Strong
from p2_styles import JumpInBullyP2, TurtleGodP2, HadoukenSpammerP2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE = "ryu vs ryu self_play"

NUM_ENV = 8
TOTAL_TIMESTEPS = 5_000_000

POOL_SIZE = 5
CKPT_INTERVAL = 500_000
LOG_INTERVAL = 20_000

# P2 噪声: 45%→25%线性衰减, 早期高以暴露智能体至波动压力
P2_NOISE_START = 0.45
P2_NOISE_END = 0.25
P2_NOISE_DECAY = 5_000_000

LEARNING_RATE = 5e-5
LR_FINAL = 5e-6
GAMMA = 0.94
GAE_LAMBDA = 0.95
ENT_COEF = 0.02  # higher diversity: prevents one-pattern strategy
VF_COEF = 0.5
BATCH_SIZE = 512
N_STEPS = 512
N_EPOCHS = 4
CLIP_RANGE = 0.10
MAX_GRAD_NORM = 0.5

# Resume from 10M base model
BASE_MODEL = os.path.join(SCRIPT_DIR, 'models', 'model_10M_v4.zip')
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs', 'training')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'models', 'checkpoints')
POOL_DIR = os.path.join(SCRIPT_DIR, 'opponents', 'pool')
_A10_TO_A12 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]


def p2_noise_prob(step):
    return P2_NOISE_START + (P2_NOISE_END - P2_NOISE_START) * min(1.0, step / P2_NOISE_DECAY)


class SelfPlayEnvV4(gym.Env):
    def __init__(self, game, state, opponent_pool_dir):
        super().__init__()
        retro_env = retro.make(
            game=game, state=state, players=2,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )
        self._env = StreetFighterCustomWrapperV4(retro_env, reset_round=True, rendering=False)
        self.action_space = spaces.MultiBinary(10)
        self.observation_space = self._env.observation_space
        self._rb_p2_pool = [
            RuleBasedP2Strong(noise_prob=0.0),
            JumpInBullyP2(),
            TurtleGodP2(),
            HadoukenSpammerP2(),
        ]
        self._current_rb_p2 = self._rb_p2_pool[0]
        self._opponent_pool_dir = opponent_pool_dir
        self._opponent_model = None
        self._opponent_path = None
        self._current_obs = None
        self._last_info = {}
        self._episode_count = 0
        self._step_in_episode = 0
        self._global_step = 0

    def set_global_step(self, step):
        self._global_step = step

    def _try_load_opponent(self):
        files = glob.glob(os.path.join(self._opponent_pool_dir, 'opponent_*.zip'))
        if not files:
            return
        chosen = random.choice(files)
        if chosen == self._opponent_path:
            return
        try:
            self._opponent_model = PPO.load(chosen, device='cpu')
            self._opponent_path = chosen
        except Exception:
            pass

    def reset(self):
        self._try_load_opponent()
        # Randomly pick a rule-based P2 style for this episode
        self._current_rb_p2 = random.choice(self._rb_p2_pool)
        obs = self._env.reset()
        self._current_obs = obs
        self._last_info = {}
        self._step_in_episode = 0
        self._episode_count += 1
        return obs

    @staticmethod
    def a10_to_a12(a10):
        a12 = [0] * 12
        for i10, i12 in enumerate(_A10_TO_A12):
            a12[i12] = int(a10[i10])
        return a12

    @staticmethod
    def _clean_action(a12):
        a12[2] = 0; a12[3] = 0
        if a12[6] and a12[7]: a12[6] = a12[7] = 0
        if a12[4] and a12[5]: a12[4] = a12[5] = 0
        atk = [i for i in [11, 10, 9, 8, 1, 0] if a12[i]]
        if len(atk) > 2:
            for i in atk[2:]: a12[i] = 0
        return a12

    def step(self, p1_action):
        p1_a12 = self.a10_to_a12(p1_action)
        noise_p = p2_noise_prob(self._global_step)

        if random.random() < noise_p or self._opponent_model is None:
            p2_a12 = self._current_rb_p2.act(self._last_info)
        else:
            try:
                flipped_obs = self._current_obs[:, ::-1, :].copy()
                p2_raw, _ = self._opponent_model.predict(flipped_obs, deterministic=False)
                if isinstance(p2_raw, np.ndarray):
                    p2_raw = p2_raw.tolist()
                if len(p2_raw) >= 12:
                    p2_a12 = p2_raw[:12]
                else:
                    p2_a12 = self.a10_to_a12(p2_raw[:10])
                p2_a12[6], p2_a12[7] = p2_a12[7], p2_a12[6]
                p2_a12 = self._clean_action(p2_a12)
            except Exception:
                p2_a12 = self._current_rb_p2.act(self._last_info)

        combined = p1_a12 + p2_a12
        obs, reward, done, info = self._env.step(combined)
        self._current_obs = obs
        self._last_info = info
        self._step_in_episode += 1

        if done and self._episode_count <= 3:
            print(f"  [Diag] Ep{self._episode_count}: "
                  f"steps={self._step_in_episode}, reward={reward:.2f}, "
                  f"P1_HP={info.get('agent_hp', '?')}, P2_HP={info.get('enemy_hp', '?')}")
        return obs, reward, done, info

    def close(self):
        self._env.close()


def make_env(pool_dir):
    def _init():
        env = SelfPlayEnvV4(GAME, STATE, pool_dir)
        env = Monitor(env)
        def _fwd(step):
            env.env.set_global_step(step)
        env.set_global_step = _fwd
        return env
    return _init


class OpponentPool:
    def __init__(self, pool_dir, pool_size):
        self.pool_dir = pool_dir
        self.pool_size = pool_size
        self.entries = []

    def seed(self, model_path):
        import shutil
        path = os.path.join(self.pool_dir, "opponent_seed.zip")
        shutil.copy(model_path, path)
        self.entries.append((path, 0))
        print(f"\n  [Pool] Seeded with best model (6.1M)")

    def add(self, model, step):
        path = os.path.join(self.pool_dir, f"opponent_{step:010d}.zip")
        model.save(path)
        self.entries.append((path, step))
        while len(self.entries) > self.pool_size:
            old_path, _ = self.entries.pop(0)
            if os.path.exists(old_path) and "seed" not in old_path:
                os.remove(old_path)
        print(f"\n  [Pool] Added at {step:,} (size: {len(self.entries)})")

    def status(self):
        return len(self.entries)


class V4Callback(BaseCallback):
    def __init__(self, save_dir, ckpt_interval, log_interval,
                 opponent_pool, start_time, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.ckpt_interval = ckpt_interval
        self.log_interval = log_interval
        self.opponent_pool = opponent_pool
        self.start_time = start_time
        self._last_save = 0
        self._last_log = 0
        self._last_sync = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_sync >= 1000:
            self._last_sync = step
            try:
                self.training_env.env_method('set_global_step', step)
            except Exception:
                pass
        if step - self._last_log >= self.log_interval:
            elapsed = time.time() - self.start_time
            speed = step / elapsed if elapsed > 0 else 0
            remaining = (TOTAL_TIMESTEPS - step) / speed if speed > 0 else 0
            buf = getattr(self.model, 'ep_info_buffer', None)
            if buf is not None and len(buf) > 0:
                recent = list(buf)[-min(50, len(buf)):]
                avg_r = np.mean([e['r'] for e in recent])
                avg_l = np.mean([e['l'] for e in recent])
            else:
                avg_r, avg_l = 0.0, 0.0
            noise = p2_noise_prob(step)
            pool_sz = self.opponent_pool.status()
            print(f"  [Step {step:>9,}/{TOTAL_TIMESTEPS:,}]  "
                  f"speed:{speed:>6.0f}  rew:{avg_r:>7.2f}  len:{avg_l:>5.0f}  "
                  f"noise:{noise:.2f}  pool:{pool_sz}  "
                  f"elapsed:{elapsed/3600:.1f}h  remain:{remaining/3600:.1f}h")
            self._last_log = step
        if step - self._last_save >= self.ckpt_interval:
            self._last_save = step
            path = os.path.join(self.save_dir, f"model_{step:010d}")
            self.model.save(path)
            self.opponent_pool.add(self.model, step)
            print(f"  [Checkpoint] {path}.zip")
        return True


def linear_schedule(initial_value, final_value):
    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    return scheduler


def main():
    print("=" * 60)
    print("  Self-Play Training (v6 reward system)")
    print("=" * 60)
    print(f"  Base model:   models/model_10M_v4.zip")
    print(f"  Steps:        {TOTAL_TIMESTEPS:,}")
    print(f"  Saves:        models/checkpoints/")
    print(f"  Logs:         logs/training/")
    print(f"  Pool:         opponents/pool/")
    print(f"  P2 noise:     {P2_NOISE_START:.0%} -> {P2_NOISE_END:.0%} (4 styles)")
    print(f"  Combo bonus:  +1.5 for damage >= 15 (knockdown)")
    print(f"  Oki bonus:    +2.0 for re-hit within 20-step pressure window")
    print(f"  Comeback:     up to 1.5x damage reward when HP behind")
    print(f"  Envs:         {NUM_ENV}")
    print(f"  lr:           {LEARNING_RATE} -> {LR_FINAL}")
    print(f"  batch:        {BATCH_SIZE}, n_steps: {N_STEPS}")
    print(f"  CUDA:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    if not os.path.exists(BASE_MODEL):
        print(f"[FATAL] Not found: {BASE_MODEL}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(POOL_DIR, exist_ok=True)

    # Seed pool with diverse checkpoints (only on first run)
    _existing = glob.glob(os.path.join(POOL_DIR, 'opponent_*.zip'))
    if not _existing:
        import shutil
        _old_ckpt_dir = os.path.join(SCRIPT_DIR, 'trained_models_self_play_v4_10M')
        # Pick diverse snapshots: ~3M, 5M, 7M, 9M intervals + 10M final
        _diverse_steps = [3_000_000, 5_000_000, 7_000_000, 9_000_000]
        for _step in _diverse_steps:
            _src = os.path.join(_old_ckpt_dir, f'model_{_step:010d}.zip')
            _dst = os.path.join(POOL_DIR, f'opponent_{_step:010d}.zip')
            if os.path.exists(_src):
                shutil.copy(_src, _dst)
                print(f"  [Pool] Seeded: opponent_{_step:010d}.zip ({_step/1e6:.0f}M)")
            else:
                print(f"  [Pool] WARNING: {_src} not found, skipping")
        # Also seed with 10M final
        _final_src = os.path.join(_old_ckpt_dir, 'model_final.zip')
        _final_dst = os.path.join(POOL_DIR, 'opponent_0010000000.zip')
        if os.path.exists(_final_src):
            shutil.copy(_final_src, _final_dst)
            print(f"  [Pool] Seeded: opponent_0010000000.zip (10M final)")
    else:
        print(f"  [Pool] Skipping seed — {len(_existing)} opponents already present")

    print(f"Creating {NUM_ENV}x envs...")
    env = SubprocVecEnv([make_env(POOL_DIR) for _ in range(NUM_ENV)])
    opponent_pool = OpponentPool(POOL_DIR, POOL_SIZE)
    # Load inherited pool entries
    for f in sorted(glob.glob(os.path.join(POOL_DIR, 'opponent_*.zip'))):
        opponent_pool.entries.append((f, 0))
    print(f"  [Pool] Loaded {opponent_pool.status()} inherited opponents")
    print()

    lr_schedule = linear_schedule(LEARNING_RATE, LR_FINAL)
    clip_schedule = linear_schedule(0.10, 0.05)

    model = PPO(
        "CnnPolicy", env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF,
        vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM,
        learning_rate=lr_schedule, clip_range=clip_schedule,
        tensorboard_log=LOG_DIR,
    )

    print(f"Loading: {os.path.basename(BASE_MODEL)}")
    base = PPO.load(BASE_MODEL, device='cpu')
    model.policy.load_state_dict(base.policy.state_dict())
    print(f"  Full weights transferred")
    print()

    start_time = time.time()
    callback = V4Callback(
        save_dir=SAVE_DIR, ckpt_interval=CKPT_INTERVAL,
        log_interval=LOG_INTERVAL, opponent_pool=opponent_pool,
        start_time=start_time,
    )

    print("Starting training...")
    print(f"  Comeback multiplier: 'never give up when behind'")
    print(f"  TensorBoard: tensorboard --logdir={LOG_DIR}")
    print()

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    final_path = os.path.join(SAVE_DIR, "model_final")
    model.save(final_path)
    env.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Complete!  Time: {elapsed/3600:.1f}h")
    print(f"  Model: {final_path}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
