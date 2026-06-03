"""模型评估：vs RuleBasedP2Strong，默认使用15M_v6，回退到10M基座。"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import retro
from stable_baselines3 import PPO

from sf_wrapper import StreetFighterCustomWrapperV4
from p2_strong import RuleBasedP2Strong

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE = "ryu vs ryu self_play"

_MODEL_V6 = os.path.join(SCRIPT_DIR, 'models', 'model_15M_v6.zip')
_MODEL_10M = os.path.join(SCRIPT_DIR, 'models', 'model_10M_v4.zip')
MODEL_PATH = _MODEL_V6 if os.path.exists(_MODEL_V6) else _MODEL_10M
NUM_EVAL = 30
_A10_TO_A12 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]


def a10_to_a12(a10):
    a12 = [0] * 12
    for i10, i12 in enumerate(_A10_TO_A12):
        a12[i12] = int(a10[i10])
    return a12


def main():
    model_name = os.path.basename(MODEL_PATH).replace('.zip', '')
    print("=" * 60)
    print(f"  {model_name} vs RuleBasedP2Strong ({NUM_EVAL} eps)")
    print("=" * 60)

    model = PPO.load(MODEL_PATH, device='cpu')
    retro_env = retro.make(
        game=GAME, state=STATE, players=2,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
    )
    env = StreetFighterCustomWrapperV4(retro_env, reset_round=True, rendering=False)
    rb = RuleBasedP2Strong()

    wins, losses, draws = 0, 0, 0
    hp_left, rewards_list, steps_list = [], [], []

    for ep in range(NUM_EVAL):
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        last_info = {}
        while not done:
            p1_raw, _ = model.predict(obs, deterministic=True)
            p1_a12 = a10_to_a12(p1_raw)
            p2_a12 = rb.act(last_info)
            obs, reward, done, info = env.step(p1_a12 + p2_a12)
            last_info = info
            ep_reward += reward
            steps += 1

        p1_hp = info.get('agent_hp', 0)
        p2_hp = info.get('enemy_hp', 0)
        rewards_list.append(ep_reward)
        steps_list.append(steps)

        if p1_hp <= 0 and p2_hp <= 0:
            draws += 1
        elif p1_hp <= 0:
            losses += 1
        elif p2_hp <= 0:
            wins += 1
            hp_left.append(p1_hp)
        else:
            draws += 1

        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{NUM_EVAL}] W:{wins} L:{losses} D:{draws}")

    env.close()
    avg_reward = np.mean(rewards_list)
    avg_hp = np.mean(hp_left) if hp_left else 0.0
    avg_steps = np.mean(steps_list)
    win_rate = wins / NUM_EVAL * 100

    print()
    print("-" * 50)
    print(f"  {wins}W {losses}L {draws}D | Win:{win_rate:.1f}% | Avg HP:{avg_hp:.0f} | Steps:{avg_steps:.0f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
