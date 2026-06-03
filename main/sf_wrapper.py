"""v6 奖励包装器：连击/压起身/血量落后加成/空闲惩罚/距离奖励/回合过渡保护。"""

import math
import time
import collections

import gym
import numpy as np


class StreetFighterCustomWrapperV4(gym.Wrapper):

    def __init__(self, env, reset_round=True, rendering=False):
        super().__init__(env)
        self.env = env

        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering

        self.reward_coeff = 3.0

        # Idle tracking
        self.idle_frames = 0
        self.idle_threshold = 120  # ~2 seconds at 60fps (20 agent steps * 6 frames)

        # Proximity tracking (encourage closing distance)
        self.prev_p1_x = 0
        self.prev_p2_x = 0
        self._first_step = True

        # Combo tracking (reward aggressive chaining)
        self.combo_counter = 0
        # Oki tracking (reward knockdown → fireball wake-up pressure)
        self.oki_window = 0  # countdown, steps remaining to hit P2 after a knockdown

    def _stack_observation(self):
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def reset(self):
        observation = self.env.reset()

        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        self.idle_frames = 0
        self._first_step = True
        self.combo_counter = 0
        self.oki_window = 0

        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def step(self, action):
        custom_done = False

        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):
            obs, _reward, _done, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']

        self.total_timesteps += self.num_step_frames

        damage_dealt = self.prev_oppont_health - curr_oppont_health
        damage_taken = self.prev_player_health - curr_player_health

        # Combo bonus: damage >= 15 = 击倒, 不依赖连续命中步数
        combo_bonus = 0.0
        if damage_dealt >= 15:
            combo_bonus = 1.5  # knockdown! reward the strong hit

        # Oki window: 命中后20步内再次命中+2.0, 受伤或超时清零
        prev_oki = self.oki_window

        if damage_taken > 0:
            self.oki_window = 0  # getting hit resets pressure
        elif damage_dealt > 0:
            self.oki_window = 20  # hit → open pressure window
        elif self.oki_window > 0:
            self.oki_window -= 1  # countdown in idle steps
        if self.idle_frames >= self.idle_threshold:
            self.oki_window = 0

        oki_bonus = 0.0
        if damage_dealt > 0 and prev_oki > 0 and prev_oki < 20:
            oki_bonus = 2.0

        # 空闲惩罚: 2秒无战斗后递增, -0.5→-3.5/步
        if damage_dealt == 0 and damage_taken == 0:
            self.idle_frames += self.num_step_frames
        else:
            self.idle_frames = 0

        idle_penalty = 0.0
        if self.idle_frames > self.idle_threshold:
            idle_sec = (self.idle_frames - self.idle_threshold) / 60.0
            idle_penalty = -0.5 - 0.1 * idle_sec

        # 距离奖励: 鼓励靠近对手, 60px内额外加点
        p1_x = info.get('agent_x', 0)
        p2_x = info.get('enemy_x', 0)
        proximity_reward = 0.0

        if not self._first_step:
            prev_dist = abs(self.prev_p1_x - self.prev_p2_x)
            curr_dist = abs(p1_x - p2_x)
            proximity_reward = 0.3 * (prev_dist - curr_dist)
            if curr_dist < 60:
                proximity_reward += 0.2

        self.prev_p1_x = p1_x
        self.prev_p2_x = p2_x
        self._first_step = False

        # 回合过渡保护: HP不可逆涨, 检测到HP上升=KO→回合切换
        if curr_player_health > self.prev_player_health or \
           curr_oppont_health > self.prev_oppont_health:
            if self.prev_oppont_health < self.prev_player_health:
                # P1 was ahead → P2 was KO'd
                custom_reward = math.pow(self.full_hp, (self.prev_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            else:
                # P1 was behind → P1 was KO'd
                custom_reward = -math.pow(self.full_hp, (self.prev_oppont_health + 1) / (self.full_hp + 1))
            custom_done = True
            return self._stack_observation(), 0.1 * custom_reward, custom_done, info

        # --- v4 reward ---

        # Game is over and player loses.
        if curr_player_health <= 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health <= 0:
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

        # Time-out: game engine says done but both still alive
        elif _done:
            # Both survived without KO — penalty proportional to HP left
            # (more HP → less fighting happened → bigger penalty)
            hp_score = (curr_player_health + curr_oppont_health) / (2 * self.full_hp)
            custom_reward = -50.0 * hp_score
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * damage_dealt - damage_taken

            # 血量落后加成: HP劣势时伤害奖励最高1.5倍
            hp_ratio = curr_player_health / max(curr_oppont_health, 1)
            if hp_ratio < 1.0:
                custom_reward *= 1.0 + 0.5 * (1.0 - hp_ratio)  # up to 1.5x

            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        if not self.reset_round:
            custom_done = False

        # Apply idle penalty, proximity reward, combo bonus, and oki bonus
        custom_reward += idle_penalty
        custom_reward += proximity_reward
        custom_reward += combo_bonus
        custom_reward += oki_bonus

        return self._stack_observation(), 0.1 * custom_reward, custom_done, info
