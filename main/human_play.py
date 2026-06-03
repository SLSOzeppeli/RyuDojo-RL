import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import numpy as np
import collections
import retro
from stable_baselines3 import PPO

import pygame

GAME_NAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE_2P = "ryu vs ryu self_play" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
_MODEL_V6 = "model_15M_v6"
_MODEL_10M = "model_10M_v4"
DEFAULT_MODEL = _MODEL_V6 if os.path.exists(os.path.join(MODEL_DIR, f"{_MODEL_V6}.zip")) else _MODEL_10M

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 448

FPS_OPTIONS = {
    'slow': 15,
    'normal': 30,
    'fast': 60
}

# Character ID mapping (SFII SCE Genesis memory layout, enemy_character memory address)
CHARACTER_IDS = {
    0: 'Ryu',
    1: 'Ken',
    2: 'Chun-Li',
    3: 'Guile',
    4: 'Zangief',
    5: 'Dhalsim',
    6: 'Blanka',
    7: 'E. Honda',
    8: 'Balrog',
    9: 'Vega',
    10: 'Sagat',
    11: 'M. Bison',
}

# Special moves for each character
# motion: direction sequence for combo detection (empty list = charge/mash move, not detectable)
# buttons: retro button names that trigger the move
# key_hint: human-readable keyboard input
CHARACTER_SPECIALS = {
    'Ryu': {
        'hadouken': {
            'name': 'Hadouken',
            'motion': ['DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'S > D + J/K/L',
        },
        'shoryuken': {
            'name': 'Shoryuken',
            'motion': ['RIGHT', 'DOWN', 'DOWN_RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'D > S > D + J/K/L',
        },
        'tatsumaki': {
            'name': 'Tatsumaki',
            'motion': ['DOWN', 'DOWN_LEFT', 'LEFT'],
            'buttons': ['A', 'B', 'C'],
            'key_hint': 'S > A + U/I/O',
        },
    },
    'Ken': {
        'hadouken': {
            'name': 'Hadouken',
            'motion': ['DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'S > D + J/K/L',
        },
        'shoryuken': {
            'name': 'Shoryuken',
            'motion': ['RIGHT', 'DOWN', 'DOWN_RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'D > S > D + J/K/L',
        },
        'tatsumaki': {
            'name': 'Tatsumaki',
            'motion': ['DOWN', 'DOWN_LEFT', 'LEFT'],
            'buttons': ['A', 'B', 'C'],
            'key_hint': 'S > A + U/I/O',
        },
    },
    'Chun-Li': {
        'kikoken': {
            'name': 'Kikoken',
            'motion': ['LEFT', 'DOWN_LEFT', 'DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'A > S > D + J/K/L',
        },
        'spinning_bird': {
            'name': 'Spinning Bird Kick',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': '[Charge] hold S > W + U/I/O',
        },
        'lightning_legs': {
            'name': 'Lightning Legs',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': 'U/I/O (rapid press)',
        },
    },
    'Guile': {
        'sonic_boom': {
            'name': 'Sonic Boom',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
        'flash_kick': {
            'name': 'Flash Kick',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': '[Charge] hold S > W + U/I/O',
        },
    },
    'Zangief': {
        'spd': {
            'name': 'Spinning Piledriver',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'A > S > D > W + J/K/L (360)',
        },
        'banishing_flat': {
            'name': 'Banishing Flat',
            'motion': ['RIGHT', 'DOWN', 'DOWN_RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'D > S > D + J/K/L',
        },
    },
    'Dhalsim': {
        'yoga_fire': {
            'name': 'Yoga Fire',
            'motion': ['DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'S > D + J/K/L',
        },
        'yoga_flame': {
            'name': 'Yoga Flame',
            'motion': ['LEFT', 'DOWN_LEFT', 'DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'A > S > D + J/K/L',
        },
        'yoga_blast': {
            'name': 'Yoga Blast',
            'motion': ['LEFT', 'DOWN_LEFT', 'DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['A', 'B', 'C'],
            'key_hint': 'A > S > D + U/I/O',
        },
    },
    'Blanka': {
        'electric_thunder': {
            'name': 'Electric Thunder',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'J/K/L (rapid press)',
        },
        'rolling_attack': {
            'name': 'Rolling Attack',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
    },
    'E. Honda': {
        'hundred_hand': {
            'name': 'Hundred Hand Slap',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'J/K/L (rapid press)',
        },
        'sumo_headbutt': {
            'name': 'Sumo Headbutt',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
    },
    'Balrog': {
        'dash_punch': {
            'name': 'Dash Punch',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
        'turn_punch': {
            'name': 'Turn Punch',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'Hold J+K+L then release',
        },
    },
    'Vega': {
        'rolling_crystal': {
            'name': 'Rolling Crystal Flash',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
        'flying_barcelona': {
            'name': 'Flying Barcelona',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': '[Charge] hold S > W + U/I/O',
        },
    },
    'Sagat': {
        'tiger_shot': {
            'name': 'Tiger Shot',
            'motion': ['DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'S > D + J/K/L',
        },
        'tiger_uppercut': {
            'name': 'Tiger Uppercut',
            'motion': ['RIGHT', 'DOWN', 'DOWN_RIGHT'],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': 'D > S > D + J/K/L',
        },
        'tiger_knee': {
            'name': 'Tiger Knee',
            'motion': ['DOWN', 'DOWN_RIGHT', 'RIGHT'],
            'buttons': ['A', 'B', 'C'],
            'key_hint': 'S > D + U/I/O',
        },
    },
    'M. Bison': {
        'psycho_crusher': {
            'name': 'Psycho Crusher',
            'motion': [],
            'buttons': ['X', 'Y', 'Z'],
            'key_hint': '[Charge] hold A > D + J/K/L',
        },
        'double_knee': {
            'name': 'Double Knee Press',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': '[Charge] hold A > D + U/I/O',
        },
        'head_press': {
            'name': 'Head Press',
            'motion': [],
            'buttons': ['A', 'B', 'C'],
            'key_hint': '[Charge] hold S > W + U/I/O',
        },
    },
}


class FrameStackProcessor:
    def __init__(self, num_frames=9):
        self.num_frames = num_frames
        self.frame_stack = collections.deque(maxlen=num_frames)

    def reset(self, initial_frame):
        self.frame_stack.clear()
        processed = initial_frame[::2, ::2, :]
        for _ in range(self.num_frames):
            self.frame_stack.append(processed)
        return self._get_stacked_obs()

    def step(self, frame):
        processed = frame[::2, ::2, :]
        self.frame_stack.append(processed)
        return self._get_stacked_obs()

    def _get_stacked_obs(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i % 3] for i in range(3)], axis=-1)


class ComboBuffer:
    """输入序列检测器 - 检测方向键序列 + 攻击键触发必杀技

    实测映射: J=轻拳 K=中拳 L=重拳  U=轻脚 I=中脚 O=重脚
    Retro内部名: action[10]=X(轻拳) [9]=Y(中拳) [11]=Z(重拳)
                 action[1]=A(轻脚) [0]=B(中脚) [8]=C(重脚)
    """

    def __init__(self, history_len=15):
        self.history = []
        self.history_len = history_len
        self.specials = {}
        self._last_triggered = None
        self._trigger_cooldown = 0

    def set_character(self, character_name):
        """加载角色对应的必杀技指令表"""
        self.specials = CHARACTER_SPECIALS.get(character_name, CHARACTER_SPECIALS.get('Ryu', {}))
        self.history.clear()
        self._last_triggered = None
        self._trigger_cooldown = 0

    def _get_direction(self, action):
        """从 12 维 action 提取方向"""
        up = action[4]
        down = action[5]
        left = action[6]
        right = action[7]

        if up and not down and not left and not right:
            return 'UP'
        if down and not up and not left and not right:
            return 'DOWN'
        if left and not right and not up and not down:
            return 'LEFT'
        if right and not left and not up and not down:
            return 'RIGHT'
        if down and right and not up and not left:
            return 'DOWN_RIGHT'
        if down and left and not up and not right:
            return 'DOWN_LEFT'
        if up and right and not down and not left:
            return 'UP_RIGHT'
        if up and left and not down and not right:
            return 'UP_LEFT'
        return None

    def _get_attack(self, action):
        """从 12 维 action 提取攻击按钮"""
        attacks = []
        if action[0]:
            attacks.append('B')
        if action[1]:
            attacks.append('A')
        if action[8]:
            attacks.append('C')
        if action[9]:
            attacks.append('Y')
        if action[10]:
            attacks.append('X')
        if action[11]:
            attacks.append('Z')
        return attacks if attacks else None

    def _match_motion(self, motion, directions):
        """检查方向历史是否匹配必杀技指令。
        从最近的输入向前回溯，允许中间有无关方向。
        空motion表示蓄力/连打技，无法通过方向序列检测。"""
        if not motion:
            return False
        if len(directions) < len(motion):
            return False
        recent = [d for d in directions[-8:] if d is not None]
        if len(recent) < len(motion):
            return False
        for start in range(len(recent) - len(motion) + 1):
            match = True
            for i, expected in enumerate(motion):
                if recent[start + i] != expected:
                    match = False
                    break
            if match:
                return True
        return False

    def update(self, action):
        """传入当前帧的 12 维 action，检测必杀技触发"""
        direction = self._get_direction(action)
        attack = self._get_attack(action)

        self.history.append((direction, attack))
        if len(self.history) > self.history_len:
            self.history.pop(0)

        if self._trigger_cooldown > 0:
            self._trigger_cooldown -= 1
            return None

        directions = [d for d, _ in self.history]
        current_attacks = attack or []

        for key, move in self.specials.items():
            if self._match_motion(move['motion'], directions):
                if any(b in current_attacks for b in move['buttons']):
                    if self._last_triggered != key or self._trigger_cooldown == 0:
                        self._last_triggered = key
                        self._trigger_cooldown = 20
                        return key

        return None

    def get_hints(self):
        """返回所有必杀技的按键提示"""
        return [(m['name'], m['key_hint']) for m in self.specials.values()]


class TwoPlayerEnv:
    def __init__(self, game, state, rendering=True, fps='normal'):
        self.env = retro.make(
            game=game,
            state=state,
            players=2,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )

        self.rendering = rendering
        self.fps = FPS_OPTIONS.get(fps, 30)
        self.clock = pygame.time.Clock() if rendering else None

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.frame_processor = FrameStackProcessor(num_frames=9)

        print(f"Action space shape: {self.action_space.shape}")
        print(f"   - Player 1 (AI/Ryu) : indices 0-11")
        print(f"   - Player 2 (You)    : indices 12-23")

    def reset(self):
        obs = self.env.reset()
        stacked_obs = self.frame_processor.reset(obs)
        return obs, stacked_obs

    def step(self, combined_action):
        obs, rewards, done, info = self.env.step(combined_action)
        stacked_obs = self.frame_processor.step(obs)

        if self.rendering and self.clock:
            self.env.render()
            self.clock.tick(self.fps)
        elif not self.rendering:
            time.sleep(1.0 / self.fps)

        return obs, stacked_obs, rewards, done, info

    def close(self):
        try:
            self.env.close()
        except:
            pass


# One-button special move macros (Z/X/C keys)
# Each macro plays a direction→attack sequence frame by frame.
# SFII input buffer: ~15f total for directions. Button must be pressed WITH final dir.
# Total macro: 4f×2 dirs + 4f dir+btn = 12f (~0.2s at 60fps)
_SPECIAL_MACROS = {
    'hadouken':   {'key': pygame.K_z},     # QCF + P
    'shoryuken':  {'key': pygame.K_x},     # DP + P
    'tatsumaki':  {'key': pygame.K_c},     # QCB + K
}

_DIR_FRAMES = 4        # frames per direction step
_ATTACK_FRAMES = 4     # frames of direction+button at end

def _build_macro_frames(directions, attack_btn):
    """Build a tight frame sequence for SFII special move input.

    directions: list of index-lists (e.g. [[5], [5,7], [7]])
    attack_btn: single retro button name (e.g. 'Z' for heavy punch)

    Returns list of 12-dim action vectors, one per env.step() call.
    """
    btn_idx = _BTN_MAP[attack_btn]
    frames = []

    # Direction-only frames (all but last direction)
    for d in directions[:-1]:
        for _ in range(_DIR_FRAMES):
            a = [0] * 12
            for idx in d:
                a[idx] = 1
            frames.append(a)

    # Final direction + attack button pressed TOGETHER
    final_dir = directions[-1]
    for _ in range(_ATTACK_FRAMES):
        a = [0] * 12
        for idx in final_dir:
            a[idx] = 1
        a[btn_idx] = 1
        frames.append(a)

    return frames

# Direction index lists for _build_macro_frames
_D = [5]            # DOWN
_R = [7]            # RIGHT
_L = [6]            # LEFT
_DR = [5, 7]        # DOWN_RIGHT
_DL = [5, 6]        # DOWN_LEFT

# Button name → action index
_BTN_MAP = {'A': 1, 'B': 0, 'C': 8, 'X': 10, 'Y': 9, 'Z': 11}

# Two macro sets: one for P2 facing LEFT (on right side), one for P2 facing RIGHT (on left side).
# When characters cross sides, the directional inputs mirror.
_MACRO_FACING_LEFT = {
    # P2 on right side, facing LEFT. Forward = LEFT, Back = RIGHT.
    'hadouken':   _build_macro_frames([_D, _DL, _L], 'Z'),    # ↓↙← + HP (QCF)
    'shoryuken':  _build_macro_frames([_L, _D, _DL], 'Z'),    # ←↓↙ + HP (DP)
    'tatsumaki':  _build_macro_frames([_D, _DR, _R], 'C'),    # ↓↘→ + HK (QCB)
}

_MACRO_FACING_RIGHT = {
    # P2 on left side, facing RIGHT. Forward = RIGHT, Back = LEFT.
    'hadouken':   _build_macro_frames([_D, _DR, _R], 'Z'),    # ↓↘→ + HP (QCF)
    'shoryuken':  _build_macro_frames([_R, _D, _DR], 'Z'),    # →↓↘ + HP (DP)
    'tatsumaki':  _build_macro_frames([_D, _DL, _L], 'C'),    # ↓↙← + HK (QCB)
}

# Current active macro set — set by main loop each frame from X positions
_ACTIVE_MACROS = _MACRO_FACING_LEFT

_macro_name = None         # current running macro key or None
_macro_frame = 0           # current frame index within macro
_macro_key_was_down = {}   # rising-edge detection per macro key


def update_p2_facing(info):
    """Call every frame to update which macro set to use based on X positions."""
    global _ACTIVE_MACROS
    p1_x = info.get('agent_x', 0)
    p2_x = info.get('enemy_x', 0)
    if p2_x > p1_x:
        _ACTIVE_MACROS = _MACRO_FACING_LEFT   # P2 on right, facing left
    else:
        _ACTIVE_MACROS = _MACRO_FACING_RIGHT  # P2 on left, facing right


def get_human_action_p2(keys_pressed, combo_buffer=None):
    """获取玩家2（人类）的12维动作。

    实测映射 (retro index -> 游戏效果):
      action[10](X) -> 轻拳    action[1](A) -> 轻脚
      action[9](Y)  -> 中拳    action[0](B) -> 中脚
      action[11](Z) -> 重拳    action[8](C) -> 重脚

    键盘布局: JKL=轻中重拳  UIO=轻中重脚
    """
    global _macro_name, _macro_frame, _macro_key_was_down

    action = [0] * 12

    # ---- 一键必杀技宏 (Z/X/C) ----
    if _macro_name is not None:
        frames = _ACTIVE_MACROS[_macro_name]
        if _macro_frame < len(frames):
            action = frames[_macro_frame][:]
            _macro_frame += 1
            if _macro_frame == len(frames):
                _macro_name = None
                _macro_frame = 0
            return action
        else:
            _macro_name = None
            _macro_frame = 0

    # ---- Check for new macro trigger (rising edge only!) ----
    for name, macro in _SPECIAL_MACROS.items():
        key = macro['key']
        was_down = _macro_key_was_down.get(key, False)
        is_down = keys_pressed[key]
        _macro_key_was_down[key] = is_down
        if is_down and not was_down:  # rising edge: just pressed
            _macro_name = name
            _macro_frame = 0
            frames = _ACTIVE_MACROS[name]
            if frames:
                action = frames[0][:]
                _macro_frame = 1
            return action

    # ---- 方向键 ----
    if keys_pressed[pygame.K_w]:
        action[4] = 1
    if keys_pressed[pygame.K_s]:
        action[5] = 1
    if keys_pressed[pygame.K_a]:
        action[6] = 1
    if keys_pressed[pygame.K_d]:
        action[7] = 1

    # ---- 拳 (J=轻 K=中 L=重) ----
    if keys_pressed[pygame.K_j]:
        action[10] = 1
    if keys_pressed[pygame.K_k]:
        action[9] = 1
    if keys_pressed[pygame.K_l]:
        action[11] = 1

    # ---- 脚 (U=轻 I=中 O=重) ----
    if keys_pressed[pygame.K_u]:
        action[1] = 1
    if keys_pressed[pygame.K_i]:
        action[0] = 1
    if keys_pressed[pygame.K_o]:
        action[8] = 1

    # ---- START / MODE ----
    if keys_pressed[pygame.K_RETURN] or keys_pressed[pygame.K_SPACE]:
        action[2] = 1
        action[3] = 1

    # ---- 必杀技检测 ----
    if combo_buffer:
        triggered = combo_buffer.update(action)
        if triggered:
            move_name = combo_buffer.specials.get(triggered, {}).get('name', triggered)
            print(f'{move_name}!')

    return action


def get_ai_action_p1(stacked_obs, model):
    """获取AI的P1动作。自对弈模型输出24维，取前12维给P1（隆）。"""
    ai_action, _ = model.predict(stacked_obs)

    if isinstance(ai_action, np.ndarray):
        ai_action = ai_action.tolist()

    if len(ai_action) == 24:
        ai_action = ai_action[:12]

    return ai_action


def combine_actions(p1_action, p2_action):
    """合并两个玩家的动作
    P1 (AI): indices 0-11
    P2 (Human): indices 12-23
    """
    combined = [0] * 24
    combined[0:12] = p1_action
    combined[12:24] = p2_action
    return combined


def display_select_screen(screen, font, small_font, detected_name):
    """显示角色选择/赛前设置画面的提示"""
    if detected_name:
        # Character already selected — stage select / turbo setting screens
        title = font.render(f"SELECTED: {detected_name.upper()}", True, (100, 255, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
        screen.blit(title, title_rect)

        instr1 = small_font.render(
            "Press ENTER or SPACE (START) to advance (stage select / turbo)",
            True, (255, 255, 100),
        )
        instr1_rect = instr1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60))
        screen.blit(instr1, instr1_rect)

        instr2 = small_font.render(
            "Move: W/A/S/D  |  Confirm: J/K/L/U/I/O  |  ESC: Quit",
            True, (180, 180, 180),
        )
        instr2_rect = instr2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 35))
        screen.blit(instr2, instr2_rect)
    else:
        # Character select screen
        title = font.render("SELECT YOUR FIGHTER - P2", True, (255, 255, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
        screen.blit(title, title_rect)

        instr1 = small_font.render(
            "Move: W/A/S/D  |  Confirm: J/K/L/U/I/O  |  ENTER/SPACE=START  |  ESC: Quit",
            True, (220, 220, 220),
        )
        instr1_rect = instr1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60))
        screen.blit(instr1, instr1_rect)

        instr2 = small_font.render(
            "P1 (AI) locked as Ryu. Choose your character on the right side.",
            True, (180, 180, 180),
        )
        instr2_rect = instr2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 35))
        screen.blit(instr2, instr2_rect)


def display_info(screen, small_font, round_num, fps_mode, p1_hp, p2_hp,
                 combo_hints, character_name, p1_round_wins, p2_round_wins,
                 p1_model_name):
    """显示战斗 HUD：HP、回合、比分、角色名、必杀技提示。"""
    # HP bars
    bar_width = 200
    bar_height = 12
    bar_y = 15

    # P1 HP (left)
    p1_hp_clamped = max(0, p1_hp)
    pygame.draw.rect(screen, (80, 80, 80), (10, bar_y, bar_width, bar_height))
    hp_width = int(bar_width * min(p1_hp_clamped / 176, 1.0))
    hp_color = (0, 255, 0) if p1_hp > 44 else (255, 255, 0) if p1_hp > 0 else (255, 0, 0)
    pygame.draw.rect(screen, hp_color, (10, bar_y, hp_width, bar_height))
    p1_text = small_font.render(f"P1 AI({p1_model_name}): {p1_hp}", True, (255, 255, 255))
    screen.blit(p1_text, (10, 30))

    # P2 HP (right)
    p2_hp_clamped = max(0, p2_hp)
    bar_x = SCREEN_WIDTH - 10 - bar_width
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
    hp_width = int(bar_width * min(p2_hp_clamped / 176, 1.0))
    hp_color = (0, 255, 0) if p2_hp > 44 else (255, 255, 0) if p2_hp > 0 else (255, 0, 0)
    pygame.draw.rect(screen, hp_color, (bar_x + bar_width - hp_width, bar_y, hp_width, bar_height))
    p2_text = small_font.render(f"P2 You({character_name}): {p2_hp}", True, (255, 255, 255))
    p2_rect = p2_text.get_rect(topright=(SCREEN_WIDTH - 10, 30))
    screen.blit(p2_text, p2_rect)

    # Round & score
    round_text = small_font.render(
        f"Round: {round_num}  |  Score: AI {p1_round_wins} - {p2_round_wins} You",
        True, (255, 255, 255),
    )
    screen.blit(round_text, (10, 50))

    # Speed
    fps_text = small_font.render(f"Speed: {fps_mode} (F1-F3)", True, (180, 180, 180))
    screen.blit(fps_text, (10, 70))

    # Special moves hints
    if combo_hints:
        y = 95
        title = small_font.render(f"-- {character_name} Special Moves --", True, (255, 200, 100))
        screen.blit(title, (10, y))
        y += 18
        for name, hint in combo_hints:
            line = small_font.render(f"{name}: {hint}", True, (200, 200, 200))
            screen.blit(line, (10, y))
            y += 16

    # Bottom controls
    ctrl_line1 = small_font.render(
        "Move: W/A/S/D  |  J/K/L=Punch  U/I/O=Kick  |  ENTER=START",
        True, (180, 180, 180),
    )
    screen.blit(ctrl_line1, (10, SCREEN_HEIGHT - 65))
    ctrl_special = small_font.render(
        "Z=Hadouken  X=Shoryuken  C=Tatsumaki (one-button)",
        True, (255, 200, 100),
    )
    screen.blit(ctrl_special, (10, SCREEN_HEIGHT - 50))
    ctrl_line2 = small_font.render(
        "F1-F3: Speed  |  ESC: Quit",
        True, (180, 180, 180),
    )
    screen.blit(ctrl_line2, (10, SCREEN_HEIGHT - 30))


def show_round_result_overlay(screen, font, small_font, winner_name, round_num,
                               p1_wins, p2_wins, character_name, p1_name):
    """回合结束时的结果覆盖层"""
    if winner_name == 'AI':
        text = f"Round {round_num}: {p1_name} WINS!"
        color = (255, 100, 100)
    elif winner_name == 'You':
        text = f"Round {round_num}: You ({character_name}) WINS!"
        color = (100, 255, 100)
    else:
        text = f"Round {round_num}: DRAW"
        color = (255, 255, 0)

    result_text = font.render(text, True, color)
    text_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(result_text, text_rect)

    score_text = small_font.render(
        f"Score: AI {p1_wins} - {p2_wins} You",
        True, (200, 200, 200),
    )
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
    screen.blit(score_text, score_rect)


def show_match_result_screen(screen, font, small_font, match_winner,
                              p1_wins, p2_wins, character_name, p1_name):
    """比赛结束时的结果画面"""
    screen.fill((0, 0, 0))

    if match_winner == 'AI':
        text = f"{p1_name} WINS THE MATCH!"
        color = (255, 100, 100)
    else:
        text = f"You ({character_name}) WIN THE MATCH!"
        color = (100, 255, 100)

    result_text = font.render(text, True, color)
    text_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
    screen.blit(result_text, text_rect)

    score_text = small_font.render(
        f"Final Score: AI {p1_wins} - {p2_wins} You  |  Character: {character_name}",
        True, (255, 255, 255),
    )
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(score_text, score_rect)

    hint = small_font.render(
        "SPACE: New Match (re-select)  |  ESC: Quit",
        True, (150, 150, 150),
    )
    hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
    screen.blit(hint, hint_rect)

    pygame.display.flip()


def main():
    print("=" * 70)
    print("Street Fighter II - AI (Ryu) vs Human (Choose P2)")
    print("=" * 70)

    model_path = input(f"\nEnter model name (default: {DEFAULT_MODEL}): ").strip()
    if not model_path:
        model_path = DEFAULT_MODEL

    model_file = os.path.join(MODEL_DIR, f"{model_path}.zip")
    if not os.path.exists(model_file):
        print(f"\nModel file not found: {model_file}")
        print(f"Available models in {MODEL_DIR}:")
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.endswith('.zip'):
                    print(f"  - {f}")
        sys.exit(1)

    # Use short name for display
    p1_name = model_path[:12] if len(model_path) > 12 else model_path

    print("\nSelect game speed:")
    print("  1. Slow (15 FPS) - Easy to react")
    print("  2. Normal (30 FPS) - Balanced")
    print("  3. Fast (60 FPS) - Original speed")
    speed_choice = input("Choose (1/2/3, default: 3): ").strip()

    fps_map = {'1': 'slow', '2': 'normal', '3': 'fast'}
    fps_mode = fps_map.get(speed_choice, 'fast')

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Speed: {fps_mode} ({FPS_OPTIONS[fps_mode]} FPS)")
    print(f"  State: {STATE_2P}")
    print(f"  P1: AI Agent (Ryu)")
    print(f"  P2: Human (character select)")

    print("\n  Controls:")
    print("    Move:     W/A/S/D")
    print("    Punch:    J=Light  K=Medium  L=Heavy")
    print("    Kick:     U=Light  I=Medium  O=Heavy")
    print("    Specials: Z=Hadouken  X=Shoryuken  C=Tatsumaki (one-button)")
    print("    Start:    ENTER")
    print("    Speed:    F1(Slow)  F2(Normal)  F3(Fast)")
    print("    Quit:     ESC")

    print("\n" + "=" * 70)
    input("Press Enter to start...")

    pygame.init()
    pygame.display.set_caption("SFII - AI (Ryu) vs Human")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 20)

    print("\nLoading environment...")
    try:
        env = TwoPlayerEnv(GAME_NAME, STATE_2P, rendering=False, fps=fps_mode)
    except Exception as e:
        print(f"\nFailed to load state '{STATE_2P}': {e}")
        print(f"Make sure the state file exists in the game data folder.")
        pygame.quit()
        sys.exit(1)

    print("Loading AI model...")
    model = PPO.load(model_file, device='cpu')
    print("Ready!\n")

    combo_buffer = ComboBuffer()

    # Statistics across all matches
    total_matches = 0
    total_ai_wins = 0
    total_human_wins = 0

    try:
        while True:
            # ============================================================
            # Phase 1: Character Select
            # ============================================================
            obs, stacked_obs = env.reset()
            character_name = None
            combo_buffer.set_character('Ryu')  # default during select

            print("\n" + "=" * 70)
            print("CHARACTER SELECT")
            print("=" * 70)
            print("Select your character on screen (P2 side)...")
            print("  Move: W/A/S/D  |  Confirm: J/K/L/U/I/O  |  ENTER=START")
            print("  P1 (AI) locked as Ryu")

            selecting = True
            start_debug_logged = False
            select_frame = 0
            while selecting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    raise KeyboardInterrupt

                p2_action = get_human_action_p2(keys, None)  # No combo during select
                # P1 mirrors all P2 inputs — stage select & turbo setting are P1-controlled
                p1_select = p2_action[:]
                combined = combine_actions(p1_select, p2_action)

                # Debug: confirm START is being sent
                if keys[pygame.K_RETURN] or keys[pygame.K_SPACE]:
                    if not start_debug_logged:
                        p1_s = f"P1[{p1_select[2]},{p1_select[3]}]"
                        p2_s = f"P2[{p2_action[2]},{p2_action[3]}]"
                        print(f"  [DEBUG] START sent -> {p1_s} {p2_s}")
                        start_debug_logged = True
                else:
                    start_debug_logged = False
                obs, stacked_obs, rewards, done, info = env.step(combined)
                select_frame += 1

                enemy_hp = info.get('enemy_hp', 0)
                agent_hp = info.get('agent_hp', 0)

                # Skip select if already in battle (e.g., "ryu vs ryu self_play" state)
                if select_frame <= 5 and enemy_hp == 176 and agent_hp == 176:
                    character_name = 'Ryu'
                    selecting = False
                    break

                char_id = info.get('enemy_character', 0)
                if char_id != 0 and char_id in CHARACTER_IDS:
                    character_name = CHARACTER_IDS[char_id]

                # Detect battle start: both HPs at full (176)
                if enemy_hp == 176 and agent_hp == 176 and character_name is not None:
                    selecting = False

                # Render select screen
                try:
                    obs_surface = pygame.transform.scale(
                        pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1)),
                        (SCREEN_WIDTH, SCREEN_HEIGHT),
                    )
                    screen.blit(obs_surface, (0, 0))
                    display_select_screen(screen, font, small_font, character_name)
                    pygame.display.flip()
                except Exception:
                    pass

            # Character confirmed
            if character_name is None:
                character_name = 'Ryu'
            combo_buffer.set_character(character_name)
            combo_hints = combo_buffer.get_hints()

            print(f"\nCharacter selected: {character_name}")
            print("Battle starting...\n")

            # Reset frame stack for clean battle start
            stacked_obs = env.frame_processor.reset(obs)

            # ============================================================
            # Phase 2: Battle (continuous, detects round transitions)
            # ============================================================
            round_num = 0
            p1_round_wins = 0  # AI
            p2_round_wins = 0  # Human
            in_round_result = False
            round_result_timer = 0
            round_start_cooldown = 0
            last_winner_name = None
            battle_active = True  # starts active since we just entered battle

            match_running = True
            while match_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    raise KeyboardInterrupt

                # Speed controls
                if keys[pygame.K_F1]:
                    env.fps = FPS_OPTIONS['slow']
                    fps_mode = 'slow'
                    print("Speed: Slow (15 FPS)")
                elif keys[pygame.K_F2]:
                    env.fps = FPS_OPTIONS['normal']
                    fps_mode = 'normal'
                    print("Speed: Normal (30 FPS)")
                elif keys[pygame.K_F3]:
                    env.fps = FPS_OPTIONS['fast']
                    fps_mode = 'fast'
                    print("Speed: Fast (60 FPS)")

                # Get actions (pause AI during round result display)
                if battle_active and not in_round_result:
                    p1_action = get_ai_action_p1(stacked_obs, model)
                    p2_action = get_human_action_p2(keys, combo_buffer)
                else:
                    p1_action = [0] * 12
                    p2_action = get_human_action_p2(keys, None) if not in_round_result else [0] * 12

                combined = combine_actions(p1_action, p2_action)
                obs, stacked_obs, rewards, done, info = env.step(combined)

                p1_hp = info.get('agent_hp', 0)
                p2_hp = info.get('enemy_hp', 0)

                # Update facing direction for macro auto-flip
                update_p2_facing(info)

                # Cooldown tick
                if round_start_cooldown > 0:
                    round_start_cooldown -= 1

                # Detect NEW round: both HPs at full (after we've seen >0 rounds)
                if not battle_active and not in_round_result and p1_hp == 176 and p2_hp == 176:
                    battle_active = True
                    round_num += 1
                    round_start_cooldown = 60  # prevent false KO detection during transition
                    stacked_obs = env.frame_processor.reset(obs)
                    print(f"\n--- Round {round_num} - FIGHT! ---")

                # Init first round
                if round_num == 0 and battle_active:
                    round_num = 1
                    round_start_cooldown = 60
                    print(f"\n--- Round {round_num} - FIGHT! ---")

                # Detect round end: HP drops to/below 0 (only outside cooldown)
                if battle_active and not in_round_result and round_start_cooldown == 0 and (p1_hp <= 0 or p2_hp <= 0):
                    battle_active = False
                    in_round_result = True
                    round_result_timer = 90  # ~1.5s at 60fps

                    if p2_hp <= 0 and p1_hp <= 0:
                        last_winner_name = 'DRAW'
                    elif p2_hp <= 0:
                        last_winner_name = 'AI'
                        p1_round_wins += 1
                    else:
                        last_winner_name = 'You'
                        p2_round_wins += 1

                    print(f"  Round {round_num}: {last_winner_name} wins! "
                          f"(AI {p1_round_wins} - {p2_round_wins} You)")

                # Decrement result timer
                if in_round_result:
                    round_result_timer -= 1
                    if round_result_timer <= 0:
                        in_round_result = False

                # Render
                try:
                    obs_surface = pygame.transform.scale(
                        pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1)),
                        (SCREEN_WIDTH, SCREEN_HEIGHT),
                    )
                    screen.blit(obs_surface, (0, 0))
                    display_info(screen, small_font, round_num, fps_mode,
                                p1_hp, p2_hp, combo_hints, character_name,
                                p1_round_wins, p2_round_wins, p1_name)

                    # Round result overlay
                    if in_round_result and round_result_timer > 0 and last_winner_name:
                        show_round_result_overlay(
                            screen, font, small_font, last_winner_name,
                            round_num, p1_round_wins, p2_round_wins,
                            character_name, p1_name,
                        )

                    pygame.display.flip()
                except Exception:
                    pass

                # Check retro done (game over / continue screen)
                if done:
                    print("  [Game signaled done]")
                    match_running = False

                # Check match over (best of 3)
                if p1_round_wins >= 2 or p2_round_wins >= 2:
                    match_running = False

            # ============================================================
            # Match Result
            # ============================================================
            total_matches += 1

            if p1_round_wins >= 2:
                match_winner = 'AI'
                total_ai_wins += 1
            elif p2_round_wins >= 2:
                match_winner = 'You'
                total_human_wins += 1
            else:
                match_winner = 'DRAW'

            print(f"\n{'=' * 50}")
            print(f"MATCH {total_matches} OVER!")
            print(f"  Winner: {match_winner}")
            print(f"  Score:  AI {p1_round_wins} - {p2_round_wins} You")
            print(f"  Character: {character_name}")
            print(f"{'=' * 50}")

            show_match_result_screen(screen, font, small_font, match_winner,
                                     p1_round_wins, p2_round_wins,
                                     character_name, p1_name)

            # Wait for player input
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        if event.key == pygame.K_SPACE:
                            waiting = False
                pygame.time.Clock().tick(30)

            print("\nStarting new match...")
            # Loop back to character select

    except KeyboardInterrupt:
        print("\n\nGame ended by user.")

    finally:
        env.close()
        pygame.quit()

        if total_matches > 0:
            print(f"\n{'=' * 70}")
            print("FINAL STATISTICS")
            print(f"{'=' * 70}")
            print(f"Total Matches:        {total_matches}")
            print(f"AI Wins:              {total_ai_wins} ({total_ai_wins / total_matches:.1%})")
            print(f"Your Wins:            {total_human_wins} ({total_human_wins / total_matches:.1%})")
            print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
