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

# 10-dim → 12-dim action mapping
_A10_TO_A12 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 448

FPS_OPTIONS = {
    'slow': 15,
    'normal': 30,
    'fast': 60,
}


def _ensure_12d(action):
    """将 10/12 维动作统一转为 12 维"""
    if len(action) >= 12:
        return list(action[:12])
    a12 = [0] * 12
    for i10, i12 in enumerate(_A10_TO_A12):
        a12[i12] = int(action[i10]) if i10 < len(action) else 0
    return a12


def _clean_action(a12):
    """清理无效按键组合（与训练脚本一致）"""
    a12[2] = 0; a12[3] = 0
    if a12[6] and a12[7]: a12[6] = a12[7] = 0
    if a12[4] and a12[5]: a12[4] = a12[5] = 0
    atk = [i for i in [11, 10, 9, 8, 1, 0] if a12[i]]
    if len(atk) > 2:
        for i in atk[2:]: a12[i] = 0
    return a12


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


class TwoPlayerEnv:
    def __init__(self, game, state, fps='normal'):
        self.env = retro.make(
            game=game,
            state=state,
            players=2,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )
        self.fps = FPS_OPTIONS.get(fps, 30)
        self.clock = pygame.time.Clock()
        self.frame_processor = FrameStackProcessor(num_frames=9)

    def reset(self):
        obs = self.env.reset()
        stacked_obs = self.frame_processor.reset(obs)
        return obs, stacked_obs

    def step(self, combined_action):
        obs, rewards, done, info = self.env.step(combined_action)
        stacked_obs = self.frame_processor.step(obs)
        self.env.render()
        self.clock.tick(self.fps)
        return obs, stacked_obs, rewards, done, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def get_action(stacked_obs, model, player_label):
    """获取模型动作。兼容12维和24维输出。"""
    action, _ = model.predict(stacked_obs, deterministic=True)
    if isinstance(action, np.ndarray):
        action = action.tolist()

    if len(action) == 24:
        # v1 自对弈模型: [0:12]=P1, [12:24]=P2
        if player_label == 'P1':
            return action[:12]
        else:
            return action[12:24]

    # v2 或单玩家模型: 直接使用 12 维
    return action[:12]


def list_models():
    if not os.path.exists(MODEL_DIR):
        return []
    files = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.zip'):
            files.append(f.replace('.zip', ''))
    files.sort()
    return files


def display_hud(screen, font, small_font, round_num, p1_hp, p2_hp,
                p1_name, p2_name, fps_mode, p1_wins, p2_wins):
    # HP 条
    bar_width = 200
    bar_height = 12
    bar_y = 15

    # P1 HP (左侧)
    p1_hp_clamped = max(0, p1_hp)
    pygame.draw.rect(screen, (80, 80, 80), (10, bar_y, bar_width, bar_height))
    hp_width = int(bar_width * min(p1_hp_clamped / 176, 1.0))
    hp_color = (0, 255, 0) if p1_hp > 44 else (255, 255, 0) if p1_hp > 0 else (255, 0, 0)
    pygame.draw.rect(screen, hp_color, (10, bar_y, hp_width, bar_height))
    p1_text = small_font.render(f"{p1_name}: {p1_hp}", True, (255, 255, 255))
    screen.blit(p1_text, (10, 30))

    # P2 HP (右侧)
    p2_hp_clamped = max(0, p2_hp)
    bar_x = SCREEN_WIDTH - 10 - bar_width
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
    hp_width = int(bar_width * min(p2_hp_clamped / 176, 1.0))
    hp_color = (0, 255, 0) if p2_hp > 44 else (255, 255, 0) if p2_hp > 0 else (255, 0, 0)
    pygame.draw.rect(screen, hp_color, (bar_x + bar_width - hp_width, bar_y, hp_width, bar_height))
    p2_text = small_font.render(f"{p2_name}: {p2_hp}", True, (255, 255, 255))
    p2_rect = p2_text.get_rect(topright=(SCREEN_WIDTH - 10, 30))
    screen.blit(p2_text, p2_rect)

    # 回合 & 比分
    round_text = small_font.render(f"Round: {round_num}", True, (255, 255, 255))
    screen.blit(round_text, (10, 50))

    score_text = small_font.render(
        f"Score - {p1_name}: {p1_wins}  |  {p2_name}: {p2_wins}",
        True, (255, 255, 0),
    )
    screen.blit(score_text, (10, 70))

    # 速度
    fps_text = small_font.render(f"Speed: {fps_mode} (F1-F3)", True, (180, 180, 180))
    screen.blit(fps_text, (10, SCREEN_HEIGHT - 50))

    # 操作提示
    ctrl_text = small_font.render("SPACE: Next Round | ESC: Quit", True, (180, 180, 180))
    screen.blit(ctrl_text, (10, SCREEN_HEIGHT - 30))


def show_round_result(screen, font, winner, p1_hp, p2_hp,
                      p1_name, p2_name, round_num, p1_wins, p2_wins):
    if winner == 'P1':
        text = f"{p1_name} WINS!"
        color = (255, 100, 100)
    elif winner == 'P2':
        text = f"{p2_name} WINS!"
        color = (100, 100, 255)
    else:
        text = "DRAW"
        color = (255, 255, 0)

    result_text = font.render(text, True, color)
    text_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
    screen.blit(result_text, text_rect)

    detail = small_font.render(
        f"HP - {p1_name}: {p1_hp}  |  {p2_name}: {p2_hp}",
        True, (200, 200, 200),
    )
    detail_rect = detail.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10))
    screen.blit(detail, detail_rect)

    hint = small_font.render("Press SPACE for next round, ESC to quit",
                             True, (150, 150, 150))
    hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
    screen.blit(hint, hint_rect)
    pygame.display.flip()


def main():
    print("=" * 70)
    print("Street Fighter II - AI vs AI 对战视觉化")
    print("=" * 70)

    models_available = list_models()

    # P1 模型
    print(f"\n可用的模型 ({MODEL_DIR}):")
    for m in models_available:
        print(f"  - {m}")
    print()

    p1_name = input(f"P1 模型名称 (默认: {DEFAULT_MODEL}): ").strip()
    if not p1_name:
        p1_name = DEFAULT_MODEL

    p1_path = os.path.join(MODEL_DIR, f"{p1_name}.zip")
    if not os.path.exists(p1_path):
        print(f"\n[ERROR] P1 模型不存在: {p1_path}")
        sys.exit(1)

    # P2 模型
    p2_name = input(f"P2 模型名称 (默认: {DEFAULT_MODEL}): ").strip()
    if not p2_name:
        p2_name = DEFAULT_MODEL

    p2_path = os.path.join(MODEL_DIR, f"{p2_name}.zip")
    if not os.path.exists(p2_path):
        print(f"\n[ERROR] P2 模型不存在: {p2_path}")
        sys.exit(1)

    # 速度
    print("\n速度选择:")
    print("  1. Slow (15 FPS)")
    print("  2. Normal (30 FPS)")
    print("  3. Fast (60 FPS)")
    speed_choice = input("选择 (1/2/3, 默认: 3): ").strip()
    fps_map = {'1': 'slow', '2': 'normal', '3': 'fast'}
    fps_mode = fps_map.get(speed_choice, 'fast')

    print(f"\n配置:")
    print(f"  P1: {p1_name}")
    print(f"  P2: {p2_name}")
    print(f"  速度: {fps_mode} ({FPS_OPTIONS[fps_mode]} FPS)")

    print("\n" + "=" * 70)
    input("按 Enter 开始...")

    pygame.init()
    pygame.display.set_caption(f"SFII - {p1_name} vs {p2_name}")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 20)

    print("\n加载环境...")
    env = TwoPlayerEnv(GAME_NAME, STATE_2P, fps=fps_mode)

    print("加载模型...")
    model_p1 = PPO.load(p1_path, device='cpu')
    model_p2 = PPO.load(p2_path, device='cpu')
    print("就绪!\n")

    round_num = 0
    p1_wins = 0
    p2_wins = 0

    try:
        running = True
        while running:
            round_num += 1
            print(f"\n{'=' * 50}")
            print(f"Round {round_num}")
            print(f"{'=' * 50}")

            obs, stacked_obs = env.reset()
            done = False
            step_count = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    raise KeyboardInterrupt
                if keys[pygame.K_F1]:
                    env.fps = FPS_OPTIONS['slow']
                    fps_mode = 'slow'
                elif keys[pygame.K_F2]:
                    env.fps = FPS_OPTIONS['normal']
                    fps_mode = 'normal'
                elif keys[pygame.K_F3]:
                    env.fps = FPS_OPTIONS['fast']
                    fps_mode = 'fast'

                # 两个 AI 各自推理
                # P1: 正常视角（面朝右，控制左侧角色）
                p1_a12 = _ensure_12d(get_action(stacked_obs, model_p1, 'P1'))

                # P2: 翻转画面 + 镜像左右键（面朝左，控制右侧角色）
                p2_obs = stacked_obs[:, ::-1, :].copy()
                p2_a12 = _ensure_12d(get_action(p2_obs, model_p2, 'P2'))
                p2_a12[6], p2_a12[7] = p2_a12[7], p2_a12[6]  # 镜像 LEFT/RIGHT
                p2_a12 = _clean_action(p2_a12)

                combined = list(p1_a12) + list(p2_a12)

                obs, stacked_obs, rewards, done, info = env.step(combined)
                step_count += 1

                p1_hp = info.get('agent_hp', 0)
                p2_hp = info.get('enemy_hp', 0)

                # 渲染
                try:
                    obs_surface = pygame.transform.scale(
                        pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1)),
                        (SCREEN_WIDTH, SCREEN_HEIGHT),
                    )
                    screen.blit(obs_surface, (0, 0))
                    display_hud(screen, font, small_font, round_num,
                                p1_hp, p2_hp, p1_name, p2_name,
                                fps_mode, p1_wins, p2_wins)
                    pygame.display.flip()
                except Exception:
                    pass

            # 回合结束
            if p2_hp <= 0:
                winner = 'P1'
                p1_wins += 1
            elif p1_hp <= 0:
                winner = 'P2'
                p2_wins += 1
            else:
                winner = 'DRAW'

            print(f"结果: {winner}  |  P1 HP: {p1_hp}  |  P2 HP: {p2_hp}  |  步数: {step_count}")
            print(f"总比分 - {p1_name}: {p1_wins}  |  {p2_name}: {p2_wins}")

            # 显示结果并等待
            show_round_result(screen, font, winner, p1_hp, p2_hp,
                              p1_name, p2_name, round_num, p1_wins, p2_wins)

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

    except KeyboardInterrupt:
        print("\n\n退出。")

    finally:
        env.close()
        pygame.quit()

        if round_num > 0:
            print(f"\n{'=' * 50}")
            print("最终统计")
            print(f"{'=' * 50}")
            print(f"总回合数:        {round_num}")
            print(f"{p1_name} 胜:    {p1_wins} ({p1_wins / round_num:.1%})")
            print(f"{p2_name} 胜:    {p2_wins} ({p2_wins / round_num:.1%})")
            print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
