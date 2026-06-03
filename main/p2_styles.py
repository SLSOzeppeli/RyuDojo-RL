"""三种风格化P2：JumpInBully（跳入压制）、TurtleGod（极致龟缩）、HadoukenSpammer（波升压起身）。"""
import random


class JumpInBullyP2:
    """高频跳入+投/中段择，惩罚蹲防。"""

    def __init__(self):
        self._prev_agent_hp = 176
        self._prev_agent_status = 512

    def act(self, info):
        a = [0] * 12
        enemy_x = info.get('enemy_x', 128)
        agent_x = info.get('agent_x', 128)
        agent_y = info.get('agent_y', 0)
        agent_hp = info.get('agent_hp', 176)
        agent_status = info.get('agent_status', 512)

        dx = enemy_x - agent_x
        distance = abs(dx)

        p1_jumping = agent_y > 180
        p1_is_attacking = agent_status >= 520
        p1_is_idle = agent_status in (512, 516)
        p1_just_whiffed = self._prev_agent_status >= 520 and p1_is_idle
        p1_got_hit = agent_hp < self._prev_agent_hp

        self._prev_agent_hp = agent_hp
        self._prev_agent_status = agent_status

        # Anti-air: essential even for jump-in style
        if p1_jumping and distance < 130:
            a[5] = 1
            a[10] = 1  # X — fierce punch
            self._walk_toward(dx, a)
            return a

        # Jump-in pressure: the core strategy
        if distance < 110 and random.random() < 0.45:
            a[4] = 1  # UP — jump
            self._walk_toward(dx, a)
            if random.random() < 0.6:
                a[11] = 1  # Z — jump fierce kick (overhead!)
            else:
                a[10] = 1  # X — jump fierce punch
            return a

        # Cross-up attempt: jump when close, land behind
        if distance < 50 and random.random() < 0.25:
            a[4] = 1
            self._walk_toward(dx, a)
            a[0] = 1  # B — light kick (fast)
            return a

        # Throws at close range (beats crouch-block)
        if distance < 30 and random.random() < 0.40:
            self._walk_toward(dx, a)
            a[10] = 1  # X — throw
            return a

        # Whiff punish (simplified)
        if p1_just_whiffed and distance < 70:
            self._walk_toward(dx, a)
            if random.random() < 0.5:
                a[5] = 1; a[11] = 1  # Z — sweep
            else:
                a[10] = 1  # X — fierce punch
            return a

        # Meaty pressure on hit
        if p1_got_hit and distance < 50:
            self._walk_toward(dx, a)
            a[10] = 1  # X
            return a

        # General approach
        if distance > 50:
            self._walk_toward(dx, a)
            return a

        # Close range mixup
        r = random.random()
        if r < 0.35:
            self._walk_toward(dx, a)
            a[10] = 1  # X — throw
        elif r < 0.55:
            a[5] = 1; a[0] = 1  # B — crouch light kick
        elif r < 0.75:
            a[5] = 1; a[11] = 1  # Z — crouch fierce kick
        else:
            a[0] = 1  # standing light kick
        return a

    def _walk_toward(self, dx, a):
        if dx > 0: a[6] = 1
        else: a[7] = 1


class TurtleGodP2:
    """极致龟缩：蹲防+确反，只在确认时出招。"""

    def __init__(self):
        self._prev_agent_hp = 176
        self._prev_agent_status = 512
        self._prev_agent_y = 0

    def act(self, info):
        a = [0] * 12
        enemy_x = info.get('enemy_x', 128)
        agent_x = info.get('agent_x', 128)
        agent_y = info.get('agent_y', 0)
        agent_hp = info.get('agent_hp', 176)
        agent_status = info.get('agent_status', 512)

        dx = enemy_x - agent_x
        distance = abs(dx)

        p1_jumping = agent_y > 180
        p1_is_idle = agent_status in (512, 516)
        p1_just_whiffed = self._prev_agent_status >= 520 and p1_is_idle
        p1_got_hit = agent_hp < self._prev_agent_hp

        self._prev_agent_hp = agent_hp
        self._prev_agent_status = agent_status
        self._prev_agent_y = agent_y

        # Anti-air: crouching fierce punch
        if p1_jumping and distance < 120:
            a[5] = 1
            a[10] = 1  # X
            return a

        # Whiff punish — only at close range, guaranteed safe
        if p1_just_whiffed and distance < 50:
            a[5] = 1
            a[11] = 1  # Z — sweep
            return a

        # Punish on hit (rare since we block most things)
        if p1_got_hit and distance < 40:
            a[5] = 1
            a[0] = 1  # B — safest fast poke
            return a

        # Walk away + crouch block (core turtle tactic)
        self._walk_back(dx, a)
        a[5] = 1  # DOWN — crouch block

        # Occasional extra-safe poke to test spacing
        if distance < 45 and random.random() < 0.10:
            a[5] = 1
            a[0] = 1  # B — crouching light kick

        # Fireball at extreme range to force approach into our wall
        if distance > 120 and random.random() < 0.20:
            a = [0] * 12
            self._walk_toward(dx, a)
            a[5] = 1
            a[10] = 1  # X

        return a

    def _walk_toward(self, dx, a):
        if dx > 0: a[6] = 1
        else: a[7] = 1

    def _walk_back(self, dx, a):
        if dx > 0: a[7] = 1
        else: a[6] = 1


class HadoukenSpammerP2:
    """全距离波动拳压制+倒地起身压波动，模拟人类Ryu波升打法。"""

    def __init__(self):
        self._prev_agent_hp = 176
        self._prev_agent_status = 512
        self._prev_agent_y = 0
        self._oki_timer = 0  # counts steps since hitting P1, for wake-up timing

    def act(self, info):
        a = [0] * 12
        enemy_x = info.get('enemy_x', 128)
        agent_x = info.get('agent_x', 128)
        agent_y = info.get('agent_y', 0)
        agent_hp = info.get('agent_hp', 176)
        agent_status = info.get('agent_status', 512)

        dx = enemy_x - agent_x
        distance = abs(dx)

        p1_jumping = agent_y > 180
        p1_is_idle = agent_status in (512, 516)
        p1_just_whiffed = self._prev_agent_status >= 520 and p1_is_idle
        p1_got_hit = agent_hp < self._prev_agent_hp

        # Oki timer: track steps since P1 took a hit
        if p1_got_hit and distance > 30:
            self._oki_timer = 5  # knockdown from range → setup oki
        elif self._oki_timer > 0:
            self._oki_timer -= 1

        self._prev_agent_hp = agent_hp
        self._prev_agent_status = agent_status
        self._prev_agent_y = agent_y

        # ── P0: Anti-air ──
        if p1_jumping:
            if distance < 130:
                a[5] = 1
                a[10] = 1  # X — crouching fierce punch
            return a

        # ── P1: Hadouken Oki — fireball on wake-up ──
        # When P1 was knocked down and is getting up, fireball right on top of them
        if self._oki_timer > 0 and distance < 70 and random.random() < 0.55:
            a[5] = 1
            self._walk_toward(dx, a)
            a[10] = 1  # X — hadouken on wake-up!
            return a

        # ── P2: Fireball at ALL ranges! ──
        fireball_chance = 0.50 if distance > 60 else 0.35 if distance > 30 else 0.22
        if random.random() < fireball_chance:
            a[5] = 1
            self._walk_toward(dx, a)
            a[10] = 1  # X — fierce hadouken
            return a

        # ── P3: Close range — fireball anyway (meaty oki, beats wake-up buttons) ──
        if distance < 55 and random.random() < 0.30:
            a[5] = 1
            self._walk_toward(dx, a)
            a[10] = 1  # X — point-blank hadouken > crouch-block
            return a

        # ── P4: Whiff punish ──
        if p1_just_whiffed and distance < 80:
            a[5] = 1
            a[11] = 1  # Z — sweep knockdown → oki setup
            self._walk_toward(dx, a)
            return a

        # ── P5: After landing a hit, chase with fireball ──
        if p1_got_hit and distance < 60:
            a[5] = 1
            self._walk_toward(dx, a)
            a[10] = 1  # X — hadouken hits on wake-up
            return a

        # ── Default: maintain fireball range, walk toward ──
        self._walk_toward(dx, a)
        return a

    def _walk_toward(self, dx, a):
        if dx > 0: a[6] = 1
        else: a[7] = 1

    def _walk_back(self, dx, a):
        if dx > 0: a[7] = 1
        else: a[6] = 1


# Registry for easy import
ALL_STYLES = {
    'jumpin': JumpInBullyP2,
    'turtle': TurtleGodP2,
    'hadouken': HadoukenSpammerP2,
}
