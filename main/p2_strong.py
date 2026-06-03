"""综合强P2：升龙对空、防御预读、挥空惩罚、版边压制、波动牵制、血量自适应切换。"""
import random


class RuleBasedP2Strong:

    def __init__(self, noise_prob=0.0):
        self.noise_prob = noise_prob
        self._prev_agent_hp = 176
        self._prev_agent_status = 512
        self._prev_agent_x = 128
        self._p1_was_jumping = False
        self._p1_was_attacking = False
        self._mode = 'rush'
        self._mode_timer = 0

    def act(self, info):
        a = [0] * 12  # [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]

        agent_x = info.get('agent_x', 128)
        enemy_x = info.get('enemy_x', 128)
        agent_y = info.get('agent_y', 0)
        agent_hp = info.get('agent_hp', 176)
        enemy_hp = info.get('enemy_hp', 176)
        agent_status = info.get('agent_status', 512)

        dx = enemy_x - agent_x       # >0: P1 left, P2 walks RIGHT
        distance = abs(dx)

        # --- Detect state transitions ---
        p1_jumping = agent_y > 180
        p1_just_landed = self._p1_was_jumping and not p1_jumping
        p1_got_hit = agent_hp < self._prev_agent_hp

        # Status: 512=idle, 516=walk, 522+=attacking
        p1_is_attacking = agent_status >= 520
        p1_is_idle = agent_status in (512, 516)
        # P1 just exited an attack: was attacking last step, now idle/walk
        p1_just_whiffed = self._p1_was_attacking and p1_is_idle

        # Corner detection
        p1_cornered_left = agent_x < 40 and enemy_x > agent_x
        p1_cornered_right = agent_x > 280 and enemy_x < agent_x
        p1_is_cornered = p1_cornered_left or p1_cornered_right

        # Update tracked state
        self._p1_was_jumping = p1_jumping
        self._p1_was_attacking = p1_is_attacking
        self._prev_agent_hp = agent_hp
        self._prev_agent_status = agent_status
        self._prev_agent_x = agent_x

        # ================================================================
        # P0: ANTI-AIR (highest priority)
        # ================================================================
        if p1_jumping and distance < 130:
            a[5] = 1   # DOWN (crouching)
            a[10] = 1  # X (fierce punch anti-air)
            if distance < 60:
                self._walk_toward(dx, a)
            return a

        # ================================================================
        # P1: WHIFF PUNISH — P1 just missed an attack, counter now
        # ================================================================
        if p1_just_whiffed and distance < 75:
            if distance < 40:
                # Close: max damage — crouching fierce kick sweep
                a[5] = 1
                a[11] = 1  # Z
            elif distance < 55:
                # Mid: advancing fierce punch
                self._walk_toward(dx, a)
                a[10] = 1  # X
            else:
                # Far whiff: chase with sweep
                self._walk_toward(dx, a)
                a[5] = 1
                a[11] = 1  # Z
            return a

        # ================================================================
        # P2: CORNER PRESSURE — P1 is trapped, never let them escape
        # ================================================================
        if p1_is_cornered:
            if distance > 60:
                # Close the gap fast
                self._walk_toward(dx, a)
                if random.random() < 0.3:
                    a[4] = 1   # Jump in
                    a[11] = 1  # Z — jump fierce kick
            elif distance > 25:
                # Mid-range corner: long normals to keep them pinned
                self._walk_toward(dx, a)
                r = random.random()
                if r < 0.35:
                    a[5] = 1; a[11] = 1  # Z — sweep knockdown
                elif r < 0.60:
                    a[5] = 1; a[9] = 1   # Y — crouching medium kick
                elif r < 0.80:
                    a[10] = 1  # X — standing fierce punch
                else:
                    a[4] = 1; a[11] = 1  # Jump fierce (overhead)
            else:
                # Point blank: relentless mixup
                r = random.random()
                if r < 0.30:
                    self._walk_toward(dx, a)
                    a[10] = 1  # X — throw
                elif r < 0.50:
                    a[5] = 1; a[0] = 1   # B — crouching light kick
                elif r < 0.68:
                    a[5] = 1; a[11] = 1  # Z — crouching fierce kick
                elif r < 0.83:
                    a[10] = 1  # X — standing fierce
                else:
                    a[5] = 1; a[8] = 1   # C — crouching medium punch
            return a

        # ================================================================
        # P3: LANDING PUNISH — P1 just landed from a jump, sweep
        # ================================================================
        if p1_just_landed and distance < 80:
            a[5] = 1
            a[11] = 1  # Z (sweep on landing)
            self._walk_toward(dx, a)
            return a

        # ================================================================
        # P4: MEATY PRESSURE — P1 just took a hit, keep hitting
        # ================================================================
        if p1_got_hit and distance < 60:
            if distance < 30:
                if random.random() < 0.45:
                    self._walk_toward(dx, a)
                    a[10] = 1  # X — throw on wake-up
                else:
                    a[5] = 1
                    a[0] = 1  # B — crouching light kick meaty
            else:
                self._walk_toward(dx, a)
                a[10] = 1  # X — chase fierce punch
            return a

        # ================================================================
        # P5: HADOUKEN ZONING — long range, force approach
        # ================================================================
        if distance > 90 and random.random() < 0.30:
            # Single-step hadouken: DOWN + FORWARD + PUNCH
            # With 6-frame hold, this often completes a quarter-circle
            a[5] = 1   # DOWN
            self._walk_toward(dx, a)  # FORWARD
            a[10] = 1  # X — fierce punch
            return a

        # ================================================================
        # Mode switching (only when no high-priority action triggered)
        # ================================================================
        self._mode_timer -= 1
        if self._mode_timer <= 0:
            hp_lead = enemy_hp - agent_hp
            if hp_lead > 50:
                r = random.random()
                self._mode = 'turtle' if r < 0.70 else 'poke' if r < 0.22 else 'rush'
            elif hp_lead > 0:
                r = random.random()
                self._mode = 'turtle' if r < 0.35 else 'poke' if r < 0.40 else 'rush'
            elif hp_lead > -50:
                r = random.random()
                self._mode = 'rush' if r < 0.60 else 'poke' if r < 0.30 else 'turtle'
            else:
                self._mode = 'rush'
            self._mode_timer = random.randint(2, 8)

        # ================================================================
        # Execute mode
        # ================================================================
        if self._mode == 'turtle':
            return self._turtle(dx, distance)
        elif self._mode == 'poke':
            return self._poke(dx, distance)
        else:
            return self._rush(dx, distance)

    # ================================================================
    # MODES
    # ================================================================

    def _turtle(self, dx, distance):
        """Preemptive defense: crouch-block + walk back. Very hard to hit."""
        a = [0] * 12

        # Walk back + crouch block
        self._walk_back(dx, a)
        a[5] = 1  # DOWN — crouch block covers high AND low in SF2

        # Occasional fast poke to interrupt approach
        if distance < 45 and random.random() < 0.20:
            a[5] = 1
            a[0] = 1  # B — crouching light kick (fastest, safest)

        # Hadouken at long range to force P1 to approach into our defense
        if distance > 100 and random.random() < 0.25:
            a = [0] * 12
            self._walk_toward(dx, a)
            a[5] = 1
            a[10] = 1  # X

        return a

    def _poke(self, dx, distance):
        """Mid-range spacing: stay at optimal range, poke with normals."""
        a = [0] * 12

        if distance > 85:
            # Approach + hadouken
            self._walk_toward(dx, a)
            if random.random() < 0.20:
                a[5] = 1; a[10] = 1  # X — hadouken

        elif distance > 50:
            # Optimal poke: crouching medium kick (safe, good reach)
            a[5] = 1
            a[9] = 1  # Y — medium kick
            # Spacing adjustment
            if distance > 65:
                self._walk_toward(dx, a)
            elif distance < 55:
                self._walk_back(dx, a)

        elif distance > 30:
            # Close-mid: sweep or back up to maintain spacing
            if random.random() < 0.50:
                a[5] = 1
                a[11] = 1  # Z — sweep
            else:
                self._walk_back(dx, a)
                a[5] = 1  # crouch
                if random.random() < 0.25:
                    a[9] = 1  # Y — poke while retreating

        else:
            # Too close for poking, back up
            self._walk_back(dx, a)
            a[5] = 1  # crouch
            if random.random() < 0.35:
                a[0] = 1  # B — fast kick to create space

        return a

    def _rush(self, dx, distance):
        """Full aggression: close distance, nonstop mixup."""
        a = [0] * 12

        if distance > 100:
            self._walk_toward(dx, a)
            return a

        elif distance > 55:
            self._walk_toward(dx, a)
            if random.random() < 0.30:
                a[5] = 1; a[11] = 1  # Z — advancing sweep

        elif distance > 30:
            r = random.random()
            if r < 0.35:
                a[4] = 1   # UP — jump-in
                self._walk_toward(dx, a)
                a[11] = 1  # Z — jump fierce kick
            elif r < 0.65:
                self._walk_toward(dx, a)
                a[10] = 1  # X — advancing fierce punch
            else:
                a[5] = 1; a[11] = 1  # Z — sweep
                self._walk_toward(dx, a)

        else:
            # Close range mixup
            r = random.random()
            if r < 0.22:
                self._walk_toward(dx, a)
                a[10] = 1  # X — throw
            elif r < 0.40:
                a[5] = 1; a[0] = 1   # B — crouching light kick
            elif r < 0.55:
                a[0] = 1  # B — standing light kick
            elif r < 0.68:
                a[5] = 1; a[8] = 1   # C — crouching medium punch
            elif r < 0.80:
                a[10] = 1  # X — standing fierce
            elif r < 0.90:
                a[5] = 1; a[11] = 1  # Z — crouching fierce kick
            else:
                self._walk_back(dx, a)
                a[5] = 1  # bait → creates whiff punish opportunity

        return a

    # P2在右侧面向左: dx>0→P1在左边→走近用LEFT(a[6]=1)
    def _walk_toward(self, dx, a):
        if dx > 0:
            a[6] = 1
        else:
            a[7] = 1

    def _walk_back(self, dx, a):
        if dx > 0:
            a[7] = 1
        else:
            a[6] = 1
