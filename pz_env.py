import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from card_stats import SPELL_STATS, TOWER_STATS, TROOP_STATS, TROOP_TYPE_MAP
from game_objects import Tower, Troop


class ClashRoyalePZ(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "clash_royale_pz_v1"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        self.arena_width = 18.0
        self.arena_height = 32.0

        # Action: [type, x, y]
        # type: 0 - Nothing, 1 - Knight, 2 - Archer, 3 - Fireball
        # x: 0 to 17
        # y: 0 to 31
        self.action_spaces = {
            agent: spaces.MultiDiscrete([4, 18, 32]) for agent in self.possible_agents
        }

        # Observation Space:
        # [elixir, self_king_hp, self_l_hp, self_r_hp, enemy_king_hp, enemy_l_hp, enemy_r_hp,
        #  self_t1_type, self_t1_x, self_t1_y, self_t1_hp, ... (x10)
        #  enemy_t1_type, enemy_t1_x, enemy_t1_y, enemy_t1_hp] ... (x10)
        # Total: 1 + 3 + 3 + (10 * 4) + (10 * 4) = 87
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(87,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.max_elixir = 10
        self.dt = 0.2
        self.max_steps = 1500

        self.bridge_y = 16.0
        self.left_bridge_x = 3.5
        self.right_bridge_x = 14.5

        self.troop_type_map = TROOP_TYPE_MAP

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.time = 0.0

        # Game State
        self.elixir = {"player_0": 5.0, "player_1": 5.0}

        # Towers: [King, Left Princess, Right Princess]
        # Coordinates: Player 0 is bottom (Y=0), Player 1 is top (Y=31)
        k_s = TOWER_STATS["king"]
        p_s = TOWER_STATS["princess"]
        self.towers = {
            "player_0": [
                Tower(
                    0,
                    (9.0, 0.5),
                    "king",
                    health=k_s["health"],
                    attack_damage=k_s["damage"],
                    attack_range=k_s["range"],
                    attack_speed=k_s["speed"],
                ),
                Tower(
                    0,
                    (3.5, 3.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),
                Tower(
                    0,
                    (14.5, 3.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),
            ],
            "player_1": [
                Tower(
                    1,
                    (9.0, 31.5),
                    "king",
                    health=k_s["health"],
                    attack_damage=k_s["damage"],
                    attack_range=k_s["range"],
                    attack_speed=k_s["speed"],
                ),
                Tower(
                    1,
                    (14.5, 28.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),  # Symmetric
                Tower(
                    1,
                    (3.5, 28.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),
            ],
        }
        self.troops = {"player_0": [], "player_1": []}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        enemy = "player_1" if agent == "player_0" else "player_0"
        is_p0 = agent == "player_0"
        obs = np.zeros(87, dtype=np.float32)

        # 1. Elixir (1)
        obs[0] = self.elixir[agent] / self.max_elixir

        # 2. Own Towers HP (3)
        for i, t in enumerate(self.towers[agent]):
            obs[1 + i] = t.health / t.max_health

        # 3. Enemy Towers HP (3)
        for i, t in enumerate(self.towers[enemy]):
            obs[4 + i] = t.health / t.max_health

        def get_rel_coords(pos):
            x, y = pos
            rx = x if is_p0 else (self.arena_width - x)
            ry = y if is_p0 else (self.arena_height - y)
            return rx / self.arena_width, ry / self.arena_height

        # 4. Own Troops (40)
        self_troops = self.troops[agent]
        for i in range(min(len(self_troops), 10)):
            rx, ry = get_rel_coords(self_troops[i].position)
            idx = 7 + i * 4
            obs[idx] = self.troop_type_map.get(self_troops[i].name, 0.0)
            obs[idx + 1] = rx
            obs[idx + 2] = ry
            obs[idx + 3] = self_troops[i].health / self_troops[i].max_health

        # 5. Enemy Troops (40)
        enemy_troops = self.troops[enemy]
        for i in range(min(len(enemy_troops), 10)):
            rx, ry = get_rel_coords(enemy_troops[i].position)
            idx = 47 + i * 4
            obs[idx] = self.troop_type_map.get(enemy_troops[i].name, 0.0)
            obs[idx + 1] = rx
            obs[idx + 2] = ry
            obs[idx + 3] = enemy_troops[i].health / enemy_troops[i].max_health

        return obs

    def step(self, actions):
        rewards = {"player_0": -0.001, "player_1": -0.001}
        terminations = {"player_0": False, "player_1": False}
        truncations = {"player_0": False, "player_1": False}
        infos = {"player_0": {}, "player_1": {}}

        self.current_step += 1
        self.time += self.dt

        # 1. Elixir Regen
        for agent in self.possible_agents:
            self.elixir[agent] = min(
                self.max_elixir, self.elixir[agent] + (self.dt / 2.8)
            )
            if self.elixir[agent] >= self.max_elixir:
                rewards[agent] -= 0.01

        # 2. Process Actions
        for agent, action in actions.items():
            action_type, ax, ay = action
            enemy = "player_1" if agent == "player_0" else "player_0"

            # Clamp and convert coordinates
            # player_0 plays on Y [0, 15], player_1 on Y [17, 31]
            if action_type in [1, 2]:  # Troops
                if agent == "player_0":
                    actual_y = min(float(ay), 15.0)
                else:
                    actual_y = max(float(ay), 17.0)
                actual_x = float(ax)
            else:  # Spells can be anywhere
                actual_x, actual_y = float(ax), float(ay)

            actual_pos = (actual_x, actual_y)

            if (
                action_type == 1
                and self.elixir[agent] >= TROOP_STATS["Knight"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["Knight"]
                self.elixir[agent] -= stats["cost"]
                self.troops[agent].append(
                    Troop(
                        agent,
                        "Knight",
                        stats["health"],
                        stats["damage"],
                        stats["speed"],
                        stats["attack_range"],
                        stats["attack_speed"],
                        actual_pos,
                        stats["cost"],
                    )
                )
            elif (
                action_type == 2
                and self.elixir[agent] >= TROOP_STATS["Archer"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["Archer"]
                self.elixir[agent] -= stats["cost"]
                # Spawn a pair of Archers with x-offset
                offsets = [-1.0, 1.0]
                for dx in offsets:
                    spawn_x = max(0.0, min(self.arena_width, actual_x + dx))
                    self.troops[agent].append(
                        Troop(
                            agent,
                            "Archer",
                            stats["health"],
                            stats["damage"],
                            stats["speed"],
                            stats["attack_range"],
                            stats["attack_speed"],
                            (spawn_x, actual_y),
                            stats["cost"],
                        )
                    )
            elif (
                action_type == 3
                and self.elixir[agent] >= SPELL_STATS["Fireball"]["cost"]
            ):
                stats = SPELL_STATS["Fireball"]
                self.elixir[agent] -= stats["cost"]
                # Fireball 2D logic
                for et in self.towers[enemy]:
                    if (
                        math.hypot(et.position[0] - actual_x, et.position[1] - actual_y)
                        <= stats["radius"]
                    ):
                        et.take_damage(stats["tower_damage"])
                for et in self.troops[enemy]:
                    if (
                        math.hypot(et.position[0] - actual_x, et.position[1] - actual_y)
                        <= stats["radius"]
                    ):
                        et.take_damage(stats["damage"])

        # 3. Tower Defense Logic (2D)
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            enemy_units = [t for t in self.troops[enemy] if t.is_alive()]

            for tower in self.towers[agent]:
                if not tower.is_alive():
                    continue

                # Find closest enemy unit
                if enemy_units:
                    target = min(
                        enemy_units,
                        key=lambda t: math.hypot(
                            t.position[0] - tower.position[0],
                            t.position[1] - tower.position[1],
                        ),
                    )
                    dist = math.hypot(
                        target.position[0] - tower.position[0],
                        target.position[1] - tower.position[1],
                    )

                    if (
                        dist <= tower.attack_range
                        and (self.time - tower.last_attack_time) >= tower.attack_speed
                    ):
                        target.take_damage(tower.attack_damage)
                        tower.last_attack_time = self.time

        # 4. Movement and Combat (Troops) with Bridge Pathfinding
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            is_p0 = agent == "player_0"

            for t in self.troops[agent]:
                # Targets: enemy towers and enemy troops
                potential_targets = [
                    et for et in self.troops[enemy] if et.is_alive()
                ] + [tw for tw in self.towers[enemy] if tw.is_alive()]

                if not potential_targets:
                    continue

                target = min(
                    potential_targets,
                    key=lambda pt: math.hypot(
                        pt.position[0] - t.position[0], pt.position[1] - t.position[1]
                    ),
                )
                target_pos = target.position

                # Simple Pathfinding: if target is across the river, go to bridge first
                target_pos = target.position
                is_across = (
                    is_p0 and target_pos[1] > 16.0 and t.position[1] < 16.0
                ) or (not is_p0 and target_pos[1] < 16.0 and t.position[1] > 16.0)

                # Check if currently in the river corridor
                in_river_zone = 15.5 <= t.position[1] <= 16.5

                move_target = target_pos
                if is_across or in_river_zone:
                    # Pick closest bridge center
                    d_left = abs(t.position[0] - self.left_bridge_x)
                    d_right = abs(t.position[0] - self.right_bridge_x)
                    bridge_x = (
                        self.left_bridge_x if d_left < d_right else self.right_bridge_x
                    )

                    if in_river_zone:
                        # If already in the river zone, move straight across (lock X)
                        move_target = (bridge_x, target_pos[1])
                    else:
                        # Move toward bridge entrance
                        entrance_y = 15.5 if is_p0 else 16.5
                        move_target = (bridge_x, entrance_y)

                dist = math.hypot(
                    target_pos[0] - t.position[0], target_pos[1] - t.position[1]
                )
                if dist <= t.attack_range:
                    if (self.time - t.last_attack_time) >= t.attack_speed:
                        target.take_damage(t.damage)
                        t.last_attack_time = self.time
                        if isinstance(target, Tower):
                            rewards[agent] += 0.05
                        else:
                            rewards[agent] += 0.001
                    # Note: No movement if in range, even if on cooldown
                else:
                    old_pos = list(t.position)
                    t.move(self.dt, move_target)

                    # River Collision Enforcement (Hard Block)
                    tx, ty = t.position
                    on_bridge = (
                        abs(tx - self.left_bridge_x) < 0.8
                        or abs(tx - self.right_bridge_x) < 0.8
                    )

                    if 15.5 < ty < 16.5:
                        if not on_bridge:
                            # Cannot enter river from the side
                            t.position = old_pos
                        else:
                            # On bridge: Lock X to bridge center to prevent walking on "river" edge
                            bridge_center_x = (
                                self.left_bridge_x
                                if abs(tx - self.left_bridge_x) < 0.8
                                else self.right_bridge_x
                            )
                            t.position[0] = bridge_center_x

        # 5. Cleanup dead troops
        for agent in self.possible_agents:
            self.troops[agent] = [t for t in self.troops[agent] if t.is_alive()]

        # 6. Check Terminations and Truncations
        win_reward = 10.0

        # Determine if game is over
        game_over = False
        if (
            not self.towers["player_0"][0].is_alive()
            or not self.towers["player_1"][0].is_alive()
        ):
            game_over = True
            for a in self.possible_agents:
                terminations[a] = True

        if self.current_step >= self.max_steps:
            game_over = True
            for a in self.possible_agents:
                truncations[a] = True

        if game_over:
            # Winner Logic: Crowns first, then Lowest HP Tower (Tie-breaker)
            def get_crowns(agent):
                enemy = "player_1" if agent == "player_0" else "player_0"
                if not self.towers[enemy][0].is_alive():
                    return 3
                return sum(1 for t in self.towers[enemy] if not t.is_alive())

            def get_min_hp(agent):
                # Returns min HP among all towers (even dead ones count as 0 if they exist)
                # But typically tie-breaker is among remaining towers.
                # If crowns are equal, it means both players have the same number of towers.
                # So we check min HP of remaining towers.
                alive_hps = [t.health for t in self.towers[agent] if t.is_alive()]
                return min(alive_hps) if alive_hps else 0

            c0 = get_crowns("player_0")
            c1 = get_crowns("player_1")

            if c0 > c1:
                rewards["player_0"] += win_reward
                rewards["player_1"] -= win_reward
            elif c1 > c0:
                rewards["player_0"] -= win_reward
                rewards["player_1"] += win_reward
            else:
                # Crown Tie-breaker: Lowest HP tower loses
                min0 = get_min_hp("player_0")
                min1 = get_min_hp("player_1")
                if min0 > min1:
                    rewards["player_0"] += win_reward
                    rewards["player_1"] -= win_reward
                elif min1 > min0:
                    rewards["player_0"] -= win_reward
                    rewards["player_1"] += win_reward
                # else: True Draw (0 reward change)

            self.agents = []

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
