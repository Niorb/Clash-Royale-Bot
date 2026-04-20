import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from card_stats import (
    BUILDING_STATS,
    BUILDING_TYPE_MAP,
    SPELL_STATS,
    SPELL_TYPE_MAP,
    TOWER_STATS,
    TROOP_STATS,
    TROOP_TYPE_MAP,
)
from game_objects import Building, ProjectileSpell, Tower, Troop


class ClashRoyalePZ(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "clash_royale_pz_v1"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        self.arena_width = 18.0
        self.arena_height = 30.0

        # Action: [type, x, y]
        # type: 0 - Nothing, 1 - Knight, 2 - Archer, 3 - Fireball, 4 - Minion, 5 - Giant, 6 - Cannon, 7 - Baby Dragon, 8 - Musketeer
        # x: 0 to 17
        # y: 0 to 29
        self.action_spaces = {
            agent: spaces.MultiDiscrete([9, 18, 30]) for agent in self.possible_agents
        }

        # Observation Space: Dict with Entities and Vector
        # entities: (40, 7) - [Active, Type, X, Y, HP, Enemy, Flying]
        # vector: [Elixir]
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "entities": spaces.Box(
                        low=0.0, high=1.0, shape=(40, 7), dtype=np.float32
                    ),
                    "vector": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.max_elixir = 10
        self.dt = 0.2
        self.max_steps = 1500

        self.bridge_y = 15.0
        self.left_bridge_x = 3.0
        self.right_bridge_x = 15.0

        self.troop_type_map = TROOP_TYPE_MAP

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.time = 0.0

        # Game State
        self.elixir = {"player_0": 5.0, "player_1": 5.0}

        # Towers: [King, Left Princess, Right Princess]
        # Coordinates: Player 0 is bottom (Y=0), Player 1 is top (Y=29)
        k_s = TOWER_STATS["king"]
        p_s = TOWER_STATS["princess"]
        self.towers = {
            "player_0": [
                Tower(
                    0,
                    (9.0, 2.0),
                    "king",
                    health=k_s["health"],
                    attack_damage=k_s["damage"],
                    attack_range=k_s["range"],
                    attack_speed=k_s["speed"],
                ),
                Tower(
                    0,
                    (3.5, 5.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),
                Tower(
                    0,
                    (14.5, 5.5),
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
                    (9.0, 28.0),
                    "king",
                    health=k_s["health"],
                    attack_damage=k_s["damage"],
                    attack_range=k_s["range"],
                    attack_speed=k_s["speed"],
                ),
                Tower(
                    1,
                    (14.5, 24.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),  # Symmetric
                Tower(
                    1,
                    (3.5, 24.5),
                    "princess",
                    health=p_s["health"],
                    attack_damage=p_s["damage"],
                    attack_range=p_s["range"],
                    attack_speed=p_s["speed"],
                ),
            ],
        }
        self.troops = {"player_0": [], "player_1": []}
        self.buildings = {"player_0": [], "player_1": []}
        self.spells = {"player_0": [], "player_1": []}
        self.hits = []

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        enemy = "player_1" if agent == "player_0" else "player_0"
        is_p0 = agent == "player_0"

        # Entities array (Max 40, Feature Size 7)
        # Features: [Active, Type, X, Y, HP, Enemy, Flying]
        entities = np.zeros((40, 7), dtype=np.float32)
        entity_count = 0

        def add_entity(obj, obj_type, is_enemy):
            nonlocal entity_count
            if entity_count >= 40:
                return
            
            x, y = obj.position
            rx = x if is_p0 else (self.arena_width - x)
            ry = y if is_p0 else (self.arena_height - y)
            
            entities[entity_count, 0] = 1.0  # Active
            entities[entity_count, 1] = obj_type / 10.0 # Type (Normalized)
            entities[entity_count, 2] = rx / self.arena_width
            entities[entity_count, 3] = ry / self.arena_height
            
            if hasattr(obj, "health") and hasattr(obj, "max_health"):
                entities[entity_count, 4] = obj.health / obj.max_health
            else:
                entities[entity_count, 4] = 1.0  # Spells or other non-destructible entities
                
            entities[entity_count, 5] = 1.0 if is_enemy else 0.0
            entities[entity_count, 6] = 1.0 if (hasattr(obj, "is_flying") and obj.is_flying) else 0.0
            
            entity_count += 1

        # 1. Towers
        for t in self.towers[agent]:
            if t.is_alive():
                add_entity(t, 9 if t.tower_type == "king" else 10, False)
        for t in self.towers[enemy]:
            if t.is_alive():
                add_entity(t, 9 if t.tower_type == "king" else 10, True)

        # 2. Troops
        for t in self.troops[agent]:
            # Action types: 1:Knight, 2:Archer, 4:Minion, 5:Giant, 7:BabyDragon, 8:Musketeer
            type_map = {"Knight": 1, "Archer": 2, "Minion": 4, "Giant": 5, "BabyDragon": 7, "Musketeer": 8}
            add_entity(t, type_map.get(t.name, 0), False)
        for t in self.troops[enemy]:
            type_map = {"Knight": 1, "Archer": 2, "Minion": 4, "Giant": 5, "BabyDragon": 7, "Musketeer": 8}
            add_entity(t, type_map.get(t.name, 0), True)

        # 3. Buildings
        for b in self.buildings[agent]:
            add_entity(b, 6, False) # Cannon
        for b in self.buildings[enemy]:
            add_entity(b, 6, True)

        # 4. Spells
        for s in self.spells[agent]:
            add_entity(s, 3, False) # Fireball
        for s in self.spells[enemy]:
            add_entity(s, 3, True)

        # Vector: Elixir (1)
        vector = np.array([self.elixir[agent] / self.max_elixir], dtype=np.float32)

        return {"entities": entities, "vector": vector}

    def step(self, actions):
        rewards = {"player_0": -0.001, "player_1": -0.001}
        terminations = {"player_0": False, "player_1": False}
        truncations = {"player_0": False, "player_1": False}
        infos = {"player_0": {}, "player_1": {}}

        self.current_step += 1
        self.time += self.dt
        self.hits = []

        # 1. Elixir Regen and Building Decay
        for agent in self.possible_agents:
            self.elixir[agent] = min(
                self.max_elixir, self.elixir[agent] + (self.dt / 2.8)
            )
            if self.elixir[agent] >= self.max_elixir:
                rewards[agent] -= 0.01

            for b in self.buildings[agent]:
                decay_damage = b.max_health * (self.dt / b.lifetime)
                b.take_damage(decay_damage)

        # 2. Process Actions
        for agent, action in actions.items():
            action_type, ax, ay = action
            enemy = "player_1" if agent == "player_0" else "player_0"

            # Clamp and convert coordinates
            if action_type in [1, 2, 4, 5, 6, 7, 8]:  # Troops and Buildings
                if agent == "player_0":
                    actual_y = min(float(ay), 13.5)
                else:
                    actual_y = max(float(ay), 16.5)
                actual_x = float(ax)
            else:  # Spells can be anywhere
                actual_x, actual_y = float(ax), float(ay)

            actual_pos = (actual_x, actual_y)
            spawn_success = False

            if action_type == 0:
                spawn_success = True  # Doing nothing is always successful
            elif (
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
                        is_flying=stats["is_flying"],
                        targets=stats["targets"],
                        splash_radius=stats.get("splash_radius", 0.0),
                    )
                )
                spawn_success = True
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
                            is_flying=stats["is_flying"],
                            targets=stats["targets"],
                            splash_radius=stats.get("splash_radius", 0.0),
                        )
                    )
                spawn_success = True
            elif (
                action_type == 4
                and self.elixir[agent] >= TROOP_STATS["Minion"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["Minion"]
                self.elixir[agent] -= stats["cost"]
                # Spawn a pack of 3 Minions in a triangle formation
                offsets = [(0.0, 0.0), (-1.0, -1.0), (1.0, -1.0)]
                for dx, dy in offsets:
                    spawn_x = max(0.0, min(self.arena_width, actual_x + dx))
                    spawn_y = actual_y + dy
                    self.troops[agent].append(
                        Troop(
                            agent,
                            "Minion",
                            stats["health"],
                            stats["damage"],
                            stats["speed"],
                            stats["attack_range"],
                            stats["attack_speed"],
                            (spawn_x, spawn_y),
                            stats["cost"],
                            is_flying=stats["is_flying"],
                            targets=stats["targets"],
                            splash_radius=stats.get("splash_radius", 0.0),
                        )
                    )
                spawn_success = True
            elif (
                action_type == 5
                and self.elixir[agent] >= TROOP_STATS["Giant"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["Giant"]
                self.elixir[agent] -= stats["cost"]
                self.troops[agent].append(
                    Troop(
                        agent,
                        "Giant",
                        stats["health"],
                        stats["damage"],
                        stats["speed"],
                        stats["attack_range"],
                        stats["attack_speed"],
                        actual_pos,
                        stats["cost"],
                        is_flying=stats["is_flying"],
                        targets=stats["targets"],
                        splash_radius=stats.get("splash_radius", 0.0),
                    )
                )
                spawn_success = True
            elif (
                action_type == 6
                and self.elixir[agent] >= BUILDING_STATS["Cannon"]["cost"]
                and len(self.buildings[agent]) < 2
            ):
                stats = BUILDING_STATS["Cannon"]
                self.elixir[agent] -= stats["cost"]
                self.buildings[agent].append(
                    Building(
                        agent,
                        "Cannon",
                        actual_pos,
                        stats["health"],
                        stats["damage"],
                        stats["attack_range"],
                        stats["attack_speed"],
                        stats["lifetime"],
                        stats["cost"],
                        targets=stats["targets"],
                    )
                )
                spawn_success = True
            elif (
                action_type == 7
                and self.elixir[agent] >= TROOP_STATS["BabyDragon"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["BabyDragon"]
                self.elixir[agent] -= stats["cost"]
                self.troops[agent].append(
                    Troop(
                        agent,
                        "BabyDragon",
                        stats["health"],
                        stats["damage"],
                        stats["speed"],
                        stats["attack_range"],
                        stats["attack_speed"],
                        actual_pos,
                        stats["cost"],
                        is_flying=stats["is_flying"],
                        targets=stats["targets"],
                        splash_radius=stats.get("splash_radius", 0.0),
                    )
                )
                spawn_success = True
            elif (
                action_type == 8
                and self.elixir[agent] >= TROOP_STATS["Musketeer"]["cost"]
                and len(self.troops[agent]) < 10
            ):
                stats = TROOP_STATS["Musketeer"]
                self.elixir[agent] -= stats["cost"]
                self.troops[agent].append(
                    Troop(
                        agent,
                        "Musketeer",
                        stats["health"],
                        stats["damage"],
                        stats["speed"],
                        stats["attack_range"],
                        stats["attack_speed"],
                        actual_pos,
                        stats["cost"],
                        is_flying=stats["is_flying"],
                        targets=stats["targets"],
                        splash_radius=stats.get("splash_radius", 0.0),
                    )
                )
                spawn_success = True
            elif (
                action_type == 3 and self.elixir[agent] >= SPELL_STATS["Fireball"]["cost"]
            ):
                stats = SPELL_STATS["Fireball"]
                self.elixir[agent] -= stats["cost"]
                king_tower = self.towers[agent][0]
                start_pos = king_tower.position
                self.spells[agent].append(
                    ProjectileSpell(
                        agent,
                        "Fireball",
                        start_pos,
                        actual_pos,
                        stats["damage"],
                        stats["tower_damage"],
                        stats["radius"],
                        stats["travel_speed"],
                    )
                )
                spawn_success = True

            # If action was requested but failed (e.g. no spawn happened), apply penalty
            if not spawn_success:
                rewards[agent] -= 0.05

        # 2.5 Update Spells
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            for s in self.spells[agent]:
                s.move(self.dt)
                if s.is_done:
                    # Apply damage
                    fx, fy = s.target_pos
                    self.hits.append(s.target_pos)
                    for et in self.towers[enemy]:
                        if math.hypot(et.position[0] - fx, et.position[1] - fy) <= s.radius:
                            et.take_damage(s.tower_damage)
                    for et in self.troops[enemy]:
                        if math.hypot(et.position[0] - fx, et.position[1] - fy) <= s.radius:
                            et.take_damage(s.damage)
                    for eb in self.buildings[enemy]:
                        if math.hypot(eb.position[0] - fx, eb.position[1] - fy) <= s.radius:
                            eb.take_damage(s.damage)
            self.spells[agent] = [s for s in self.spells[agent] if not s.is_done]

        # 3. Tower and Building Defense Logic (2D)
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            enemy_units = [t for t in self.troops[enemy] if t.is_alive()]

            # Defensive structures: Towers + Buildings
            defenders = self.towers[agent] + self.buildings[agent]

            for tower in defenders:
                if not tower.is_alive():
                    continue

                # Check if current target is still valid
                if (
                    hasattr(tower, "target")
                    and tower.target
                    and tower.target.is_alive()
                ):
                    dist = math.hypot(
                        tower.target.position[0] - tower.position[0],
                        tower.target.position[1] - tower.position[1],
                    )
                    if dist <= tower.attack_range:
                        # Keep current target if in range
                        if (self.time - tower.last_attack_time) >= tower.attack_speed:
                            dmg = (
                                tower.damage
                                if hasattr(tower, "damage")
                                else tower.attack_damage
                            )
                            tower.target.take_damage(dmg)
                            tower.last_attack_time = self.time
                        continue
                    else:
                        tower.target = None

                # Find closest enemy unit
                valid_targets = [
                    u
                    for u in enemy_units
                    if (tower.targets == "both")
                    or (not u.is_flying and tower.targets == "ground")
                ]

                if valid_targets:
                    target = min(
                        valid_targets,
                        key=lambda t: math.hypot(
                            t.position[0] - tower.position[0],
                            t.position[1] - tower.position[1],
                        ),
                    )
                    dist = math.hypot(
                        target.position[0] - tower.position[0],
                        target.position[1] - tower.position[1],
                    )

                    if dist <= tower.attack_range:
                        tower.target = target  # Lock in target
                        if (self.time - tower.last_attack_time) >= tower.attack_speed:
                            dmg = (
                                tower.damage
                                if hasattr(tower, "damage")
                                else tower.attack_damage
                            )
                            target.take_damage(dmg)
                            tower.last_attack_time = self.time

        # 4. Movement and Combat (Troops) with Bridge Pathfinding
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            is_p0 = agent == "player_0"

            for t in self.troops[agent]:
                # Targets: towers, buildings, and optionally troops
                if t.targets == "building":
                    potential_targets = [
                        tw for tw in self.towers[enemy] if tw.is_alive()
                    ] + [b for b in self.buildings[enemy] if b.is_alive()]
                else:
                    potential_targets = (
                        [
                            et
                            for et in self.troops[enemy]
                            if et.is_alive()
                            and (t.targets == "both" or not et.is_flying)
                        ]
                        + [tw for tw in self.towers[enemy] if tw.is_alive()]
                        + [b for b in self.buildings[enemy] if b.is_alive()]
                    )

                if not potential_targets:
                    t.target = None
                    continue

                # Lock-in logic: only keep target if alive AND in range
                if t.target and t.target.is_alive():
                    dist_to_current = math.hypot(
                        t.target.position[0] - t.position[0],
                        t.target.position[1] - t.position[1],
                    )
                    if dist_to_current > t.attack_range:
                        # Not in range, retarget to closest
                        t.target = min(
                            potential_targets,
                            key=lambda pt: math.hypot(
                                pt.position[0] - t.position[0],
                                pt.position[1] - t.position[1],
                            ),
                        )
                    # Else: In range, keep target (lock)
                else:
                    # No target or dead, pick closest
                    t.target = min(
                        potential_targets,
                        key=lambda pt: math.hypot(
                            pt.position[0] - t.position[0],
                            pt.position[1] - t.position[1],
                        ),
                    )

                target = t.target
                target_pos = target.position

                # Simple Pathfinding: if target is across the river, go to bridge first
                # Flying units ignore the river
                is_across = not t.is_flying and (
                    (is_p0 and target_pos[1] > 16.0 and t.position[1] < 16.0)
                    or (not is_p0 and target_pos[1] < 14.0 and t.position[1] > 14.0)
                )

                # Check if currently in the river corridor (Ground units only)
                in_river_zone = not t.is_flying and (14.0 < t.position[1] < 16.0)

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
                        entrance_y = 14.0 if is_p0 else 16.0
                        # If already at bridge X, aim for river center to keep moving
                        if abs(t.position[0] - bridge_x) < 0.2:
                            move_target = (bridge_x, 15.0)
                        else:
                            move_target = (bridge_x, entrance_y)

                dist = math.hypot(
                    target_pos[0] - t.position[0], target_pos[1] - t.position[1]
                )
                if dist <= t.attack_range:
                    if (self.time - t.last_attack_time) >= t.attack_speed:
                        if t.splash_radius > 0.0:
                            for pt in potential_targets:
                                if (
                                    math.hypot(
                                        pt.position[0] - target.position[0],
                                        pt.position[1] - target.position[1],
                                    )
                                    <= t.splash_radius
                                ):
                                    pt.take_damage(t.damage)
                        else:
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

                    # River Collision Enforcement (Hard Block) - Only for ground units
                    if not t.is_flying:
                        tx, ty = t.position
                        on_bridge = (
                            abs(tx - self.left_bridge_x) < 1.0
                            or abs(tx - self.right_bridge_x) < 1.0
                        )

                        if 14.0 < ty < 16.0:
                            if not on_bridge:
                                # Cannot enter river from the side
                                t.position = old_pos
                            else:
                                # On bridge: Lock X to bridge center to prevent walking on "river" edge
                                bridge_center_x = (
                                    self.left_bridge_x
                                    if abs(tx - self.left_bridge_x) < 1.0
                                    else self.right_bridge_x
                                )
                                t.position[0] = bridge_center_x

        # 5. Cleanup dead troops and buildings
        for agent in self.possible_agents:
            self.troops[agent] = [t for t in self.troops[agent] if t.is_alive()]
            self.buildings[agent] = [b for b in self.buildings[agent] if b.is_alive()]

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
