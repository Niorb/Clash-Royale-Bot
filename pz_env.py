import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv
from game_objects import Tower, Troop

class ClashRoyalePZ(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "clash_royale_pz_v1"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        # Action: 0 - Nothing, 1 - Knight, 2 - Archer, 3 - Fireball
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

        # Observation Space:
        # [elixir, self_tower_hp, enemy_tower_hp, self_t1_pos, self_t1_hp, ... enemy_t1_pos, enemy_t1_hp]
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.max_elixir = 10
        self.dt = 0.2
        self.max_steps = 1500
        self.arena_length = 32.0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.time = 0.0

        # Game State
        self.elixir = {"player_0": 5.0, "player_1": 5.0}
        self.towers = {
            "player_0": Tower(owner_id=0, position=0.0),
            "player_1": Tower(owner_id=1, position=self.arena_length)
        }
        self.troops = {"player_0": [], "player_1": []}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        enemy = "player_1" if agent == "player_0" else "player_0"
        obs = np.zeros(15, dtype=np.float32)
        
        # Elixir and Towers
        obs[0] = self.elixir[agent] / self.max_elixir
        obs[1] = self.towers[agent].health / self.towers[agent].max_health
        obs[2] = self.towers[enemy].health / self.towers[enemy].max_health

        # Troop observations are symmetric
        self_troops = self.troops[agent]
        enemy_troops = self.troops[enemy]

        def get_rel_pos(pos):
            return pos / self.arena_length if agent == "player_0" else (self.arena_length - pos) / self.arena_length

        for i in range(min(len(self_troops), 3)):
            obs[3 + i*2] = get_rel_pos(self_troops[i].position)
            obs[4 + i*2] = self_troops[i].health / self_troops[i].max_health

        for i in range(min(len(enemy_troops), 3)):
            obs[9 + i*2] = get_rel_pos(enemy_troops[i].position)
            obs[10 + i*2] = enemy_troops[i].health / enemy_troops[i].max_health

        return obs

    def step(self, actions):
        rewards = {"player_0": -0.001, "player_1": -0.001} # Step penalty to encourage speed
        terminations = {"player_0": False, "player_1": False}
        truncations = {"player_0": False, "player_1": False}
        infos = {"player_0": {}, "player_1": {}}

        self.current_step += 1
        self.time += self.dt

        # 1. Elixir Regen and Leak Penalty
        for agent in self.possible_agents:
            self.elixir[agent] = min(self.max_elixir, self.elixir[agent] + (self.dt / 2.8))
            # Penalty for leaking elixir (staying at max)
            if self.elixir[agent] >= self.max_elixir:
                rewards[agent] -= 0.01  # Small penalty per step at max elixir

        # 2. Process Actions
        for agent, action in actions.items():
            enemy = "player_1" if agent == "player_0" else "player_0"
            if action == 1 and self.elixir[agent] >= 3.0 and len(self.troops[agent]) < 3:
                self.elixir[agent] -= 3.0
                pos = 0.0 if agent == "player_0" else self.arena_length
                self.troops[agent].append(Troop(agent, "Knight", 1350, 160, 1.2, 1.2, 1.2, pos, 3))
            elif action == 2 and self.elixir[agent] >= 3.0 and len(self.troops[agent]) < 3:
                self.elixir[agent] -= 3.0
                pos = 0.0 if agent == "player_0" else self.arena_length
                self.troops[agent].append(Troop(agent, "Archer", 250, 80, 1.2, 5.0, 1.0, pos, 3))
            elif action == 3 and self.elixir[agent] >= 4.0:
                self.elixir[agent] -= 4.0
                # Fireball at enemy tower
                self.towers[enemy].take_damage(200)
                for et in self.troops[enemy]:
                    dist = abs(et.position - self.towers[enemy].position)
                    if dist <= 2.5:
                        et.take_damage(200)

        # 3. Tower Defense Logic
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            enemy_troops = [t for t in self.troops[enemy] if t.is_alive()]
            if enemy_troops:
                target = min(enemy_troops, key=lambda t: abs(t.position - self.towers[agent].position))
                dist = abs(target.position - self.towers[agent].position)
                if dist <= self.towers[agent].attack_range and \
                   (self.time - self.towers[agent].last_attack_time) >= self.towers[agent].attack_speed:
                    target.take_damage(self.towers[agent].attack_damage)
                    self.towers[agent].last_attack_time = self.time

        # 4. Movement and Combat (Troops)
        for agent in self.possible_agents:
            enemy = "player_1" if agent == "player_0" else "player_0"
            for t in self.troops[agent]:
                enemy_troops = [et for et in self.troops[enemy] if et.is_alive()]
                if enemy_troops:
                    target = min(enemy_troops, key=lambda et: abs(et.position - t.position))
                    if t.can_attack(target.position, self.time):
                        target.take_damage(t.damage)
                        t.last_attack_time = self.time
                        rewards[agent] += 0.01
                    elif abs(t.position - target.position) > t.attack_range:
                        t.move(self.dt, target.position)
                else:
                    if t.can_attack(self.towers[enemy].position, self.time):
                        self.towers[enemy].take_damage(t.damage)
                        t.last_attack_time = self.time
                        rewards[agent] += 0.05
                    elif abs(t.position - self.towers[enemy].position) > t.attack_range:
                        t.move(self.dt, self.towers[enemy].position)

        # 5. Penalties for damage taken (Implicitly handled by opponent reward)

        # 6. Cleanup dead troops
        for agent in self.possible_agents:
            self.troops[agent] = [t for t in self.troops[agent] if t.is_alive()]

        # 7. Check Terminations
        win_reward = 10.0 + (1.0 - self.current_step / self.max_steps) * 10.0
        if not self.towers["player_1"].is_alive():
            rewards["player_0"] += win_reward
            rewards["player_1"] -= win_reward
            for a in self.possible_agents: terminations[a] = True
        elif not self.towers["player_0"].is_alive():
            rewards["player_0"] -= win_reward
            rewards["player_1"] += win_reward
            for a in self.possible_agents: terminations[a] = True
        
        if self.current_step >= self.max_steps:
            for a in self.possible_agents: truncations[a] = True

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
