import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_objects import Tower, Troop

class ClashRoyaleEnv(gym.Env):
    def __init__(self):
        super(ClashRoyaleEnv, self).__init__()

        # Action: 0 - Nothing, 1 - Knight (cost 3) at position 0
        self.action_space = spaces.Discrete(2)

        # Observation Space:
        # [elixir, p_tower_hp, e_tower_hp, p_t1_pos, p_t1_hp, p_t2_pos, p_t2_hp, p_t3_pos, p_t3_hp, e_t1_pos, e_t1_hp, e_t2_pos, e_t2_hp, e_t3_pos, e_t3_hp]
        # Normalized between 0 and 1 roughly.
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        self.max_elixir = 10
        self.dt = 0.5  # Time step in seconds
        self.max_steps = 600  # 300 seconds (5 mins) at 0.5s per step
        self.arena_length = 32.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_elixir = 5.0
        self.p_tower = Tower(owner_id=0, position=0)
        self.e_tower = Tower(owner_id=1, position=self.arena_length)
        self.p_troops = []
        self.e_troops = []
        self.current_step = 0
        self.time = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = self.player_elixir / self.max_elixir
        obs[1] = self.p_tower.health / self.p_tower.max_health
        obs[2] = self.e_tower.health / self.e_tower.max_health

        # Player troops
        for i in range(min(len(self.p_troops), 3)):
            obs[3 + i*2] = self.p_troops[i].position / self.arena_length
            obs[4 + i*2] = self.p_troops[i].health / self.p_troops[i].max_health

        # Enemy troops
        for i in range(min(len(self.e_troops), 3)):
            obs[9 + i*2] = self.e_troops[i].position / self.arena_length
            obs[10 + i*2] = self.e_troops[i].health / self.e_troops[i].max_health

        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        self.current_step += 1
        self.time += self.dt

        # 1. Handle Elixir (roughly 1 elixir every 2.8 seconds)
        self.player_elixir = min(self.max_elixir, self.player_elixir + (self.dt / 2.8))

        # 2. Handle Action (Player only for now)
        if action == 1 and self.player_elixir >= 3.0 and len(self.p_troops) < 3:
            self.player_elixir -= 3.0
            # Knight: health 1350, damage 160, speed 1.2, range 1.2, attack_speed 1.2, cost 3
            knight = Troop(0, "Knight", 1350, 160, 1.2, 1.2, 1.2, 0.0, 3)
            self.p_troops.append(knight)

        # 3. Simple Enemy Logic (Spawn Knight at position 32 if elixir permits)
        # For simplicity, we just randomly spawn an enemy knight
        if np.random.rand() < 0.05 and len(self.e_troops) < 3:
             enemy_knight = Troop(1, "E_Knight", 1350, 160, 1.2, 1.2, 1.2, self.arena_length, 3)
             self.e_troops.append(enemy_knight)

        # 4. Movement and Combat
        # Player troops move towards enemy tower (32)
        for t in self.p_troops:
            targets = [et for et in self.e_troops if et.is_alive()]
            if targets:
                # Find closest enemy troop
                target = min(targets, key=lambda et: abs(et.position - t.position))
                if t.can_attack(target.position, self.time):
                    target.take_damage(t.damage)
                    t.last_attack_time = self.time
                    reward += 0.01 # Small reward for hitting
                elif abs(t.position - target.position) > t.attack_range:
                    t.move(self.dt, target.position)
            else:
                # Move towards tower
                if t.can_attack(self.e_tower.position, self.time):
                    self.e_tower.take_damage(t.damage)
                    t.last_attack_time = self.time
                    reward += 0.05 # Reward for hitting tower
                elif abs(t.position - self.e_tower.position) > t.attack_range:
                    t.move(self.dt, self.e_tower.position)

        # Enemy troops move towards player tower (0)
        for t in self.e_troops:
            targets = [pt for pt in self.p_troops if pt.is_alive()]
            if targets:
                target = min(targets, key=lambda pt: abs(pt.position - t.position))
                if t.can_attack(target.position, self.time):
                    target.take_damage(t.damage)
                    t.last_attack_time = self.time
                elif abs(t.position - target.position) > t.attack_range:
                    t.move(self.dt, target.position)
            else:
                if t.can_attack(self.p_tower.position, self.time):
                    self.p_tower.take_damage(t.damage)
                    t.last_attack_time = self.time
                    reward -= 0.05 # Punishment for taking tower damage
                elif abs(t.position - self.p_tower.position) > t.attack_range:
                    t.move(self.dt, self.p_tower.position)

        # Clean up dead troops
        self.p_troops = [t for t in self.p_troops if t.is_alive()]
        self.e_troops = [t for t in self.e_troops if t.is_alive()]

        # 5. Check Termination
        if not self.e_tower.is_alive():
            reward += 10.0
            terminated = True
        elif not self.p_tower.is_alive():
            reward -= 10.0
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
