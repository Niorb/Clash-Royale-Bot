import os
import sys

import numpy as np
import pygame
from stable_baselines3 import PPO

from game_objects import Tower, Troop
from pz_env import ClashRoyalePZ

# Constants
ARENA_WIDTH = 18
ARENA_HEIGHT = 32
CELL_SIZE = 25
MARGIN = 60
WIDTH = ARENA_WIDTH * CELL_SIZE + 2 * MARGIN  # 570
HEIGHT = ARENA_HEIGHT * CELL_SIZE + 2 * MARGIN  # 920

# Colors
GREEN = (34, 139, 34)
ROAD = (210, 180, 140)
BLUE = (50, 50, 255)
RED = (255, 50, 50)
GREY = (169, 169, 169)
DARK_GREEN = (0, 100, 0)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RIVER = (100, 149, 237)


class Visualizer:
    def __init__(self, model_p0_path=None, model_p1_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Clash Royale 2-Lane 2D Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.env = ClashRoyalePZ()
        self.explosions = []  # (x, y, time_remaining)

        # Load Model for Player 0
        self.model_0 = None
        if model_p0_path and os.path.exists(model_p0_path):
            try:
                self.model_0 = PPO.load(model_p0_path, device="cpu")
                print(f"Loaded Player 0 from {model_p0_path}")
            except Exception as e:
                print(f"Could not load Player 0: {e}")

        # Load Model for Player 1
        self.model_1 = None
        if model_p1_path and os.path.exists(model_p1_path):
            try:
                self.model_1 = PPO.load(model_p1_path, device="cpu")
                print(f"Loaded Player 1 from {model_p1_path}")
            except Exception as e:
                print(f"Could not load Player 1: {e}")

    def world_to_screen(self, pos):
        x, y = pos
        sx = MARGIN + x * CELL_SIZE
        sy = MARGIN + (ARENA_HEIGHT - y) * CELL_SIZE
        return int(sx), int(sy)

    def draw_health_bar(self, sx, sy, current, max_hp, width=40, height=6):
        if max_hp == 0:
            return
        pct = max(0, current / max_hp)
        pygame.draw.rect(self.screen, BLACK, (sx - width // 2, sy - 25, width, height))
        color = (
            (0, 255, 0) if pct > 0.5 else (255, 165, 0) if pct > 0.2 else (255, 0, 0)
        )
        pygame.draw.rect(
            self.screen, color, (sx - width // 2, sy - 25, int(width * pct), height)
        )

    def run(self):
        obs_dict, _ = self.env.reset()
        running = True
        game_over = False
        cum_rewards = {"player_0": 0.0, "player_1": 0.0}

        while running:
            # Default actions: [Nothing, 0, 0]
            actions = {"player_0": [0, 0, 0], "player_1": [0, 0, 0]}

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs_dict, _ = self.env.reset()
                        game_over = False
                        self.explosions = []
                        cum_rewards = {"player_0": 0.0, "player_1": 0.0}

            # 2. Step Environment (if not game over)
            if not game_over:
                # Agent 0 control
                if self.model_0:
                    try:
                        pred, _ = self.model_0.predict(
                            obs_dict["player_0"], deterministic=True
                        )
                        actions["player_0"] = pred
                    except:
                        pass  # Space mismatch probably

                # Agent 1 control
                if self.model_1:
                    try:
                        pred, _ = self.model_1.predict(
                            obs_dict["player_1"], deterministic=True
                        )
                        actions["player_1"] = pred
                    except:
                        pass

                # Render explosion indicators
                for agent, action in actions.items():
                    act_type = action[0]
                    if act_type == 3 and self.env.elixir[agent] >= 4.0:
                        ex, ey = self.world_to_screen((action[1], action[2]))
                        self.explosions.append([ex, ey, 10])

                obs_dict, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )
                for agent in ["player_0", "player_1"]:
                    cum_rewards[agent] += rewards[agent]

                if any(terminations.values()) or any(truncations.values()):
                    game_over = True

            # 3. Draw
            self.screen.fill(GREEN)

            # Draw River and Bridges
            river_y_world = 16.0
            _, river_sy = self.world_to_screen((0, river_y_world))
            pygame.draw.rect(
                self.screen, RIVER, (MARGIN, river_sy - 10, ARENA_WIDTH * CELL_SIZE, 20)
            )

            # Left Bridge
            lbx_world, lby_world = 3.5, 16.0
            lsx, lsy = self.world_to_screen((lbx_world, lby_world))
            pygame.draw.rect(self.screen, ROAD, (lsx - 20, lsy - 15, 40, 30))

            # Right Bridge
            rbx_world, rby_world = 14.5, 16.0
            rsx, rsy = self.world_to_screen((rbx_world, rby_world))
            pygame.draw.rect(self.screen, ROAD, (rsx - 20, rsy - 15, 40, 30))

            # Draw Towers
            for agent in ["player_0", "player_1"]:
                color = BLUE if agent == "player_0" else RED
                for tower in self.env.towers[agent]:
                    if not tower.is_alive():
                        continue
                    tx, ty = self.world_to_screen(tower.position)
                    size = 40 if tower.tower_type == "king" else 30
                    pygame.draw.rect(
                        self.screen, color, (tx - size // 2, ty - size // 2, size, size)
                    )
                    self.draw_health_bar(tx, ty, tower.health, tower.max_health, 50)
                    # if tower.is_invulnerable:
                    #     pygame.draw.rect(self.screen, WHITE, (tx - size // 2, ty - size // 2, size, size), 2)

            # Draw Troops
            for agent in ["player_0", "player_1"]:
                color_border = BLUE if agent == "player_0" else RED
                for t in self.env.troops[agent]:
                    tx, ty = self.world_to_screen(t.position)
                    color_body = GREY if "Knight" in t.name else DARK_GREEN
                    pygame.draw.circle(self.screen, color_body, (tx, ty), 12)
                    pygame.draw.circle(self.screen, color_border, (tx, ty), 12, 2)
                    self.draw_health_bar(tx, ty, t.health, t.max_health)

            # Draw Explosions
            for exp in self.explosions[:]:
                pygame.draw.circle(self.screen, ORANGE, (exp[0], exp[1]), 50, 5)
                exp[2] -= 1
                if exp[2] <= 0:
                    self.explosions.remove(exp)

            # Draw UI
            p0_elixir = self.font.render(
                f"P0 Elixir: {self.env.elixir['player_0']:.1f}", True, BLUE
            )
            self.screen.blit(p0_elixir, (20, 10))
            p1_elixir = self.font.render(
                f"P1 Elixir: {self.env.elixir['player_1']:.1f}", True, RED
            )
            self.screen.blit(p1_elixir, (WIDTH - 150, 10))

            p0_reward_text = self.font.render(
                f"P0 Reward: {cum_rewards['player_0']:.2f}", True, BLUE
            )
            self.screen.blit(p0_reward_text, (20, 35))
            p1_reward_text = self.font.render(
                f"P1 Reward: {cum_rewards['player_1']:.2f}", True, RED
            )
            self.screen.blit(p1_reward_text, (WIDTH - 150, 35))

            time_text = self.font.render(f"Time: {self.env.time:.1f}s", True, WHITE)
            self.screen.blit(time_text, (WIDTH // 2 - 40, 10))

            if game_over:

                def get_crowns(agent):
                    enemy = "player_1" if agent == "player_0" else "player_0"
                    if not self.env.towers[enemy][0].is_alive():
                        return 3
                    return sum(1 for t in self.env.towers[enemy] if not t.is_alive())

                def get_min_hp(agent):
                    hps = [t.health for t in self.env.towers[agent] if t.is_alive()]
                    return min(hps) if hps else 0

                c0, c1 = get_crowns("player_0"), get_crowns("player_1")
                if c0 > c1:
                    msg = "P0 WINS!"
                elif c1 > c0:
                    msg = "P1 WINS!"
                else:
                    m0, m1 = get_min_hp("player_0"), get_min_hp("player_1")
                    if m0 > m1:
                        msg = "P0 WINS! (HP)"
                    elif m1 > m0:
                        msg = "P1 WINS! (HP)"
                    else:
                        msg = "DRAW"

                color = BLUE if "P0" in msg else RED if "P1" in msg else BLACK
                over_text = self.font.render(msg, True, color)
                self.screen.blit(over_text, (WIDTH // 2 - 60, HEIGHT // 2))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    vis = Visualizer(
        model_p0_path="models/ppo_p0.zip", model_p1_path="models/ppo_p1.zip"
    )
    vis.run()
