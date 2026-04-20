import os
import sys

import numpy as np
import pygame
from stable_baselines3 import PPO

from game_objects import Troop
from pz_env import ClashRoyalePZ

# Constants
WIDTH, HEIGHT = 1000, 400
LANE_WIDTH = 800
LANE_X_OFFSET = 100
LANE_Y = HEIGHT // 2
SCALE = LANE_WIDTH / 32.0

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


class Visualizer:
    def __init__(self, model_p0_path=None, model_p1_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Clash Royale Two-Agent Battle Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.env = ClashRoyalePZ()
        self.explosions = []  # (x, time_remaining)

        # Load Model for Player 0
        self.model_0 = None
        if model_p0_path and os.path.exists(model_p0_path):
            self.model_0 = PPO.load(model_p0_path, device="cpu")
            print(f"Loaded Player 0 from {model_p0_path}")

        # Load Model for Player 1
        self.model_1 = None
        if model_p1_path and os.path.exists(model_p1_path):
            self.model_1 = PPO.load(model_p1_path, device="cpu")
            print(f"Loaded Player 1 from {model_p1_path}")

    def world_to_screen(self, pos):
        return int(LANE_X_OFFSET + pos * SCALE)

    def draw_health_bar(self, x, y, current, max_hp, width=40, height=6):
        if max_hp == 0:
            return
        pct = max(0, current / max_hp)
        pygame.draw.rect(self.screen, BLACK, (x - width // 2, y, width, height))
        color = (
            (0, 255, 0) if pct > 0.5 else (255, 165, 0) if pct > 0.2 else (255, 0, 0)
        )
        pygame.draw.rect(
            self.screen, color, (x - width // 2, y, int(width * pct), height)
        )

    def run(self):
        obs_dict, _ = self.env.reset()
        running = True
        game_over = False

        while running:
            actions = {"player_0": 0, "player_1": 0}

            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if not game_over and event.type == pygame.KEYDOWN:
                    # Manual control for player_0 (Keys 1, 2, 3)
                    if event.key == pygame.K_1:
                        actions["player_0"] = 1  # Knight
                    if event.key == pygame.K_2:
                        actions["player_0"] = 2  # Archer
                    if event.key == pygame.K_3:
                        actions["player_0"] = 3  # Fireball

                    # Manual control for player_1 (Keys 7, 8, 9)
                    if event.key == pygame.K_7:
                        actions["player_1"] = 1  # Knight
                    if event.key == pygame.K_8:
                        actions["player_1"] = 2  # Archer
                    if event.key == pygame.K_9:
                        actions["player_1"] = 3  # Fireball

                    if event.key == pygame.K_r:
                        obs_dict, _ = self.env.reset()
                        game_over = False
                        self.explosions = []

            # 2. Step Environment (if not game over)
            if not game_over:
                # Agent 0 control
                if self.model_0 and actions["player_0"] == 0:
                    actions["player_0"], _ = self.model_0.predict(
                        obs_dict["player_0"], deterministic=True
                    )

                # Agent 1 control
                if self.model_1 and actions["player_1"] == 0:
                    actions["player_1"], _ = self.model_1.predict(
                        obs_dict["player_1"], deterministic=True
                    )

                # Render explosion indicators
                for agent, action in actions.items():
                    if action == 3 and self.env.elixir[agent] >= 4.0:
                        target_pos = 32.0 if agent == "player_0" else 0.0
                        self.explosions.append([self.world_to_screen(target_pos), 10])

                obs_dict, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )
                if any(terminations.values()) or any(truncations.values()):
                    game_over = True

            # 3. Draw
            self.screen.fill(GREEN)
            pygame.draw.rect(
                self.screen, ROAD, (LANE_X_OFFSET, LANE_Y - 40, LANE_WIDTH, 80)
            )

            # Draw Towers
            for agent in ["player_0", "player_1"]:
                tower = self.env.towers[agent]
                tx = self.world_to_screen(tower.position)
                color = BLUE if agent == "player_0" else RED
                pygame.draw.rect(self.screen, color, (tx - 30, LANE_Y - 50, 60, 100))
                self.draw_health_bar(
                    tx, LANE_Y - 65, tower.health, tower.max_health, 80
                )

            # Draw Troops
            for agent in ["player_0", "player_1"]:
                color_border = BLUE if agent == "player_0" else RED
                for t in self.env.troops[agent]:
                    tx = self.world_to_screen(t.position)
                    color_body = GREY if "Knight" in t.name else DARK_GREEN
                    pygame.draw.circle(self.screen, color_body, (tx, LANE_Y), 15)
                    pygame.draw.circle(self.screen, color_border, (tx, LANE_Y), 15, 2)
                    self.draw_health_bar(tx, LANE_Y - 30, t.health, t.max_health)

            # Draw Explosions
            for exp in self.explosions[:]:
                pygame.draw.circle(self.screen, ORANGE, (exp[0], LANE_Y), 50, 5)
                exp[1] -= 1
                if exp[1] <= 0:
                    self.explosions.remove(exp)

            # Draw UI
            p0_elixir = self.font.render(
                f"P0 Elixir: {self.env.elixir['player_0']:.1f}", True, BLUE
            )
            self.screen.blit(p0_elixir, (20, 20))
            p1_elixir = self.font.render(
                f"P1 Elixir: {self.env.elixir['player_1']:.1f}", True, RED
            )
            self.screen.blit(p1_elixir, (WIDTH - 180, 20))

            time_text = self.font.render(f"Time: {self.env.time:.1f}s", True, WHITE)
            self.screen.blit(time_text, (20, 50))

            p0_mode = "AI_P0" if self.model_0 else "MAN_P0"
            p1_mode = "AI_P1" if self.model_1 else "MAN_P1"
            mode_text = self.font.render(f"{p0_mode} vs {p1_mode}", True, BLACK)
            self.screen.blit(mode_text, (WIDTH // 2 - 80, 50))

            controls_text = self.font.render(
                "P0: 1,2,3 | P1: 7,8,9 | R: Reset", True, BLACK
            )
            self.screen.blit(controls_text, (WIDTH // 2 - 150, 20))

            if game_over:
                msg = (
                    "P0 WINS!"
                    if self.env.towers["player_1"].health <= 0
                    else "P1 WINS!"
                    if self.env.towers["player_0"].health <= 0
                    else "DRAW"
                )
                color = BLUE if "P0" in msg else RED if "P1" in msg else BLACK
                over_text = self.font.render(msg, True, color)
                self.screen.blit(over_text, (WIDTH // 2 - 50, HEIGHT // 2 - 20))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    # Check for latest versions
    vis = Visualizer(
        model_p0_path="models/ppo_p0.zip", model_p1_path="models/ppo_p1.zip"
    )
    vis.run()
