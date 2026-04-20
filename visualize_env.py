import os
import sys

import numpy as np
import pygame
import torch
from sb3_contrib import RecurrentPPO

from game_objects import Tower, Troop
from pz_env import ClashRoyalePZ

# Constants
ARENA_WIDTH = 18
ARENA_HEIGHT = 30
CELL_SIZE = 25
MARGIN = 60
WIDTH = ARENA_WIDTH * CELL_SIZE + 2 * MARGIN  # 570
HEIGHT = ARENA_HEIGHT * CELL_SIZE + 2 * MARGIN  # 870

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Visualizer using device: {self.device}")

        # Assets
        self.assets = {}
        self.load_assets()

        # Load Model for Player 0
        self.model_0 = None
        if model_p0_path and os.path.exists(model_p0_path):
            try:
                self.model_0 = RecurrentPPO.load(model_p0_path, device=self.device)
                print(f"Loaded Player 0 from {model_p0_path} (RecurrentPPO)")
            except Exception as e:
                print(f"Could not load Player 0 as RecurrentPPO: {e}")
                print("Checking if it is a standard PPO model...")
                try:
                    from stable_baselines3 import PPO

                    self.model_0 = PPO.load(model_p0_path, device=self.device)
                    print(f"Loaded Player 0 from {model_p0_path} (PPO)")
                except Exception as e2:
                    print(f"Could not load Player 0 as standard PPO either: {e2}")

        # Load Model for Player 1
        self.model_1 = None
        if model_p1_path and os.path.exists(model_p1_path):
            try:
                self.model_1 = RecurrentPPO.load(model_p1_path, device=self.device)
                print(f"Loaded Player 1 from {model_p1_path} (RecurrentPPO)")
            except Exception as e:
                print(f"Could not load Player 1 as RecurrentPPO: {e}")
                print("Checking if it is a standard PPO model...")
                try:
                    from stable_baselines3 import PPO

                    self.model_1 = PPO.load(model_p1_path, device=self.device)
                    print(f"Loaded Player 1 from {model_p1_path} (PPO)")
                except Exception as e2:
                    print(f"Could not load Player 1 as standard PPO either: {e2}")

    def load_assets(self):
        assets_dir = "assets"
        # Simple mapping of troop and building names to filenames
        image_assets = [
            "Knight",
            "Giant",
            "Archer",
            "Minion",
            "BabyDragon",
            "Cannon",
            "Musketeer",
        ]
        for name in image_assets:
            path = os.path.join(assets_dir, f"{name}.png")
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    # Scale to fit unit size (approx 24-30 pixels)
                    self.assets[name] = pygame.transform.scale(img, (30, 30))
                except Exception as e:
                    print(f"Error loading {name} image: {e}")

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

        # Recurrent state tracking
        lstm_states = {"player_0": None, "player_1": None}
        episode_starts = {
            "player_0": np.ones((1,), dtype=bool),
            "player_1": np.ones((1,), dtype=bool),
        }

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
                        lstm_states = {"player_0": None, "player_1": None}
                        episode_starts = {
                            "player_0": np.ones((1,), dtype=bool),
                            "player_1": np.ones((1,), dtype=bool),
                        }

            # 2. Step Environment (if not game over)
            if not game_over:
                # Agent 0 control
                if self.model_0:
                    try:
                        if isinstance(self.model_0, RecurrentPPO):
                            pred, lstm_states["player_0"] = self.model_0.predict(
                                obs_dict["player_0"],
                                state=lstm_states["player_0"],
                                episode_start=episode_starts["player_0"],
                                deterministic=True,
                            )
                            episode_starts["player_0"] = np.zeros((1,), dtype=bool)
                        else:
                            pred, _ = self.model_0.predict(
                                obs_dict["player_0"], deterministic=True
                            )
                        actions["player_0"] = pred
                    except Exception as e:
                        # print(f"P0 Prediction Error: {e}")
                        pass

                # Agent 1 control
                if self.model_1:
                    try:
                        if isinstance(self.model_1, RecurrentPPO):
                            pred, lstm_states["player_1"] = self.model_1.predict(
                                obs_dict["player_1"],
                                state=lstm_states["player_1"],
                                episode_start=episode_starts["player_1"],
                                deterministic=True,
                            )
                            episode_starts["player_1"] = np.zeros((1,), dtype=bool)
                        else:
                            pred, _ = self.model_1.predict(
                                obs_dict["player_1"], deterministic=True
                            )
                        actions["player_1"] = pred
                    except Exception as e:
                        # print(f"P1 Prediction Error: {e}")
                        pass

                obs_dict, rewards, terminations, truncations, infos = self.env.step(
                    actions
                )
                for agent in ["player_0", "player_1"]:
                    cum_rewards[agent] += rewards[agent]

                # Handle explosions from hits
                for hit_pos in self.env.hits:
                    ex, ey = self.world_to_screen(hit_pos)
                    self.explosions.append([ex, ey, 10])

                if any(terminations.values()) or any(truncations.values()):
                    game_over = True

            # 3. Draw
            self.screen.fill(GREEN)

            # Draw River and Bridges
            # River spans Y=14.0 to Y=16.0
            _, sy_16 = self.world_to_screen((0, 16.0))
            pygame.draw.rect(
                self.screen, RIVER, (MARGIN, sy_16, ARENA_WIDTH * CELL_SIZE, 50)
            )

            # Left Bridge (Center at X=3.0, Y=15.0)
            lsx, lsy = self.world_to_screen((3.0, 15.0))
            pygame.draw.rect(self.screen, ROAD, (lsx - 25, lsy - 25, 50, 50))

            # Right Bridge (Center at X=15.0, Y=15.0)
            rsx, rsy = self.world_to_screen((15.0, 15.0))
            pygame.draw.rect(self.screen, ROAD, (rsx - 25, rsy - 25, 50, 50))

            # Draw Towers
            for agent in ["player_0", "player_1"]:
                color = BLUE if agent == "player_0" else RED
                for tower in self.env.towers[agent]:
                    if not tower.is_alive():
                        continue
                    tx, ty = self.world_to_screen(tower.position)
                    size = 100 if tower.tower_type == "king" else 75
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

                    # Determine color based on troop name
                    if "Knight" in t.name:
                        color_body = GREY
                    elif "Archer" in t.name:
                        color_body = DARK_GREEN
                    elif "Minion" in t.name:
                        color_body = (100, 100, 200)  # Light Purple/Blue
                    elif "Giant" in t.name:
                        color_body = (160, 82, 45)  # Sienna
                    elif "PEKKA" in t.name:
                        color_body = (75, 0, 130)  # Indigo
                    elif "HogRider" in t.name:
                        color_body = (139, 69, 19)  # Brown
                    elif "BabyDragon" in t.name:
                        color_body = (124, 252, 0)  # Lawn Green
                    elif "Wizard" in t.name:
                        color_body = (65, 105, 225)  # Royal Blue
                    elif "Musketeer" in t.name:
                        color_body = (218, 112, 214)  # Orchid
                    elif "Skeletons" in t.name:
                        color_body = WHITE
                    else:
                        color_body = DARK_GREEN

                    # Visuals for flying units
                    if t.is_flying:
                        # Draw unit with an upward offset
                        v_tx, v_ty = tx, ty - 15
                        # Optional: small shadow at the actual position
                        pygame.draw.ellipse(
                            self.screen, (50, 50, 50), (tx - 8, ty - 4, 16, 8)
                        )
                    else:
                        v_tx, v_ty = tx, ty

                    # Draw Image if available, else Circle
                    if t.name in self.assets:
                        img = self.assets[t.name]
                        # Center the image
                        self.screen.blit(img, (v_tx - 15, v_ty - 15))
                        # Draw owner border
                        pygame.draw.circle(
                            self.screen, color_border, (v_tx, v_ty), 16, 2
                        )
                    else:
                        pygame.draw.circle(self.screen, color_body, (v_tx, v_ty), 12)
                        pygame.draw.circle(
                            self.screen, color_border, (v_tx, v_ty), 12, 2
                        )

                    self.draw_health_bar(v_tx, v_ty, t.health, t.max_health)

            # Draw Buildings
            for agent in ["player_0", "player_1"]:
                color_border = BLUE if agent == "player_0" else RED
                for b in self.env.buildings[agent]:
                    bx, by = self.world_to_screen(b.position)

                    # Draw Image if available, else Square
                    if b.name in self.assets:
                        img = self.assets[b.name]
                        # Center the image
                        self.screen.blit(img, (bx - 15, by - 15))
                        # Draw owner border
                        pygame.draw.rect(
                            self.screen,
                            color_border,
                            (bx - 18, by - 18, 36, 36),
                            2,
                        )
                    else:
                        if "Cannon" in b.name:
                            color_body = GREY
                        else:
                            color_body = BLACK

                        # Buildings are squares
                        size = 24
                        pygame.draw.rect(
                            self.screen,
                            color_body,
                            (bx - size // 2, by - size // 2, size, size),
                        )
                        pygame.draw.rect(
                            self.screen,
                            color_border,
                            (bx - size // 2, by - size // 2, size, size),
                            2,
                        )
                    self.draw_health_bar(bx, by, b.health, b.max_health)

            # Draw Spells
            for agent in ["player_0", "player_1"]:
                color = BLUE if agent == "player_0" else RED
                for s in self.env.spells[agent]:
                    fx, fy = self.world_to_screen(s.position)
                    # Use different colors/sizes based on spell name if needed
                    spell_color = ORANGE if s.name == "Fireball" else (255, 255, 0)
                    pygame.draw.circle(self.screen, spell_color, (fx, fy), 15)
                    pygame.draw.circle(self.screen, color, (fx, fy), 15, 2)

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
            self.clock.tick(20)

        pygame.quit()


if __name__ == "__main__":
    vis = Visualizer(
        model_p0_path="models/ppo_p0.zip", model_p1_path="models/ppo_p1.zip"
    )
    vis.run()
