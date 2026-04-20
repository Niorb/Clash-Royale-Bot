import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from opponent_env import OpponentWrapper
from pz_env import ClashRoyalePZ


def train():
    # 1. Create Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Initialize Models
    # We create two separate models for Player 0 and Player 1
    # Initially they have no opponent model (opponent_model=None)

    # Simple base environment for initializing models
    base_pz = ClashRoyalePZ()

    # Policy kwargs to make them robust
    policy_kwargs = dict(net_arch=[128, 128])

    print("Initializing models...")
    model_0 = PPO(
        "MlpPolicy",
        OpponentWrapper(base_pz, "player_0"),
        verbose=0,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/player_0",
        device="cpu",
    )

    model_1 = PPO(
        "MlpPolicy",
        OpponentWrapper(base_pz, "player_1"),
        verbose=0,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/player_1",
        device="cpu",
    )

    # 3. Iterative Training Loop
    generations = 5  # Number of times we switch training
    steps_per_gen = 40000  # steps per agent per generation

    print(
        f"Starting Alternating Training: {generations} generations of {steps_per_gen} steps each."
    )

    for gen in range(generations):
        print(f"\n--- GENERATION {gen + 1} ---")

        # If first generation, create a shift in training so the newly trained always in theory becomes better than the opponent
        if gen == 0:
            num_steps_gen_0 = steps_per_gen / 2
        else:
            num_steps_gen_0 = steps_per_gen

        # --- Phase A: Train Player 0 against Player 1 ---
        print(f"Phase A: Training Player 0 (Opponent: Player 1 Version {gen})")
        # Create a fresh wrapped env with the latest version of the opponent
        env_p0 = OpponentWrapper(ClashRoyalePZ(), "player_0", opponent_model=model_1)
        model_0.set_env(env_p0)
        model_0.learn(total_timesteps=num_steps_gen_0, reset_num_timesteps=False)
        model_0.save(f"models/ppo_p0_gen{gen + 1}")
        model_0.save("models/ppo_p0")  # Latest version for easy loading

        # --- Phase B: Train Player 1 against Player 0 ---
        print(f"Phase B: Training Player 1 (Opponent: Player 0 Version {gen + 1})")
        # Create a fresh wrapped env with the latest version of Player 0
        env_p1 = OpponentWrapper(ClashRoyalePZ(), "player_1", opponent_model=model_0)
        model_1.set_env(env_p1)
        model_1.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)
        model_1.save(f"models/ppo_p1_gen{gen + 1}")
        model_1.save("models/ppo_p1")  # Latest version for easy loading

    print("\nTraining Complete!")
    print("Final models saved as models/ppo_p0.zip and models/ppo_p1.zip")


if __name__ == "__main__":
    train()
