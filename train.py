import os

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from opponent_env import OpponentWrapper
from pz_env import ClashRoyalePZ


def make_masked_env(pz_env, agent_id, opponent_model=None):
    env = OpponentWrapper(pz_env, agent_id, opponent_model=opponent_model)
    env = ActionMasker(env, lambda e: e.action_masks())
    return env


def train():
    # 1. Create Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Initialize or Load Models
    # We create two separate models for Player 0 and Player 1
    base_pz = ClashRoyalePZ()
    policy_kwargs = dict(net_arch=[512, 128])
    model_0_path = "models/ppo_p0.zip"
    model_1_path = "models/ppo_p1.zip"

    if os.path.exists(model_0_path) and os.path.exists(model_1_path):
        print("Found existing models. Resuming training with MaskablePPO...")
        # Note: Loading a standard PPO model as MaskablePPO might require caution.
        # If it fails, we might need to initialize fresh or use a conversion script.
        try:
            model_0 = MaskablePPO.load(
                model_0_path,
                env=make_masked_env(base_pz, "player_0"),
                device="cpu",
                custom_objects={"tensorboard_log": "./logs/player_0"},
                gamma=0.995,
            )
            model_1 = MaskablePPO.load(
                model_1_path,
                env=make_masked_env(base_pz, "player_1"),
                device="cpu",
                custom_objects={"tensorboard_log": "./logs/player_1"},
                gamma=0.995,
            )
        except Exception as e:
            print(f"Failed to load existing models as MaskablePPO: {e}")
            print("Initializing fresh MaskablePPO models instead.")
            model_0 = None
            model_1 = None

    if 'model_0' not in locals() or model_0 is None:
        model_0 = MaskablePPO(
            "MlpPolicy",
            make_masked_env(base_pz, "player_0"),
            verbose=0,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/player_0",
            device="cpu",
        )

        model_1 = MaskablePPO(
            "MlpPolicy",
            make_masked_env(base_pz, "player_1"),
            verbose=0,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/player_1",
            device="cpu",
        )

    # 3. Iterative Training Loop
    generations = 10  # Number of times we switch training
    steps_per_gen = 40000  # steps per agent per generation

    print(
        f"Starting Alternating Training with Action Masking: {generations} generations of {steps_per_gen} steps each."
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
        env_p0 = make_masked_env(ClashRoyalePZ(), "player_0", opponent_model=model_1)
        model_0.set_env(env_p0)
        model_0.learn(total_timesteps=num_steps_gen_0, reset_num_timesteps=False)
        model_0.save(f"models/ppo_p0_gen{gen + 1}")
        model_0.save("models/ppo_p0")  # Latest version for easy loading

        # --- Phase B: Train Player 1 against Player 0 ---
        print(f"Phase B: Training Player 1 (Opponent: Player 0 Version {gen + 1})")
        # Create a fresh wrapped env with the latest version of Player 0
        env_p1 = make_masked_env(ClashRoyalePZ(), "player_1", opponent_model=model_0)
        model_1.set_env(env_p1)
        model_1.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)
        model_1.save(f"models/ppo_p1_gen{gen + 1}")
        model_1.save("models/ppo_p1")  # Latest version for easy loading

    print("\nTraining Complete!")
    print("Final models saved as models/ppo_p0.zip and models/ppo_p1.zip")


if __name__ == "__main__":
    train()
