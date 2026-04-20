import os

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from custom_policy import TransformerExtractor
from opponent_env import OpponentWrapper
from pz_env import ClashRoyalePZ


def make_opponent_env(pz_env, agent_id, opponent_model=None):
    env = OpponentWrapper(pz_env, agent_id, opponent_model=opponent_model)
    return env


def train():
    # 1. Create Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Initialize or Load Models
    # We create two separate models for Player 0 and Player 1
    base_pz = ClashRoyalePZ()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # net_arch for the MLP part after feature extraction
    # features_dim is 65 (64 from Transformer + 1 from Elixir)
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(features_dim=65),
        net_arch=[256, 128]
    ) 
    model_0_path = "models/ppo_p0.zip"
    model_1_path = "models/ppo_p1.zip"

    if os.path.exists(model_0_path) and os.path.exists(model_1_path):
        print("Found existing models. Resuming training with RecurrentPPO...")
        try:
            # We exclude 'use_sde' and 'sde_sample_freq' because RecurrentPPO's policy 
            # might not support them, causing errors when loading standard PPO models.
            custom_objs = {
                "tensorboard_log": "./logs/player_0",
                "use_sde": None,
                "sde_sample_freq": None
            }
            model_0 = RecurrentPPO.load(
                model_0_path,
                env=make_opponent_env(base_pz, "player_0"),
                device=device,
                custom_objects=custom_objs,
                gamma=0.995,
            )
            
            custom_objs["tensorboard_log"] = "./logs/player_1"
            model_1 = RecurrentPPO.load(
                model_1_path,
                env=make_opponent_env(base_pz, "player_1"),
                device=device,
                custom_objects=custom_objs,
                gamma=0.995,
            )

            # Check for observation space mismatch (Dict spaces don't have .shape)
            new_env = make_opponent_env(base_pz, "player_0")
            if isinstance(model_0.observation_space, gym.spaces.Dict):
                if model_0.observation_space.keys() != new_env.observation_space.keys():
                     print("Observation space mismatch detected. Re-initializing models.")
                     model_0 = None
                     model_1 = None
            elif model_0.observation_space.shape != new_env.observation_space.shape:
                print("Observation space mismatch detected. Re-initializing models.")
                model_0 = None
                model_1 = None
        except Exception as e:
            print(f"Note: Could not resume from old models: {e}")
            print("This is normal when switching architectures. Initializing fresh RecurrentPPO models.")
            model_0 = None
            model_1 = None

    if 'model_0' not in locals() or model_0 is None:
        model_0 = RecurrentPPO(
            "MultiInputLstmPolicy",
            make_opponent_env(base_pz, "player_0"),
            verbose=0,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/player_0",
            device=device,
        )

        model_1 = RecurrentPPO(
            "MultiInputLstmPolicy",
            make_opponent_env(base_pz, "player_1"),
            verbose=0,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/player_1",
            device=device,
        )

    # 3. Iterative Training Loop
    generations = 10  # Number of times we switch training
    steps_per_gen = 40000  # steps per agent per generation

    print(
        f"Starting Alternating Training with RecurrentPPO: {generations} generations of {steps_per_gen} steps each."
    )

    for gen in range(generations):
        print(f"\n--- GENERATION {gen + 1} ---")

        # If first generation, create a shift in training
        if gen == 0:
            num_steps_gen_0 = steps_per_gen / 2
        else:
            num_steps_gen_0 = steps_per_gen

        # --- Phase A: Train Player 0 against Player 1 ---
        print(f"Phase A: Training Player 0 (Opponent: Player 1 Version {gen})")
        # Create a fresh wrapped env with the latest version of the opponent
        env_p0 = make_opponent_env(ClashRoyalePZ(), "player_0", opponent_model=model_1)
        model_0.set_env(env_p0)
        model_0.learn(total_timesteps=num_steps_gen_0, reset_num_timesteps=False)
        model_0.save(f"models/ppo_p0_gen{gen + 1}")
        model_0.save("models/ppo_p0")  # Latest version for easy loading

        # --- Phase B: Train Player 1 against Player 0 ---
        print(f"Phase B: Training Player 1 (Opponent: Player 0 Version {gen + 1})")
        # Create a fresh wrapped env with the latest version of Player 0
        env_p1 = make_opponent_env(ClashRoyalePZ(), "player_1", opponent_model=model_0)
        model_1.set_env(env_p1)
        model_1.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)
        model_1.save(f"models/ppo_p1_gen{gen + 1}")
        model_1.save("models/ppo_p1")  # Latest version for easy loading

    print("\nTraining Complete!")
    print("Final models saved as models/ppo_p0.zip and models/ppo_p1.zip")


if __name__ == "__main__":
    train()
