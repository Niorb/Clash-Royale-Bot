import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class OpponentWrapper(gym.Env):
    """
    A wrapper that turns a multi-agent PettingZoo environment into a 
    single-agent Gymnasium environment by using a fixed opponent model.
    """
    def __init__(self, pz_env, learning_agent_id, opponent_model=None):
        super().__init__()
        self.env = pz_env
        self.learning_agent_id = learning_agent_id
        self.opponent_id = "player_1" if learning_agent_id == "player_0" else "player_0"
        self.opponent_model = opponent_model
        
        # Define single-agent spaces based on the PettingZoo spaces
        self.action_space = self.env.action_space(self.learning_agent_id)
        self.observation_space = self.env.observation_space(self.learning_agent_id)
        
        self.last_obs = None

    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        self.last_obs = obs_dict
        return obs_dict[self.learning_agent_id], infos[self.learning_agent_id]

    def step(self, action):
        # 1. Get Opponent's action if model is available, otherwise Nothing ([0, 0, 0])
        opp_action = [0, 0, 0]
        if self.opponent_model:
            # Predict the opponent's move using their observation
            opp_obs = self.last_obs[self.opponent_id]
            opp_action, _ = self.opponent_model.predict(opp_obs, deterministic=True)

        # 2. Package actions for PettingZoo
        actions = {self.learning_agent_id: action, self.opponent_id: opp_action}

        # 3. Step the environment
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        self.last_obs = obs_dict
        
        # 4. Extract learning agent's data
        reward = rewards[self.learning_agent_id]
        terminated = terminations[self.learning_agent_id]
        truncated = truncations[self.learning_agent_id]
        info = infos[self.learning_agent_id]
        
        return obs_dict[self.learning_agent_id], reward, terminated, truncated, info

    def render(self):
        return self.env.render()
