import os
import argparse
import json
import numpy as np
import v0_turbine_env 
import v1_turbine_env 
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C, DQN, PPO

def get_unique_path(base_path):
    """
    Generate a unique file path by appending an incrementing number if the file already exists.
    Args:
        base_path (str): The desired file path.
    """
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}-{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def train_sb3(env_args, timesteps, env_version, model_type):
    """
    Train an RL model on the Turbine environment using Stable Baselines3.
    Args:
        env_args (dict): Arguments for the environment.
        timesteps (int): Number of timesteps to train the model.
        env_version (str): Version of the environment ('v0', 'v1').
        model_type (str): Type of the RL model ('A2C', 'DQN', 'PPO').
    """
    # Save training parameters
    os.makedirs("parameters", exist_ok=True)
    param_path = get_unique_path(f"parameters/params_{model_type.lower()}_{env_version}.json")
    with open(param_path, "w") as f:
        params = {
            "env_args": env_args,
            "timesteps": timesteps,
            "model_type": model_type,
            "env_version": env_version
        }
        json.dump(params, f, indent=2)

    # Create environment
    env = gym.make('turbine-env-' + env_version, **env_args)

    # Create evaluation callback
    os.makedirs("best_models", exist_ok=True)
    eval_callback = EvalCallback(env, n_eval_episodes=5, best_model_save_path="best_models/",
                             log_path="best_models/", eval_freq=timesteps//10,
                             deterministic=True, render=False, verbose=1)

    # Create and train model
    os.makedirs("tensorboard_logs", exist_ok=True)
    model_class = {"A2C": A2C, "DQN": DQN, "PPO": PPO}[model_type]
    model = model_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log="tensorboard_logs/")
    print("Model Policy Architecture:")
    print(model.policy)
    print("Starting training...")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=eval_callback, progress_bar=True)

    # Save the model and parameters
    os.makedirs("saved_models", exist_ok=True)
    model_path = get_unique_path(f"saved_models/{model_type.lower()}-{env_version}.zip")
    model.save(model_path)
    print(f"Training completed and model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test the Turbine environment using Stable Baselines3.")
    parser.add_argument("--model_type", type=str, choices=["A2C", "DQN", "PPO"], default="DQN", help="Type of model to use.")
    parser.add_argument("--env_version", type=str, default="v0", help="Version of the environment to use.")
    parser.add_argument("--timesteps", type=int, default=500000, help="Number of timesteps for training.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--history_length", type=int, default=20, help="Length of the history buffer.")
    parser.add_argument("--reward_1", type=float, default=-20, help="Reward for alert and no action")
    parser.add_argument("--reward_2", type=float, default=-3, help="Reward for alert and action")
    parser.add_argument("--reward_3", type=float, default=0.001, help="Reward for no alert and no action")
    parser.add_argument("--reward_4", type=float, default=-0.2, help="Reward for no alert and action")
    parser.add_argument("--lower_threshold", type=float, default=None, help="Lower threshold for the turbine's H2 level.")
    parser.add_argument("--upper_threshold", type=float, default=None, help="Upper threshold for the turbine's H2 level.")
    args = parser.parse_args()

    env_args = {
        "seed": args.seed,
        "history_length": args.history_length,
        "reward_1": args.reward_1,
        "reward_2": args.reward_2,
        "reward_3": args.reward_3,
        "reward_4": args.reward_4,
        "lower_threshold": args.lower_threshold,
        "upper_threshold": args.upper_threshold
    }

    train_sb3(env_args, args.timesteps, args.env_version, args.model_type)