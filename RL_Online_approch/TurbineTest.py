import os
import argparse
import json
import pandas as pd
import numpy as np
import v0_turbine_env 
import v1_turbine_env 
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
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

def test_sb3(env_args, timesteps, env_version, model_type, model_path):
    """
    Test a trained RL model on the Turbine environment using Stable Baselines3.
    Args:
        env_args (dict): Arguments for the environment.
        timesteps (int): Number of timesteps to test the model.
        env_version (str): Version of the environment ('v0', 'v1').
        model_type (str): Type of the RL model ('A2C', 'DQN', 'PPO').
        model_path (str): Path to the trained model file.
    """
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load environment
    env = gym.make('turbine-env-' + env_version, **env_args)

    # Load model
    model_class = {"A2C": A2C, "DQN": DQN, "PPO": PPO}[model_type]
    model = model_class.load(model_path, env=env)

    evaluation = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
    print(f"Evaluation over 10 episodes: Mean reward = {np.mean(evaluation[0])}, Std reward = {np.std(evaluation[0])}")

    # Set seed for reproducibility
    env_args['seed'] = 42
    env = gym.make('turbine-env-' + env_version, **env_args)
    model_class = {"A2C": A2C, "DQN": DQN, "PPO": PPO}[model_type]
    model = model_class.load(model_path, env=env)

    # Keep track of observations and actions
    H2_history = []
    Metano_history = []
    MaxEnergy_history = []
    GeneratedEnergy_history = []
    lower_threshold_history = []
    upper_threshold_history = []
    action_history = []
    reward_history = []

    # Run a test
    obs = env.reset()[0]
    save_every = env_args.get("history_length")
    for i in range(timesteps):
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        action_history.append(int(action))
        reward_history.append(reward)
        if (i + 1) % save_every == 0:
            H2_history.extend(obs[0])
            Metano_history.extend(obs[1])
            MaxEnergy_history.extend(obs[2])
            GeneratedEnergy_history.extend(obs[3])
            # In case thresholds are part of the observation
            if len(obs) > 4:
                lower_threshold_history.extend(obs[4])
            if len(obs) > 5:
                upper_threshold_history.extend(obs[5])

    csv_path = get_unique_path(f"{results_dir}/test_results_{model_type}_{env_version}.csv")

    # save the lists to a csv file using pandas
    data = {
        "actions": action_history,
        "rewards": reward_history,
        "H2": H2_history,
        "Metano": Metano_history,
        "MaxEnergy": MaxEnergy_history,
        "GeneratedEnergy": GeneratedEnergy_history,
    }
    if lower_threshold_history:
        data["lower_threshold"] = lower_threshold_history
    else:
        data["lower_threshold"] = [env_args.get("lower_threshold")] * len(action_history)
    if upper_threshold_history:
        data["upper_threshold"] = upper_threshold_history
    else:
        data["upper_threshold"] = [env_args.get("upper_threshold")] * len(action_history)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    print(f"Testing completed:\n 1's: {sum(action_history)} \n0's: {len(action_history) - sum(action_history)}")
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained RL model on the Turbine environment using Stable Baselines3.')
    parser.add_argument("--parameters", type=str, required=True, help="Path to the JSON file containing environment parameters.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of timesteps to test the model.")
    args = parser.parse_args()

    try:
        with open(args.parameters, 'r') as f:
            parameters = json.load(f)
        if not parameters:
            raise ValueError("Parameters file is empty.")
        env_args = parameters.get("env_args")
        model_type = parameters.get("model_type")
        env_version = parameters.get("env_version")
        if env_args is None or model_type is None or env_version is None:
            raise KeyError("Missing one or more required keys: 'env_args', 'model_type', 'env_version'.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading parameters file: {e}")
        exit(1)
    except (ValueError, KeyError) as e:
        print(f"Invalid parameters: {e}")
        exit(1)

    test_sb3(env_args, args.steps, env_version, model_type, args.model)