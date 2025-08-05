import os
import argparse
import json
import numpy as np
import re
import v0_turbine_env 
import v1_turbine_env 
import v2_turbine_env 
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    return val

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def get_latest_model_path(model_dir, model_type, env_version):
    base_name = f"{model_type.lower()}-{env_version}"
    pattern = re.compile(rf"^{re.escape(base_name)}(\d+)?\.zip$")
    candidates = []
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            num = match.group(1)
            num = int(num) if num else 0
            candidates.append((num, fname))
    if not candidates:
        raise FileNotFoundError(f"No model files found for {base_name} in {model_dir}")
    latest_fname = max(candidates, key=lambda x: x[0])[1]
    return os.path.join(model_dir, latest_fname)


def train_sb3(env_args, timesteps=300000, env_version='v0', model_type='DQN'):
    parameters_dir = "parameters"
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('turbine-env-' + env_version, **env_args)

    model_class = {"A2C": A2C, "DQN": DQN, "PPO": PPO}[model_type]
    # Use MlpPolicy for observation space 1D vector.
    model = model_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

    # Save the model and parameters
    model_path = get_unique_path(f"{model_dir}/{model_type.lower()}-{env_version}")
    param_path = get_unique_path(f"{parameters_dir}/params_{model_type.lower()}_{env_version}.json")

    model.save(model_path)
    with open(param_path, "w") as f:
        params = {
            "env_args": env_args,
            "timesteps": timesteps,
            "model_type": model_type,
            "env_version": env_version
        }
        json.dump(params, f, indent=2)
    print(f"Training completed and model saved to {model_path}")


def test_sb3(env_args, timesteps_test=100000, env_version='v0', model_type='DQN', model_path=None):
    model_dir = "models"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    env = gym.make('turbine-env-' + env_version, **env_args)

    # Load model
    model_class = {"A2C": A2C, "DQN": DQN, "PPO": PPO}[model_type]
    if model_path is not None:
        model_path = os.path.join(model_dir, model_path)
    else:
        model_path = get_latest_model_path(model_dir, model_type, env_version)
    model = model_class.load(model_path, env=env)

    # Keep track of observations and actions
    H2_history = []
    Metano_history = []
    MaxEnergy_history = []
    GeneratedEnergy_history = []
    lower_threshold_history = []
    upper_threshold_history = []
    action_history = []

    # Run a test
    obs = env.reset()[0]
    save_every = env_args.get("history_length", 20)
    for i in range(timesteps_test):
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        action_history.append(int(action))
        if (i + 1) % save_every == 0:
            H2_history.extend(obs[0])
            Metano_history.extend(obs[1])
            MaxEnergy_history.extend(obs[2])
            GeneratedEnergy_history.extend(obs[3])
            if len(obs) > 4:
                lower_threshold_history.extend(obs[4])
            if len(obs) > 5:
                upper_threshold_history.extend(obs[5])

    save_data = {
        "actions": action_history,
        "H2": [to_serializable(h) for h in H2_history],
        "Metano": [to_serializable(m) for m in Metano_history],
        "MaxEnergy": [to_serializable(e) for e in MaxEnergy_history],
        "GeneratedEnergy": [to_serializable(g) for g in GeneratedEnergy_history]
    }
    if lower_threshold_history:
        save_data["lower_threshold"] = [to_serializable(l) for l in lower_threshold_history]
    if upper_threshold_history:
        save_data["upper_threshold"] = [to_serializable(u) for u in upper_threshold_history]

    with open(f"{results_dir}/test_results_{model_type.lower()}_{env_version}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print("Testing completed:\n", "1's:", sum(action_history), "\n0's:", len(action_history) - sum(action_history))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test the Turbine environment using Stable Baselines3.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--model_type", type=str, choices=["A2C", "DQN", "PPO"], default="DQN", help="Type of model to use.")
    parser.add_argument("--env_version", type=str, default="v0", help="Version of the environment to use.")
    parser.add_argument("--timesteps", type=int, default=300000, help="Number of timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--history_length", type=int, default=20, help="Length of the history buffer.")
    parser.add_argument("--reward_1", type=float, default=-20, help="Reward for alert and no action")
    parser.add_argument("--reward_2", type=float, default=-3, help="Reward for alert and action")
    parser.add_argument("--reward_3", type=float, default=0.001, help="Reward for no alert and no action")
    parser.add_argument("--reward_4", type=float, default=-0.2, help="Reward for no alert and action")
    parser.add_argument("--lower_threshold", type=float, default=50, help="Lower threshold for the turbine's H2 level.")
    parser.add_argument("--upper_threshold", type=float, default=70, help="Upper threshold for the turbine's H2 level.")

    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file for testing. If not provided, the latest model will be used.")
    parser.add_argument("--timesteps_test", type=int, default=100000, help="Number of timesteps for testing.")
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

    if args.train:
        train_sb3(env_args, args.timesteps, args.env_version, args.model_type)
    if args.test:
        test_sb3(env_args, args.timesteps_test, args.env_version, args.model_type, args.model_path)