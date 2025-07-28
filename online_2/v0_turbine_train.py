import argparse
import json
import gymnasium as gym
from stable_baselines3 import A2C, DQN
import v0_turbine_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3(env_args, timesteps=300000):
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('turbine-env-v0', **env_args)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # Train the model
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(f"{model_dir}/a2c_{timesteps}")

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(env_args, timesteps, render):
    env = gym.make('turbine-env-v0', **env_args)

    # Load model
    model = DQN.load(f"models/a2c_{timesteps}", env=env)

    H2_history = []
    Metano_history = []
    MaxEnergy_history = []
    GeneratedEnergy_history = []

    action_history = []
    save_every = env_args.get("history_length", 20)
    save_data = []

    # Run a test
    obs = env.reset()[0]
    for i in range(20000):
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        action_history.append(int(action))
        if (i + 1) % save_every == 0:
            H2_history.extend(obs[0])
            Metano_history.extend(obs[1])
            MaxEnergy_history.extend(obs[2])
            GeneratedEnergy_history.extend(obs[3])

    def to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        return val

    save_data = {
        "actions": action_history,
        "H2": [to_serializable(h) for h in H2_history],
        "Metano": [to_serializable(m) for m in Metano_history],
        "MaxEnergy": [to_serializable(e) for e in MaxEnergy_history],
        "GeneratedEnergy": [to_serializable(g) for g in GeneratedEnergy_history]
    }
    print(save_data)
    with open("test_observations_actions.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print("Testing completed:", sum(action_history))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test the Turbine environment using Stable Baselines3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--timesteps", type=int, default=300000, help="Number of timesteps for training.")
    parser.add_argument("--render", action="store_true", help="Render the environment during testing.")
    parser.add_argument("--history_length", type=int, default=20, help="Length of the history buffer.")
    parser.add_argument("--reward_1", type=float, default=-20, help="Reward for alert and no action")
    parser.add_argument("--reward_2", type=float, default=-3, help="Reward for alert and action")
    parser.add_argument("--reward_3", type=float, default=0.001, help="Reward for no alert and no action")
    parser.add_argument("--reward_4", type=float, default=-0.2, help="Reward for no alert and action")
    args = parser.parse_args()

    env_args = {
        "seed": args.seed,
        "history_length": args.history_length,
        "reward_1": args.reward_1,
        "reward_2": args.reward_2,
        "reward_3": args.reward_3,
        "reward_4": args.reward_4,
    }

    if args.train:
        train_sb3(env_args, args.timesteps)
    if args.test:
        test_sb3(env_args, args.timesteps, args.render)