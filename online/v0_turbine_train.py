import argparse
import gymnasium as gym
from stable_baselines3 import A2C
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
    model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
    # Train the model
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(f"{model_dir}/a2c_{timesteps}")

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(env_args, timesteps, render):
    env = gym.make('turbine-env-v0', **env_args)

    # Load model
    model = A2C.load(f"models/a2c_{timesteps}", env=env)

    y_true = []
    y_pred = []

    # Run a test
    obs = env.reset()[0]
    for _ in range(100000):
        action, _ = model.predict(observation=obs, deterministic=True)  # Turn on deterministic, so predict always returns the same behavior
        obs, _, _, _, info = env.step(action)
        y_true.append(info["last_alert"])
        y_pred.append(info["last_action"])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=1))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot as an image file
    plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test the Turbine environment using Stable Baselines3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--timesteps", type=int, default=300000, help="Number of timesteps for training.")
    parser.add_argument("--render", action="store_true", help="Render the environment during testing.")
    parser.add_argument("--history_length", type=int, default=20, help="Length of the history buffer.")
    parser.add_argument("--reward_alert_no_action", type=float, default=-2, help="Reward for alert and no action.")
    parser.add_argument("--reward_alert_action", type=float, default=1, help="Reward for alert and action.")
    parser.add_argument("--reward_no_alert_no_action", type=float, default=0, help="Reward for no alert and no action.")
    parser.add_argument("--reward_no_alert_action", type=float, default=-0.2, help="Reward for no alert and action.")
    args = parser.parse_args()

    env_args = {
        "seed": args.seed,
        "history_length": args.history_length,
        "reward_alert_no_action": args.reward_alert_no_action,
        "reward_alert_action": args.reward_alert_action,
        "reward_no_alert_no_action": args.reward_no_alert_no_action,
        "reward_no_alert_action": args.reward_no_alert_action,
    }

    if args.train:
        train_sb3(env_args, args.timesteps)
    if args.test:
        test_sb3(env_args, args.timesteps, args.render)