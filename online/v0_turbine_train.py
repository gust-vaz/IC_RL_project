import gymnasium as gym
from stable_baselines3 import A2C
import v0_turbine_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('turbine-env-v0')

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 300000
    iters = 0
    while iters < 1:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True):
    env = gym.make('turbine-env-v0', render_mode='human' if render else None)

    # Load model
    model = A2C.load('models/a2c_300000', env=env)

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
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # Train/test using StableBaseline3
    train_sb3()
    test_sb3()