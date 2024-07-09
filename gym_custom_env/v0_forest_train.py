import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
import os
import v0_forest_env

def train_sb3():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('forest-env-v0')
    model = A2C('MlpPolicy', env, verbose=1)

    TIMESTEPS = 1000
    iters = 0
    while iters < 10:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

def test_sb3(render=True):
    env = gym.make('forest-env-v0', render_mode='human' if render else None)
    model = A2C.load('models/a2c_2000', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            break

if __name__ == '__main__':
    # Train/test using StableBaseline3
    train_sb3()
    test_sb3()
