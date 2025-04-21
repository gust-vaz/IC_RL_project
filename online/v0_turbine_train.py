import gymnasium as gym
from stable_baselines3 import A2C
import v0_turbine_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.
import os

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

    al = 0
    unmatches = 0
    # Run a test
    obs = env.reset()[0]
    terminated = False
    for _ in range (100000):
        action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
        obs, _, _, _, info = env.step(action)
        if info["last_alert"]:
            al += 1
        if info["last_alert"] != info["last_action"]:
            unmatches += 1
    print("alerts = ", al)
    print("unmatches = ", unmatches)

if __name__ == '__main__':
    # Train/test using StableBaseline3
    train_sb3()
    test_sb3()