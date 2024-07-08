import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
import pandas as pd

def estimate_precipitation(week):
    avg_value = avg_precipitation.loc[avg_precipitation['week'] == week]['precipitacao(mm)'][0]
    variation_percentage = 0.10
    variation = np.random.uniform(-variation_percentage, variation_percentage) * avg_value
    estimated_value = avg_value + variation
    return estimated_value

weekly_sum = pd.read_csv('chuva_semanal.csv')
avg_precipitation = weekly_sum.groupby('week')['precipitacao(mm)'].mean().reset_index()

register(
    id='forest-env-v0',
    entry_point='v0_forest_env:ForestEnv', # module_name:class_name
)

class ForestEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, plant_space=10000, render_mode=None):

        self.plant_space = plant_space
        self.n_trees = plant_space // 6
        self.water_per_week = self.n_tress * 30
        self.render_mode = render_mode
        self.carbon_stock_per_week = 3 * plant_space / 10000

        self.state = None
        self.last_action = None
        self.cost = 0
        self.carbon_stock = 0


        # Act: [gasto_agua]
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([self.water_per_week * 1.]), shape=(1,), dtype=np.float32)

        # Obs: [week, rain]
        self.observation_space = spaces.Box(
            low=np.array([1, 0.0]),
            high=np.array([53, self.water_per_week * 1.]),
            shape=(2,),
            dtype=np.float32
        )
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(53),  # Week of the year
        #     spaces.Box(low=np.array([0]), high=np.array([self.water_per_week * 1.]), dtype=np.float32)  # Rain observation
        # ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.array([0,0])
        info = {}

        self.last_action = None
        self.state = obs
        self.cost = 0
        self.carbon_stock = 0

        if(self.render_mode=='human'):
            self.render()

        return obs, info

    def step(self, action: np.ndarray):
        self.state = np.array([self.state[0]+1, estimate_precipitation(self.state[0]+1)])
        self.last_action = action

        reward = np.array([action[0] * 0.3, self.carbon_stock_per_week * (action[0] + self.state[1])])
        weight = np.array([0.5,0.5])

        self.cost += reward[0]
        self.carbon_stock += reward[1]

        obs = self.state
        terminated = obs[0] == 53
        truncated=False
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, reward@weight, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        if(self.last_action != None):
            print("Week: ", self.state[0])
            print("Rain: ", self.state[1])
            print("Action: ", self.last_action)
            print("Total Cost: ", self.cost)
            print("Total Carbon Stock: ", self.carbon_stock)
        else:
            print("Environment initializing...")
        print("")

# For unit testing
if __name__=="__main__":
    env = gym.make('forest-env-v0', render_mode='human')

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    obs, _ = env.reset()

    terminated = False
    while(not terminated):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)