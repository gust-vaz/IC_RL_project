import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
import pandas as pd

def estimate_precipitation(week):
    week = int(week)   
    avg_value = avg_precipitation[week-1]
    variation_percentage = 0.10
    variation = np.random.uniform(-variation_percentage, variation_percentage) * avg_value
    estimated_value = avg_value + variation
    return estimated_value

weekly_sum = pd.read_csv('IC_RL_project/gym_custom_env/chuva_semanal.csv')
avg_precipitation = weekly_sum.groupby('week')['precipitacao(mm)'].mean().reset_index() 
avg_precipitation = avg_precipitation['precipitacao(mm)'].tolist()

register(
    id='factory-forest-env-v0',
    entry_point='v0_factory_forest_env:FactoryForestEnv', # module_name:class_name
)

class FactoryForestEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, plant_space=10000, total_production_per_year=2000, render_mode=None):
        self.n_trees = plant_space // 6

        # uso máximo de água e produção máximos por semana em litros e toneladas
        self.water_max_per_week = self.n_trees * 30 * 7                     # 350.000
        self.production_max_per_week = total_production_per_year / 53       # 37.7

        # absorção e geração máxima de carbono por semana em toneladas
        self.carbon_max_credit_per_week = 0.0003 * plant_space              # 3
        self.carbon_max_debit_per_week = 0.001 * total_production_per_year  # 2

        # custo e lucro máximo por semana em dólares
        self.cost_max_per_week = 0.003 * self.water_max_per_week            # 1050
        self.profit_max_per_week = 22 * self.production_max_per_week        # 830

        self.state = None
        self.last_action = None
        self.cashier = 0
        self.carbon_stock = 0

        self.low_state = np.array(
            [0, 0], dtype=np.float32
        )
        self.high_state = np.array(
            [53, self.water_per_week], dtype=np.float32
        )

        # Reward: [money, ghg]
        self.reward_space = self.reward_space = spaces.Box(low=np.array([-1000.0, -1000.0]), high=np.array([100, 100]), dtype=np.float32)
        self.reward_dim = 2

        # Act: [gasto_agua, producao]
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.water_max_per_week, self.production_max_per_week]), shape=(2,), dtype=np.float32)

        # Obs: [week, rain]
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        # Render: None or human
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.array([0,0], dtype=np.float32)
        info = {}

        self.last_action = None
        self.state = obs
        self.cashier = 0
        self.carbon_stock = 0

        if(self.render_mode=='human'):
            self.render()

        return obs, info

    def step(self, action: np.ndarray):
        self.state = np.array([self.state[0]+1, estimate_precipitation(self.state[0]+1)], dtype=np.float32)
        self.last_action = action

        reward = np.zeros(2, dtype=np.float32)
        if(action+self.state[1] > self.water_per_week):
            reward[0] = reward[1] = -1000
        else:
            reward[0] = - action[0] * 100 / (self.water_max_per_week - self.state[1])
            self.cashier -= 0.003 * action[0]
            self.carbon_stock += self.carbon_max_credit_per_week * (action[0]+self.state[1]) / self.water_max_per_week

            reward[1] = action[1] * 100 / self.production_max_per_week
            self.cashier += 22 * action[1]
            self.carbon_stock -= self.carbon_max_debit_per_week * action[1] / self.production_max_per_week


        obs = self.state
        terminated = obs[0] == 53
        truncated=False
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        if(self.last_action != None):
            print("Week: ", self.state[0])
            print("Rain: ", self.state[1])
            print("Action: ", self.last_action)
            print("Carbon: ", self.carbon_stock)
            print("Cash: ", self.cashier)
        else:
            print("Environment initializing...")
        print("")

# For unit testing
if __name__=="__main__":
    env = gym.make('factory-forest-env-v0', render_mode='human')
    print(env.observation_space)
    print(env.action_space)
    print(env.reward_space)

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    print("\nCOMEÇANDO RUN RANDOM\n")
    env.reset()
    terminated = False
    while(not terminated):
        action = env.action_space.sample()
        obs, _, terminated, _, _ = env.step(action)