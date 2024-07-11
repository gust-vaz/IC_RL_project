import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
from morl_baselines.multi_policy.morld.morld import MORLD
import v0_forest_env

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


if __name__ == '__main__':
    # Train/test
    GAMMA = 0.9
    ref_point = np.array([-53000., -53000.])
    env = mo_gym.make('forest-env-v0')
    env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics
    eval_env = mo_gym.make('forest-env-v0') # environment used for evaluation

    agent = PGMORL(
        env_id='forest-env-v0',
        warmup_iterations=20,
        evolutionary_iterations=20,
        origin=np.array([0.0, 0.0]),
        gamma=GAMMA,
        log=True
    )
    agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    for i in range(len(agent.archive.individuals)):
        print("Run Number: ", i)
        env = mo_gym.make('forest-env-v0', render_mode='human')
        obs, _ = env.reset()
        terminated = False
        while(not terminated):
            action = agent.archive.individuals[i].eval(obs, None)
            obs, reward, terminated, _, _ = env.step(action)

    # agent = CAPQL(
    #     env=env,
    #     gamma=GAMMA,
    #     log=True
    # )
    # agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    # agent.save()
    # agent.load("weights/CAPQL.tar")
    # agent._sample_batch_experiences()

    # agent = GPIPDContinuousAction(
    #     env,
    #     gamma=GAMMA,
    #     log=True
    # )
    # agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    # agent.save()
    # agent._sample_batch_experiences()

    # agent = MORLD(
    #     env=env,
    #     gamma=GAMMA,
    #     log=True
    # )
    # agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    # for i in range(len(agent.archive.individuals)):
    #     env = mo_gym.make('forest-env-v0', render_mode='human')
    #     obs, _ = env.reset()
    #     terminated = False
    #     while(not terminated):
    #         action = agent.archive.individuals[i].wrapped.eval(obs)
    #         obs, reward, terminated, _, _ = env.step(action)
