import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
# from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import v0_forest_env

if __name__ == '__main__':
    # Train/test
    GAMMA = 0.9
    ref_point = np.array([-53000., -53000.])
    env = mo_gym.make('forest-env-v0')
    env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics
    eval_env = mo_gym.make('forest-env-v0') # environment used for evaluation

    agent = PGMORL(
        env_id='forest-env-v0',
        origin=np.array([0.0, 0.0]),
        gamma=GAMMA,
        log=True,  # use weights and biases to see the results!
    )
    agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    

    # agent = MPMOQLearning(
    #     env,
    #     initial_epsilon=1.0,
    #     final_epsilon=0.05,
    #     epsilon_decay_steps=100000,
    #     gamma=GAMMA,
    #     dyna=True,
    #     gpi_pd=True,
    #     weight_selection_algo='gpi-ls',
    #     use_gpi_policy=True
    # )
    # agent.train(total_timesteps=100000, timesteps_per_iteration=10000, eval_env=eval_env, num_eval_episodes_for_front=50, ref_point=ref_point)

