import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
import v0_forest_env

if __name__ == '__main__':
    # Train/test
    GAMMA = 0.9
    ref_point = np.array([-53000., -53000.])
    env = mo_gym.make('forest-env-v0')
    env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics
    eval_env = mo_gym.make('forest-env-v0') # environment used for evaluation

    # agent = PGMORL(
    #     env_id='forest-env-v0',
    #     origin=np.array([0.0, 0.0]),
    #     gamma=GAMMA,
    #     log=True
    # )
    # agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)

    # agent = CAPQL(
    #     env=env,
    #     gamma=GAMMA,
    #     log=True
    # )
    # agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    # agent.save()
    # agent.load("weights/CAPQL.tar")
    # agent._sample_batch_experiences()

    agent = GPIPDContinuousAction(
        env,
        gamma=GAMMA,
        log=True
    )
    agent.train(total_timesteps=10000, eval_env=eval_env, ref_point=ref_point)
    agent.save()
    agent._sample_batch_experiences()