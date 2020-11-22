from gym.envs.registration import register

register(
    id='chkCentaurEnv-v0',
    entry_point='ChkGymEnv.envs:ChkCentaurEnv',
    max_episode_steps=3000
)