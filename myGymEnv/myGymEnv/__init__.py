from gym.envs.registration import register

register(
    id='centaur-v0',
    entry_point='myGymEnv.envs:CentaurEnv'
)