python -m spinup.run ppo --exp_name ppo_pendulum --env HumanoidFlagrunBulletEnv --clip_ratio 0.1 0.2 \
--hid[h] [32,32] [64,32] --act torch.nn.Tanh \
--seed 0 10 20 --dt