from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch, ddpg_pytorch
import time
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name=time.strftime("%d-%m:%H_",time.localtime())+'ppo'+'_four_legs')
    eg.add('env_name', 'ChkDogBulletEnv-v0', '', False)
    eg.add('seed', [i for i in range(args.num_runs)])
    eg.add('epochs', 10000)
    # eg.add('clip_ratio', 0.3)
    # eg.add('target_kl', 100)
    # eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [[32,16]], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)
""" 
    eg2 = ExperimentGrid(name=time.strftime("%Y-%m%d",time.localtime())+'ddpg')
    eg2.add('env_name', 'HumanoidFlagrunBulletEnv-v0', '', True)
    eg2.add('seed', [10*i for i in range(args.num_runs)])
    eg2.add('epochs', 100)
    eg2.add('ac_kwargs:hidden_sizes', [[32,16,8],[64,32,16],[32,32,16],[32,16,16]], 'hid')
    eg2.run(ddpg_pytorch, num_cpu=args.cpu) """