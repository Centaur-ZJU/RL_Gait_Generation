from q_learning import q_learning, test_Q
from environment import PendulumEnv
import plotting


def main():
    env = PendulumEnv(precision_th=3600, precision_thdot=100, max_speed=10, max_torque=10, dt=.05, nA=81, debug=False)
    Q, stats = q_learning(env, 10)
    # plotting.plot_episode_stats(stats)
    # test_Q(env,Q)


if __name__ == '__main__':
    main()

