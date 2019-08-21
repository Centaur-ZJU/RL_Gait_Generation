import numpy as np
record_num = 40
max_epoch_steps = 10000


class PendulumEnv(object):
    def __init__(self, precision_th=3600, precision_thdot=100, max_speed=1, max_torque=2, dt=.05, nA=81, debug=False):
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.viewer = None
        self.nA = nA
        self.n_th = precision_th
        self.n_thdot = precision_thdot
        self.nS = self.n_th * self.n_thdot
        self.time = 0
        self.action_space = np.linspace(-self.max_torque, self.max_torque, self.nA)
        self.state = np.array([0., 0.])
        self.last_u = None
        self.debug = debug
        # self.recent_rewards = np.zeros(record_num)
        self.recent_theta = np.zeros(record_num)
        self.reset()

    def reset(self):
        th = np.random.random_sample() * np.pi * 2 - np.pi
        thdot = np.random.random_sample() * self.max_speed * 2 - self.max_speed
        self.state = th, thdot
        self.time = 0
        return self.get_state()

    def get_state(self):
        th, thdot = self.state
        s = int((th + np.pi) / np.pi / 2 * self.n_th) % self.n_th
        a = int((thdot + self.max_speed) / self.max_speed / 2 * self.n_thdot) % self.n_thdot
        return s * self.n_thdot + a

    def get_theta(self):
        return self.state

    def if_done(self):
        if self.time > max_epoch_steps:
            # print('epoch steps max, steps = '+str(self.time))
            return True
        # if np.all(np.abs(self.recent_theta) < 0.2) and self.time > 10:
        #     print(self.recent_theta)
        #     print('theta var = ', np.var(self.recent_theta))
        #     return True
        return False

    def step(self, action):
        th, thdot = self.state
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt
        u = np.clip([self.action_space[action]], -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th) + 3. / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        newth = ((newth + np.pi) % (2 * np.pi)) - np.pi
        # print('th={},thdot={},newthdot={},newth={},costs={}'.format(th, thdot, newthdot, newth, costs))
        self.state = np.array([newth, newthdot])
        if self.debug:
            self.debug_info()
        # print(self.state)
        # self.recent_rewards[self.time % record_num] = costs
        self.recent_theta[self.time % record_num] = newth
        self.time += 1
        return self.get_state(), -costs, self.if_done(), {}

    def debug_info(self):
        th, thdot = self.state
        g = 10.
        m = 1.
        l = 1.
        force_g = -3 * g / (2 * l) * np.sin(th + np.pi)
        force_u = 3. / (m * l ** 2) * self.last_u
        delta_th = (force_u + force_g) * self.dt
        # print('100 * sin(theta+ pi) = ', 100 * np.sin(th + np.pi))
        print('theta={}, delta_th={}, thdot={}, force_g={}, force_u={}'.format(th, delta_th, thdot, force_g, force_u))