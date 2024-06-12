from neurolib.utils.stimulus import ZeroInput

import numpy as np
import scipy

import gymnasium as gym
from gymnasium import spaces

from neurolib.models.wc import WCModel


class PhaseShiftingEnv(gym.Env):

    def __init__(
        self,
        duration=300,
        dt=0.1,
        random_target_shift=True,
        target_shift=1 * np.pi,
        exc_ext_baseline=2.8,
        inh_ext_baseline=1.2,
        x_init=0.04201540010391125,
        y_init=0.1354067401509556,
        sigma_ou=0.0,
        c_inhexc=16,
        c_excinh=10,
        c_inhinh=1,
        l1_control_strength_loss_scale=0.01,
        l2_control_strength_loss_scale=0.01,
    ):
        self.exc_ext_baseline = exc_ext_baseline
        self.inh_ext_baseline = inh_ext_baseline

        self.duration = duration
        self.dt = dt
        self.random_target_shift = random_target_shift
        self.target_shift = target_shift
        self.x_init = x_init
        self.y_init = y_init
        self.l1_control_strength_loss_scale = l1_control_strength_loss_scale
        self.l2_control_strength_loss_scale = l2_control_strength_loss_scale

        assert 0 < self.target_shift < 2 * np.pi

        self.model = WCModel()
        self.model.params["dt"] = self.dt
        self.model.params["sigma_ou"] = sigma_ou
        self.model.params["duration"] = self.dt  # one step at a time
        self.model.params["exc_init"] = np.array([[x_init]])
        self.model.params["inh_init"] = np.array([[y_init]])
        self.model.params["exc_ext_baseline"] = self.exc_ext_baseline
        self.model.params["inh_ext_baseline"] = self.inh_ext_baseline

        self.model.params["c_inhexc"] = c_inhexc
        self.model.params["c_excinh"] = c_excinh
        self.model.params["c_inhinh"] = c_inhinh
        self.params = self.model.params.copy()

        self.n_steps = round(self.duration / self.dt)

        self.target = self.get_target()

        self.observation_space = spaces.Dict(
            {
                "exc": spaces.Box(0, 1, shape=(self.period_n,), dtype=float),
                "inh": spaces.Box(0, 1, shape=(self.period_n,), dtype=float),
                "target_phase": spaces.Box(0, 2 * np.pi, shape=(1,), dtype=float),
            }
        )

        self.action_space = spaces.Tuple(
            (
                spaces.Box(-5, 5, shape=(1,), dtype=float),  # exc
                spaces.Box(-5, 5, shape=(1,), dtype=float),  # inh
            )
        )

    def get_target(self):
        wc = WCModel()
        wc.params = self.model.params.copy()
        wc.params["duration"] = self.duration + 100.0
        wc.run()

        peaks = scipy.signal.find_peaks(wc.exc[0, :])[0]
        p_list = []
        for i in range(3, len(peaks)):
            p_list.append(peaks[i] - peaks[i - 1])

        self.period_n = np.ceil(np.mean(p_list)).astype(int)

        period = np.mean(p_list) * self.dt
        self.period = period

        raw = np.stack((wc.exc, wc.inh), axis=1)[0]
        if self.random_target_shift:
            target_shift = np.random.random() * 2 * np.pi
        else:
            target_shift = self.target_shift
        index = np.round(target_shift * period / (2.0 * np.pi) / self.dt).astype(int)
        target = raw[:, index : index + np.round(1 + self.duration / self.dt, 1).astype(int)]
        self.target_time = wc.t[index : index + target.shape[1]]
        self.target_phase = (self.target_time % self.period) / self.period * 2 * np.pi

        return target

    def _get_obs(self):
        return {
            "exc": self.exc_history,
            "inh": self.inh_history,
            "target_phase": np.array([self.target_phase[self.t_i]]),
        }

    def _get_info(self):
        return {"t": self.t_i * self.dt}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.t_i = 0
        self.model.clearModelState()

        self.model.params = self.params.copy()

        # init history window
        self.model.params["duration"] = self.period_n * self.dt
        self.model.exc = np.array([[self.x_init]])
        self.model.inh = np.array([[self.y_init]])
        self.model.run()
        self.exc_history = self.model.exc[0]
        self.inh_history = self.model.inh[0]

        # reset duration parameter
        self.model.params = self.params.copy()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _loss(self, obs, action):
        control_loss = np.sqrt(
            (self.target[0, self.t_i] - obs["exc"][-1]) ** 2 + (self.target[1, self.t_i] - obs["inh"][-1]) ** 2
        )
        control_strength_loss = np.abs(action).sum() * self.l1_control_strength_loss_scale
        control_strength_loss += np.sqrt(np.sum(np.square(action))) * self.l2_control_strength_loss_scale

        return control_loss + control_strength_loss

    def step(self, action):
        assert self.action_space.contains(action)
        exc, inh = action
        self.model.params["exc_ext"] = np.array([exc])
        self.model.params["inh_ext"] = np.array([inh])
        self.model.run(continue_run=True)

        # shift observation window
        self.exc_history = np.concatenate((self.exc_history[-self.period_n + 1 :], self.model.exc[0]))
        self.inh_history = np.concatenate((self.inh_history[-self.period_n + 1 :], self.model.inh[0]))

        observation = self._get_obs()

        reward = -self._loss(observation, action)

        self.t_i += 1
        terminated = self.t_i >= self.n_steps
        info = self._get_info()

        return observation, reward, terminated, False, info
