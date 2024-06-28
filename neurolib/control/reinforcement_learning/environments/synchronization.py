import numpy as np

import gymnasium as gym
from gymnasium import spaces

from neurolib.models.wc import WCModel


class SynchronizationEnv(gym.Env):

    def __init__(
        self,
        duration=200,
        dt=0.1,
        observation_window=300,  # number of observed integration steps
        target="sync",
        l1_control_strength_loss_scale=1.0,
        l2_control_strength_loss_scale=1.0,
    ):
        self.duration = duration
        self.dt = dt
        self.observation_window = observation_window
        self.target = target
        self.l1_control_strength_loss_scale = l1_control_strength_loss_scale
        self.l2_control_strength_loss_scale = l2_control_strength_loss_scale

        assert target in ("sync", "desync")
        if target == "sync":
            self.exc_ext_baseline = 1.6  # starts in desync
        elif target == "desync":
            self.exc_ext_baseline = 1.0  # starts in sync
        self.inh_ext_baseline = 0.4
        self.coupling = 0.8
        self.N = 6
        self.cmat = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.dmat = np.array(
            [
                [0.0, 12.0, 0.0, 0.0, 0.0, 8.0],
                [8.0, 0.0, 13.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 11.0],
                [5.0, 17.0, 0.0, 14.0, 0.0, 18.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            ]
        )
        assert np.max(self.dmat / dt) < self.observation_window

        # numerically determined dominant freq in sync state, corresponds to ~19ms period
        self.oscillation_freq = 0.052
        """
        def get_oscillation_freq(data, dt):
            ps = np.abs(np.fft.fft(data, axis=-1))
            ps[:, 0] = 0.
            freqs = np.fft.fftfreq(round(ps.shape[1]*dt))
            return freqs[scipy.stats.mode(ps.argmax(axis=-1)).mode]
        """

        self.model = WCModel(Cmat=self.cmat, Dmat=self.dmat)
        self.model.params["dt"] = self.dt
        self.model.params["K_gl"] = self.coupling
        self.model.params["exc_ext_baseline"] = self.exc_ext_baseline
        self.model.params["inh_ext_baseline"] = self.inh_ext_baseline
        self.model.params["duration"] = self.dt  # one step at a time

        self.n_steps = round(self.duration / self.dt)

        # TODO
        self.observation_space = spaces.Dict(
            {
                "exc": spaces.Box(0, 1, shape=(self.N, self.observation_window), dtype=float),
                "inh": spaces.Box(0, 1, shape=(self.N, self.observation_window), dtype=float),
            }
        )

        self.action_space = spaces.Tuple(
            (
                spaces.Box(-5, 5, shape=(self.N,), dtype=float),  # exc
                spaces.Box(-5, 5, shape=(self.N,), dtype=float),  # inh
            )
        )

    def _get_obs(self):
        return {
            "exc": self.model.exc[:, -self.observation_window :],
            "inh": self.model.inh[:, -self.observation_window :],
        }

    def _get_info(self):
        return {"t": self.t_i * self.dt}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.t_i = 0
        self.model.clearModelState()

        self.model.params["duration"] = self.observation_window * self.dt
        self.model.run(continue_run=True, append_outputs=True)
        self.model.params["duration"] = self.dt  # one step at a time

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def synchronization(self, data):
        summed = data.sum(0)
        ps = np.abs(np.fft.fft(summed))
        freqs = np.fft.fftfreq(round(summed.shape[0] * self.dt))
        return ps[np.argmin(np.abs(freqs - self.oscillation_freq))]

    def _reward(self, obs, action):
        if self.target == "sync":
            control_reward = self.synchronization(obs["exc"])
        elif self.target == "desync":
            control_reward = -1 * self.synchronization(obs["exc"])
        control_strength_loss = np.abs(action).sum() * self.l1_control_strength_loss_scale
        control_strength_loss += np.sqrt(np.sum(np.square(action))) * self.l2_control_strength_loss_scale
        return control_reward - control_strength_loss

    def step(self, action):
        assert self.action_space.contains(action)
        exc, inh = action
        self.model.params["exc_ext"] = np.array(exc)
        self.model.params["inh_ext"] = np.array(inh)
        self.model.run(continue_run=True, append_outputs=True)

        observation = self._get_obs()

        reward = self._reward(observation, action)

        self.t_i += 1
        terminated = self.t_i >= self.n_steps
        info = self._get_info()

        return observation, reward, terminated, False, info
