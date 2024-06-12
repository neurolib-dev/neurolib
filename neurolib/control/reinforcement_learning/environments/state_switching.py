from neurolib.utils.stimulus import ZeroInput

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from neurolib.models.wc import WCModel


class StateSwitchingEnv(gym.Env):

    def __init__(
        self,
        duration=200,
        dt=0.1,
        target="up",
        exc_ext_baseline=2.9,
        inh_ext_baseline=3.3,
        l1_control_strength_loss_scale=0.005,
        l2_control_strength_loss_scale=0.005,
    ):
        self.exc_ext_baseline = exc_ext_baseline
        self.inh_ext_baseline = inh_ext_baseline
        self.compute_up_and_down_states()

        self.duration = duration
        self.dt = dt
        self.target = target
        self.l1_control_strength_loss_scale = l1_control_strength_loss_scale
        self.l2_control_strength_loss_scale = l2_control_strength_loss_scale

        assert self.target in ("up", "down")
        if self.target == "up":
            self.targetstate = self.upstate
            self.initstate = self.downstate
        elif self.target == "down":
            self.targetstate = self.downstate
            self.initstate = self.upstate

        self.model = WCModel()
        self.model.params["dt"] = self.dt
        self.model.params["duration"] = self.dt  # one step at a time
        self.model.params["exc_init"] = np.array([[self.initstate[0]]])
        self.model.params["inh_init"] = np.array([[self.initstate[1]]])
        self.model.params["exc_ext_baseline"] = self.exc_ext_baseline
        self.model.params["inh_ext_baseline"] = self.inh_ext_baseline

        self.n_steps = round(self.duration / self.dt)

        self.observation_space = spaces.Dict(
            {
                "exc": spaces.Box(0, 1, shape=(1,), dtype=float),
                "inh": spaces.Box(0, 1, shape=(1,), dtype=float),
            }
        )

        self.action_space = spaces.Tuple(
            (
                spaces.Box(-5, 5, shape=(1,), dtype=float),  # exc
                spaces.Box(-5, 5, shape=(1,), dtype=float),  # inh
            )
        )

    def compute_up_and_down_states(self):
        model = WCModel()

        dt = model.params["dt"]
        duration = 500
        model.params["duration"] = duration

        zero_input = ZeroInput().generate_input(duration=duration + dt, dt=dt)
        bi_control = zero_input.copy()
        bi_control[0, :500] = -5.0
        bi_control[0, 2500:3000] = +5.0

        model.params["exc_ext_baseline"] = self.exc_ext_baseline
        model.params["inh_ext_baseline"] = self.inh_ext_baseline
        model.params["exc_ext"] = bi_control
        model.params["inh_ext"] = zero_input
        model.run()
        self.downstate = [model.exc[0, 2000], model.inh[0, 2000]]
        self.upstate = [model.exc[0, -1], model.inh[0, -1]]

    def _get_obs(self):
        return {"exc": self.model.exc[0], "inh": self.model.inh[0]}

    def _get_info(self):
        return {"t": self.t_i * self.dt}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.t_i = 0
        self.model.clearModelState()

        self.model.params["exc_init"] = np.array([[self.initstate[0]]])
        self.model.params["inh_init"] = np.array([[self.initstate[1]]])
        self.model.exc = np.array([[self.initstate[0]]])
        self.model.inh = np.array([[self.initstate[1]]])

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _loss(self, obs, action):
        control_loss = abs(self.targetstate[0] - obs["exc"].item()) + abs(self.targetstate[1] - obs["inh"].item())
        control_strength_loss = np.abs(action).sum() * self.l1_control_strength_loss_scale
        control_strength_loss += np.sqrt(np.sum(action**2)) * self.l2_control_strength_loss_scale
        return control_loss + control_strength_loss

    def step(self, action):
        assert self.action_space.contains(action)
        exc, inh = action
        self.model.params["exc_ext"] = np.array([exc])
        self.model.params["inh_ext"] = np.array([inh])
        self.model.run(continue_run=True)

        observation = self._get_obs()

        reward = -self._loss(observation, action)

        self.t_i += 1
        terminated = self.t_i >= self.n_steps
        info = self._get_info()

        return observation, reward, terminated, False, info
