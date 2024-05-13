from gymnasium.envs.registration import register

register(
    id="StateSwitching-v0",
    entry_point="neurolib.control.reinforcement_learning.environments.state_switching:StateSwitchingEnv",
)
