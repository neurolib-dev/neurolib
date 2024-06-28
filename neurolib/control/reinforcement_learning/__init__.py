from gymnasium.envs.registration import register

register(
    id="StateSwitching-v0",
    entry_point="neurolib.control.reinforcement_learning.environments.state_switching:StateSwitchingEnv",
)

register(
    id="PhaseShifting-v0",
    entry_point="neurolib.control.reinforcement_learning.environments.phase_shifting:PhaseShiftingEnv",
)

register(
    id="Synchronization-v0",
    entry_point="neurolib.control.reinforcement_learning.environments.synchronization:SynchronizationEnv",
)
