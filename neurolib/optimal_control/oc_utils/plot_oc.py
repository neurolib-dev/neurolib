import matplotlib.pyplot as plt
import numpy as np


def plot_oc_singlenode(duration, dt, state, target, control, orig_input, cost_array=(), color_x="red", color_y="blue"):
    """Plot target and controlled dynamics for a network with a single node.
    :param duration:    Duration of simulation (in ms).
    :param dt:          Time discretization (in ms).
    :param state:       The state of the system controlled with the found oc-input.
    :param target:      The target state.
    :param control:     The control signal found by the oc-algorithm.
    :param orig_input:  The inputs that were used to generate target time series.
    :param cost_array:  Array of costs in optimization iterations.
    :param color_x:     Color used for plots of x-population variables.
    :param color_y:     Color used for plots of y-population variables.
    """
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)

    # Plot the target (dashed line) and unperturbed activity
    t_array = np.arange(0, duration + dt, dt)

    ax[0].plot(t_array, state[0, 0, :], label="x", color=color_x)
    ax[0].plot(t_array, state[0, 1, :], label="y", color=color_y)
    ax[0].plot(t_array, target[0, 0, :], linestyle="dashed", label="Target x", color=color_x)
    ax[0].plot(t_array, target[0, 1, :], linestyle="dashed", label="Target y", color=color_y)
    ax[0].legend()
    ax[0].set_title("Activity without stimulation and target activity")

    # Plot the target control signal (dashed line) and "initial" zero control signal
    ax[1].plot(t_array, control[0, 0, :], label="stimulation x", color=color_x)
    ax[1].plot(t_array, control[0, 1, :], label="stimulation y", color=color_y)
    ax[1].plot(t_array, orig_input[0, 0, :], linestyle="dashed", label="input x", color=color_x)
    ax[1].plot(t_array, orig_input[0, 1, :], linestyle="dashed", label="input y", color=color_y)
    ax[1].legend()
    ax[1].set_title("Active stimulation and input stimulation")

    ax[2].plot(cost_array)
    ax[2].set_title("Cost throughout optimization.")

    plt.show()


def plot_oc_network(
    N, duration, dt, state, target, control, orig_input, cost_array=(), step_array=(), color_x="red", color_y="blue"
):
    """Plot target and controlled dynamics for a network with a single node.
    :param N:           Number of nodes in the network.
    :param duration:    Duration of simulation (in ms).
    :param dt:          Time discretization (in ms).
    :param state:       The state of the system controlled with the found oc-input.
    :param target:      The target state.
    :param control:     The control signal found by the oc-algorithm.
    :param orig_input:  The inputs that were used to generate target time series.
    :param cost_array:  Array of costs in optimization iterations.
    :param step_array:  Number of iterations in the step-size algorithm in each optimization iteration.
    :param color_x:     Color used for plots of x-population variables.
    :param color_y:     Color used for plots of y-population variables.
    """

    t_array = np.arange(0, duration + dt, dt)
    fig, ax = plt.subplots(3, N, figsize=(12, 8), constrained_layout=True)

    for n in range(N):
        ax[0, n].plot(t_array, state[n, 0, :], label="x", color=color_x)
        ax[0, n].plot(t_array, state[n, 1, :], label="y", color=color_y)
        ax[0, n].plot(t_array, target[n, 0, :], linestyle="dashed", label="Target x", color=color_x)
        ax[0, n].plot(t_array, target[n, 1, :], linestyle="dashed", label="Target y", color=color_y)
        ax[0, n].legend()
        ax[0, n].set_title(f"Activity and target, node %s" % (n))

        # Plot the target control signal (dashed line) and "initial" zero control signal
        ax[1, n].plot(t_array, control[n, 0, :], label="stimulation x", color=color_x)
        ax[1, n].plot(t_array, control[n, 1, :], label="stimulation y", color=color_y)
        ax[1, n].plot(t_array, orig_input[n, 0, :], linestyle="dashed", label="input x", color=color_x)
        ax[1, n].plot(t_array, orig_input[n, 1, :], linestyle="dashed", label="input y", color=color_y)
        ax[1, n].legend()
        ax[1, n].set_title(f"Stimulation and input, node %s" % (n))

    ax[2, 0].plot(cost_array)
    ax[2, 0].set_title("Cost throughout optimization.")

    ax[2, 1].plot(step_array)
    ax[2, 1].set_title("Step size throughout optimization.")

    ax[2, 1].set_ylim(bottom=0, top=None)

    plt.show()
