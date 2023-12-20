import matplotlib.pyplot as plt
import numpy as np

colors = ["red", "blue", "green", "orange"]

def plot_oc_singlenode(
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
):
    """Plot target and controlled dynamics for a single node.
    :param duration:    Duration of simulation (in ms).
    :type duration:     float
    :param dt:          Time discretization (in ms).
    :type dt:           float
    :param state:       The state of the system controlled with the found oc-input.
    :type state:        np.ndarray
    :param target:      The target state.
    :type target:       np.ndarray
    :param control:     The control signal found by the oc-algorithm.
    :type control:      np.ndarray
    :param orig_input:  The inputs that were used to generate target time series.
    :type orig_input:   np.ndarray
    :param cost_array:  Array of costs in optimization iterations.
    :type cost_array:   np.ndarray, optional
    :param plot_state_vars:  List of indices of state variables that should be plotted
    :type plot_state_vars:   List, optional
    :param plot_control_vars:  List of indices of control variables that should be plotted
    :type plot_control_vars:   List, optional
    
    """
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)

    # Plot the target (dashed line) and unperturbed activity
    t_array = np.arange(0, duration + dt, dt)

    # Plot the controlled state and the initial/ original state (dashed line)
    for v in plot_state_vars:
        ax[0].plot(t_array, state[0, v, :], label="state var " + str(v), color=colors[v])
        ax[0].plot(t_array, target[0, v, :], linestyle="dashed", label="target var " + str(v), color=colors[v])
    ax[0].legend(loc="upper right")
    ax[0].set_title("Activity without stimulation and target activity")

    # Plot the computed control signal and the initial/ original control signal (dashed line)
    for v in plot_control_vars:
        ax[1].plot(t_array, control[0, v, :], label="stimulation var " + str(v), color=colors[v])
        ax[1].plot(t_array, orig_input[0, v, :], linestyle="dashed", label="input var " + str(v), color=colors[v])
    ax[1].legend(loc="upper right")
    ax[1].set_title("Active stimulation and input stimulation")

    ax[2].plot(cost_array)
    ax[2].set_title("Cost throughout optimization.")

    plt.show()


def plot_oc_network(
    N,
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    step_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
):
    """Plot target and controlled dynamics for a network of N nodes.
    :param N:           Number of nodes in the network.
    :type N:            int
    :param duration:    Duration of simulation (in ms).
    :type duration:     float
    :param dt:          Time discretization (in ms).
    :type dt:           float
    :param state:       The state of the system controlled with the found oc-input.
    :type state:        np.ndarray
    :param target:      The target state.
    :type target:       np.ndarray
    :param control:     The control signal found by the oc-algorithm.
    :type control:      np.ndarray
    :param orig_input:  The inputs that were used to generate target time series.
    :type orig_input:   np.ndarray
    :param cost_array:  Array of costs in optimization iterations.
    :type cost_array:   np.ndarray, optional
    :param step_array:  Array of step sizes in optimization iterations.
    :type step_array:   np.ndarray, optional
    :param plot_state_vars:  List of indices of state variables that should be plotted
    :type plot_state_vars:   List, optional
    :param plot_control_vars:  List of indices of control variables that should be plotted
    :type plot_control_vars:   List, optional
    """

    t_array = np.arange(0, duration + dt, dt)
    fig, ax = plt.subplots(3, N, figsize=(12, 8), constrained_layout=True)

    # Plot the controlled state and the initial/ original state (dashed line)
    for n in range(N):
        for v in plot_state_vars:
            ax[0, n].plot(t_array, state[n, v, :], label="state var " + str(v), color=colors[v])
            ax[0, n].plot(t_array, target[n, v, :], linestyle="dashed", label="target var " + str(v), color=colors[v])
        # ax[0, n].legend(loc="upper right")
        ax[0, n].set_title(f"Activity and target, node %s" % (n))

        # Plot the computed control signal and the initial/ original control signal (dashed line)
        for v in plot_control_vars:
            ax[1, n].plot(t_array, control[n, v, :], label="stimulation var " + str(v), color=colors[v])
            ax[1, n].plot(
                t_array, orig_input[n, v, :], linestyle="dashed", label="input var " + str(v), color=colors[v]
            )
        ax[1, n].set_title(f"Stimulation and input, node %s" % (n))

    ax[2, 0].plot(cost_array)
    ax[2, 0].set_title("Cost throughout optimization.")

    ax[2, 1].plot(step_array)
    ax[2, 1].set_title("Step size throughout optimization.")

    ax[2, 1].set_ylim(bottom=0, top=None)

    plt.show()
