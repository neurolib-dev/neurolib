import numpy as np

import neurolib.models.hopf.loadDefaultParams as dp
import numba


def timeIntegration(params):
    """
    TIMEINTEGRATION : Simulate a network of Hopf modules

    Return:
      x:  N*L array   : containing the x time series of the N nodes
      y:  N*L array   : containing the y time series of the N nodes
      t:          L array     : the time value at which the time series are evaluated
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    # ------------------------------------------------------------------------
    # local parameters
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    epsilon = params["epsilon"]
    tau = params["tau"]

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process) (mV/ms)
    x_ext_mean = params["x_ext_mean"]
    # Mean external inhibitory input (OU process) (mV/ms)
    y_ext_mean = params["y_ext_mean"]

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    K_gl = params["K_gl"]  # global coupling strength
    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    # Additive or diffusive coupling scheme
    coupling = params["coupling"]
    # convert to integer for faster integration later
    if coupling == "diffusive":
        coupling = 0
    elif coupling == "additive":
        coupling = 1
    else:
        raise ValueError('Paramter "coupling" must be either "diffusive" or "additive"')

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = dp.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
    params["Dmat_ndt"] = Dmat_ndt
    # ------------------------------------------------------------------------

    # Initialization
    t = np.arange(0, duration, dt)  # Time variable (ms)
    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    x_ext = np.zeros((N,))
    y_ext = np.zeros((N,))

    # Set the initial firing rates.
    if np.shape(params["xs_init"])[1] == 1:
        # If the initial firing rate is a 1D array, we use a fixed firing for a time "max_delay" before the simulation
        xs = np.dot(params["xs_init"], np.ones((1, len(t))))
        ys = np.dot(params["ys_init"], np.ones((1, len(t))))
    else:
        # Reuse the firing rates computed in a precedent simulation
        xs = np.zeros((N, len(t)))
        ys = np.zeros((N, len(t)))
        xs[:, :startind] = params["xs_init"][:, -startind:]
        ys[:, :startind] = params["ys_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    xs_input_d = np.zeros(N)  # delayed input to x
    ys_input_d = np.zeros(N)  # delayed input to x

    # Save the noise in the rates array to save memory
    if RNGseed:
        np.random.seed(RNGseed)
    xs[:, startind:] = np.random.standard_normal((N, len(range(startind, len(t)))))
    ys[:, startind:] = np.random.standard_normal((N, len(range(startind, len(t)))))

    noise_xs = np.zeros((N,))
    noise_ys = np.zeros((N,))

    zeros4 = np.zeros((4,))

    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        duration,
        N,
        Cmat,
        Dmat,
        K_gl,
        signalV,
        coupling,
        Dmat_ndt,
        xs,
        ys,
        xs_input_d,
        ys_input_d,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        tau,
        noise_xs,
        noise_ys,
        x_ext,
        y_ext,
        x_ext_mean,
        y_ext_mean,
        tau_ou,
        sigma_ou,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    duration,
    N,
    Cmat,
    Dmat,
    K_gl,
    signalV,
    coupling,
    Dmat_ndt,
    xs,
    ys,
    xs_input_d,
    ys_input_d,
    alpha,
    beta,
    gamma,
    delta,
    epsilon,
    tau,
    noise_xs,
    noise_ys,
    x_ext,
    y_ext,
    x_ext_mean,
    y_ext_mean,
    tau_ou,
    sigma_ou,
):
    """
    Fitz-Hugh Nagumo equations
    du/dt = -alpha u^3 + beta u^2 - gamma u - w + I_{ext}
    dw/dt = 1/tau (u + delta  - epsilon w)
    """
    ### integrate ODE system:
    for i in range(startind, len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the rates array
            noise_xs[no] = xs[no, i]
            noise_ys[no] = ys[no, i]

            # delayed input to each node
            xs_input_d[no] = 0
            ys_input_d[no] = 0

            # diffusive coupling
            if coupling == 0:
                for l in range(N):
                    xs_input_d[no] += K_gl * Cmat[no, l] * (xs[l, i - Dmat_ndt[no, l] - 1] - xs[no, i - 1])
                    # ys_input_d[no] += K_gl * Cmat[no, l] * (ys[l, i - Dmat_ndt[no, l] - 1] - ys[no, i - 1])
            # additive coupling
            elif coupling == 1:
                for l in range(N):
                    xs_input_d[no] += K_gl * Cmat[no, l] * (xs[l, i - Dmat_ndt[no, l] - 1])
                    # ys_input_d[no] += K_gl * Cmat[no, l] * (ys[l, i - Dmat_ndt[no, l] - 1])

            # Fitz-Hugh Nagumo equations
            x_rhs = (
                -alpha * xs[no, i - 1] ** 3
                + beta * xs[no, i - 1] ** 2
                + gamma * xs[no, i - 1]
                - ys[no, i - 1]
                + xs_input_d[no]  # input from other nodes
                + x_ext[no]  # input from external sources / noise
            )
            y_rhs = (
                (xs[no, i - 1] - delta - epsilon * ys[no, i - 1])
                / tau
                # + ys_input_d[no]
                # + y_ext[no]
            )

            # Euler integration
            xs[no, i] = xs[no, i - 1] + dt * x_rhs
            ys[no, i] = ys[no, i - 1] + dt * y_rhs

            # Ornstein-Uhlenberg process
            x_ext[no] = x_ext[no] + (x_ext_mean - x_ext[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_xs[no]  # mV/ms
            y_ext[no] = y_ext[no] + (y_ext_mean - y_ext[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_ys[no]  # mV/ms

    return t, xs, ys
