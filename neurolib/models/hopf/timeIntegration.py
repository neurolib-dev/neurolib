import numpy as np
import numba

from . import loadDefaultParams as dp
from ...utils import model_utils as mu


def timeIntegration(params):
    """Sets up the parameters for time integration
    
    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    # ------------------------------------------------------------------------
    # local parameters
    a = params["a"]  # Hopf bifurcation parameter
    w = params["w"]  # Oscillator frequency

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process) (mV/ms)
    x_ou_mean = params["x_ou_mean"]
    # Mean external inhibitory input (OU process) (mV/ms)
    y_ou_mean = params["y_ou_mean"]

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
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    x_ou = params["x_ou"]
    y_ou = params["y_ou"]

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    xs = np.zeros((N, startind + len(t)))
    ys = np.zeros((N, startind + len(t)))

    x_ext = mu.adjustArrayShape(params["x_ext"], xs)
    y_ext = mu.adjustArrayShape(params["y_ext"], ys)

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["xs_init"])[1] == 1:
        xs_init = np.dot(params["xs_init"], np.ones((1, startind)))
        ys_init = np.dot(params["ys_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        xs_init = params["xs_init"][:, -startind:]
        ys_init = params["ys_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    xs_input_d = np.zeros(N)  # delayed input to x
    ys_input_d = np.zeros(N)  # delayed input to y

    np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    xs[:, startind:] = np.random.standard_normal((N, len(t)))
    ys[:, startind:] = np.random.standard_normal((N, len(t)))

    xs[:, :startind] = xs_init
    ys[:, :startind] = ys_init

    noise_xs = np.zeros((N,))
    noise_ys = np.zeros((N,))

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
        x_ext,
        y_ext,
        a,
        w,
        noise_xs,
        noise_ys,
        x_ou,
        y_ou,
        x_ou_mean,
        y_ou_mean,
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
    x_ext,
    y_ext,
    a,
    w,
    noise_xs,
    noise_ys,
    x_ou,
    y_ou,
    x_ou_mean,
    y_ou_mean,
    tau_ou,
    sigma_ou,
):
    ### integrate ODE system:
    for i in range(startind, startind + len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the activity array
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

            # Stuart-Landau / Hopf Oscillator
            x_rhs = (
                (a - xs[no, i - 1] ** 2 - ys[no, i - 1] ** 2) * xs[no, i - 1]
                - w * ys[no, i - 1]
                + xs_input_d[no]  # input from other nodes
                + x_ou[no]  # ou noise
                + x_ext[no, i-1]  # external input
            )
            y_rhs = (
                (a - xs[no, i - 1] ** 2 - ys[no, i - 1] ** 2) * ys[no, i - 1]
                + w * xs[no, i - 1]
                + ys_input_d[no]  # input from other nodes
                + y_ou[no]  # ou noise
                + y_ext[no, i-1]  # external input
            )

            # Euler integration
            xs[no, i] = xs[no, i - 1] + dt * x_rhs
            ys[no, i] = ys[no, i - 1] + dt * y_rhs

            # Ornstein-Uhlenbeck process
            x_ou[no] = x_ou[no] + (x_ou_mean - x_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_xs[no]  # mV/ms
            y_ou[no] = y_ou[no] + (y_ou_mean - y_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_ys[no]  # mV/ms

    return t, xs, ys, x_ou, y_ou
