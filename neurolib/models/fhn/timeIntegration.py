import numpy as np
import numba

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

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # Additive or diffusive coupling scheme
    coupling = params["coupling"]
    # convert to integer for faster integration later
    if coupling == "diffusive":
        coupling = 0
    elif coupling == "additive":
        coupling = 1
    else:
        raise ValueError('Paramter "coupling" must be either "diffusive" or "additive"')

    # ------------------------------------------------------------------------

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    x_ou = params["x_ou"].copy()
    y_ou = params["y_ou"].copy()

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
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        tau,
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
    alpha,
    beta,
    gamma,
    delta,
    epsilon,
    tau,
    noise_xs,
    noise_ys,
    x_ou,
    y_ou,
    x_ou_mean,
    y_ou_mean,
    tau_ou,
    sigma_ou,
):
    """
    Fitz-Hugh Nagumo equations
    du/dt = -alpha u^3 + beta u^2 + gamma u - w + I_{x, ext}
    dw/dt = 1/tau (u + delta  - epsilon w) + I_{y, ext}
    """
    ### integrate ODE system:
    for i in range(startind, startind + len(t)):
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
                + x_ou[no]  # ou noise
                + x_ext[no, i - 1]  # external input
            )
            y_rhs = (
                (xs[no, i - 1] - delta - epsilon * ys[no, i - 1]) / tau
                + ys_input_d[no]  # input from other nodes
                + y_ou[no]  # ou noise
                + y_ext[no, i - 1]  # external input
            )

            # Euler integration
            xs[no, i] = xs[no, i - 1] + dt * x_rhs
            ys[no, i] = ys[no, i - 1] + dt * y_rhs

            # Ornstein-Uhlenberg process
            x_ou[no] = x_ou[no] + (x_ou_mean - x_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_xs[no]  # mV/ms
            y_ou[no] = y_ou[no] + (y_ou_mean - y_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_ys[no]  # mV/ms

    return t, xs, ys, x_ou, y_ou


@numba.njit
def jacobian_fhn(
    model_params,
    x,
    V,
    sv,
):
    """Jacobian of a single node of the FHN dynamical system wrt. its 'state_vars' ('x', 'y', 'x_ou', 'y_ou'). The
       Jacobian of the FHN systems dynamics depends only on the constant model parameters and the values of the 'x'-
       population.

    :param model_params:    Ordered tuple of parameters in the FHN Model in order
    :type model_params:     tuple of float
    :param x:                   Value of the 'x'-population in the FHN node at a specific time step.
    :type x:                    float
    :param V:                   Number of system variables.
    :type V:                    int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict


    :return:                    V x V Jacobian matrix.
    :rtype:                     np.ndarray
    """
    (
        alpha,
        beta,
        gamma,
        tau,
        epsilon,
    ) = model_params
    jacobian = np.zeros((V, V))
    jacobian[sv["x"], sv["x"]] = 3 * alpha * x**2 - 2 * beta * x - gamma
    jacobian[sv["x"], sv["y"]] = 1.0
    jacobian[sv["y"], sv["x"]] = -1 / tau
    jacobian[sv["y"], sv["y"]] = epsilon / tau
    return jacobian


@numba.njit
def compute_hx(
    model_params,
    N,
    V,
    T,
    dyn_vars,
    sv,
):
    """Jacobians  of FHN model wrt. its 'state_vars' at each time step.

    :param model_params:    Ordered tuple of parameters in the FHN Model in order
    :type model_params:     tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Values of the 'x' and 'y' variable of FHN of all nodes through time.
    :type dyn_vars:             np.ndarray of shape N x 2 x T
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    Array that contains Jacobians for all nodes in all time steps.
    :rtype:                     np.ndarray of shape N x T x v X v
    """
    hx = np.zeros((N, T, V, V))

    for n in range(N):  # Iterate through nodes.
        for t in range(T):
            hx[n, t, :, :] = jacobian_fhn(model_params, dyn_vars[n, sv["x"], t], V, sv)
    return hx


@numba.njit
def compute_hx_nw(
    K_gl,
    cmat,
    coupling,
    N,
    V,
    T,
    sv,
):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:     Model parameter of global coupling strength.
    :type K_gl:      float
    :param cmat:     Model parameter, connectivity matrix.
    :type cmat:      ndarray
    :param coupling: Model parameter, which specifies the coupling type. E.g. "additive" or "diffusive".
    :type coupling:  str
    :param N:        Number of nodes in the network.
    :type N:         int
    :param V:        Number of system variables.
    :type V:         int
    :param T:        Length of simulation (time dimension).
    :type T:         int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:         Jacobians for network connectivity in all time steps.
    :rtype:          np.ndarray of shape N x N x T x 4 x 4
    """
    hx_nw = np.zeros((N, N, T, V, V))

    for n1 in range(N):
        for n2 in range(N):
            hx_nw[n1, n2, :, sv["x"], sv["x"]] = K_gl * cmat[n1, n2]  # term corresponding to additive coupling
            if coupling == "diffusive":
                hx_nw[n1, n1, :, sv["x"], sv["x"]] += -K_gl * cmat[n1, n2]

    return -hx_nw


@numba.njit
def Duh(
    N,
    V_in,
    V_vars,
    T,
    sv,
):
    """Jacobian of systems dynamics wrt. external inputs (control signals).

    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :rtype:     np.ndarray of shape N x V x V x T
    """

    duh = np.zeros((N, V_vars, V_in, T))
    for t in range(T):
        for n in range(N):
            duh[n, sv["x"], sv["x"], t] = -1.0
            duh[n, sv["y"], sv["y"], t] = -1.0
    return duh


@numba.njit
def Dxdoth(N, V):
    """Derivative of system dynamics wrt x dot

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of system variables.
    :type V:        int

    :return:        N x V x V matrix.
    :rtype:         np.ndarray
    """
    dxdoth = np.zeros((N, V, V))
    for n in range(N):
        for v in range(V):
            dxdoth[n, v, v] = 1

    return dxdoth
