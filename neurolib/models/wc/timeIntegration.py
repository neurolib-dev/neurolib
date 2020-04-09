import numpy as np
import numba

from . import loadDefaultParams as dp


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
    # See Papadopoulos et al., Relations between large-scale brain connectivity and effects of regional stimulation
    # depend on collective dynamical state, arXiv, 2020
    tau_e = params["tau_e"]  #
    tau_i = params["tau_i"]  #
    c_ee = params["c_ee"]  #
    c_ei = params["c_ei"]  #
    c_ie = params["c_ie"]  #
    c_ii = params["c_ii"]  #
    a_e = params["a_e"]  #
    a_i = params["a_i"]  #
    mu_e = params["mu_e"]  #
    mu_i = params["mu_i"]  #

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process) (mV/ms)
    e_ou_mean = params["e_ou_mean"]
    # Mean external inhibitory input (OU process) (mV/ms)
    i_ou_mean = params["i_ou_mean"]

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connection from jth to ith
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

    e_ou = params["e_ou"]
    i_ou = params["i_ou"]

    e_ext = params["e_ext"]
    i_ext = params["i_ext"]


    # set of the state variable arrays
    exc = np.zeros((N, len(t)))
    inh = np.zeros((N, len(t)))

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["es_init"])[1] == 1:
        es_init = np.dot(params["es_init"], np.ones((1, startind)))
        is_init = np.dot(params["is_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        es_init = params["es_init"][:, -startind:]
        is_init = params["is_init"][:, -startind:]

    es_input_d = np.zeros(N)  # delayed input to e
    is_input_d = np.zeros(N)  # delayed input to i

    if RNGseed:
        np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    exc[:, startind:] = np.random.standard_normal((N, len(t) - startind))
    inh[:, startind:] = np.random.standard_normal((N, len(t) - startind))


    exc[:, :startind] = es_init
    inh[:, :startind] = is_init

    noise_es = np.zeros((N,))
    noise_is = np.zeros((N,))

    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        exc,
        inh,
        es_input_d,
        is_input_d,
        e_ext,
        i_ext,
        tau_e,
        tau_i,
        a_e,
        a_i,
        mu_e,
        mu_i,
        c_ee,
        c_ei,
        c_ie,
        c_ii,
        noise_es,
        noise_is,
        e_ou,
        i_ou,
        e_ou_mean,
        i_ou_mean,
        tau_ou,
        sigma_ou,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    N,
    Cmat,
    K_gl,
    Dmat_ndt,
    exc,
    inh,
    es_input_d,
    is_input_d,
    e_ext,
    i_ext,
    tau_e,
    tau_i,
    a_e,
    a_i,
    mu_e,
    mu_i,
    c_ee,
    c_ei,
    c_ie,
    c_ii,
    noise_es,
    noise_is,
    e_ou,
    i_ou,
    e_ou_mean,
    i_ou_mean,
    tau_ou,
    sigma_ou,
):
    ### integrate ODE system:

    def S_E(x):
        return 1.0 / (1.0 + np.exp(-a_e * (x - mu_e)))

    def S_I(x):
        return 1.0 / (1.0 + np.exp(-a_i * (x - mu_i)))

    for i in range(startind, startind + len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the activity array
            noise_es[no] = exc[no, i]
            noise_is[no] = inh[no, i]

            # delayed input to each node
            es_input_d[no] = 0
            is_input_d[no] = 0

            for l in range(N):
                es_input_d[no] += K_gl * Cmat[no, l] * (exc[l, i - Dmat_ndt[no, l] - 1])

            # Wilson-Cowan model
            e_rhs = (
                1
                / tau_e
                * (
                    -exc[no, i - 1]
                    + (1 - exc[no, i - 1])
                    * S_E(
                        c_ee * exc[no, i - 1]  # input from within the excitatory population
                        - c_ie * inh[no, i - 1]  # input from the inhibitory population
                        + es_input_d[no]  # input from other nodes
                        + e_ext[no]
                    )  # external input
                    + e_ou[no]  # ou noise
                )
            )
            i_rhs = (
                1
                / tau_i
                * (
                    -inh[no, i - 1]
                    + (1 - inh[no, i - 1])
                    * S_I(
                        c_ei * exc[no, i - 1]  # input from the excitatori population
                        - c_ii * inh[no, i - 1]  # input from within the inhibitori population
                        + es_input_d[no]  # input from other nodes
                        + i_ext[no]
                    )  # external input
                    + e_ou[no]  # ou noise
                )
            )

            # Euler integration
            exc[no, i] = exc[no, i - 1] + dt * e_rhs
            inh[no, i] = inh[no, i - 1] + dt * i_rhs

            # Ornstein-Uhlenbeck process
            e_ou[no] = e_ou[no] + (e_ou_mean - e_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_es[no]  # mV/ms
            i_ou[no] = i_ou[no] + (i_ou_mean - i_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_is[no]  # mV/ms

    return t, exc, inh, e_ou, i_ou
