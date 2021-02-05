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
    a_exc = params["a_exc"]
    b_exc = params["b_exc"]
    d_exc = params["d_exc"]
    tau_exc = params["tau_exc"]
    gamma_exc = params["gamma_exc"]
    w_exc = params["w_exc"]
    exc_current = params["exc_current"]

    a_inh = params["a_inh"]
    b_inh = params["b_inh"]
    d_inh = params["d_inh"]
    tau_inh = params["tau_exc"]
    w_inh = params["w_inh"]
    inh_current = params["inh_current"]

    J_NMDA = params["J_NMDA"]
    J_I = params["J_I"]
    w_ee = params["w_ee"]

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process)
    exc_ou_mean = params["exc_ou_mean"]
    # Mean external inhibitory input (OU process)
    inh_ou_mean = params["inh_ou_mean"]
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
        Dmat = dp.computeDelayMatrix(lengthMat, signalV)
        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
    params["Dmat_ndt"] = Dmat_ndt

    # # Additive or diffusive coupling scheme
    # version = params["version"]
    # # convert to integer for faster integration later
    # if version == "original":
    #     version = 0
    # elif version == "reduced":
    #     version = 1
    # else:
    #     raise ValueError('Paramter "version" must be either "original" or "reduced"')

    # ------------------------------------------------------------------------

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    # noise variable
    exc_ou = params["exc_ou"]
    inh_ou = params["inh_ou"]

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    ses = np.zeros((N, startind + len(t)))
    sis = np.zeros((N, startind + len(t)))

    # holds firing rates
    r_exc = np.zeros((N, startind + len(t)))
    r_inh = np.zeros((N, startind + len(t)))

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["ses_init"])[1] == 1:
        ses_init = np.dot(params["ses_init"], np.ones((1, startind)))
        sis_init = np.dot(params["sis_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        ses_init = params["ses_init"][:, -startind:]
        sis_init = params["sis_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    ses_input_d = np.zeros(N)  # delayed input to x

    np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    ses[:, startind:] = np.random.standard_normal((N, len(t)))
    sis[:, startind:] = np.random.standard_normal((N, len(t)))

    ses[:, :startind] = ses_init
    sis[:, :startind] = sis_init

    noise_se = np.zeros((N,))
    noise_si = np.zeros((N,))

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
        Dmat_ndt,
        ses,
        sis,
        ses_input_d,
        a_exc,
        b_exc,
        d_exc,
        tau_exc,
        gamma_exc,
        w_exc,
        exc_current,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current,
        J_NMDA,
        J_I,
        w_ee,
        r_exc,
        r_inh,
        noise_se,
        noise_si,
        exc_ou,
        inh_ou,
        exc_ou_mean,
        inh_ou_mean,
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
    Dmat_ndt,
    ses,
    sis,
    ses_input_d,
    a_exc,
    b_exc,
    d_exc,
    tau_exc,
    gamma_exc,
    w_exc,
    exc_current,
    a_inh,
    b_inh,
    d_inh,
    tau_inh,
    w_inh,
    inh_current,
    J_NMDA,
    J_I,
    w_ee,
    r_exc,
    r_inh,
    noise_se,
    noise_si,
    exc_ou,
    inh_ou,
    exc_ou_mean,
    inh_ou_mean,
    tau_ou,
    sigma_ou,
):
    """
    Wong-Wang model equations (Deco2014, no long-rage feedforward inhibition):

    currents
    I_e = w_e * I_0 + w_ee * J_NMDA * s_e - J_I * s_i + K * J_NMDA * \sum Gij * s_e_j + I_ext
    I_i = w_i * I_0 + J_NMDA * s_e - s_i

    synaptic activity
    d_se/dt = (-s / tau) + (1.0 - s) * gamma * r + noise
    d_si/dt = (-s / tau) + r + noise

    firing rate transfer function
    r = (a * I - b) / (1.0 - exp(-d * (a * I - b)))

    """
    #  firing rate transfer function
    def r(I, a, b, d):
        return (a * I - b) / (1.0 - np.exp(-d * (a * I - b)))

    ### integrate ODE system:
    for i in range(startind, startind + len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the activity array
            noise_se[no] = ses[no, i]
            noise_si[no] = sis[no, i]

            # delayed input to each node
            ses_input_d[no] = 0

            # input from other nodes
            for l in range(N):
                ses_input_d[no] += K_gl * Cmat[no, l] * (ses[l, i - Dmat_ndt[no, l] - 1])

            # Wong-Wang
            se = ses[no, i - 1]
            si = sis[no, i - 1]

            I_exc = w_exc * exc_current + w_ee * J_NMDA * se - J_I * si + J_NMDA * ses_input_d[no]
            I_inh = w_inh * inh_current + J_NMDA * se - si

            r_exc[no, i] = r(I_exc, a_exc, b_exc, d_exc)
            r_inh[no, i] = r(I_inh, a_inh, b_inh, d_inh)

            se_rhs = -(se / tau_exc) + (1 - se) * gamma_exc * r_exc[no, i] + exc_ou[no]  # exc_ou = ou noise
            si_rhs = -(si / tau_inh) + r_inh[no, i] + inh_ou[no]

            # Euler integration
            ses[no, i] = ses[no, i - 1] + dt * se_rhs
            sis[no, i] = sis[no, i - 1] + dt * si_rhs

            # Ornstein-Uhlenberg process
            exc_ou[no] = (
                exc_ou[no] + (exc_ou_mean - exc_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_se[no]
            )  # mV/ms
            inh_ou[no] = (
                inh_ou[no] + (inh_ou_mean - inh_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_si[no]
            )  # mV/ms

    return t, r_exc, r_inh, ses, sis, exc_ou, inh_ou
