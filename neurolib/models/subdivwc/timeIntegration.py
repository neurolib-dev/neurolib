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
    tau_exc = params["tau_exc"]  #
    tau_inh_s = params["tau_inh_s"]  #
    tau_inh_d = params["tau_inh_d"]  #
    w1 = params["w1"]  #
    w2 = params["w2"]  #
    w3 = params["w3"]  #
    w4 = params["w4"]  #
    w5 = params["w5"]  #
    w6 = params["w6"]  #
    w7 = params["w7"]  #
    alpha_exc = params["alpha_exc"]  #
    alpha_inh_s = params["alpha_inh_s"]  #
    alpha_inh_d =  params["alpha_inh_d"]  #
    theta_exc =  params["theta_exc"]  #
    theta_inh_s = params["theta_inh_s"]  #
    theta_inh_d = params["theta_inh_d"]  #
    q = params["q"]  #

    # external input parameters:
    # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    tau_ou = params["tau_ou"]
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    # Mean external excitatory input (OU process) (mV/ms)
    exc_ou_mean = params["exc_ou_mean"]
    # Mean external inhibitory input to somatic inhibition (OU process) (mV/ms)
    inh_s_ou_mean = params["inh_s_ou_mean"]
    # Mean external inhibitory input to dendritic inhibition (OU process) (mV/ms)
    inh_d_ou_mean = params["inh_d_ou_mean"]

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

    exc_ou = params["exc_ou"]
    inh_s_ou = params["inh_s_ou"]
    inh_d_ou = params["inh_d_ou"]

    exc_ext = params["exc_ext"]
    inh_s_ext = params["inh_s_ext"]
    inh_d_ext = params["inh_d_ext"]

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    excs = np.zeros((N, startind + len(t)))
    inh_ss = np.zeros((N, startind + len(t)))
    inh_ds = np.zeros((N, startind + len(t)))

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["exc_init"])[1] == 1:
        exc_init = np.dot(params["exc_init"], np.ones((1, startind)))
        inh_s_init = np.dot(params["inh_s_init"], np.ones((1, startind)))
        inh_d_init = np.dot(params["inh_d_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        exc_init = params["exc_init"][:, -startind:]
        inh_s_init = params["inh_s_init"][:, -startind:]
        inh_d_init = params["inh_d_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    exc_input_d = np.zeros(N)  # delayed input to exc
    inh_s_input_d = np.zeros(N)  # delayed input to inh_s
    inh_d_input_d = np.zeros(N)  # delayed input to inh_d

    if RNGseed:
        np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    excs[:, startind:] = np.random.standard_normal((N, len(t)))
    inh_ss[:, startind:] = np.random.standard_normal((N, len(t)))
    inh_ds[:, startind:] = np.random.standard_normal((N, len(t)))

    excs[:, :startind] = exc_init
    inh_ss[:, :startind] = inh_s_init
    inh_ds[:, :startind] = inh_d_init

    noise_exc = np.zeros((N,))
    noise_inh_s = np.zeros((N,))
    noise_inh_d = np.zeros((N,))

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
        excs,
        inh_ss,
        inh_ds,
        exc_input_d,
        inh_s_input_d,
        inh_d_input_d,
        exc_ext,
        inh_s_ext,
        inh_d_ext,
        tau_exc,
        tau_inh_s,
        tau_inh_d,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        alpha_exc,
        alpha_inh_s,
        alpha_inh_d,
        theta_exc,
        theta_inh_s,
        theta_inh_d,
        q,
        noise_exc,
        noise_inh_s,
        noise_inh_d,
        exc_ou,
        inh_s_ou,
        inh_d_ou,
        exc_ou_mean,
        inh_s_ou_mean,
        inh_d_ou_mean,
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
        excs,
        inh_ss,
        inh_ds,
        exc_input_d,
        inh_s_input_d,
        inh_d_input_d,
        exc_ext,
        inh_s_ext,
        inh_d_ext,
        tau_exc,
        tau_inh_s,
        tau_inh_d,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        alpha_exc,
        alpha_inh_s,
        alpha_inh_d,
        theta_exc,
        theta_inh_s,
        theta_inh_d,
        q,
        noise_exc,
        noise_inh_s,
        noise_inh_d,
        exc_ou,
        inh_s_ou,
        inh_d_ou,
        exc_ou_mean,
        inh_s_ou_mean,
        inh_d_ou_mean,
        tau_ou,
        sigma_ou,
):
    ### integrate ODE system:

    def F_exc(x, Theta, A):
        return (alpha_exc/(alpha_exc+q*A))*((1/(1+np.exp(-alpha_exc*(x-(theta_exc+Theta+(1-q)*A)))))
                                            -(1/(1+np.exp(alpha_exc*theta_exc))))

    def F_inh_s(x, Theta, A):
        return (alpha_inh_s/(alpha_inh_s+q*A))*((1/(1+np.exp(-alpha_inh_s*(x-(theta_inh_s+Theta+(1-q)*A)))))
                                            -(1/(1+np.exp(alpha_inh_s*theta_inh_s))))
    def F_inh_d(x, Theta, A):
        return (alpha_inh_d/(alpha_inh_d+q*A))*((1/(1+np.exp(-alpha_inh_d*(x-(theta_inh_d+Theta+(1-q)*A)))))
                                            -(1/(1+np.exp(alpha_inh_d*theta_inh_d))))

    k_exc = np.exp(alpha_exc)/(1+np.exp(alpha_exc))
    k_inh_s = np.exp(alpha_inh_s)/(1+np.exp(alpha_inh_s))
    k_inh_d = np.exp(alpha_inh_d)/(1+np.exp(alpha_inh_d))

    for i in range(startind, startind + len(t)):

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the activity array
            noise_exc[no] = excs[no, i]
            noise_inh_s[no] = inh_ss[no, i]
            noise_inh_d[no] = inh_ds[no, i]

            # delayed input to each node
            exc_input_d[no] = 0
            inh_s_input_d[no] = 0
            inh_d_input_d[no] = 0

            for l in range(N):
                exc_input_d[no] += K_gl * Cmat[no, l] * (excs[l, i - Dmat_ndt[no, l] - 1])



            # subtractive/divisive Wilson-Cowan model
            exc_rhs = (
                1/tau_exc*(-excs[no, i - 1]+(k_exc-excs[no, i - 1])*F_exc(w1*excs[no, i - 1]+exc_ext[no],
                           w2*inh_ds[no,i-1],w3*inh_ss[no,i-1]))
                    + exc_ou[no])
            inh_s_rhs = (
                1/tau_inh_s*(-inh_ss[no, i - 1]+(k_inh_s-inh_ss[no, i - 1])*F_inh_s(w5*excs[no, i - 1]+inh_s_ext[no],
                             w6*inh_ds[no,i-1]+w7*inh_ss[no,i-1],0))
                    + inh_s_ou[no])
            inh_d_rhs = (
                1/tau_inh_d*(-inh_ds[no, i - 1]+(k_inh_d-inh_ds[no, i - 1])*F_inh_d(w4*excs[no, i - 1]+inh_d_ext[no],0,0))
                    + inh_d_ou[no])

            # Euler integration
            excs[no, i] = excs[no, i - 1] + dt * exc_rhs
            inh_ss[no, i] = inh_ss[no, i - 1] + dt * inh_s_rhs
            inh_ds[no, i] = inh_ds[no, i - 1] + dt * inh_d_rhs

            # Ornstein-Uhlenbeck process
            exc_ou[no] = exc_ou[no] + (exc_ou_mean - exc_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]  # mV/ms
            inh_s_ou[no] = inh_s_ou[no] + (inh_s_ou_mean - inh_s_ou[no]) * dt / tau_ou + \
                           sigma_ou * sqrt_dt * noise_inh_s[no]  # mV/ms
            inh_d_ou[no] = inh_d_ou[no] + (inh_d_ou_mean - inh_d_ou[no]) * dt / tau_ou + \
                           sigma_ou * sqrt_dt * noise_inh_d[no]  # mV/ms

    return t, excs, inh_ss, inh_ds, exc_ou, inh_s_ou, inh_d_ou
