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
    a_exc = params["a_exc"]
    b_exc = params["b_exc"]
    d_exc = params["d_exc"]
    tau_exc = params["tau_exc"]
    gamma_exc = params["gamma_exc"]
    w_exc = params["w_exc"]
    exc_current_baseline = params["exc_current_baseline"]

    a_inh = params["a_inh"]
    b_inh = params["b_inh"]
    d_inh = params["d_inh"]
    tau_inh = params["tau_exc"]
    w_inh = params["w_inh"]
    inh_current_baseline = params["inh_current_baseline"]

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
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

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
    exc_ou = params["exc_ou"].copy()
    inh_ou = params["inh_ou"].copy()

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    ses = np.zeros((N, startind + len(t)))
    sis = np.zeros((N, startind + len(t)))

    # holds firing rates
    r_exc = np.zeros((N, startind + len(t)))
    r_inh = np.zeros((N, startind + len(t)))

    exc_current = mu.adjustArrayShape(params["exc_current"], r_exc)
    inh_current = mu.adjustArrayShape(params["inh_current"], r_inh)

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
        exc_current_baseline,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current,
        inh_current_baseline,
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
    exc_current_baseline,
    a_inh,
    b_inh,
    d_inh,
    tau_inh,
    w_inh,
    inh_current,
    inh_current_baseline,
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
                ses_input_d[no] += (
                    K_gl * Cmat[no, l] * (ses[l, i - Dmat_ndt[no, l] - 1])
                )

            # Wong-Wang
            se = ses[no, i - 1]
            si = sis[no, i - 1]

            I_exc = (
                w_exc * (exc_current_baseline + exc_current[no, i - 1])
                + w_ee * J_NMDA * se
                - J_I * si
                + J_NMDA * ses_input_d[no]
            )
            I_inh = (
                w_inh * (inh_current_baseline + inh_current[no, i - 1])
                + J_NMDA * se
                - si
            )

            r_exc[no, i] = r(I_exc, a_exc, b_exc, d_exc)
            r_inh[no, i] = r(I_inh, a_inh, b_inh, d_inh)

            se_rhs = (
                -(se / tau_exc) + (1 - se) * gamma_exc * r_exc[no, i] + exc_ou[no]
            )  # exc_ou = ou noise
            si_rhs = -(si / tau_inh) + r_inh[no, i] + inh_ou[no]

            # Euler integration
            ses[no, i] = ses[no, i - 1] + dt * se_rhs
            sis[no, i] = sis[no, i - 1] + dt * si_rhs

            # Ornstein-Uhlenberg process
            exc_ou[no] = (
                exc_ou[no]
                + (exc_ou_mean - exc_ou[no]) * dt / tau_ou
                + sigma_ou * sqrt_dt * noise_se[no]
            )  # mV/ms
            inh_ou[no] = (
                inh_ou[no]
                + (inh_ou_mean - inh_ou[no]) * dt / tau_ou
                + sigma_ou * sqrt_dt * noise_si[no]
            )  # mV/ms

    return t, r_exc, r_inh, ses, sis, exc_ou, inh_ou


@numba.njit
def logistic(x, a, b, d):
    """Logistic function evaluated at point 'x'.

    :type x:    float
    :param a:   Parameter of logistic function.
    :type a:    float
    :param b:  Parameter of logistic function.
    :type b:   float
    :param d:   Parameter of logistic function.
    :type d:    float

    :rtype:     float
    """
    return (a * x - b) / (1.0 - np.exp(-d * (a * x - b)))


@numba.njit
def logistic_der(x, a, b, d):
    """Derivative of logistic function, evaluated at point 'x'.

    :type x:    float
    :param a:   Parameter of logistic function.
    :type a:    float
    :param b:  Parameter of logistic function.
    :type b:   float
    :param d:   Parameter of logistic function.
    :type d:    float

    :rtype:     float
    """
    exp = np.exp(-d * (a * x - b))
    return (a * (1.0 - exp) - (a * x - b) * d * a * exp) / (1.0 - exp) ** 2


@numba.njit
def jacobian_ww(
    model_params,
    nw_se,
    re,
    se,
    si,
    ue,
    ui,
    V,
    sv,
):
    """Jacobian of the WW dynamical system.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param  nw_se:          N x T input of network into each node's 'exc'
    :type  nw_se:           np.ndarray
    :param re:              Value of the r_exc-variable at specific time.
    :type re:               float
    :param se:              Value of the se-variable at specific time.
    :type se:               float
    :param si:              Value of the si-variable at specific time.
    :type si:               float
    :param ue:              Value of control input to into 'exc' at specific time.
    :type ue:               float
    :param ui:              Value of control input to into 'ihn' at specific time.
    :type ui:               float
    :param V:               Number of system variables.
    :type V:                int
    :param sv:              dictionary of state vars and respective indices
    :type sv:               dict

    :return:        4 x 4 Jacobian matrix.
    :rtype:         np.ndarray
    """
    (
        a_exc,
        b_exc,
        d_exc,
        tau_exc,
        gamma_exc,
        w_exc,
        exc_current_baseline,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current_baseline,
        J_NMDA,
        J_I,
        w_ee,
    ) = model_params

    jacobian = np.zeros((V, V))
    IE = (
        w_exc * (exc_current_baseline + ue)
        + w_ee * J_NMDA * se
        - J_I * si
        + J_NMDA * nw_se
    )
    jacobian[sv["r_exc"], sv["se"]] = (
        -logistic_der(IE, a_exc, b_exc, d_exc) * w_ee * J_NMDA
    )
    jacobian[sv["r_exc"], sv["si"]] = logistic_der(IE, a_exc, b_exc, d_exc) * J_I
    II = w_inh * (inh_current_baseline + ui) + J_NMDA * se - si
    jacobian[sv["r_inh"], sv["se"]] = -logistic_der(II, a_inh, b_inh, d_inh) * J_NMDA
    jacobian[sv["r_inh"], sv["si"]] = logistic_der(II, a_inh, b_inh, d_inh)

    jacobian[sv["se"], sv["r_exc"]] = -(1.0 - se) * gamma_exc
    jacobian[sv["se"], sv["se"]] = 1.0 / tau_exc + gamma_exc * re

    jacobian[sv["si"], sv["r_inh"]] = -1.0
    jacobian[sv["si"], sv["si"]] = 1.0 / tau_inh
    return jacobian


@numba.njit
def compute_hx(
    wc_model_params,
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    dyn_vars,
    dyn_vars_delay,
    control,
    sv,
):
    """Jacobians of WWModel wrt. the  all variables for each time step.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:            Model parameter of global coupling strength.
    :type K_gl:             float
    :param cmat:            Model parameter, connectivity matrix.
    :type cmat:             ndarray
    :param dmat_ndt:        N x N delay matrix in multiples of dt.
    :type dmat_ndt:         np.ndarray
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V:               Number of system variables.
    :type V:                int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param dyn_vars:        N x V x T array containing all values of 'exc' and 'inh'.
    :type dyn_vars:         np.ndarray
    :param dyn_vars_delay:
    :type dyn_vars_delay:   np.ndarray
    :param control:     N x 2 x T control inputs to 'exc' and 'inh'.
    :type control:      np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:            N x T x 4 x 4 Jacobians.
    :rtype:             np.ndarray
    """
    hx = np.zeros((N, T, V, V))
    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, dyn_vars_delay[:, sv["se"], :])

    for n in range(N):
        for t in range(T):
            re = dyn_vars[n, sv["r_exc"], t]
            se = dyn_vars[n, sv["se"], t]
            si = dyn_vars[n, sv["si"], t]
            ue = control[n, sv["r_exc"], t]
            ui = control[n, sv["r_inh"], t]
            hx[n, t, :, :] = jacobian_ww(
                wc_model_params,
                nw_e[n, t],
                re,
                se,
                si,
                ue,
                ui,
                V,
                sv,
            )
    return hx


@numba.njit
def jacobian_ww_min1(
    model_params,
    se,
    V,
    sv,
):
    """Jacobian of the WW dynamical system.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param se:              Value of the se-variable at specific time.
    :type se:               float
    :param V:               Number of system variables.
    :type V:                int
    :param sv:              dictionary of state vars and respective indices
    :type sv:               dict

    :return:        4 x 4 Jacobian matrix.
    :rtype:         np.ndarray
    """
    (
        a_exc,
        b_exc,
        d_exc,
        tau_exc,
        gamma_exc,
        w_exc,
        exc_current_baseline,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current_baseline,
        J_NMDA,
        J_I,
        w_ee,
    ) = model_params

    jacobian = np.zeros((V, V))

    jacobian[sv["se"], sv["r_exc"]] = -(1.0 - se) * gamma_exc
    jacobian[sv["si"], sv["r_inh"]] = -1.0
    return jacobian


@numba.njit
def compute_hx_min1(
    wc_model_params,
    N,
    V,
    T,
    dyn_vars,
    sv,
):
    """Jacobians of WWModel wrt. the  all variables for each time step.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V:               Number of system variables.
    :type V:                int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param dyn_vars:        N x V x T array containing all values of 'exc' and 'inh'.
    :type dyn_vars:         np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:            N x T x 4 x 4 Jacobians.
    :rtype:             np.ndarray
    """
    hx = np.zeros((N, T, V, V))

    for n in range(N):
        for t in range(T):
            se = dyn_vars[n, sv["se"], t]
            hx[n, t, :, :] = jacobian_ww_min1(
                wc_model_params,
                se,
                V,
                sv,
            )
    return hx


@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, se):
    """Compute input by other nodes of network into each node's 'exc' population at every timestep.

    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param se:          N x T array containing values of 'exc' of all nodes through time.
    :type se:           np.ndarray
    :return:            N x T network inputs.
    :rytpe:             np.ndarray
    """
    nw_input = np.zeros((N, T))

    for t in range(1, T):
        for n in range(N):
            for l in range(N):
                nw_input[n, t] += K_gl * cmat[n, l] * (se[l, t - dmat_ndt[n, l] - 1])
    return nw_input


@numba.njit
def compute_hx_nw(
    model_params,
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    se,
    si,
    se_delay,
    ue,
    sv,
):
    """Jacobians for network connectivity in all time steps.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:            Model parameter of global coupling strength.
    :type K_gl:             float
    :param cmat:            Model parameter, connectivity matrix.
    :type cmat:             ndarray
    :param dmat_ndt:        N x N delay matrix in multiples of dt.
    :type dmat_ndt:         np.ndarray
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V:               Number of system variables.
    :type V:                int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param se:              Array of the se-variable.
    :type se:               np.ndarray
    :param si:              Array of the se-variable.
    :type si:               np.ndarray
    :param se_delay:        Value of delayed se-variable.
    :type se_delay:         np.ndarray
    :param ue:              N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:               np.ndarray
    :param sv:              dictionary of state vars and respective indices
    :type sv:               dict

    :return:                Jacobians for network connectivity in all time steps.
    :rtype:                 np.ndarray of shape N x N x T x 4 x 4
    """
    (
        a_exc,
        b_exc,
        d_exc,
        tau_exc,
        gamma_exc,
        w_exc,
        exc_current_baseline,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current_baseline,
        J_NMDA,
        J_I,
        w_ee,
    ) = model_params
    hx_nw = np.zeros((N, N, T, V, V))

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, se_delay)
    IE = (
        w_exc * (exc_current_baseline + ue)
        + w_ee * J_NMDA * se
        - J_I * si
        + J_NMDA * nw_e
    )

    for n1 in range(N):
        for n2 in range(N):
            for t in range(T - 1):
                hx_nw[n1, n2, t, sv["r_exc"], sv["se"]] = (
                    logistic_der(IE[n1, t], a_exc, b_exc, d_exc)
                    * J_NMDA
                    * K_gl
                    * cmat[n1, n2]
                )

    return -hx_nw


@numba.njit
def Duh(
    model_params,
    N,
    V_in,
    V_vars,
    T,
    ue,
    ui,
    se,
    si,
    K_gl,
    cmat,
    dmat_ndt,
    se_delay,
    sv,
):
    """Jacobian of systems dynamics wrt. external inputs (control signals).

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param  nw_e:           N x T input of network into each node's 'exc'
    :type  nw_e:            np.ndarray
    :param ue:              N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:               np.ndarray
    :param ui:              N x T array of the total input received by 'inh' population in every node at any time.
    :type ui:               np.ndarray
    :param se:              Value of the se-variable for each node and timepoint
    :type se:               np.ndarray
    :param si:              Value of the si-variable for each node and timepoint
    :type si:               np.ndarray
    :param K_gl:            global coupling strength
    :type K_gl              float
    :param cmat:            coupling matrix
    :type cmat:             np.ndarray
    :param dmat_ndt:        delay index matrix
    :type dmat_ndt:         np.ndarray
    :param se_delay:        N x T array containing values of 'exc' of all nodes through time.
    :type se_delay:         np.ndarray
    :param sv:              dictionary of state vars and respective indices
    :type sv:               dict

    :rtype:     np.ndarray of shape N x V x V x T
    """

    (
        a_exc,
        b_exc,
        d_exc,
        tau_exc,
        gamma_exc,
        w_exc,
        exc_current_baseline,
        a_inh,
        b_inh,
        d_inh,
        tau_inh,
        w_inh,
        inh_current_baseline,
        J_NMDA,
        J_I,
        w_ee,
    ) = model_params

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, se_delay)

    duh = np.zeros((N, V_vars, V_in, T))
    for t in range(T):
        for n in range(N):
            IE = (
                w_exc * (exc_current_baseline + ue[n, t])
                + w_ee * J_NMDA * se[n, t]
                - J_I * si[n, t]
                + J_NMDA * nw_e[n, t]
            )
            duh[n, sv["r_exc"], sv["r_exc"], t] = (
                -logistic_der(IE, a_exc, b_exc, d_exc) * w_exc
            )
            II = (
                w_inh * (inh_current_baseline + ui[n, t]) + J_NMDA * se[n, t] - si[n, t]
            )
            duh[n, sv["r_inh"], sv["r_inh"], t] = (
                -logistic_der(II, a_inh, b_inh, d_inh) * w_inh
            )
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
        for v in range(2, V):
            dxdoth[n, v, v] = 1

    return dxdoth
