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
    # See Papadopoulos et al., Relations between large-scale brain connectivity and effects of regional stimulation
    # depend on collective dynamical state, arXiv, 2020
    tau_exc = params["tau_exc"]  #
    tau_inh = params["tau_inh"]  #
    c_excexc = params["c_excexc"]  #
    c_excinh = params["c_excinh"]  #
    c_inhexc = params["c_inhexc"]  #
    c_inhinh = params["c_inhinh"]  #
    a_exc = params["a_exc"]  #
    a_inh = params["a_inh"]  #
    mu_exc = params["mu_exc"]  #
    mu_inh = params["mu_inh"]  #

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
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
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
    excs = np.zeros((N, startind + len(t)))
    inhs = np.zeros((N, startind + len(t)))

    exc_ext_baseline = params["exc_ext_baseline"]
    inh_ext_baseline = params["inh_ext_baseline"]

    exc_ext = mu.adjustArrayShape(params["exc_ext"], excs)
    inh_ext = mu.adjustArrayShape(params["inh_ext"], inhs)

    # ------------------------------------------------------------------------
    # Set initial values
    # if initial values are just a Nx1 array
    if np.shape(params["exc_init"])[1] == 1:
        exc_init = np.dot(params["exc_init"], np.ones((1, startind)))
        inh_init = np.dot(params["inh_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        exc_init = params["exc_init"][:, -startind:]
        inh_init = params["inh_init"][:, -startind:]

    # xsd = np.zeros((N,N))  # delayed activity
    exc_input_d = np.zeros(N)  # delayed input to exc
    inh_input_d = np.zeros(N)  # delayed input to inh (note used)

    np.random.seed(RNGseed)

    # Save the noise in the activity array to save memory
    excs[:, startind:] = np.random.standard_normal((N, len(t)))
    inhs[:, startind:] = np.random.standard_normal((N, len(t)))

    excs[:, :startind] = exc_init
    inhs[:, :startind] = inh_init

    noise_exc = np.zeros((N,))
    noise_inh = np.zeros((N,))

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
        inhs,
        exc_input_d,
        inh_input_d,
        exc_ext_baseline,
        inh_ext_baseline,
        exc_ext,
        inh_ext,
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_excinh,
        c_inhexc,
        c_inhinh,
        noise_exc,
        noise_inh,
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
    N,
    Cmat,
    K_gl,
    Dmat_ndt,
    excs,
    inhs,
    exc_input_d,
    inh_input_d,
    exc_ext_baseline,
    inh_ext_baseline,
    exc_ext,
    inh_ext,
    tau_exc,
    tau_inh,
    a_exc,
    a_inh,
    mu_exc,
    mu_inh,
    c_excexc,
    c_excinh,
    c_inhexc,
    c_inhinh,
    noise_exc,
    noise_inh,
    exc_ou,
    inh_ou,
    exc_ou_mean,
    inh_ou_mean,
    tau_ou,
    sigma_ou,
):
    ### integrate ODE system:

    def S_E(x):
        return 1.0 / (1.0 + np.exp(-a_exc * (x - mu_exc)))

    def S_I(x):
        return 1.0 / (1.0 + np.exp(-a_inh * (x - mu_inh)))

    for i in range(startind, startind + len(t)):
        # loop through all the nodes
        for no in range(N):
            # To save memory, noise is saved in the activity array
            noise_exc[no] = excs[no, i]
            noise_inh[no] = inhs[no, i]

            # delayed input to each node
            exc_input_d[no] = 0

            for l in range(N):
                exc_input_d[no] += K_gl * Cmat[no, l] * (excs[l, i - Dmat_ndt[no, l] - 1])

            # Wilson-Cowan model
            exc_rhs = (
                1
                / tau_exc
                * (
                    -excs[no, i - 1]
                    + (1 - excs[no, i - 1])
                    * S_E(
                        c_excexc * excs[no, i - 1]  # input from within the excitatory population
                        - c_inhexc * inhs[no, i - 1]  # input from the inhibitory population
                        + exc_input_d[no]  # input from other nodes
                        + exc_ext_baseline  # baseline external input (static)
                        + exc_ext[no, i - 1]  # time-dependent external input
                    )
                    + exc_ou[no]  # ou noise
                )
            )
            inh_rhs = (
                1
                / tau_inh
                * (
                    -inhs[no, i - 1]
                    + (1 - inhs[no, i - 1])
                    * S_I(
                        c_excinh * excs[no, i - 1]  # input from the excitatory population
                        - c_inhinh * inhs[no, i - 1]  # input from within the inhibitory population
                        + inh_ext_baseline  # baseline external input (static)
                        + inh_ext[no, i - 1]  # time-dependent external input
                    )
                    + inh_ou[no]  # ou noise
                )
            )

            # Euler integration
            excs[no, i] = excs[no, i - 1] + dt * exc_rhs
            inhs[no, i] = inhs[no, i - 1] + dt * inh_rhs

            # make sure e and i variables do not exceed 1 (can only happen with noise)
            if excs[no, i] > 1.0:
                excs[no, i] = 1.0
            if excs[no, i] < 0.0:
                excs[no, i] = 0.0

            if inhs[no, i] > 1.0:
                inhs[no, i] = 1.0
            if inhs[no, i] < 0.0:
                inhs[no, i] = 0.0

            # Ornstein-Uhlenbeck process
            exc_ou[no] = (
                exc_ou[no] + (exc_ou_mean - exc_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]
            )  # mV/ms
            inh_ou[no] = (
                inh_ou[no] + (inh_ou_mean - inh_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[no]
            )  # mV/ms

    return t, excs, inhs, exc_ou, inh_ou


@numba.njit
def logistic(x, a, mu):
    """Logistic function evaluated at point 'x'.

    :type x:    float
    :param a:   Slope parameter.
    :type a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    return 1.0 / (1.0 + np.exp(-a * (x - mu)))


@numba.njit
def logistic_der(x, a, mu):
    """Derivative of logistic function, evaluated at point 'x'.

    :type x:    float
    :param a:   Slope parameter.
    :typa a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    return (a * np.exp(-a * (x - mu))) / (1.0 + np.exp(-a * (x - mu))) ** 2


@numba.njit
def jacobian_wc(
    model_params,
    nw_e,
    e,
    i,
    ue,
    ui,
    V,
    sv,
):
    """Jacobian of the WC dynamical system.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param  nw_e:   N x T input of network into each node's 'exc'
    :type  nw_e:    np.ndarray
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T combined input of 'background' and 'control' into 'exc'.
    :type ue:       np.ndarray
    :param ui:      N x T combined input of 'background' and 'control' into 'inh'.
    :type ui:       np.ndarray
    :param V:       Number of system variables.
    :type V:        int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:        4 x 4 Jacobian matrix.
    :rtype:         np.ndarray
    """
    (
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_inhexc,
        c_excinh,
        c_inhinh,
        exc_ext_baseline,
        inh_ext_baseline,
    ) = model_params

    jacobian = np.zeros((V, V))
    input_exc = c_excexc * e - c_inhexc * i + nw_e + exc_ext_baseline + ue
    jacobian[sv["exc"], sv["exc"]] = (
        -(-1.0 - logistic(input_exc, a_exc, mu_exc) + (1.0 - e) * c_excexc * logistic_der(input_exc, a_exc, mu_exc))
        / tau_exc
    )
    jacobian[sv["exc"], sv["inh"]] = -((1.0 - e) * (-c_inhexc) * logistic_der(input_exc, a_exc, mu_exc)) / tau_exc
    input_inh = c_excinh * e - c_inhinh * i + inh_ext_baseline + ui
    jacobian[sv["inh"], sv["exc"]] = -((1.0 - i) * c_excinh * logistic_der(input_inh, a_inh, mu_inh)) / tau_inh
    jacobian[sv["inh"], sv["inh"]] = (
        -(-1.0 - logistic(input_inh, a_inh, mu_inh) + (1.0 - i) * (-c_inhinh) * logistic_der(input_inh, a_inh, mu_inh))
        / tau_inh
    )
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
    """Jacobians of WCModel wrt. the 'e'- and 'i'-variable for each time step.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dyn_vars:    N x V x T array containing all values of 'exc' and 'inh'.
    :type dyn_vars:     np.ndarray
    :param dyn_vars_delay:
    :type dyn_vars_delay:     np.ndarray
    :param control:     N x 2 x T control inputs to 'exc' and 'inh'.
    :type control:      np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:            N x T x 4 x 4 Jacobians.
    :rtype:             np.ndarray
    """
    hx = np.zeros((N, T, V, V))
    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, dyn_vars_delay[:, sv["exc"], :])

    for n in range(N):
        for t in range(T):
            e = dyn_vars[n, sv["exc"], t]
            i = dyn_vars[n, sv["inh"], t]
            ue = control[n, sv["exc"], t]
            ui = control[n, sv["inh"], t]
            hx[n, t, :, :] = jacobian_wc(
                wc_model_params,
                nw_e[n, t],
                e,
                i,
                ue,
                ui,
                V,
                sv,
            )
    return hx


@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, exc_values):
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
    :param exc_values:  N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:   np.ndarray
    :return:            N x T network inputs.
    :rytpe:             np.ndarray
    """
    nw_input = np.zeros((N, T))

    for t in range(1, T):
        for n in range(N):
            for l in range(N):
                nw_input[n, t] += K_gl * cmat[n, l] * (exc_values[l, t - dmat_ndt[n, l] - 1])
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
    e,
    i,
    e_delay,
    ue,
    sv,
):
    """Jacobians for network connectivity in all time steps.

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:       np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:         Jacobians for network connectivity in all time steps.
    :rtype:          np.ndarray of shape N x N x T x 4 x 4
    """
    (
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_inhexc,
        c_excinh,
        c_inhinh,
        exc_ext_baseline,
        inh_ext_baseline,
    ) = model_params
    hx_nw = np.zeros((N, N, T, V, V))

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, e_delay)
    exc_input = c_excexc * e - c_inhexc * i + nw_e + exc_ext_baseline + ue

    for n1 in range(N):
        for n2 in range(N):
            for t in range(T - 1):
                hx_nw[n1, n2, t, sv["exc"], sv["exc"]] = (
                    logistic_der(exc_input[n1, t], a_exc, mu_exc) * K_gl * cmat[n1, n2]
                ) / tau_exc

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
    e,
    i,
    K_gl,
    cmat,
    dmat_ndt,
    exc_values,
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
    :param e:               Value of the E-variable for each node and timepoint
    :type e:                np.ndarray
    :param i:               Value of the I-variable for each node and timepoint
    :type i:                np.ndarray
    :param K_gl:            global coupling strength
    :type K_gl              float
    :param cmat:            coupling matrix
    :type cmat:             np.ndarray
    :param dmat_ndt:        delay index matrix
    :type dmat_ndt:         np.ndarray
    :param exc_values:      N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:       np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :rtype:     np.ndarray of shape N x V x V x T
    """

    (
        tau_exc,
        tau_inh,
        a_exc,
        a_inh,
        mu_exc,
        mu_inh,
        c_excexc,
        c_inhexc,
        c_excinh,
        c_inhinh,
        exc_ext_baseline,
        inh_ext_baseline,
    ) = model_params

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, exc_values)

    duh = np.zeros((N, V_vars, V_in, T))
    for t in range(T):
        for n in range(N):
            input_exc = c_excexc * e[n, t] - c_inhexc * i[n, t] + nw_e[n, t] + exc_ext_baseline + ue[n, t]
            duh[n, sv["exc"], sv["exc"], t] = -(1.0 - e[n, t]) * logistic_der(input_exc, a_exc, mu_exc) / tau_exc
            input_inh = c_excinh * e[n, t] - c_inhinh * i[n, t] + inh_ext_baseline + ui[n, t]
            duh[n, sv["inh"], sv["inh"], t] = -(1.0 - i[n, t]) * logistic_der(input_inh, a_inh, mu_inh) / tau_inh
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
