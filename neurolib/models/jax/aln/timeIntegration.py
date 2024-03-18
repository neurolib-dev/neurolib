import jax
from jax import jit
import jax.numpy as jnp

from functools import partial

from ....utils import model_utils as mu


def timeIntegration(params):
    """Sets up the parameters for time integration

    Return:
      rates_exc:  N*L array   : containing the exc. neuron rates in kHz time series of the N nodes
      rates_inh:  N*L array   : containing the inh. neuron rates in kHz time series of the N nodes
      t:          L array     : time in ms
      mufe:       N vector    : final value of mufe for each node
      mufi:       N vector    : final value of mufi for each node
      IA:         N vector    : final value of IA   for each node
      seem :      N vector    : final value of seem  for each node
      seim :      N vector    : final value of seim  for each node
      siem :      N vector    : final value of siem  for each node
      siim :      N vector    : final value of siim  for each node
      seev :      N vector    : final value of seev  for each node
      seiv :      N vector    : final value of seiv  for each node
      siev :      N vector    : final value of siev  for each node
      siiv :      N vector    : final value of siiv  for each node

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    if params.distr_delay:
        raise NotImplementedError("The model parameter distr_delay=1 is not supported for jax.aln yet.")

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)

    key, _ = jax.random.split(params["key"])
    params["key"] = key

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matric
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    Cmat = params["Cmat"]
    c_gl = params["c_gl"]  # EPSP amplitude between areas
    Ke_gl = params["Ke_gl"]  # number of incoming E connections (to E population) from each area

    N = len(Cmat)  # Number of areas

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = jnp.ones((N, N)) * params["de"]
    else:
        Dmat = mu.computeDelayMatrix(
            lengthMat, signalV
        )  # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = Dmat.at[jnp.diag_indices(N)].set(params["de"])

    Dmat_ndt = jnp.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------

    # local network (area) parameters [identical for all areas for now]

    ### model parameters
    filter_sigma = params["filter_sigma"]

    # external input parameters:
    tau_ou = params["tau_ou"]  # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    mue_ext_mean = params["mue_ext_mean"]  # Mean external excitatory input (OU process) (mV/ms)
    mui_ext_mean = params["mui_ext_mean"]  # Mean external inhibitory input (OU process) (mV/ms)
    sigmae_ext = params["sigmae_ext"]  # External exc input standard deviation ( mV/sqrt(ms) )
    sigmai_ext = params["sigmai_ext"]  # External inh input standard deviation ( mV/sqrt(ms) )

    # recurrent coupling parameters
    Ke = params["Ke"]  # Recurrent Exc coupling. "EE = IE" assumed for act_dep_coupling in current implementation
    Ki = params["Ki"]  # Recurrent Exc coupling. "EI = II" assumed for act_dep_coupling in current implementation

    # Recurrent connection delays
    de = params["de"]  # Local constant delay "EE = IE" (ms)
    di = params["di"]  # Local constant delay "EI = II" (ms)

    tau_se = params["tau_se"]  # Synaptic decay time constant for exc. connections "EE = IE" (ms)
    tau_si = params["tau_si"]  # Synaptic decay time constant for inh. connections  "EI = II" (ms)
    tau_de = params["tau_de"]
    tau_di = params["tau_di"]

    cee = params["cee"]  # strength of exc. connection
    #  -> determines ePSP magnitude in state-dependent way (in the original model)
    cie = params["cie"]  # strength of inh. connection
    #   -> determines iPSP magnitude in state-dependent way (in the original model)
    cei = params["cei"]
    cii = params["cii"]

    # Recurrent connections coupling strength
    Jee_max = params["Jee_max"]  # ( mV/ms )
    Jei_max = params["Jei_max"]  # ( mV/ms )
    Jie_max = params["Jie_max"]  # ( mV/ms )
    Jii_max = params["Jii_max"]  # ( mV/ms )

    # rescales c's here: multiplication with tau_se makes
    # the increase of s subject to a single input spike invariant to tau_se
    # division by J ensures that mu = J*s will result in a PSP of exactly c
    # for a single spike!

    cee = cee * tau_se / Jee_max  # ms
    cie = cie * tau_se / Jie_max  # ms
    cei = cei * tau_si / abs(Jei_max)  # ms
    cii = cii * tau_si / abs(Jii_max)  # ms
    c_gl = c_gl * tau_se / Jee_max  # ms

    # neuron model parameters
    a = params["a"]  # Adaptation coupling term ( nS )
    b = params["b"]  # Spike triggered adaptation ( pA )
    EA = params["EA"]  # Adaptation reversal potential ( mV )
    tauA = params["tauA"]  # Adaptation time constant ( ms )
    # if params below are changed, preprocessing required
    C = params["C"]  # membrane capacitance ( pF )
    gL = params["gL"]  # Membrane conductance ( nS )
    EL = params["EL"]  # Leak reversal potential ( mV )
    DeltaT = params["DeltaT"]  # Slope factor ( EIF neuron ) ( mV )
    VT = params["VT"]  # Effective threshold (in exp term of the aEIF model)(mV)
    Vr = params["Vr"]  # Membrane potential reset value (mV)
    Vs = params["Vs"]  # Cutoff or spike voltage value, determines the time of spike (mV)
    Tref = params["Tref"]  # Refractory time (ms)
    taum = C / gL  # membrane time constant

    # ------------------------------------------------------------------------

    # Lookup tables for the transfer functions
    precalc_r, precalc_V, precalc_tau_mu, precalc_tau_sigma = (
        params["precalc_r"],
        params["precalc_V"],
        params["precalc_tau_mu"],
        params["precalc_tau_sigma"],
    )

    # parameter for the lookup tables
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = jnp.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = jnp.sqrt(dt)

    ndt_de = jnp.around(de / dt).astype(int)
    ndt_di = jnp.around(di / dt).astype(int)

    rd_exc = jnp.zeros((N, N))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = jnp.zeros(N)

    # Already done above when Dmat_ndt is built
    # for l in range(N):
    #    Dmat_ndt[l, l] = ndt_de  # if no distributed, this is a fixed value (E-E coupling)

    max_global_delay = max(jnp.max(Dmat_ndt), ndt_de, ndt_di)
    startind = int(max_global_delay + 1)

    # ------------------------------------------------------------------------
    # Set initial values
    mufe = params["mufe_init"].copy()  # Filtered mean input (mu) for exc. population
    mufi = params["mufi_init"].copy()  # Filtered mean input (mu) for inh. population
    IA_init = params["IA_init"].copy()  # Adaptation current (pA)
    seem = params["seem_init"].copy()  # Mean exc synaptic input
    seim = params["seim_init"].copy()
    seev = params["seev_init"].copy()  # Exc synaptic input variance
    seiv = params["seiv_init"].copy()
    siim = params["siim_init"].copy()  # Mean inh synaptic input
    siem = params["siem_init"].copy()
    siiv = params["siiv_init"].copy()  # Inh synaptic input variance
    siev = params["siev_init"].copy()

    mue_ou = params["mue_ou"].copy()  # Mean of external exc OU input (mV/ms)
    mui_ou = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)

    # Set the initial firing rates.
    # if initial values are just a Nx1 array
    if params["rates_exc_init"].shape[1] == 1:
        # repeat the 1-dim value stardind times
        rates_exc_init = jnp.dot(params["rates_exc_init"], jnp.ones((1, startind)))  # kHz
        rates_inh_init = jnp.dot(params["rates_inh_init"], jnp.ones((1, startind)))  # kHz
        # set initial adaptation current
        IA_init = jnp.dot(params["IA_init"], jnp.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        rates_exc_init = params["rates_exc_init"][:, -startind:]
        rates_inh_init = params["rates_inh_init"][:, -startind:]
        IA_init = params["IA_init"][:, -startind:]

    # tile external inputs to appropriate shape
    ext_exc_current = mu.adjustArrayShape_jax(params["ext_exc_current"], jnp.zeros((N, startind + len(t))))
    ext_inh_current = mu.adjustArrayShape_jax(params["ext_inh_current"], jnp.zeros((N, startind + len(t))))
    ext_exc_rate = mu.adjustArrayShape_jax(params["ext_exc_rate"], jnp.zeros((N, startind + len(t))))
    ext_inh_rate = mu.adjustArrayShape_jax(params["ext_inh_rate"], jnp.zeros((N, startind + len(t))))

    # ------------------------------------------------------------------------

    return timeIntegration_elementwise(
        dt,
        duration,
        filter_sigma,
        Cmat,
        Dmat,
        c_gl,
        Ke_gl,
        tau_ou,
        sigma_ou,
        mue_ext_mean,
        mui_ext_mean,
        sigmae_ext,
        sigmai_ext,
        Ke,
        Ki,
        de,
        di,
        tau_se,
        tau_si,
        tau_de,
        tau_di,
        cee,
        cie,
        cii,
        cei,
        Jee_max,
        Jei_max,
        Jie_max,
        Jii_max,
        a,
        b,
        EA,
        tauA,
        C,
        gL,
        EL,
        DeltaT,
        VT,
        Vr,
        Vs,
        Tref,
        taum,
        mufe,
        mufi,
        IA_init,
        seem,
        seim,
        seev,
        seiv,
        siim,
        siem,
        siiv,
        siev,
        precalc_r,
        precalc_V,
        precalc_tau_mu,
        precalc_tau_sigma,
        dI,
        ds,
        sigmarange,
        Irange,
        N,
        Dmat_ndt,
        t,
        rates_exc_init,
        rates_inh_init,
        rd_exc,
        rd_inh,
        sqrt_dt,
        startind,
        ndt_de,
        ndt_di,
        mue_ou,
        mui_ou,
        ext_exc_rate,
        ext_inh_rate,
        ext_exc_current,
        ext_inh_current,
        key,
    )


@partial(jit, static_argnames=["N"])
def timeIntegration_elementwise(
    dt,
    duration,
    filter_sigma,
    Cmat,
    Dmat,
    c_gl,
    Ke_gl,
    tau_ou,
    sigma_ou,
    mue_ext_mean,
    mui_ext_mean,
    sigmae_ext,
    sigmai_ext,
    Ke,
    Ki,
    de,
    di,
    tau_se,
    tau_si,
    tau_de,
    tau_di,
    cee,
    cie,
    cii,
    cei,
    Jee_max,
    Jei_max,
    Jie_max,
    Jii_max,
    a,
    b,
    EA,
    tauA,
    C,
    gL,
    EL,
    DeltaT,
    VT,
    Vr,
    Vs,
    Tref,
    taum,
    mufe,
    mufi,
    IA_init,
    seem,
    seim,
    seev,
    seiv,
    siim,
    siem,
    siiv,
    siev,
    precalc_r,
    precalc_V,
    precalc_tau_mu,
    precalc_tau_sigma,
    dI,
    ds,
    sigmarange,
    Irange,
    N,
    Dmat_ndt,
    t,
    rates_exc_init,
    rates_inh_init,
    rd_exc,
    rd_inh,
    sqrt_dt,
    startind,
    ndt_de,
    ndt_di,
    mue_ou,
    mui_ou,
    ext_exc_rate,
    ext_inh_rate,
    ext_exc_current,
    ext_inh_current,
    key,
):
    # squared Jee_max
    sq_Jee_max = Jee_max**2
    sq_Jei_max = Jei_max**2
    sq_Jie_max = Jie_max**2
    sq_Jii_max = Jii_max**2

    key, subkey_exc = jax.random.split(key)
    noise_exc = jax.random.normal(subkey_exc, (N, len(t)))
    key, subkey_inh = jax.random.split(key)
    noise_inh = jax.random.normal(subkey_inh, (N, len(t)))

    range_N = jnp.arange(N)

    ### integrate ODE system:
    def update_step(state, _):
        (
            rates_exc_history,
            rates_inh_history,
            IA_history,
            mufe,
            mufi,
            seem,
            seim,
            siem,
            siim,
            seev,
            seiv,
            siev,
            siiv,
            mue_ou,
            mui_ou,
            # sigmae_f_prev,
            # sigmai_f_prev,
            i,
        ) = state

        rd_exc = rates_exc_history[range_N, -Dmat_ndt - 1] * 1e-3
        rd_inh = rates_inh_history[range_N, -ndt_di - 1] * 1e-3

        mue = Jee_max * seem + Jei_max * seim + mue_ou + ext_exc_current[:, i - 1]
        mui = Jie_max * siem + Jii_max * siim + mui_ou + ext_inh_current[:, i - 1]

        rowsum = jnp.sum(Cmat * rd_exc, axis=1)
        rowsumsq = jnp.sum(Cmat**2 * rd_exc, axis=1)

        z1ee = cee * Ke * rd_exc.diagonal() + c_gl * Ke_gl * rowsum + c_gl * Ke_gl * ext_exc_rate[:, i - 1]
        z1ei = cei * Ki * rd_inh
        z1ie = cie * Ke * rd_exc.diagonal() + c_gl * Ke_gl * ext_inh_rate[:, i - 1]
        z1ii = cii * Ki * rd_inh

        z2ee = cee**2 * Ke * rd_exc.diagonal() + c_gl**2 * Ke_gl * rowsumsq + c_gl**2 * Ke_gl * ext_exc_rate[:, i - 1]
        z2ei = cei**2 * Ki * rd_inh
        z2ie = cie**2 * Ke * rd_exc.diagonal() + c_gl**2 * Ke_gl * ext_inh_rate[:, i - 1]
        z2ii = cii**2 * Ki * rd_inh

        sigmae = jnp.sqrt(
            2 * sq_Jee_max * seev * tau_se * taum / ((1 + z1ee) * taum + tau_se)
            + 2 * sq_Jei_max * seiv * tau_si * taum / ((1 + z1ei) * taum + tau_si)
            + sigmae_ext**2
        )
        sigmai = jnp.sqrt(
            2 * sq_Jie_max * siev * tau_se * taum / ((1 + z1ie) * taum + tau_se)
            + 2 * sq_Jii_max * siiv * tau_si * taum / ((1 + z1ii) * taum + tau_si)
            + sigmai_ext**2
        )
        """
        if filter_sigma:
            sigmae_f = sigmae_f_prev
            sigmai_f = sigmai_f_prev
        else:"""
        sigmae_f = sigmae
        sigmai_f = sigmai

        xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae_f, Irange, dI, mufe - IA_history[:, -2] / C)

        rates_exc_new = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3
        Vmean_exc = interpolate_values(precalc_V, xid1, yid1, dxid, dyid)
        tau_exc = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
        # if filter_sigma:
        #    tau_sigmae_eff = interpolate_values(precalc_tau_sigma, xid1, yid1, dxid, dyid)

        xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f, Irange, dI, mufi)
        rates_inh_new = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3
        tau_inh = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)

        """
        if filter_sigma:
            tau_sigmai_eff = interpolate_values(precalc_tau_sigma, xid1, yid1, dxid, dyid)
            sigmae_f_rhs = (sigmae - sigmae_f) / tau_sigmae_eff
            sigmai_f_rhs = (sigmai - sigmai_f) / tau_sigmai_eff
            sigmae_f_new = sigmae_f + dt * sigmae_f_rhs
            sigmai_f_new = sigmai_f + dt * sigmai_f_rhs
        """

        mufe_rhs = (mue - mufe) / tau_exc
        mufi_rhs = (mui - mufi) / tau_inh

        # rate has to be kHz
        IA_rhs = (a * (Vmean_exc - EA) - IA_history[:, -1] + tauA * b * rates_exc_history[:, -1] * 1e-3) / tauA

        # integration of synaptic input (eq. 4.36)
        seem_rhs = ((1 - seem) * z1ee - seem) / tau_se
        seim_rhs = ((1 - seim) * z1ei - seim) / tau_si
        siem_rhs = ((1 - siem) * z1ie - siem) / tau_se
        siim_rhs = ((1 - siim) * z1ii - siim) / tau_si
        seev_rhs = ((1 - seem) ** 2 * z2ee + (z2ee - 2 * tau_se * (z1ee + 1)) * seev) / tau_se**2
        seiv_rhs = ((1 - seim) ** 2 * z2ei + (z2ei - 2 * tau_si * (z1ei + 1)) * seiv) / tau_si**2
        siev_rhs = ((1 - siem) ** 2 * z2ie + (z2ie - 2 * tau_se * (z1ie + 1)) * siev) / tau_se**2
        siiv_rhs = ((1 - siim) ** 2 * z2ii + (z2ii - 2 * tau_si * (z1ii + 1)) * siiv) / tau_si**2

        mufe_new = mufe + dt * mufe_rhs
        mufi_new = mufi + dt * mufi_rhs
        IA_new = IA_history[:, -1] + dt * IA_rhs

        seem_new = seem + dt * seem_rhs
        seim_new = seim + dt * seim_rhs
        siem_new = siem + dt * siem_rhs
        siim_new = siim + dt * siim_rhs

        seev_update = seev + dt * seev_rhs
        seiv_update = seiv + dt * seiv_rhs
        siev_update = siev + dt * siev_rhs
        siiv_update = siiv + dt * siiv_rhs

        # Ensure the variance does not get negative for low activity
        seev_new = jnp.where(seev_update < 0, 0.0, seev_update)
        seiv_new = jnp.where(seiv_update < 0, 0.0, seiv_update)
        siev_new = jnp.where(siev_update < 0, 0.0, siev_update)
        siiv_new = jnp.where(siiv_update < 0, 0.0, siiv_update)

        # Update Ornstein-Uhlenbeck process for noise
        mue_ou_new = (
            mue_ou + (mue_ext_mean - mue_ou) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[:, i - startind]
        )  # mV/ms
        mui_ou_new = (
            mui_ou + (mui_ext_mean - mui_ou) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[:, i - startind]
        )  # mV/ms

        rates_exc_history_new = jnp.concatenate(
            (rates_exc_history[:, 1:], jnp.expand_dims(rates_exc_new, axis=1)), axis=1
        )
        rates_inh_history_new = jnp.concatenate(
            (rates_inh_history[:, 1:], jnp.expand_dims(rates_inh_new, axis=1)), axis=1
        )
        IA_history_new = jnp.concatenate((IA_history[:, 1:], jnp.expand_dims(IA_new, axis=1)), axis=1)

        return (
            (
                rates_exc_history_new,
                rates_inh_history_new,
                IA_history_new,
                mufe_new,
                mufi_new,
                seem_new,
                seim_new,
                siem_new,
                siim_new,
                seev_new,
                seiv_new,
                siev_new,
                siiv_new,
                mue_ou_new,
                mui_ou_new,
                # sigmae_f_new if filter_sigma else None,
                # sigmai_f_new if filter_sigma else None,
                i + 1,
            ),
            (rates_exc_new, rates_inh_new, IA_new),
        )

    # Iterating through time steps
    (
        rates_exc_history,
        rates_inh_history,
        IA_history,
        mufe,
        mufi,
        seem,
        seim,
        siem,
        siim,
        seev,
        seiv,
        siev,
        siiv,
        mue_ou,
        mui_ou,
        # sigmae_f_prev,
        # sigmai_f_prev,
        i,
    ), (rates_exc_new, rates_inh_new, IA_new) = jax.lax.scan(
        update_step,
        (
            rates_exc_init,
            rates_inh_init,
            IA_init,
            mufe,
            mufi,
            seem,
            seim,
            siem,
            siim,
            seev,
            seiv,
            siev,
            siiv,
            mue_ou,
            mui_ou,
            # sigmae_ext if filter_sigma else None,
            # sigmai_ext if filter_sigma else None,
            startind,
        ),
        xs=None,
        length=len(t),
    )

    return (
        t,
        jnp.concatenate((rates_exc_init, rates_exc_new.T), axis=1),
        jnp.concatenate((rates_inh_init, rates_inh_new.T), axis=1),
        mufe,
        mufi,
        jnp.concatenate((IA_init, IA_new.T), axis=1),
        seem,
        seim,
        siem,
        siim,
        seev,
        seiv,
        siev,
        siiv,
        mue_ou,
        mui_ou,
    )


def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output


def lookup_no_interp(x, dx, xi, y, dy, yi):
    """
    Return the indices for the closest values for a look-up table
    Choose the closest point in the grid

    x     ... range of x values
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0])
               (same for y)

    return:   idxX and idxY
    """

    if xi > x[0] and xi < x[-1]:
        xid = (xi - x[0]) / dx
        xid_floor = jnp.floor(xid)
        if xid - xid_floor < dx / 2:
            idxX = xid_floor
        else:
            idxX = xid_floor + 1
    elif xi < x[0]:
        idxX = 0
    else:
        idxX = len(x) - 1

    if yi > y[0] and yi < y[-1]:
        yid = (yi - y[0]) / dy
        yid_floor = jnp.floor(yid)
        if yid - yid_floor < dy / 2:
            idxY = yid_floor
        else:
            idxY = yid_floor + 1

    elif yi < y[0]:
        idxY = 0
    else:
        idxY = len(y) - 1

    return idxX, idxY


def fast_interp2_opt(x, dx, xi, y, dy, yi):
    """
    Vectorized version of fast_interp2_opt to work with arrays of xi and yi.
    """
    # Initialize outputs with the shape of xi and yi, filled with zeros
    xid1 = jnp.zeros_like(xi)
    dxid = jnp.zeros_like(xi)
    yid1 = jnp.zeros_like(yi)
    dyid = jnp.zeros_like(yi)

    # Calculate indices and distances for all xi and yi within boundaries
    within_x_bounds = (xi >= x[0]) & (xi < x[-1])
    within_y_bounds = (yi >= y[0]) & (yi < y[-1])
    within_bounds = within_x_bounds & within_y_bounds

    xid = (xi - x[0]) / dx
    yid = (yi - y[0]) / dy

    xid1 = jnp.where(within_bounds, jnp.floor(xid), xid1)
    dxid = jnp.where(within_bounds, xid - jnp.floor(xid), dxid)

    yid1 = jnp.where(within_bounds, jnp.floor(yid), yid1)
    dyid = jnp.where(within_bounds, yid - jnp.floor(yid), dyid)

    # Handle outside one boundary conditions for xi
    xi_less_than_x0 = xi < x[0]
    xi_greater_than_xend = xi >= x[-1]

    xid1 = jnp.where(xi_less_than_x0, 0, xid1)
    xid1 = jnp.where(xi_greater_than_xend, -1, xid1)

    dxid = jnp.where(xi_less_than_x0 | xi_greater_than_xend, 0.0, dxid)

    # Handle outside one boundary conditions for yi
    yi_less_than_y0 = yi < y[0]
    yi_greater_than_yend = yi >= y[-1]

    yid1 = jnp.where(yi_less_than_y0, 0, yid1)
    yid1 = jnp.where(yi_greater_than_yend, -1, yid1)

    dyid = jnp.where(yi_less_than_y0 | yi_greater_than_yend, 0.0, dyid)

    return xid1.astype(int), yid1.astype(int), dxid, dyid
