import numba
import numpy as np


def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    sqrt_dt = np.sqrt(dt)
    duration = params["duration"]  # Simulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    startind = 1  # int(max_global_delay + 1)
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    # parameters
    tau = params["tau"]
    Q_max = params["Q_max"]
    C1 = params["C1"]
    theta = params["theta"]
    sigma = params["sigma"]
    g_L = params["g_L"]
    E_L = params["E_L"]
    g_AMPA = params["g_AMPA"]
    g_GABA = params["g_GABA"]
    E_AMPA = params["E_AMPA"]
    E_GABA = params["E_GABA"]
    g_LK = params["g_LK"]
    E_K = params["E_K"]
    g_T_t = params["g_T_t"]
    g_T_r = params["g_T_r"]
    E_Ca = params["E_Ca"]
    g_h = params["g_h"]
    g_inc = params["g_inc"]
    E_h = params["E_h"]
    C_m = params["C_m"]
    alpha_Ca = params["alpha_Ca"]
    Ca_0 = params["Ca_0"]
    tau_Ca = params["tau_Ca"]
    k1 = params["k1"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    n_P = params["n_P"]
    gamma_e = params["gamma_e"]
    gamma_r = params["gamma_r"]
    d_phi = params["d_phi"]
    N_rt = params["N_rt"]
    N_tr = params["N_tr"]
    N_rr = params["N_rr"]

    ext_current_t = params["ext_current_t"]
    ext_current_r = params["ext_current_r"]

    # model output
    V_t = np.zeros((1, startind + len(t)))
    V_r = np.zeros((1, startind + len(t)))
    Q_t = np.zeros((1, startind + len(t)))
    Q_r = np.zeros((1, startind + len(t)))
    # init
    V_t[:, :startind] = params["V_t_init"]
    V_r[:, :startind] = params["V_r_init"]
    Ca = float(params["Ca_init"])
    h_T_t = float(params["h_T_t_init"])
    h_T_r = float(params["h_T_r_init"])
    m_h1 = float(params["m_h1_init"])
    m_h2 = float(params["m_h2_init"])
    s_et = float(params["s_et_init"])
    s_gt = float(params["s_gt_init"])
    s_er = float(params["s_er_init"])
    s_gr = float(params["s_gr_init"])
    ds_et = float(params["ds_et_init"])
    ds_gt = float(params["ds_gt_init"])
    ds_er = float(params["ds_er_init"])
    ds_gr = float(params["ds_gr_init"])

    np.random.seed(RNGseed)
    noise = np.random.standard_normal((len(t)))

    (
        t,
        V_t,
        V_r,
        Q_t,
        Q_r,
        Ca,
        h_T_t,
        h_T_r,
        m_h1,
        m_h2,
        s_et,
        s_gt,
        s_er,
        s_gr,
        ds_et,
        ds_gt,
        ds_er,
        ds_gr,
    ) = timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        Q_max,
        C1,
        theta,
        sigma,
        g_L,
        E_L,
        g_AMPA,
        g_GABA,
        E_AMPA,
        E_GABA,
        g_LK,
        E_K,
        g_T_t,
        g_T_r,
        E_Ca,
        g_h,
        g_inc,
        E_h,
        C_m,
        tau,
        alpha_Ca,
        Ca_0,
        tau_Ca,
        k1,
        k2,
        k3,
        k4,
        n_P,
        gamma_e,
        gamma_r,
        d_phi,
        noise,
        ext_current_t,
        ext_current_r,
        N_rt,
        N_tr,
        N_rr,
        V_t,
        V_r,
        Q_t,
        Q_r,
        Ca,
        h_T_t,
        h_T_r,
        m_h1,
        m_h2,
        s_et,
        s_gt,
        s_er,
        s_gr,
        ds_et,
        ds_gt,
        ds_er,
        ds_gr,
    )
    return (
        t,
        V_t,
        V_r,
        Q_t,
        Q_r,
        np.array(Ca),
        np.array(h_T_t),
        np.array(h_T_r),
        np.array(m_h1),
        np.array(m_h2),
        np.array(s_et),
        np.array(s_gt),
        np.array(s_er),
        np.array(s_gr),
        np.array(ds_et),
        np.array(ds_gt),
        np.array(ds_er),
        np.array(ds_gr),
    )


@numba.njit()
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    Q_max,
    C1,
    theta,
    sigma,
    g_L,
    E_L,
    g_AMPA,
    g_GABA,
    E_AMPA,
    E_GABA,
    g_LK,
    E_K,
    g_T_t,
    g_T_r,
    E_Ca,
    g_h,
    g_inc,
    E_h,
    C_m,
    tau,
    alpha_Ca,
    Ca_0,
    tau_Ca,
    k1,
    k2,
    k3,
    k4,
    n_P,
    gamma_e,
    gamma_r,
    d_phi,
    noise,
    ext_current_t,
    ext_current_r,
    N_rt,
    N_tr,
    N_rr,
    V_t,
    V_r,
    Q_t,
    Q_r,
    Ca,
    h_T_t,
    h_T_r,
    m_h1,
    m_h2,
    s_et,
    s_gt,
    s_er,
    s_gr,
    ds_et,
    ds_gt,
    ds_er,
    ds_gr,
):
    def _firing_rate(voltage):
        return Q_max / (1.0 + np.exp(-C1 * (voltage - theta) / sigma))

    def _leak_current(voltage):
        return g_L * (voltage - E_L)

    def _potassium_leak_current(voltage):
        return g_LK * (voltage - E_K)

    def _syn_exc_current(voltage, synaptic_rate):
        return g_AMPA * synaptic_rate * (voltage - E_AMPA)

    def _syn_inh_current(voltage, synaptic_rate):
        return g_GABA * synaptic_rate * (voltage - E_GABA)

    for i in range(startind, startind + len(t)):
        # leak current
        I_leak_t = _leak_current(V_t[0, i - 1])
        I_leak_r = _leak_current(V_r[0, i - 1])

        # synaptic currents
        I_et = _syn_exc_current(V_t[0, i - 1], s_et)
        I_gt = _syn_inh_current(V_t[0, i - 1], s_gt)
        I_er = _syn_exc_current(V_r[0, i - 1], s_er)
        I_gr = _syn_inh_current(V_r[0, i - 1], s_gr)

        # potassium leak current
        I_LK_t = _potassium_leak_current(V_t[0, i - 1])
        I_LK_r = _potassium_leak_current(V_r[0, i - 1])

        # T-type Ca current
        m_inf_T_t = 1.0 / (1.0 + np.exp(-(V_t[0, i - 1] + 59.0) / 6.2))
        m_inf_T_r = 1.0 / (1.0 + np.exp(-(V_r[0, i - 1] + 52.0) / 7.4))
        I_T_t = g_T_t * m_inf_T_t * m_inf_T_t * h_T_t * (V_t[0, i - 1] - E_Ca)
        I_T_r = g_T_r * m_inf_T_r * m_inf_T_r * h_T_r * (V_r[0, i - 1] - E_Ca)

        # h-type current
        I_h = g_h * (m_h1 + g_inc * m_h2) * (V_t[0, i - 1] - E_h)

        ### define derivatives
        # membrane potential
        d_V_t = -(I_leak_t + I_et + I_gt + ext_current_t) / tau - (1.0 / C_m) * (I_LK_t + I_T_t + I_h)
        d_V_r = -(I_leak_r + I_er + I_gr + ext_current_r) / tau - (1.0 / C_m) * (I_LK_r + I_T_r)
        # Calcium concentration
        d_Ca = alpha_Ca * I_T_t - (Ca - Ca_0) / tau_Ca
        # channel dynamics
        h_inf_T_t = 1.0 / (1.0 + np.exp((V_t[0, i - 1] + 81.0) / 4.0))
        h_inf_T_r = 1.0 / (1.0 + np.exp((V_r[0, i - 1] + 80.0) / 5.0))
        tau_h_T_t = (
            30.8 + (211.4 + np.exp((V_t[0, i - 1] + 115.2) / 5.0)) / (1.0 + np.exp((V_t[0, i - 1] + 86.0) / 3.2))
        ) / 3.7371928
        tau_h_T_r = (
            85.0 + 1.0 / (np.exp((V_r[0, i - 1] + 48.0) / 4.0) + np.exp(-(V_r[0, i - 1] + 407.0) / 50.0))
        ) / 3.7371928
        d_h_T_t = (h_inf_T_t - h_T_t) / tau_h_T_t
        d_h_T_r = (h_inf_T_r - h_T_r) / tau_h_T_r
        m_inf_h = 1.0 / (1.0 + np.exp((V_t[0, i - 1] + 75.0) / 5.5))
        tau_m_h = 20.0 + 1000.0 / (np.exp((V_t[0, i - 1] + 71.5) / 14.2) + np.exp(-(V_t[0, i - 1] + 89.0) / 11.6))
        # Calcium channel dynamics
        P_h = k1 * Ca ** n_P / (k1 * Ca ** n_P + k2)
        d_m_h1 = (m_inf_h * (1.0 - m_h2) - m_h1) / tau_m_h - k3 * P_h * m_h1 + k4 * m_h2
        d_m_h2 = k3 * P_h * m_h1 - k4 * m_h2
        # synaptic dynamics
        d_s_et = ds_et
        d_s_er = ds_er
        d_s_gt = ds_gt
        d_s_gr = ds_gr
        d_ds_et = 0.0
        # d_ds_et = gamma_e ** 2 * (N_tp * cortical_rowsum - s_et) - 2 * gamma_e * ds_et
        d_ds_er = gamma_e ** 2 * (N_rt * _firing_rate(V_t[0, i - 1]) - s_er) - 2 * gamma_e * ds_er
        # d_ds_er = gamma_e ** 2 * (N_rt * _firing_rate(V_t[0, i - 1]) + N_rp * cortical_rowsum - s_er) - 2 * gamma_e * ds_er
        d_ds_gt = gamma_r ** 2 * (N_tr * _firing_rate(V_r[0, i - 1]) - s_gt) - 2 * gamma_r * ds_gt
        d_ds_gr = gamma_r ** 2 * (N_rr * _firing_rate(V_r[0, i - 1]) - s_gr) - 2 * gamma_r * ds_gr

        ### Euler integration
        V_t[0, i] = V_t[0, i - 1] + dt * d_V_t
        V_r[0, i] = V_r[0, i - 1] + dt * d_V_r
        Q_t[0, i] = _firing_rate(V_t[0, i]) * 1e3  # convert kHz to Hz
        Q_r[0, i] = _firing_rate(V_r[0, i]) * 1e3  # convert kHz to Hz
        Ca = Ca + dt * d_Ca
        h_T_t = h_T_t + dt * d_h_T_t
        h_T_r = h_T_r + dt * d_h_T_r
        m_h1 = m_h1 + dt * d_m_h1
        m_h2 = m_h2 + dt * d_m_h2
        s_et = s_et + dt * d_s_et
        s_gt = s_gt + dt * d_s_gt
        s_er = s_er + dt * d_s_er
        s_gr = s_gr + dt * d_s_gr
        # noisy variable
        ds_et = ds_et + dt * d_ds_et + gamma_e ** 2 * d_phi * sqrt_dt * noise[i - startind]
        ds_gt = ds_gt + dt * d_ds_gt
        ds_er = ds_er + dt * d_ds_er
        ds_gr = ds_gr + dt * d_ds_gr

    return t, V_t, V_r, Q_t, Q_r, Ca, h_T_t, h_T_r, m_h1, m_h2, s_et, s_gt, s_er, s_gr, ds_et, ds_gt, ds_er, ds_gr
