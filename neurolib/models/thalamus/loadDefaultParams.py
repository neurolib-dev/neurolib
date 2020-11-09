import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(seed=None):
    """
    Load default parameters for the thalamic mass model due to Costa et al.
    Subscript t (_t) referes to thalamocortical relay population (TCR), while
    sucscript r (_r) referes to the thalamic reticular nuclei (TRN).

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    ### runtime parameters
    # thalamus is really sensitive, so either you integrate with very small dt or use an adaptive integration step
    params.dt = 0.01  # ms
    params.duration = 60000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    params.seed = seed

    # local parameters for both populations
    params.tau = 20.0
    params.Q_max = 400.0e-3  # 1/ms
    params.theta = -58.5  # mV
    params.sigma = 6.0
    params.C1 = 1.8137993642
    params.C_m = 1.0  # muF/cm^2
    params.gamma_e = 70.0e-3  # 1/ms
    params.gamma_r = 100.0e-3  # 1/ms
    params.g_L = 1.0  # AU
    params.g_GABA = 1.0  # ms
    params.g_AMPA = 1.0  # ms
    params.g_LK = 0.018  # mS/cm^2
    params.E_AMPA = 0.0  # mV
    params.E_GABA = -70.0  # mV
    params.E_L = -70.0  # mV
    params.E_K = -100.0  # mV
    params.E_Ca = 120.0  # mV

    # specific thalamo-cortical neurons population - TCR (excitatory)
    params.g_T_t = 3.0  # mS/cm^2
    params.g_h = 0.062  # mS/cm^2
    params.E_h = -40.0  # mV
    params.alpha_Ca = -51.8e-6  # nmol
    params.tau_Ca = 10.0  # ms
    params.Ca_0 = 2.4e-4
    params.k1 = 2.5e7
    params.k2 = 4.0e-4
    params.k3 = 1.0e-1
    params.k4 = 1.0e-3
    params.n_P = 4.0
    params.g_inc = 2.0
    # connectivity
    params.N_tr = 5.0
    # noise
    params.d_phi = 0.0

    # specific thalamic reticular nuclei population - TRN (inhibitory)
    params.g_T_r = 2.3  # mS/cm^2
    # connectivity
    params.N_rt = 3.0
    params.N_rr = 25.0

    # external input
    params.ext_current_t = 0.0
    params.ext_current_r = 0.0

    # init
    (
        params.V_t_init,
        params.V_r_init,
        params.Q_t_init,
        params.Q_r_init,
        params.Ca_init,
        params.h_T_t_init,
        params.h_T_r_init,
        params.m_h1_init,
        params.m_h2_init,
        params.s_et_init,
        params.s_gt_init,
        params.s_er_init,
        params.s_gr_init,
        params.ds_et_init,
        params.ds_gt_init,
        params.ds_er_init,
        params.ds_gr_init,
    ) = generateRandomICs(seed=seed)

    # always 1 node only - no network of multiple "thalamuses"
    params.N = 1
    params.Cmat = np.zeros((1, 1))
    params.lengthMat = np.zeros((1, 1))

    return params


def generateRandomICs(seed=None):
    """Generates random Initial Conditions for the interareal network

    :returns:   A tuple of 15 floats for representing initial state of the
                thalamus
    """
    np.random.seed(seed)

    V_t_init = np.random.uniform(-75, -50, (1,))
    V_r_init = np.random.uniform(-75, -50, (1,))
    Q_t_init = np.random.uniform(0.0, 200.0, (1,))
    Q_r_init = np.random.uniform(0.0, 200.0, (1,))
    Ca_init = 2.4e-4
    h_T_t_init = 0.0
    h_T_r_init = 0.0
    m_h1_init = 0.0
    m_h2_init = 0.0
    s_et_init = 0.0
    s_gt_init = 0.0
    s_er_init = 0.0
    s_gr_init = 0.0
    ds_et_init = 0.0
    ds_gt_init = 0.0
    ds_er_init = 0.0
    ds_gr_init = 0.0

    return (
        V_t_init,
        V_r_init,
        Q_t_init,
        Q_r_init,
        np.array(Ca_init),
        np.array(h_T_t_init),
        np.array(h_T_r_init),
        np.array(m_h1_init),
        np.array(m_h2_init),
        np.array(s_et_init),
        np.array(s_gt_init),
        np.array(s_er_init),
        np.array(s_gr_init),
        np.array(ds_et_init),
        np.array(ds_gt_init),
        np.array(ds_er_init),
        np.array(ds_gr_init),
    )
