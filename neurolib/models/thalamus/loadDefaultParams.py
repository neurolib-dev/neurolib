import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(seed=None):
    """
    Load default parameters for the thalamic mass model due to Costa et al.

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    ### runtime parameters
    # thalamus is really sensitive, so either you integrate with very small dt or use an adaptive integration step
    params.dt = 0.01  # ms
    params.duration = 60000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    # set seed to 0, pypet will complain otherwise
    params.seed = seed or 0.0

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
    params.V_t_init = -68.0
    params.V_r_init = -68.0
    params.Ca_init = 2.4e-4
    params.h_T_t_init = 0.0
    params.h_T_r_init = 0.0
    params.m_h1_init = 0.0
    params.m_h2_init = 0.0
    params.s_et_init = 0.0
    params.s_gt_init = 0.0
    params.s_er_init = 0.0
    params.s_gr_init = 0.0
    params.ds_et_init = 0.0
    params.ds_gt_init = 0.0
    params.ds_er_init = 0.0
    params.ds_gr_init = 0.0

    # always 1 node only - no network of multiple "thalamuses"
    params.N = 1
    params.Cmat = np.zeros((1, 1))
    params.lengthMat = np.zeros((1, 1))

    return params
