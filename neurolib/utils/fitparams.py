import warnings

import neurolib.models.aln.loadDefaultParams as dp


def loadpoint(params, whichpoint="A1", reload_params=True, newIC=True):
    """
    Loads mean input values for points of interest in Bifurcation plots
    for the mean-field model
    See Paper for locateion in bifurcation diagram
    Parameters A1, A2, A3, ... are for the system without adaptation
    Parameters B1, B2, B3, ... are for the system with adaptation

    whichpoint:     specifies point of interest, pass as string, ex "A2"
    reload_params:  reloads full set of default parameters
    newIC:          reloads new set of random initial conditions
    """
    if reload_params:
        if whichpoint[0] == "A":
            params = load_pars_noadapt(params, reloadDefaults=newIC)
        if whichpoint[0] == "B":
            params = load_pars_adapt(params, reloadDefaults=newIC)

    def pA1(params):  # left of limit cycle
        params["mue_ext_mean"] = 1.2
        params["mui_ext_mean"] = 1.2
        return params

    def pA2(params):  # inside limit cycle
        params["mue_ext_mean"] = 1.3
        params["mui_ext_mean"] = 0.5
        return params

    def pA3(params):  # at bistable region
        params["mue_ext_mean"] = 2.05
        params["mui_ext_mean"] = 1.7
        return params

    ################################
    def pB1(params):  # adaptation: left of e-i lc
        params["mue_ext_mean"] = 1.52
        params["mui_ext_mean"] = 0.84
        return params

    def pB2(params):  # adaptation: inside e-i lc
        params["mue_ext_mean"] = 3.4
        params["mui_ext_mean"] = 1.2
        return params

    def pB3(params):  # adaptation: inside a-E lc
        params["mue_ext_mean"] = 4.0
        params["mui_ext_mean"] = 1.8
        return params

    def pB4(params):  # left of a-E lc
        params["mue_ext_mean"] = 3.8
        params["mui_ext_mean"] = 2.0
        return params

    # execute options
    try:
        params = locals()["p" + whichpoint](params)
    except:
        warnings.warn("loadpoint: No such point {}".format(whichpoint))
    return params


def loadpoint_network(params, whichpoint="A1", reload_params=True):
    """
    Load equivalent points like in loadpoint() but for the adEx network
    See Paper for locateion in bifurcation diagram
    """

    if reload_params:
        if whichpoint[0] == "A":
            params = load_pars_noadapt(params, reloadDefaults=False)
        if whichpoint[0] == "B":
            params = load_pars_adapt(params, reloadDefaults=False)

    def pA1(params):  # left of limit cycle
        params["mue_ext_mean"] = 1.1
        params["mui_ext_mean"] = 0.6
        return params

    def pA2(params):  # inside limit cycle
        params["mue_ext_mean"] = 1.6
        params["mui_ext_mean"] = 0.3
        return params

    def pA3(params):  # at bistable region
        params["mue_ext_mean"] = 2.0
        params["mui_ext_mean"] = 1.2
        return params

    def pB2(params):  # adaptation: inside a-E lc
        params["mue_ext_mean"] = 3.7
        params["mui_ext_mean"] = 0.5
        return params

    def pB3(params):  # adaptation: inside a-E lc
        params["mue_ext_mean"] = 3.8
        params["mui_ext_mean"] = 1.2
        return params

    def pB4(params):  # left of a-E lc
        params["mue_ext_mean"] = 3.4
        params["mui_ext_mean"] = 1.2
        return params

    # execute options
    try:
        params = locals()["p" + whichpoint](params)
    except:
        warnings.warn("loadpoint: No such point {}".format(whichpoint))
    return params


def load_pars_noadapt(params, reloadDefaults=True):

    # load full set of parameters as specified in dp.loadDefaultParams
    if reloadDefaults:
        params = dp.loadDefaultParams(Cmat=[])
    params["dt"] = 0.1
    params["fast_interp"] = 1
    params["global_delay"] = 1
    params["distr_delay"] = 0

    params["c_gl"] = 0.40

    params["ext_exc_rate"] = 0.0
    params["ext_exc_current"] = 0.0

    params["a"] = 0.0
    params["b"] = 0.0

    params["Jee_max"] = 2.43
    params["Jii_max"] = -1.64
    params["Jie_max"] = 2.60
    params["Jei_max"] = -3.3

    params["signalV"] = 20.0
    params["mui_ext_mean"] = 1.40
    params["mue_ext_mean"] = 1.35

    params["sigma_ou"] = 0.046
    params["Ke_gl"] = 252.0

    params["de"] = 4.0
    params["di"] = 2.0

    return params


def load_pars_adapt(params, reloadDefaults=True):
    if reloadDefaults:
        params = dp.loadDefaultParams(Cmat=[])
    params["dt"] = 0.1
    params["fast_interp"] = 1
    params["global_delay"] = 1
    params["distr_delay"] = 0

    params["c_gl"] = 0.40

    params["ext_exc_rate"] = 0.0
    params["ext_exc_current"] = 0.0

    params["a"] = 15.0
    params["b"] = 40.0

    params["Jee_max"] = 2.43
    params["Jii_max"] = -1.64
    params["Jie_max"] = 2.60
    params["Jei_max"] = -3.3

    params["signalV"] = 20.0
    params["mue_ext_mean"] = 1.520
    params["mui_ext_mean"] = 0.840

    params["Ke_gl"] = 325.700
    params["sigma_ou"] = 0.120

    params["de"] = 4.0
    params["di"] = 2.0

    return params


def newIC(params):
    N = params["Cmat"].shape[0]
    # Randomly create the initial parameters
    (
        mufe_init,
        IA_init,
        mufi_init,
        seem_init,
        seim_init,
        seev_init,
        seiv_init,
        siim_init,
        siem_init,
        siiv_init,
        siev_init,
        rates_exc_init,
        rates_inh_init,
    ) = dp.generateRandomICs(N)

    params["mufe_init"] = mufe_init
    params["IA_init"] = IA_init
    params["mufi_init"] = mufi_init
    params["seem_init"] = seem_init
    params["seim_init"] = seim_init
    params["seev_init"] = seev_init
    params["seiv_init"] = seiv_init
    params["siim_init"] = siim_init
    params["siem_init"] = siem_init
    params["siiv_init"] = siiv_init
    params["siev_init"] = siev_init
    params["rates_exc_init"] = rates_exc_init
    params["rates_inh_init"] = rates_inh_init

    return params


def zeroIC(params):
    N = params["Cmat"].shape[0]
    # Randomly create the initial parameters
    (
        mufe_init,
        IA_init,
        mufi_init,
        seem_init,
        seim_init,
        seev_init,
        seiv_init,
        siim_init,
        siem_init,
        siiv_init,
        siev_init,
        rates_exc_init,
        rates_inh_init,
    ) = dp.generateRandomICs(N)

    params["mufe_init"] = mufe_init * 0.0
    params["IA_init"] = IA_init * 0.0
    params["mufi_init"] = mufi_init * 0.0
    params["seem_init"] = seem_init * 0.0
    params["seim_init"] = seim_init * 0.0
    params["seev_init"] = seev_init * 0.0
    params["seiv_init"] = seiv_init * 0.0
    params["siim_init"] = siim_init * 0.0
    params["siem_init"] = siem_init * 0.0
    params["siiv_init"] = siiv_init * 0.0
    params["siev_init"] = siev_init * 0.0
    params["rates_exc_init"] = rates_exc_init * 0.0
    params["rates_inh_init"] = rates_inh_init * 0.0

    return params
