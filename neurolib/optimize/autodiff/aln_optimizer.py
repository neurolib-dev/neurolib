from jax import jit
from neurolib.models.jax.aln.timeIntegration import timeIntegration_args, timeIntegration_elementwise

args_names = [
    "dt",
    "duration",
    "filter_sigma",
    "Cmat",
    "Dmat",
    "c_gl",
    "Ke_gl",
    "tau_ou",
    "sigma_ou",
    "mue_ext_mean",
    "mui_ext_mean",
    "sigmae_ext",
    "sigmai_ext",
    "Ke",
    "Ki",
    "de",
    "di",
    "tau_se",
    "tau_si",
    "tau_de",
    "tau_di",
    "cee",
    "cie",
    "cii",
    "cei",
    "Jee_max",
    "Jei_max",
    "Jie_max",
    "Jii_max",
    "a",
    "b",
    "EA",
    "tauA",
    "C",
    "gL",
    "EL",
    "DeltaT",
    "VT",
    "Vr",
    "Vs",
    "Tref",
    "taum",
    "mufe",
    "mufi",
    "IA_init",
    "seem",
    "seim",
    "seev",
    "seiv",
    "siim",
    "siem",
    "siiv",
    "siev",
    "precalc_r",
    "precalc_V",
    "precalc_tau_mu",
    "precalc_tau_sigma",
    "dI",
    "ds",
    "sigmarange",
    "Irange",
    "N",
    "Dmat_ndt",
    "t",
    "rates_exc_init",
    "rates_inh_init",
    "rd_exc",
    "rd_inh",
    "sqrt_dt",
    "startind",
    "ndt_de",
    "ndt_di",
    "mue_ou",
    "mui_ou",
    "ext_exc_rate",
    "ext_inh_rate",
    "ext_exc_current",
    "ext_inh_current",
    "key",
]


def get_loss(model_params, loss_f, opt_params):
    args_values = timeIntegration_args(model_params)
    args = dict(zip(args_names, args_values))

    @jit
    def loss(x):
        args_local = args.copy()
        args_local.update(dict(zip(opt_params, x)))
        simulation_outputs = timeIntegration_elementwise(**args_local)
        return loss_f(*simulation_outputs)

    return loss
