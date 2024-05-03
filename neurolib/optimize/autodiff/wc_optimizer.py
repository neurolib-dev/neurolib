from jax import jit
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise

args_names = [
    "startind",
    "t",
    "dt",
    "sqrt_dt",
    "N",
    "Cmat",
    "K_gl",
    "Dmat_ndt",
    "exc_init",
    "inh_init",
    "exc_ext_baseline",
    "inh_ext_baseline",
    "exc_ext",
    "inh_ext",
    "tau_exc",
    "tau_inh",
    "a_exc",
    "a_inh",
    "mu_exc",
    "mu_inh",
    "c_excexc",
    "c_excinh",
    "c_inhexc",
    "c_inhinh",
    "exc_ou_init",
    "inh_ou_init",
    "exc_ou_mean",
    "inh_ou_mean",
    "tau_ou",
    "sigma_ou",
    "key",
]


# example usage:
# model = WCModel()
# wc_loss = get_loss(model.params, loss_f, ['exc_ext'])
# grad_wc_loss = jax.jit(jax.grad(wc_loss))
# grad_wc_loss([exc_ext])
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
