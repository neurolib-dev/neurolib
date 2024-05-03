import jax
from jax import jit
import jax.numpy as jnp
from jax import random

from functools import partial

from ....utils import model_utils as mu


def timeIntegration(params):
    return timeIntegration_elementwise(*timeIntegration_args(params))


def timeIntegration_args(params):
    """Sets up the parameters for time integration in a JAX-compatible manner.

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (jax.numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler integration (ms)
    duration = params["duration"]  # Simulation duration (ms)

    # ------------------------------------------------------------------------
    # local parameters
    # See Papadopoulos et al., Relations between large-scale brain connectivity and effects of regional stimulation
    # depend on collective dynamical state, arXiv, 2020
    tau_exc = params["tau_exc"]
    tau_inh = params["tau_inh"]
    c_excexc = params["c_excexc"]
    c_excinh = params["c_excinh"]
    c_inhexc = params["c_inhexc"]
    c_inhinh = params["c_inhinh"]
    a_exc = params["a_exc"]
    a_inh = params["a_inh"]
    mu_exc = params["mu_exc"]
    mu_inh = params["mu_inh"]

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

    key, _ = jax.random.split(params["key"])
    params["key"] = key

    if N == 1:
        Dmat = jnp.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        Dmat = Dmat.at[jnp.diag_indices(N)].set(0)
    Dmat_ndt = jnp.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------
    # Initialization
    t = jnp.arange(1, jnp.round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = jnp.sqrt(dt)

    max_global_delay = int(jnp.max(Dmat_ndt))
    startind = max_global_delay + 1  # timestep to start integration at

    # noise variable
    exc_ou_init = params["exc_ou"].copy()
    inh_ou_init = params["inh_ou"].copy()

    exc_ext_baseline = params["exc_ext_baseline"]
    inh_ext_baseline = params["inh_ext_baseline"]

    exc_ext = mu.adjustArrayShape_jax(params["exc_ext"], jnp.zeros((N, startind + len(t))))
    inh_ext = mu.adjustArrayShape_jax(params["inh_ext"], jnp.zeros((N, startind + len(t))))

    # Set initial values
    # if initial values are just a Nx1 array
    if params["exc_init"].shape[1] == 1:
        exc_init = jnp.dot(params["exc_init"], jnp.ones((1, startind)))
        inh_init = jnp.dot(params["inh_init"], jnp.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        exc_init = params["exc_init"][:, -startind:]
        inh_init = params["inh_init"][:, -startind:]

    # ------------------------------------------------------------------------

    return (
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        exc_init,
        inh_init,
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
        exc_ou_init,
        inh_ou_init,
        exc_ou_mean,
        inh_ou_mean,
        tau_ou,
        sigma_ou,
        key,
    )


@partial(jit, static_argnames=["N"])
def timeIntegration_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    N,
    Cmat,
    K_gl,
    Dmat_ndt,
    exc_init,
    inh_init,
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
    exc_ou_init,
    inh_ou_init,
    exc_ou_mean,
    inh_ou_mean,
    tau_ou,
    sigma_ou,
    key,
):

    update_step = get_update_step(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        Cmat,
        K_gl,
        Dmat_ndt,
        exc_init,
        inh_init,
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
        exc_ou_init,
        inh_ou_init,
        exc_ou_mean,
        inh_ou_mean,
        tau_ou,
        sigma_ou,
        key,
    )

    # Iterating through time steps
    (exc_history, inh_history, exc_ou, inh_ou, i), (excs_new, inhs_new) = jax.lax.scan(
        update_step,
        (exc_init, inh_init, exc_ou_init, inh_ou_init, startind),
        xs=None,
        length=len(t),
    )

    return (
        t,
        jnp.concatenate((exc_init, excs_new.T), axis=1),
        jnp.concatenate((inh_init, inhs_new.T), axis=1),
        exc_ou,
        inh_ou,
    )


def get_update_step(
    startind,
    t,
    dt,
    sqrt_dt,
    N,
    Cmat,
    K_gl,
    Dmat_ndt,
    exc_init,
    inh_init,
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
    exc_ou_init,
    inh_ou_init,
    exc_ou_mean,
    inh_ou_mean,
    tau_ou,
    sigma_ou,
    key,
):
    key, subkey_exc = random.split(key)
    noise_exc = random.normal(subkey_exc, (N, len(t)))
    key, subkey_inh = random.split(key)
    noise_inh = random.normal(subkey_inh, (N, len(t)))

    range_N = jnp.arange(N)

    def S_E(x):
        return 1.0 / (1.0 + jnp.exp(-a_exc * (x - mu_exc)))

    def S_I(x):
        return 1.0 / (1.0 + jnp.exp(-a_inh * (x - mu_inh)))

    def update_step(state, _):
        exc_history, inh_history, exc_ou, inh_ou, i = state

        # Vectorized calculation of delayed excitatory input
        exc_input_d = jnp.sum(K_gl * Cmat * exc_history[range_N, -Dmat_ndt - 1], axis=1)

        # Wilson-Cowan model
        exc_rhs = (
            1
            / tau_exc
            * (
                -exc_history[:, -1]
                + (1 - exc_history[:, -1])
                * S_E(
                    c_excexc * exc_history[:, -1]  # input from within the excitatory population
                    - c_inhexc * inh_history[:, -1]  # input from the inhibitory population
                    + exc_input_d  # input from other nodes
                    + exc_ext_baseline  # baseline external input (static)
                    + exc_ext[:, i - 1]  # time-dependent external input
                )
                + exc_ou  # ou noise
            )
        )
        inh_rhs = (
            1
            / tau_inh
            * (
                -inh_history[:, -1]
                + (1 - inh_history[:, -1])
                * S_I(
                    c_excinh * exc_history[:, -1]  # input from the excitatory population
                    - c_inhinh * inh_history[:, -1]  # input from within the inhibitory population
                    + inh_ext_baseline  # baseline external input (static)
                    + inh_ext[:, i - 1]  # time-dependent external input
                )
                + inh_ou  # ou noise
            )
        )
        # Euler integration
        # make sure e and i variables do not exceed 1 (can only happen with noise)
        exc_new = jnp.clip(exc_history[:, -1] + dt * exc_rhs, 0, 1)
        inh_new = jnp.clip(inh_history[:, -1] + dt * inh_rhs, 0, 1)

        # Update Ornstein-Uhlenbeck process for noise
        exc_ou = (
            exc_ou + (exc_ou_mean - exc_ou) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[:, i - startind]
        )  # mV/ms
        inh_ou = (
            inh_ou + (inh_ou_mean - inh_ou) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[:, i - startind]
        )  # mV/ms

        return (
            (
                jnp.concatenate((exc_history[:, 1:], jnp.expand_dims(exc_new, axis=1)), axis=1),
                jnp.concatenate((inh_history[:, 1:], jnp.expand_dims(inh_new, axis=1)), axis=1),
                exc_ou,
                inh_ou,
                i + 1,
            ),
            (exc_new, inh_new),
        )

    return update_step
