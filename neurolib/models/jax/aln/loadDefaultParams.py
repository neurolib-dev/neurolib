import jax.numpy as jnp
from jax import random

from ...aln.loadDefaultParams import loadDefaultParams as loadDefaultParams_numpy


def loadDefaultParams(Cmat=None, Dmat=None, lookupTableFileName=None, seed=None):
    """Load default parameters for a network of aLN nodes.
    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lookUpTableFileName: Filename of lookup table with aln non-linear transfer functions and other precomputed quantities., defaults to aln-precalc/quantities_cascade.h
    :type lookUpTableFileName: str, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = loadDefaultParams_numpy(Cmat, Dmat, lookupTableFileName, seed)

    # Use JAX's PRNGKey for RNG
    key = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
    params.key = key

    params.Cmat = jnp.array(params.Cmat)
    params.lengthMat = jnp.array(params.lengthMat)

    params.mue_ou = jnp.array(params.mue_ou)
    params.mui_ou = jnp.array(params.mui_ou)

    params.mufe_init = jnp.array(params.mufe_init)
    params.mufi_init = jnp.array(params.mufi_init)
    params.IA_init = jnp.array(params.IA_init)
    params.seem_init = jnp.array(params.seem_init)
    params.seim_init = jnp.array(params.seim_init)
    params.seev_init = jnp.array(params.seev_init)
    params.seiv_init = jnp.array(params.seiv_init)
    params.siim_init = jnp.array(params.siim_init)
    params.siem_init = jnp.array(params.siem_init)
    params.siiv_init = jnp.array(params.siiv_init)
    params.siev_init = jnp.array(params.siev_init)
    params.rates_exc_init = jnp.array(params.rates_exc_init)
    params.rates_inh_init = jnp.array(params.rates_inh_init)

    params.Irange = jnp.array(params.Irange)
    params.sigmarange = jnp.array(params.sigmarange)
    params.precalc_r = jnp.array(params.precalc_r)
    params.precalc_V = jnp.array(params.precalc_V)
    params.precalc_tau_mu = jnp.array(params.precalc_tau_mu)
    params.precalc_tau_sigma = jnp.array(params.precalc_tau_sigma)

    return params
