import jax.numpy as jnp
from jax import random

from ...wc.loadDefaultParams import loadDefaultParams as loadDefaultParams_numpy


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Wilson-Cowan model

    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: jax.numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: jax.numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = loadDefaultParams_numpy(Cmat, Dmat, seed)

    # Use JAX's PRNGKey for RNG
    key = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
    params.key = key
    params.Cmat = jnp.array(params.Cmat)
    params.lengthMat = jnp.array(params.lengthMat)

    params.exc_init = jnp.array(params.exc_init)
    params.inh_init = jnp.array(params.inh_init)

    params.exc_ou = jnp.array(params.exc_ou)
    params.inh_ou = jnp.array(params.inh_ou)

    return params
