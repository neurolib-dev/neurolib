import numpy as np
from kuramoto.utils import set_seed


def kuramoto(omega, t, k, theta, simulate_noise=False, seed=None):
    """
    Kuramoto Model
    
    omega: frequencies of oscillators
    t: time steps (not used)
    k: coupling strength
    theta: phases of oscillators
    """
    set_seed(seed)
    
    n_osc = len(omega)
    one_per_osc = 1.0 / n_osc
    
    d_omega_dt = omega.copy()

    for i, j in np.ndindex((n_osc, n_osc)):
        d_omega_dt[i] += k * one_per_osc * np.sin(theta[j] - theta[i])

    return d_omega_dt
