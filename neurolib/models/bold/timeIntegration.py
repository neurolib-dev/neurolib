import numpy as np
import numba


def simulateBOLD(Z, dt, voxelCounts, X=None, F=None, Q=None, V=None):
    """Simulate BOLD activity using the Balloon-Windkessel model.
    See Friston 2000, Friston 2003 and Deco 2013 for reference on how the BOLD signal is simulated.
    The returned BOLD signal should be downsampled to be comparable to a recorded fMRI signal.

    :param Z: Synaptic activity
    :type Z: numpy.ndarray
    :param dt: dt of input activity in s
    :type dt: float
    :param voxelCounts: Number of voxels in each region (not used yet!)
    :type voxelCounts: numpy.ndarray
    :param X: Initial values of Vasodilatory signal, defaults to None
    :type X: numpy.ndarray, optional
    :param F: Initial values of Blood flow, defaults to None
    :type F: numpy.ndarray, optional
    :param Q: Initial values of Deoxyhemoglobin, defaults to None
    :type Q: numpy.ndarray, optional
    :param V: Initial values of Blood volume, defaults to None
    :type V: numpy.ndarray, optional

    :return: BOLD, X, F, Q, V
    :rtype: (numpy.ndarray,)
    """

    N = np.shape(Z)[0]

    if "voxelCounts" not in globals():
        voxelCounts = np.ones((N,))

    # Balloon-Windkessel model parameters (from Friston 2003):
    # Friston paper: Nonlinear responses in fMRI: The balloon model, Volterra kernels, and other hemodynamics
    # Note: the distribution of each Balloon-Windkessel models parameters are given per voxel
    # Since we usually average the empirical fMRI of each voxel for a given area, the standard
    # deviation of the gaussian distribution should be divided by the number of voxels in each area
    # voxelCountsSqrtInv = 1 / np.sqrt(voxelCounts)
    #
    # See Friston 2003, Table 1 mean values and variances:
    # rho     = np.random.normal(0.34, np.sqrt(0.0024) / np.sqrt( sum(voxelCounts) ) )    # Capillary resting net oxygen extraction
    # alpha   = np.random.normal(0.32, np.sqrt(0.0015) / np.sqrt( sum(voxelCounts) ) )    # Grubb's vessel stiffness exponent
    # V0      = 0.02
    # k1      = 7 * rho
    # k2      = 2.0
    # k3      = 2 * rho - 0.2
    # Gamma   = np.random.normal(0.41 * np.ones(N), np.sqrt(0.002) * voxelCountsSqrtInv)   # Rate constant for autoregulatory feedback by blood flow
    # K       = np.random.normal(0.65 * np.ones(N), np.sqrt(0.015) * voxelCountsSqrtInv)   # Vasodilatory signal decay
    # Tau     = np.random.normal(0.98 * np.ones(N), np.sqrt(0.0568) * voxelCountsSqrtInv)   # Transit time
    #
    # If no voxel counts are given, we can use scalar values for each region's parameter:
    rho = 0.34  # Capillary resting net oxygen extraction (dimensionless), E_0 in Friston2000
    alpha = 0.32  # Grubb's vessel stiffness exponent (dimensionless), \alpha in Friston2000
    V0 = 0.02  # Resting blood volume fraction (dimensionless)
    k1 = 7 * rho  # (dimensionless)
    k2 = 2.0  # (dimensionless)
    k3 = 2 * rho - 0.2  # (dimensionless)
    Gamma = 0.41 * np.ones((N,))  # Rate constant for autoregulatory feedback by blood flow (1/s)
    K = 0.65 * np.ones((N,))  # Vasodilatory signal decay (1/s)
    Tau = 0.98 * np.ones((N,))  # Transit time  (s)

    # initialize state variables
    # NOTE: We need to use np.copy() because these variables
    # will be overwritten later and numba doesn't like to do that
    # with anything that was defined outside the scope of the @njit'ed function
    X = np.zeros((N,)) if X is None else np.copy(X)  # Vasso dilatory signal
    F = np.zeros((N,)) if F is None else np.copy(F)  # Blood flow
    Q = np.zeros((N,)) if Q is None else np.copy(Q)  # Deoxyhemoglobin
    V = np.zeros((N,)) if V is None else np.copy(V)  # Blood volume

    BOLD = np.zeros(np.shape(Z))
    # return integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau)
    BOLD, X, F, Q, V = integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau)
    return BOLD, X, F, Q, V


@numba.njit
def integrateBOLD_numba(BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau):
    """Integrate the Balloon-Windkessel model.

    Reference:

    Friston et al. (2000), Nonlinear responses in fMRI: The balloon model, Volterra kernels, and other hemodynamics.
    Friston et al. (2003), Dynamic causal modeling

    Variable names in Friston2000:
    X = x1, Q = x4, V = x3, F = x2

    Friston2003: see Equation (3)

    NOTE: A very small constant EPS is added to F to avoid F become too small / negative
    and cause a floating point error in EQ. Q due to the exponent **(1 / F[j])

    """

    EPS = 1e-120  # epsilon for softening

    for i in range(len(Z[0, :])):  # loop over all timesteps
        # component-wise loop for compatibilty with numba
        for j in range(N):  # loop over all areas
            X[j] = X[j] + dt * (Z[j, i] - K[j] * X[j] - Gamma[j] * (F[j] - 1))
            Q[j] = Q[j] + dt / Tau[j] * (F[j] / rho * (1 - (1 - rho) ** (1 / F[j])) - Q[j] * V[j] ** (1 / alpha - 1))
            V[j] = V[j] + dt / Tau[j] * (F[j] - V[j] ** (1 / alpha))
            F[j] = F[j] + dt * X[j]

            F[j] = max(F[j], EPS)

            BOLD[j, i] = V0 * (k1 * (1 - Q[j]) + k2 * (1 - Q[j] / V[j]) + k3 * (1 - V[j]))
    return BOLD, X, F, Q, V
