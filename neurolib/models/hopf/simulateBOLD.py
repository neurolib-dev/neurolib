import numpy as np

import numba


def simulateBOLD(Z, dt, voxelCounts, X=None, F=None, Q=None, V=None):
    # PREDICTEDBOLD Compute the simulated BOLD signal out of the synaptic activity using the Ballon-Windkessel model
    #   BOLD = predictBold(synapticActivity,dt,voxelCounts)
    #
    #   Params:
    #       synapticActivity    N x length matrix. Synaptic activity of the Nth brain area at each time steps (Hz)
    #       dt                  Scalar. Sampling period for the synaptic activity. (s)
    #                           length x dt = simulation duration
    #       voxelCounts         (optional) N x 1 matrix. Number of fMRI voxels contained in each brain area.
    #       X                   (optional) N x 1 matrix. Initial values: Vasso dilatory signal
    #       F                   (optional) N x 1 matrix. Initial values: Blood flow
    #       Q                   (optional) N x 1 matrix. Initial values: Deoxyhemoglobin
    #       V                   (optional) N x 1 matrix. Initial values: Blood volume
    #
    #   Return:
    #       BOLD    N x length matrix: simulated BOLD for the N brain areas.
    #               Note: This signal needs then to be downsampled to be compared to the empirical fMRI signal
    #
    #   See Friston 2003 and Deco 2013 for reference on how the BOLD signal is simulated

    N = np.shape(Z)[0]

    if not "voxelCounts" in globals():
        voxelCounts = np.ones((N,))

    voxelCountsSqrtInv = 1 / np.sqrt(voxelCounts)

    # Ballon-Windkessel model parameters (Deco 2013, Friston 2003):
    # Note: the distribution of each Ballon-Windkessel models parameters values are given pro Voxels
    #   Since we average the empirical fMRI of each voxel for a given area, the standard deviation of the gaussian distribution
    #   should be divided by the number of voxels present in the area
    # rho     = np.random.normal(0.34, np.sqrt(0.0024) / np.sqrt( sum(voxelCounts) ) )    # Capillary resting net oxygen extraction
    # alpha   = np.random.normal(0.32, np.sqrt(0.0015) / np.sqrt( sum(voxelCounts) ) )    # Grubb's vessel stiffness exponent
    # V0      = 0.02
    # k1      = 7 * rho
    # k2      = 2.0
    # k3      = 2 * rho - 0.2
    # Gamma   = np.random.normal(0.41 * np.ones(N), np.sqrt(0.002) * voxelCountsSqrtInv)   # Rate constant for autoregulatory feedback by blood flow
    # K       = np.random.normal(0.65 * np.ones(N), np.sqrt(0.015) * voxelCountsSqrtInv)   # Vasodilatory signal decay
    # Tau     = np.random.normal(0.98 * np.ones(N), np.sqrt(0.0568) * voxelCountsSqrtInv)   # Transit time
    rho = 0.34  # Capillary resting net oxygen extraction (dimensionless)
    alpha = 0.32  # Grubb's vessel stiffness exponent (dimensionless)
    V0 = 0.02  # Resting blood volume fraction (dimensionless)
    k1 = 7 * rho  # (dimensionless)
    k2 = 2.0  # (dimensionless)
    k3 = 2 * rho - 0.2  # (dimensionless)
    Gamma = 0.41 * np.ones(
        (N,)
    )  # Rate constant for autoregulatory feedback by blood flow (1/s)
    K = 0.65 * np.ones((N,))  # Vasodilatory signal decay (1/s)
    Tau = 0.98 * np.ones((N,))  # Transit time  (s)

    # ballon-Windkessel state variables

    if X is None:
        X = np.zeros((N,))  # Vasso dilatory signal
    if F is None:
        F = np.zeros((N,))  # Blood flow
    if Q is None:
        Q = np.zeros((N,))  # Deoxyhemoglobin
    if V is None:
        V = np.zeros((N,))  # Blood volume

    BOLD = np.zeros(np.shape(Z))

    return integrateBOLD_numba(
        BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau
    )


@numba.njit
def integrateBOLD_numba(
    BOLD, X, Q, F, V, Z, dt, N, rho, alpha, V0, k1, k2, k3, Gamma, K, Tau
):

    for i in range(len(Z[0, :])):  # loop over all timesteps

        #        X = X + dt * ( Z[:,i] - K * X - Gamma * ( F - 1 ) )
        #        Q = Q + dt / Tau * ( F / rho * ( 1 - ( 1 - rho ) ** (1 / F) )  \
        #                             - Q * V ** ( 1 / alpha - 1 ) )
        #        V = V + dt / Tau * ( F - V ** ( 1 / alpha ) )
        #        F = F + dt * X
        #
        #        BOLD[:,i] = V0 * ( k1 * ( 1 - Q) + k2 * (1 - Q / V ) + k3 * (1 - V) )

        # component-wise loop for compatibilty with numba
        for j in range(N):  # loop over all areas
            X[j] = X[j] + dt * (Z[j, i] - K[j] * X[j] - Gamma[j] * (F[j] - 1))
            Q[j] = Q[j] + dt / Tau[j] * (
                F[j] / rho * (1 - (1 - rho) ** (1 / F[j]))
                - Q[j] * V[j] ** (1 / alpha - 1)
            )
            V[j] = V[j] + dt / Tau[j] * (F[j] - V[j] ** (1 / alpha))
            F[j] = F[j] + dt * X[j]

            BOLD[j, i] = V0 * (
                k1 * (1 - Q[j]) + k2 * (1 - Q[j] / V[j]) + k3 * (1 - V[j])
            )

    return BOLD, X, F, Q, V
