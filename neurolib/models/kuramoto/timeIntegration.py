import numpy as np
import numba

from ...utils import model_utils as mu


def timeIntegration(params):
    """
    setting up parameters for time integration
    
    :param params: Parameter dictionary of the model
    :type params: dict

    :return: Integrated activity of the model
    :rtype: (numpy.ndarray, )
    """ 
    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG
   
    np.random.seed(RNGseed)
    
    # ------------------------------------------------------------------------
    # model parameters
    # ------------------------------------------------------------------------

    N = params["N"]  # number of oscillators

    omega = params["omega"]  # frequencies of oscillators

    # ornstein uhlenbeck noise param
    tau_ou = params["tau_ou"]  # noise time constant
    sigma_ou = params["sigma_ou"]  # noise strength
    
    # ------------------------------------------------------------------------
    # global coupling parameters
    # ------------------------------------------------------------------------

    # Connectivity matrix and Delay
    Cmat = params["Cmat"]

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]
    k = params["k"]  # coupling strength

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)

        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat = Dmat.astype(int)
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt
   
    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------

    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)  # maximum global delay
    startind = int(max_global_delay + 1)  # start simulation after delay 

    # Placeholders
    theta_ou = params['theta_ou'].copy()
    theta = np.zeros((N, startind + len(t)))

    theta_ext = mu.adjustArrayShape(params["theta_ext"], theta)

    # ------------------------------------------------------------------------
    # initial values
    # ------------------------------------------------------------------------  

    if params["theta_init"].shape[1] == 1:
        theta_init = np.dot(params["theta_init"], np.ones((1, startind)))
    else:
        theta_init = params["theta_init"][:, -startind:]
    
    # put noise to instantiated array to save memory
    theta[:, :startind] = theta_init
    theta[:, startind:] = np.random.standard_normal((N, len(t)))
    
    theta_input_d = np.zeros(N)

    noise_theta = 0

    # ------------------------------------------------------------------------
    # some helper variables
    # ------------------------------------------------------------------------

    k_n = k/N
    theta_rhs = np.zeros((N,))

    # ------------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------------
    
    return timeIntegration_njit_elementwise(
        startind,
        t, 
        dt, 
        sqrt_dt,
        N,
        omega,
        k_n, 
        Cmat,
        Dmat,
        theta,
        theta_input_d,
        theta_ext,
        tau_ou,
        sigma_ou,
        theta_ou,
        noise_theta,
        theta_rhs,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t, 
    dt, 
    sqrt_dt,
    N,
    omega,
    k_n, 
    Cmat,
    Dmat,
    theta,
    theta_input_d,
    theta_ext,
    tau_ou,
    sigma_ou,
    theta_ou,
    noise_theta,
    theta_rhs, 
):
    """
    Kuramoto Model 
    """
    for i in range(startind, startind+len(t)):
        # Kuramoto model
        for n in range(N): 
            noise_theta = theta[n, i]
            
            theta_input_d[n] = 0.0

            # adding input from other nodes
            for m in range(N):
                theta_input_d[n] +=  k_n * Cmat[n, m] * np.sin(theta[m, i-1-Dmat[n, m]] - theta[n, i-1])

            theta_rhs[n] = omega[n] + theta_input_d[n] + theta_ou[n] + theta_ext[n, i-1]
            
            # time integration
            theta[n, i] = theta[n, i-1] + dt * theta_rhs[n]
            
            # phase reset
            theta[n, i] = np.mod(theta[n, i], 2*np.pi)

            # Ornstein-Uhlenbeck
            theta_ou[n] = theta_ou[n] - theta_ou[n] * dt / tau_ou + sigma_ou * sqrt_dt * noise_theta

    return t, theta, theta_ou
