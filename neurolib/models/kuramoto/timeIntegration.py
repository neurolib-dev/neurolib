import numpy as np
import numba

from ...utils import model_utils as mu

# TODO: rename variables to theta
# TODO: loop over nodes
# TODO: remove time_integrateion '
# TODO: integrate input

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
    theta_ou_mean = params["theta_ou_mean"]
    
    # ------------------------------------------------------------------------
    # global coupling parameters
    # ------------------------------------------------------------------------
    #TO CHECK, added from other models

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
    thetas = np.zeros((N, startind + len(t)))

    # ------------------------------------------------------------------------
    # initial values
    # ------------------------------------------------------------------------  
    if params["thetas_init"].shape[1] == 1:
        thetas_init = np.dot(params["thetas_init"], np.ones((1, startind)))
    else:
        thetas_init = params["thetas_init"][:, -startind:]
    
    # put noise to instantiated array to save memory
    thetas[:, :startind] = thetas_init
    thetas[:, startind:] = np.random.standard_normal((N, len(t)))
    
    noise_thetas = np.zeros((N,))

    # ------------------------------------------------------------------------
    # some helper variables
    # ------------------------------------------------------------------------
    k_n = k/N
    theta_rhs = np.zeros((N,))

    # ------------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------------
    return timeIntegration_njit_elementwise(
        startind=startind,
        t=t,
        dt=dt,
        sqrt_dt=sqrt_dt,
        N=N,
        omega=omega,
        k_n=k_n,
        Cmat=Cmat,
        Dmat=Dmat_ndt,
        thetas=thetas,
        tau_ou=tau_ou,
        sigma_ou=sigma_ou,
        theta_ou=theta_ou,
        theta_ou_mean=theta_ou_mean,
        noise_thetas=noise_thetas,
        theta_rhs=theta_rhs,
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
    thetas,
    tau_ou,
    sigma_ou,
    theta_ou,
    theta_ou_mean,
    noise_thetas,
    theta_rhs, # right hand side
):
    """
    Kuramoto Model 
    """
    for i in range(startind, startind+len(t)):
        # get noise from thetas
        noise_thetas = thetas[:, i]

        # # Kuramoto model
        # theta_rhs[:] = 0.0
        # for n in range(N): 
        #     for m in range(N):
        #         theta_rhs[n] += Cmat[n, m] * np.sin(thetas[m, i-Dmat[n, m]] - thetas[n, i-1])
        #     theta_rhs *= k_n

        theta_rhs[:] = 0.0
        for n, m in np.ndindex((N, N)):
            theta_rhs[n] += Cmat[n, m] * np.sin(thetas[m, i-1-Dmat[n, m]] - thetas[n, i-1])
        theta_rhs *= k_n

        # Ornstein-Uhlenbeck process
        theta_ou = theta_ou + (theta_ou_mean - theta_ou) * dt / tau_ou + sigma_ou * sqrt_dt * noise_thetas
    
        sum_theta = omega + theta_rhs + theta_ou
        # Euler integration
        new_theta = thetas[:, i-1] + dt * sum_theta

        new_theta = np.mod(new_theta, 2*np.pi)

        thetas[:, i] = new_theta
    

    return t, thetas, theta_ou
