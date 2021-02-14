# Parameters

Model parameters in `neurolib` are stored as a dictionary-like object `params` as one of a model's attributes. Changing parameters is straightforward:

```
from neurolib.models.aln import ALNModel # Import the model
model = ALNModel() # Create an instance

model.params['duration'] = 10 * 1000 # in ms
model.run() # Run it
```

Parameters are `dotdict` objects that can also be accessed using the more simple syntax `model.params.parameter_name = 123` (see [Collections](/utils/collections/)).

## Default parameters

The default parameters of a model are stored in the `loadDefaultParams.py` within each model's directory. This function is called by the `model.py` file upon initialisation and returns all necessary parameters of the model.

Below is an example function that prepares the structural connectivity matrices `Cmat` and `Dmat`, all parameters of the model, and its initial values.

```
def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for a model

    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    # set seed to 0 if None, pypet will complain otherwise
    params.seed = seed or 0

    # make sure that seed=0 remains None
    if seed == 0:
        seed = None

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    # the coupling parameter determines how nodes are coupled.
    # "diffusive" for diffusive coupling, "additive" for additive coupling
    params.coupling = "diffusive"

    params.signalV = 20.0
    params.K_gl = 0.6  # global coupling strength

    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity
    params.x_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.y_ou_mean = 0.0  # mV/ms (OU process) [0-5]

    # neural mass model parameters
    params.a = 0.25  # Hopf bifurcation parameter
    params.w = 0.2  # Oscillator frequency, 32 Hz at w = 0.2

    # ------------------------------------------------------------------------

    # initial values of the state variables
    params.xs_init = 0.5 * np.random.uniform(-1, 1, (params.N, 1))
    params.ys_init = 0.5 * np.random.uniform(-1, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.x_ou = np.zeros((params.N,))
    params.y_ou = np.zeros((params.N,))

    # values of the external inputs
    params.x_ext = np.zeros((params.N,))
    params.y_ext = np.zeros((params.N,))

    return params

```