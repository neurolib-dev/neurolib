from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ...model import Model
from ...aln import ALNModel as ALNModel_numba


class ALNModel(ALNModel_numba):
    """
    Multi-population mean-field model with exciatory and inhibitory neurons per population.
    """

    def __init__(self, params=None, Cmat=None, Dmat=None, lookupTableFileName=None, seed=None):
        """
        :param params: parameter dictionary of the model
        :param Cmat: Global connectivity matrix (connects E to E)
        :param Dmat: Distance matrix between all nodes (in mm)
        :param lookupTableFileName: Filename for precomputed transfer functions and tables
        :param seed: Random number generator seed
        :param simulateChunkwise: Chunkwise time integration (for lower memory use)
        """

        # Global attributes
        self.Cmat = Cmat  # Connectivity matrix
        self.Dmat = Dmat  # Delay matrix
        self.lookupTableFileName = lookupTableFileName  # Filename for aLN lookup functions
        self.seed = seed  # Random seed

        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(
                Cmat=self.Cmat, Dmat=self.Dmat, lookupTableFileName=self.lookupTableFileName, seed=self.seed
            )

        # Initialize base class Model
        Model.__init__(self, integration=integration, params=params)
