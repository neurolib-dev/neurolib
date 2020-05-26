import numpy as np

from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class ALNModel(Model):
    """
    Multi-population mean-field model with exciatory and inhibitory neurons per population.
    """

    name = "aln"
    description = "Adaptive linear-nonlinear model of exponential integrate-and-fire neurons"

    init_vars = [
        "rates_exc_init",
        "rates_inh_init",
        "mufe_init",
        "mufi_init",
        "IA_init",
        "seem_init",
        "seim_init",
        "siem_init",
        "siim_init",
        "seev_init",
        "seiv_init",
        "siev_init",
        "siiv_init",
        "mue_ou",
        "mui_ou",
    ]

    state_vars = [
        "rates_exc",
        "rates_inh",
        "mufe",
        "mufi",
        "IA",
        "seem",
        "seim",
        "siem",
        "siim",
        "seev",
        "seiv",
        "siev",
        "siiv",
        "mue_ou",
        "mui_ou",
    ]
    output_vars = ["rates_exc", "rates_inh", "IA"]
    default_output = "rates_exc"
    input_vars = ["ext_exc_current", "ext_exc_rate"]
    default_input = "ext_exc_rate"

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
        super().__init__(integration=integration, params=params)

    def getMaxDelay(self):
        # compute maximum delay of model
        ndt_de = round(self.params["de"] / self.params["dt"])
        ndt_di = round(self.params["di"] / self.params["dt"])
        max_dmat_delay = super().getMaxDelay()
        return int(max(max_dmat_delay, ndt_de, ndt_di))
