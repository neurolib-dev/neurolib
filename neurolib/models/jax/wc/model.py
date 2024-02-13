from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ...model import Model
from ...wc import WCModel as WCModel_numba


class WCModel(WCModel_numba):
    """
    The two-population Wilson-Cowan model
    """

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):

        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        Model.__init__(self, integration=integration, params=params)
