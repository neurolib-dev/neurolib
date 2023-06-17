from . import loadDefaultParams as dp
from . import timeIntegration as ti

from neurolib.models.model import Model


class KuramotoModel(Model):
    """
    Kuramoto Model
    """

    name = "kuramoto"
    description = "Kuramoto Model"

    init_vars = ['theta_init', 'theta_ou']
    state_vars = ['theta', 'theta_ou'] # change x to theta
    output_vars = ['theta']
    default_output = 'theta'
    input_vars = None
    default_input = None

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed
        
        integration = ti.timeIntegration
        
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)
        
        super().__init__(params=params, integration=integration)