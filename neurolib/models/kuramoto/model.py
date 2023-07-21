from . import loadDefaultParams as dp
from . import timeIntegration as ti

from neurolib.models.model import Model


class KuramotoModel(Model):
    """
    Kuramoto Model

    Based on:
    Kuramoto, Yoshiki (1975). H. Araki (ed.). Lecture Notes in Physics, International Symposium on Mathematical Problems in Theoretical Physics.
    """

    name = "kuramoto"
    description = "Kuramoto Model"

    init_vars = ['theta_init', 'theta_ou']
    state_vars = ['theta', 'theta_ou'] 
    output_vars = ['theta']
    default_output = 'theta'
    input_vars = ['theta_ext']
    default_input = 'theta_ext'

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed
        
        integration = ti.timeIntegration
        
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)
        
        super().__init__(params=params, integration=integration)