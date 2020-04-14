from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class SubDivWCModel(Model):
    """
    A three-population Wilson-Cowan model with subtractive and divisive inhibition.

    This model is usually used for cortical nodes within a whole-brain network. Each node is modelled as three
    populations of WC neural masses, one excitatory representing cortical pyramidal cells and two inhibitory,
    representing dendrite-targeting, somatostatin-positive (SST+) interneurons and soma-targeting,
    parvalbumin-positive (PV+) interneurons, respectively. While the SST+ interneuron population simply provides
    subtractive inhibition, the PV+ population can provide both subtractive and divisive inhibition.
    The brain network is realised by coupling excitatory to excitatory masses within the network using the structural
    connectivity matrix and supports network delays.

    References:
        *Papasavvas et al., Divisive gain modulation enables flexible and rapid entrainment in a
         neocortical microcircuit model, J. Neurophysiol., 2020"""

    name = "sdwc"
    description = "Three-population Wilson-Cowan model with both subtractive and divisive inhibition"

    init_vars = ["exc_init", "inh_s_init","inh_d_init", "exc_ou", "inh_s_ou", "inh_d_ou"]
    state_vars = ["exc", "inh_s", "inh_d", "exc_ou", "inh_s_ou", "inh_d_ou"]
    output_vars = ["exc", "inh_s", "inh_d"]
    default_output = "exc"
    input_vars = ["exc_ext", "inh_s_ext", "inh_d_ext"]
    default_input = "exc_ext"

    # because this is not a rate model, the input
    # to the bold model must be transformed
    boldInputTransform = lambda self, x: x * 50

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
        super().__init__(integration=integration, params=params)
