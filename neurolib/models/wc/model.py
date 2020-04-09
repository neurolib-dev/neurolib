from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class WCModel(Model):
    """
    The two-population Wilson-Cowan model
    """

    name = "wc"
    description = "Wilson-Cowan model"

    init_vars = ["es_init", "is_init", "e_ou", "i_ou"]
    state_vars = ["e", "i", "e_ou", "i_ou"]
    output_vars = ["e", "i"]
    default_output = "e"
    input_vars = ["e_ext", "i_ext"]
    default_input = "e_ext"

    # because this is not a rate model, the input
    # to the bold model must be normalized
    normalize_bold_input = True
    normalize_bold_input_max = 50

    def __init__(
        self, params=None, Cmat=None, Dmat=None, lookupTableFileName=None, seed=None, bold=False,
    ):

        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(
            integration=integration, params=params, bold=bold,
        )
