import neurolib.models.fhn.loadDefaultParams as dp
import neurolib.models.fhn.timeIntegration as ti
from neurolib.models.model import Model


class FHNModel(Model):
    """
    Fitz-Hugh Nagumo oscillator.
    """

    name = "fhn"
    description = "Fitz-Hugh Nagumo oscillator"

    init_vars = ["xs_init", "ys_init", "x_ou", "y_ou"]
    state_vars = ["x", "y", "x_ou", "y_ou"]
    output_vars = ["x", "y"]
    default_output = "x"
    input_vars = ["x_ext", "y_ext"]
    default_input = "x_ext"

    # because this is not a rate model, the input
    # to the bold model must be normalized
    normalize_bold_input = True

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
