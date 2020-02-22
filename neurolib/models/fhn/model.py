import numpy as np

from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class FHNModel(Model):
    """
    Fitz-Hugh Nagumo oscillator.
    """

    name = "fhn"
    description = "Fitz-Hugh Nagumo oscillator"

    init_vars = ["xs_init", "ys_init", "x_ou", "y_ou"]
    state_vars = ["x", "y", "x_ou", "y_ou"]
    output_vars = ["x", "y"]
    defaultOutput = "x"
    input_vars = ["x_ext", "y_ext"]
    defaultInput = "x_ext"

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
            integration=integration,
            params=params,
            state_vars=self.state_vars,
            init_vars=self.init_vars,
            output_vars=self.output_vars,
            input_vars=self.input_vars,
            default_output=self.defaultOutput,
            bold=bold,
            normalize_bold_input=self.normalize_bold_input,
            name=self.name,
            description=self.description,
        )
