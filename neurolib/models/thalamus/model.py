from ..model import Model
from . import loadDefaultParams as dp
from . import timeIntegration as ti


class ThalamicMassModel(Model):
    """
    Two population thalamic model
    """

    name = "thalamus"
    description = "Two population thalamic mass model"

    init_vars = []
    state_vars = ["V_t", "V_r", "Q_t", "Q_r"]
    output_vars = ["V_t", "V_r", "Q_t", "Q_r"]
    default_output = "V_t"
    input_vars = []
    default_input = None

    def __init__(self, params=None, seed=None):
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams()

        # Initialize base class Model
        super().__init__(
            integration=integration, params=params, bold=False,
        )
