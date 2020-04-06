from ..model import Model
from . import loadDefaultParams as dp
from . import timeIntegration as ti


class ThalamicMassModel(Model):
    """
    Two population thalamic model
    """

    name = "thalamus"
    description = "Two population thalamic mass model"

    init_vars = [
        "V_t_init"
        "V_r_init"
        "Ca_init"
        "h_T_t_init"
        "h_T_r_init"
        "m_h1_init"
        "m_h2_init"
        "s_et_init"
        "s_gt_init"
        "s_er_init"
        "s_gr_init"
        "ds_et_init"
        "ds_gt_init"
        "ds_er_init"
        "ds_gr_init"
    ]
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
        super().__init__(integration=integration, params=params)
