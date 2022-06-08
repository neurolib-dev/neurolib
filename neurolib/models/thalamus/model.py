from ..model import Model
from . import loadDefaultParams as dp
from . import timeIntegration as ti


class ThalamicMassModel(Model):
    """
    Two population thalamic model

    Reference:
    Costa, M. S., Weigenand, A., Ngo, H. V. V., Marshall, L., Born, J.,
    Martinetz, T., & Claussen, J. C. (2016). A thalamocortical neural mass
    model of the EEG during NREM sleep and its response to auditory stimulation.
    PLoS computational biology, 12(9).

    """

    name = "thalamus"
    description = "Two population thalamic mass model"

    init_vars = [
        "V_t_init",
        "V_r_init",
        "Q_t_init",
        "Q_r_init",
        "Ca_init",
        "h_T_t_init",
        "h_T_r_init",
        "m_h1_init",
        "m_h2_init",
        "s_et_init",
        "s_gt_init",
        "s_er_init",
        "s_gr_init",
        "ds_et_init",
        "ds_gt_init",
        "ds_er_init",
        "ds_gr_init",
    ]
    state_vars = [
        "V_t",
        "V_r",
        "Q_t",
        "Q_r",
        "Ca",
        "h_T_t",
        "h_T_r",
        "m_h1",
        "m_h2",
        "s_et",
        "s_gt",
        "s_er",
        "s_gr",
        "ds_et",
        "ds_gt",
        "ds_er",
        "ds_gr",
    ]
    output_vars = ["V_t", "V_r", "Q_t", "Q_r"]
    default_output = "Q_t"
    input_vars = []
    default_input = None

    def __init__(self, params=None, seed=None):
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(seed=seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)

    def randomICs(self):
        ics = dp.generateRandomICs()
        for idx, iv in enumerate(self.init_vars):
            self.params[iv] = ics[idx]
