from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class WWModel(Model):
    """
    Wong-Wang model. Original version and reduced version.

    Main reference:
        [original] Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism
        of time integration in perceptual decisions. Journal of Neuroscience, 26(4),
        1314-1328.

    Additional references:
        [reduced] Deco, G., Ponce-Alvarez, A., Mantini, D., Romani, G. L., Hagmann,
        P., & Corbetta, M. (2013). Resting-state functional connectivity emerges
        from structurally and dynamically shaped slow linear fluctuations. Journal
        of Neuroscience, 33(27), 11239-11252.

        [original] Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., Mantini,
        D., & Corbetta, M. (2014). How local excitation–inhibition ratio impacts the
        whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
    """

    name = "wongwang"
    description = "Wong-Wang neural mass model"

    init_vars = ["r_exc", "r_inh", "ses_init", "sis_init", "exc_ou", "inh_ou"]
    state_vars = ["r_exc", "r_inh", "se", "si", "exc_ou", "inh_ou"]
    output_vars = ["r_exc", "r_inh", "se", "si"]
    default_output = "r_exc"

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
