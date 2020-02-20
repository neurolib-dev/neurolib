import numpy as np

import neurolib.models.hopf.chunkwiseIntegration as cw
import neurolib.models.hopf.loadDefaultParams as dp
import neurolib.models.hopf.timeIntegration as ti
from neurolib.models.model import Model


class HopfModel(Model):
    """
    Todo.
    """

    name = "hopf"
    description = "Stuart-Landau model with Hopf bifurcation"

    integration = ti.timeIntegration

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
        self,
        params=None,
        Cmat=None,
        Dmat=None,
        lookupTableFileName=None,
        seed=None,
        simulateChunkwise=False,
        chunkSize=10000,
        simulateBOLD=False,
        saveAllActivity=False,
    ):

        # if Cmat is None:
        #     self.singleNode = True
        # else:
        #     self.singleNode = False
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        self.simulateChunkwise = simulateChunkwise
        self.chunkSize = chunkSize  # Size of integration chunks in chunkwise integration
        self.simulateBOLD = simulateBOLD  # BOLD
        if simulateBOLD:
            self.simulateChunkwise = True  # Override this setting if BOLD is simulated!
        # Save data from all chunks? Can be very memory demanding if simulations are long or large
        self.saveAllActivity = saveAllActivity

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(
            integration=ti.timeIntegration,
            params=params,
            state_vars=self.state_vars,
            init_vars=self.init_vars,
            output_vars=self.output_vars,
            input_vars=self.input_vars,
            default_output=self.defaultOutput,
            simulate_bold=self.simulateBOLD,
            normalize_bold_input=self.normalize_bold_input,
            name=self.name,
            description=self.description,
        )

    def getMaxDelay(self):
        # compute maximum delay of model
        dt = self.params["dt"]
        Dmat = dp.computeDelayMatrix(self.params["lengthMat"], self.params["signalV"])
        Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
        max_global_delay = int(np.amax(Dmat_ndt))
        return max_global_delay

    # def run(self, chunkwise=False):
    #     """Run the model.
    #     """
    #     if chunkwise:
    #         t, x, y, x_ou, y_ou, t_BOLD, BOLD = cw.chunkwiseTimeIntegration(
    #             self.params,
    #             chunkSize=self.chunkSize,
    #             simulateBOLD=self.simulateBOLD,
    #             saveAllActivity=self.saveAllActivity,
    #         )
    #         self.setOutput("BOLD.t", t_BOLD)
    #         self.setOutput("BOLD.BOLD", BOLD)

    #     else:
    #         t, x, y, x_ou, y_ou = ti.timeIntegration(self.params)

    #     self.setOutput("t", t)
    #     self.setOutput("x", x)
    #     self.setOutput("y", y)
    #     self.setOutput("x_ou", x_ou)
    #     self.setOutput("y_ou", y_ou)

