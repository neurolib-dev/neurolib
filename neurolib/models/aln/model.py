import numpy as np

import neurolib.models.aln.chunkwiseIntegration as cw
import neurolib.models.aln.loadDefaultParams as dp
import neurolib.models.aln.timeIntegration as ti
from neurolib.models.model import Model


class ALNModel(Model):
    """
    Multi-population mean-field model with exciatory and inhibitory neurons per population.
    """

    name = "aln"
    description = "Adaptive linear-nonlinear model of exponential integrate-and-fire neurons"

    integration = ti.timeIntegration

    init_vars = [
        "rates_exc_init",
        "rates_inh_init",
        "mufe_init",
        "mufi_init",
        "IA_init",
        "seem_init",
        "seim_init",
        "siim_init",
        "siem_init",
        "seev_init",
        "seiv_init",
        "siiv_init",
        "siev_init",
    ]
    state_vars = [
        "rates_exc",
        "rates_inh",
        "mufe",
        "mufi",
        "IA",
        "seem",
        "seim",
        "siim",
        "siem",
        "seev",
        "seiv",
        "siiv",
        "siev",
    ]
    output_vars = ["rates_exc", "rates_inh"]
    defaultOutput = "rates_exc"
    input_vars = ["ext_exc_current", "ext_exc_rate"]
    defaultInput = "ext_exc_rate"

    modelInputNames = ["ext_exc_current", "ext_exc_rate"]
    modelOutputNames = ["rates_exc", "rates_inh"]

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
        """
        :param params: parameter dictionary of the model
        :param Cmat: Global connectivity matrix (connects E to E)
        :param Dmat: Distance matrix between all nodes (in mm)
        :param lookupTableFileName: Filename for precomputed transfer functions and tables
        :param seed: Random number generator seed
        :param simulateChunkwise: Chunkwise time integration (for lower memory use)
        :param simulateBOLD: Parallel (chunkwise) BOLD simulation
        """

        # Model.addOutputs(self, self.outputNames, self.outputNames)

        # Global attributes
        self.Cmat = Cmat  # Connectivity matrix
        self.Dmat = Dmat  # Delay matrix
        self.lookupTableFileName = lookupTableFileName  # Filename for aLN lookup functions
        self.seed = seed  # Random seed

        # Chunkwise simulation and BOLD
        self.simulateChunkwise = simulateChunkwise  # Chunkwise time integration
        self.simulateBOLD = simulateBOLD  # BOLD
        if simulateBOLD:
            self.simulateChunkwise = True  # Override this setting if BOLD is simulated!
        # Size of integration chunks in chunkwise integration in case of simulateBOLD == True
        self.chunkSize = chunkSize
        # Save data from all chunks? Can be very memory demanding if simulations are long or large
        self.saveAllActivity = saveAllActivity

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(
                Cmat=self.Cmat, Dmat=self.Dmat, lookupTableFileName=self.lookupTableFileName, seed=self.seed
            )

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
            name=self.name,
            description=self.description,
        )

    def getMaxDelay(self):
        # compute maximum delay of model
        dt = self.params["dt"]
        Dmat = dp.computeDelayMatrix(self.params["lengthMat"], self.params["signalV"])
        Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
        ndt_de = round(self.params["de"] / dt)
        ndt_di = round(self.params["di"] / dt)
        max_global_delay = int(max(np.amax(Dmat_ndt), ndt_de, ndt_di))
        return max_global_delay

    # def run(self):
    #     """
    #     Runs an aLN mean-field model simulation.
    #     """

    #     if self.simulateChunkwise:
    #         t_BOLD, BOLD, return_tuple = cw.chunkwiseTimeIntAndBOLD(
    #             self.params, self.chunkSize, self.simulateBOLD, self.saveAllActivity
    #         )
    #         (t, rates_exc, rates_inh, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv,) = return_tuple
    #         self.t_BOLD = t_BOLD
    #         self.BOLD = BOLD
    #         self.setOutput("BOLD.t_BOLD", t_BOLD)
    #         self.setOutput("BOLD.BOLD", BOLD)
    #     else:
    #         (
    #             t,
    #             rates_exc,
    #             rates_inh,
    #             mufe,
    #             mufi,
    #             IA,
    #             seem,
    #             seim,
    #             siem,
    #             siim,
    #             seev,
    #             seiv,
    #             siev,
    #             siiv,
    #         ) = ti.timeIntegration(self.params)

    #     # convert output from kHz to Hz
    #     rates_exc = rates_exc * 1000.0
    #     rates_inh = rates_inh * 1000.0

    #     self.setOutput("t", t)
    #     self.setOutput("rates_exc", rates_exc)
    #     self.setOutput("rates_inh", rates_inh)
