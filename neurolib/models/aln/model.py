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

    modelInputNames = ["ext_exc_current", "ext_exc_rate"]
    modelOutputNames = ["rates_exc", "rates_inh"]

    def __init__(
        self,
        params=None,
        Cmat=[],
        Dmat=[],
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
        # Initialize base class Model
        super().__init__(self.name)
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
        self.chunkSize = (
            chunkSize  # Size of integration chunks in chunkwise integration in case of simulateBOLD == True
        )
        self.saveAllActivity = (
            saveAllActivity  # Save data of all chunks? Can be very memory demanding if simulations are long or large
        )

        # load default parameters if none were given
        if params is None:
            self.params = dp.loadDefaultParams(
                Cmat=self.Cmat, Dmat=self.Dmat, lookupTableFileName=self.lookupTableFileName, seed=self.seed,
            )
        else:
            self.params = params

    def run(self):
        """
        Runs an aLN mean-field model simulation.
        """

        if self.simulateChunkwise:
            t_BOLD, BOLD, return_tuple = cw.chunkwiseTimeIntAndBOLD(
                self.params, self.chunkSize, self.simulateBOLD, self.saveAllActivity
            )
            (rates_exc, rates_inh, t, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv,) = return_tuple
            self.t_BOLD = t_BOLD
            self.BOLD = BOLD
            self.setOutput("BOLD.t_BOLD", t_BOLD)
            self.setOutput("BOLD.BOLD", BOLD)
        else:
            (
                rates_exc,
                rates_inh,
                t,
                mufe,
                mufi,
                IA,
                seem,
                seim,
                siem,
                siim,
                seev,
                seiv,
                siev,
                siiv,
            ) = ti.timeIntegration(self.params)

        # convert output from kHz to Hz
        rates_exc = rates_exc * 1000.0
        rates_inh = rates_inh * 1000.0

        self.setOutput("t", t)
        self.setOutput("rates_exc", rates_exc)
        self.setOutput("rates_inh", rates_inh)
