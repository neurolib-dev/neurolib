import neurolib.models.fhn.chunkwiseIntegration as cw
import neurolib.models.fhn.loadDefaultParams as dp
import neurolib.models.fhn.timeIntegration as ti
from neurolib.models.model import Model


class FHNModel(Model):
    """
    Todo.
    """

    name = "fhn"
    description = "Fitz-Hugh Nagumo oscillator"

    modelOutputs = {"activity": ["x", "y"]}

    defaultOutput = "x"

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
        # Initialize base class Model
        super().__init__(self.name)

        if Cmat is None:
            self.singleNode = True
        else:
            self.singleNode = False
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
            self.params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)
        else:
            self.params = params

        # set default output
        self.setDefaultOutput(self.defaultOutput)

    def run(self):
        """
        Runs the aLN mean-field model simulation
        """
        if self.simulateChunkwise:
            t, x, y, t_BOLD, BOLD = cw.chunkwiseTimeIntegration(
                self.params,
                chunkSize=self.chunkSize,
                simulateBOLD=self.simulateBOLD,
                saveAllActivity=self.saveAllActivity,
            )
            self.setOutput("BOLD.t", t_BOLD)
            self.setOutput("BOLD.BOLD", BOLD)

        else:
            t, x, y = ti.timeIntegration(self.params)

        self.setOutput("t", t)
        self.setOutput("x", x)
        self.setOutput("y", y)
