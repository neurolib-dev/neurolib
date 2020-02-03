import numpy as np

import xarray as xr

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

    # multiple outputs can be specified as
    # {"outputname1" : ["variablename1", "variablename2"],
    # {"outputname2" : ["output2varname1"]}}
    # deprecated
    modelOutputs = {"activity": ["x", "y"]}

    defaultOutput = "x"

    def __init__(
        self, params=None, Cmat=[], Dmat=[], lookupTableFileName=None, seed=None, simulateChunkwise=False, chunkSize=10000, simulateBOLD=False,
    ):
        # Initialize base class Model
        Model.__init__(self, self.name)

        if len(Cmat) == 0:
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
        self.saveAllActivity = False  # Save data from all chunks? Can be very memory demanding if simulations are long or large

        # load default parameters if none were given
        if params == None:
            self.params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)
        else:
            self.params = params

        # set default output
        Model.setDefaultOutput(self, self.defaultOutput)

    def run(self):
        """
        Runs the aLN mean-field model simulation
        """
        if self.simulateChunkwise:
            t, x, y, t_BOLD, BOLD = cw.chunkwiseTimeIntegration(self.params, chunkSize=self.chunkSize, simulateBOLD=self.simulateBOLD, saveAllActivity=self.saveAllActivity,)
            # self.t_BOLD = t_BOLD
            # self.BOLD = BOLD
            Model.setOutput(self, "BOLD.t", t_BOLD)
            Model.setOutput(self, "BOLD.BOLD", BOLD)

        else:
            t, x, y = ti.timeIntegration(self.params)

        Model.setOutput(self, "t", t)
        Model.setOutput(self, "x", x)
        Model.setOutput(self, "y", y)

