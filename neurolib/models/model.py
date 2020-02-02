import logging
import xarray as xr
import numpy as np


class Model:
    # possiby deprecated
    inputNames = []
    inputs = []
    nInputs = 0
    outputNames = None

    # working
    defaultOutput = None
    outputs = {}
    xrs = {}

    def __init__(self, name, description=None):
        assert isinstance(name, str), f"Model name is not a string."
        self.name = name
        logging.info(f"Model {name} created")

    def setOutput(self, name, data):
        """Adds an output to the model, typically a simulation result.
        :params name: Name of the output split by dots for each dictionary key, a la "rates.rates_exc"
        :type name: str
        :params data: Output data, can't be a dictionary!
        """
        assert not isinstance(data, dict), "Output data cannot be a dictionary."
        assert isinstance(name, str), "Output name must be a string."
        # build results dictionary and write into self.outputs
        keys = name.split(".")
        level = self.outputs
        for i, k in enumerate(keys):
            # if it's the last iteration, it's data
            if i == len(keys) - 1:
                level[k] = data
            # if it's a known key, then go deeper
            elif k in level:
                level = level[k]
            # if it's a new key, create new nested dictionary and go deeper
            else:
                level[k] = {}
                level = level[k]

    def getOutput(self, name):
        """Get an output.
        :param name: A key of the form group.subgroup.variable
        :type name: str
        """
        assert isinstance(name, str), "Output name must be a string."
        keys = name.split(".")
        lastOutput = self.outputs
        for i, k in enumerate(keys):
            lastOutput = lastOutput[k]
        return lastOutput

    def __getitem__(self, key):
        """Index outputs with a dictionary-like key
        """
        return self.getOutput(key)

    def setDefaultOutput(self, name):
        assert isinstance(name, str), "Default output name must be a string."
        self.defaultOutput = name

    def getDefaultOutput(self):
        assert not self.defaultOutput == None, "Default output has not been set yet. Use setDefaultOutput() to set it."
        return self.getOutput(self.defaultOutput)

    def addOutputs(self, name, t, outputs, outputNames=None):
        # if no names are provided, make up names
        # if outputs is a list
        if outputNames == None and isinstance(outputs, list):
            outputNames = [self.name + "-output-" + str(i) for i in range(len(outputs))]
        elif outputNames == None:
            outputNames = [self.name + "-output"]

        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(outputNames, list):
            outputNames = [outputNames]

        # sanity check
        assert len(outputs) == len(
            outputNames
        ), f"Length of output ({name}) = {len(outputs)} doesn't match the length of the names provided = {outputNames}"

        # save outputs
        self.outputs[name] = {}
        self.outputs[name]["t"] = t
        for o, on in zip(outputs, outputNames):
            self.outputs[name][on] = o

        self.xrs[name] = self.outputsToXarray(t, outputs, outputNames)

    def outputsToXarray(self, t, outputs, outputNames):
        # assume
        nNodes = outputs[0].shape[0]
        nodes = list(range(nNodes))
        allOutputsStacked = np.stack(outputs)  # What? Where? When?
        return xr.DataArray(allOutputsStacked, coords=[outputNames, nodes, t], dims=["variable", "space", "time"],)
