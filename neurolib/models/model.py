import logging
import xarray as xr
import numpy as np

from neurolib.utils.collections import dotdict


class Model:
    """The Model superclass manages inputs and outputs of all models.
    """

    defaultOutput = None

    def __init__(self, name, description=None):
        assert isinstance(name, str), f"Model name is not a string."
        self.name = name
        self.outputs = dotdict({})
        logging.info(f"Model {name} created")

    def setOutput(self, name, data):
        """Adds an output to the model, typically a simulation result.
        :params name: Name of the output in dot.notation, a la "outputgroup.output"
        :type name: str
        :params data: Output data, can't be a dictionary!
        """
        assert not isinstance(data, dict), "Output data cannot be a dictionary."
        assert isinstance(name, str), "Output name must be a string."

        # if the output is a single name (not dot.separated)
        if "." not in name:
            # save into output dict
            self.outputs[name] = data
            # set output as an attribute
            setattr(self, name, self.outputs[name])
        else:
            # build results dictionary and write into self.outputs
            # dot.notation iteration
            keys = name.split(".")
            level = self.outputs  # not copy, reference!
            for i, k in enumerate(keys):
                # if it's the last iteration, store data
                if i == len(keys) - 1:
                    level[k] = data
                # if key is in outputs, then go deeper
                elif k in level:
                    level = level[k]
                    setattr(self, k, level)
                # if it's a new key, create new nested dictionary, set attribute, then go deeper
                else:
                    level[k] = dotdict({})
                    setattr(self, k, level[k])
                    level = level[k]

    def getOutput(self, name):
        """Get an output of a given name (dot.semarated)
        :param name: A key, grouped outputs in the form group.subgroup.variable
        :type name: str

        :returns: Output data
        """
        assert isinstance(name, str), "Output name must be a string."
        keys = name.split(".")
        lastOutput = self.outputs.copy()
        for i, k in enumerate(keys):
            assert k in lastOutput, f"Key {k} not found in outputs."
            lastOutput = lastOutput[k]
        return lastOutput

    def __getitem__(self, key):
        """Index outputs with a dictionary-like key
        """
        return self.getOutput(key)

    def getOutputs(self, group=""):
        """Get all outputs of an output group. Examples: getOutputs("BOLD") or simply getOutputs()

        :param group: Group name, subgroups separated by dots. If left empty (default), all outputs of the root group
            are returned.
        :type group: str
        """
        assert isinstance(group, str), "Group name must be a string."

        def filterOutputsFromGroupDict(groupDict):
            """Return a dictionary with the output data of a group disregarding all other nested dicts.
            :param groupDict: Dictionary of outputs (can include other groups)
            :type groupDict: dict
            """
            assert isinstance(groupDict, dict), "Not a dictionary."
            # make a deep copy of the dictionary
            returnDict = groupDict.copy()
            for key, value in groupDict.items():
                if isinstance(value, dict):
                    del returnDict[key]
            return returnDict

        # if a group deeper than the root is given, select the last node
        lastOutput = self.outputs.copy()
        if len(group) > 0:
            keys = group.split(".")
            for i, k in enumerate(keys):
                assert k in lastOutput, f"Key {k} not found in outputs."
                lastOutput = lastOutput[k]
                assert isinstance(lastOutput, dict), f"Key {k} does not refer to a group."
        # filter out all output *groups* that might be in this node and return only output data
        return filterOutputsFromGroupDict(lastOutput)

    def setDefaultOutput(self, name):
        """Sets the default output of the model.
        :param name: Name of the default output.
        :type name: str
        """
        assert isinstance(name, str), "Default output name must be a string."
        self.defaultOutput = name

    def getDefaultOutput(self):
        """Returns value of default output.
        """
        assert self.defaultOutput is not None, "Default output has not been set yet. Use setDefaultOutput() to set it."
        return self.getOutput(self.defaultOutput)

    def xr(self, group=""):
        """Converts a group of outputs to xarray. Output group needs to contain an
        element called "t" or it will not recognize any time axis.

        :param group: Output group name, example:  "BOLD". Leave blank for top group.
        :type group: str
        """
        assert isinstance(group, str), "Group name must be a string."
        # take all outputs of one group: disregard all dictionaries because they are subgroups
        outputDict = self.getOutputs(group)
        # make sure that there is a time array
        timeDictKey = ""
        if "t" in outputDict:
            timeDictKey = "t"
        else:
            for k in outputDict:
                if k.startswith("t"):
                    timeDictKey = k
                    logging.info(f"Assuming {k} to be the time axis.")
                    break
        assert len(timeDictKey) > 0, f"No time array found (starting with t) in output group {group}."
        t = outputDict[timeDictKey].copy()
        del outputDict[timeDictKey]
        outputs = []
        outputNames = []
        for key, value in outputDict.items():
            outputNames.append(key)
            outputs.append(value)

        nNodes = outputs[0].shape[0]
        nodes = list(range(nNodes))
        allOutputsStacked = np.stack(outputs)  # What? Where? When?
        result = xr.DataArray(allOutputsStacked, coords=[outputNames, nodes, t], dims=["output", "space", "time"])
        return result
