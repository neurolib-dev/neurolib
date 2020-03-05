import logging
import xarray as xr
import numpy as np

from ..models import bold

from ..utils.collections import dotdict


class Model:
    """The Model superclass runs simulations and manages inputs and outputs of all models.
    """

    def __init__(
        self, integration, params, bold=False,
    ):
        if hasattr(self, "name"):
            if self.name is not None:
                assert isinstance(self.name, str), f"Model name is not a string."

        assert integration is not None, "Model integration function not given."
        self.integration = integration

        assert isinstance(params, dict), "Parameters must be a dictionary."
        self.params = dotdict(params)

        # assert self.state_vars not None:
        assert hasattr(
            self, "state_vars"
        ), f"Model {self.name} has no attribute `state_vars`, which should be alist of strings containing all variable names."
        assert np.all([type(s) is str for s in self.state_vars]), "All entries in state_vars must be strings."

        assert hasattr(
            self, "default_output"
        ), f"Model {self.name} needs to define a default output variable in `default_output`."

        assert isinstance(self.default_output, str), "`default_output` must be a string."

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()

        # set up bold model
        self.boldInitialized = False
        if not hasattr(self, "normalize_bold_input"):
            self.normalize_bold_input = False
        if not hasattr(self, "normalize_bold_input_max"):
            self.normalize_bold_input_max = 50

        # bold initialization at model init
        # if not initialized yet, it will be done when run(bold=True) is called
        # for the first time.
        if bold:
            self.initializeBold(self.normalize_bold_input, self.normalize_bold_input_max)

        logging.info(f"{self.name}: Model initialized.")

    def initializeBold(self, normalize_bold_input, normalize_bold_input_max):
        """Initialize BOLD model.
        
        :param normalize_bold_input: whether or not to normalize the output of the model to a range between 0 and normalize_bold_input_max (in Hz)
        :type normalize_bold_input: bool
        :param normalize_bold_input_max: maximum value in Hz to normalize input to
        :type normalize_bold_input_max: float
        """
        self.normalize_bold_input = normalize_bold_input
        self.normalize_bold_input_max = normalize_bold_input_max
        if self.normalize_bold_input:
            logging.info(f"{self.name}: BOLD input will be normalized to a maxmimum of {normalize_bold_input_max} Hz")

        self.boldModel = bold.BOLDModel(
            self.params["N"], self.params["dt"], normalize_bold_input, normalize_bold_input_max
        )
        self.boldInitialized = True
        logging.info(f"{self.name}: BOLD model initialized.")

    def simulateBold(self, append=False):
        """Gets the default output of the model and simulates the BOLD model. 
        Adds the simulated BOLD signal to outputs.
        """
        if self.boldInitialized:
            self.boldModel.run(self.state[self.default_output], append=append)
            t_BOLD = self.boldModel.t_BOLD
            BOLD = self.boldModel.BOLD
            self.setOutput("BOLD.t", t_BOLD)
            self.setOutput("BOLD.BOLD", BOLD)
        else:
            logging.warn("BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")

    def checkChunkwise(self):
        """Checks if the model fulfills requirements for chunkwise simulation. Throws errors if not.
        """
        assert self.state_vars is not None, "State variable names not given."
        assert self.init_vars is not None, "Initial value variable names not given."
        assert len(self.state_vars) == len(self.init_vars), "State variables are not same length as initial values."

    def initializeRun(self, initializeBold=False):
        """Initialization before each run.

        :param initializeBold: initialize BOLD model
        :type initializeBold: bool
        """
        # NOTE: this if clause causes an error if signalV or Dmat has changed since
        # last calulcateion of max_delay. For every run, we need to compute the new
        # max delay (which is not very good for performance).
        # get the maxDelay of the system
        self.maxDelay = self.getMaxDelay()

        # set up the bold model, if it didn't happen yet
        if initializeBold and not self.boldInitialized:
            self.initializeBold(self.normalize_bold_input, self.normalize_bold_input_max)

    def run(
        self, inputs=None, cont=False, chunkwise=False, chunksize=10000, bold=False, append=False, append_outputs=False
    ):
        """Main interfacing function to run a model. 
        The model can be run in three different ways:
        1) `model.run()` starts a new run.
        2) `model.run(chunkwise=True)` runs the simulation in chunks of length `chunksize`.
        3) `mode.run(cont=True)` continues the simulation of a previous run.
        
        :param inputs: list of inputs to the model, must have the same order as model.input_vars. Note: no sanity check is performed for performance reasons. Take care of the inputs yourself.
        :type inputs: list[np.ndarray|]
        :param cont: continue a simulation by using the initial values from a previous simulation
        :type cont: bool
        :param chunkwise: simulate model chunkwise or in one single run, defaults to False
        :type chunkwise: bool, optional
        :param chunksize: size of the chunk to simulate in dt, defaults to 10000
        :type chunksize: int, optional
        :param bold: simulate BOLD signal (only for chunkwise integration), defaults to False
        :type bold: bool, optional
        :param append: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append: bool, optional
        """
        # TODO legacy argument for compatibility
        append = append_outputs

        self.initializeRun(bold)

        # override some settings if cont==True
        # simulates one dt
        if cont:
            self.autochunk(inputs=inputs, append=append)
            return

        if chunkwise is False:
            self.integrate()
            if bold:
                self.simulateBold()
            return
        else:
            # check if model is safe for chunkwise integration
            self.checkChunkwise()
            if bold and not self.boldInitialized:
                logging.warn(f"{self.name}: BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")
                bold = False
            self.integrateChunkwise(chunksize=chunksize, bold=bold, append=append)
            return

    def integrate(self, append=False):
        """Calls each models `integration` function and saves the state and the outputs of the model.
        
        :param append: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append: bool, optional
        """
        # run integration
        t, *variables = self.integration(self.params)

        # save time array
        self.setOutput("t", t, append=append)
        self.setStateVariables("t", t)
        # save outputs
        for svn, sv in zip(self.state_vars, variables):
            if svn in self.output_vars:
                self.setOutput(svn, sv, append=append)
            self.setStateVariables(svn, sv)

    def integrateChunkwise(self, chunksize, bold=False, append=False):
        """Repeatedly calls the chunkwise integration for the whole duration of the simulation.
        If `bold==True`, the BOLD model is simulated after each chunk.     
        
        :param chunksize: size of each chunk to simulate in units of dt
        :type chunksize: int
        :param bold: simulate BOLD model after each chunk, defaults to False
        :type bold: bool, optional
        :param append: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append: bool, optional
        """

        totalDuration = self.params["duration"]

        dt = self.params["dt"]
        # create a shallow copy of the parameters
        lastT = 0
        while lastT < totalDuration:
            # Determine the size of the next chunk
            currentChunkSize = min(chunksize, (totalDuration - lastT) / dt)
            # currentChunkSize += self.maxDelay + 1
            self.autochunk(chunksize=currentChunkSize, append=append)

            if bold and self.boldInitialized:
                self.simulateBold(append=True)

            # we save the last simulated time step
            lastT += self.state["t"][-1]
        # set duration back to its original value
        self.params["duration"] = totalDuration

    def autochunk(self, inputs=None, chunksize=1, append=False):
        """Executes a single chunk of integration, either for a given duration
        or a single timestep `dt`. Gathers all inputs to the model and resets
        the initial conditions as a preparation for the next chunk. 
        
        :param inputs: list of input values, ordered according to self.input_vars, defaults to None
        :type inputs: list[np.ndarray|], optional
        :param chunksize: length of a chunk to simulate in dt, defaults 1
        :type chunksize: int, optional
        :param append: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append: bool, optional
        """

        # number of time steps of the initialization
        startindt = self.maxDelay + 1
        # set the duration for this chunk
        chunkDuration = startindt + chunksize
        self.params["duration"] = chunkDuration * self.params["dt"]

        # set inputs
        if inputs is not None:
            for i, iv in enumerate(self.input_vars):
                self.params[iv] = inputs[i]

        # run integration
        self.integrate(append=append)

        # reset initial conditions to last state
        for iv, sv in zip(self.init_vars, self.state_vars):
            # if state variables are one-dimensional (in space only)
            if self.state[sv].ndim == 1:
                self.params[iv] = self.state[sv]
            # if they are space-time arrays
            else:
                # we set the next initial condition to the last state
                self.params[iv] = self.state[sv][:, -startindt:]

    def getMaxDelay(self):
        """Computes the maximum delay of the model. This function should be overloaded
        if the model has internal delays (additional to delay between nodes defined by Dmat)
        such as the delay between an excitatory and inhibitory population within each brain area. 
        If this function is not overloaded, the maximum delay is assumed to be defined from the 
        global delay matrix `Dmat`. 
        
        Note: Maxmimum delay is given in units of dt.
        
        :return: maxmimum delay of the model in units of dt
        :rtype: int
        """
        dt = self.params["dt"]
        Dmat = self.params["lengthMat"]

        if "signalV" in self.params:
            signalV = self.params["signalV"]
            if signalV > 0:
                Dmat = Dmat / signalV
            else:
                Dmat = Dmat * 0.0

        Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
        max_global_delay = int(np.amax(Dmat_ndt))
        return max_global_delay

    def setStateVariables(self, name, data):
        """Saves the models current state variables. 
        
        TODO: Cut state variables to length of self.maxDelay
        However, this could be time-memory tradeoff
        
        :param name: name of the state variable
        :type name: str
        :param data: value of the variable
        :type data: np.ndarray
        """
        self.state[name] = data.copy()

    def setOutput(self, name, data, append=False, removeICs=True):
        """Adds an output to the model, typically a simulation result.
        :params name: Name of the output in dot.notation, a la "outputgroup.output"
        :type name: str
        :params data: Output data, can't be a dictionary!
        :type data: `numpy.ndarray`
        """
        assert not isinstance(data, dict), "Output data cannot be a dictionary."
        assert isinstance(name, str), "Output name must be a string."

        # if the output is a single name (not dot.separated)
        if "." not in name:
            # append data
            if append and name in self.outputs:
                if isinstance(self.outputs[name], np.ndarray):
                    assert isinstance(data, np.ndarray), "Cannot append output, not the old type np.ndarray."
                    # remove initial conditions from data
                    if removeICs:
                        startindt = self.maxDelay + 1
                        # if data is one-dim (for example time array)
                        if len(data.shape) == 1:
                            # cut off initial condition
                            data = data[startindt:].copy()
                            # if data is a time array, we need to treat it specially
                            # and increment the time by the last recorded duration
                            if name == "t":
                                data += self.outputs[name][-1] - (startindt - 1) * self.params["dt"]
                        elif len(data.shape) == 2:
                            data = data[:, startindt:].copy()
                        else:
                            raise ValueError(f"Don't know how to truncate data of shape {data.shape}.")
                    self.outputs[name] = np.hstack((self.outputs[name], data))
                # if isinstance(self.outputs[name], list):
                #     assert isinstance(data, np.ndarray), "Cannot append output, not the old type list."
                #     self.outputs[name] = self.outputs[name] + data
                else:
                    raise TypeError(
                        f"Previous output {name} if of type {type(self.outputs[name])}. I can't append to it."
                    )
            else:
                # save all data into output dict
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
                    # todo: this needs to be append-aware like above
                    # todo: for dotted outputs
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
        """Get all outputs of an output group. Examples: `getOutputs("BOLD")` or simply `getOutputs()`

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

    @property
    def output(self):
        """Returns value of default output.
        """
        assert self.default_output is not None, "Default output has not been set yet. Use `setDefaultOutput()`."
        return self.getOutput(self.default_output)

    def xr(self, group=""):
        """Converts a group of outputs to xarray. Output group needs to contain an
        element that starts with the letter "t" or it will not recognize any time axis.

        :param group: Output group name, example:  "BOLD". Leave empty for top group.
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
