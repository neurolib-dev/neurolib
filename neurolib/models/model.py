import logging
import xarray as xr
import numpy as np

from neurolib.models import bold

from neurolib.utils.collections import dotdict


class Model:
    """The Model superclass manages inputs and outputs of all models.
    """

    def __init__(
        self,
        integration,
        params,
        state_vars,
        init_vars=None,
        output_vars=None,
        input_vars=None,
        default_output=None,
        bold=False,
        normalize_bold_input=False,
        normalize_bold_input_max=50,
        name=None,
        description=None,
    ):
        if name is not None:
            assert isinstance(name, str), f"Model name is not a string."
            self.name = name

        assert integration is not None, "Model integration function not given."
        self.integration = integration

        assert isinstance(params, dict), "Parameters must be a dictionary."
        self.params = dotdict(params)

        self.state_vars = state_vars
        self.init_vars = init_vars
        self.output_vars = output_vars
        self.input_vars = input_vars

        # possibly redundant
        self.default_output = default_output
        self.setDefaultOutput(default_output)

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.max_delay = None
        self.initialize_run()

        # set up bold model
        # NOTE: obsolete, will be called if run(bold==True)
        self.bold_initialized = False
        self.normalize_bold_input = normalize_bold_input
        self.normalize_bold_input_max = normalize_bold_input_max
        if bold:
            self.initialize_bold(self.normalize_bold_input, self.normalize_bold_input_max)

        logging.info(f"{name}: Model initialized.")

    def initialize_bold(self, normalize_bold_input, normalize_bold_input_max):
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
        self.bold_initialized = True
        logging.info(f"{self.name}: BOLD model initialized.")

    def simulate_bold(self):
        """Gets the default output of the model and simulates the BOLD model. 
        Adds the simulated BOLD signal to outputs.
        """
        if self.bold_initialized:
            self.boldModel.run(self.state[self.default_output])
            t_BOLD = self.boldModel.t_BOLD
            BOLD = self.boldModel.BOLD
            self.setOutput("BOLD.t", t_BOLD)
            self.setOutput("BOLD.BOLD", BOLD)
        else:
            logging.warn("BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")

    def check_chunkwise(self):
        """Checks if the model fulfills requirements for chunkwise simulation. Throws errors if not.
        """
        assert self.state_vars is not None, "State variable names not given."
        assert self.init_vars is not None, "Initial value variable names not given."
        assert len(self.state_vars) == len(self.init_vars), "State variables are not same length as initial values."

    def initialize_run(self, initialize_bold=False):
        """Initialization before each run.

        :param initialize_bold: initialize BOLD model
        :type initialize_bold: bool
        """
        # NOTE: this if clause causes an error if signalV or Dmat has changed since
        # last calulcateion of max_delay. For every run, we need to compute the new
        # max delay (which is not very good for performance).

        # if self.max_delay is None:
        self.max_delay = self.getMaxDelay()
        if initialize_bold and not self.bold_initialized:
            self.initialize_bold(self.normalize_bold_input, self.normalize_bold_input_max)

    def run(self, inputs=None, onedt=False, chunkwise=False, chunksize=10000, bold=False, append_outputs=False):
        """Main function to run a model. 
        
        :param inputs: list of inputs to the model, must have the same order as model.input_vars. Note: no sanity check is performed for performance reasons. Take care of the inputs yourself.
        :type inputs: list[np.ndarray|]
        :param onedt: simulate for a single dt
        :type onedt: bool
        :param chunkwise: simulate model chunkwise or in one single run, defaults to False
        :type chunkwise: bool, optional
        :param chunksize: size of the chunk to simulate in dt, defaults to 10000
        :type chunksize: int, optional
        :param bold: simulate BOLD signal (only for chunkwise integration), defaults to False
        :type bold: bool, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append_outputs: bool, optional
        """

        self.initialize_run(bold)

        # override some settings if onedt==True
        if onedt:
            self.autochunk(inputs=inputs, append_outputs=append_outputs)
            return

        if chunkwise is False:
            self.integrate()
            if bold:
                self.simulate_bold()
            return
        else:
            # check if model is safe for chunkwise integration
            self.check_chunkwise()
            if bold and not self.bold_initialized:
                logging.warn(f"{self.name}: BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")
                bold = False
            self.integrate_chunkwise(chunksize=chunksize, bold=bold, append_outputs=append_outputs)
            return

    def integrate(self, append_outputs=False):
        """Calls each models `integration` function and saves the state and the outputs of the model.
        
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append_outputs: bool, optional
        """
        # run integration
        t, *variables = self.integration(self.params)

        # save time array
        self.setOutput("t", t, append=append_outputs)
        self.setStateVariables("t", t)
        # save outputs
        for svn, sv in zip(self.state_vars, variables):
            if svn in self.output_vars:
                self.setOutput(svn, sv, append=append_outputs)
            self.setStateVariables(svn, sv)

    def integrate_chunkwise(self, chunksize, bold=False, append_outputs=False):
        """Repeatedly calls the chunkwise integration for the whole duration of the simulation.
        If `bold==True`, the BOLD model is simulated after each chunk.     
        
        :param chunksize: size of each chunk to simulate in units of dt
        :type chunksize: int
        :param bold: simulate BOLD model after each chunk, defaults to False
        :type bold: bool, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append_outputs: bool, optional
        """
        totalDuration = self.params["duration"]

        dt = self.params["dt"]
        # create a shallow copy of the parameters
        lastT = 0
        while lastT < totalDuration:
            # Determine the size of the next chunk
            currentChunkSize = min(chunksize, (totalDuration - lastT) / dt)
            currentChunkSize += self.max_delay + 1

            self.autochunk(duration=currentChunkSize * dt, append_outputs=append_outputs)

            if bold and self.bold_initialized:
                self.simulate_bold()

            # we save the last simulated time step
            lastT += self.state["t"][-1]

    def autochunk(self, inputs=None, duration=None, append_outputs=False):
        """Executes a single chunk of integration, either for a given duration
        or a single timestep `dt`. Gathers all inputs to the model and resets
        the initial conditions as a preparation for the next chunk. 
        
        :param inputs: list of input values, ordered according to self.input_vars, defaults to None
        :type inputs: list[np.ndarray|], optional
        :param duration: length of a chunk to simulate in ms, defaults to a single dt, defaults to None
        :type duration: float, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append_outputs: bool, optional
        """
        startindt = self.max_delay + 1
        if duration is not None:
            chunkDuration = duration
        else:
            chunkDuration = startindt * self.params["dt"] + self.params["dt"]
        self.params["duration"] = chunkDuration
        # set inputs
        if inputs is not None:
            for i, iv in enumerate(self.input_vars):
                self.params[iv] = inputs[i]

        # run integration
        self.integrate(append_outputs=append_outputs)

        # reset initial conditions to last state
        for iv, sv in zip(self.init_vars, self.state_vars):
            # if state variables are one-dimensional (in space only)
            if len(self.state[sv].shape) == 1:
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
        
        TODO: Cut state variables to length of self.max_delay
        However, this could be time-memory tradeoff
        
        :param name: name of the state variable
        :type name: str
        :param data: value of the variable
        :type data: np.ndarray
        """
        self.state[name] = data.copy()

    def setOutput(self, name, data, append=False, remove_ics=True):
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
                    if remove_ics:
                        startindt = self.max_delay + 1
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
                            raise ValueError("Don't know how to truncate data.")
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
        assert self.defaultOutput is not None, "Default output has not been set yet. Use `setDefaultOutput()`."
        return self.getOutput(self.defaultOutput)

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
