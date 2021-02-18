import logging
import xarray as xr
import numpy as np

from ..models import bold

from ..utils.collections import dotdict


class Model:
    """The Model base class runs models, manages their outputs, parameters and more.
    This class should serve as the base class for all implemented models.
    """

    def __init__(self, integration, params):
        if hasattr(self, "name"):
            if self.name is not None:
                assert isinstance(self.name, str), f"Model name is not a string."
        else:
            self.name = "Noname"

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

        # if no output_vars is set, it will be replaced by state_vars
        if not hasattr(self, "output_vars"):
            self.output_vars = self.state_vars

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()

        self.boldInitialized = False

        logging.info(f"{self.name}: Model initialized.")

    def initializeBold(self):
        """Initialize BOLD model."""
        self.boldInitialized = False

        # function to transform model state before passing it to the bold model
        # Note: This can be used like the parameter \epsilon in Friston2000
        # (neural efficacy) by multiplying the input with a constant via
        # self.boldInputTransform = lambda x: x * epsilon
        if not hasattr(self, "boldInputTransform"):
            self.boldInputTransform = None

        self.boldModel = bold.BOLDModel(self.params["N"], self.params["dt"])
        self.boldInitialized = True
        # logging.info(f"{self.name}: BOLD model initialized.")

    def simulateBold(self, t, variables, append=False):
        """Gets the default output of the model and simulates the BOLD model.
        Adds the simulated BOLD signal to outputs.
        """
        if self.boldInitialized:
            # first we loop through all state variables
            for svn, sv in zip(self.state_vars, variables):
                # the default output is used as the input for the bold model
                if svn == self.default_output:
                    bold_input = sv[:, self.startindt :]
                    # logging.debug(f"BOLD input `{svn}` of shape {bold_input.shape}")
                    if bold_input.shape[1] >= self.boldModel.samplingRate_NDt:
                        # only if the length of the output has a zero mod to the sampling rate,
                        # the downsampled output from the boldModel can correctly appended to previous data
                        # so: we are lazy here and simply disable appending in that case ...
                        if not bold_input.shape[1] % self.boldModel.samplingRate_NDt == 0:
                            append = False
                            logging.warn(
                                f"Output size {bold_input.shape[1]} is not a multiple of BOLD sampling length { self.boldModel.samplingRate_NDt}, will not append data."
                            )
                        logging.debug(f"Simulating BOLD: boldModel.run(append={append})")

                        # transform bold input according to self.boldInputTransform
                        if self.boldInputTransform:
                            bold_input = self.boldInputTransform(bold_input)

                        # simulate bold model
                        self.boldModel.run(bold_input, append=append)

                        t_BOLD = self.boldModel.t_BOLD
                        BOLD = self.boldModel.BOLD
                        self.setOutput("BOLD.t_BOLD", t_BOLD)
                        self.setOutput("BOLD.BOLD", BOLD)
                    else:
                        logging.warn(
                            f"Will not simulate BOLD if output {bold_input.shape[1]*self.params['dt']} not at least of duration {self.boldModel.samplingRate_NDt*self.params['dt']}"
                        )
        else:
            logging.warn("BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")

    def checkChunkwise(self, chunksize):
        """Checks if the model fulfills requirements for chunkwise simulation.
        Checks whether the sampling rate for outputs fits to chunksize and duration.
        Throws errors if not."""
        assert self.state_vars is not None, "State variable names not given."
        assert self.init_vars is not None, "Initial value variable names not given."
        assert len(self.state_vars) == len(self.init_vars), "State variables are not same length as initial values."

        # throw a warning if the user is nasty
        if int(self.params["duration"] / self.params["dt"]) % chunksize != 0:
            logging.warning(
                f"It is strongly advised to use a `chunksize` ({chunksize}) that is a divisor of `duration / dt` ({int(self.params['duration']/self.params['dt'])})."
            )

        # if `sampling_dt` is set, do some checks
        if self.params.get("sampling_dt") is not None:
            # sample_dt checks are required after setting chunksize
            assert (
                chunksize * self.params["dt"] >= self.params["sampling_dt"]
            ), "`chunksize * dt` must be >= `sampling_dt`"

            # ugly floating point modulo hack: instead of float1%float2==0, we do
            # (float1/float2)%1==0
            assert ((chunksize * self.params["dt"]) / self.params["sampling_dt"]) % 1 == 0, (
                f"Chunksize {chunksize * self.params['dt']} must be divisible by sampling dt "
                f"{self.params['sampling_dt']}"
            )
            assert (
                (self.params["duration"] % (chunksize * self.params["dt"])) / self.params["sampling_dt"]
            ) % 1 == 0, (
                f"Last chunk of size {self.params['duration'] % (chunksize * self.params['dt'])} must be divisible by sampling dt "
                f"{self.params['sampling_dt']}"
            )

    def setSamplingDt(self):
        """Checks if sampling_dt is set correctly and sets self.`sample_every`
        1) Check if sampling_dt is multiple of dt
        2) Check if semplind_dt is greater than duration
        """

        if self.params.get("sampling_dt") is None:
            self.sample_every = 1
        elif self.params.get("sampling_dt") > 0:
            assert self.params["sampling_dt"] >= self.params["dt"], "`sampling_dt` needs to be >= `dt`"
            assert (
                self.params["duration"] >= self.params["sampling_dt"]
            ), "`sampling_dt` needs to be lower than `duration`"
            self.sample_every = int(self.params["sampling_dt"] / self.params["dt"])
        else:
            raise ValueError(f"Can't handle `sampling_dt`={self.params.get('sampling_dt')}")

    def initializeRun(self, initializeBold=False):
        """Initialization before each run.

        :param initializeBold: initialize BOLD model
        :type initializeBold: bool
        """
        # get the maxDelay of the system
        self.maxDelay = self.getMaxDelay()

        # length of the initial condition
        self.startindt = self.maxDelay + 1

        # check dt / sampling_dt
        self.setSamplingDt()

        # force bold if params['bold'] == True
        if self.params.get("bold"):
            initializeBold = True
        # set up the bold model, if it didn't happen yet
        if initializeBold and not self.boldInitialized:
            self.initializeBold()

    def run(
        self,
        inputs=None,
        chunkwise=False,
        chunksize=None,
        bold=False,
        append=False,
        append_outputs=None,
        continue_run=False,
    ):
        """
        Main interfacing function to run a model.

        The model can be run in three different ways:
        1) `model.run()` starts a new run.
        2) `model.run(chunkwise=True)` runs the simulation in chunks of length `chunksize`.
        3) `mode.run(continue_run=True)` continues the simulation of a previous run.

        :param inputs: list of inputs to the model, must have the same order as model.input_vars. Note: no sanity check is performed for performance reasons. Take care of the inputs yourself.
        :type inputs: list[np.ndarray|]
        :param chunkwise: simulate model chunkwise or in one single run, defaults to False
        :type chunkwise: bool, optional
        :param chunksize: size of the chunk to simulate in dt, if set will imply chunkwise=True, defaults to 2s
        :type chunksize: int, optional
        :param bold: simulate BOLD signal (only for chunkwise integration), defaults to False
        :type bold: bool, optional
        :param append: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append: bool, optional
        :param continue_run: continue a simulation by using the initial values from a previous simulation
        :type continue_run: bool
        """
        # TODO: legacy argument support
        if append_outputs is not None:
            append = append_outputs

        # if a previous run is not to be continued clear the model's state
        if continue_run is False:
            self.clearModelState()

        self.initializeRun(initializeBold=bold)

        # enable chunkwise if chunksize is set
        chunkwise = chunkwise if chunksize is None else True

        if chunkwise is False:
            self.integrate(append_outputs=append, simulate_bold=bold)
            if continue_run:
                self.setInitialValuesToLastState()

        else:
            if chunksize is None:
                chunksize = int(2000 / self.params["dt"])

            # check if model is safe for chunkwise integration
            # and whether sampling_dt is compatible with duration and chunksize
            self.checkChunkwise(chunksize)
            if bold and not self.boldInitialized:
                logging.warn(f"{self.name}: BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")
                bold = False
            self.integrateChunkwise(chunksize=chunksize, bold=bold, append_outputs=append)

        # check if there was a problem with the simulated data
        self.checkOutputs()

    def checkOutputs(self):
        # check nans in output
        if np.isnan(self.output).any():
            logging.error("nan in model output!")
        else:
            EXPLOSION_THRESHOLD = 1e20
            if (self.output > EXPLOSION_THRESHOLD).any() > 0:
                logging.error("nan in model output!")

        # check nans in BOLD
        if "BOLD" in self.outputs:
            if np.isnan(self.outputs.BOLD.BOLD).any():
                logging.error("nan in BOLD output!")

    def integrate(self, append_outputs=False, simulate_bold=False):
        """Calls each models `integration` function and saves the state and the outputs of the model.

        :param append: append the chunkwise outputs to the outputs attribute, defaults to False, defaults to False
        :type append: bool, optional
        """
        # run integration
        t, *variables = self.integration(self.params)
        self.storeOutputsAndStates(t, variables, append=append_outputs)

        # force bold if params['bold'] == True
        if self.params.get("bold"):
            simulate_bold = True

        # bold simulation after integration
        if simulate_bold and self.boldInitialized:
            self.simulateBold(t, variables, append=True)

    def integrateChunkwise(self, chunksize, bold=False, append_outputs=False):
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
        while totalDuration - lastT >= dt - 1e-6:
            # Determine the size of the next chunk
            # account for floating point errors
            remainingChunkSize = int(round((totalDuration - lastT) / dt))
            currentChunkSize = min(chunksize, remainingChunkSize)

            self.autochunk(chunksize=currentChunkSize, append_outputs=append_outputs, bold=bold)
            # we save the last simulated time step
            lastT += currentChunkSize * dt
            # or
            # lastT = self.state["t"][-1]

        # set duration back to its original value
        self.params["duration"] = totalDuration

    def clearModelState(self):
        """Clears the model's state to create a fresh one"""
        self.state = dotdict({})
        self.outputs = dotdict({})
        # reinitialize bold model
        if self.params.get("bold"):
            self.initializeBold()

    def storeOutputsAndStates(self, t, variables, append=False):
        """Takes the simulated variables of the integration and stores it to the appropriate model output and state object.

        :param t: time vector
        :type t: list
        :param variables: variable from time integration
        :type variables: numpy.ndarray
        :param append: append output to existing output or overwrite, defaults to False
        :type append: bool, optional
        """
        # save time array
        self.setOutput("t", t, append=append, removeICs=True)
        self.setStateVariables("t", t)
        # save outputs
        for svn, sv in zip(self.state_vars, variables):
            if svn in self.output_vars:
                self.setOutput(svn, sv, append=append, removeICs=True)
            self.setStateVariables(svn, sv)

    def setInitialValuesToLastState(self):
        """Reads the last state of the model and sets the initial conditions to that state for continuing a simulation."""
        for iv, sv in zip(self.init_vars, self.state_vars):
            # if state variables are one-dimensional (in space only)
            if (self.state[sv].ndim == 0) or (self.state[sv].ndim == 1):
                self.params[iv] = self.state[sv]
            # if they are space-time arrays
            else:
                # we set the next initial condition to the last state
                self.params[iv] = self.state[sv][:, -self.startindt :]

    def randomICs(self, min=0, max=1):
        """Generates a new set of uniformly-distributed random initial conditions for the model.

        TODO: All parameters are drawn from the same distribution / range. Allow for independent ranges.

        :param min: Minium of uniform distribution
        :type min: float
        :param max: Maximum of uniform distribution
        :type max: float
        """
        for iv in self.init_vars:
            if self.params[iv].ndim == 1:
                self.params[iv] = np.random.uniform(min, max, (self.params["N"]))
            elif self.params[iv].ndim == 2:
                self.params[iv] = np.random.uniform(min, max, (self.params["N"], 1))

    def setInputs(self, inputs):
        """Take inputs from a list and store it in the appropriate model parameter for external input.
        TODO: This is not safe yet, checks should be implemented whether the model has inputs defined or not for example.

        :param inputs: list of inputs
        :type inputs: list[np.ndarray(), ...]
        """
        for i, iv in enumerate(self.input_vars):
            self.params[iv] = inputs[i].copy()

    def autochunk(self, inputs=None, chunksize=1, append_outputs=False, bold=False):
        """Executes a single chunk of integration, either for a given duration
        or a single timestep `dt`. Gathers all inputs to the model and resets
        the initial conditions as a preparation for the next chunk.

        :param inputs: list of input values, ordered according to self.input_vars, defaults to None
        :type inputs: list[np.ndarray|], optional
        :param chunksize: length of a chunk to simulate in dt, defaults 1
        :type chunksize: int, optional
        :param append_outputs: append the chunkwise outputs to the outputs attribute, defaults to False
        :type append_outputs: bool, optional
        """

        # set the duration for this chunk
        self.params["duration"] = chunksize * self.params["dt"]

        # set inputs
        if inputs is not None:
            self.setInputs(inputs)

        # run integration
        self.integrate(append_outputs=append_outputs, simulate_bold=bold)

        # set initial conditions to last state for the next chunk
        self.setInitialValuesToLastState()

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
        dt = self.params.get("dt")
        Dmat = self.params.get("lengthMat")

        if Dmat is not None:
            # divide Dmat by signalV
            signalV = self.params.get("signalV") or 0
            if signalV > 0:
                Dmat = Dmat / signalV
            else:
                # if signalV is 0, eliminate delays
                Dmat = Dmat * 0.0

        # only if Dmat and dt exist, a global max delay can be computed
        if Dmat is not None and dt is not None:
            Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
            max_global_delay = int(np.amax(Dmat_ndt))
        else:
            max_global_delay = 0
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
        # old
        # self.state[name] = data.copy()

        # if the data is temporal, cut off initial values
        # NOTE: this shuold actually check for
        # if data.shape[1] > 1:
        # else: data.copy()
        # there coulb be (N, 1)-dimensional output, right now
        # it is requred to be of shape (N, )
        if data.ndim == 2:
            self.state[name] = data[:, -self.startindt :].copy()
        else:
            self.state[name] = data.copy()

    def setOutput(self, name, data, append=False, removeICs=False):
        """Adds an output to the model, typically a simulation result.
        :params name: Name of the output in dot.notation, a la "outputgroup.output"
        :type name: str
        :params data: Output data, can't be a dictionary!
        :type data: `numpy.ndarray`
        """
        assert not isinstance(data, dict), "Output data cannot be a dictionary."
        assert isinstance(name, str), "Output name must be a string."
        assert isinstance(data, np.ndarray), "Output must be a `numpy.ndarray`."

        # remove initial conditions from output
        if removeICs and name != "t":
            if data.ndim == 1:
                data = data[self.startindt :]
            elif data.ndim == 2:
                data = data[:, self.startindt :]
            else:
                raise ValueError(f"Don't know how to truncate data of shape {data.shape}.")

        # subsample to sampling dt
        if data.ndim == 1:
            data = data[:: self.sample_every]
        elif data.ndim == 2:
            data = data[:, :: self.sample_every]
        else:
            raise ValueError(f"Don't know how to subsample data of shape {data.shape}.")

        # if the output is a single name (not dot.separated)
        if "." not in name:
            # append data
            if append and name in self.outputs:
                # special treatment for time data:
                # increment the time by the last recorded duration
                if name == "t":
                    data += self.outputs[name][-1]
                self.outputs[name] = np.hstack((self.outputs[name], data))
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
                    # TODO: this needs to be append-aware like above
                    # if append:
                    #     if k == "t":
                    #         data += level[k][-1]
                    #     level[k] = np.hstack((level[k], data))
                    # else:
                    #     level[k] = data
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
        """Index outputs with a dictionary-like key, e.g., `model['rates_exc']`."""
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
        """Returns value of default output as defined by `self.default_output`.
        Note that all outputs are saved in the attribute `self.outputs`.
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
