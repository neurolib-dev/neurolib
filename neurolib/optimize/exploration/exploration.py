import copy
import datetime
import logging
import multiprocessing
import os
import pathlib

import numpy as np
import pandas as pd
import psutil
import pypet
import tqdm
import xarray as xr

from ...utils import paths
from ...utils import pypetUtils as pu
from ...utils.collections import dotdict, flat_dict_to_nested, flatten_nested_dict, unwrap_star_dotdict


class BoxSearch:
    """
    Paremter box search for a given model and a range of parameters.
    """

    def __init__(self, model=None, parameterSpace=None, evalFunction=None, filename=None, saveAllModelOutputs=False):
        """Either a model has to be passed, or an evalFunction. If an evalFunction
        is passed, then the evalFunction will be called and the model is accessible to the
        evalFunction via `self.getModelFromTraj(traj)`. The parameters of the current
        run are accible via `self.getParametersFromTraj(traj)`.

        If no evaluation function is passed, then the model is simulated using `Model.run()`
        for every parameter.

        :param model: Model to run for each parameter (or model to pass to the evaluation funciton if an evaluation
            function is used), defaults to None
        :type model: `neurolib.models.model.Model`, optional
        :param parameterSpace: Parameter space to explore, defaults to None
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`, optional
        :param evalFunction: Evaluation function to call for each run., defaults to None
        :type evalFunction: function, optional
        :param filename: HDF5 storage file name, if left empty, defaults to ``exploration.hdf``
        :type filename: str
        :param saveAllModelOutputs: If True, save all outputs of model, else only default output of the model will be
            saved. Note: if saveAllModelOutputs==False and the model's parameter model.params['bold']==True, then BOLD
            output will be saved as well, defaults to False
        :type saveAllModelOutputs: bool
        """
        self.model = model
        if evalFunction is None and model is not None:
            self.evalFunction = self.runModel
        elif evalFunction is not None:
            self.evalFunction = evalFunction

        assert (evalFunction is not None) or (
            model is not None
        ), "Either a model has to be specified or an evalFunction."

        assert parameterSpace is not None, "No parameters to explore."

        self.parameterSpace = parameterSpace
        self.exploreParameters = parameterSpace.dict()

        # TODO: use random ICs for every explored point or rather reuse the ones that are generated at model
        # initialization
        self.useRandomICs = False

        filename = filename or "exploration.hdf"
        self.filename = filename

        self.saveAllModelOutputs = saveAllModelOutputs

        # bool to check whether pypet was initialized properly
        self.initialized = False
        self.initializeExploration(self.filename)

        self.results = None

    def initializeExploration(self, filename="exploration.hdf"):
        """Initialize the pypet environment

        :param filename: hdf filename to store the results in , defaults to "exploration.hdf"
        :type filename: str, optional
        """
        # create hdf file path if it does not exist yet
        pathlib.Path(paths.HDF_DIR).mkdir(parents=True, exist_ok=True)

        # set default hdf filename
        self.HDF_FILE = os.path.join(paths.HDF_DIR, filename)

        # initialize pypet environment
        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        trajectoryfilename = self.HDF_FILE

        nprocesses = multiprocessing.cpu_count()
        logging.info("Number of processes: {}".format(nprocesses))

        # set up the pypet environment
        env = pypet.Environment(
            trajectory=trajectoryName,
            filename=trajectoryfilename,
            multiproc=True,
            ncores=nprocesses,
            complevel=9,
            log_config=paths.PYPET_LOGGING_CONFIG,
        )
        self.env = env
        # Get the trajectory from the environment
        self.traj = env.trajectory
        self.trajectoryName = self.traj.v_name

        # Add all parameters to the pypet trajectory
        if self.model is not None:
            # if a model is specified, use the default parameter of the
            # model to initialize pypet
            self.addParametersToPypet(self.traj, self.model.params)
        else:
            # else, use a random parameter of the parameter space
            self.addParametersToPypet(self.traj, self.parameterSpace.getRandom(safe=True))

        # Tell pypet which parameters to explore
        self.pypetParametrization = pypet.cartesian_product(self.exploreParameters)
        # explicitely add all parameters within star notation, hence unwrap star notation into actual params names
        if self.parameterSpace.star:
            assert self.model is not None, "With star notation, model cannot be None"
            self.pypetParametrization = unwrap_star_dotdict(self.pypetParametrization, self.model)
        self.nRuns = len(self.pypetParametrization[list(self.pypetParametrization.keys())[0]])
        logging.info(f"Number of parameter configurations: {self.nRuns}")

        self.traj.f_explore(self.pypetParametrization)

        # initialization done
        logging.info("BoxSearch: Environment initialized.")
        self.initialized = True

    def addParametersToPypet(self, traj, params):
        """This function registers the parameters of the model to Pypet.
        Parameters can be nested dictionaries. They are unpacked and stored recursively.

        :param traj: Pypet trajectory to store the parameters in
        :type traj: `pypet.trajectory.Trajectory`
        :param params: Parameter dictionary
        :type params: dict, dict[dict,]
        """

        def addParametersRecursively(traj, params, current_level):
            # make dummy list if just string
            if isinstance(current_level, str):
                current_level = [current_level]
            # iterate dict
            for key, value in params.items():
                # if another dict - recurse and increase level
                if isinstance(value, dict):
                    addParametersRecursively(traj, value, current_level + [key])
                else:
                    param_address = ".".join(current_level + [key])
                    value = "None" if value is None else value
                    traj.f_add_parameter(param_address, value)

        addParametersRecursively(traj, params, [])

    # TODO: remove legacy
    def saveOutputsToPypet(self, outputs, traj):
        return self.saveToPypet(outputs, traj)

    def saveToPypet(self, outputs, traj):
        """This function takes simulation results in the form of a nested dictionary
        and stores all data into the pypet hdf file.

        :param outputs: Simulation outputs as a dictionary.
        :type outputs: dict
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """

        def makeSaveStringForPypet(value, savestr):
            """Builds the pypet-style results string from the results
            dictionary's keys.
            """
            for k, v in value.items():
                if isinstance(v, dict):
                    _savestr = savestr + k + "."
                    makeSaveStringForPypet(v, _savestr)
                else:
                    _savestr = savestr + k
                    self.traj.f_add_result(_savestr, v)

        assert isinstance(outputs, dict), "Outputs must be an instance of dict."
        value = outputs
        savestr = "results.$."
        makeSaveStringForPypet(value, savestr)

    def runModel(self, traj):
        """If not evaluation function is given, we assume that a model will be simulated.
        This function will be called by pypet directly and therefore wants a pypet trajectory as an argument

        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """
        if self.useRandomICs:
            logging.warn("Random initial conditions not implemented yet")
        # get parameters of this run from pypet trajectory
        runParams = self.getParametersFromTraj(traj)
        if self.parameterSpace.star:
            runParams = flatten_nested_dict(flat_dict_to_nested(runParams)["parameters"])

        # set the parameters for the model
        self.model.params.update(runParams)

        # get kwargs from Exploration.run()
        runKwargs = {}
        if hasattr(self, "runKwargs"):
            runKwargs = self.runKwargs
        # run it
        self.model.run(**runKwargs)
        # save outputs
        self.saveModelOutputsToPypet(traj)

    def saveModelOutputsToPypet(self, traj):
        # save all data to the pypet trajectory
        if self.saveAllModelOutputs:
            # save all results from exploration
            self.saveToPypet(self.model.outputs, traj)
        else:
            # save only the default output
            self.saveToPypet(
                {
                    self.model.default_output: self.model.output,
                    "t": self.model.outputs["t"],
                },
                traj,
            )
            # save BOLD output
            # if "bold" in self.model.params:
            #     if self.model.params["bold"] and "BOLD" in self.model.outputs:
            #         self.saveToPypet(self.model.outputs["BOLD"], traj)
            if "BOLD" in self.model.outputs:
                self.saveToPypet(self.model.outputs["BOLD"], traj)

    def validatePypetParameters(self, runParams):
        """Helper to handle None's in pypet parameters
        (used for random number generator seed)

        :param runParams: parameters as returned by traj.parameters.f_to_dict()
        :type runParams: dict of pypet.parameter.Parameter
        """

        # fix rng seed, which is saved as a string if None
        if "seed" in runParams:
            if runParams["seed"] == "None":
                runParams["seed"] = None
        return runParams

    def getParametersFromTraj(self, traj):
        """Returns the parameters of the current run as a (dot.able) dictionary

        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Parameter set of the current run
        :rtype: dict
        """
        # DO NOT use short names for star notation dicts
        runParams = self.traj.parameters.f_to_dict(short_names=not self.parameterSpace.star, fast_access=True)
        runParams = self.validatePypetParameters(runParams)
        return dotdict(runParams)

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this run
        :params traj: Pypet trajectory of current run

        :returns model: Model with the parameters of this run.
        """
        model = self.model
        runParams = self.getParametersFromTraj(traj)

        model.params.update(runParams)
        return model

    def run(self, **kwargs):
        """
        Call this function to run the exploration
        """
        self.runKwargs = kwargs
        assert self.initialized, "Pypet environment not initialized yet."
        self._t_start_exploration = datetime.datetime.now()
        self.env.run(self.evalFunction)
        self._t_end_exploration = datetime.datetime.now()

    def loadResults(self, all=True, filename=None, trajectoryName=None, pypetShortNames=True, memory_cap=95.0):
        """Load results from a hdf file of a previous simulation.

        :param all: Load all simulated results into memory, which will be available as the `.results` attribute. Can
            use a lot of RAM if your simulation is large, please use this with caution. , defaults to True
        :type all: bool, optional
        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults
            to None
        :type trajectoryName: str, optional
        :param pypetShortNames: Use pypet short names as keys for the results dictionary. Use if you are experiencing
            errors due to natural naming collisions.
        :type pypetShortNames: bool
        :param memory_cap: Percentage memory cap between 0 and 100. If `all=True` is used, a memory cap can be set to
            avoid filling up the available RAM. Example: use `memory_cap = 95` to avoid loading more data if memory is
            at 95% use, defaults to 95
        :type memory_cap: float, int, optional
        """

        self.loadDfResults(filename, trajectoryName)

        # make a list of dictionaries with results
        self.results = dotdict({})
        if all:
            logging.info("Loading all results to `results` dictionary ...")
            for rInd in tqdm.tqdm(range(self.nResults), total=self.nResults):

                # check if enough memory is available
                if memory_cap:
                    assert isinstance(memory_cap, (int, float)), "`memory_cap` must be float."
                    assert (memory_cap > 0) and (memory_cap < 100), "`memory_cap` must be between 0 and 100"
                    # check ram usage with psutil
                    used_memory_percent = psutil.virtual_memory()[2]
                    if used_memory_percent > memory_cap:
                        raise MemoryError(
                            f"Memory use is at {used_memory_percent}% and capped at {memory_cap}. Aborting."
                        )

                self.pypetTrajectory.results[rInd].f_load()
                result = self.pypetTrajectory.results[rInd].f_to_dict(fast_access=True, short_names=pypetShortNames)
                result = dotdict(result)
                self.pypetTrajectory.results[rInd].f_remove()
                self.results[rInd] = copy.deepcopy(result)

            # Postprocess result keys if pypet short names aren't used
            # Before: results.run_00000001.outputs.rates_inh
            # After: outputs.rates_inh
            if not pypetShortNames:
                for i, r in self.results.items():
                    new_dict = dotdict({})
                    for key, value in r.items():
                        new_key = "".join(key.split(".", 2)[2:])
                        new_dict[new_key] = r[key]
                    self.results[i] = copy.deepcopy(new_dict)

        self.aggregateResultsToDfResults()

        logging.info("All results loaded.")

    def aggregateResultsToDfResults(self, arrays=True, fillna=False):
        """Aggregate all results in to dfResults dataframe.

        :param arrays: Load array results (like timeseries) if True. If False, only load scalar results, defaults to
            True
        :type arrays: bool, optional
        :param fillna: Fill nan results (for example if they're not returned in a subset of runs) with zeros, default
            to False
        :type fillna: bool, optional
        """
        nan_value = np.nan
        logging.info("Aggregating results to `dfResults` ...")
        # for i, result in tqdm.tqdm(self.results.items()):

        for runId, parameters in tqdm.tqdm(self.dfResults.iterrows(), total=len(self.dfResults)):
            # if the results were previously loaded into memory, use them
            if hasattr(self, "results"):
                # only if the length matches the number of results
                if len(self.results) == len(self.dfResults):
                    result = self.results[runId]
                # else, load results individually from hdf file
                else:
                    result = self.getRun(runId)
            # else, load results individually from hdf file
            else:
                result = self.getRun(runId)

            for key, value in result.items():
                # only save floats, ints and arrays
                if isinstance(value, (float, int, np.ndarray)):
                    # save 1-dim arrays
                    if isinstance(value, np.ndarray) and arrays:
                        # to save a numpy array, convert column to object type
                        if key not in self.dfResults:
                            self.dfResults[key] = None
                        self.dfResults[key] = self.dfResults[key].astype(object)
                        self.dfResults.at[runId, key] = value
                    elif isinstance(value, (float, int)):
                        # save numbers
                        self.dfResults.loc[runId, key] = value
                else:
                    self.dfResults.loc[runId, key] = nan_value
        # drop nan columns
        self.dfResults = self.dfResults.dropna(axis="columns", how="all")

        if fillna:
            self.dfResults = self.dfResults.fillna(0)

    def loadDfResults(self, filename=None, trajectoryName=None):
        """Load results from a previous simulation.

        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults
            to None
        :type trajectoryName: str, optional
        """
        # chose HDF file to load
        filename = filename or self.HDF_FILE
        self.pypetTrajectory = pu.loadPypetTrajectory(filename, trajectoryName)
        self.nResults = len(self.pypetTrajectory.f_get_run_names())

        exploredParameters = self.pypetTrajectory.f_get_explored_parameters()

        # create pandas dataframe of all runs with parameters as keys
        logging.info("Creating `dfResults` dataframe ...")
        niceParKeys = [p[11:] for p in exploredParameters.keys()]
        if not self.parameterSpace:
            niceParKeys = [p.split(".")[-1] for p in niceParKeys]
        self.dfResults = pd.DataFrame(columns=niceParKeys, dtype=object)
        for nicep, p in zip(niceParKeys, exploredParameters.keys()):
            self.dfResults[nicep] = exploredParameters[p].f_get_range()

    @staticmethod
    def _filt_dict_bold(filt_dict, bold):
        """
        Filters result dictionary: either keeps ONLY BOLD results, or remove
        BOLD results.

        :param filt_dict: dictionary to filter for BOLD keys
        :type filt_dict: dict
        :param bold: whether to remove BOLD keys (bold=False) or keep only BOLD
            keys (bold=True)
        :return: filtered dict, without or only BOLD keys
        :rtype: dict
        """
        filt_dict = copy.deepcopy(filt_dict)
        if bold:
            return {k: v for k, v in filt_dict.items() if "BOLD" in k}
        else:
            return {k: v for k, v in filt_dict.items() if "BOLD" not in k}

    def _get_coords_from_run(self, run_dict, bold=False):
        """
        Find coordinates of a single run - time, output and space dimensions.

        :param run_dict: dictionary with run results
        :type run_dict: dict
        :param bold: whether to do only BOLD or without BOLD results
        :type bold: bool
        :return: dictionary of coordinates for xarray
        :rtype: dict
        """
        run_dict = copy.deepcopy(run_dict)
        run_dict = self._filt_dict_bold(run_dict, bold=bold)
        timeDictKey = ""
        if "t" in run_dict:
            timeDictKey = "t"
        else:
            for k in run_dict:
                if k.startswith("t"):
                    timeDictKey = k
                    logging.info(f"Assuming {k} to be the time axis.")
                    break
        assert len(timeDictKey) > 0, "No time array found (starting with t) in model output."
        t = run_dict[timeDictKey].copy()
        del run_dict[timeDictKey]
        return timeDictKey, {
            "output": list(run_dict.keys()),
            "space": list(range(next(iter(run_dict.values())).shape[0])),
            "time": t,
        }

    def xr(self, bold=False):
        """
        Return xr.Dataset from the exploration. The coordinates are in the star
        notation if applicable. The whole list of affected model parameters by
        star notated exploration is listed in attributes.

        :param bold: if True, will load and return only BOLD output
        :type bold: bool
        """
        assert self.results is not None, "Run `loadResults()` first to populate the results"
        assert len(self.results) == len(self.dfResults)
        # create intrisinsic dims for one run
        timeDictKey, run_coords = self._get_coords_from_run(self.results[0], bold=bold)
        dataarrays = []
        orig_search_coords = pypet.cartesian_product(self.exploreParameters)
        for runId, run_result in self.results.items():
            # take exploration coordinates for this run
            expl_coords = {k: v[runId] for k, v in orig_search_coords.items()}
            outputs = []
            run_result = self._filt_dict_bold(run_result, bold=bold)
            for key, value in run_result.items():
                if key == timeDictKey:
                    continue
                outputs.append(value)
            # create DataArray for run only - we need to add exploration coordinates
            data_temp = xr.DataArray(
                np.stack(outputs), dims=["output", "space", "time"], coords=run_coords, name="exploration"
            )
            expand_coords = {}
            # iterate exploration coordinates
            for k, v in expl_coords.items():
                # if single values, just assing
                if isinstance(v, (str, float, int)):
                    expand_coords[k] = [v]
                # if arrays, check whether they can be sqeezed into one value
                elif isinstance(v, np.ndarray):
                    if np.unique(v).size == 1:
                        # if yes, just assing that one value
                        expand_coords[k] = [float(np.unique(v))]
                    else:
                        # if no, sorry - coordinates cannot be array
                        raise ValueError("Cannot squeeze coordinates")
            # assing exploration coordinates to the DataArray
            dataarrays.append(data_temp.expand_dims(expand_coords))

        # finally, combine all arrays into one
        combined = xr.combine_by_coords(dataarrays)["exploration"]
        if self.parameterSpace.star:
            combined.attrs = {k: list(self.model.params[k].keys()) for k in orig_search_coords.keys()}

        return combined

    def getRun(self, runId, filename=None, trajectoryName=None, pypetShortNames=True):
        """Load the simulated data of a run and its parameters from a pypetTrajectory.

        :param runId: ID of the run
        :type runId: int

        :return: Dictionary with simulated data and parameters of the run.
        :type return: dict
        """
        # chose HDF file to load
        filename = self.HDF_FILE or filename

        # either use loaded pypetTrajectory or load from HDF file if it isn't available
        pypetTrajectory = (
            self.pypetTrajectory
            if hasattr(self, "pypetTrajectory")
            else pu.loadPypetTrajectory(filename, trajectoryName)
        )

        # # if there was no pypetTrajectory loaded before
        # if pypetTrajectory is None:
        #     # chose HDF file to load
        #     filename = self.HDF_FILE or filename
        #     pypetTrajectory = pu.loadPypetTrajectory(filename, trajectoryName)

        return pu.getRun(runId, pypetTrajectory, pypetShortNames=pypetShortNames)

    def getResult(self, runId):
        """Returns either a loaded result or reads from disk.

        :param runId: runId of result
        :type runId: int
        :return: result
        :rtype: dict
        """
        # if hasattr(self, "results"):
        #     # load result from either the preloaded .result attribute (from .loadResults)
        #     result = self.results[runId]
        # else:
        #     # or from disk if results haven't been loaded yet
        #     result = self.getRun(runId)

        # load result from either the preloaded .result attribute (from .loadResults)
        # or from disk if results haven't been loaded yet
        # result = self.results[runId] if hasattr(self, "results") else self.getRun(runId)
        return self.results[runId] if hasattr(self, "results") else self.getRun(runId)

    def info(self):
        """Print info about the current search."""
        now = datetime.datetime.now().strftime("%Y-%m-%d-%HH-%MM-%SS")
        print(f"Exploration info ({now})")
        print(f"HDF name: {self.HDF_FILE}")
        print(f"Trajectory name: {self.trajectoryName}")
        if self.model is not None:
            print(f"Model: {self.model.name}")
        if hasattr(self, "nRuns"):
            print(f"Number of runs {self.nRuns}")
        print(f"Explored parameters: {self.exploreParameters.keys()}")
        if hasattr(self, "_t_end_exploration") and hasattr(self, "_t_start_exploration"):
            print(f"Duration of exploration: {self._t_end_exploration-self._t_start_exploration}")
