import multiprocessing
import datetime
import os
import logging
import pathlib

import h5py
import pypet
import pandas as pd
import tqdm

import neurolib.utils.paths as paths
import neurolib.utils.pypetUtils as pu


class BoxSearch:
    """
    Paremter box search for a given model and a range of parameters.
    """

    def __init__(self, model, exploreParameters):
        """Defines the model to explore and the parameter range

        :param model: Model to explore
        :type model: Model

        :param exploreParameters: Parameter dictionary with each exploration parameter as a key and a list of all parameter values as a value
        :type param: dict
        """
        self.model = model

        self.exploreParameters = exploreParameters
        # pypet interaction starts here
        self.pypetParametrization = pypet.cartesian_product(exploreParameters)
        logging.info("Number of parameter configurations: {}".format(len(self.pypetParametrization[list(self.pypetParametrization.keys())[0]])))

        # create hdf file path if it does not exist yet
        pathlib.Path(paths.HDF_DIR).mkdir(parents=True, exist_ok=True)

        # set default hdf filename
        self.HDF_FILE = os.path.join(paths.HDF_DIR, "exploration.hdf")

        # todo: use random ICs for every explored point or rather reuse the ones that are generated at model initialization
        self.useRandomICs = False

        # bool to check whether pypet was initialized properly
        self.initialized = False

    def initializeExploration(self, fileName="exploration.hdf"):
        # ---- initialize pypet environment ----
        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        self.HDF_FILE = os.path.join(paths.HDF_DIR, fileName)
        trajectoryFileName = self.HDF_FILE

        nprocesses = multiprocessing.cpu_count()
        logging.info("Number of processes: {}".format(nprocesses))

        # set up the pypet environment
        env = pypet.Environment(trajectory=trajectoryName, filename=trajectoryFileName, multiproc=True, ncores=nprocesses, complevel=9, log_stdout=False)
        self.env = env
        # Get the trajectory from the environment
        self.traj = env.trajectory
        self.trajectoryName = self.traj.v_name

        # Add all parameters to the pypet trajectory
        self.addParametersToPypet(self.traj, self.model.params)
        # Tell pypet which parameters to explore
        self.traj.f_explore(self.pypetParametrization)

        # initialization done
        logging.info("Pypet environment initialized.")
        self.initialized = True

    def addParametersToPypet(self, traj, params):
        """This function registers the parameters of the model to Pypet.
        Parameters can be nested dictionaries. They are unpacked here recursively.
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
                    traj.f_add_parameter(param_address, value)

        addParametersRecursively(traj, params, [])

    def saveOutputsToPypet(self, outputs, traj):
        """This function takes all outputs in the form of a nested dictionary
        and stores all data into the pypet hdf file.
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

        value = outputs
        savestr = "results.$."
        makeSaveStringForPypet(value, savestr)
        # print(outputs["rates_exc"])
        # traj.f_add_result("rates_exc", outputs["rates_exc"])
        # traj.f_add_result("results.$", rates_exc=outputs["rates_exc"], t=outputs["t"])
        # traj.f_add_result('results.$', t = t, rates_exc = rates_exc)

    def runModel(self, traj):
        """This function will be called by pypet directly and therefore 
        wants a trajectory `traj` as an argument
        """
        if self.useRandomICs:
            logging.warn("Random initial conditions not implemented yet")
        # get parameters of this run from pypet trajectory
        runParams = traj.parameters.f_to_dict(short_names=True, fast_access=True)
        # set the parameters for the model
        self.model.params.update(runParams)
        # run it
        self.model.run()
        # save all results from exploration
        self.saveOutputsToPypet(self.model.outputs, traj)

    def run(self):
        """
        Call this function to run the exploration
        """
        assert self.initialized, "Pypet environment not initialized yet."
        self.env.run(self.runModel)

    def loadResults(self, filename=None, trajectoryName=None):
        """
        Load simulation results an hdf file.
        """
        # chose
        if filename == None:
            filename = self.HDF_FILE
        trajLoaded = pu.loadPypetTrajectory(filename, trajectoryName)
        self.nResults = len(trajLoaded.f_get_run_names())

        exploredParameters = trajLoaded.f_get_explored_parameters()

        # create pandas dataframe of all runs with parameters as keys
        logging.info("Creating pandas dataframe ...")
        niceParKeys = [p.split(".")[-1] for p in exploredParameters.keys()]
        self.dfResults = pd.DataFrame(columns=niceParKeys, dtype=object)
        for nicep, p in zip(niceParKeys, exploredParameters.keys()):
            self.dfResults[nicep] = exploredParameters[p].f_get_range()

        # make a list of dictionaries with results
        logging.info("Creating results dictionary ...")
        self.results = []
        for rInd in tqdm.tqdm(range(self.nResults), total=self.nResults):
            trajLoaded.results[rInd].f_load()
            result = trajLoaded.results[rInd].f_to_dict(fast_access=True, short_names=True)
            trajLoaded.results[rInd].f_remove()
            self.results.append(result)
        logging.info("All results loaded.")
