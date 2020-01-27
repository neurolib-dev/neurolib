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
        logging.info(
            "Number of parameter configurations: {}".format(
                len(
                    self.pypetParametrization[list(self.pypetParametrization.keys())[0]]
                )
            )
        )

        # create hdf file path if it does not exist yet
        pathlib.Path(paths.HDF_DIR).mkdir(parents=True, exist_ok=True)

        # set default hdf filename
        self.HDF_FILE = os.path.join(paths.HDF_DIR, "exploration.hdf")

        # todo: use random ICs for every explored point or rather reuse the ones that are generated at model initialization
        self.useRandomICs = False

        # bool to check whether pypet was initialized properly
        self.initialized = False

    def initializeExploration(
        self, explorationName="exploration", fileName="exploration.hdf"
    ):
        # ---- initialize pypet environment ----
        trajectoryName = (
            "results"
            + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
            + "-"
            + explorationName
        )
        self.HDF_FILE = os.path.join(paths.HDF_DIR, fileName)
        trajectoryFileName = self.HDF_FILE

        nprocesses = multiprocessing.cpu_count()
        logging.info("Number of processes: {}".format(nprocesses))

        # set up the pypet environment
        env = pypet.Environment(
            trajectory=trajectoryName,
            filename=trajectoryFileName,
            file_title=explorationName,
            large_overview_tables=True,
            multiproc=True,
            ncores=nprocesses,
            # log_config=None,
            wrap_mode="QUEUE",
            log_stdout=False,
        )
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

    def runModel(self, traj):
        """This function will be called by pypet directly and therefore 
        wants a trajectory `traj` as an argument
        """
        if self.useRandomICs:
            logging.warn("Random initial conditions not implemented yet")

        # lambda function for an dictionary update without inplace replacement
        # (lambda d: d.update(newDict) or d)(oldDict)

        # get parameters of this run from pypet trajectory
        runParams = traj.parameters.f_to_dict(short_names=True, fast_access=True)

        # set the parameters for the model
        self.model.params.update(runParams)

        # run it
        self.model.run()

        # get retuls from the model
        # attention: this is not model-agnostic yet, only works with aln model
        # should return a `result` object rather than each dimension
        t, rates_exc, rates_inh = (
            self.model.t,
            self.model.rates_exc,
            self.model.rates_inh,
        )

        traj.f_add_result("results.$", t=t, rates_exc=rates_exc, rates_inh=rates_inh)

    def run(self):
        """
        Call this function to run the exploration
        """
        assert self.initialized, "Pypet environment not initialized yet."
        self.env.run(self.runModel)

    def getTrajectorynamesInFile(self, filename):
        """
        Return a list of all pypet trajectories name saved in a a given hdf5 file.

        Parameter:
            :param filename:   Name of the hdf5 we want to explore

        Return:
            List of string containing the trajectory name
        """
        hdf = h5py.File(filename)
        all_traj_names = list(hdf.keys())
        hdf.close()
        return all_traj_names

    def loadPypetTrajectory(self, filename, trajectoryName):
        """Read HDF file with simulation results and return the chosen trajectory.

        :param filename: HDF file path
        :type filename: str

        :return: pypet trajectory
        """
        if filename == None:
            filename = self.HDF_FILE

        assert pathlib.Path(filename).exists(), f"{filename} does not exist!"
        logging.info(f"Loading results from {filename}")

        # if trajectoryName is not specified, load the most resent trajectory
        if trajectoryName == None:
            trajectoryName = self.getTrajectorynamesInFile(filename)[-1]
        logging.info(f"Analyzing trajectory {trajectoryName}")

        trajLoaded = pypet.Trajectory(trajectoryName, add_time=False)
        trajLoaded.f_load(trajectoryName, filename=filename, force=True)
        trajLoaded.v_auto_load = True
        return trajLoaded

    def getResults(self, filename=None, trajectoryName=None):
        """
        Load simulation results.
        """
        trajLoaded = self.loadPypetTrajectory(filename, trajectoryName)
        nResults = len(trajLoaded.f_get_run_names())

        # this is very wonky, might break if nested parameters are used!
        dt = trajLoaded.f_get_parameters()["parameters.dt"].f_get()

        exploredParameters = trajLoaded.f_get_explored_parameters()
        # split dot-separated pypet parameters back
        niceParKeys = [p.split(".")[-1] for p in exploredParameters.keys()]

        # ---- lcreate pandas df with results as keys ----
        self.dfResults = pd.DataFrame(columns=niceParKeys, dtype=object)

        # range of parameters
        for nicep, p in zip(niceParKeys, exploredParameters.keys()):
            self.dfResults[nicep] = exploredParameters[p].f_get_range()

        # ---- make a dictionary with results ----
        resultDicts = []
        print("Creating results dictionary ...")
        self.runResults = []
        for rInd in tqdm.tqdm(range(len(self.dfResults)), total=len(self.dfResults)):
            result = trajLoaded.results[rInd].f_to_dict()
            self.runResults.append(result)
        print("done.")

