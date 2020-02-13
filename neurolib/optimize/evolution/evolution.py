import datetime
import os
import logging
import multiprocessing
import sys

import deap
from deap import base
from deap import creator
from deap import tools

import numpy as np
import pypet as pp
import pandas as pd

import neurolib.utils.paths as paths

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.optimize.evolution.deapUtils as du
import neurolib.utils.pypetUtils as pu


class Evolution:
    """Evolutionary parameter optimization. This class helps you to optimize any function or model using an evlutionary algorithm. 
    It uses the package `deap` and supports its builtin mating and selection functions as well as custom ones. 
    """

    def __init__(
        self,
        evalFunction,
        parameterSpace,
        weightList=None,
        model=None,
        hdf_filename="evolution.hdf",
        ncores=None,
        POP_INIT_SIZE=100,
        POP_SIZE=20,
        NGEN=10,
        matingFunction=None,
        CXP=0.5,
        selectionFunction=None,
        RANKP=1.5,
    ):
        """Initialize evolutionary optimization.
        :param evalFunction: Evaluation function of a run that provides a fitness vector and simulation outputs
        :type evalFunction: function
        :param parameterSpace: Parameter space to run evolution in.
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`
        :param weightList: List of floats that defines the dimensionality of the fitness vector returned from evalFunction and the weights of each component for multiobjective optimization (positive = maximize, negative = minimize). If not given, then a single positive weight will be used, defaults to None
        :type weightList: list[float], optional
        :param model: Model to simulate, defaults to None
        :type model: `neurolib.models.model.Model`, optional
        :param hdf_filename: HDF file to store all results in, defaults to "evolution.hdf"
        :type hdf_filename: str, optional
        :param ncores: Number of cores to simulate on (max cores default), defaults to None
        :type ncores: int, optional
        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed), defaults to 100
        :type POP_INIT_SIZE: int, optional
        :param POP_SIZE: Size of the population during evolution, defaults to 20
        :type POP_SIZE: int, optional
        :param NGEN: Numbers of generations to evaluate, defaults to 10
        :type NGEN: int, optional
        :param matingFunction: Custom mating function, defaults to blend crossover if not set., defaults to None
        :type matingFunction: function, optional
        :param CXP: Parameter handed to the mating function (for blend crossover, this is `alpha`), defaults to 0.5
        :type CXP: float, optional
        :param selectionFunction: Custom parent selection function, defaults to rank selection if not set., defaults to None
        :type selectionFunction: function, optional
        :param RANKP: Parent selection parameter (for rank selection, this is `s` in Eiben&Smith p.81), defaults to 1.5
        :type RANKP: float, optional
        """

        if weightList is None:
            logging.info("weightList not set, assuming single fitness value to be maximized.")
            weightList = [1.0]

        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        self.HDF_FILE = os.path.join(paths.HDF_DIR, hdf_filename)
        trajectoryFileName = self.HDF_FILE

        logging.info("Storing data to: {}".format(trajectoryFileName))
        logging.info("Trajectory Name: {}".format(trajectoryName))
        if ncores is None:
            ncores = multiprocessing.cpu_count()
        logging.info("Number of cores: {}".format(ncores))

        # initialize pypet environment
        # env = pp.Environment(trajectory=trajectoryName, filename=trajectoryFileName)
        env = pp.Environment(
            trajectory=trajectoryName,
            filename=trajectoryFileName,
            use_pool=False,
            multiproc=True,
            ncores=ncores,
            complevel=9,
        )

        # Get the trajectory from the environment
        traj = env.traj
        # Sanity check if everything went ok
        assert (
            trajectoryName == traj.v_name
        ), f"Pypet trajectory has a different name than trajectoryName {trajectoryName}"
        # trajectoryName = traj.v_name

        self.model = model
        self.evalFunction = evalFunction
        self.weightList = weightList

        self.CXP = CXP
        self.RANKP = RANKP
        self.NGEN = NGEN
        assert POP_SIZE % 2 == 0, "Please chose an even number for POP_SIZE!"
        self.POP_SIZE = POP_SIZE
        assert POP_INIT_SIZE % 2 == 0, "Please chose an even number for POP_INIT_SIZE!"
        self.POP_INIT_SIZE = POP_INIT_SIZE
        self.ncores = ncores

        self.traj = env.traj
        self.env = env
        self.trajectoryName = trajectoryName
        self.trajectoryFileName = trajectoryFileName

        self._initialPopulationSimulated = False

        # -------- settings
        self.verbose = False

        # -------- simulation
        self.parameterSpace = parameterSpace
        self.ParametersInterval = parameterSpace.named_tuple_constructor
        self.paramInterval = parameterSpace.named_tuple

        self.toolbox = deap.base.Toolbox()

        if matingFunction is None:
            # this is our custom uniform mating function
            # matingFunction = du.cxUniform_adapt
            # this is blend crossover (with alpha)
            matingFunction = tools.cxBlend
        self.matingFunction = matingFunction

        if selectionFunction is None:
            selectionFunction = du.selRank
        self.selectionFunction = selectionFunction

        self.initDEAP(
            self.toolbox,
            self.env,
            self.paramInterval,
            self.evalFunction,
            weights_list=self.weightList,
            matingFunction=self.matingFunction,
            selectionFunction=self.selectionFunction,
        )

        # comment string for storing info
        self.comments = ""

        # set up pypet trajectory
        self.initPypetTrajectory(
            self.traj, self.paramInterval, self.POP_SIZE, self.CXP, self.NGEN, self.model,
        )

        # population history: dict of all valid individuals per generation
        self.popHist = {}

        # initialize population
        self.evaluationCounter = 0
        self.last_id = 0

    def getIndividualFromTraj(self, traj):
        """Get individual from pypet trajectory
        
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Individual (`DEAP` type)
        :rtype: `deap.creator.Individual`
        """
        # either pass an individual or a pypet trajectory with the attribute individual
        if type(traj).__name__ == "Individual":
            individual = traj
        else:
            individual = traj.individual
            ind_id = traj.id
            individual = [p for p in self.pop if p.id == ind_id]
            if len(individual) > 0:
                individual = individual[0]
        return individual

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this individual
        :params traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual

        :returns model: Model with the parameters of this individual.
        
        :param traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual
        :type traj: `pypet.trajectory.Trajectory`
        :return: Model with the parameters of this individual.
        :rtype: `neurolib.models.model.Model`
        """
        model = self.model
        model.params.update(self.individualToDict(self.getIndividualFromTraj(traj)))
        return model

    def individualToDict(self, individual):
        """Convert an individual to a parameter dictionary.
        
        :param individual: Individual (`DEAP` type)
        :type individual: `deap.creator.Individual`
        :return: Parameter dictionary of this individual
        :rtype: dict
        """
        return self.ParametersInterval(*(individual[: len(self.paramInterval)]))._asdict().copy()

    def initPypetTrajectory(self, traj, paramInterval, POP_SIZE, CXP, NGEN, model):
        """Initializes pypet trajectory and store all simulation parameters for later analysis.
        
        :param traj: Pypet trajectory (must be already initialized!)
        :type traj: `pypet.trajectory.Trajectory`
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param POP_SIZE: Population size
        :type POP_SIZE: int
        :param CXP: Crossover parameter
        :type CXP: float
        :param NGEN: Number of generations
        :type NGEN: int
        :param model: Model to store the default parameters of
        :type model: `neurolib.models.model.Model`
        """
        # Initialize pypet trajectory and add all simulation parameters
        traj.f_add_parameter("popsize", POP_SIZE, comment="Population size")  #
        traj.f_add_parameter("CXP", CXP, comment="Crossover parameter")
        traj.f_add_parameter("NGEN", NGEN, comment="Number of generations")

        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter("generation", 0, comment="Current generation")

        traj.f_add_result("scores", [], comment="Score of all individuals for each generation")
        traj.f_add_result_group("evolution", comment="Contains results for each generation")
        traj.f_add_result_group("outputs", comment="Contains simulation results")

        # if a model was given, save its parameters
        if model is not None:
            traj.f_add_result("params", model.params, comment="Default parameters")

        # todo: initialize this after individuals have been defined!
        traj.f_add_parameter("id", 0, comment="Index of individual")
        traj.f_add_parameter("ind_len", 20, comment="Length of individual")
        traj.f_add_derived_parameter(
            "individual", [0 for x in range(traj.ind_len)], "An indivudal of the population",
        )

    def initDEAP(
        self, toolbox, pypetEnvironment, paramInterval, evalFunction, weights_list, matingFunction, selectionFunction
    ):
        """Initializes DEAP and registers all methods to the deap.toolbox
        
        :param toolbox: Deap toolbox
        :type toolbox: deap.base.Toolbox
        :param pypetEnvironment: Pypet environment (must be initialized first!)
        :type pypetEnvironment: [type]
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param evalFunction: Evaluation function
        :type evalFunction: function
        :param weights_list: List of weiths for multiobjective optimization
        :type weights_list: list[float]
        :param matingFunction: Mating function (crossover)
        :type matingFunction: function
        :param selectionFunction: Parent selection function
        :type selectionFunction: function
        """
        # ------------- register everything in deap
        deap.creator.create("FitnessMulti", deap.base.Fitness, weights=tuple(weights_list))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        # initially, each individual has randomized genes
        # need to create a lambda funciton because du.generateRandomParams wants an argument but
        # toolbox.register cannot pass an argument to it.
        toolbox.register(
            "individual",
            deap.tools.initIterate,
            deap.creator.Individual,
            lambda: du.randomParametersAdaptive(paramInterval),
        )
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("selBest", du.selBest_multiObj)
        toolbox.register("selRank", du.selRank)
        # "select" is not used, instead selRank is used but this is more general and should be changed
        toolbox.register("select", selectionFunction)
        logging.info(f"Evolution: Registered {selectionFunction} as selection function.")
        toolbox.register("evaluate", evalFunction)
        toolbox.register("mate", matingFunction)
        logging.info(f"Evolution: Registered {matingFunction} as mating function.")
        toolbox.register("mutate", du.gaussianAdaptiveMutation_nStepSizes)
        toolbox.register("map", pypetEnvironment.run)
        toolbox.register("run_map", pypetEnvironment.run_map)

        # recording the history
        # self.history = tools.History()
        # toolbox.decorate("mate", self.history.decorator)
        # toolbox.decorate("mutate", self.history.decorator)

    def evalPopulationUsingPypet(self, traj, toolbox, pop, gIdx):
        """Evaluate the fitness of the popoulation of the current generation using pypet
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :param toolbox: `deap` toolbox
        :type toolbox: deap.base.Toolbox
        :param pop: Population
        :type pop: list
        :param gIdx: Index of the current generation
        :type gIdx: int
        :return: Evaluated population with fitnesses
        :rtype: list
        """
        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(
            pp.cartesian_product(
                {"generation": [gIdx], "id": [x.id for x in pop], "individual": [list(x) for x in pop],},
                [("id", "individual"), "generation"],
            )
        )  # the current generation  # unique id of each individual

        # increment the evaluationCounter
        self.evaluationCounter += len(pop)

        # run simulations for one generation
        evolutionResult = toolbox.map(toolbox.evaluate)

        # This error can have different reasons but is most likely
        # due to multiprocessing problems. One possibility is that your evaluation
        # funciton is not pickleable or that it returns an object that is not pickleable.
        assert len(evolutionResult) > 0, "No results returned from simulations."

        for idx, result in enumerate(evolutionResult):
            runIndex, packedReturnFromEvalFunction = result

            # packedReturnFromEvalFunction is the return from the evaluation function
            # it has length two, the first is the fitness, second is the model output
            assert (
                len(packedReturnFromEvalFunction) == 2
            ), "Evaluation function must return tuple with shape (fitness, output_data)"

            fitnessesResult, returnedOutputs = packedReturnFromEvalFunction
            pop[idx].outputs = returnedOutputs
            # store outputs of simulations in population
            pop[idx].fitness.values = fitnessesResult
            # mean fitness value
            pop[idx].fitness.score = np.nansum(pop[idx].fitness.wvalues) / (len(pop[idx].fitness.wvalues))
        return pop

    def getValidPopulation(self, pop):
        return [p for p in pop if not np.any(np.isnan(p.fitness.values))]

    def getInvalidPopulation(self, pop):
        return [p for p in pop if np.any(np.isnan(p.fitness.values))]

    def tagPopulation(self, pop):
        """Take a fresh population and add id's and attributes such as parameters that we can use later

        :param pop: Fresh population
        :type pop: list
        :return: Population with tags
        :rtype: list
        """
        for i, ind in enumerate(pop):
            assert not hasattr(ind, "id"), "Individual has an id already, will not overwrite it!"
            ind.id = self.last_id
            ind.gIdx = self.gIdx
            ind.simulation_stored = False
            ind_dict = self.individualToDict(ind)
            for key, value in ind_dict.items():
                # set the parameters as attributes for easy access
                setattr(ind, key, value)
            ind.params = ind_dict
            # increment id counter
            self.last_id += 1
        return pop

    def runInitial(self):
        """Run the first round of evolution with the initial population of size `POP_INIT_SIZE`
        and select the best `POP_SIZE` for the following evolution. This needs to be run before `runEvolution()`
        """
        self._t_start_initial_population = datetime.datetime.now()

        # Create the initial population
        self.pop = self.toolbox.population(n=self.POP_INIT_SIZE)

        ### Evaluate the initial population
        logging.info("Evaluating initial population of size %i ..." % len(self.pop))
        self.gIdx = 0  # set generation index
        self.pop = self.tagPopulation(self.pop)

        # evaluate
        self.pop = self.evalPopulationUsingPypet(self.traj, self.toolbox, self.pop, self.gIdx)

        if self.verbose:
            eu.printParamDist(self.pop, self.paramInterval, self.gIdx)

        # save all simulation data to pypet
        try:
            self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)
        except:
            logging.warn("Error: Write to pypet failed!")

        # Only the best indviduals are selected for the population the others do not survive
        self.pop[:] = self.toolbox.selBest(self.pop, k=self.traj.popsize)
        self._initialPopulationSimulated = True

        # populate history for tracking
        self.popHist[self.gIdx] = self.getValidPopulation(self.pop)

        self._t_end_initial_population = datetime.datetime.now()

    def runEvolution(self):
        """Run the evolutionary optimization process for `NGEN` generations.
        """
        # Start evolution
        logging.info("Start of evolution")
        self._t_start_evolution = datetime.datetime.now()
        for self.gIdx in range(self.gIdx + 1, self.gIdx + self.traj.NGEN):
            # ------- Weed out the invalid individuals and replace them by random new indivuals -------- #
            validpop = self.getValidPopulation(self.pop)
            # replace invalid individuals
            invalidpop = self.getInvalidPopulation(self.pop)
            logging.info("Replacing {} invalid individuals.".format(len(invalidpop)))
            newpop = self.toolbox.population(n=len(invalidpop))
            newpop = self.tagPopulation(newpop)

            # ------- Create the next generation by crossover and mutation -------- #
            ### Select parents using rank selection and clone them ###
            offspring = list(map(self.toolbox.clone, self.toolbox.selRank(self.pop, self.POP_SIZE, self.RANKP)))
            # n_offspring = min(len(validpop), self.POP_SIZE)
            # if n_offspring % 2 != 0:
            #     n_offspring -= 1
            # offspring = list(map(self.toolbox.clone, self.toolbox.selRank(validpop, n_offspring)))

            ##### cross-over ####
            for i in range(1, len(offspring), 2):
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i], self.CXP)
                # del offspring[i - 1].fitness, offspring[i].fitness
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                del offspring[i - 1].fitness.wvalues, offspring[i].fitness.wvalues
                #
                offspring[i - 1].parentIds = offspring[i - 1].id, offspring[i].id
                offspring[i].parentIds = offspring[i - 1].id, offspring[i].id
                del offspring[i - 1].id, offspring[i].id

            ##### Mutation ####
            # Apply adaptive mutation
            du.mutateUntilValid(offspring, self.paramInterval, self.toolbox)

            offspring = self.tagPopulation(offspring)

            # ------- Evaluate next generation -------- #

            self.pop = offspring + newpop
            self.evalPopulationUsingPypet(self.traj, self.toolbox, offspring + newpop, self.gIdx)
            # ------- Select surviving population -------- #

            # select next population
            self.pop = self.toolbox.selBest(validpop + offspring + newpop, k=self.traj.popsize)

            # ------- END OF ROUND -------

            # add all valid individuals to the population history
            self.popHist[self.gIdx] = self.getValidPopulation(self.pop)
            # self.history.update(self.getValidPopulation(self.pop))
            # save all simulation data to pypet
            try:
                self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)
            except:
                logging.warn("Error: Write to pypet failed!")

            # select best individual for logging
            self.best_ind = self.toolbox.selBest(self.pop, 1)[0]

            # text log
            logging.info("----------- Generation %i -----------" % self.gIdx)
            logging.info("Best individual is {}".format(self.best_ind))
            logging.info("Score: {}".format(self.best_ind.fitness.score))
            logging.info("Fitness: {}".format(self.best_ind.fitness.values))
            logging.info("--- Population statistics ---")

            # plotting
            if self.verbose:
                eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
                # plotting
                eu.plotPopulation(
                    self.pop, self.paramInterval, self.gIdx, plotScattermatrix=True, save_plots=self.trajectoryName
                )

        logging.info("--- End of evolution ---")
        logging.info("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        logging.info("--- End of evolution ---")

        self.traj.f_store()  # We switched off automatic storing, so we need to store manually
        self._t_end_evolution = datetime.datetime.now()

    def run(self, verbose=False):
        """Run the evolution or continue previous evolution. If evolution was not initialized first
        using `runInitial()`, this will be done.
        
        :param verbose: Print and plot state of evolution during run, defaults to False
        :type verbose: bool, optional
        """
        self.verbose = verbose
        if not self._initialPopulationSimulated:
            self.runInitial()
        self.runEvolution()

    def info(self, plot=True, bestN=5):
        """Print and plot information about the evolution and the current population
        
        :param plot: plot a plot using `matplotlib`, defaults to True
        :type plot: bool, optional
        :param bestN: Print summary of `bestN` best individuals, defaults to 5
        :type bestN: int, optional
        """
        eu.printEvolutionInfo(self)
        validPop = [p for p in self.pop if not np.any(np.isnan(p.fitness.values))]
        popArray = np.array([p[0 : len(self.paramInterval._fields)] for p in validPop]).T
        scores = np.array([validPop[i].fitness.score for i in range(len(validPop))])
        # Text output
        print("--- Info summary ---")
        print("Valid: {}".format(len(validPop)))
        print("Mean score (weighted fitness): {:.2}".format(np.mean(scores)))
        eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
        print("--------------------")
        print(f"Best {bestN} individuals:")
        eu.printIndividuals(self.toolbox.selBest(self.pop, bestN), self.paramInterval)
        print("--------------------")
        # Plotting
        if plot:
            eu.plotPopulation(self.pop, self.paramInterval, self.gIdx, plotScattermatrix=True)

    def plotProgress(self):
        """Plots progress of fitnesses of current evolution run
        """
        eu.plotProgress(self)

    @property
    def dfPop(self):
        """Returns a `pandas` dataframe of the current generation's population parameters 
        for post processing. This object can be further used to easily analyse the population.
        :return: Pandas dataframe with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        validPop = [p for p in self.pop if not np.any(np.isnan(p.fitness.values))]
        indIds = [p.id for p in validPop]
        popArray = np.array([p[0 : len(self.paramInterval._fields)] for p in validPop]).T
        scores = np.array([validPop[i].fitness.score for i in range(len(validPop))])
        # gridParameters = [k for idx, k in enumerate(paramInterval._fields)]
        dfPop = pd.DataFrame(popArray, index=self.parameterSpace.parameter_names).T
        dfPop["score"] = scores
        dfPop["id"] = indIds
        return dfPop

    def loadResults(self, filename=None, trajectoryName=None):
        """Load results from a hdf file of a previous evolution and store the
        pypet trajectory in `self.traj`
        
        :param filename: hdf filename of the previous run, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory in the hdf file to load. If not given, the last one will be loaded, defaults to None
        :type trajectoryName: str, optional
        """
        if filename == None:
            filename = self.HDF_FILE
        self.traj = pu.loadPypetTrajectory(filename, trajectoryName)

    def getScoresDuringEvolution(self, traj=None, drop_first=True, reverse=False):
        """Get the scores of each generation's population.
        
        :param traj: Pypet trajectory. If not given, the current trajectory is used, defaults to None
        :type traj: `pypet.trajectory.Trajectory`, optional
        :param drop_first: Drop the first (initial) generation. This can be usefull because it can have a different size (`POP_INIT_SIZE`) than the succeeding populations (`POP_SIZE`) which can make data handling tricky, defaults to True
        :type drop_first: bool, optional
        :param reverse: Reverse the order of each generation. This is a necessary workaraound because loading from the an hdf file returns the generations in a reversed order compared to loading each generation from the pypet trajectory in memory, defaults to False
        :type reverse: bool, optional
        :return: Tuple of list of all generations and an array of the scores of all individuals
        :rtype: tuple[list, numpy.ndarray]
        """
        if traj == None:
            traj = self.traj

        generation_names = list(traj.results.evolution.f_to_dict(nested=True).keys())

        if reverse:
            generation_names = generation_names[::-1]
        if drop_first:
            generation_names.remove("gen_000000")

        npop = len(traj.results.evolution[generation_names[0]].scores)

        gens = []
        all_scores = np.empty((len(generation_names), npop))

        for i, r in enumerate(generation_names):
            gens.append(i)
            scores = traj.results.evolution[r].scores
            all_scores[i] = scores

        if drop_first:
            gens = np.add(gens, 1)

        return gens, all_scores

