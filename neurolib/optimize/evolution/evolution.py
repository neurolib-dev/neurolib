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
    """Evolutionary parameter optimization.
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
        CXPB=0.04,
    ):
        """
        :param model: Model to run
        :type model: Model
        :param evalFunction: Evaluation function of a run that provides a fitness vector and simulation outputs
        :type evalFunction: Function shuold retiurn a tuple of the form (fitness_tuple, model.output)
        :param weightList: List of floats that defines the dimensionality of the fitness vector returned from evalFunction and the weights of each component (positive = maximize, negative = minimize) 

        :param hdf_filename: HDF file to store all results in (data/hdf/evolution.hdf default)
        :param ncores: Number of cores to simulate on (max cores default)
        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed)
        :param POP_SIZE: Size of the population during evolution
        :param NGEN: Numbers of generations to evaluate
        :param CXPB: Crossover probability of each individual gene
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

        self.CXPB = CXPB
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

        self.initDEAP(
            self.toolbox, self.env, self.paramInterval, self.evalFunction, weights_list=self.weightList,
        )

        # comment string for storing info
        self.comments = ""

        # set up pypet trajectory
        self.initPypetTrajectory(
            self.traj, self.paramInterval, self.POP_SIZE, self.CXPB, self.NGEN, self.model,
        )

        # population history: dict of all valid individuals per generation
        self.popHist = {}

        # initialize population
        self.evaluationCounter = 0
        self.last_id = 0

    def getIndividualFromTraj(self, traj):
        """Get individual from pypet trajectory
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
        """
        model = self.model
        model.params.update(self.individualToDict(self.getIndividualFromTraj(traj)))
        return model

    def individualToDict(self, individual):
        """
        Convert an individual to a parameter dictionary.
        """
        return self.ParametersInterval(*(individual[: len(self.paramInterval)]))._asdict().copy()

    def initPypetTrajectory(self, traj, paramInterval, POP_SIZE, CXPB, NGEN, model):
        """Initializes pypet trajectory and store all simulation parameters.
        """
        # Initialize pypet trajectory and add all simulation parameters
        traj.f_add_parameter("popsize", POP_SIZE, comment="Population size")  #
        traj.f_add_parameter("CXPB", CXPB, comment="Crossover term")  # Crossover probability
        traj.f_add_parameter("NGEN", NGEN, comment="Number of generations")

        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter("generation", 0, comment="Current generation")

        traj.f_add_result("scores", [], comment="Mean_score for each generation")
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

    def initDEAP(self, toolbox, pypetEnvironment, paramInterval, evalFunction, weights_list):
        """Initializes DEAP and registers all methods to the deap.toolbox
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
        toolbox.register("evaluate", evalFunction)
        toolbox.register("mate", du.cxUniform_adapt)
        toolbox.register("mutate", du.gaussianAdaptiveMutation_nStepSizes)
        toolbox.register("map", pypetEnvironment.run)
        toolbox.register("run_map", pypetEnvironment.run_map)

        # recording the history
        # self.history = tools.History()
        # toolbox.decorate("mate", self.history.decorator)
        # toolbox.decorate("mutate", self.history.decorator)

    def evalPopulationUsingPypet(self, traj, toolbox, pop, gIdx):
        """
        Eval the population fitness using pypet
        
        Params:
            pop:    List of inidivual (population to evaluate)
            gIdx:   Index of the current generation
            
        Return:
            The population with updated fitness values
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

        :param pop: Population
        :type pop: list
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
        """First round of evolution: 
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
        # Start evolution
        logging.info("Start of evolution")
        self._t_start_evolution = datetime.datetime.now()
        for self.gIdx in range(self.gIdx + 1, self.gIdx + self.traj.NGEN):
            # ------- Weed out the invalid individuals and replace them by random new indivuals -------- #
            validpop = self.getValidPopulation(self.pop)

            # replace invalid individuals
            nanpop = self.getInvalidPopulation(self.pop)
            logging.info("Replacing {} invalid individuals.".format(len(nanpop)))
            newpop = self.toolbox.population(n=len(nanpop))
            newpop = self.tagPopulation(newpop)

            # self.pop = validpop + newpop

            # ------- Create the next generation by crossover and mutation -------- #
            ### Select parents using rank selection and clone them ###
            offspring = list(map(self.toolbox.clone, self.toolbox.selRank(self.pop, self.POP_SIZE)))
            # n_offspring = min(len(validpop), self.POP_SIZE)
            # if n_offspring % 2 != 0:
            #     n_offspring -= 1
            # offspring = list(map(self.toolbox.clone, self.toolbox.selRank(validpop, n_offspring)))

            ##### cross-over ####
            for i in range(1, len(offspring), 2):
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i], indpb=self.CXPB)
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
        """Run full evolution (or continue previous evolution)
        """
        self.verbose = verbose
        if not self._initialPopulationSimulated:
            self.runInitial()
        self.runEvolution()

    def info(self, plot=True, bestN=5):
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

    @property
    def dfPop(self):
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
        """Load results from evolution.
        """
        if filename == None:
            filename = self.HDF_FILE
        self.traj = pu.loadPypetTrajectory(filename, trajectoryName)

    def getScoresDuringEvolution(self, traj=None, drop_first=True, reverse=False):
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

