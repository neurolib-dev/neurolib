import datetime
import os
import logging
import multiprocessing

import deap
from deap import base
from deap import creator
from deap import tools

import numpy as np
import pypet as pp

import neurolib.utils.paths as paths

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.optimize.evolution.deapUtils as du
import neurolib.utils.pypetUtils as pu


class Evolution:
    def __init__(
        self, evalFunction, parameterSpace, weightList, model=None, hdf_filename="evolution.hdf", ncores=None, POP_INIT_SIZE=100, POP_SIZE=20, NGEN=10, CXPB=1 - 0.96,
    ):
        """Evolutionary parameter optimization
        
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
        env = pp.Environment(trajectory=trajectoryName, filename=trajectoryFileName, use_pool=False, multiproc=True, ncores=ncores, complevel=9,)

        # Get the trajectory from the environment
        traj = env.traj
        trajectoryName = traj.v_name

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

        self.initialPopulationSimulated = False

        # ------------- settings
        self.verbose = False

        # ------------- define parameters
        self.ParametersInterval = parameterSpace.named_tuple_constructor
        self.paramInterval = parameterSpace.named_tuple

        self.toolbox = deap.base.Toolbox()

        self.initDEAP(
            self.toolbox, self.env, self.paramInterval, self.evalFunction, weights_list=self.weightList,
        )

        # set up pypet trajectory
        self.initPypetTrajectory(
            self.traj, self.paramInterval, self.ParametersInterval, self.POP_SIZE, self.CXPB, self.NGEN, self.model,
        )

        # ------------- initialize population
        self.last_id = 0
        self.pop = self.toolbox.population(n=self.POP_INIT_SIZE)
        self.pop = self.tagPopulation(self.pop)

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

    def initPypetTrajectory(self, traj, paramInterval, ParametersInterval, POP_SIZE, CXPB, NGEN, model):
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
            "individual", deap.tools.initIterate, deap.creator.Individual, lambda: du.generate_random_pars_adapt(paramInterval),
        )
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("selBest", du.selBest_multiObj)
        toolbox.register("selRank", du.selRank)
        toolbox.register("evaluate", evalFunction)
        toolbox.register("mate", du.cxUniform_normDraw_adapt)
        toolbox.register("mutate", du.adaptiveMutation_nStepSize)
        toolbox.register("map", pypetEnvironment.run)
        toolbox.register("run_map", pypetEnvironment.run_map)

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
        traj.f_expand(pp.cartesian_product({"generation": [gIdx], "id": [x.id for x in pop], "individual": [list(x) for x in pop],}, [("id", "individual"), "generation"],))  # the current generation  # unique id of each individual
        # SIMULUATE INDIVIDUALS

        results = toolbox.map(toolbox.evaluate)

        assert len(results) > 0, "No results returned from simulations."

        for idx, result in enumerate(results):
            runIdx, packed_result = result
            # this is the return from the evaluation function
            fitnesses_result, outputs = packed_result

            # store outputs of simulations in population
            pop[idx].outputs = outputs
            pop[idx].fitness.values = fitnesses_result
            # mean fitness value
            pop[idx].fitness.score = np.nansum(pop[idx].fitness.wvalues) / (len(pop[idx].fitness.wvalues))
        return pop

    def getValidPopulation(self, pop):
        return [p for p in pop if not np.any(np.isnan(p.fitness.values))]

    def getInvalidPopulation(self, pop):
        return [p for p in pop if np.any(np.isnan(p.fitness.values))]

    def runInitial(self):
        """First round of evolution: 
        """
        ### Evaluate the initial population
        logging.info("Evaluating initial population of size %i ..." % len(self.pop))
        self.gIdx = 0  # set generation index
        # evaluate
        self.pop = self.evalPopulationUsingPypet(self.traj, self.toolbox, self.pop, self.gIdx)
        if self.verbose:
            eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
        self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)
        # Only the best indviduals are selected for the population the others do not survive
        self.pop[:] = self.toolbox.selBest(self.pop, k=self.traj.popsize)
        self.initialPopulationSimulated = True

    def tagPopulation(self, pop):
        """Take a fresh population and add id's and attributes such as parameters that we can use later

        :param pop: Population
        :type pop: list
        """
        for i, ind in enumerate(pop):
            assert not hasattr(ind, "id"), "Individual has an id already, will not overwrite it!"
            ind.id = self.last_id
            ind.simulation_stored = False
            ind_dict = self.individualToDict(ind)
            for key, value in ind_dict.items():
                # set the parameters as attributes for easy access
                setattr(ind, key, value)
            ind.params = ind_dict
            # increment id counter
            self.last_id += 1
        return pop

    def runEvolution(self):
        # Start evolution
        logging.info("Start of evolution")
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

            ##### cross-over ####
            for i in range(1, len(offspring), 2):
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i], indpb=self.CXPB)
                # del offspring[i - 1].fitness, offspring[i].fitness
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                del offspring[i - 1].fitness.wvalues, offspring[i].fitness.wvalues
                # print("Deleting ID of {} and {}".format(offspring[i - 1].id, offspring[i].id))
                del offspring[i - 1].id, offspring[i].id

            ##### Mutation ####
            # Apply adaptive mutation
            eu.mutateUntilValid(offspring, self.paramInterval, self.toolbox)

            offspring = self.tagPopulation(offspring)

            # ------- Evaluate next generation -------- #

            logging.info("----------- Generation %i -----------" % self.gIdx)

            self.pop = offspring + newpop
            self.evalPopulationUsingPypet(self.traj, self.toolbox, offspring + newpop, self.gIdx)
            # ------- Select surviving population -------- #

            # select next population
            self.pop = self.toolbox.selBest(validpop + offspring + newpop, k=self.traj.popsize)

            self.best_ind = self.toolbox.selBest(self.pop, 1)[0]
            logging.info("Best individual is {}".format(self.best_ind))
            logging.info("Score: {}".format(self.best_ind.fitness.score))
            logging.info("Fitness: {}".format(self.best_ind.fitness.values))

            logging.info("--- Population statistics ---")

            if self.verbose:
                eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
                eu.printPopFitnessStats(self.pop, self.paramInterval, self.gIdx, draw_scattermatrix=True, save_plots="evo")

            # save all simulation data to pypet
            try:
                self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)
            except:
                logging.warn("Error: Write to pypet failed!")

        logging.info("--- End of evolution ---")
        self.best_ind = self.toolbox.selBest(self.pop, 1)[0]
        logging.info("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        logging.info("--- End of evolution ---")

        self.traj.f_store()  # We switched off automatic storing, so we need to store manually

    def run(self, verbose=False):
        """Run full evolution (or continue previous evolution)
        """
        self.verbose = verbose
        if not self.initialPopulationSimulated:
            self.runInitial()
        self.runEvolution()

    def info(self, plot=True):
        eu.printParamDist(self.pop, self.paramInterval, self.gIdx)
        if plot:
            eu.printPopFitnessStats(self.pop, self.paramInterval, self.gIdx, draw_scattermatrix=True)
        bestN = 20
        print("--------------------------")
        print(f"Best {bestN} individuals:")
        eu.printIndividuals(self.toolbox.selBest(self.pop, bestN), self.paramInterval)

    def loadResults(self, filename=None, trajectoryName=None):
        """Load results from evolution.
        """
        if filename == None:
            filename = self.HDF_FILE
        trajLoaded = pu.loadPypetTrajectory(filename, trajectoryName)
        return trajLoaded

    def getScoresDuringEvolution(self, traj=None, drop_first=True, reverse=False):
        if traj == None:
            traj = self.traj

        generation_names = list(traj.results.evolution.f_to_dict(nested=True).keys())

        if drop_first:
            # drop first (initial) generation 0
            generation_names = generation_names[1:]
        if reverse:
            generation_names = generation_names[::-1]

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

