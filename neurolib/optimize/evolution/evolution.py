import datetime
import logging
import multiprocessing
import os

import deap
import numpy as np
import pandas as pd
import pypet as pp
from deap import base, creator, tools

from ...utils import paths as paths
from ...utils import pypetUtils as pu
from ...utils.parameterSpace import ParameterSpace
from ...utils.collections import unwrap_star_dotdict, BACKWARD_REPLACE
from . import deapUtils as du
from . import evolutionaryUtils as eu


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
        filename="evolution.hdf",
        ncores=None,
        POP_INIT_SIZE=100,
        POP_SIZE=20,
        NGEN=10,
        algorithm="adaptive",
        matingOperator=None,
        MATE_P=None,
        mutationOperator=None,
        MUTATE_P=None,
        selectionOperator=None,
        SELECT_P=None,
        parentSelectionOperator=None,
        PARENT_SELECT_P=None,
        individualGenerator=None,
        IND_GENERATOR_P=None,
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

        :param filename: HDF file to store all results in, defaults to "evolution.hdf"
        :type filename: str, optional
        :param ncores: Number of cores to simulate on (max cores default), defaults to None
        :type ncores: int, optional

        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed), defaults to 100
        :type POP_INIT_SIZE: int, optional
        :param POP_SIZE: Size of the population during evolution, defaults to 20
        :type POP_SIZE: int, optional
        :param NGEN: Numbers of generations to evaluate, defaults to 10
        :type NGEN: int, optional

        :param matingOperator: Custom mating operator, defaults to deap.tools.cxBlend
        :type matingOperator: deap operator, optional
        :param MATE_P: Mating operator keyword arguments (for the default crossover operator cxBlend, this defaults `alpha` = 0.5)
        :type MATE_P: dict, optional

        :param mutationOperator: Custom mutation operator, defaults to du.gaussianAdaptiveMutation_nStepSizes
        :type mutationOperator: deap operator, optional
        :param MUTATE_P: Mutation operator keyword arguments
        :type MUTATE_P: dict, optional

        :param selectionOperator: Custom selection operator, defaults to du.selBest_multiObj
        :type selectionOperator: deap operator, optional
        :param SELECT_P: Selection operator keyword arguments
        :type SELECT_P: dict, optional

        :param parentSelectionOperator: Operator for parent selection, defaults to du.selRank
        :param PARENT_SELECT_P: Parent selection operator keyword arguments (for the default operator selRank, this defaults to `s` = 1.5 in Eiben&Smith p.81)
        :type PARENT_SELECT_P: dict, optional

        :param individualGenerator: Function to generate initial individuals, defaults to du.randomParametersAdaptive
        """

        if weightList is None:
            logging.info("weightList not set, assuming single fitness value to be maximized.")
            weightList = [1.0]

        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        logging.info(f"Trajectory Name: {trajectoryName}")
        self.HDF_FILE = os.path.join(paths.HDF_DIR, filename)
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
            log_config=paths.PYPET_LOGGING_CONFIG,
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

        self.NGEN = NGEN
        assert POP_SIZE % 2 == 0, "Please chose an even number for POP_SIZE!"
        self.POP_SIZE = POP_SIZE
        assert POP_INIT_SIZE % 2 == 0, "Please chose an even number for POP_INIT_SIZE!"
        self.POP_INIT_SIZE = POP_INIT_SIZE
        self.ncores = ncores

        # comment string for storing info
        self.comments = "no comments"

        self.traj = env.traj
        self.env = env
        self.trajectoryName = trajectoryName
        self.trajectoryFileName = trajectoryFileName

        self._initialPopulationSimulated = False

        # -------- settings
        self.verbose = False
        self.plotColor = "C0"

        # -------- simulation
        self.parameterSpace = parameterSpace
        self.ParametersInterval = self.parameterSpace.named_tuple_constructor
        self.paramInterval = self.parameterSpace.named_tuple

        self.toolbox = deap.base.Toolbox()

        # -------- algorithms
        if algorithm == "adaptive":
            logging.info(f"Evolution: Using algorithm: {algorithm}")
            self.matingOperator = tools.cxBlend
            self.MATE_P = {"alpha": 0.5} or MATE_P
            self.mutationOperator = du.gaussianAdaptiveMutation_nStepSizes
            self.selectionOperator = du.selBest_multiObj
            self.parentSelectionOperator = du.selRank
            self.PARENT_SELECT_P = {"s": 1.5} or PARENT_SELECT_P
            self.individualGenerator = du.randomParametersAdaptive

        elif algorithm == "nsga2":
            logging.info(f"Evolution: Using algorithm: {algorithm}")
            self.matingOperator = tools.cxSimulatedBinaryBounded
            self.MATE_P = {
                "low": self.parameterSpace.lowerBound,
                "up": self.parameterSpace.upperBound,
                "eta": 20.0,
            } or MATE_P
            self.mutationOperator = tools.mutPolynomialBounded
            self.MUTATE_P = {
                "low": self.parameterSpace.lowerBound,
                "up": self.parameterSpace.upperBound,
                "eta": 20.0,
                "indpb": 1.0 / len(self.weightList),
            } or MUTATE_P
            self.selectionOperator = tools.selNSGA2
            self.parentSelectionOperator = tools.selTournamentDCD
            self.individualGenerator = du.randomParameters

        else:
            raise ValueError("Evolution: algorithm must be one of the following: ['adaptive', 'nsga2']")

        # if the operators are set manually, then overwrite them
        self.matingOperator = self.matingOperator if hasattr(self, "matingOperator") else matingOperator
        self.mutationOperator = self.mutationOperator if hasattr(self, "mutationOperator") else mutationOperator
        self.selectionOperator = self.selectionOperator if hasattr(self, "selectionOperator") else selectionOperator
        self.parentSelectionOperator = (
            self.parentSelectionOperator if hasattr(self, "parentSelectionOperator") else parentSelectionOperator
        )
        self.individualGenerator = (
            self.individualGenerator if hasattr(self, "individualGenerator") else individualGenerator
        )

        # let's also make sure that the parameters are set correctly
        self.MATE_P = self.MATE_P if hasattr(self, "MATE_P") else {}
        self.PARENT_SELECT_P = self.PARENT_SELECT_P if hasattr(self, "PARENT_SELECT_P") else {}
        self.MUTATE_P = self.MUTATE_P if hasattr(self, "MUTATE_P") else {}
        self.SELECT_P = self.SELECT_P if hasattr(self, "SELECT_P") else {}

        self.initDEAP(
            self.toolbox,
            self.env,
            self.paramInterval,
            self.evalFunction,
            weightList=self.weightList,
            matingOperator=self.matingOperator,
            mutationOperator=self.mutationOperator,
            selectionOperator=self.selectionOperator,
            parentSelectionOperator=self.parentSelectionOperator,
            individualGenerator=self.individualGenerator,
        )

        # set up pypet trajectory
        self.initPypetTrajectory(
            self.traj,
            self.paramInterval,
            self.POP_SIZE,
            self.NGEN,
            self.model,
        )

        # population history: dict of all valid individuals per generation
        self.history = {}

        # initialize population
        self.evaluationCounter = 0
        self.last_id = 0

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
        # resolve star notation - MultiModel
        individual_params = self.individualToDict(self.getIndividualFromTraj(traj))
        if self.parameterSpace.star:
            individual_params = unwrap_star_dotdict(individual_params, self.model, replaced_dict=BACKWARD_REPLACE)
        model.params.update(individual_params)
        return model

    def getIndividualFromHistory(self, id):
        """Searches the entire evolution history for an individual with a specific id and returns it.

        :param id: Individual id
        :type id: int
        :return: Individual (`DEAP` type)
        :rtype: `deap.creator.Individual`
        """
        for key, value in self.history.items():
            for p in value:
                if p.id == id:
                    return p
        logging.warning(f"No individual with id={id} found. Returning `None`")
        return None

    def individualToDict(self, individual):
        """Convert an individual to a parameter dictionary.

        :param individual: Individual (`DEAP` type)
        :type individual: `deap.creator.Individual`
        :return: Parameter dictionary of this individual
        :rtype: dict
        """
        return self.ParametersInterval(*(individual[: len(self.paramInterval)]))._asdict().copy()

    def initPypetTrajectory(self, traj, paramInterval, POP_SIZE, NGEN, model):
        """Initializes pypet trajectory and store all simulation parameters for later analysis.

        :param traj: Pypet trajectory (must be already initialized!)
        :type traj: `pypet.trajectory.Trajectory`
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param POP_SIZE: Population size
        :type POP_SIZE: int
        :param MATE_P: Crossover parameter
        :type MATE_P: float
        :param NGEN: Number of generations
        :type NGEN: int
        :param model: Model to store the default parameters of
        :type model: `neurolib.models.model.Model`
        """
        # Initialize pypet trajectory and add all simulation parameters
        traj.f_add_parameter("popsize", POP_SIZE, comment="Population size")  #
        traj.f_add_parameter("NGEN", NGEN, comment="Number of generations")

        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter("generation", 0, comment="Current generation")

        traj.f_add_result("scores", [], comment="Score of all individuals for each generation")
        traj.f_add_result_group("evolution", comment="Contains results for each generation")
        traj.f_add_result_group("outputs", comment="Contains simulation results")

        # TODO: save evolution parameters and operators as well, MATE_P, MUTATE_P, etc..

        # if a model was given, save its parameters
        # NOTE: Convert model.params to dict() since it is a dotdict() and pypet doesn't like that
        if model is not None:
            params_dict = dict(model.params)
            # replace all None with zeros, pypet doesn't like None
            for key, value in params_dict.items():
                if value is None:
                    params_dict[key] = "None"
            traj.f_add_result("params", params_dict, comment="Default parameters")

        # todo: initialize this after individuals have been defined!
        traj.f_add_parameter("id", 0, comment="Index of individual")
        traj.f_add_parameter("ind_len", 20, comment="Length of individual")
        traj.f_add_derived_parameter(
            "individual",
            [0 for x in range(traj.ind_len)],
            "An indivudal of the population",
        )

    def initDEAP(
        self,
        toolbox,
        pypetEnvironment,
        paramInterval,
        evalFunction,
        weightList,
        matingOperator,
        mutationOperator,
        selectionOperator,
        parentSelectionOperator,
        individualGenerator,
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
        :param weightList: List of weiths for multiobjective optimization
        :type weightList: list[float]
        :param matingOperator: Mating function (crossover)
        :type matingOperator: function
        :param selectionOperator: Parent selection function
        :type selectionOperator: function
        :param individualGenerator: Function that generates individuals
        """
        # ------------- register everything in deap
        deap.creator.create("FitnessMulti", deap.base.Fitness, weights=tuple(weightList))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        # initially, each individual has randomized genes
        # need to create a lambda funciton because du.generateRandomParams wants an argument but
        # toolbox.register cannot pass an argument to it.
        toolbox.register(
            "individual",
            deap.tools.initIterate,
            deap.creator.Individual,
            lambda: individualGenerator(paramInterval),
        )
        logging.info(f"Evolution: Individual generation: {individualGenerator}")

        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("map", pypetEnvironment.run)
        toolbox.register("evaluate", evalFunction)
        toolbox.register("run_map", pypetEnvironment.run_map)

        # Operator registering

        toolbox.register("mate", matingOperator)
        logging.info(f"Evolution: Mating operator: {matingOperator}")

        toolbox.register("mutate", mutationOperator)
        logging.info(f"Evolution: Mutation operator: {mutationOperator}")

        toolbox.register("selBest", du.selBest_multiObj)
        toolbox.register("selectParents", parentSelectionOperator)
        logging.info(f"Evolution: Parent selection: {parentSelectionOperator}")
        toolbox.register("select", selectionOperator)
        logging.info(f"Evolution: Selection operator: {selectionOperator}")

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

        # this function is necessary for the NSGA-2 algorithms because
        # some operators return np.float64 instead of float and pypet
        # does not like individuals with mixed types... sigh.
        def _cleanIndividual(ind):
            return [float(i) for i in ind]

        traj.f_expand(
            pp.cartesian_product(
                {
                    "generation": [gIdx],
                    "id": [x.id for x in pop],
                    "individual": [list(_cleanIndividual(x)) for x in pop],
                },
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

            # store simulation outputs
            pop[idx].outputs = returnedOutputs

            # store fitness values
            pop[idx].fitness.values = fitnessesResult

            # compute score
            pop[idx].fitness.score = np.ma.masked_invalid(pop[idx].fitness.wvalues).sum() / (
                len(pop[idx].fitness.wvalues)
            )
        return pop

    def getValidPopulation(self, pop=None):
        """Returns a list of the valid population.

        :params pop: Population to check, defaults to self.pop
        :type pop: deap population
        :return: List of valid population
        :rtype: list
        """
        pop = pop or self.pop
        return [p for p in pop if not (np.isnan(p.fitness.values).any() or np.isinf(p.fitness.values).any())]

    def getInvalidPopulation(self, pop=None):
        """Returns a list of the invalid population.

        :params pop: Population to check, defaults to self.pop
        :type pop: deap population
        :return: List of invalid population
        :rtype: list
        """
        pop = pop or self.pop
        return [p for p in pop if np.isnan(p.fitness.values).any() or np.isinf(p.fitness.values).any()]

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
        self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)

        # reduce initial population to popsize
        self.pop = self.toolbox.select(self.pop, k=self.traj.popsize, **self.SELECT_P)

        self._initialPopulationSimulated = True

        # populate history for tracking
        self.history[self.gIdx] = self.pop  # self.getValidPopulation(self.pop)

        self._t_end_initial_population = datetime.datetime.now()

    def runEvolution(self):
        """Run the evolutionary optimization process for `NGEN` generations."""
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
            offspring = list(
                map(
                    self.toolbox.clone,
                    self.toolbox.selectParents(self.pop, self.POP_SIZE, **self.PARENT_SELECT_P),
                )
            )

            ##### cross-over ####
            for i in range(1, len(offspring), 2):
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i], **self.MATE_P)
                # delete fitness inherited from parents
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
                del offspring[i - 1].fitness.wvalues, offspring[i].fitness.wvalues

                # assign parent IDs to new offspring
                offspring[i - 1].parentIds = offspring[i - 1].id, offspring[i].id
                offspring[i].parentIds = offspring[i - 1].id, offspring[i].id

                # delete id originally set from parents, needs to be deleted here!
                # will be set later in tagPopulation()
                del offspring[i - 1].id, offspring[i].id

            ##### Mutation ####
            # Apply mutation
            du.mutateUntilValid(offspring, self.paramInterval, self.toolbox, MUTATE_P=self.MUTATE_P)

            offspring = self.tagPopulation(offspring)

            # ------- Evaluate next generation -------- #

            self.pop = offspring + newpop
            self.evalPopulationUsingPypet(self.traj, self.toolbox, offspring + newpop, self.gIdx)

            # log individuals
            self.history[self.gIdx] = validpop + offspring + newpop  # self.getValidPopulation(self.pop)

            # ------- Select surviving population -------- #

            # select next generation
            self.pop = self.toolbox.select(validpop + offspring + newpop, k=self.traj.popsize, **self.SELECT_P)

            # ------- END OF ROUND -------

            # save all simulation data to pypet
            self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)

            # select best individual for logging
            self.best_ind = self.toolbox.selBest(self.pop, 1)[0]

            # text log
            next_print = print if self.verbose else logging.info
            next_print("----------- Generation %i -----------" % self.gIdx)
            next_print("Best individual is {}".format(self.best_ind))
            next_print("Score: {}".format(self.best_ind.fitness.score))
            next_print("Fitness: {}".format(self.best_ind.fitness.values))
            next_print("--- Population statistics ---")

            # verbose output
            if self.verbose:
                self.info(plot=True, info=True)

        logging.info("--- End of evolution ---")
        logging.info("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        logging.info("--- End of evolution ---")

        self.traj.f_store()  # We switched off automatic storing, so we need to store manually
        self._t_end_evolution = datetime.datetime.now()

        self.buildEvolutionTree()

    def buildEvolutionTree(self):
        """Builds a genealogy tree that is networkx compatible.

        Plot the tree using:

            import matplotlib.pyplot as plt
            import networkx as nx
            from networkx.drawing.nx_pydot import graphviz_layout

            G = nx.DiGraph(evolution.tree)
            G = G.reverse()     # Make the graph top-down
            pos = graphviz_layout(G, prog='dot')
            plt.figure(figsize=(8, 8))
            nx.draw(G, pos, node_size=50, alpha=0.5, node_color=list(evolution.genx.values()), with_labels=False)
            plt.show()
        """
        self.tree = dict()
        self.id_genx = dict()
        self.id_score = dict()

        for gen, pop in self.history.items():
            for p in pop:
                self.tree[p.id] = p.parentIds if hasattr(p, "parentIds") else ()
                self.id_genx[p.id] = p.gIdx
                self.id_score[p.id] = p.fitness.score

    def info(self, plot=True, bestN=5, info=True, reverse=False):
        """Print and plot information about the evolution and the current population

        :param plot: plot a plot using `matplotlib`, defaults to True
        :type plot: bool, optional
        :param bestN: Print summary of `bestN` best individuals, defaults to 5
        :type bestN: int, optional
        :param info: Print information about the evolution environment
        :type info: bool, optional
        """
        if info:
            eu.printEvolutionInfo(self)
        validPop = self.getValidPopulation(self.pop)
        scores = self.getScores()
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
            # hack: during the evolution we need to use reverse=True
            # after the evolution (with evolution.info()), we need False
            try:
                self.plotProgress(reverse=reverse)
            except:
                logging.warning("Could not plot progress, is this a previously saved simulation?")
            eu.plotPopulation(
                self,
                plotScattermatrix=True,
                save_plots=self.trajectoryName,
                color=self.plotColor,
            )

    def plotProgress(self, reverse=False):
        """Plots progress of fitnesses of current evolution run"""
        eu.plotProgress(self, reverse=reverse)

    def saveEvolution(self, fname=None):
        """Save evolution to file using dill.

        :param fname: Filename, defaults to a path in ./data/
        :type fname: str, optional
        """
        import dill

        fname = fname or os.path.join("data/", "evolution-" + self.trajectoryName + ".dill")
        dill.dump(self, open(fname, "wb"))
        logging.info(f"Saving evolution to {fname}")

    def loadEvolution(self, fname):
        """Load evolution from previously saved simulatoins.

        Example usage:
        ```
        evaluateSimulation = lambda x: x # the funciton can be ommited, that's why we define a lambda here
        pars = ParameterSpace(['a', 'b'], # should be same as previously saved evolution
                      [[0.0, 4.0], [0.0, 5.0]])
        evolution = Evolution(evaluateSimulation, pars, weightList = [1.0])
        evolution = evolution.loadEvolution("data/evolution-results-2020-05-15-00H-24M-48S.dill")
        ```

        :param fname: Filename, defaults to a path in ./data/
        :type fname: str
        :return: Evolution
        :rtype: self
        """
        import dill

        evolution = dill.load(open(fname, "rb"))
        # parameter space is not saved correctly in dill, don't know why
        # that is why we recreate it using the values of
        # the parameter space in the dill
        pars = ParameterSpace(
            evolution.parameterSpace.parameterNames,
            evolution.parameterSpace.parameterValues,
        )

        evolution.parameterSpace = pars
        evolution.paramInterval = evolution.parameterSpace.named_tuple
        evolution.ParametersInterval = evolution.parameterSpace.named_tuple_constructor
        return evolution

    def _outputToDf(self, pop, df):
        """Loads outputs dictionary from evolution from the .outputs attribute
        and writes data into a dataframe.

        :param pop: Population of which to get outputs from.
        :type pop: list
        :param df: Dataframe to which outputs are written
        :type df: pandas.core.frame.DataFrame
        :return: Dataframe with outputs
        :rtype: pandas.core.frame.DataFrame
        """
        assert len(pop) == len(df), "Dataframe and population do not have same length."
        nan_value = np.nan
        # load outputs into dataframe
        for i, p in enumerate(pop):
            if hasattr(p, "outputs"):
                for key, value in p.outputs.items():
                    # only save floats, ints and arrays
                    if isinstance(value, (float, int, np.ndarray)):
                        # save 1-dim arrays
                        if isinstance(value, np.ndarray):
                            # to save a numpy array, convert column to object type
                            if key not in df:
                                df[key] = None
                            df[key] = df[key].astype(object)
                            df.at[i, key] = value
                        elif isinstance(value, (float, int)):
                            # save numbers
                            df.loc[i, key] = value
                    else:
                        df.loc[i, key] = nan_value
        return df

    def _dropDuplicatesFromDf(self, df):
        """Drops duplicates from dfEvolution dataframe.
        Tries vanilla drop_duplicates, which fails if the Dataframe contains
        data objects like numpy.arrays. Tries to drop via key "id" if it fails.

        :param df: Input dataframe with duplicates to drop
        :type df: pandas.core.frame.DataFrame
        :return: Dataframe without duplicates
        :rtype: pandas.core.frame.DataFrame
        """
        try:
            df = df.drop_duplicates()
        except:
            logging.info('Failed to drop_duplicates() without column name. Trying by column "id".')
            try:
                df = df.drop_duplicates(subset="id")
            except:
                logging.warning("Failed to drop_duplicates from dataframe.")
        return df

    def dfPop(self, outputs=False):
        """Returns a `pandas` DataFrame of the current generation's population parameters.
        This object can be further used to easily analyse the population.
        :return: Pandas DataFrame with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        # add the current population to the dataframe
        validPop = self.getValidPopulation(self.pop)
        indIds = [p.id for p in validPop]
        popArray = np.array([p[0 : len(self.paramInterval._fields)] for p in validPop]).T

        dfPop = pd.DataFrame(popArray, index=self.parameterSpace.parameterNames).T

        # add more information to the dataframe
        scores = self.getScores()
        dfPop["score"] = scores
        dfPop["id"] = indIds
        dfPop["gen"] = [p.gIdx for p in validPop]

        if outputs:
            dfPop = self._outputToDf(validPop, dfPop)

        # add fitness columns
        # NOTE: when loading an evolution with dill using loadingEvolution
        # MultiFitness values dissappear and only one is left.
        # See dfEvolution() for a solution using wvalues
        n_fitnesses = len(validPop[0].fitness.values)
        for i in range(n_fitnesses):
            for ip, p in enumerate(validPop):
                column_name = "f" + str(i)
                dfPop.loc[ip, column_name] = p.fitness.values[i]
        return dfPop

    def dfEvolution(self, outputs=False):
        """Returns a `pandas` DataFrame with the individuals of the the whole evolution.
        This method can be usef after loading an evolution from disk using loadEvolution()

        :return: Pandas DataFrame with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        parameters = self.parameterSpace.parameterNames
        allIndividuals = [p for gen, pop in self.history.items() for p in pop]
        popArray = np.array([p[0 : len(self.paramInterval._fields)] for p in allIndividuals]).T
        dfEvolution = pd.DataFrame(popArray, index=parameters).T
        # add more information to the dataframe
        scores = [float(p.fitness.score) for p in allIndividuals]
        indIds = [p.id for p in allIndividuals]
        dfEvolution["score"] = scores
        dfEvolution["id"] = indIds
        dfEvolution["gen"] = [p.gIdx for p in allIndividuals]

        if outputs:
            dfEvolution = self._outputToDf(allIndividuals, dfEvolution)

        # add fitness columns
        # NOTE: have to do this with wvalues and divide by weights later, why?
        # Because after loading the evolution with dill, somehow multiple fitnesses
        # dissappear and only the first one is left. However, wvalues still has all
        # fitnesses, and we have acces to weightList, so this hack kind of helps
        n_fitnesses = len(self.pop[0].fitness.wvalues)
        for i in range(n_fitnesses):
            for ip, p in enumerate(allIndividuals):
                dfEvolution.loc[ip, f"f{i}"] = p.fitness.wvalues[i] / self.weightList[i]

        # the history keeps all individuals of all generations
        # there can be duplicates (in elitism for example), which we filter
        # out for the dataframe
        dfEvolution = self._dropDuplicatesFromDf(dfEvolution)
        dfEvolution = dfEvolution.reset_index(drop=True)
        return dfEvolution

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

    def getScores(self):
        """Returns the scores of the current valid population"""
        validPop = self.getValidPopulation(self.pop)
        return np.array([pop.fitness.score for pop in validPop])

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
        if drop_first and "gen_000000" in generation_names:
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
