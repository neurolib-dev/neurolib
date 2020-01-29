import datetime
import os
import random 
import copy
import logging
import multiprocessing

import deap
from deap import base
from deap import creator
from deap import tools

import collections
import numpy as np
import pypet as pp

import neurolib.utils.paths as paths

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.optimize.evolution.deapUtils as du

class Evolution:
    def __init__(self, model, evalFunction, weightList, hdf_filename='evolution.hdf', ncores = None, POP_INIT_SIZE = 100, POP_SIZE=20, NGEN = 10, CXPB = 1 - 0.96):
        """Evolutionary parameter optimization
        
        :param model: Model to run
        :type model: Model
        :param evalFunction: Evaluation function that outputs the fitness vector
        :param weightList: List of floats that defines the dimensionality of the fitness vector returned from evalFunction and the weights of each component (positive = maximize, negative = minimize) 

        :param hdf_filename: HDF file to store all results in (data/hdf/evolution.hdf default)
        :param ncores: Number of cores to simulate on (max cores default)
        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed)
        :param POP_SIZE: Size of the population during evolution
        :param NGEN: Numbers of generations to evaluate
        :param CXPB: Crossover probability of each individual gene
        """
        trajectoryName = 'results' + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        HDF_FILE = os.path.join(paths.HDF_DIR, hdf_filename)
        trajectoryFileName = HDF_FILE

        logging.info("Storing data to: {}".format(trajectoryFileName))
        logging.info("Trajectory Name: {}".format(trajectoryName))
        if ncores is None:
            ncores = multiprocessing.cpu_count()
        logging.info("Number of cores: {}".format(ncores))        

        # initialize pypet environment
        env = pp.Environment(trajectory=trajectoryName,filename=trajectoryFileName,
                            file_title='Evolutionary optimization',
                            large_overview_tables=True,
                            multiproc=True,           
                            ncores=ncores,
                            wrap_mode='QUEUE',
                            log_stdout = False,
                            automatic_storing=False,
                            complevel = 9
                            )

        # Get the trajectory from the environment
        traj = env.traj
        trajectoryName = traj.v_name


        self.model = model
        self.evalFunction = evalFunction
        self.weightList = weightList

        self.CXPB = CXPB
        self.NGEN = NGEN
        self.POP_SIZE = POP_SIZE
        self.POP_INIT_SIZE = POP_INIT_SIZE
        self.ncores = ncores 

        self.traj = env.traj        
        self.env = env
        self.trajectoryName = trajectoryName
        self.trajectoryFileName = trajectoryFileName

        # ------------- define parameters
        self.ParametersInterval = collections.namedtuple('ParametersInterval',['mue_ext_mean', 'mui_ext_mean', 'sigma_ou'])
        self.paramInterval = self.ParametersInterval([0.0, 4.0], [0.0, 4.0], [0.01, 0.3])
        self.toolbox = deap.base.Toolbox()
        self.initDEAP(self.toolbox, self.env, self.paramInterval, self.evalFunction, weights_list = self.weightList)

        # set up pypet trajectory
        self.initPypetTrajectory(self.traj, self.paramInterval, self.ParametersInterval, \
            self.POP_SIZE, self.CXPB, self.NGEN, self.model.params)

        # ------------- initialize population
        #pop = toolbox.population(n=traj.popsize)
        self.pop = self.toolbox.population(n=self.POP_INIT_SIZE)

        for i in range(len(self.pop)):
            self.pop[i].id = i
            self.pop[i].simulation_stored = False

        self.last_id = self.pop[-1].id    

    def loadIndividual(self, traj):
        def getIndividual(traj):
            # either pass an individual or a pypet trajectory with the attribute individual
            if type(traj).__name__ == 'Individual':
                individual = traj
            else:
                individual = traj.individual
            return individual

        model = self.model
        model.params.update(self.individualToDict(getIndividual(traj)))
        return model

    def individualToDict(self, individual):
        """
        Convert an individual to a dictionary
        """
        return self.ParametersInterval(*(individual[: len(self.paramInterval)]))._asdict().copy()

    def printParamDist(self, pop, paramInterval):
        print("Parameters dictribution:")
        for idx, k in enumerate(paramInterval._fields):
            print('{}: \t mean: {:.4},\t std: {:.4}'.format(k,
                                    np.mean([indiv[idx] for indiv in pop]),
                                    np.std([indiv[idx] for indiv in pop]) ))

    def initPypetTrajectory(self, traj, paramInterval, ParametersInterval, POP_SIZE, CXPB, NGEN, params):
        # Initialize pypet trajectory and add all simulation parameters
        traj.f_add_parameter('popsize', POP_SIZE, comment='Population size') # 
        traj.f_add_parameter('CXPB',    CXPB,   comment='Crossover term') # Crossover probability
        traj.f_add_parameter('NGEN',    NGEN, comment='Number of generations')

        # Placeholders for individuals and results that are about to be explored
        traj.f_add_parameter('generation', 0, comment='Current generation')

        traj.f_add_result('scores', [], comment='Mean_score for each generation')
        traj.f_add_result_group("evolution", comment='Contains results for each generation')
        traj.f_add_result_group("outputs", comment='Contains simulation results')

        traj.f_add_result('params', params, comment='Default parameters')

        # todo: initialize this after individuals have been defined!
        traj.f_add_parameter('id', 0, comment='Index of individual')
        traj.f_add_parameter('ind_len', 20, comment='Length of individual')    
        traj.f_add_derived_parameter('individual', [0 for x in range(traj.ind_len)], 'An indivudal of the population')

    def initDEAP(self, toolbox, pypetEnvironment, paramInterval, evalFunction, weights_list):
        # ------------- register everything in deap
        deap.creator.create("FitnessMulti", deap.base.Fitness, weights=tuple(weights_list))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        
        # initially, each individual has randomized genes
        # need to create a lambda funciton because du.generateRandomParams wants an argument but
        # toolbox.register cannot pass an argument to it.
        toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, lambda: du.generateRandomParams_withAdaptation(paramInterval))
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual) 
        
        # Operator registering
        toolbox.register("selBest",     du.selBest_multiObj)
        toolbox.register("selRank",     du.selRank)
        
        toolbox.register("evaluate",    evalFunction)

        toolbox.register("mate",        du.cxUniform_normDraw_adapt)
        toolbox.register("mutate",      du.adaptiveMutation_nStepSize)
        toolbox.register("map",         pypetEnvironment.run)
        toolbox.register("run_map",     pypetEnvironment.run_map)         

    def evalPopulationUsingPypet(self, traj,toolbox,pop,gIdx):
        '''
        Eval the population fitness using pypet
        
        Params:
            pop:    List of inidivual (population to evaluate)
            gIdx:   Index of the current generation
            
        Return:
            The population with updated fitness values
        ''' 
        
        # Add as many explored runs as individuals that need to be evaluated.
        # Furthermore, add the individuals as explored parameters.
        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`:
        # This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(pp.cartesian_product({'generation': [gIdx],
                                            #'ind_idx': range(len(pop)),
                                            'id': [x.id for x in pop],
                                            'individual':[list(x) for x in pop]},
                                            [('id', 'individual'),'generation']))
        
        #traj.f_expand(cartesian_product({'generation': [gIdx], 'ind_idx': range(len(pop))}))
        
        # SIMULUATE INDIVIDUALS
        results = toolbox.map(toolbox.evaluate)

        for idx, result in enumerate(results):
            runIdx, packed_result = result
            fitnesses_result, outputs = packed_result
            
            # store outputs of simulations in population
            pop[idx].outputs = outputs
            pop[idx].fitness.values = fitnesses_result
            # mean fitness value
            pop[idx].fitness.score = np.nansum(pop[idx].fitness.wvalues) / (len(pop[idx].fitness.wvalues))
        
        return pop   

    def runInitial(self):
        ### Evaluate the initial population
        print("Evaluating initial population of size %i ..."%len(self.pop))
        self.evalPopulationUsingPypet(self.traj,self.toolbox,self.pop,0)
        self.gIdx = 0 # set generation index

        self.printParamDist(self.pop, self.paramInterval)
        self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)

        # Only the best indviduals are selected for the population the others do not survive
        self.pop[:] = self.toolbox.selBest(self.pop, k = self.traj.popsize)

        for p in self.pop:
            p.isOffspring = False
            p.isCrossOver = False

    def runEvolution(self):
        # Start evolution
        print("Start of evolution")
        for self.gIdx in range(self.gIdx+1,self.gIdx+self.traj.NGEN):
            # ------- Weed out the invalid individuals and replace them by random new indivuals -------- #
            validpop = [p for p in self.pop if not np.any(np.isnan(p.fitness.values))]
            nanpop = [p for p in self.pop if np.any(np.isnan(p.fitness.values))]
            # replace invalid individuals
            print("Replacing {} invalid individuals.".format(len(nanpop)))
            newpop = self.toolbox.population(n=len(nanpop))
            for i, n in enumerate(newpop):
                n.id = self.last_id + i + 1
                n.simulation_stored = False

            #pop = validpop + newpop    
            
            # ------- Create the next generation by crossover and mutation -------- #
            ### Select parents using rank selection and clone them ###
            offspring = list(map( self.toolbox.clone, self.toolbox.selRank(self.pop,self.POP_SIZE) ))
            for i, o in enumerate(offspring):
                o.isOffspring = True # mark them for statistics
                o.id = self.last_id + i + 1
                o.simulation_stored = False
            
            # increase the id counter
            self.last_id = self.last_id + len(offspring)
            
            ##### cross-over ####
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2, indpb = self.CXPB)
                del child1.fitness.values
                del child2.fitness.values
                    
            ##### Mutation ####
            # Apply adaptive mutation
            eu.mutateUntilValid(offspring, self.paramInterval, self.toolbox)
            
            # ------- Evaluate next generation -------- #
            
            print("----------- Generation %i -----------" % self.gIdx)

            self.evalPopulationUsingPypet(self.traj,self.toolbox,offspring + newpop,self.gIdx)
            
            # ------- Select surviving population -------- #
            
            # select next population
            self.pop = self.toolbox.selBest(validpop + offspring + newpop, k = self.traj.popsize)
            
            self.best_ind = self.toolbox.selBest(self.pop, 1)[0]
            print("Best individual is {}".format(self.best_ind))
            print("Score: {}".format(self.best_ind.fitness.score))        
            print("Fitness: {}".format(self.best_ind.fitness.values))
            
            print( "--- Population statistics ---")
            self.printParamDist(self.pop, self.paramInterval)
            eu.printPopFitnessStats(self.pop, self.paramInterval, self.gIdx, draw_scattermatrix = True)
            
            # ------- Save data using pypet
            try:
                self.pop = eu.saveToPypet(self.traj, self.pop, self.gIdx)
            except:
                print("Error: Write to pypet failed!")

            #print( "Offspring in the new pop: %f %%"%(1.*len([i for i in pop if i.isOffspring])
            #                                                          /len(pop)))

            # unmark offsprings
            for iv in self.pop:
                o.isOffspring = False

        print("--- End of evolution ---")
        self.best_ind = self.toolbox.selBest(self.pop, 1)[0]
        print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        print("--- End of evolution ---")

        self.traj.f_store()  # We switched off automatic storing, so we need to store manually            