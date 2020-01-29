import datetime
import os
import random 
import copy

import deap
import matplotlib.pyplot as plt
import numpy as np
import pypet as pp
import pandas as pd
import seaborn as sns

import neurolib.utils.paths as paths 
import neurolib.optimize.evolution.deapUtils as du


def printParamDist(pop, paramInterval):
    print("Parameters dictribution:")
    for idx, k in enumerate(paramInterval._fields):
        print('{}: \t mean: {:.4},\t std: {:.4}'.format(k,
                                   np.mean([indiv[idx] for indiv in pop]),
                                   np.std([indiv[idx] for indiv in pop]) ))

from pandas.plotting import scatter_matrix

def printPopFitnessStats(pop, paramInterval, gIdx = 0, draw_distribution = True, draw_scattermatrix = False, save_plots = None):
    '''
    Print some stats of a population fitness
    '''
    # Gather all the fitnesses in one list and print the stats
    #selectPop = [p for p in pop if not np.isnan(p.fitness.score)]
    selectPop = [p for p in pop if p.fitness.score > 0]
    candidates = np.array([p[0:len(paramInterval._fields)] for p in selectPop]).T
    scores = np.array([selectPop[i].fitness.score for i in range(len(selectPop))])
    print("There are {} valid individuals".format(len(selectPop)))
    print("Mean score across population: {:.2}".format(np.mean(scores)))
    
    if draw_distribution:
        plt.figure(figsize=(4, 2))
        plt.hist(scores, color='grey');
        plt.title("Generation: %i, Individuals: %i"%(gIdx,len(scores)))
        plt.xlabel('Score')
        plt.ylabel('Count')
        if save_plots is not None:
            plt.savefig(os.path.join(paths.FIGURES_DIR, '%s_hist_%i.jpg'%(save_plots,gIdx)))
        plt.show()
    
    if draw_scattermatrix:
        plt.figure()
        gridParameters = [k for idx, k in enumerate(paramInterval._fields)]
        pcandidates = pd.DataFrame(candidates, index=gridParameters).T
        pcandidates = pcandidates.loc[:, :]

        sm = sns.pairplot(pcandidates, diag_kind="kde", kind="reg")
        
        if save_plots is not None:
            plt.savefig(os.path.join(paths.FIGURES_DIR, '%s_sns_params_%i.jpg'%(save_plots,gIdx)))        
        
        plt.show()
        
        # SEABORN TESTING
        # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
        # Create an instance of the PairGrid class.
        grid = sns.PairGrid(data= pcandidates)
        # Map a scatter plot to the upper triangle
        grid = grid.map_upper(plt.scatter, color = 'darkred', alpha=0.5)
        # Map a histogram to the diagonal
        grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                             edgecolor = 'k')
        # Map a density plot to the lower triangle
        grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
        
        #print(len(scores))
        #sm = scatter_matrix(pcandidates, grid=True, s=scores*100, alpha=0.8, figsize=(12, 12), diagonal='kde', c='K');
        #for ax in sm.ravel():
        #    ax.set_xlabel(ax.get_xlabel(), fontsize = 14, rotation = 0)
        #    ax.set_ylabel(ax.get_ylabel(), fontsize = 14, rotation = 90)        
        
        if save_plots is not None:
            plt.savefig(os.path.join(paths.FIGURES_DIR, '%s_sns2_params_%i.jpg'%(save_plots,gIdx)))        
        plt.show()

def countLiving(pop):
    nValid = 0 
    for i in range(len(pop)): nValid += 1 if not np.isnan(pop[i].fitness.score) else 0
    #print("Number of living individuals: %i"%nValid)
    return nValid


def saveToPypet(self, traj, pop, gIdx):
    traj.f_add_result_group('{}.gen_{:06d}'.format("evolution", gIdx))
    traj.f_add_result('{}.gen_{:06d}.fitness'.format("evolution", gIdx), np.array([p.fitness.values for p in pop]))
    traj.f_add_result('{}.gen_{:06d}.scores'.format("evolution", gIdx), np.array([p.fitness.score for p in pop]))
    traj.f_add_result('{}.gen_{:06d}.population'.format("evolution", gIdx), np.array([list(p) for p in pop]))
    traj.f_add_result('{}.gen_{:06d}.ind_ids'.format("evolution", gIdx), np.array([p.id for p in pop]))
    
    # recursively store all simulated outputs into hdf
    for i, p in enumerate(pop):
        if not np.any(np.isnan(p.fitness.values)) and not p.simulation_stored:
            pop[i].simulation_stored = True

            traj.f_add_result_group('{}.ind_{:06d}'.format("outputs", p.id))
            
            assert isinstance(p.outputs, dict), 'outputs are not a dict, can\'t unpack!'

            def unpackOutputsAndStore(outputs, save_string):
                new_save_string = save_string
                for key, value in outputs.items():
                    if isinstance(value, dict):
                        new_save_string = save_string + "." + key
                        traj.f_add_result_group(new_save_string)
                        unpackOutputsAndStore(value, new_save_string)
                    else:
                        traj.f_add_result('{}.{}'.format(new_save_string, key), value)
            unpackOutputsAndStore(p.outputs, save_string = '{}.ind_{:06d}'.format("outputs", p.id))

    traj.f_store()
    return pop   
    
def mutateUntilValid(offspring, paramInterval, toolbox, maxTries = 100):
    #### Check validity of new individuals ### 
    # mutate individuald until valid, max 100 times
    for i, o in enumerate(offspring):
        o_bak = copy.copy(o)
        toolbox.mutate(offspring[i])
        nMutations = 0
        while not du.check_param_validity(offspring[i], paramInterval) and nMutations < maxTries:
            #print("Gen {} Offspring {}, not valid, repeating mutation...".format(gIdx, i))
            offspring[i] = copy.copy(o_bak)
            toolbox.mutate(offspring[i])
            nMutations += 1
        del offspring[i].fitness.values
        #print("> Offspring {}, {} mutations tried".format(i, nMutations))    
