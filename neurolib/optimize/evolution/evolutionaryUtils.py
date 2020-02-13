import logging
import datetime
import os
import random
import copy

import deap
import numpy as np
import pypet as pp
import pandas as pd

import neurolib.utils.paths as paths
import neurolib.optimize.evolution.deapUtils as du


def saveToPypet(traj, pop, gIdx):
    traj.f_add_result_group("{}.gen_{:06d}".format("evolution", gIdx))
    traj.f_add_result("{}.gen_{:06d}.fitness".format("evolution", gIdx), np.array([p.fitness.values for p in pop]))
    traj.f_add_result("{}.gen_{:06d}.scores".format("evolution", gIdx), np.array([p.fitness.score for p in pop]))
    traj.f_add_result("{}.gen_{:06d}.population".format("evolution", gIdx), np.array([list(p) for p in pop]))
    traj.f_add_result("{}.gen_{:06d}.ind_ids".format("evolution", gIdx), np.array([p.id for p in pop]))
    # recursively store all simulated outputs into hdf
    for i, p in enumerate(pop):
        if not np.any(np.isnan(p.fitness.values)) and not p.simulation_stored:
            pop[i].simulation_stored = True

            traj.f_add_result_group("{}.ind_{:06d}".format("outputs", p.id))

            assert isinstance(p.outputs, dict), "outputs are not a dict, can't unpack!"

            def unpackOutputsAndStore(outputs, save_string):
                new_save_string = save_string
                for key, value in outputs.items():
                    if isinstance(value, dict):
                        new_save_string = save_string + "." + key
                        traj.f_add_result_group(new_save_string)
                        unpackOutputsAndStore(value, new_save_string)
                    else:
                        traj.f_add_result("{}.{}".format(new_save_string, key), value)

            unpackOutputsAndStore(p.outputs, save_string="{}.ind_{:06d}".format("outputs", p.id))

    traj.f_store()
    return pop


# visualization and infos


def printParamDist(pop=None, paramInterval=None, gIdx=None):
    if pop == None:
        pop = self.pop
    if paramInterval == None:
        paramInterval = self.paramInterval

    print("Parameter dictribution (Generation {}):".format(gIdx))
    for idx, k in enumerate(paramInterval._fields):
        print(
            "{}: \t mean: {:.4},\t std: {:.4}".format(
                k, np.mean([indiv[idx] for indiv in pop]), np.std([indiv[idx] for indiv in pop]),
            )
        )


def printIndividuals(pop, paramInterval, stats=False):
    print("Printing {} individuals".format(len(pop)))
    pars = []
    for i, ind in enumerate(pop):
        thesepars = {}
        for ki, k in enumerate(paramInterval._fields):
            thesepars[k] = ind[ki]
        thesepars["fit"] = np.mean(ind.fitness.values)
        pars.append(thesepars)
        print(
            "Individual",
            i,
            "pars",
            ", ".join([" ".join([k, "{0:.4}".format(ind[ki])]) for ki, k in enumerate(paramInterval._fields)]),
        )
        print("\tFitness values: ", *np.round(ind.fitness.values, 4))
        if stats:
            print(
                "\t > mean {0:.4}, std {0:.4}, min {0:.4} max {0:.4}".format(
                    np.mean(ind.fitness.values),
                    np.std(ind.fitness.values),
                    np.min(ind.fitness.values),
                    np.max(ind.fitness.values),
                )
            )


def plotScoresDistribution(scores, gIdx, save_plots=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 2))
    plt.hist(scores, color="grey", edgecolor="black", linewidth=1.2)
    plt.title("Generation: %i, Individuals: %i" % (gIdx, len(scores)))
    plt.xlabel("Score")
    plt.ylabel("Count")
    if save_plots is not None:
        logging.info("Saving plot to {}".format(os.path.join(paths.FIGURES_DIR, "%s_hist_%i.png" % (save_plots, gIdx))))
        plt.savefig(os.path.join(paths.FIGURES_DIR, "%s_hist_%i.png" % (save_plots, gIdx)))
    plt.show()


def plotSeabornScatter1(dfPop, pop, paramInterval, gIdx, save_plots):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    sm = sns.pairplot(dfPop, diag_kind="kde", kind="reg")
    if save_plots is not None:
        plt.savefig(os.path.join(paths.FIGURES_DIR, "{}_sns_params_{}.png".format(save_plots, gIdx)))
    plt.show()


def plotSeabornScatter2(dfPop, pop, paramInterval, gIdx, save_plots):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    grid = sns.PairGrid(data=dfPop)
    grid = grid.map_upper(plt.scatter, color="darkred", alpha=0.5)
    grid = grid.map_diag(plt.hist, bins=10, color="darkred", edgecolor="k")
    grid = grid.map_lower(sns.kdeplot, cmap="Reds")
    if save_plots is not None:
        plt.savefig(os.path.join(paths.FIGURES_DIR, "{}_sns_params_red_{}.png".format(save_plots, gIdx)))
    plt.show()


def plotPopulation(
    pop, paramInterval, gIdx=0, plotDistribution=True, plotScattermatrix=False, save_plots=None,
):

    """
    Print some stats of a population fitness
    """
    if save_plots:
        if not os.path.exists(paths.FIGURES_DIR):
            os.makedirs(paths.FIGURES_DIR)

    # Gather all the fitnesses in one list and print the stats
    # validPop = [p for p in pop if not np.isnan(p.fitness.score)]
    validPop = [p for p in pop if not np.any(np.isnan(p.fitness.values))]
    popArray = np.array([p[0 : len(paramInterval._fields)] for p in validPop]).T
    scores = np.array([validPop[i].fitness.score for i in range(len(validPop))])
    print("There are {} valid individuals".format(len(validPop)))
    print("Mean score across population: {:.2}".format(np.mean(scores)))

    # plots can only be drawn if there are enough individuals
    MIN_POP_SIZE_PLOTTING = 4
    if len(validPop) > MIN_POP_SIZE_PLOTTING and plotDistribution:
        plotScoresDistribution(scores, gIdx, save_plots)

    if len(validPop) > MIN_POP_SIZE_PLOTTING and plotScattermatrix:
        # make a pandas dataframe for the seaborn pairplot
        gridParameters = [k for idx, k in enumerate(paramInterval._fields)]
        dfPop = pd.DataFrame(popArray, index=gridParameters).T
        dfPop = dfPop.loc[:, :]
        plotSeabornScatter1(dfPop, pop, paramInterval, gIdx, save_plots)
        plotSeabornScatter2(dfPop, pop, paramInterval, gIdx, save_plots)


def plotProgress(evolution, drop_first=True):
    gens, all_scores = evolution.getScoresDuringEvolution(reverse=False, drop_first=drop_first)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(gens, np.nanmean(all_scores, axis=1))
    plt.fill_between(gens, np.nanmin(all_scores, axis=1), np.nanmax(all_scores, axis=1), alpha=0.3)
    plt.xlabel("Generation #")
    plt.ylabel("Score")
    plt.show()


def printEvolutionInfo(evolution):
    """Function that prints all important parameters of the evolution.
    :param evolution: evolution object
    """
    print("> Simulation parameters")
    print(f"HDF file storage: {evolution.trajectoryFileName}")
    print(f"Trajectory Name: {evolution.trajectoryName}")
    if hasattr(evolution, "_t_end_initial_population"):
        print(
            f"Duration of evaluating initial population {evolution._t_end_initial_population-evolution._t_start_initial_population}"
        )
    if hasattr(evolution, "_t_end_evolution"):
        print(f"Duration of evolution {evolution._t_end_evolution-evolution._t_start_evolution}")
    if evolution.model is not None:
        print(f"Model: {type(evolution.model)}")
        if hasattr(evolution.model, "name"):
            if isinstance(evolution.model.name, str):
                print(f"Model name: {evolution.model.name}")
    print(f"Eval function: {evolution.evalFunction}")
    print(f"Parameter space: {evolution.paramInterval}")
    # print(f"Weights: {evolution.weightList}")
    print("> Evolution parameters")
    print(f"Number of generations: {evolution.NGEN}")
    print(f"Initial population size: {evolution.POP_INIT_SIZE}")
    print(f"Population size: {evolution.POP_SIZE}")
    print(f"Crossover paramter: {evolution.CXP}")
    print(f"Selection paramter: {evolution.RANKP}")
    if len(evolution.comments) > 0:
        if isinstance(evolution.comments, str):
            print(f"Comments: {evolution.comments}")
        elif isinstance(evolution.comments, list):
            for c in evolution.comments:
                print(f"Comments: {c}")
