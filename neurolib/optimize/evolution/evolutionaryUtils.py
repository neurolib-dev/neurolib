import logging
import os

import numpy as np
import pandas as pd

from ...utils import paths as paths
from . import deapUtils as du


def saveToPypet(traj, pop, gIdx):
    try:
        traj.f_add_result_group("{}.gen_{:06d}".format("evolution", gIdx))
        traj.f_add_result("{}.gen_{:06d}.fitness".format("evolution", gIdx), np.array([p.fitness.values for p in pop]))
        traj.f_add_result("{}.gen_{:06d}.scores".format("evolution", gIdx), np.array([p.fitness.score for p in pop]))
        traj.f_add_result("{}.gen_{:06d}.population".format("evolution", gIdx), np.array([list(p) for p in pop]))
        traj.f_add_result("{}.gen_{:06d}.ind_ids".format("evolution", gIdx), np.array([p.id for p in pop]))
        # recursively store all simulated outputs into hdf
        for i, p in enumerate(pop):
            if not np.isnan(p.fitness.values).any() and not p.simulation_stored:
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
    except:
        logging.warn("Error: Write to pypet failed!")
    return pop


# visualization and infos


def printParamDist(pop=None, paramInterval=None, gIdx=None):
    print("Parameter distribution (Generation {}):".format(gIdx))
    for idx, k in enumerate(paramInterval._fields):
        print(
            "{}: \t mean: {:.4},\t std: {:.4}".format(
                k, np.mean([indiv[idx] for indiv in pop]), np.std([indiv[idx] for indiv in pop]),
            )
        )


def printIndividuals(pop, paramInterval, stats=True):
    print("Printing {} individuals".format(len(pop)))
    for i, ind in enumerate(pop):
        print(f"Individual {i}")
        
        print("\tFitness values: ", *np.round(ind.fitness.values, 2))
        if stats:
            print("\tScore: ", np.round(ind.fitness.score, 2))
            print("\tWeighted fitness: ", *np.round(ind.fitness.wvalues, 2))
            print(f"\tStats mean {np.mean(ind.fitness.values):.2f} std {np.std(ind.fitness.values):.2f} min {np.min(ind.fitness.values):.2f} max {np.max(ind.fitness.values):.2f}")
        for ki, k in enumerate(paramInterval._fields):
            print(f"\tmodel.params[\"{k}\"] = {ind[ki]:.2f}")
        


def plotScoresDistribution(scores, gIdx, save_plots=None, color='C0'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 2))
    plt.hist(scores, color=color, edgecolor="black", linewidth=1.2)
    plt.title("Generation: %i, Individuals: %i" % (gIdx, len(scores)))
    plt.xlabel("Score")
    plt.ylabel("Count")
    if save_plots is not None:
        logging.info("Saving plot to {}".format(os.path.join(paths.FIGURES_DIR, "%s_hist_%i.png" % (save_plots, gIdx))))
        plt.savefig(os.path.join(paths.FIGURES_DIR, "%s_hist_%i.png" % (save_plots, gIdx)))
    plt.show()


def plotSeabornScatter1(evolution, vars, save_plots, color='C0'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    sm = sns.pairplot(evolution.dfPop, vars=vars, 
                    diag_kind="kde", kind="reg", 
                    plot_kws={'line_kws':{'color': color}, 
                    'scatter_kws': {'alpha': 0.5, 'color' : color}},
                    diag_kws={'color' :  color})

    # adjust axis to parameter boundaries
    for axi, ax in enumerate(sm.axes):
        for ayi, ay in enumerate(ax):
            ay.set_ylim(evolution.paramInterval[axi])
            ay.set_xlim(evolution.paramInterval[ayi])

    if save_plots is not None:
        plt.savefig(os.path.join(paths.FIGURES_DIR, "{}_sns_params_{}.png".format(save_plots, evolution.gIdx)), bbox_inches='tight')
    plt.show()


def plotSeabornScatter2(evolution, vars, save_plots, color='C0'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    grid = sns.PairGrid(data=evolution.dfPop, vars=vars)
    grid = grid.map_upper(plt.scatter, color=color, alpha=0.5)
    grid = grid.map_diag(plt.hist, bins=10, color=color, edgecolor="k")
    grid = grid.map_lower(sns.kdeplot, cmap="Reds")

    # adjust axis to parameter boundaries
    for axi, ax in enumerate(grid.axes):
        for ayi, ay in enumerate(ax):
            # this block will only work if only parameters (and not other measures like score)
            # are plotted (since they don't have a range, and thus no xlim/ylim)
            try:
                ay.set_ylim(evolution.paramInterval[axi])
                ay.set_xlim(evolution.paramInterval[ayi])
            except:
                pass 
    if save_plots is not None:
        plt.savefig(os.path.join(paths.FIGURES_DIR, "{}_sns_params_red_{}.png".format(save_plots, evolution.gIdx)), bbox_inches='tight')
    plt.show()


def plotPopulation(
    evolution, plotDistribution=True, plotScattermatrix=False, save_plots=None, color='C0'
):

    """
    Print some stats of a population fitness
    """
    if save_plots:
        if not os.path.exists(paths.FIGURES_DIR):
            os.makedirs(paths.FIGURES_DIR)

    validPop = evolution.getValidPopulation()
    scores = evolution.getScores()
    print("There are {} valid individuals".format(len(validPop)))
    print("Mean score across population: {:.2}".format(np.mean(scores)))

    # plots can only be drawn if there are enough individuals, to avoid errors
    MIN_POP_SIZE_PLOTTING = 4
    vars_to_plot = list(evolution.parameterSpace.dict().keys())

    if len(validPop) > MIN_POP_SIZE_PLOTTING and plotDistribution:
        plotScoresDistribution(scores, evolution.gIdx, save_plots=save_plots, color=color)
        plotSeabornScatter1(evolution, vars=vars_to_plot, save_plots=save_plots, color=color)
        plotSeabornScatter2(evolution, vars=vars_to_plot, save_plots=save_plots, color=color)


def plotProgress(evolution, reverse=True):
    import matplotlib.pyplot as plt

    gens, all_scores = evolution.getScoresDuringEvolution(reverse=reverse)

    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3), dpi=150, sharex=True, gridspec_kw={"height_ratios" : [2, 1]})   
    im = axs[0].imshow(all_scores.T, aspect='auto', origin='lower')
    axs[0].set_ylabel("Individuals")
    cbar_ax = fig.add_axes([0.93, 0.42, 0.02, 0.3])
    fig.colorbar(im, cax=cbar_ax, label='Score')

    axs[1].plot(gens, np.nanmean(all_scores, axis=1), label='Average')
    axs[1].fill_between(gens, np.nanmin(all_scores, axis=1), np.nanmax(all_scores, axis=1), alpha=0.3, label='Bound')
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Score")
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
    print("> Evolutionary operators")
    print(f"Mating operator: {evolution.matingOperator}")
    print(f"Mating paramter: {evolution.MATE_P}")
    print(f"Selection operator: {evolution.selectionOperator}")
    print(f"Selection paramter: {evolution.SELECT_P}")
    print(f"Parent selection operator: {evolution.parentSelectionOperator}")
    if len(evolution.comments) > 0:
        if isinstance(evolution.comments, str):
            print(f"Comments: {evolution.comments}")
        elif isinstance(evolution.comments, list):
            for c in evolution.comments:
                print(f"Comments: {c}")
