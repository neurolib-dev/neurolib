import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import tqdm

from ...utils import functions as func


def plotExplorationResults(
    dfResults,
    par1,
    par2,
    plot_key,
    by=None,
    by_label=None,
    plot_key_label=None,
    symmetric_colorbar=False,
    one_figure=False,
):
    """
    """
    # copy here, because we add another column that we do not want to keep later
    dfResults = dfResults.copy()

    if isinstance(par1, str):
        par1_label = par1
    elif isinstance(par1, (list, tuple)):
        par1_label = par1[1]
        par1 = par1[0]

    if isinstance(par2, str):
        par2_label = par2
    elif isinstance(par2, (list, tuple)):
        par2_label = par2[1]
        par2 = par2[0]

    # if no by value was given, create a dummy value
    if by is None:
        dfResults["_by"] = 0
        by = ["_by"]

    n_plots = len(dfResults.groupby(by=by))

    if by_label is None:
        by_label = by

    # if by is not None:
    #     n_plots = len(dfResults.groupby(by=by))
    # else:
    #     n_plots = 1

    if one_figure == True:
        fig, axs = plt.subplots(nrows=1, ncols=n_plots, figsize=(n_plots * 4, 3.5), dpi=150)

    # if by is not None:
    #     _dfResults = dfResults.groupby(by=by)
    # else:
    #     _dfResults = dfResults

    axi = 0
    for i, df in dfResults.groupby(by=by):

        if one_figure == True:
            if n_plots > 1:
                ax = axs[axi]
            else:
                ax = axs

        df_pivot = df.pivot_table(values=plot_key, index=par2, columns=par1)

        plot_clim = None
        if symmetric_colorbar:
            plot_clim = (-np.max(df_pivot.values), np.max(df_pivot.values))

        if one_figure == False:
            fig = plt.figure(figsize=(5, 4), dpi=150)
            ax = plt.gca()

        im = ax.imshow(
            df_pivot,
            extent=[min(df[par1]), max(df[par1]), min(df[par2]), max(df[par2])],
            origin="lower",
            aspect="auto",
            clim=plot_clim,
        )

        if one_figure == False:
            cbar = plt.colorbar(im, ax=ax, orientation="vertical", label=plot_key_label)
        else:
            # # below is code for one colorbar only but it only displays the values of
            # # the last plot so it's useless
            # # if this is the last plot
            # if axi == n_plots - 1:
            #     cbar_ax = fig.add_axes([0.91, 0.18, 0.005, 0.65])
            #     cbar = fig.colorbar(im, cax=cbar_ax, label=plot_key_label)

            # colorbar per plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        ax.set_xlabel(par1_label)
        ax.set_ylabel(par2_label)

        # single by-values need to become tuple
        if not isinstance(i, tuple):
            i = (i,)

        ax.set_title(" ".join([f"{bb}={bi}" for bb, bi in zip(by_label, i)]))
        if one_figure == False:
            plt.show()
        else:
            axi += 1

    if one_figure == True:
        plt.tight_layout()
        plt.show()


def processExplorationResults(results, dfResults, **kwargs):
    """Process results from the exploration. 
    """

    #
    # if bold_transient is given as an argument
    if "bold_transient" in kwargs:
        bold_transient = kwargs["bold_transient"]
        logging.info(f"Bold transient: {bold_transient} ms")
    else:
        # else, set it to 10 seconds
        bold_transient = 10000

    # cycle through all results
    for i in tqdm.tqdm(dfResults.index):
        # if the result has an output
        if "t" in results[i].keys() or "outputs.t" in results[i].keys():
            # if a dataset was passed as an argument
            if "ds" in kwargs:
                ds = kwargs["ds"]

                # calculate mean correlation of functional connectivity
                # of the simulation and the empirical data
                dfResults.loc[i, "fc"] = np.mean(
                    [
                        func.matrix_correlation(
                            func.fc(results[i]["BOLD"][:, results[i]["t_BOLD"] > bold_transient]), fc,
                        )
                        for fc in ds.FCs
                    ]
                )

            # if the model is passed as an argument
            if "model" in kwargs:
                model = kwargs["model"]

                # get the output of the model
                if "output" in kwargs:
                    output_name = kwargs["output"]
                else:
                    output_name = model.default_output

                # use the last x ms for analysis
                if "output_last_ms" in kwargs:
                    last_ms = kwargs["output_last_ms"]
                    logging.info(f"Analyzing last {bold_transient} ms of the output {output_name}.")
                else:
                    last_ms = 1000

                # calculate the maximum of the output
                dfResults.loc[i, "max_" + output_name] = np.max(
                    results[i][output_name][:, -int(last_ms / model.params["dt"]) :]
                )

                # calculate the amplitude of the output
                dfResults.loc[i, "amp_" + output_name] = np.max(
                    np.max(results[i][output_name][:, -int(last_ms / model.params["dt"]) :], axis=1)
                    - np.min(results[i][output_name][:, -int(last_ms / model.params["dt"]) :], axis=1)
                )
    return dfResults


def findCloseResults(dfResults, dist=0.01, **kwargs):
    """Usage: findCloseResults(search.dfResults, mue_ext_mean=2.0, mui_ext_mean=2.5)
    """
    # dist = 0.01
    selectors = True
    for key, value in kwargs.items():
        new_selector = abs(dfResults[key] - value) < dist
        selectors = selectors & new_selector
    filtered_df = dfResults[selectors]
    return filtered_df


def paramsRun(dfResults, runNr):
    return dfResults.loc[runNr].to_dict()
