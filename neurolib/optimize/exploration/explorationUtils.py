import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import tqdm

from scipy import stats

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
    **kwargs
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

    # the `by` argument is used to slice the data
    # if no by value was given, create a dummy value
    if by is None:
        dfResults["_by"] = 0
        by = ["_by"]

    if by_label is None:
        by_label = by        

    n_plots = len(dfResults.groupby(by=by))


    # create subfigures
    if one_figure == True:
        fig, axs = plt.subplots(nrows=1, ncols=n_plots, figsize=(n_plots * 4, 3.5), dpi=150)
        if plot_key_label:
            fig.suptitle(plot_key_label)

    axi = 0
    # cycle through all slices
    for i, df in dfResults.groupby(by=by):
        # chose the current axis
        if one_figure == True:
            if n_plots > 1:
                ax = axs[axi]
            else:
                ax = axs
        # pivot data
        df_pivot = df.pivot_table(values=plot_key, index=par2, columns=par1, dropna=False)

        plot_clim = None
        if symmetric_colorbar:
            plot_clim = (-np.max(df_pivot.values), np.max(df_pivot.values))

        if one_figure == False:
            fig = plt.figure(figsize=(5, 4), dpi=150)
            if plot_key_label:
                plt.title(plot_key_label)
            ax = plt.gca()

        image_extent = [min(df[par1]), max(df[par1]), min(df[par2]), max(df[par2])]
        im = ax.imshow(
            df_pivot,
            extent=image_extent,
            origin="lower",
            aspect="auto",
            clim=plot_clim,
        )

        # plot contours
        if "contour" in kwargs:
            # check if this is a dataframe
            if isinstance(kwargs["contour"], pd.DataFrame):
                contour_color = kwargs["contour_color"] if "contour_color" in kwargs else "white"
                contourPlotDf(kwargs["contour"], color=contour_color, ax=ax)


        # colorbar
        if one_figure == False:
            # useless and wrong if images don't have the same range! should be used only if plot_clim is used
            cbar = plt.colorbar(im, ax=ax, orientation="vertical", label=plot_key_label)
        else:
            # colorbar per plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        # labels and style
        ax.set_xlabel(par1_label)
        ax.set_ylabel(par2_label)

        # single by-values need to become tuple
        if not isinstance(i, tuple):
            i = (i,)

        if by != ["_by"]:
            ax.set_title(" ".join([f"{bb}={bi}" for bb, bi in zip(by_label, i)]))
        if one_figure == False:
            plt.show()
        else:
            axi += 1

    if one_figure == True:
        plt.tight_layout()
        plt.show()


def contourPlotDf(dataframe, color="white", ax=None):
    #Xi, Yi = np.meshgrid(range(len(dataframe.columns)), range(len(dataframe.index)))
    Xi, Yi = np.meshgrid(dataframe.columns, dataframe.index)
    ax = ax or plt
    cset2 = ax.contour(
        Xi, Yi, dataframe, colors=color, linestyles="solid", levels=[0, 1.0001], linewidths=(4,), zorder=1
    )

def plotResult(search, runId, z_bold = False, **kwargs):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2), dpi=300, gridspec_kw={'width_ratios': [1, 1.2, 2]})

    if "bold_transient" in kwargs:
        bold_transient = int(kwargs["bold_transient"] / 2)
    else:
        bold_transient = int(30 / 2)

    bold = search.results[runId].BOLD[:, bold_transient:]
    bold_z = stats.zscore(bold, axis=1)
    t_bold = np.linspace(2, len(bold.T)*2, len(bold.T), )

    output = search.results[runId].output[:, :]
    output_dt = search.model.params.dt
    t_output = np.linspace(output_dt, len(output.T)*output_dt, len(output.T), )

    axs[0].set_title(f"FC (run {runId})")
    axs[0].imshow(func.fc(bold))
    axs[0].set_ylabel("Node")
    axs[0].set_xlabel("Node")
    
    #axs[1].set_title("BOLD")
    if z_bold:
        axs[1].plot(t_bold, bold_z.T, lw=1.5, alpha=0.8);
    else:
        axs[1].plot(t_bold, bold.T, lw=1.5, alpha=0.8);
    axs[1].set_xlabel("Time [s]")
    if z_bold:
        axs[1].set_ylabel("Normalized BOLD")
    else:
        axs[1].set_ylabel("BOLD")

    axs[2].set_ylabel("Activity")
    axs[2].plot(t_output, output.T, lw=1.5, alpha=0.6);
    axs[2].set_xlabel("Time [ms]")
    if "xlim" in kwargs:
        axs[2].set_xlim(kwargs["xlim"])
    plt.tight_layout()

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
        #if "t" in results[i].keys() or "outputs.t" in results[i].keys():
        if "BOLD" in results[i].keys():
            # if a dataset was passed as an argument
            bold = results[i]["BOLD"][:, int(bold_transient/1000/2):]
            if "ds" in kwargs:
                ds = kwargs["ds"]

                # calculate mean correlation of functional connectivity
                # of the simulation and the empirical data
                dfResults.loc[i, "fc"] = np.mean(
                    [
                        func.matrix_correlation(
                            func.fc(bold), fc,
                        )
                        for fc in ds.FCs
                    ]
                )         
                # if BOLD simulation is longer than 5 minutes, calculate kolmogorov of FCD
                if len(bold.T) > 5 * 30
                dfResults.loc[i, "fcd"] = np.mean(
                    [
                        func.ts_kolmogorov(bold, bold)
                        for bold in ds.BOLDs
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


def findCloseResults(dfResults, dist=0.01, relative = False, **kwargs):
    """Usage: findCloseResults(search.dfResults, mue_ext_mean=2.0, mui_ext_mean=2.5)
    """
    # dist = 0.01
    selectors = True
    for key, value in kwargs.items():
        if relative:
            new_selector = abs(dfResults[key] - value) <= dist * value
        else:
            new_selector = abs(dfResults[key] - value) <= dist
        selectors = selectors & new_selector
    filtered_df = dfResults[selectors]
    return filtered_df

def paramsRun(dfResults, runNr):
    return dfResults.loc[runNr].to_dict()
