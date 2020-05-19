import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

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
    contour=None,
    alpha_mask=None,
    **kwargs
):
    """
    """
    # PREPARE DATA
    # ------------------    
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

    # PLOT
    # ------------------    

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
        else:
            fig = plt.figure(figsize=(5, 4), dpi=150)
            if plot_key_label:
                plt.title(plot_key_label)
            ax = plt.gca()       


        # -----
        # pivot data and plot
        df_pivot = df.pivot_table(values=plot_key, index=par2, columns=par1, dropna=False)

        plot_clim = (np.nanmin(df_pivot.values), np.nanmax(df_pivot.values))

        if symmetric_colorbar:
            plot_clim = (-np.max(np.abs(plot_clim)), np.max(np.abs(plot_clim)))
        image_extent = [min(df[par1]), max(df[par1]), min(df[par2]), max(df[par2])]
        image = np.array(df_pivot)
        
        # -----
        # alpha mask
        if alpha_mask is not None:
            mask_threshold = kwargs["mask_threshold"] if "mask_threshold" in kwargs else 1
            mask_alpha = kwargs["mask_alpha"] if "mask_alpha" in kwargs else 0.5
            mask_style = kwargs["mask_style"] if "mask_style" in kwargs else None
            mask_invert = kwargs["mask_invert"] if "mask_invert" in kwargs else False

            # alpha_mask can either be a pd.DataFrame or an np.ndarray that is 
            # layed over the image, a string that is a key in the results df
            # or simply True, which means that the image itself will be used as a 
            # threshold for the alpha map (default in alphaMask()). 

            if isinstance(alpha_mask, (pd.DataFrame, np.ndarray)):
                mask = np.array(alpha_mask)
            elif alpha_mask == "custom":
                mask = df.pivot_table(values=alpha_mask, index=par2, columns=par1, dropna=False)
            elif isinstance(alpha_mask, str):
                mask = df.pivot_table(values=alpha_mask, index=par2, columns=par1, dropna=False)
                mask = np.array(mask)
            else:
                mask = None
            
            image = alphaMask(image, mask_threshold, mask_alpha, mask=mask, invert=mask_invert, style=mask_style)

        im = ax.imshow(
            image,
            extent=image_extent,
            origin="lower",
            aspect="auto",
            clim=plot_clim,
        )

        # ANNOTATIONs
        # ------------------
        # plot contours
        if contour is not None:
            contour_color = kwargs["contour_color"] if "contour_color" in kwargs else "white"
            contour_levels = kwargs["contour_levels"] if "contour_levels" in kwargs else None
            # check if this is a dataframe
            if isinstance(contour, pd.DataFrame):
                contourPlotDf(contour, color=contour_color, ax=ax, levels=contour_levels)
            # if it's a string, take that value as the contour plot value
            elif isinstance(contour, str):
                df_contour = df.pivot_table(values=contour, index=par2, columns=par1, dropna=False)
                contourPlotDf(df_contour, color=contour_color, ax=ax, levels=contour_levels)            

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


def contourPlotDf(dataframe, color="white", levels=None, ax=None):
    levels = levels or [0, 1.0001]
    Xi, Yi = np.meshgrid(dataframe.columns, dataframe.index)
    ax = ax or plt
    cset2 = ax.contour(
        Xi, Yi, dataframe, colors=color, linestyles="solid", levels=levels, linewidths=(4,), zorder=1
    )

def alphaMask(image, threshold, alpha, mask=None, invert=False, style=None):
    if mask is None:
        mask = image    
    #alphas = Normalize(0, threshold, clip=True)(np.abs(mask))
    alphas = mask>threshold if not invert else mask<threshold
    alphas = np.clip(alphas, alpha, 1)

    if style == "stripes":
        f = mask.shape[0]/5
        style_mask = np.sin(np.linspace(0, 2*np.pi*f, mask.shape[0]))
        style_mask=style_mask>0
        alphas = alphas + style_mask[:, None]
        alphas = np.clip(alphas, 0, 1)

    cmap = plt.cm.plasma
    colors = Normalize(np.nanmin(image), np.nanmax(image), clip=True)(image)
    colors = cmap(colors)
    colors[..., -1] = alphas
    return colors

def plotResult(search, runId, z_bold = False, **kwargs):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2), dpi=300, gridspec_kw={'width_ratios': [1, 1.2, 2]})

    bold_transient = int(kwargs["bold_transient"] / 2) if "bold_transient" in kwargs else int(30 / 2)

    # get result from search
    result = search.getResult(runId)

    bold = result.BOLD[:, bold_transient:]
    bold_z = stats.zscore(bold, axis=1)
    t_bold = np.linspace(2, len(bold.T)*2, len(bold.T), )

    output = result[search.model.default_output]
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

def processExplorationResults(search, **kwargs):
    """Process results from the exploration. 
    """

    dfResults = search.dfResults


    # cycle through each result's runID
    for i in tqdm.tqdm(dfResults.index):
        # get result
        result = search.getResult(i)
        # ------------------------
        # analyse model outputs
        # to know the name of the output, either a model has to be passed to this function
        # alternatively an output name can be directly specified using the output="output_name" argument
        if "model" in kwargs or "output" in kwargs:
            dt = None
            if "model" in kwargs:
                model = kwargs["model"]
                output_name = model.default_output
                dt = model.params["dt"]
            if "output" in kwargs:
                output_name = kwargs["output"]
                if "dt" in kwargs:
                    dt = kwargs["dt"] 

            assert output_name in result, f"Results do not contain output `{output_name}`."
            assert dt > 0, f"dt could not be determined from model, use dt=0.1 for example."

            # use the last x ms for analysis
            last_ms = kwargs["output_last_ms"] if "output_last_ms" in kwargs else 1000
            output = result[output_name][:, -int(last_ms / dt) :]
            
            dfResults = computeMinMax(dfResults, i, output, output_name)

        # ------------------------
        # analyse BOLD output
        if "BOLD" in result.keys():
            # set bold transient 
            bold_transient = kwargs["bold_transient"] if "bold_transient" in kwargs else 10000
            
            # load BOLD data
            if "BOLD" in result["BOLD"]:
                # if the output is a nested dictionary (default output of a model)\
                bold = result["BOLD"]["BOLD"]
                t_bold = result["BOLD"]["t_BOLD"]
            elif isinstance(result["BOLD"], np.ndarray):
                # if not, then we hope the first BOLD key contains an array
                # and infer the time axis by assuming fs=0.5 Hz BOLD sampling rate
                bold = result["BOLD"]
                t_bold = np.linspace(0, bold.shape[1]*2*1000, bold.shape[1])
            else:
                raise ValueError("Could not load BOLD data. Wrong format?")
            
            bold = result["BOLD"][:, t_bold>bold_transient]
            t_bold = t_bold[t_bold>bold_transient]

            # cut the bold signal until a time but only as the input
            # for computeMinMax(), that's why we create a copy here 
            # and use the original bold signal later for fc and fcd
            if "bold_until" in kwargs:
                bold_minmax = bold[:, t_bold<kwargs["bold_until"]]
                t_bold_minmax = t_bold[t_bold<kwargs["bold_until"]]
            else:
                bold_minmax = bold
                t_bold_minmax = t_bold

            output_name = "BOLD"
            dfResults = computeMinMax(dfResults, i, bold_minmax, output_name)     
            
            # -----
            # compare to BOLD dataset
            # if a dataset was passed as an argument
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
                skip_fcd = kwargs['skip_fcd'] if 'skip_fcd' in kwargs else False
                if t_bold[-1] > 5 * 1000 * 60 and not skip_fcd:
                    sim_fcd = func.fcd(bold)
                    if hasattr(ds, "FCDs"):
                        emp_fcds = ds.FCDs
                    else:
                        emp_fcds = [func.fcd(emp_bold) for emp_bold in ds.BOLDs]
                    dfResults.loc[i, "fcd"] = np.mean(
                        [
                            func.matrix_kolmogorov(sim_fcd, emp_fcd)
                            for emp_fcd in emp_fcds
                        ]
                    )    


def computeMinMax(dfResults, i, output, output_name):
        # calculate the maximum of the output
        dfResults.loc[i, "max_" + output_name] = np.nanmax(
            output
        )
        # calculate the minimum of the output
        dfResults.loc[i, "min_" + output_name] = np.nanmin(
            output
        )            

        # calculate the maximum amplitude of the output
        dfResults.loc[i, "max_amp_" + output_name] = np.nanmax(
            np.nanmax(output, axis=1)
            - np.nanmin(output, axis=1)
        )

        # calculate the minimum amplitude of the output
        dfResults.loc[i, "min_amp_" + output_name] = np.nanmin(
            np.nanmax(output, axis=1)
            - np.nanmin(output, axis=1)
        )

        # compute relative amplitude
        dfResults['relative_amplitude_' + output_name] = dfResults['max_amp_' + output_name] / (dfResults['max_' + output_name] - dfResults['min_' + output_name])
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
