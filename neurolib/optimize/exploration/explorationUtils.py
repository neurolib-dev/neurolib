import numpy as np
import pandas as pd

def pivot_plots(dfResults, by, par1, par2, plot_key, plot_key_label="", symmetric_colorbar=False, one_figure=False):
    n_plots = len(dfResults.groupby(by=by))
    if one_figure == True:
        fig, axs = plt.subplots(nrows=1, ncols=n_plots, figsize=(n_plots*4, 4), dpi=150)
    axi = 0
    for i,k in dfResults.groupby(by=by):
        
        if one_figure == True:
            ax = axs[axi]
            
        df = k

        df_pivot = df.pivot_table(values=plot_key, index = par2, columns=par1)
        
        plot_clim = None
        if symmetric_colorbar:
            plot_clim = (-np.max(df_pivot.values), np.max(df_pivot.values))
        
        if one_figure == False:
            fig = plt.figure(figsize=(5, 4), dpi=150)
            ax = plt.gca()
            
        im = ax.imshow(df_pivot, \
               extent = [min(df[par1]), max(df[par1]),
                         min(df[par2]), max(df[par2])], origin='lower', aspect='equal', clim=plot_clim)
        
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        if one_figure == False:
            plt.colorbar(im, ax=ax, orientation='vertical');        
        else:
            # if this is the last plot
            if axi == n_plots - 1:
                #plt.colorbar(im, ax=ax, orientation='vertical');   
                cbar_ax = fig.add_axes([0.92, 0.18, 0.01, 0.65])
                fig.colorbar(im, cax=cbar_ax)
        
        ax.set_xlabel(par1)
        ax.set_ylabel(par2)

        # single by-values need to become tuple    
        if not isinstance(i, tuple):
            i = (i,)

        ax.set_title(" ".join([f"{bb}={bi}" for bb, bi in zip(by, i)]))
        if one_figure == False:
            plt.show()
        else:
            axi += 1
            
    if one_figure == True:
        plt.show()
        plt.tight_layout()