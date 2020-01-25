# -*- coding: utf-8 -*-
'''
 script for computing the quantities required by the cascade-based spike rate 
 models (LNexp, LNdos), i.e., steady-state spike rate and mean membrane voltage 
 of an exponential/leaky integrate-and-fire neuron subject to white noise input 
 as well as quantities derived from the first order rate response to modulations 
 of the input moments, for a range of values for the baseline input moments 
 (mu, sigma) -- written by Josef Ladenbauer in 2016 
'''

from params import get_params
import methods_cascade as mc
import numpy as np
import multiprocessing
from collections import OrderedDict
import os

# WHAT TO DO ------------------------------------------------------------------
load_EIF_output = False  # True if EIF output has been computed and saved before
load_quantities = False  # True if quantities have been computed and saved before
compute_EIF_output = True
compute_quantities = True  # True -> needs compute_EIF_output set to True
save_rate_mod = False  # True to save linear rate response functions, 
                       # False is default to save memory
save_EIF_output = True
save_quantities = True
plot_filters = False  # True -> needs EIF_output
plot_quantities = False

# location for saving/loading
folder = os.path.dirname(os.path.realpath(__file__))  # directory for the files
# currently the same directory as for the script itself
output_filename = 'EIF_output_for_cascade.h5'
quantities_filename = 'quantities_cascade.h5'


# PREPARE ---------------------------------------------------------------------
params = get_params()  # loads default parameter dictionary

#params['t_ref'] = 0.0  # refractory period can be >0 (but not all reduced models 
                       #                              in the paper support this)

# choose a plausible range of values for mu and sigma:
# (which range is plausible depends on the neuron model parameter values)
# e.g., for mu from -1 to 5 with spacing 0.025 mV/ms, 
# for sigma from 0.5 to 5 with spacing 0.1 mV/sqrt(ms)  
N_mu_vals = 350 #241  
N_sigma_vals = 64
mu_vals = np.linspace(-1.0, 5.0, N_mu_vals) 
sigma_vals = np.linspace(0.5, 5.0, N_sigma_vals) 
# these values above were used to generate the files EIF_output_for_cascade.h5
# and quantities_cascade.h5 available on Github

# note that sigma should not be too small, and for small values of sigma the 
# lower limit of mu should be sufficiently large (to avoid large regions of 
# vanishing spike rate activity, for which the numerical schemes are not optimized) 

d_V = 0.01  # mV, membrane voltage spacing for calculations of steady-state and 
             # first order rate response (note: Vr should be a grid point)
             # 0.01 mV seems sufficient for many neuron model parametrizations,
             # otherwise reduce to e.g. 0.005
d_freq = 0.25  # Hz, frequency spacing for calculation of first order rate response 
f_max = 1000.0  # Hz

# choose background mu and sigma values for filter visualization;  
mus_plot = [-0.5, 1.0, 2.5] 
sigmas_plot = [1.5, 2.5]
# choose background sigma values for quantity visualization        
sigmas_quant_plot = np.arange(0.5, 4.501, 0.2) 
#sigmas_quant_plot = sigma_vals  # all sigma values used

# some more pre-calculation parameters
params['N_procs'] = int(multiprocessing.cpu_count())  # nb. of parallel procs. 
# note that multiprocessing is only used for >1 sigma value (in sigma_vals)

params['V_vals'] = np.arange(params['Vlb'],params['Vcut']+d_V/2,d_V)
params['freq_vals'] = np.arange(d_freq, f_max+d_freq/2, d_freq)/1000  # kHz
params['d_mu'] = 1e-5 # mV/ms
params['d_sigma'] = 1e-5 # mV/sqrt(ms)

EIF_output_dict = OrderedDict()
LN_quantities_dict = OrderedDict()

EIF_output_names = ['r_ss', 'dr_ss_dmu', 'dr_ss_dsigma', 'V_mean_ss',
                    'r1_mumod', 'r1_sigmamod', 
                    'peak_real_r1_mumod', 'f_peak_real_r1_mumod', # last 4 only
                    'peak_imag_r1_mumod', 'f_peak_imag_r1_mumod'] # for LN_dos

LN_quantity_names = ['r_ss', 'V_mean_ss', 
                     'tau_mu_exp', 'tau_sigma_exp',
                     'tau_mu_dosc', 'f0_mu_dosc'] # last 2 only needed for LN_dos

plot_EIF_output_names = ['r1_mumod', 'ifft_r1_mumod', 
                         'r1_sigmamod', 'ifft_r1_sigmamod']
plot_quantitiy_names =  ['r_ss', 'V_mean_ss',
                         'tau_mu_exp', 'tau_sigma_exp',
                         'tau_mu_dosc', 'f0_mu_dosc'] # last 2 only for LN_dos

if __name__ == '__main__':
    print("Main")
    # LOAD --------------------------------------------------------------------
    if load_EIF_output:
        print("Loading EIF output...")
        mc.load(folder+'/'+output_filename, EIF_output_dict,
                EIF_output_names + ['mu_vals', 'sigma_vals', 'freq_vals'], params)
        # optional shortcuts:
        mu_vals = EIF_output_dict['mu_vals']
        sigma_vals = EIF_output_dict['sigma_vals']
        freq_vals = EIF_output_dict['freq_vals']       
     
    if load_quantities:
        print("Loading quantities...")
        mc.load(folder+'/'+quantities_filename, LN_quantities_dict,
                LN_quantity_names + ['mu_vals', 'sigma_vals', 'freq_vals'], params)
        # optional shortcuts:
        mu_vals = LN_quantities_dict['mu_vals']
        sigma_vals = LN_quantities_dict['sigma_vals']
        freq_vals = LN_quantities_dict['freq_vals'] 
    
    
    # COMPUTE -----------------------------------------------------------------
    if compute_EIF_output and compute_quantities:
        print('')
        print('Computing {}'.format(EIF_output_names))
        print('This may take a while for large numbers of mu & sigma values...')
        EIF_output_dict, LN_quantities_dict = \
            mc.calc_EIF_output_and_cascade_quants(mu_vals, sigma_vals, params, 
                                                  EIF_output_dict, EIF_output_names, 
                                                  save_rate_mod, LN_quantities_dict, 
                                                  LN_quantity_names)                                          
                                                 
    # SAVE --------------------------------------------------------------------                             
    if save_EIF_output:
        mc.save(folder+'/'+output_filename, EIF_output_dict, params) 
        print('EIF/LIF output saved')
        
    if save_quantities:
        mc.save(folder+'/'+quantities_filename, LN_quantities_dict, params) 
        print('LN quantities saved')
         
    # PLOT --------------------------------------------------------------------
    if plot_filters:
        recalc_filters = True
        mc.plot_filters(EIF_output_dict, LN_quantities_dict, plot_EIF_output_names, 
                        params, mus_plot, sigmas_plot, recalc_filters)
        
    if plot_quantities: 
        mc.plot_quantities(LN_quantities_dict, plot_quantitiy_names, sigmas_quant_plot)
        #mc.plot_quantities_forpaper(LN_quantities_dict, plot_quantitiy_names, 
        #                            sigmas_quant_plot, mus_plot, sigmas_plot)
    

