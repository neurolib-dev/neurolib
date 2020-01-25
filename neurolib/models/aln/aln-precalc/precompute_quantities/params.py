'''manages the parameter sets for all models. all parameter are given without brian units except the ones that are used
are used for brian2 simulations.'''

def get_params():
    params = dict()

    # runtime options #
    # reduced models
    params['rectify_spec_models'] = True
    params['uni_dt'] = 0.05  #[ms]
    params['uni_int_order'] = 1 # (1:Euler, 2:Heun)
    params['grid_warn'] = True #outside grid warn

    # params for ou process #
    params['ou_stationary'] = True # if False --> ou process starts at X0

    # parameters for network simulation
    params['neuron_model'] = 'EIF' #'EIF' #current options EIF/PIF
    params['net_record_spikes'] = 25 # 200 #number of neurons to record spikes from
    params['net_record_example_v_traces'] = 0. # 100 #number of v_traces that are recorded, if no trace should be computed: 0
    params['net_record_all_neurons'] = False # keep this high; otherwise great deal of memory, zero if not
    params['net_record_all_neurons_dt'] = 10. # keep this high; otherwise great deal of memory, zero if not
    params['net_record_v_stats'] = True # mean and std
    params['net_record_w_stats'] = True # mean and std
    params['net_record_w'] = 10# 0000 # record 100 w traces
    params['net_record_dt'] = 1. #[ms]
    params['net_w_refr'] = True  #clamp (True) or don't clamp (False) w during the refractory period
    params['net_w_init'] = 0.#20. # [pA]
    params['net_v_lower_bound'] = None #-100. #if None --> NO lower bound , else value of lower bound e.g. -200 in [mV]
    params['connectivity_type'] = 'fixed' #'binomial'#'fixed'
    #initial conditions
    #the transient response of the network and the solution of the fp eq are in very good agreement for the init: normal
    #note for uniform distribution: by default the uniform initial distribution is set on the interval [Vr, Vcut]
    params['net_v_init'] = 'normal' # 'uniform' #'normal', 'delta'
    params['net_delta_peak'] = -70.
    params['net_normal_mean'] = -100.   #mV
    params['net_normal_sigma'] = 10.    #mV
    # standalone mode for network sim
    params['brian2_standalone'] = True
    params['brian2_device'] = 'cpp_standalone'
    # integration method for (should be specified for brian2_rc3)
    params['net_integration_method'] = 'heun' #'heun'



    #neuron model parameters (AdEX)
    params['C'] = 200.     #[pF]
    params['gL'] = 10.     #[nS]
    params['taum'] = params['C'] / params['gL'] #[ms]
    params['EL'] = -65.    #[mV] # reversal potential for membrane potential v
    params['Ew'] = -80.    #[mV] # reversal potential for adaptation param w
    params['VT'] = -50.    #[mV]
    params['deltaT'] = 1.5 #[mV]
    params['Vcut'] = -40.  #[mV]
    params['tauw'] = 200.  #[ms]
    params['a'] =  4.  #4.  #[nS]              subhreshold adaptation param
    params['b'] =  40.  #40 #[pA]              spike-frequency adaptation param
    params['Vr'] = -70.    #[mV]
    params['t_ref'] = 1.5  #[ms]
    # for all derived models and the scharfetter gummel method
    params['Vlb'] = -200. #[mV]

    #for recurrency
    params['J'] = .1          #[mV]
    params['K'] = 100         #500      #number of connections



    #FOKKER-PLANCK IMPLEMENTATION (SCHARFETTER-GUMMEL)
    params['N_centers_fp'] = 1000     # number of centers for V-space discretization
    params['fp_dt'] = 0.05  # [ms]
    params['integration_method'] = 'implicit' 

    #initial conditions for the fp model. by default they match initial conditions for the net
    params['fp_v_init'] = params['net_v_init'] #'delta'#'uniform','normal', 'delta'
    params['fp_delta_peak'] =   params['net_delta_peak'] #[mV]
    params['fp_normal_mean'] =  params['net_normal_mean']
    params['fp_normal_sigma'] = params['net_normal_sigma']
    # rather use these for the delay
    # params['tau_d'] =
    # params['d_0'] =
    #FOR UNIFORM DISTRIBUTION
    params['start_uniform'] = .05
    params['end_uniform'] = .75
    # type of finite-size noise and type of normalization
    params['noise_type'] = 'gauss'
    params['norm_type']  = 'full_domain'

    

    # SPECTRALSOLVER PARAMETERS
    # => model params
    print('TODO: rename parameter to match dict from model comparison')
    params['model'] = params['neuron_model']
    params['V_r'] = params['Vr']
    params['g_L'] = params['gL']
    params['E_L'] = params['EL']
    params['V_T'] = params['VT']
    params['delta_T'] = params['deltaT']
    params['V_s'] = params['Vcut']
    params['tau_ref'] = params['t_ref']
    params['V_lb'] = params['Vlb']
    # rename dmu_couplingterms as well...

    # => mu/sigma step limit from below to prevent hopping to another eigenvalue curve
    params['diff_mu_sigma_min'] = 5e-3 # minimal mu grid distance that is enforced (even if the 
                                       # actually provided grid is coarser) during numerical 
                                       # computation of the spectrum
                                       # note: if two eigenvalues lie close together or 
                                       # lambda(mu_sigma_curve) is very steep then this param
                                       # must be small otherwise it possibly convergence to wrong eigval!
    params['lambda_real_abs_thresh'] = 3.0 # ignore all eigenvalue abs realparts below this value
    # => ODE solver
    params['grid_V_points'] = 20000 # discretization for exponential backwards integration
    # => algebraic solver
    params['solver_abstol'] = 1e-4 # absolute tolerance (scipy solver wrapper)
    params['solver_maxrefinements'] = 30 #COMMENT THIS -- NOT INCLUDED IN PAPER # mu, sigma refinements (interval bisection) if non-convergence
    params['solver_init_approx_type'] = 'const' #COMMENT THIS -- NOT INCLUDED IN PAPER  #'linear' # 'linear' or (default) NOne = lastpoint
    params['solver_smoothtol'] = 0.1 #COMMENT THIS -- NOT INCLUDED IN PAPER  # limit the relative difference of eigenvalues for successive values of mu,sigma  
    params['solver_workaround_tolreduction_refinements'] = 5 # in case of workaround: every these many refinements to increase abs and rel tol
    params['solver_workaround_max'] = 10 # in mu/sigma curve steps
    # => root solver for 2d unknown (real/imag part of eigenvalue)
    params['root_method'] = 'hybr'
    params['root_options'] = {'eps':  1e-10, 
                              'xtol': 1e-8}
#                               'diag': [1., 1.]} # [1., 10.] 
    # => quantities
#    params['quantities_grid_V_addpoints'] = 5000 # not required for the new threshold flux method
#    params['threshold_interp_order'] = 3 # not required anymore for the new threshold flux method
    params['dmu_couplingterms'] = 1e-3 # 1e-2 or 1e-4 produce identical results up to very small numerical differences
    params['dsigma_couplingterms'] = 1e-3 # 1e-2 or 1e-4 produce identical results up to very small numerical differences
    # for postprocessing spectrum and quantities    
    params['tolerance_conjugation'] = 1e-4 #COMMENT THIS -- NOT INCLUDED IN PAPER  also: imag parts significance (to be different from zero)
    # => misc
    params['verboselevel'] = 0          # 1: some debug text, 2: +intermediateplots, 3: verbose text



    #record variables for reduced models
    #lnexp model
    params['rec_lne'] = ['Vm', 'wm'] #'tau_mu_f', 'tau_sigma_f', 'w_m',
    #lndosc model
    params['rec_lnd'] = ['wm']
    #spec1 model
    params['rec_s1'] = ['wm', 'lambda_1_real']
    #spec m model
    params['filter_type'] = 'gauss'
    params['filter_gauss_sigma'] = 400. # [ms]

    params['rec_sm'] = ['wm']

    # initial value for the mean adaptation current
    params['wm_init'] = 0. #[pA]


    #for reccurrency
    params['taud'] = 3.   #[ms]
    params['const_delay'] = 5. #[ms]
    params['delay_type'] = 1 #options [0: no delay, 1: const, 2: exp, 3: const+exp]


    #for plotting
    #colors[modelname] = color
    params['color'] = {'net':'b', 'fp':'0.6','ln_exp':'darkmagenta', 'ln_dos':'cyan',
                       'ln_bexdos':'green', 'spec1':'darkgreen', 
                       'spec2_red':'pink', 'spec2':'orangered'}
    params['lw'] = {'net':'1', 'fp':'2','ln_exp':'1', 'ln_dos':'2', 'ln_bexdos':'2',
                    'spec1':'1', 'spec2_red':'1', 'spec2': '1'}
    return params