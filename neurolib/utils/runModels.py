from brian2 import *
import os
import time
import warnings

import loadparams as lp
import functions as func
import fitparams as fp
import timeIntegration as ti # aLN mean-field simulation
import network_sim as net # AdEx network simulation
import paths

def runModels(traj=0, manual_params=None):
    """ 
    Wrapper function designed to be called by pypet
    to either run mean-field model or AdEx network model
    without external stimulus. 

    If traj is defined, it assumes that runModels() was called by pypet 
    (for parameter exploration) and loads the parameters from the trajectory
    and returns the results the way pypet expects it. 

    If traj==0, parameters are loaded from global variable "params", make
    sure to set them before calling this function.
    """
    np.random.seed() 
    if (traj != 0):
        params = traj.parameters.f_to_dict(short_names=True, fast_access=True) 
    elif (manual_params != None):
        params = manual_params
    else:
        warnings.warn("params can't be None!")
        return

    model = params['model']
    point = params['load_point']
    if model == 'aln':
        if point != 'none':
            params = fp.loadpoint(params, point, reload_params=True, newIC=False)        
        t, rates_exc, rates_inh, stimulus = runaLN(params)
        
    elif model == 'brian':      
        if point != 'none':
            params = fp.loadpoint_network(params, point, reload_params=True)
        t, rates_exc, rates_inh, stimulus = runAdEx(params)
        
    if (traj != 0):
        traj.f_add_result('results.$', t = t, rate_exc = rates_exc, rate_inh = rates_inh, stimulus = stimulus)      
    else:
        return t, rates_exc, rates_inh, stimulus


def runAdEx(params):
    '''
    Runs an AdEx network simulation
    '''

    stimulus = params['ext_exc_current']
    if not hasattr(stimulus, "__len__"): # in case stimulus is only a number
        stimulus = [stimulus] # convert it to a list (the way that set_network_params expects it)

    net_params, ext_input = lp.set_network_params(params, stimulus)

    # prepare brian2 simulation 
    set_device('cpp_standalone', build_on_run=False)
    device.insert_code(
        'main', 'srand(' + str(int(time.time()) + os.getpid()) + ');')
    results = lp.init_resultsdict()

    # destination for build files of brian2 cpp standalone mode
    try:
        compile_dir = paths.BRIAN2_COMPILE_DIR
    except:
        compile_dir = "models/brian2/brian2_compile/"
    if not os.path.isdir(compile_dir): os.makedirs(compile_dir)

    # run
    results['model_results']['net'] = net.network_sim(
        ext_input, net_params, rec=True, standalone_dir = compile_dir)
    
    rates_exc_net = results['model_results']['net']['r_e']
    rates_inh_net = results['model_results']['net']['r_i'][0]

    t_net = results['model_results']['net']['t']
    
    device.reinit()
    device.activate()    

    return t_net, rates_exc_net, rates_inh_net, stimulus



def runaLN(params):
    '''
    Runs an aLN mean-field model simulation
    '''  
    rates_exc, rates_inh, t, \
            mufe, mufi, IA, seem, seim, siem, siim, \
                seev, seiv, siev, siiv, return_tuple, rhs = ti.timeIntegration(params)

    rates_exc = rates_exc[0, :]*1000.0
    rates_inh = rates_inh[0, :]*1000.0
    stimulus = params['ext_exc_current']

    t = np.dot(range(len(rates_exc)),params['dt'])    
    
    return t, rates_exc, rates_inh, stimulus





def runModels_stimulus(traj=0, manual_params=None):
    """ 
    Wrapper function designed to be called by pypet
    to either run mean-field model or AdEx network model
    with predefined external stimulus. 

    Note: if you want to run either model without a stimulus, call 
    runModels() or run each model directly with runaLN() or runAdex() 
    after loading a certain parameter set. 
    params = fp.loadpoint(params, point) # for mean-field
    params = fp.loadpoint_network(params, point) # for AdEx

    If traj is defined, it assumes that runModels() was called by pypet 
    (for parameter exploration) and loads the parameters from the trajectory
    and returns the results the way pypet expects it. 

    If traj==0, parameters are loaded from global variable "params", make
    sure to set them before calling this function.
    """
    
    np.random.seed() 
    if (traj != 0):
        params = traj.parameters.f_to_dict(short_names=True, fast_access=True) 
    elif (manual_params != None):
        params = manual_params
    else:
        print("params can't be None!")
        return

    model = params['model']
    point = params['load_point']
    
    mue = params['mue_ext_mean'] 
    mui = params['mui_ext_mean'] 
    
    duration = params['duration'] 
    
    params['ext_exc_current'] = 0 # delete external stimulus in case it was somehow before
    
    if model == 'aln':
        # setting params['load_point'] overrides previous mue and mui
        if point != 'none':
            params = fp.loadpoint(params, point, reload_params=True, newIC=False)
        else:
            params['mue_ext_mean'] = mue
            params['mui_ext_mean'] = mui
        t, rates_exc, rates_inh, stimulus = runaLN_stimulus(params)
        
    elif model == 'brian':
        if point != 'none':
            params = fp.loadpoint_network(params, point, reload_params=True)
        else:
            params['mue_ext_mean'] = mue
            params['mui_ext_mean'] = mui        
        t, rates_exc, rates_inh, stimulus = runAdEx_stimulus(params)
        
    if (traj != 0):
        traj.f_add_result('results.$', t = t, rate_exc = rates_exc, rate_inh = rates_inh, stimulus = stimulus)      
    else:
        return t, rates_exc, rates_inh, stimulus

def runaLN_stimulus(params):
    stimulus = []
    if params['A_sin'] > 0 and params['f_sin'] > 0:
        delay = 0
        stimulus = construct_stimulus('ac', duration=params['duration'], dt=params['dt'], stim_amp=params['A_sin'], stim_freq=params['f_sin'], nostim_before=1000+delay, n_periods=1000)
        params['ext_exc_current'] = stimulus
    else:
        stimulus = construct_stimulus('rect', duration=params['duration'], dt=params['dt'], stim_amp=1.0)
        params['ext_exc_current'] = stimulus
        
    # run    
    rates_exc, rates_inh, t, \
            mufe, mufi, IA, seem, seim, siem, siim, \
                seev, seiv, siev, siiv, return_tuple, rhs = ti.timeIntegration(params)

    rates_exc = rates_exc[0, :]*1000.0
    rates_inh = rates_inh[0, :]*1000.0
    
    t = np.dot(range(len(rates_exc)),params['dt'])    
    
    return t, rates_exc, rates_inh, stimulus

def runAdEx_stimulus(params):
    
    stimulus = []
    if params['A_sin'] > 0 and params['f_sin'] > 0:
        delay = 0
        stimulus = construct_stimulus('ac', duration=params['duration'], dt=params['dt'], stim_amp=params['A_sin'], stim_freq=params['f_sin'], nostim_before=1000+delay, n_periods=1000)
        params['ext_exc_current'] = stimulus
    else:
        stimulus = construct_stimulus('rect', duration=params['duration'], dt=params['dt'], stim_amp=1.0)
        params['ext_exc_current'] = stimulus      

    net_params, ext_input = lp.set_network_params(params, stimulus)

    # prepare brian2 simulation 
    set_device('cpp_standalone', build_on_run=False)
    device.insert_code(
        'main', 'srand(' + str(int(time.time()) + os.getpid()) + ');')
    results = lp.init_resultsdict()

    # destination for build files of brian2 cpp standalone mode
    try:
        compile_dir = paths.BRIAN2_COMPILE_DIR
    except:
        compile_dir = "models/brian2/brian2_compile/"
    if not os.path.isdir(compile_dir): os.makedirs(compile_dir)

    # run
    results['model_results']['net'] = net.network_sim(
        ext_input, net_params, rec=True, standalone_dir = compile_dir)
    
    rates_exc_net = results['model_results']['net']['r_e']
    rates_inh_net = results['model_results']['net']['r_i'][0]

    t_net = results['model_results']['net']['t']
    
    device.reinit()
    device.activate()    

    return t_net, rates_exc_net, rates_inh_net, stimulus


def construct_stimulus(stim='dc', duration=6000, dt=0.1, stim_amp=0.2, stim_freq=1, stim_bias=0, n_periods=0, nostim_before=0, nostim_after=0):
    '''
    Constructs an appropriate sitmulus for the experiments

    stim:       stimulation stype ['ac':oscillatory stimulus, 'dc': stimple step current, 
                'rect': step current in negative then positive direction with slowly
                decaying amplitude, used for bistability detection]
    stim_amp:   amplitude of stimulus in mV/ms (multiply by C to get pA value)
    '''
    def sinus_stim(f=1, amplitude=0.2, positive=0, phase=0, cycles=1, t_pause=0):
        x = np.linspace(np.pi, -np.pi, 1000 / dt / f)
        sinus_function = np.hstack(((np.sin(x + phase) + positive), np.tile(0, t_pause)))
        sinus_function *= amplitude
        return np.tile(sinus_function, cycles)

    if stim == 'ac':
        if n_periods == 0:
            n_periods = int(stim_freq) * 1
        stimulus = np.hstack(([stim_bias] * int(nostim_before / dt),
                              np.tile(sinus_stim(stim_freq, stim_amp) + stim_bias, n_periods)))
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
    elif stim == 'dc':
        stimulus = np.hstack(
            ([stim_bias] * int(nostim_before / dt), [stim_bias + stim_amp] * int(1000 / dt)))
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
        stimulus[stimulus < 0] = 0
    elif stim == 'rect':
        # construct input
        stimulus = np.zeros(int(duration / dt))
        tot_len = int(duration / dt)
        stim_epoch = tot_len / 6

        stim_increase_counter = 0
        stim_decrease_counter = 0
        stim_step_increase = 5.0 / stim_epoch

        for i, m in enumerate(stimulus):
            if 0 * stim_epoch <= i < 0.5 * stim_epoch:
                stimulus[i] -= stim_amp
            elif 0.5 * stim_epoch <= i < 3.0 * stim_epoch:
                stimulus[i] = -np.exp(-stim_increase_counter) * stim_amp
                stim_increase_counter += stim_step_increase
            elif 3.0 * stim_epoch <= i < 3.5 * stim_epoch:
                stimulus[i] += stim_amp
            elif 3.5 * stim_epoch <= i < 5 * stim_epoch:
                stimulus[i] = np.exp(-stim_decrease_counter) * stim_amp
                stim_decrease_counter += stim_step_increase
    else:
        print("ERROR, stim protocol {} not found")

    # repeat stimulus until full length
    steps = int(duration / dt)
    stimlength = int(len(stimulus))
    stimulus = np.tile(stimulus, steps // stimlength + 2)
    stimulus = stimulus[:steps]

    return stimulus
