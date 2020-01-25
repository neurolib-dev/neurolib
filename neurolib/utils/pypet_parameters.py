from pypet import Environment, cartesian_product, Parameter
import h5py


def add_parameters(traj, params):
    """ Adds parameters to existing Trajectory 'traj' using dict params
    The added parameters are the one from the interareal network with the aLN cascade model for the local dynamic.
    
    WARNING: pypet 3.0 doesn't support lazy loading anymore, this function will not work!
    """
    
    
    ### simulation parameters
    
    traj.f_add_parameter_group('simulation', comment = 'Group containing simulation parameters')

    traj.simulation.model           = Parameter('model', params['model'], comment='Which model to run (aln/brian)')
                                            
    traj.simulation.dt              = Parameter('dt', params['dt'], comment='I am a useful comment!') # params['dt'], 'integration time step'
    traj.simulation.duration        = Parameter('duration', params['duration'], comment='total simulation time [ms]')
    traj.simulation.warn            = Parameter('warn', params['warn'], comment='warns if out of precalc limits in \
                                        interpolation, 0 for faster computation')
    traj.simulation.dosc_version    = Parameter('dosc_version', params['dosc_version'], comment='use dosc_version (1/0)')
    traj.simulation.fast_interp     = Parameter('fast_interp', params['fast_interp'], comment='Interpolate the value from the look-up table \
                                                                instead of taking the closest value')
    traj.simulation.distr_delay     = Parameter('distr_delay', params['distr_delay'], comment='use distributed delay instead of fixed (1/0)') 
    traj.simulation.filter_sigma    = Parameter('filter_sigma', params['filter_sigma'], comment='full filter used for sigmae/sigmai(1),\
                                                                else use dirac filter(0)')
    traj.simulation.seed            = Parameter('seed', params['seed'], comment='Random Seed for Noise and ICs')


    # global coupling parameters

    traj.f_add_parameter_group('globalNetwork', comment = 'Group containing \
                                           global network coupling parameters')
                                           
    traj.globalNetwork.N      = Parameter('N', params['N'], 'Number of nodes in SC (or for brian: number of neurons in one pop)')                                       
    traj.globalNetwork.density      = Parameter('density', params['density'],\
                                        comment='density of SC: cutoff weakest links, from 1 to 0')
    traj.globalNetwork.global_delay = Parameter('global_delay', params['global_delay'],\
                                        comment='use global interareal delays, 1 or 0')
    #traj.globalNetwork.CmatFileName = Parameter('CmatFileName', params['CmatFileName'],\
                                        #comment='File name for the connectivity matrix')
    #traj.globalNetwork.DmatFileName = Parameter('DmatFileName', params['DmatFileName'],\
                                        #comment='File name for the delay matrix')                                

    traj.globalNetwork.Cmat = Parameter('Cmat', params['Cmat'], comment='relative coupling strengths, \
                Cmat[i,j]: connection from jth to ith (values between 0 and 1)')
    traj.globalNetwork.lengthMat = Parameter('lengthMat', params['lengthMat'], comment='fiber length matrix')
    traj.globalNetwork.signalV = Parameter('signalV', params['signalV'], comment='[m/s] Signal velocity for the interareal connections')
   

    traj.globalNetwork.c_gl = Parameter('c_gl', params['c_gl'], comment='global coupling strength \
                                                between areas(unitless)')                                            
    traj.globalNetwork.Ke_gl = Parameter('Ke_gl', params['Ke_gl'], comment='number of incoming E connect-\
                                                  ions from each area')
                                                  
    
    # local network (area) parameters
    
    traj.f_add_parameter_group('localNetwork', comment = 'Group containing \
                                            local network (area) parameters')   

    traj.localNetwork.load_point   = Parameter('load_point', params['load_point'], comment='Specifies which parameter point to load (from Cakan 2019)')
    
    # external input parameters:
    traj.localNetwork.tau_ou       = Parameter('tau_ou', params['tau_ou'], comment='[ms]')
    traj.localNetwork.sigma_ou     = Parameter('sigma_ou', params['sigma_ou'], comment='[mV/ms/sqrt(ms)]')
    traj.localNetwork.mue_ext_mean = Parameter('mue_ext_mean', params['mue_ext_mean'], comment='[mV/ms] (OU process)')
    traj.localNetwork.mui_ext_mean = Parameter('mui_ext_mean', params['mui_ext_mean'], comment='[mV/ms] (OU process)')
    traj.localNetwork.sigmae_ext   = Parameter('sigmae_ext', params['sigmae_ext'], comment='[mV/sqrt(ms)]')
    traj.localNetwork.sigmai_ext   = Parameter('sigmai_ext', params['sigmai_ext'], comment='[mV/sqrt(ms)]')
    traj.localNetwork.ext_exc_rate = Parameter('ext_exc_rate', params['ext_exc_rate'], comment='external excitatory rate input [kHz]')
    traj.localNetwork.ext_inh_rate = Parameter('ext_inh_rate', params['ext_inh_rate'], comment='external inhibitory rate input [kHz]')
    traj.localNetwork.ext_inh_current = Parameter('ext_inh_current', params['ext_inh_current'], comment='external inhibitory field input [mV/ms?]')    
    traj.localNetwork.ext_exc_current = Parameter('ext_exc_current', params['ext_exc_current'], comment='external excitatory field input [mV/ms?]')
    
    # sinusodial input paramters
    traj.localNetwork.A_sin        = Parameter('A_sin', params['A_sin'])
    traj.localNetwork.f_sin        = Parameter('f_sin', params['f_sin'])
    traj.localNetwork.ph_sin       = Parameter('ph_sin', params['ph_sin'])
    
        
    # recurrent coupling parameters
    traj.localNetwork.Ke           = Parameter('Ke', params['Ke'], comment='"EE = IE" assumed')
    traj.localNetwork.Ki           = Parameter('Ki', params['Ki'], comment='"EI = II" assumed')

    traj.localNetwork.de           = Parameter('de', params['de'], comment='[ms] local constant delay, \
                                                                    "EE = IE"')
    traj.localNetwork.di           = Parameter('di', params['di'], comment='[ms] local constant delay, \
                                                                    "EI = II"')
    
    traj.localNetwork.tau_se       = Parameter('tau_se', params['tau_se'], comment='[ms], "EE = IE"')
    traj.localNetwork.tau_si       = Parameter('tau_si', params['tau_si'], comment='[ms], "EI = II"')
    traj.localNetwork.cee           = Parameter('cee', params['cee'], comment='(unitless)')
    traj.localNetwork.cei           = Parameter('cei', params['cei'], comment='(unitless)')
    traj.localNetwork.cii           = Parameter('cii', params['cii'], comment='(unitless)')
    traj.localNetwork.cie           = Parameter('cie', params['cie'], comment='(unitless)')

    traj.localNetwork.Jee_max      = Parameter('Jee_max', params['Jee_max'], comment='[mV/ms]')
    traj.localNetwork.Jie_max      = Parameter('Jie_max', params['Jie_max'], comment='[mV/ms]')  
    traj.localNetwork.Jei_max      = Parameter('Jei_max', params['Jei_max'], comment='[mV/ms]') 
    traj.localNetwork.Jii_max      = Parameter('Jii_max', params['Jii_max'], comment='[mV/ms]')

    traj.localNetwork.tau_di        = Parameter('tau_di', params['tau_di'], comment='[ms] Distributed delay time constant, local inhibitory connection')
    traj.localNetwork.tau_de        = Parameter('tau_de', params['tau_de'], comment='[ms] Distributed delay time constant, local excitatory connection')


    # neuron model parameters

    traj.f_add_parameter_group('neuron', comment = 'Group containing \
                                                    neuron model parameters')

    traj.neuron.a            = Parameter('a', params['a'], comment='nS')
    traj.neuron.b            = Parameter('b', params['b'], comment='pA')
    traj.neuron.EA           = Parameter('EA', params['EA'], comment='mV')
    traj.neuron.tauA         = Parameter('tauA', params['tauA'], comment='ms')

    # if params below are changed, preprocessing required
    traj.neuron.C            = Parameter('C', params['C'], comment='pF')
    traj.neuron.gL           = Parameter('gL', params['gL'], comment='nS')
    traj.neuron.EL           = Parameter('EL', params['EL'], comment='mV')
    traj.neuron.DeltaT       = Parameter('DeltaT', params['DeltaT'], comment='mV')
    traj.neuron.VT           = Parameter('VT', params['VT'], comment='mV')
    traj.neuron.Vr           = Parameter('Vr', params['Vr'], comment='mV')
    traj.neuron.Vs           = Parameter('Vs', params['Vs'], comment='mV')
    traj.neuron.Tref         = Parameter('Tref', params['Tref'], comment='ms')
    
    
    ## initial parameters IC, randomly generated in 'loadDefaultParams'
    
    traj.f_add_parameter_group('IC', comment = 'Group containing initial params')
    
    traj.IC.mufe_init   = Parameter('mufe_init', params['mufe_init'], comment='[mV/ms], dosc_version: complex variable')
    traj.IC.IA_init     = Parameter('IA_init', params['IA_init'], comment='[pA]')
    traj.IC.mufi_init   = Parameter('mufi_init', params['mufi_init'], comment='[mV/ms], dosc_version: complex variable')
    traj.IC.seem_init    = Parameter('seem_init', params['seem_init'], comment='COMMENT')
    traj.IC.seim_init    = Parameter('seim_init', params['seim_init'], comment='COMMENT')
    traj.IC.seev_init    = Parameter('seev_init', params['seev_init'], comment='COMMENT')
    traj.IC.seiv_init    = Parameter('seiv_init', params['seiv_init'], comment='COMMENT')
    traj.IC.siim_init    = Parameter('siim_init', params['siim_init'], comment='COMMENT')
    traj.IC.siem_init    = Parameter('siem_init', params['siem_init'], comment='COMMENT')
    traj.IC.siiv_init    = Parameter('siiv_init', params['siiv_init'], comment='COMMENT')
    traj.IC.siev_init    = Parameter('siev_init', params['siev_init'], comment='COMMENT')
    traj.IC.rates_exc_init = Parameter('rates_exc_init', params['rates_exc_init'], comment='COMMENT')
    traj.IC.rates_inh_init = Parameter('rates_inh_init', params['rates_inh_init'], comment='COMMENT')

    # Look up tables used to integrate the local network dynamics
    traj.f_add_parameter_group('LT', comment = 'Look-up tables containing the parameter of the aLN model')
    #traj.LT.Z_dosc      = Parameter('Z_dosc', params['Z_dosc'], comment='Look up table - dosc version')
    #traj.LT.Ze_dosc     = Parameter('Ze_dosc', params['Ze_dosc'], comment='Look up table - dosc version')
    #traj.LT.Zi_dosc     = Parameter('Zi_dosc', params['Zi_dosc'], comment='Look up table - dosc version')
    traj.LT.precalc_r    = Parameter('precalc_r', params['precalc_r'], comment='Look up table')
    traj.LT.precalc_V    = Parameter('precalc_V', params['precalc_V'], comment='Look up table')
    traj.LT.precalc_tau_mu    = Parameter('precalc_tau_mu', params['precalc_tau_mu'], comment='Look up table')
    traj.LT.precalc_tau_sigma    = Parameter('precalc_tau_sigma', params['precalc_tau_sigma'], comment='Look up table')
    #traj.LT.Zi          = Parameter('Zi', params['Zi'], comment='Look up table')
    traj.LT.Irange      = Parameter('Irange', params['Irange'], comment='Range of the mean input for the look up tables')
    traj.LT.sigmarange  = Parameter('sigmarange', params['sigmarange'], comment='Range of the input variance for the look up tables')
    traj.LT.dI          = Parameter('dI', params['dI'], comment='Step size of the mean input for the look up tables')
    traj.LT.ds          = Parameter('ds', params['ds'], comment='Step size of the input variance for the look up tables')

########################################################################
###### Function for loading parameter exploration results and   ########
###### computing the correlation with the empirical BOLD signal ########
########################################################################
def getTrajectoryNameInsideFile(fName):
    '''
    Return a list of all pypet trajectories name saved in a a given hdf5 file.

    Parameter:
        :param fName:   Name of the hdf5 we want to explore

    Return:
        List of string containing the trajectory name
    '''
    hdf = h5py.File(fName)
    all_traj_names = list(hdf.keys())
    hdf.close()
    return all_traj_names

