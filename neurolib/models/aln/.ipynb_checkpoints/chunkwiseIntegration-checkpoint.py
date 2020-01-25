import numpy as np
import copy as cp
import h5py
import time
from numpy import dtype

from neurolib.models.aln.timeIntegration import timeIntegration
from neurolib.models.aln.simulateBOLD import simulateBOLD

import neurolib.models.aln.loadDefaultParams as dp

def chunkwiseTimeIntAndBOLD(params, chunkSize=10000, callFunct = None, returnAllRates = False):
    '''
    Run the interareal network simulation with the parametrization params, compute the corresponding BOLD signal
    and store the result ( currently only BOLD signal ) in the hdf5 file fname_out
    The simulation runs in chunks to save working memory.
    chunkSize corresponds to the size in ms of a single chunk
    '''
    
    totalDuration = params["duration"]

    dt = params["dt"]
    samplingRate_NDt = int(round(2000/dt))
    
    Dmat = dp.computeDelayMatrix(params["lengthMat"],params["signalV"])
    Dmat_ndt    = np.around(Dmat/dt)  # delay matrix in multiples of dt
    ndt_de      = round(params["de"]/dt)
    ndt_di      = round(params["di"]/dt)
    max_global_delay = np.amax(Dmat_ndt) 
    delay_Ndt   = int(max(max_global_delay, ndt_de, ndt_di))

    paramsChunk = cp.deepcopy(params)
    
    N = params["Cmat"].shape[0]
        
    t_BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    all_Rates = np.array([], dtype = 'f', ndmin = 2)
    
    
    idxLastT = 0    # Index of the last computed t
    
    X_BOLD       = np.ones((N,));   # Vasso dilatory signal
    F_BOLD       = np.ones((N,));   # Blood flow
    Q_BOLD       = np.ones((N,));   # Deoxyhemoglobin
    V_BOLD       = np.ones((N,));   # Blood volume
    

    
    while dt * idxLastT< totalDuration:
        #print "%d"%(float(dt * float(idxLastT) / float(totalDuration)) * 100.0)
        # Determine the size of the next chunk
        currentChunkSize = min( chunkSize + delay_Ndt, totalDuration - dt * idxLastT+ (delay_Ndt + 1 )* dt )
        paramsChunk["duration"] = currentChunkSize

        # Time Integration
        # inititlize the variables, because they are initiated in the if ... else ... block otherwise
        t_chunk, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ext, mui_ext = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        rates_exc_chunk, rates_inh_chunk = np.array([], dtype = 'f', ndmin = 2), np.array([], dtype = 'f', ndmin = 2)      
        
        rates_exc_chunk, rates_inh_chunk, t_chunk, mufe, mufi, IA, seem, seim, siem, siim, \
                    seev, seiv, siev, siiv, integrated_chunk, rhs_chunk = timeIntegration(paramsChunk)

        # Prepare integration parameters for the next chunk
        paramsChunk["mufe_init"]    = mufe
        paramsChunk["mufi_init"]    = mufi
        paramsChunk["IA_init"]      = IA
        paramsChunk["seem_init"]     = seem
        paramsChunk["seim_init"]     = seim
        paramsChunk["siim_init"]     = siim
        paramsChunk["siem_init"]     = siem
        paramsChunk["seev_init"]     = seev
        paramsChunk["seiv_init"]     = seiv
        paramsChunk["siiv_init"]     = siiv
        paramsChunk["siev_init"]     = siev
        paramsChunk["mue_ext_init"] = mue_ext
        paramsChunk["mui_ext_init"] = mui_ext

        paramsChunk["rates_exc_init"] = rates_exc_chunk[:,-int(delay_Ndt):]
        paramsChunk["rates_inh_init"] = rates_inh_chunk[:,-int(delay_Ndt):]

        #del rates_inh_chunk
        
        if returnAllRates:
            if all_Rates.shape[1] == 0: # first time?
                all_Rates = rates_exc_chunk
            else:
                all_Rates = np.hstack((all_Rates, rates_exc_chunk))
        
        if BOLD_return.shape[1] == 0:
            rates_exc_return    = rates_exc_chunk
        else:
            rates_exc_return    = rates_exc_chunk[:,int(delay_Ndt):]
        del rates_exc_chunk

        
        # Compute the BOLD signal for the chunk
        BOLD_chunk, X_BOLD, F_BOLD, Q_BOLD, V_BOLD = simulateBOLD(rates_exc_return*1e3, dt*1e-3, 10000*np.ones((N,)),X=X_BOLD,F=F_BOLD,Q=Q_BOLD,V=V_BOLD);
        
        BOLD_resampled = BOLD_chunk[:, samplingRate_NDt - np.mod(idxLastT-1,samplingRate_NDt) : : samplingRate_NDt]
        del BOLD_chunk;

#        oldShape            = ds_t_BOLD.shape
        t_new_idx           = (idxLastT + np.arange(rates_exc_return.shape[1]))
        t_BOLD_resampled    = t_new_idx[ samplingRate_NDt - np.mod(idxLastT-1,samplingRate_NDt) : : samplingRate_NDt] * dt
        del t_new_idx
        
        if BOLD_return.shape[1] == 0:
            t_BOLD_return = t_BOLD_resampled
            BOLD_return = BOLD_resampled
        else:
            t_BOLD_return = np.hstack((t_BOLD_return, t_BOLD_resampled))
            BOLD_return = np.hstack((BOLD_return, BOLD_resampled))
    
        
        idxLastT = idxLastT + rates_exc_return.shape[1]
       
        del BOLD_resampled
        del t_BOLD_resampled
            
        if returnAllRates:
            rates_exc_return = all_Rates

        return_from_timeIntegration = (rates_exc_return, rates_inh_chunk, \
                        mufe, mufi, IA, seem, seim, siem, siim, \
                        seev, seiv, siev, siiv, integrated_chunk, rhs_chunk)
            
    return t_BOLD_return, BOLD_return, return_from_timeIntegration
    

def chunkwiseTimeIntegration(params,fname_out):
    chunkSize = 60000   # Chunk duration (ms) 

    totalDuration = params["duration"]

    dt = params["dt"]
   
    Dmat = dp.computeDelayMatrix(params["lengthMat"],params["signalV"])
    Dmat_ndt    = np.around(Dmat/dt)  # delay matrix in multiples of dt
    ndt_de      = round(params["de"]/dt)
    ndt_di      = round(params["di"]/dt)
    max_global_delay = np.amax(Dmat_ndt)
    delay_Ndt   = int(max(max_global_delay, ndt_de, ndt_di))

    paramsChunk = cp.deepcopy(params)

    t           = []
    N = params["Cmat"].shape[0]

    f = h5py.File(fname_out,'w')
    
    ds_rates_exc = f.require_dataset('/results/rates_exc',(N,0),dtype("f"),maxshape=(N,None),chunks=True)
    ds_rates_inh = f.require_dataset('/results/rates_inh',(N,0),dtype("f"),maxshape=(N,None),chunks=True)
    ds_t         = f.require_dataset('/results/t',(0,),dtype("f"),maxshape=(None,),chunks=True)
    idxH5 = 0

    while dt * idxH5 < totalDuration:

        currentChunkSize = min( chunkSize + delay_Ndt, totalDuration - dt * idxH5+ (delay_Ndt + 1 )* dt )
        currentChunkSize = max(currentChunkSize,delay_Ndt)
        paramsChunk["duration"] = currentChunkSize

        rates_exc_chunk,rates_inh_chunk,t_chunk, mufe, mufi, IA, seem, seim, siem, siim, \
                seev, seiv, siev, siiv, mue_ext, mui_ext = timeIntegration(paramsChunk)

        paramsChunk["mufe_init"]    = mufe
        paramsChunk["mufi_init"]    = mufi
        paramsChunk["IA_init"]      = IA
        paramsChunk["seem_init"]     = seem
        paramsChunk["seim_init"]     = seim
        paramsChunk["siim_init"]     = siim
        paramsChunk["siem_init"]     = siem
        paramsChunk["seev_init"]     = seev
        paramsChunk["seiv_init"]     = seiv
        paramsChunk["siiv_init"]     = siiv
        paramsChunk["siev_init"]     = siev
        paramsChunk["mue_ext_init"] = mue_ext
        paramsChunk["mui_ext_init"] = mui_ext

        paramsChunk["rates_exc_init"] = rates_exc_chunk[:,-delay_Ndt:]
        paramsChunk["rates_inh_init"] = rates_inh_chunk[:,-delay_Ndt:]

        if idxH5 == 0:
            rates_exc_return    = rates_exc_chunk
            rates_inhNew    = rates_inh_chunk
        else:
            rates_exc_return    = rates_exc_chunk[:,delay_Ndt:]
            rates_inhNew    = rates_inh_chunk[:,delay_Ndt:]

        t_new           = np.arange(idxH5, idxH5 + rates_exc_return.shape[1]) * dt

        ds_rates_exc.resize( (N, ds_rates_exc.shape[1] + rates_exc_return.shape[1]) )
        ds_rates_exc[:,idxH5:] = rates_exc_return 
        
        ds_rates_inh.resize( (N, ds_rates_inh.shape[1] + rates_inhNew.shape[1]) )
        ds_rates_inh[:,idxH5:] = rates_inhNew 
        
        ds_t.resize( ( ds_t.shape[0] + len(t_new), ) )
        ds_t[idxH5:] = t_new 
        
        idxH5 = idxH5 + rates_exc_return.shape[1]

        del rates_exc_chunk
        del rates_inh_chunk
        del rates_exc_return
        del rates_inhNew
        del t_new

    
    ds = f.require_dataset('/results/mufe',(N,),dtype("f"),chunks=True)
    ds = mufe
    ds = f.require_dataset('/results/mufi',(N,),dtype("f"),chunks=True)
    ds = mufi
    ds = f.require_dataset('/results/IA',(N,),dtype("f"),chunks=True)
    ds = IA
    ds = f.require_dataset('/results/seem',(N,),dtype("f"),chunks=True)
    ds = seem
    ds = f.require_dataset('/results/seim',(N,),dtype("f"),chunks=True)
    ds = seim
    ds = f.require_dataset('/results/siim',(N,),dtype("f"),chunks=True)
    ds = siim
    ds = f.require_dataset('/results/siem',(N,),dtype("f"),chunks=True)
    ds = siem
    ds = f.require_dataset('/results/seev',(N,),dtype("f"),chunks=True)
    ds = seev
    ds = f.require_dataset('/results/seiv',(N,),dtype("f"),chunks=True)
    ds = seiv
    ds = f.require_dataset('/results/siiv',(N,),dtype("f"),chunks=True)
    ds = siiv
    ds = f.require_dataset('/results/siev',(N,),dtype("f"),chunks=True)
    ds = siev
    ds = f.require_dataset('/results/mue_ext',(N,),dtype("f"),chunks=True)
    ds = mue_ext
    ds = f.require_dataset('/results/mui_ext',(N,),dtype("f"),chunks=True)
    ds = mui_ext

    f.close()

def graph_density(SC):
    nodes = float(SC.shape[1])
    if nodes > 1:
        edges = float(np.count_nonzero(SC)) 
        density = edges / (nodes * (nodes - 1.0))
        return density
    else:
        return 0

def adjust_density(structSC, taget_density):
    SC = structSC.copy()
    density = graph_density(SC)
    counter = 1
    while(density > taget_density and counter < 100000):  
        counter += 1
        SC[SC < 0.0001 * counter] = 0 
        density = graph_density(SC)
    return SC
