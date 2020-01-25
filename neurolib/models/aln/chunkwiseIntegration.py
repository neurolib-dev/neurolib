import numpy as np
import copy as cp
import h5py
import time
from numpy import dtype

from neurolib.models.aln.timeIntegration import timeIntegration
from neurolib.models import bold

import neurolib.models.aln.loadDefaultParams as dp

def chunkwiseTimeIntAndBOLD(params, chunkSize=10000, simulateBOLD = True, returnAllRates = False):
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
    delay_Ndt   = int(max(max_global_delay, ndt_de, ndt_di)+1)

    paramsChunk = cp.deepcopy(params)
    
    N = params["Cmat"].shape[0]
    
    if simulateBOLD:
        boldModel = bold.BOLDModel(N, dt)
    
    t_BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    all_Rates = np.array([], dtype = 'f', ndmin = 2)
    rates_exc_return = np.array([], dtype = 'f', ndmin = 2)

    idxLastT = 0    # Index of the last computed t

    while dt * idxLastT< totalDuration:
        # Determine the size of the next chunk
        currentChunkSize = min( chunkSize + delay_Ndt, totalDuration - dt * idxLastT + (delay_Ndt + 1) * dt )
        paramsChunk["duration"] = currentChunkSize

        # Time Integration
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

        paramsChunk["rates_exc_init"] = rates_exc_chunk[:,-int(delay_Ndt):]
        paramsChunk["rates_inh_init"] = rates_inh_chunk[:,-int(delay_Ndt):]


        rates_exc_return    = rates_exc_chunk[:,int(delay_Ndt):] # cut off initial condition transient, otherwise it would repeat
        del rates_exc_chunk

        if returnAllRates:
            if all_Rates.shape[1] == 0: # first time?
                all_Rates = rates_exc_return
            else:
                all_Rates = np.hstack((all_Rates, rates_exc_return))

        # Run BOLD model
        boldModel.run(rates_exc_return*1e3)
        BOLD_return = boldModel.BOLD
        t_BOLD_return = boldModel.t_BOLD
        # in crement time counter
        idxLastT = idxLastT + rates_exc_return.shape[1]
    
        if returnAllRates:
            rates_exc_return = all_Rates

        return_from_timeIntegration = (rates_exc_return, rates_inh_chunk, t_chunk,\
                        mufe, mufi, IA, seem, seim, siem, siim, \
                        seev, seiv, siev, siiv, integrated_chunk, rhs_chunk)
            
    return t_BOLD_return, BOLD_return, return_from_timeIntegration
    