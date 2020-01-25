import numpy as np
import copy as cp
import h5py
import time
from numpy import dtype

from neurolib.models.hopf.timeIntegration import timeIntegration
import neurolib.models.hopf.loadDefaultParams as dp

def chunkwiseTimeIntegration(params, chunkSize=10000, saveAllActivity = False):
    totalDuration = params["duration"]

    dt = params["dt"]
    samplingRate_NDt = int(round(2000/dt))
    
    Dmat = dp.computeDelayMatrix(params["lengthMat"],params["signalV"])
    Dmat_ndt    = np.around(Dmat/dt)  # delay matrix in multiples of dt
    max_global_delay = np.amax(Dmat_ndt) 
    delay_Ndt   = int(max_global_delay+1)

    paramsChunk = cp.deepcopy(params)
    
    N = params["Cmat"].shape[0]
        
    t_BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    BOLD_return = np.array([], dtype = 'f', ndmin = 2)
    all_xs = np.array([], dtype = 'f', ndmin = 2)
    
    
    idxLastT = 0    # Index of the last computed t

    nround = 0 # how many cunks simulated?
    while dt * idxLastT< totalDuration:
        # Determine the size of the next chunk
        currentChunkSize = min( chunkSize + delay_Ndt, totalDuration - dt * idxLastT + (delay_Ndt + 1) * dt)
        paramsChunk["duration"] = currentChunkSize

        # Time Integration
        t_chunk, xs_chunk, ys_chunk = timeIntegration(paramsChunk)

        # Prepare integration parameters for the next chunk
        paramsChunk["xs_init"]    = xs_chunk[:,-int(delay_Ndt):]
        paramsChunk["ys_init"]    = ys_chunk[:,-int(delay_Ndt):]



        if nround == 0:
            xs_return = xs_chunk
        else:
            xs_return = xs_chunk[:,int(delay_Ndt):]
            
        if saveAllActivity:
            if all_xs.shape[1] == 0: # first time?
                all_xs = xs_return
            else:
                all_xs = np.hstack((all_xs, xs_return))

        del xs_chunk
        
        idxLastT = idxLastT + xs_return.shape[1]
        t_return = np.dot(range(xs_return.shape[1], idxLastT + xs_return.shape[1]), dt)    

        nround += 1 # increase chunk counter
    
    if saveAllActivity:
        xs_return = all_xs
            
    return t_return, xs_return, ys_chunk