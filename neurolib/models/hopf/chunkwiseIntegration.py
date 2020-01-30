import copy as cp

import numpy as np

import neurolib.models.hopf.loadDefaultParams as dp
from neurolib.models import bold
from neurolib.models.hopf.timeIntegration import timeIntegration


def chunkwiseTimeIntegration(
    params, chunkSize=10000, simulateBOLD=True, saveAllActivity=False
):
    # time stuff
    totalDuration = params["duration"]

    dt = params["dt"]
    samplingRate_NDt = int(round(2000 / dt))

    Dmat = dp.computeDelayMatrix(params["lengthMat"], params["signalV"])
    Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
    max_global_delay = np.amax(Dmat_ndt)
    delay_Ndt = int(max_global_delay + 1)

    paramsChunk = cp.deepcopy(params)

    N = params["Cmat"].shape[0]

    if simulateBOLD:
        boldModel = bold.BOLDModel(N, dt)

    # initialize data arrays
    t_BOLD_return = np.array([], dtype="f", ndmin=2)
    BOLD_return = np.array([], dtype="f", ndmin=2)
    all_xs = np.array([], dtype="f", ndmin=2)
    xs_return = np.array([], dtype="f", ndmin=2)
    all_ys = np.array([], dtype="f", ndmin=2)
    ys_return = np.array([], dtype="f", ndmin=2)

    idxLastT = 0  # Index of the last computed t

    nround = 0  # how many cunks simulated?
    while dt * idxLastT < totalDuration:
        # Determine the size of the next chunk
        currentChunkSize = min(
            chunkSize + delay_Ndt, totalDuration - dt * idxLastT + (delay_Ndt + 1) * dt
        )
        paramsChunk["duration"] = currentChunkSize

        # Time Integration
        t_chunk, xs_chunk, ys_chunk = timeIntegration(paramsChunk)

        # Prepare integration parameters for the next chunk
        paramsChunk["xs_init"] = xs_chunk[:, -int(delay_Ndt) :]
        paramsChunk["ys_init"] = ys_chunk[:, -int(delay_Ndt) :]

        xs_return = xs_chunk[:, int(delay_Ndt) :]
        ys_return = ys_chunk[:, int(delay_Ndt) :]
        del xs_chunk, ys_chunk

        if saveAllActivity:
            all_xs = np.hstack((all_xs, xs_return))
            all_ys = np.hstack((all_ys, ys_return))

        # BOLD model
        xsNormalized = xs_return
        xsNormalized = np.abs(xsNormalized)
        xsNormalized /= np.max(xsNormalized)
        xsNormalized *= 80.0

        if simulateBOLD:
            boldModel.run(xsNormalized)
            BOLD_return = boldModel.BOLD
            t_BOLD_return = boldModel.t_BOLD

        idxLastT = idxLastT + xs_return.shape[1]
        t_return = np.dot(range(xs_return.shape[1], idxLastT + xs_return.shape[1]), dt)

        nround += 1  # increase chunk counter

    if saveAllActivity:
        xs_return = all_xs
        ys_return = all_ys

    return t_return, xs_return, ys_return, t_BOLD_return, BOLD_return
