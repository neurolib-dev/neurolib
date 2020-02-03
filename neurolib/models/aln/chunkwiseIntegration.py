import copy as cp

import numpy as np

import neurolib.models.aln.loadDefaultParams as dp
from neurolib.models import bold
from neurolib.models.aln.timeIntegration import timeIntegration


def chunkwiseTimeIntAndBOLD(params, chunkSize=10000, simulateBOLD=True, saveAllActivity=False):
    """
    Run the interareal network simulation with the parametrization params, compute the corresponding BOLD signal
    and store the result ( currently only BOLD signal ) in the hdf5 file fname_out
    The simulation runs in chunks to save working memory.
    chunkSize corresponds to the size in ms of a single chunk
    """
    # time stuff
    totalDuration = params["duration"]

    dt = params["dt"]
    samplingRate_NDt = int(round(2000 / dt))

    Dmat = dp.computeDelayMatrix(params["lengthMat"], params["signalV"])
    Dmat_ndt = np.around(Dmat / dt)  # delay matrix in multiples of dt
    ndt_de = round(params["de"] / dt)
    ndt_di = round(params["di"] / dt)
    max_global_delay = np.amax(Dmat_ndt)
    delay_Ndt = int(max(max_global_delay, ndt_de, ndt_di) + 1)

    # make a real copy of params
    paramsChunk = cp.deepcopy(params)
    N = params["Cmat"].shape[0]

    # initialize BOLD model?
    if simulateBOLD:
        boldModel = bold.BOLDModel(N, dt)

    # initialize data arrays
    t_BOLD_return = np.array([], dtype="f", ndmin=1)
    BOLD_return = np.array([], dtype="f", ndmin=2)

    all_t = np.array([], dtype="f", ndmin=1)
    t_return = np.array([], dtype="f", ndmin=1)
    all_rates_exc = np.array([], dtype="f", ndmin=2)
    rates_exc_return = np.array([], dtype="f", ndmin=2)
    all_rates_inh = np.array([], dtype="f", ndmin=2)
    rates_inh_return = np.array([], dtype="f", ndmin=2)

    lastT = 0
    while lastT < totalDuration:
        # Determine the size of the next chunk
        currentChunkSize = min(chunkSize + delay_Ndt, totalDuration - lastT + (delay_Ndt + 1) * dt)
        paramsChunk["duration"] = currentChunkSize

        rates_exc, rates_inh, t, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv = timeIntegration(paramsChunk)

        # Prepare integration parameters for the next chunk
        paramsChunk["mufe_init"] = mufe
        paramsChunk["mufi_init"] = mufi
        paramsChunk["IA_init"] = IA
        paramsChunk["seem_init"] = seem
        paramsChunk["seim_init"] = seim
        paramsChunk["siim_init"] = siim
        paramsChunk["siem_init"] = siem
        paramsChunk["seev_init"] = seev
        paramsChunk["seiv_init"] = seiv
        paramsChunk["siiv_init"] = siiv
        paramsChunk["siev_init"] = siev

        paramsChunk["rates_exc_init"] = rates_exc[:, -int(delay_Ndt) :]
        paramsChunk["rates_inh_init"] = rates_inh[:, -int(delay_Ndt) :]

        # rates_exc_return = rates_exc[:, int(delay_Ndt) :]  # cut off initial condition transient, otherwise it would repeat
        # rates_inh_return = rates_inh[:, int(delay_Ndt) :]
        # t_return = t[int(delay_Ndt) :]

        rates_exc_return = rates_exc[:, : -int(delay_Ndt)]  # cut off initial condition transient, otherwise it would repeat
        rates_inh_return = rates_inh[:, : -int(delay_Ndt)]
        t_return = t[: -int(delay_Ndt)]

        del rates_exc, rates_inh, t

        if saveAllActivity:
            all_rates_exc = np.hstack((all_rates_exc, rates_exc_return))
            all_rates_inh = np.hstack((all_rates_inh, rates_inh_return))
            all_t = np.hstack((all_t, np.add(t_return, lastT)))

        if simulateBOLD:
            # Run BOLD model
            boldModel.run(rates_exc_return * 1e3)
            BOLD_return = boldModel.BOLD
            t_BOLD_return = boldModel.t_BOLD

        # in crement time counter
        lastT = lastT + t_return[-1]

        if saveAllActivity:
            rates_exc_return = all_rates_exc
            rates_inh_return = all_rates_inh
            t_return = all_t

        return_from_timeIntegration = (rates_exc_return, rates_inh_return, t_return, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv)

    return t_BOLD_return, BOLD_return, return_from_timeIntegration
