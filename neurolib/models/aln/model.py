import numpy as np 

import neurolib.models.aln.loadDefaultParams as dp
import neurolib.models.aln.timeIntegration as ti
import neurolib.models.aln.chunkwiseIntegration as cw


class ALNModel():
    """
    Multi-population mean-field model with exciatory and inhibitory neurons per population.
    """
    def __init__(self, params = None, Cmat = [], Dmat = [], lookupTableFileName = None, seed=None, \
        simulateChunkwise = False, chunkSize = 10000, simulateBOLD=False):
        """
        :param params: parameter dictionary of the model
        :param Cmat: Global connectivity matrix (connects E to E)
        :param Dmat: Distance matrix between all nodes (in mm)
        :param lookupTableFileName: Filename for precomputed transfer functions and tables
        :param seed: Random number generator seed
        :param simulateChunkwise: Chunkwise time integration (for lower memory use)
        :param simulateBOLD: Parallel (chunkwise) BOLD simulation 
        """
        # Global parameters
        self.Cmat = Cmat # Connectivity matrix
        self.Dmat = Dmat # Delay matrix
        self.lookupTableFileName = lookupTableFileName # Filename for aLN lookup functions
        self.seed = seed # Random seed
        
        # Chunkwise simulation and BOLD
        self.simulateChunkwise = simulateChunkwise # Chunkwise time integration
        self.simulateBOLD = simulateBOLD # BOLD
        if simulateBOLD:
            self.simulateChunkwise = True # Override this setting if BOLD is simulated!
        self.chunkSize = chunkSize # Size of integration chunks in chunkwise integration in case of simulateBOLD == True
        self.saveAllActivity = False # Save data of all chunks? Can be very memory demanding if simulations are long or large

        # load default parameters if none were given
        if params == None:
            self.params = dp.loadDefaultParams(Cmat = self.Cmat, Dmat = self.Dmat, \
                                               lookupTableFileName = self.lookupTableFileName, seed=self.seed)
        else:
            self.params = params
        
        
        
    def run(self):
        '''
        Runs an aLN mean-field model simulation
        '''  
        if self.simulateChunkwise:
            t_BOLD, BOLD, return_tuple = cw.chunkwiseTimeIntAndBOLD(self.params, self.chunkSize, \
                                                                    self.simulateBOLD, self.saveAllActivity)
            rates_exc, rates_inh, t,\
                            mufe, mufi, IA, seem, seim, siem, siim, \
                            seev, seiv, siev, siiv, integrated_chunk, rhs_chunk = return_tuple
            self.t_BOLD = t_BOLD
            self.BOLD = BOLD
            
        else:
            rates_exc, rates_inh, t, \
                    mufe, mufi, IA, seem, seim, siem, siim, \
                        seev, seiv, siev, siiv, integrated_chunk, rhs_chunk = ti.timeIntegration(self.params)
        
        # convert output from kHz to Hz
        rates_exc = rates_exc*1000.0
        rates_inh = rates_inh*1000.0
        stimulus = self.params['ext_exc_current']

        t = np.dot(range(rates_exc.shape[1]),self.params['dt'])    
        
        self.t = t
        self.rates_exc = rates_exc
        self.rates_inh = rates_inh
        self.input = stimulus
        
        #return t, rates_exc, rates_inh