import numpy as np 

import neurolib.models.hopf.loadDefaultParams as dp
import neurolib.models.hopf.timeIntegration as ti
import neurolib.models.hopf.chunkwiseIntegration as cw


class HopfModel():
    """
    Todo.
    """
    def __init__(self, params = None, Cmat = [], Dmat = [], lookupTableFileName = None, seed=None, simulateBOLD=False):
        if len(Cmat) == 0:
            self.singleNode = True
        else:
            self.singleNode = False
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.lookupTableFileName = lookupTableFileName
        self.seed = seed
        
        self.simulateBOLD = simulateBOLD
        
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
        if self.simulateBOLD:
            t_BOLD, BOLD, return_tuple = cw.chunkwiseTimeIntAndBOLD(self.params)
            rates_exc, rates_inh, \
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

        t = np.dot(range(len(rates_exc)),self.params['dt'])    
        
        self.t = t
        self.rates_exc = rates_exc
        self.rates_inh = rates_inh
        self.input = stimulus
        
        return t, rates_exc, rates_inh