from scipy.optimize import bisect
import numpy as np
import numba

from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
#import Simulation

#import analytic_sol as Analytic

# 4 different background noise states (mu,sigma) in nA: (low mu, low sig), (low mu, high sig), (high mu, low sig), (high mu, high sig)
noiseBackGroundStates_LIF_OU = [(0.0049913216140622316, 0.015342881944444442),
 (0.0049913216140622316, 0.060536024305555547),
 (0.010151039399773795, 0.015342881944444442),
 (0.010151039399773795, 0.060536024305555547)]

noiseBackGroundStates_EIF_OU = [
        (0.0053945861303803695, 0.027832709418402771),
        (0.0053945861303803695, 0.098247612847222193),
        (0.013377155530318902, 0.027832709418402771),
        (0.013377155530318902, 0.098247612847222193)]

# 4 different background noise states (mu,sigma) in nA: (low mu, low sig), (low mu, high sig), (high mu, low sig), (high mu, high sig)
# For the LIF neuron with a white noise input and dt=0.025 ms
noiseBackGroundStates_LIF_WN_0_25 = [
        (0.0010399202179046496, 0.11879340277777777),
        (0.0083193617432371966, 0.11879340277777777),
        (0.0010399202179046496, 0.31870659722222222),
        (0.0083193617432371966, 0.31870659722222222)]

# 4 different background noise states (mu,sigma) in nA: (low mu, low sig), (low mu, high sig), (high mu, low sig), (high mu, high sig)
# For the EIF neuron with a white noise input and dt=0.025 ms
noiseBackGroundStates_EIF_WN_0_25 = [
        (0.0010399202179046496, 0.18350694444444443),
        (0.0010399202179046496, 0.39618055555555554),
        (0.0083193617432371966, 0.18350694444444443),
        (0.0083193617432371966, 0.39618055555555554)]

class IFModel:
    '''
    Simple leak IF models used to compute the sensitivity to the field.
    This simple neuron as no time integration function and cannot fire
    '''
    def __init__(self):
        self.rhom_dend  = 28.0e-1   # Dendritic tree membrane resistivity (Ohm m2)
        self.rha_dend   = 1.5       # Dendritic tree membrane internal resistivity (Ohm m)
        self.d_dend     = 1.2e-6    # Dendritic tree diameter (m)
        self.L          = 700.0e-6  # Dendritic tree length (m)

        # Soma (here we take the same values for the soma and dendritic tree as was done in Rattay 1999)
        self.C_soma     = 1.0e-2   # Soma membrane capacitance (F / m2)
        self.rhom_soma  = 28.0e-1  # Soma membrane resistivity (Ohm m2)
        self.d_soma     = 10e-6    # Soma diameter ( m )

        self.updateParams()

        # Firing parameters 
        self.V_cut      = -40e-3    # Voltage cut for the firing (V)
        self.T_ref      = 1.5e-3    # Refractory time (s)
        self.V_reset    = -70e-3    # Reset membrane potential (V)
        
        self.EL     = -65e-3    # V

    def updateParams(self):
        '''
        This function should be called everytime some parameters of the model are changed.
        It is used to update the parameters used in the polarization transfer function. 
        '''
        # Cable resistance per unit length
        self.r_m    = self.rhom_dend / ( np.pi * self.d_dend )        # Tangential resistance
        self.r_i    = self.rha_dend / ( np.pi * (self.d_dend/2) ** 2) # Axial resistance

        # Cable specific length
        self.R_s    = self.rhom_soma / ( np.pi * self.d_soma ** 2 )
        self.C_s    = self.C_soma * np.pi * self.d_soma ** 2
        self.lambd  = np.sqrt( self.r_m / self.r_i ) 
        self.gamma  = self.R_s / ( self.r_i * self.lambd )
        self.tau_m  = self.R_s * self.C_s
        
        self.gL     = 1 / self.R_s
        
    def polarizationTransfer(self,freqs):
            w = 2 * np.pi * freqs
            
            Rs = self.rhom_soma / ( np.pi * self.d_soma ** 2 )
            Cs = self.C_soma * np.pi * self.d_soma ** 2 
            Gs = 1 / Rs
            gm = 1 / self.r_m
            cm = self.C_soma * (self.d_dend * np.pi)
            gi = 1 / self.r_i

            alpha   = np.sqrt( ( gm     + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            beta    = np.sqrt( ( -gm    + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            z       = alpha + 1.j*beta  # z2 = -z

            dummy   = 1 + np.exp(-2 * z * self.L)
            denom   = dummy * ( Cs * w * 1.j + Gs) + z * gi * ( 2 - dummy )
            #V1_BS_over_I1(iw) = dummy / denom
            return gi * ( 2 * np.exp( -z * self.L ) - dummy ) / denom

    def fieldEqInputCurrent(self,V0,freqs):
        '''
        Return the input current equivalent to the extracellular electrical field for the given
            baseline membrane potential and field frequency

        The equivalent input current is complex, its imaginary part should be taken in case of a sinusoidal field
        '''
        # We just need to compute the factor which is due to the time integration of the current
        return (self.gL + self.C_s * 1j* 2*np.pi * freqs) * self.polarizationTransfer(freqs)

    def runSim(self,duration=500e-3,E_onset=100e-3,E_amp=1,E_freq=10,E_n=1000,V0=-65e-3,dt=0.1e-3,I_ext=None,I_amp=0,I_freq=0): 
        '''
        Run a time integration of the IF neuron model to which the input current equivalent to the field is added
        
        Parameters:
        :param duration:    Simulation duration (s)
        :param offset:      Time at which the field start (s)
        :param E_amp:       Field amplitude (V/m)
        :param E_freq:      Field frequency (Hz)
        :param V0:          Base Membrane potential (resting value when no field is applied) (V)
        :param I_ext:       External input received by the neuron at the soma  (A)
                                Numpy.array of length (duration/dt)

        :returns:   (t,V,E): 3 numpy.array of length duration/dt corresponding to:
                            - the time values (s)
                            - the membrane voltage ( mV )
                            - the field intensity ( V/m )
        '''
        
        # Compute the synaptic input corresponding to the resting membrane potential
        Isyn = self.gL*(V0-self.EL)
        
        t = np.arange(0,duration,dt)
        nT = len(t)
        # Precompute the equivalent input current
        E_comp = np.zeros((nT,),dtype=np.complex128)
        E_offset = E_onset + 1.0 * E_n / E_freq
        E_comp[(t > E_onset) & (t < E_offset)] = E_amp * np.exp(2*np.pi * E_freq * (t[(t>E_onset) & (t < E_offset)] - t[t>E_onset][0]) * 1j)
        
        IE = np.imag(E_comp * self.fieldEqInputCurrent(V0,E_freq))
        E = np.imag(E_comp)
        
        V = np.zeros(E.shape)
        V[0] = V0
       
        if I_ext is None:
            I_ext = I_amp * np.sin(2*np.pi*I_freq*t)

        T_refIndex = int(np.ceil(self.T_ref / dt))
        
        IFModel._integrationIF(V,IE,nT, \
                                            self.C_s,self.gL,self.EL,\
                                            Isyn,dt,I_ext,self.V_reset,self.V_cut,T_refIndex)
        
        return t,V,E


    ### Numba functions for the time integration ###
    @numba.njit
    def _integrationIF(V,IE,nT,C_s,gL,EL,Isyn,dt,I_ext,V_reset,V_cut,T_refIndex):
        
        lastFiringIndex = -T_refIndex
        
        for i in xrange(nT-1):
            
            # Are we in the refractory time?
            if i - lastFiringIndex < T_refIndex:
                V[i+1] = V_reset
                continue
           
            
            V[i+1] = V[i] + dt / C_s * (-gL * ( V[i]-EL ) + IE[i] + Isyn + I_ext[i])
            
            if V[i+1] > V_cut:
                lastFiringIndex = i+1
                V[i+1] = V_reset

    def computeEquivalentInputCurrentFromBSInput(self,t_BS,I_BS,V0=-65.e-3):
        '''
        Compute the equivalent input current to apply on the point neuron model to reproduce the oscillations
        observed in the B&S model for an input current at the soma

        WARNING:
            this is an analytical solutions mvalid oly for the passive case:
                - linearization of the exp mech.
                - Slow adaptation limit
            

        Parameters:
            :param t_BS:    Time stamp corresponding to I_BS (s)
            :param I_BS:    Input current applied at the B&S soma (nA)
            :param V0:      Base membrane potential (V)

        :returns:   I_Point: numpy.array containing the time serie of the equivalent input current to apply
                                on the point neuron (A)

        '''
        # Decompose the input current on the BS model into its fourier coefs
        fcoefs_I_BS = np.fft.fft(I_BS)
        freqs = np.fft.fftfreq(len(I_BS),np.diff(t_BS)[0])
            
        # Compute the membrane potential oscillations due to the input currents in the fourier space
        factors_BS = self.computeFourierCoeffFactors_I_to_V(freqs)
        fcoefs_V = fcoefs_I_BS * factors_BS

        # Compute the equivalent input current for Point neuron model in the foureier space
        factors_PointNeuron = 1/self.computePointNeuronImpedance(freqs)
    
        fcoefs_I_Point = fcoefs_V * factors_PointNeuron

        # Reconstitute the input current from its foureier coefs
        I_Point = np.real( np.fft.ifft(fcoefs_I_Point) )

        return I_Point

    def computePointNeuronImpedance(self,freqs):
        '''
        Compute the point neuron impedance.

        Parameters:
            :param freqs:   Frequencies for which the impedance should be computed (Hz)lt

        :returns:   Complex impedances
        '''
        return 1/(self.gL +  2 * 1.j * np.pi * freqs * self.C_s)



    def computeFourierCoeffFactors_I_to_V(self, freqs):
            '''
            Compute the Fourier coeff factor to apply on the input current fourier decomposition
            in order to get the membrane voltage variation at the B&S soma

            For legacy, this function returns the coefficients in V/nA
        
            Use the fourier method instead of the Green's function
            '''
            w = 2 * np.pi * freqs
            
            Gs = 1 / self.R_s
            gm = 1 / self.r_m
            cm = self.C_soma * (self.d_dend * np.pi)
            gi = 1 / self.r_i

            alpha   = np.sqrt( ( gm     + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            beta    = np.sqrt( ( -gm    + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            z       = alpha + 1.j*beta  # z2 = -z
            z[w<0]  = np.conj(z[w<0])

            dummy   = 1 + np.exp(-2 * z * self.L)
            denom   = dummy * ( self.C_s * w * 1.j + Gs) + z * gi * ( 2 - dummy )
            return dummy / denom * 1e-9

    def setNeuronParamForMembranePotential(self):
        '''
        Set the neuron parameter in order to have a mebrane potential oscillating around the 0 value and a positive threshold.
        '''
        self.EL     = 0
        self.V_cut  = 10e-3
        self.V_reset= 0

class EIFModel(IFModel):
    def __init__(self):
        IFModel.__init__(self)

        self.deltaT = 1.5e-3      # V
        self.VT     = -50e-3    # V
        self.EL     = -65e-3    # V
        self.factorExp = 1

        self.linearVersion = False  # True if we want to use the linear approximation ( for test ), in which
                                    #   the exponential ion conductance are passive

    def setNeuronParamForMembranePotential(self):
        self.EL     = 0
        self.V_cut  = 20e-3
        self.VT     = 10e-3
        self.V_reset= 0

    def fieldEqInputCurrent(self,V0,freqs):
        '''
        Return the input current equivalent to the extracellular electrical field for the given
            baseline membrane potential and field frequency

        The equivalent input current is complex, its imaginary part should be taken in case of a sinusoidal field
        '''
        # We just need to compute the factor which is due to the time integration of the current
        return (self.gL * (1-self.factorExp * np.exp((V0-self.VT)/self.deltaT)) + self.C_s * 1j* 2*np.pi * freqs) * self.polarizationTransfer(freqs)

    def runSim(self,duration=500e-3,E_onset=100e-3,E_amp=1,E_freq=10,E_n=1000,V0=-65e-3,dt=0.1e-3): 
        '''
        Run a time integration of the EIF neuron model to which the input current equivalent to the field is added
        
        :param duration: Simulation duration (s)
        :param offset:   Time at which the field start (s)
        :param E_amp:    Field amplitude (V/m)
        :param E_freq:   Field frequency (Hz)
        :param V0:       Base Membrane potential (resting value when no field is applied) -(V)

        :returns:   (t,V,E): 3 numpy.array of length duration/dt corresponding to:
                            - the time values (s)
                            - the membrane voltage ( mV )
                            - the field intensity ( V/m )
        '''
        # Compute the synaptic input corresponding to the resting membrane potential
        if not self.linearVersion:
            Isyn = self.gL * (V0-self.EL) - self.gL * self.factorExp * self.deltaT * np.exp((V0-self.VT)/self.deltaT)
        else:
            Isyn = self.gL*(V0-self.EL) - self.gL * self.factorExp * self.deltaT * (1+(V0-self.VT)/self.deltaT)
        
        t = np.arange(0,duration,dt)
        nT = len(t)

        # Precompute the equivalent input current
        E_comp = np.zeros((nT,),dtype=np.complex128)
        E_offset = E_onset + 1.0 * E_n / E_freq
        E_comp[(t > E_onset) & (t < E_offset)] = E_amp * np.exp(2*np.pi * E_freq * (t[(t>E_onset) & (t < E_offset)] - t[t>E_onset][0]) * 1j)
        
        IE = np.imag(E_comp * self.fieldEqInputCurrent(V0,E_freq))
        E = np.imag(E_comp)
        
        V = np.zeros(E.shape)
        V[0] = V0
       
        for i in xrange(nT-1):

            # Non linear version
            if not self.linearVersion:
                V[i+1] = V[i] + dt / self.C_s * (-self.gL * ( V[i]-self.EL ) \
                                                    + self.gL * self.factorExp * self.deltaT * np.exp((V[i] - self.VT)/self.deltaT) \
                                                    + IE[i] + Isyn)
            else:
                V[i+1] = V[i] + dt / self.C_s * (-self.gL * ( V[i]-self.EL ) \
                                                    + self.gL * self.factorExp * self.deltaT * (1+(V[i] - self.VT)/self.deltaT) \
                                                                                * np.exp((V0-self.VT)/self.deltaT) \
                                                    + IE[i] + Isyn)
        return t,V,E

    def computeEquivalentInputCurrentFromBSInput(self,t_BS,I_BS,V0=-65.e-3):
        '''
        Compute the equivalent input current to apply on the point neuron model to reproduce the oscillations
        observed in the B&S model for an input current at the soma

        WARNING:
            this is an analytical solutions mvalid oly for the passive case:
                - linearization of the exp mech.
                - Slow adaptation limit
            

        Parameters:
            :param t_BS:    Time stamp corresponding to I_BS (s)
            :param I_BS:    Input current applied at the B&S soma (nA)
            :param V0:      Base membrane potential (V)

        :returns:   I_Point: numpy.array containing the time serie of the equivalent input current to apply
                                on the point neuron (A)

        '''
        # Decompose the input current on the BS model into its fourier coefs
        fcoefs_I_BS = np.fft.fft(I_BS)
        freqs = np.fft.fftfreq(len(I_BS),np.diff(t_BS)[0])
            
        # Compute the membrane potential oscillations due to the input currents in the fourier space
        factors_BS = self.computeFourierCoeffFactors_I_to_V(freqs)
        fcoefs_V = fcoefs_I_BS * factors_BS

        # Compute the equivalent input current for Point neuron model in the foureier space
        factors_PointNeuron = 1/self.computePointNeuronImpedance(freqs,V0)
    
        fcoefs_I_Point = fcoefs_V * factors_PointNeuron

        # Reconstitute the input current from its foureier coefs
        I_Point = np.real( np.fft.ifft(fcoefs_I_Point) )

        return I_Point

    def computePointNeuronImpedance(self,freqs, V0 = -65e-3):
        '''
        Compute the point neuron impedance.

        Parameters:
            :param freqs:   Frequencies for which the impedance should be computed (Hz)
            :param V0:      Base membrane potential (V)

        :returns:   Complex impedances
        '''
        return 1/ ( self.gL * (1 - self.factorExp * np.exp( ( V0 - self.VT ) / self.deltaT ) ) \
                    +  2 * 1.j * np.pi * freqs * self.C_s )


class aEIFModel(EIFModel):
    '''
    This class corresponds to an adaptive Exonential Integrate and Fire neuron model.
    The equivalent input current accounting for the extracellular field is computed using the slow adaptation
        approximation ( see "Equivalent current" notebook ). In this case the eq input current equation is the same as for
        the EIF model

    This neuron model enable simulating the firing of the neuron.
    This is currently implented for the the case where slowLimit=False
    '''
    
    def __init__(self):
        EIFModel.__init__(self)

        self.EW         = -80e-3 # V
        self.aAdapt     = 0.06*1e-3 # S/cm2
        self.TAdapt     = 0.2    # s
        self.bAdapt     = 10 # TODO
        self.slowLimit  = False  # True if we want to the the slow adaptation limit

        # Firing parameters 
        self.V_cut      = -40e-3    # Voltage cut for the firing (V)
        self.T_ref      = 1.5e-3    # Refractory time (s)
        self.V_reset    = -65e-3    # Reset membrane potential (V)

    def runSim(self,duration=500e-3,E_onset=100e-3,E_amp=1,E_freq=10,E_n=1000,V0=-65e-3,dt=0.1e-3,I_ext=None,I_amp=0,I_freq=0,WOutput=None): 
        '''
        Run a time integration of the aEIF neuron model to which the input current equivalent to the field is added
        
        slowLimit: The slow limit corresponds to the limit were the adaptation does not change in time ( its time constant is infinite)
        linearVersion: The linear version ( avalaible onlyin the slowLimit ) corresponds to the linearized version of the integral
        
        Parameters:
        :param duration:    Simulation duration (s)
        :param offset:      Time at which the field start (s)
        :param E_amp:       Field amplitude (V/m)
        :param E_freq:      Field frequency (Hz)
        :param V0:          Base Membrane potential (resting value when no field is applied) (V)
        :param I_ext:       External input received by the neuron at the soma  (A)
                                Numpy.array of length (duration/dt)
        :param WOutput:     Output list in which the value of the adaptation current will be saved (W)

        :returns:   (t,V,E): 3 numpy.array of length duration/dt corresponding to:
                            - the time values (s)
                            - the membrane voltage ( mV )
                            - the field intensity ( V/m )
        '''
        aAdapt_Soma = self.aAdapt * np.pi * (self.d_soma*100)**2    # S/cm2 * area /1000 (S)
        
        # Compute the synaptic input corresponding to the resting membrane potential
        if not self.slowLimit:
            Isyn = self.gL*(V0-self.EL) - self.gL * self.factorExp * self.deltaT * np.exp((V0-self.VT)/self.deltaT) + aAdapt_Soma * (V0-self.EW)
        elif not self.linearVersion:
            Isyn = self.gL*(V0-self.EL) - self.gL * self.factorExp * self.deltaT * np.exp((V0-self.VT)/self.deltaT) + aAdapt_Soma * (V0-self.EW)
        else:
            Isyn = self.gL*(V0-self.EL) - self.gL * self.factorExp * self.deltaT * (1+(V0-self.VT)/self.deltaT) * np.exp((V0-self.VT)/self.deltaT) \
                    + aAdapt_Soma * (V0-self.EW)
        
        t = np.arange(0,duration,dt)
        nT = len(t)
        # Precompute the equivalent input current
        E_comp = np.zeros((nT,),dtype=np.complex128)
        E_offset = E_onset + 1.0 * E_n / E_freq
        E_comp[(t > E_onset) & (t < E_offset)] = E_amp * np.exp(2*np.pi * E_freq * (t[(t>E_onset) & (t < E_offset)] - t[t>E_onset][0]) * 1j)
        
        IE = np.imag(E_comp * self.fieldEqInputCurrent(V0,E_freq))
        E = np.imag(E_comp)
        
        V = np.zeros(E.shape)
        V[0] = V0
        W = np.ones(E.shape) * aAdapt_Soma * (V0-self.EW)
       
        if I_ext is None:
            I_ext = I_amp * np.sin(2*np.pi*I_freq*t)

        T_refIndex = int(np.ceil(self.T_ref / dt))
        
        if not self.slowLimit: # Without the slow limit we use automatically the non-linear version
            aEIFModel._integrationNoSlowLimit(V,W,IE,nT,self.TAdapt,aAdapt_Soma,self.EW, \
                                                self.C_s,self.gL, self.factorExp, self.EL,self.deltaT,self.VT,\
                                                Isyn,dt,V0,I_ext,self.V_reset,self.V_cut,T_refIndex)
        elif self.linearVersion:
            aEIFModel._integrationSlowLimitLinear(V,W,IE,nT,self.TAdapt,aAdapt_Soma,self.EW, \
                                                self.C_s,self.gL, self.factorExp, self.EL,self.deltaT,self.VT,Isyn,dt,V0,I_ext)
        else:
            aEIFModel._integrationSlowLimitNonLinear(V,W,IE,nT,self.TAdapt,aAdapt_Soma,self.EW, \
                                                self.C_s,self.gL, self.factorExp, self.EL,self.deltaT,self.VT,Isyn,dt,V0,I_ext)
       
        if not WOutput is None:
            WOutput.extend(W.tolist())
        return t,V,E


    ### Numba functions for the time integration ###
    @numba.njit
    def _integrationNoSlowLimit(V,W,IE,nT,TAdapt,aAdapt_Soma,EW,C_s,gL, factorExp, EL,deltaT,VT,Isyn,dt,V0,I_ext,V_reset,V_cut,T_refIndex):
        
        lastFiringIndex = -T_refIndex
        
        for i in xrange(nT-1):
            W[i+1] = W[i] + dt/TAdapt * (aAdapt_Soma * (V[i]-EW) - W[i])
            
            if i - lastFiringIndex < T_refIndex:
                V[i+1] = V_reset
                continue
            
            V[i+1] = V[i] + dt / C_s * (-gL * ( V[i]-EL ) - W[i] \
                                                + gL * factorExp * deltaT * np.exp((V[i] - VT)/deltaT) \
                                                + IE[i] + Isyn + I_ext[i])
            
            if V[i+1] > V_cut:
                lastFiringIndex = i+1
                #W[i+1] += b # TODO spike triggered adaptation
                V[i+1] = V_reset
    @numba.njit
    def _integrationSlowLimitNonLinear(V,W,IE,nT,TAdapt,aAdapt_Soma,EW,C_s,gL, factorExp, EL,deltaT,VT,Isyn,dt,V0,I_ext):
        for i in xrange(nT-1):
            V[i+1] = V[i] + dt / C_s * (-gL * ( V[i]-EL ) \
                                            + gL * factorExp * deltaT * np.exp((V[i] - VT)/deltaT) \
                                            - W[i] + IE[i] + Isyn + I_ext[i])

    @numba.njit
    def _integrationSlowLimitLinear(V,W,IE,nT,TAdapt,aAdapt_Soma,EW,C_s,gL, factorExp, EL,deltaT,VT,Isyn,dt,V0,I_ext):
        for i in xrange(nT-1):
            V[i+1] = V[i] + dt / C_s * (-gL * ( V[i]-EL ) \
                                            + gL * factorExp * deltaT * (1+(V[i] - VT)/deltaT) * np.exp((V0-VT)/deltaT) \
                                            - W[i] + IE[i] + Isyn + I_ext[i])


def computeFreqDepSensitivityAndPhaseShift(freqs,E_amp,V0,runIFSim,corrCoefs=[],dt=0.1e-3,plot=False,**kw):
    tOnset = 100e-3

    sensitivities = []
    phases = []

    for f in freqs:
        # Determine the necessary simulation length
        tStop = tOnset + np.max([400e-3, 2/f])
        
        T,V,E = runIFSim(duration=tStop,E_onset=tOnset,E_amp=E_amp,E_freq=f,V0=V0,dt=dt,**kw)
        
        if plot: # Plotting for debug
            plt.figure()
            ax = plt.subplot(211)
            plt.plot(T,V*1e3)
            plt.ylabel("Membrane potential (mV)")
            plt.subplot(212,sharex=ax)
            plt.plot(T,E)
            plt.xlabel("Time (s)")
            plt.ylabel("Field intensity (V/m)")
            plt.suptitle("Field frequency: "+str(f))
        
        idxStimStart    = find(T > tOnset)[0]
        idxStimStop     = find(T < tOnset + 1.0 * 1000 / f)[-1]

        argRelMinimaV = idxStimStart + argrelextrema(V[idxStimStart:idxStimStop],np.less_equal)[0]
        argRelMinimaV = argRelMinimaV[argRelMinimaV < idxStimStop-1]

        argRelMaximaV = idxStimStart + argrelextrema(V[idxStimStart:idxStimStop],np.greater_equal)[0]
        argRelMaximaV = argRelMaximaV[argRelMaximaV < idxStimStop-1]
        
        argRelMinimaE = idxStimStart + argrelextrema(E[idxStimStart:idxStimStop],np.less_equal)[0]
        argRelMinimaE = argRelMinimaE[argRelMinimaE < idxStimStop-1]

        # Compute the amplitude
        lower   = V[ argRelMinimaV[-1] ]
        higher  = V[ argRelMaximaV[-1]  ]
        amp = (higher - lower) / 2
        sensitivities.append(amp / E_amp)
        
        # Compute the phase shift
        tPeakE     =  T[ argRelMinimaE[-1] ]
        tPeakV     =  T[ argRelMaximaV[-1] ]
       
        # Note: the soma polarization and the field are out of phase, therefore we measure the last field minima
        #   and the last polarization maxima for computing the phase to which we had np.pi (out-of-phase)
        # Note: since the soma is delayed in comparison to the field, the phase shift (phase difference) should be negative
        phase = 2 * np.pi * (tPeakE - tPeakV)* f - np.pi
        if phase < -2*np.pi:
            phase += 2 * np.pi
        if phase > 0:
            phase -= 2 * np.pi
        phases.append( phase )
        
        if not corrCoefs is None:
            scaledField, shiftTime = Simulation.scaleAndShiftField(T,V,E, E_amp, f,sensitivities[-1],phases[-1],tOnset)
            corrCoef =np.corrcoef(V[T>tOnset + shiftTime],scaledField[T>tOnset+shiftTime])[1,0]
            corrCoefs.append(corrCoef) 

    return sensitivities,phases
