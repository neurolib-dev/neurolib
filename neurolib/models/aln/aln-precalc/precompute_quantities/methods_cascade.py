# -*- coding: utf-8 -*-
'''
 functions to calculate and plot the quantities required by the LN cascade models, 
 including functions to compute the steady-state and the first order rate response 
 of an exponential/leaky integrate-and-fire neuron subject to white noise input
 (and modulations of the input moments) -- written by Josef Ladenbauer in 2016 
'''

import numpy as np
import scipy.optimize
import numba
import multiprocessing
import time
import tables
from warnings import warn
import matplotlib.pyplot as plt


# COMPUTING FUNCTIONS ---------------------------------------------------------

# prepares data structures and calls computing functions (possibly in parallel)
def calc_EIF_output_and_cascade_quants(mu_vals, sigma_vals, params, 
                                       EIF_output_dict, output_names, save_rate_mod,
                                       LN_quantities_dict, quantity_names):
    
    N_mu_vals = len(mu_vals)    
    N_sigma_vals = len(sigma_vals)
    if N_sigma_vals==1:
        N_procs = 1
    else:
        N_procs = params['N_procs']
        
    # create EIF_output_dict arrays to be filled
    for n in output_names:  
        # complex values dependent on mu, sigma, frequency
        if n in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals, 
                                           len(params['freq_vals']))) + 0j
            
        # complex values dependent on mu, sigma
        elif n in ['peak_real_r1_mumod', 'peak_imag_r1_mumod', 
                   'peak_real_r1_sigmamod', 'peak_imag_r1_sigmamod']:
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals)) + 0j
            
        # real values dependent on mu, sigma
        else: 
            EIF_output_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
    
    # create quantities_dict arrays to be filled
    for n in quantity_names:  
        # real values dependent on mu, sigma
        LN_quantities_dict[n] = np.zeros((N_mu_vals, N_sigma_vals))
                

    arg_tuple_list = [(isig, sigma_vals[isig], mu_vals, params, output_names,
                       quantity_names, save_rate_mod) for isig in range(N_sigma_vals)]    
                      
    comp_total_start = time.time()
                  
    if N_procs <= 1:
        # single process version
        pool = False
        result = (output_and_quantities_given_sigma_wrapper(arg_tuple) 
                  for arg_tuple in arg_tuple_list) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(params['N_procs'])
        result = pool.imap_unordered(output_and_quantities_given_sigma_wrapper, 
                                     arg_tuple_list)
        
    finished = 0 
    for isig, res_given_sigma_dict in result:
        finished += 1
        print(('{count} of {tot} steady-state / rate response and LN quantity ' + \
               'calculations completed').format(count=finished*N_mu_vals, 
                                                tot=N_mu_vals*N_sigma_vals)) 
        for k in res_given_sigma_dict.keys():
            for imu, mu in enumerate(mu_vals):
                if k in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
                    EIF_output_dict[k][imu,isig,:] = res_given_sigma_dict[k][imu,:]
                elif k in output_names and k not in ['r1_mumod', 'r1_sigmamod']:
                    EIF_output_dict[k][imu,isig] = res_given_sigma_dict[k][imu] 
                if k in quantity_names:
                    LN_quantities_dict[k][imu,isig] = res_given_sigma_dict[k][imu]    
    if pool:
        pool.close()
    
    # also include mu_vals, sigma_vals, and freq_vals in output dictionaries
    EIF_output_dict['mu_vals'] = mu_vals
    EIF_output_dict['sigma_vals'] = sigma_vals
    EIF_output_dict['freq_vals'] = params['freq_vals'].copy()

    LN_quantities_dict['mu_vals'] = mu_vals
    LN_quantities_dict['sigma_vals'] = sigma_vals
    LN_quantities_dict['freq_vals'] = params['freq_vals'].copy()
    
    print('Computation of: {} done'.format(output_names))
    print('Total time for computation (N_mu_vals={Nmu}, N_sigma_vals={Nsig}): {rt}s'.
          format(rt=np.round(time.time()-comp_total_start,2), Nmu=N_mu_vals, 
                 Nsig=N_sigma_vals))
      
    return EIF_output_dict, LN_quantities_dict
    

# wrapper function that calls computing functions for a given sigma value and 
# looping over all given mu values (depending on what needs to be computed)
def output_and_quantities_given_sigma_wrapper(arg_tuple):
    isig, sigma, mu_vals, params, output_names, quantity_names, save_rate_mod = arg_tuple
    # a few shortcuts
    V_vec = params['V_vals']
    Vr = params['Vr']
    kr = np.argmin(np.abs(V_vec-Vr))  # reset index value
    VT = params['VT']
    taum = params['taum']
    EL = params['EL']
    DeltaT = params['deltaT']
    
    dV = V_vec[1]-V_vec[0]
    Tref = params['t_ref']
    
    # for damped oscillator fitting
    init_vals = [10.0, 0.01]  # tau (ms), f0=omega/(2*pi) (kHz) [omega used in the paper]
        
    N_mu_vals = len(mu_vals)    
    res_given_sigma_dict = dict()
    
    for n in output_names:  
        # complex values dependent on mu, sigma, frequency
        if n in ['r1_mumod', 'r1_sigmamod'] and save_rate_mod:
            res_given_sigma_dict[n] = np.zeros((N_mu_vals, len(params['freq_vals']))) + 0j
            
        # complex values dependent on mu, sigma
        elif n in ['peak_real_r1_mumod', 'peak_imag_r1_mumod', 
                   'peak_real_r1_sigmamod', 'peak_imag_r1_sigmamod']:
            res_given_sigma_dict[n] = np.zeros(N_mu_vals) + 0j
            
        # real values dependent on mu, sigma
        else: 
            res_given_sigma_dict[n] = np.zeros(N_mu_vals)
     
    for n in quantity_names:    
        if n not in ['r_ss', 'V_mean_ss']:  # omit doubling
            res_given_sigma_dict[n] = np.zeros(N_mu_vals)
            
    
    for imu, mu in enumerate(mu_vals):    
        # first, steady state output & derivatives drdmu, drdsigma
        p_ss, r_ss, q_ss = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma) 
        _, r_ss_dmu, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                          mu+params['d_mu'], sigma)
        _, r_ss_dsig, _ = EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                           mu, sigma+params['d_sigma'])
           
        if 'V_mean_ss' in output_names:
            # disregarding refr. period
            # when in the AdEx model both V and w are clamped during the refr. 
            # period use this Vmean (otherwise see below)
            V_mean = dV*np.sum(V_vec*p_ss)  
            res_given_sigma_dict['V_mean_ss'][imu] = V_mean
        
        r_ss_ref = r_ss/(1+r_ss*Tref)    
        p_ss = r_ss_ref * p_ss/r_ss
        q_ss = r_ss_ref * q_ss/r_ss  # prob. flux needed for sigma-mod calculation              
        r_ss_dmu_ref = r_ss_dmu/(1+r_ss_dmu*Tref) - r_ss_ref
        r_ss_dsig_ref = r_ss_dsig/(1+r_ss_dsig*Tref) - r_ss_ref   
        
        if 'V_mean_sps_ss' in output_names:
            # when considering spike shape (during refr. period) use this Vmean;
            # note that the density reflecting nonrefr. proportion integrates to 
            # r_ss_ref/r_ss 
            Vmean_sps = dV*np.sum(V_vec*p_ss) + (1-r_ss_ref/r_ss)*(params['Vcut']+Vr)/2  
            # note: (1-r_ss_ref/r_ss)==r_ss_ref*Tref 
            res_given_sigma_dict['V_mean_sps_ss'][imu] = Vmean_sps
            
        if 'r_ss' in output_names:
            res_given_sigma_dict['r_ss'][imu] = r_ss_ref
            
        if 'dr_ss_dmu' in output_names:      
            dr_ss_dmu = r_ss_dmu_ref/params['d_mu']
            res_given_sigma_dict['dr_ss_dmu'][imu] = dr_ss_dmu
            
        if 'dr_ss_dsigma' in output_names:        
            dr_ss_dsig = r_ss_dsig_ref/params['d_sigma']
            res_given_sigma_dict['dr_ss_dsigma'][imu] = dr_ss_dsig
        
            
        # next, rate response for mu-modulation and sigma-modulation across the 
        # given modulation frequency range, and (optionally) accurate peaks of 
        # the response modulations |r1|, |real(r1)|, |imag(r1)| determined by 
        # binary search (assuming a uniform spacing in freq_vals) 
        
        if 'r1_mumod' in output_names or 'peak_real_r1_mumod' in output_names \
        or 'peak_imag_r1_mumod' in output_names:
            w_vec = 2*np.pi*params['freq_vals']
            inhom = params['d_mu']*p_ss
            r1mu_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                    Tref, mu, sigma, inhom, w_vec)

            if save_rate_mod:                                        
                res_given_sigma_dict['r1_mumod'][imu,:] = r1mu_vec/params['d_mu']
            
            if 'peak_real_r1_mumod' in output_names:
                abs_re_im = 'real'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_real_r1_mumod'][imu] = peak_val/params['d_mu']
                res_given_sigma_dict['f_peak_real_r1_mumod'][imu] = w_peak/(2*np.pi)
                                              
            if 'peak_imag_r1_mumod' in output_names:                                              
                abs_re_im = 'imag'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec,r1mu_vec,r_ss_dmu_ref,
                                                          V_vec,kr,taum,EL,Vr,VT,DeltaT,
                                                          Tref, mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_imag_r1_mumod'][imu] = peak_val/params['d_mu']
                res_given_sigma_dict['f_peak_imag_r1_mumod'][imu] = w_peak/(2*np.pi)                                          
       
        if 'r1_sigmamod' in output_names or 'peak_real_r1_sigmamod' in output_names \
        or 'peak_imag_r1_sigmamod' in output_names:
            w_vec = 2*np.pi*params['freq_vals']
            if DeltaT>0:
                Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
            else:
                Psi = 0.0*V_vec   
            driftterm = ( mu + ( EL-V_vec+Psi )/taum ) * p_ss
            inhom = params['d_sigma'] * 2/sigma * (q_ss - driftterm)  
            # == -sigma*params['d_sigma']*dp_ssdV
            # Remark: for sigma^2 modulation of amplitude d_sigma2, 
            # inhom = -d_sigma2*dp_ssdV/2
            r1sig_vec = EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, 
                                                     Tref, mu, sigma, inhom, w_vec)
 
            if save_rate_mod:                                         
                res_given_sigma_dict['r1_sigmamod'][imu,:] = r1sig_vec/params['d_sigma']   
            
            if 'peak_real_r1_sigmamod' in output_names:
                abs_re_im = 'real'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec, r1sig_vec,
                                                        r_ss_dsig_ref, V_vec,kr,
                                                        taum,EL,Vr,VT,DeltaT,Tref, 
                                                        mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_real_r1_sigmamod'][imu] = \
                                                        peak_val/params['d_sigma']
                res_given_sigma_dict['f_peak_real_r1_sigmamod'][imu] = w_peak/(2*np.pi)
                                              
            if 'peak_imag_r1_sigmamod' in output_names:                                              
                abs_re_im = 'imag'    
                w_peak, peak_val = EIF_find_lin_response_peak(w_vec, r1sig_vec,
                                                        r_ss_dsig_ref, V_vec,kr,
                                                        taum,EL,Vr,VT,DeltaT,Tref, 
                                                        mu,sigma,inhom,abs_re_im)
                res_given_sigma_dict['peak_imag_r1_sigmamod'][imu] = \
                                                        peak_val/params['d_sigma']
                res_given_sigma_dict['f_peak_imag_r1_sigmamod'][imu] = w_peak/(2*np.pi)
        
        
        # next, the quantities obtained by fitting the filters semi-analytically
        # (in the Fourier domain)
        
        # fitting exponential function A*exp(-t/tau) to normalized rate response 
        # in Fourier domain: A*tau / (1 + 1i*2*pi*f*tau)  f in kHz, tau in ms 
        # with A = 1/tau to guarantee equality at f=0,
        # see the paper before Eq. 85 for more details
        if 'tau_mu_exp' in quantity_names:            
            # use normalized rate response r1 for fitting, normalize such that 
            # its value at f=0 is one (by dividing by r1_mumod_f0, the real value 
            # equal to the time-integral of the filter from 0 to inf)         
            #r1_mumod_f0 = dr_ss_dmu 
            #r1_mumod_normalized = r1mu_vec/params['d_mu'] /r1_mumod_f0
            r1_mumod_normalized = r1mu_vec/r1mu_vec[0]  # this version is more robust 
            # because r1 for f->0 and dr_ss_dmu deviate for some parametrizations
            init_val = 1.0 #ms
            tau = fit_exponential_freqdom(params['freq_vals'], r1_mumod_normalized, 
                                          init_val)
            res_given_sigma_dict['tau_mu_exp'][imu] = tau
            
        # same as above with an additional constraint, see the paper before Eq. 
        # 89 for more details
        if 'tau_sigma_exp' in quantity_names:
            # use normalized rate response r1 for fitting, normalize such that 
            # its value at f=0 is one (by dividing by r1_sigmamod_f0, the real value 
            # equal to the time-integral of the filter from 0 to inf)        
            r1_sigmamod_f0 = dr_ss_dsig
            r1_sigmamod_normalized = r1sig_vec/params['d_sigma'] /r1_sigmamod_f0
            # r1_sigmamod_normalized = r1sig_vec/r1sig_vec[0]  # alternative, which 
            # might be more robust for some parametrizations
            init_val = 0.1 #ms
            if r1_sigmamod_f0>0:  # see the paper around Eq. 88 for an explanation
                tau = fit_exponential_freqdom(params['freq_vals'], 
                                              r1_sigmamod_normalized, init_val)
            else: 
                tau = 0.0
            res_given_sigma_dict['tau_sigma_exp'][imu] = tau
           
        # fitting damped oscillator function (with exponential decay) 
        # B*exp(-t/tau)*cos(2*pi*f0*t) to normalized rate response in Fourier domain:
        # B*tau/2 * ( 1/(1 + 2*pi*1i*tau*(f-f0)) + 1/(1 + 2*pi*1i*tau*(f+f0)) )
        # with B = (1 + (2*pi*f0*tau)^2)/tau to guarantee equality at f=0,
        # see the paper before Eq. 87 for more details
        if 'tau_mu_dosc' in quantity_names:
            r1_mumod_f0 = dr_ss_dmu  # real value equal to the time-integral 
                                     # of the filter from 0 to inf
            # shortcuts for peak values and corresponding frequencies:
            fpeak_real_r1_mumod = res_given_sigma_dict['f_peak_real_r1_mumod'][imu]
            peak_real_r1_mumod = res_given_sigma_dict['peak_real_r1_mumod'][imu]/r1_mumod_f0
            fpeak_imag_r1_mumod = res_given_sigma_dict['f_peak_imag_r1_mumod'][imu]
            peak_imag_r1_mumod = res_given_sigma_dict['peak_imag_r1_mumod'][imu]/r1_mumod_f0

            firstfit = imu==0
            if (firstfit and mu > -0.5) or (mu_vals[1]-mu_vals[0] > 0.05):
                print firstfit, mu
                print mu_vals[1]-mu_vals[0]
                print('WARNING: damped oscillator fitting parameter values may ' + \
                      'not be suitable')
                print('--> check fit_exp_damped_osc_freqdom function')
            tau, f0 = fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_mumod, 
                                                 peak_real_r1_mumod, fpeak_imag_r1_mumod, 
                                                 peak_imag_r1_mumod, firstfit)
            res_given_sigma_dict['tau_mu_dosc'][imu] = tau
            res_given_sigma_dict['f0_mu_dosc'][imu] = f0
            init_vals =[tau, f0]
      
        #print 'calculation for mu =', mu, 'done'  
    #print 'calculations for sigma =', sigma, 'done'  
    return isig, res_given_sigma_dict

    
    
# CORE FUNCTIONS that calculate steady state and 1st order spike rate response 
# to modulations for an EIF/LIF neuron subject to white noise input 

@numba.njit
def EIF_steady_state(V_vec, kr, taum, EL, Vr, VT, DeltaT, mu, sigma):
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    p_ss = np.zeros(n);  q_ss = np.ones(n);
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-EL-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, kr, -1):
        if not F[k]==0.0:
            p_ss[k-1] = p_ss[k] * A[k] + B[k]
        else:
            p_ss[k-1] = p_ss[k] * A[k] + sig2term_dV
        q_ss[k-1] = 1.0    
    for k in xrange(kr, 0, -1):  
        p_ss[k-1] = p_ss[k] * A[k]
        q_ss[k-1] = 0.0
    p_ss_sum = np.sum(p_ss)   
    r_ss = 1.0/(dV*p_ss_sum)
    p_ss *= r_ss;  q_ss *= r_ss;
    return p_ss, r_ss, q_ss

    
@numba.njit
def EIF_lin_rate_response_frange(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                 mu, sigma, inhom, w_vec): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)
    r1_vec = 1j*np.ones(len(w_vec))    
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-EL-Psi )/taum - mu )
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for iw in range(len(w_vec)):
        q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
        fw = dV*1j*w_vec[iw]
        refterm = np.exp(-1j*w_vec[iw]*Tref)  
        for k in xrange(n-1, 0, -1):
            if not k==kr+1:
                q1a_new = q1a + fw*p1a
            else:
                q1a_new = q1a + fw*q1a - refterm
            if not F[k]==0.0:    
                p1a_new = p1a * A[k] + B[k] * q1a
                p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
            else:
                p1a_new = p1a * A[k] + sig2term_dV * q1a
                p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
            q1b += fw*p1b
            q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
        r1_vec[iw] = -q1b/q1a
    return r1_vec   
    
 
def EIF_find_lin_response_peak(w_vec, r1_vec, r1_f0, V_vec, kr, taum, EL, Vr, VT, 
                               DeltaT, Tref, mu, sigma, inhom, abs_re_im):                              
   if abs_re_im == 'abs':
       val = np.max(abs(r1_vec))
       ind = np.argmax(abs(r1_vec))
   elif abs_re_im == 'real':
       val = np.max(abs(np.real(r1_vec)))
       ind = np.argmax(abs(np.real(r1_vec)))
   elif abs_re_im == 'imag':
       val = np.max(abs(np.imag(r1_vec)))
       ind = np.argmax(abs(np.imag(r1_vec)))
   # now, binary search to get peak of r1_vec with higher accuracy 
   # if it is not very close to f=0
   if ind>0: 
      w_peak = w_vec[ind];  dw = w_vec[1]-w_vec[0];
      w_min = 2*np.pi*1e-5
      bounds = [np.max([w_min,w_peak-dw]), w_peak+dw]
      cval = r1_vec[ind]
      while dw>2*np.pi*5e-6:
          r1l = EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                      mu, sigma, inhom, bounds[0])
          if abs_re_im == 'abs':                             
              vall = abs(r1l)
          elif abs_re_im == 'real':
              vall = abs(np.real(r1l))
          elif abs_re_im == 'imag':
              vall = abs(np.imag(r1l))
          if vall>val:
              val = vall
              cval = r1l
              w_peak = bounds[0]
          else:
              r1r = EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                                          mu, sigma, inhom, bounds[1])
              if abs_re_im == 'abs':                             
                  valr = abs(r1r)
              elif abs_re_im == 'real':
                  valr = abs(np.real(r1r))
              elif abs_re_im == 'imag':
                  valr = abs(np.imag(r1r))               
              if valr>val:
                  val = valr
                  cval = r1r
                  w_peak = bounds[1]
          dw /= 2.0
          bounds = [np.max([w_min,w_peak-dw]), w_peak+dw]
      peak_val = cval
   else:
      w_peak = 0.0
      peak_val = r1_f0
   return w_peak, peak_val
  

@numba.njit
def EIF_lin_rate_response(V_vec, kr, taum, EL, Vr, VT, DeltaT, Tref,
                          mu, sigma, inhom, w): 
    dV = V_vec[1]-V_vec[0]
    sig2term = 2.0/sigma**2
    n = len(V_vec)   
    q1a = 1.0 + 0.0*1j;  p1a = 0.0*1j;  q1b = 0.0*1j;  p1b = 0.0*1j;
    fw = dV*1j*w
    refterm = np.exp(-1j*w*Tref)
    if DeltaT>0:
        Psi = DeltaT*np.exp((V_vec-VT)/DeltaT)
    else:
        Psi = 0.0*V_vec
    F = sig2term*( ( V_vec-EL-Psi )/taum - mu ) 
    A = np.exp(dV*F)
    F_dummy = F.copy()
    F_dummy[F_dummy==0.0] = 1.0
    B = (A - 1.0)/F_dummy * sig2term
    sig2term_dV = dV * sig2term
    for k in xrange(n-1, 0, -1):
        if not k==kr+1:
            q1a_new = q1a + fw*p1a
        else:
            q1a_new = q1a + fw*q1a - refterm
        if not F[k]==0.0:    
            p1a_new = p1a * A[k] + B[k] * q1a
            p1b_new = p1b * A[k] + B[k] * (q1b - inhom[k])
        else:
            p1a_new = p1a * A[k] + sig2term_dV * q1a
            p1b_new = p1b * A[k] + sig2term_dV * (q1b - inhom[k])
        q1b += fw*p1b
        q1a = q1a_new;  p1a = p1a_new;  p1b = p1b_new;   
    r1 = -q1b/q1a
    return r1    

# the functions above work efficiently in practice, but the integration schemes 
# might be improved (e.g., based on the Magnus expansion) to allow for larger 
# membrane voltage discretization steps  
 

           
def fit_exponential_freqdom(f, r1_mod_normalized, init_val):
    
    tau_lb = 0.001  #ms, lower bound 
    tau_ub = 100.0  #ms, upper bound
    # first global brute-force optimization on a coarse grid to reduce risk of 
    # finding a local optimum
    tau_step = 1.0  #ms
    tau_vals = np.arange(tau_lb, tau_ub, tau_step)
    errors = np.zeros_like(tau_vals)
    for i, tau in enumerate(tau_vals):
        errors[i] = exp_mean_sq_dist(tau, f, r1_mod_normalized)
    idx = np.argmin(errors)
    # then refinement using a finer grid
    if idx<len(tau_vals)-1:
        tau_ub = tau_vals[idx+1]
    if idx>0:
        tau_lb = tau_vals[idx-1]
    tau_step = 0.01  #ms
    tau_vals = np.arange(tau_lb, tau_ub, tau_step)
    errors = np.zeros_like(tau_vals)
    for i, tau in enumerate(tau_vals):
        errors[i] = exp_mean_sq_dist(tau, f, r1_mod_normalized)
    idx = np.argmin(errors)
    tau = tau_vals[idx]   
    # or, alternatively, use an "off-the-shelf" optimization method  
#    sol = scipy.optimize.minimize_scalar(exp_mean_sq_dist, args=(f, r1_mod_normalized), 
#                                         bounds=(tau_lb, tau_ub), method='bounded', 
#                                         options={'disp':True, 'maxiter':500, 'xatol':1e-3})
#    tau = sol.x                            
    return tau
    


def exp_mean_sq_dist(tau, *args):
    f, r1_mod_normalized = args
    exp_fdom = 1.0 / (1.0 + 1j*2.0*np.pi*f*tau)  # exp. function in Fourier domain
    error = np.sum(np.abs(exp_fdom - r1_mod_normalized)**2)  
    # correct normalization for mean squared error is not important for optim.
    return error

    

def fit_exp_damped_osc_freqdom(init_vals, fpeak_real_r1_mumod, peak_real_r1_mumod, 
                               fpeak_imag_r1_mumod, peak_imag_r1_mumod, firstfit):
    
    # first global brute-force optimization on a coarse grid to reduce risk of 
    # finding a local optimum (narrow ranges used here)
    # Note that some of the values set here might not be optimal for certain 
    # parametrizations of the EIF/LIF model
    if firstfit:  
        tau_vals = np.arange(10.0, 35, 0.025)  #ms
        f0_vals = np.arange(0.0, 0.005, 1e-4)  #kHz
    else:
        tau_vals = np.arange(np.max([0.0, init_vals[0]-5]), init_vals[0]+2, 0.005)  
        f0_vals = np.arange(np.max([0.0, init_vals[1]-0.001]), init_vals[1]+0.02, 5e-5) 

    args = (fpeak_real_r1_mumod, peak_real_r1_mumod, 
            fpeak_imag_r1_mumod, peak_imag_r1_mumod)
    errors = dosc_mean_sq_dist_2f_tauf0grid(tau_vals, f0_vals, args)     
    imin, jmin = np.unravel_index(errors.argmin(), errors.shape)
    tau = tau_vals[imin]
    f0 = f0_vals[jmin]
    
    # then refine using gradient- or simplex-based opimization methods and 
    # previously determined solution as starting point
    init_vals = [tau, f0]
    # when using gradient-based optim. method the discretization parameter values 
    # to compute the derivatives (jacobian etc.) need to be determined first; 
    # instead we here use a simplex-based method
    if f0==0:                     
        sol = scipy.optimize.minimize(dosc_mean_sq_dist_2f, tau, args=args,
                                      method='Nelder-Mead', 
                                      options={'xtol':1e-9, 'ftol':1e-9})
        tau = sol.x
        f0 = 0.0
    else:
        sol = scipy.optimize.minimize(dosc_mean_sq_dist_2f, init_vals, args=args,
                                      method='Nelder-Mead', 
                                      options={'xtol':1e-9, 'ftol':1e-9})
        tau, f0 = sol.x  
    return tau, f0
                                 
             
            
def dosc_mean_sq_dist_2f(p, *args):
    f_val1, r1_val1, f_val2, r1_val2 = args
    if np.size(p)<2:
        p = np.array([p[0], 0.0])
    dosc_at_f1 = eval_dosc_fdom(p, f_val1)
    dosc_at_f2 = eval_dosc_fdom(p, f_val2)
    error = np.abs(dosc_at_f1 - r1_val1)**2 + np.abs(dosc_at_f2 - r1_val2)**2  
    # correct normalization for mean squared error is not important for optim.
    return error



@numba.njit
def dosc_mean_sq_dist_2f_tauf0grid(tau_vals, f0_vals, args):
    f_val1, r1_val1, f_val2, r1_val2 = args
    errors = np.zeros((len(tau_vals),len(f0_vals)))
    for i in range(len(tau_vals)):
        for j in range(len(f0_vals)):
            tau = tau_vals[i];  f0 = f0_vals[j];
            dosc_at_f1 = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
                         ( 1.0 / (1 + 2*np.pi*1j*tau*(f_val1-f0)) + \
                           1.0/ (1 + 2*np.pi*1j*tau*(f_val1+f0)) )
            dosc_at_f2 = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
                         ( 1.0 / (1 + 2*np.pi*1j*tau*(f_val2-f0)) + \
                           1.0/ (1 + 2*np.pi*1j*tau*(f_val2+f0)) )
            errors[i,j] = np.abs(dosc_at_f1 - r1_val1)**2 + \
                          np.abs(dosc_at_f2 - r1_val2)**2
    return errors  



@numba.njit
def eval_dosc_fdom(p, f):
    tau = p[0]  
    f0 = p[1]
    out = (1.0 + (2*np.pi*f0*tau)**2) / 2 * \
          ( 1.0 / (1 + 2*np.pi*1j*tau*(f-f0)) + 1.0/ (1 + 2*np.pi*1j*tau*(f+f0)) )
    return out



# LOAD / SAVE FUNCTIONS --------------------------------------------------------

def load(filepath, input_dict, quantities, param_dict):
    print('')
    print('Loading {} from file {}'.format(quantities, filepath))
    try:
        h5file = tables.open_file(filepath, mode='r')
        root = h5file.root
        
        for q in quantities:
            input_dict[q] = h5file.get_node(root, q).read()            
                   
        # loading parameters
        # only overwrite what is in the file, do not start params from scratch, 
        # otherwise: uncomment following line
        #param_dict = {} 
        for child in root.params._f_walknodes('Array'):
            param_dict[child._v_name] = child.read()[0]
        for group in root.params._f_walk_groups():
            if group != root.params: # walk group first yields the group itself, 
                                     # then its children
                param_dict[group._v_name] = {}
                for subchild in group._f_walknodes('Array'):
                    param_dict[group._v_name][subchild._v_name] = subchild.read()[0]            
        
        h5file.close()
    
    except IOError:
        warn('Could not load quantities from file '+filepath)
    except:
        h5file.close()
        

def save(filepath, output_dict, param_dict):
    print('')
    print('Saving {} to file {}'.format(output_dict.keys(), filepath))
    try:
        h5file = tables.open_file(filepath, mode='w')
        root = h5file.root
            
        for k in output_dict.keys():
            h5file.create_array(root, k, output_dict[k])
            print('created array {}'.format(k))
            
        h5file.create_group(root, 'params', 'Neuron model and numerics parameters')
        for name, value in param_dict.items():
            # for python2/3 compat.:
            # if isinstance(name, str):
            # name = name.encode(encoding='ascii') # do not allow unicode keys
            if isinstance(value, (int, float, bool)):
                h5file.create_array(root.params, name, [value])
            elif isinstance(value, str):
                h5file.create_array(root.params, name, [value.encode(encoding='ascii')])
            elif isinstance(value, dict):
                params_sub = h5file.create_group(root.params, name)
                for nn, vv in value.items():
                    # for python2/3 compat.:
                    # if isinstance(nn, str):
                    # nn = nn.encode(encoding='ascii') # do not allow unicode keys
                    if isinstance(vv, str):
                        h5file.create_array(params_sub, nn, [vv.encode(encoding='ascii')])
                    else:
                        h5file.create_array(params_sub, nn, [vv])
        h5file.close()
    
    except IOError:
        warn('Could not write quantities into file {}'.format(filepath))
    except:
        h5file.close()



# PLOTTING FUNCTIONS -----------------------------------------------------------

def plot_quantities_forpaper(quantities_dict, quantity_names, sigmas_quant_plot, 
                             mus_plot, sigmas_plot):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']   
    
    plt.figure()
    plt.suptitle('LN quantities')
    
    mu_lim = [-1, 4]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_quant_plot]
    mu_plot_ind = np.argmin(np.abs(mu_vals-mus_plot[0]))
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j+1)/(N_sigma), 0]
        linecolor = rgb
        
        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(1, 4, 1)
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(
                            sigma_vals[j])
                           
            else:
                siglabel = None
    
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_mu_exp'][mu_plot_ind,j],
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title(r'$r_{\infty}$', fontsize=14)
                plt.ylabel('[kHz]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            
                               
        if 'V_mean_ss' in quantity_names:
            plt.subplot(1, 4, 2, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['V_mean_ss'][inds_mu_plot,j],
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_sigma_exp'][mu_plot_ind,j],
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title('$\langle V \\rangle_{\infty}$', fontsize=14)
                plt.ylabel('[mV]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
            if k_j==N_sigma-1:
                plt.legend(loc='best')
        
        if 'tau_mu_dosc' in quantity_names:
            plt.subplot(1, 4, 3, sharex=ax1)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['tau_mu_dosc'][mu_plot_ind,j], 
                    'o', color=linecolor) #, markeredgewidth=0.0)
            if k_j==0:
                plt.title(r'$\tau$', fontsize=14)
                plt.ylabel('[ms]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                plt.ylim([-1, 46])
        
        if 'f0_mu_dosc' in quantity_names:
            plt.subplot(1, 4, 4, sharex=ax1)                       
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            for l in range(len(sigmas_plot)): 
                if np.round(sigmas_plot[l],2)==np.round(sigma_vals[j],2):
                    plt.plot(mus_plot[0], quantities_dict['f0_mu_dosc'][mu_plot_ind,j], 
                    'o', color=linecolor) #, markeredgewidth=0.0)  #markeredgecolor=...,
            if k_j==0:
                plt.title(r'$\omega/2\pi$', fontsize=14)
                plt.ylabel('[kHz]', fontsize=12)
                plt.xlabel('$\mu$ [mV/ms]', fontsize=12)
                plt.ylim([-0.005, 0.12])
    plt.show()            



def plot_quantities(quantities_dict, quantity_names, sigmas_plot):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']   
    
    plt.figure()
    plt.suptitle('LN quantities')
    
    mu_lim = [-1, 5]
    inds_mu_plot = [i for i in range(len(mu_vals)) if mu_lim[0] <= mu_vals[i] <= mu_lim[1]]
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_plot]
    N_sigma = len(inds_sigma_plot)    
    
    for k_j, j in enumerate(inds_sigma_plot):
        # color    
        rgb = [0, float(k_j+1)/(N_sigma), 0]
        linecolor = rgb
        
        if 'r_ss' in quantity_names:
            ax1 = plt.subplot(3, 2, 1)
            # labels
            if k_j in [0, N_sigma//2, N_sigma-1]:
                siglabel = '$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(
                            sigma_vals[j])
                           
            else:
                siglabel = None
    
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['r_ss'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
    
            if k_j==0:
                plt.ylabel('$r_{\infty}$ [kHz]')
            if k_j==N_sigma-1:
                plt.legend(loc='best')
        
        if 'V_mean_ss' in quantity_names:
            plt.subplot(3, 2, 2, sharex=ax1)              
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['V_mean_ss'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel('$\langle V \\rangle_{\infty}$ [mV]')
            
        if 'tau_mu_exp' in quantity_names:
            plt.subplot(3, 2, 3, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_exp'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_{\mu}$ [ms]')
            
        if 'tau_sigma_exp' in quantity_names:
            plt.subplot(3, 2, 4, sharex=ax1)
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_sigma_exp'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau_{\sigma}$[ms]')
        
        if 'tau_mu_dosc' in quantity_names:
            plt.subplot(3, 2, 5, sharex=ax1)           
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\tau$ [ms]')
                plt.xlabel('$\mu$ [mV/ms]')
        
        if 'f0_mu_dosc' in quantity_names:
            plt.subplot(3, 2, 6, sharex=ax1)                       
            plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_dosc'][inds_mu_plot,j], 
                     label=siglabel, color=linecolor)
            if k_j==0:
                plt.ylabel(r'$\omega/2\pi$ [kHz]')
                plt.xlabel('$\mu$ [mV/ms]')
                
    
    if 'tau1_mu_bedosc' in quantity_names:  # not included here
        plt.figure()
        plt.suptitle('LN quantities for bexdox variant')
        for k_j, j in enumerate(inds_sigma_plot):
            # color    
            rgb = [0, float(k_j+1)/(N_sigma), 0]
            linecolor = rgb
            if 'B_mu_bedosc' in quantity_names:
                ax2 = plt.subplot(2, 2, 2)              
                plt.plot(mu_vals[inds_mu_plot], quantities_dict['B_mu_bedosc'][inds_mu_plot,j], 
                         label=siglabel, color=linecolor)
                if k_j==0:
                    plt.ylabel('$B$ [...]')
                
            if 'tau1_mu_bedosc' in quantity_names:
                plt.subplot(2, 2, 1, sharex=ax2)
                plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau1_mu_bedosc'][inds_mu_plot,j], 
                         label=siglabel, color=linecolor)
                if k_j==0:
                    plt.ylabel(r'$\tau_1$ [ms]')
                
            if 'tau2_mu_bedosc' in quantity_names:
                plt.subplot(2, 2, 3, sharex=ax2)
                plt.plot(mu_vals[inds_mu_plot], quantities_dict['tau2_mu_bedosc'][inds_mu_plot,j], 
                         label=siglabel, color=linecolor)
                if k_j==0:
                    plt.ylabel(r'$\tau_2$ [ms]')
                    plt.xlabel('$\mu$ [mV/ms]')
            
            if 'f0_mu_bedosc' in quantity_names:
                plt.subplot(2, 2, 4, sharex=ax2)           
                plt.plot(mu_vals[inds_mu_plot], quantities_dict['f0_mu_bedosc'][inds_mu_plot,j], 
                         label=siglabel, color=linecolor)
                if k_j==0:
                    plt.ylabel(r'$f_0$ [kHz]')
                    plt.xlabel('$\mu$ [mV/ms]')
    plt.show()
            
        
    
def plot_filters(output_dict, quantities_dict, output_names, params, 
                 mus_plot, sigmas_plot, recalc_filters):
    
    mu_vals = quantities_dict['mu_vals']
    sigma_vals = quantities_dict['sigma_vals']
    f_vals = quantities_dict['freq_vals']
    
    if recalc_filters:
       f_vals = np.arange(0.25, 3000.1, 0.25)/1000
       params_tmp = params.copy()
       params_tmp['freq_vals'] = f_vals
       params_tmp['N_procs'] = 1
       mumod_tmp_dict = {};  sigmamod_tmp_dict = {}
       
    df = f_vals[1] - f_vals[0]
    if any(f_vals<0):
        n = len(f_vals)+1
    else: 
        n = 2*len(f_vals)+1
    dt = 1.0/(df*n)  #ms
    t_vals = np.arange(0,n)*dt    
    tmax = 50
    # note that dt for the filter is 1/(df*(2*len(f_vals)+1)), 
    # factor 2 for appended negative frequencies, +1 for appended zero freq.,
    # both not calculated explicitly because rate response r1 for pos. and neg. 
    # frequencies are complex conjugates, and r1 for f=0 is obtained from steady state
    
    inds_mu_plot = [np.argmin(np.abs(mu_vals-mus)) for mus in mus_plot]  
    inds_sigma_plot = [np.argmin(np.abs(sigma_vals-sig)) for sig in sigmas_plot]    
    N_mu = len(inds_mu_plot)    
    N_sigma = len(inds_sigma_plot)
    
    if 'r1_mumod' in output_names:
        plt.figure()
        plt.suptitle('EIF rate response to $\mu$-modulation in in 1/V')
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor (see Eq. 79 in the paper)
                drdmu = output_dict['dr_ss_dmu'][imu,isig]
                
                # analytical exponential fit (using asymptotics)
                if params['deltaT']>0:  #otherwise it makes no sense
                    tau = params['deltaT']*drdmu/output_dict['r_ss'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'b--')
                    
                # semi-analytical exponential fit
                if 'tau_mu_exp' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_exp'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_mu_dosc' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_dosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_dosc'][imu,isig]
                    A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                    dosc_fit = drdmu * A*tau/2 * \
                               ( 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals-f0)) + \
                                 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals+f0)) )
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(dosc_fit[f_vals<=1]), 'm')
                                                
                # semi-analytical bi-exp, damped oscillator fit (not included here)
                if 'tau1_mu_bedosc' in quantities_dict.keys():
                    tau1 = quantities_dict['tau1_mu_bedosc'][imu,isig]
                    tau2 = quantities_dict['tau2_mu_bedosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_bedosc'][imu,isig]
                    B = quantities_dict['B_mu_bedosc'][imu,isig]
                    A = 1.0/tau1 - B*tau2/((1.0 + (2*np.pi*f0*tau2)**2)*tau1)
                    bexdos_fit = drdmu * ( A*tau1/(1.0 + 2*np.pi*1j*f_vals*tau1) + \
                                 B*tau2/2 * ( 1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals-f0)) + \
                                              1.0/(1.0 + 2*np.pi*1j*tau2*(f_vals+f0)) ) )
                    plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(bexdos_fit[f_vals<=1]), 'g')

                # rate response
                if recalc_filters:
                    tmp_dict = {};  tmp_output_name = ['r1_mumod']; save_rmod = True
                    print('')
                    print('(Re)computing {} at higher resolution for plot'.format(
                            tmp_output_name))
                    tmp_dict, _ = calc_EIF_output_and_cascade_quants([mu_vals[imu]],
                                                                     [sigma_vals[isig]],
                                                                     params_tmp, tmp_dict,
                                                                     tmp_output_name, 
                                                                     save_rmod, {}, [])
                    r1_mumod = tmp_dict['r1_mumod'][0,0,:]
                    mumod_tmp_dict[imu,isig] = r1_mumod
                else:                               
                    r1_mumod = output_dict['r1_mumod'][imu,isig,:]
                plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(r1_mumod[f_vals<=1]), 
                           'k', linewidth=1.5)              
                plt_min = np.min(1000*np.abs(r1_mumod[f_vals<=1]))
                plt_max = np.max(1000*np.abs(r1_mumod[f_vals<=1]))
                plt.ylim([plt_min, plt_max])
                plt.xlim([1000*f_vals[0], 1000])
                if k_imu==0:
                    plt.ylabel('$\sigma={0:.3}$ [mV/$\sqrt{{\mathrm{{ms}}}}$]'.format(
                                sigma_vals[isig]))
                if k_isig==len(inds_sigma_plot)-1:    
                    plt.xlabel('$\mu={0:.3}$ [mV/ms]'.format(mu_vals[imu]))
                
    if 'r1_sigmamod' in output_names:
        plt.figure()
        plt.suptitle('EIF rate response to $\sigma$-modulation in 1/(V*sqrt(ms))')
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor (see Eq. 80 in the paper)
                drdsigma = output_dict['dr_ss_dsigma'][imu,isig]
                   
                # semi-analytical exponential fit
                if 'tau_sigma_exp' in quantities_dict.keys():
                    if drdsigma>0:  # see the paper before Eq. 89
                        tau = quantities_dict['tau_sigma_exp'][imu,isig]
                        A = 1.0/tau
                        exp_fit = drdsigma * A*tau/(1.0 + 2*np.pi*1j*f_vals*tau)
                        plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(exp_fit[f_vals<=1]), 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_sigma_dosc' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use a "delta" filter
                        tau = quantities_dict['tau_sigma_dosc'][imu,isig]
                        f0 = quantities_dict['f0_sigma_dosc'][imu,isig]
                        A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                        dosc_fit = drdsigma * A*tau/2 * \
                                   ( 1.0/(1.0 + 2*np.pi*1j*tau*(f_vals-f0)) + \
                                     1.0/(1.0 + 2*np.pi*1j*tau*(f_vals+f0)) )
                        plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(dosc_fit[f_vals<=1]), 'm')

                # rate response
                if recalc_filters:
                    tmp_dict = {};  tmp_output_name = ['r1_sigmamod'];  save_rmod = True
                    print('')
                    print('(Re)computing {} at higher resolution for plot'.format(
                            tmp_output_name))
                    tmp_dict, _ = calc_EIF_output_and_cascade_quants([mu_vals[imu]],
                                                                     [sigma_vals[isig]],
                                                                     params_tmp, tmp_dict,
                                                                     tmp_output_name, 
                                                                     save_rmod, {}, [])
                    r1_sigmamod = tmp_dict['r1_sigmamod'][0,0,:] 
                    sigmamod_tmp_dict[imu,isig] = r1_sigmamod
                else:                               
                    r1_sigmamod = output_dict['r1_sigmamod'][imu,isig,:]              
                plt.loglog(1000*f_vals[f_vals<=1], 1000*np.abs(r1_sigmamod[f_vals<=1]), 'k')
                plt_min = np.min(1000*np.abs(r1_sigmamod[f_vals<=1]))
                plt_max = np.max(1000*np.abs(r1_sigmamod[f_vals<=1]))
                plt.ylim([plt_min, plt_max])
                plt.xlim([1000*f_vals[0], 1000])

    if 'ifft_r1_mumod' in output_names:
        plt.figure()
        plt.suptitle('linear filter for $\mu$ (unnormalized) in kHz/V')
        inds = t_vals<=tmax
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor (see Eq. 79 in the paper)
                drdmu = output_dict['dr_ss_dmu'][imu,isig]
                
                # analytical exponential fit (using asymptotics)
                if params['deltaT']>0:  #otherwise it makes no sense
                    tau = params['deltaT']*drdmu/output_dict['r_ss'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*np.exp(-t_vals/tau)
                    plt.plot(t_vals[inds], 1000*exp_fit[inds], 'b--')
                   
                # semi-analytical exponential fit
                if 'tau_mu_exp' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_exp'][imu,isig]
                    A = 1.0/tau
                    exp_fit = drdmu * A*np.exp(-t_vals/tau)
                    plt.plot(t_vals[inds], 1000*exp_fit[inds], 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_mu_dosc' in quantities_dict.keys():
                    tau = quantities_dict['tau_mu_dosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_dosc'][imu,isig]
                    A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                    dosc_fit = drdmu * A*np.exp(-t_vals/tau)*np.cos(2*np.pi*f0*t_vals)
                    plt.plot(t_vals[inds], 1000*dosc_fit[inds], 'm')
                
                # semi-analytical bi-exp, damped oscillator fit (not included here)
                if 'tau1_mu_bedosc' in quantities_dict.keys():
                    tau1 = quantities_dict['tau1_mu_bedosc'][imu,isig]
                    tau2 = quantities_dict['tau2_mu_bedosc'][imu,isig]
                    f0 = quantities_dict['f0_mu_bedosc'][imu,isig]
                    B = quantities_dict['B_mu_bedosc'][imu,isig]
                    A = 1.0/tau1 - B*tau2/((1.0 + (2*np.pi*f0*tau2)**2)*tau1)
                    bexdos_fit = drdmu * ( A*np.exp(-t_vals/tau1) + \
                                 B*np.exp(-t_vals/tau2)*np.cos(2*np.pi*f0*t_vals) )
                    plt.plot(t_vals[inds], 1000*bexdos_fit[inds], 'g')

                # ifft of rate response
                if recalc_filters:
                    r1_mumod = mumod_tmp_dict[imu,isig]
                else:
                    r1_mumod = output_dict['r1_mumod'][imu,isig,:]  #1/mV
                # linear filter (ifft of rate response)
                # get mumod rate response in the right shape so that ifft can be applied;
                # for info on the reshaping above, see numpy documentation on ifft 
                if any(f_vals<0):
                    r1_reshaped = np.concatenate([np.array([drdmu]), r1_mumod])
                else:
                    r1_reshaped = np.concatenate([np.array([drdmu]), r1_mumod, 
                                                  np.flipud(np.conj(r1_mumod))])
                mu_filter_unnorm = np.fft.ifft(r1_reshaped)/dt  #kHz/mV
                # minor correction to avoid plotting errors (vanishingly small 
                # neg. values and imag. parts might occur for numerical reasons)
                mu_filter_unnorm = np.real(mu_filter_unnorm)
                plt.plot(t_vals[inds], 1000*mu_filter_unnorm[inds], 'k')     
                #plt_min = np.min(mu_filter_unnorm[inds])
                plt_max = np.max(1000*mu_filter_unnorm[inds])
                plt.ylim([-0.12*plt_max, 1.2*plt_max]) 
                plt.xticks([0, tmax])        

      
    if 'ifft_r1_sigmamod' in output_names:
        plt.figure()
        plt.suptitle('linear filter for $\sigma$ (unnormalized) in kHz/(V*sqrt(ms))')
        inds = t_vals<=tmax
        for k_isig, isig in enumerate(inds_sigma_plot):        
            for k_imu, imu in enumerate(inds_mu_plot):           
                plt.subplot(N_sigma,N_mu,k_isig*N_mu+k_imu+1)                
                
                # scaling factor (see Eq. 80 in the paper)
                drdsigma = output_dict['dr_ss_dsigma'][imu,isig]
                
                # semi-analytical exponential fit
                if 'tau_sigma_exp' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use a "delta" filter   
                        tau = quantities_dict['tau_sigma_exp'][imu,isig]
                        A = 1.0/tau
                        exp_fit = drdsigma * A*np.exp(-t_vals/tau)
                        plt.plot(t_vals[inds], 1000*exp_fit[inds], 'c')                
                
                # semi-analytical damped oscillator fit
                if 'tau_sigma_dosc' in quantities_dict.keys():
                    if drdsigma>0:  # otherwise we use delta filter   
                        tau = quantities_dict['tau_sigma_dosc'][imu,isig]
                        f0 = quantities_dict['f0_sigma_dosc'][imu,isig]
                        A = (1.0 + (2*np.pi*f0*tau)**2)/tau
                        dosc_fit = drdsigma * A*np.exp(-t_vals/tau)*np.cos(2*np.pi*f0*t_vals)
                        plt.plot(t_vals[inds], 1000*dosc_fit[inds], 'm')

                # ifft of rate response
                if recalc_filters:
                    r1_sigmamod = sigmamod_tmp_dict[imu,isig]
                else:
                    r1_sigmamod = output_dict['r1_sigmamod'][imu,isig,:]  #1/mV
                # linear filter (ifft of rate response)
                # get sigmamod rate response in the right shape so that ifft can be applied;
                # for info on the reshaping above, see numpy documentation on ifft 
                if any(f_vals<0):
                    r1_reshaped = np.concatenate([np.array([drdsigma]), r1_sigmamod])
                else:
                    r1_reshaped = np.concatenate([np.array([drdsigma]), r1_sigmamod, 
                                                  np.flipud(np.conj(r1_sigmamod))])
                sigma_filter_unnorm = np.fft.ifft(r1_reshaped)/dt  #kHz/mV
                # minor correction to avoid plotting errors (vanishingly small 
                # neg. values and imag. parts might occur for numerical reasons)
                sigma_filter_unnorm = np.real(sigma_filter_unnorm)
                plt.plot(t_vals[inds], 1000*sigma_filter_unnorm[inds], 'k') 
                plt_max = np.max(1000*sigma_filter_unnorm[inds])
                plt.ylim([-0.16*plt_max, 1.2*plt_max]) 
                plt.xticks([0, tmax])
    plt.show()    