#!/usr/bin/env python3

# Author: Minyan Zhong

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
import pickle

from matplotlib import cm

import seaborn as sns

import datetime
from datetime import date

import multiprocessing

import time

from scipy import linalg

from fourdvel import fourdvel
from display import display

from basics import basics

class simulation(basics):

    def flow_model(self):

        # Constants for ice flow model.
        A = 2.4e-24
        alpha = 0.04
        g = 9.81
        h = 1000
        n_g = 3
        rho = 900
        s_v = 0.6

        # Gravitional driving stress.
        tau_d = rho*g*h*alpha

        # Basal drag.
        tau_b = 0.8 * tau_d

        w = 50*1000 #m
        L = 150*1000 #m

        # End of constants.

        # Beginning of simulations.
        x = np.linspace(0,L,num=np.round(L/500))
        y = np.linspace(0,2*w,num=np.round(2*w/500))

        yy,xx = np.meshgrid(y,x)
        #print(xx.shape)
        #print(tau_d)
        #print(tau_d * w/h)
        #print(tau_d * w/h * 0.2)

        #print(2*A*w/(n_g+1))

        v_ideal_center = 2*A*w/(n_g+1) * (tau_d * w / h * 0.2)**n_g
        print(v_ideal_center)

        v_ideal = v_ideal_center * (1 - (1-yy/w)**(n_g+1))

        k_h=10**(-1*np.abs(np.log10(L)-0.8))
        gamma = (1 + np.tanh(k_h * (x-0.6*L)))/2

        Gamma = {}

        Gamma_const = v_ideal/v_ideal_center


        # Add all tide signals together.
        p_e = np.zeros(shape=t_axis.shape)
        p_n = np.zeros(shape=t_axis.shape)
        p_u = np.zeros(shape=t_axis.shape)

        for key in syn_tidesRut:
            p_e = p_e + sim[(key,'e')]
            p_n = p_n + sim[(key,'n')]
            p_u = p_u + sim[(key,'u')]

        # Full velocity signals.
        # Constant velocity + tidal signals.
        # Consider the transition from ice stream to ice shelf.
        v_e = np.zeros(shape=t_axis.shape)

        v_n = s_v * x_loc/L * (-v_ideal[ind_x,ind_y]*t_axis + p_n)
        v_n_tides = s_v * x_loc/L * p_n

        v_u = s_v * (x_loc-L)/(10*L) * v_ideal[ind_x,ind_y] + p_u
        v_u_tides = p_u

        # Plotting.
        fig = plt.figure(2,figsize=(7,7))
        ax = fig.add_subplot(111)
        
        # Choice 1.
        #p1 = ax.imshow(v_ideal/v_ideal_center,cmap=plt.cm.coolwarm)
        #p1 = ax.imshow(v_ideal,cmap=plt.cm.coolwarm)
        #fig.colorbar(p1)

        # Choice 2. Summed tidal signals.
        ax.plot(t_axis, v_u)

        # velocity
        #v_e = np.zeros(shape=xx.shape)
        #v_n = np.z

        plt.show()

        return

    def __init__(self):

        super(simulation, self).__init__()

        tide_periods = self.tide_periods

        # Rutford data.
        self.tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']
        self.tidesRut_params = {}
        tidesRut_params = self.tidesRut_params

        # Displacement.        
        #tidesRut['K2'] =    [0.31,  163,    29.1,   99]
        #tidesRut['S2'] =    [0.363, 184,    101.6,  115]
        #tidesRut['M2'] =    [0.259, 177,    156.3,  70]
        #tidesRut['K1'] =    [0.19,  79,     49,     73]
        #tidesRut['P1'] =    [0.24,  77.0,   16.6,   64]
        #tidesRut['O1'] =    [0.264, 81.0,   43,     54]
        #tidesRut['Mf'] =    [2.54,  250.0,  2.9,    163]
        #tidesRut['Msf'] =   [13.28, 18.8,   0.3,    164]
        #tidesRut['Mm'] =    [5.04,  253.0,  1.6,    63]
        #tidesRut['Ssa'] =   [26.74, 256.0,  1.5,    179]
        #tidesRut['Sa'] =    [19.18, 273.0,  0.2,    179]

        # Velocity
        # First two columns are horizontal responses. (amplitude cm/d)
        # Last two columns are vertical forcings. (amplitude cm)
        
        tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']

        # Original data from paper.
        # Horizontal: assume this corresponds to 1 meter /day.
        self.ref_speed = 1
        # Vertical: assume the vertical scale.
        self.verti_scale = 0.5

        # Actual parameters from Murray (2007)
        tidesRut_params['K2'] =    [3.91,   163,    29.1,   99  ]
        tidesRut_params['S2'] =    [4.56,   184,    101.6,  115 ]
        tidesRut_params['M2'] =    [3.15,   177,    156.3,  70  ]
        tidesRut_params['K1'] =    [1.22,   79,     49,     73  ]
        tidesRut_params['P1'] =    [1.48,   77.0,   16.6,   64  ]
        tidesRut_params['O1'] =    [1.54,   81.0,   43.1,   54  ]
        tidesRut_params['Mf'] =    [1.17,   250.0,  2.9,    163 ]
        tidesRut_params['Msf'] =   [5.65,   18.8,   0.3,    164 ]
        tidesRut_params['Mm'] =    [1.15,   253.0,  1.6,    63  ]
        tidesRut_params['Ssa'] =   [0.92,   256.0,  1.5,    179 ]
        tidesRut_params['Sa'] =    [0.33,   273.0,  0.2,    179 ]

        # Find angular frequency
        # Convert displacement to velocity.
        for tide_name in tidesRut_params.keys():
            omega = 2*np.pi / tide_periods[tide_name] 
            tidesRut_params[tide_name][2] = tidesRut_params[tide_name][2] * omega
            tidesRut_params[tide_name][3] = tidesRut_params[tide_name][3] + 270

        print(tidesRut_params)

        # Included constituents in synthetic data.
        #self.syn_tidesRut = ['M2','O1','Msf']
        #self.syn_tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf']
        self.syn_tidesRut = ['K2','S2','M2','K1','P1','O1','Msf','Mf','Mm','Ssa','Sa']
        #self.syn_tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']

    def true_tide_vec(self, secular_v, modeling_tides, tide_amp, tide_phase):

        # Model parameters.
        n_modeling_tides = len(modeling_tides)
        syn_tidesRut = self.syn_tidesRut
        tidesRut_params = self.tidesRut_params


        num_params = 3 + n_modeling_tides*6
        param_vec = np.zeros(shape=(num_params,1))

        param_vec[0:3,0] = secular_v

        # M2, O1, Msf, etc
        for k in range(n_modeling_tides):

            tide_name = modeling_tides[k]

            amp_E = tide_amp[(tide_name,'e')]
            phi_E = tide_phase[(tide_name,'e')]


            amp_N = tide_amp[(tide_name,'n')]
            phi_N = tide_phase[(tide_name,'n')]

            amp_U = tide_amp[(tide_name,'u')]
            phi_U = tide_phase[(tide_name,'u')]

            # Amplitudes.
            amp_E = amp_E
            amp_N = amp_N
            amp_U = amp_U

            # Phases.
            phi_E = self.wrapped(phi_E)
            phi_N = self.wrapped(phi_N)
            phi_U = self.wrapped(phi_U)

            # Put them into the vector.
            param_vec[3 + k*6 : 3 + (k+1)*6] = np.expand_dims(np.asarray([amp_E, amp_N, amp_U, phi_E, phi_N, phi_U]), axis=1)

        return param_vec


    def syn_velocity(self, velo_model):
        
        # Tides.
        tide_periods = self.tide_periods

        print(velo_model)

        #print(stop)

        # Reference speed to tide amplitudes.
        ref_speed = self.ref_speed
        verti_scale = self.verti_scale

        # Rutford tide model.
        tidesRut_params = self.tidesRut_params
        syn_tidesRut = self.syn_tidesRut

        # Secular velcity.
        secular_v_e = velo_model[0]
        secular_v_n = velo_model[1]
        secular_v_u = 0

        verti_ratio = velo_model[2]

        print(secular_v_e, secular_v_n)
        secular_speed = np.sqrt(secular_v_e**2 + secular_v_n**2)
        secular_v = (secular_v_e, secular_v_n, secular_v_u)

        # Tides
        tide_amp = {}
        tide_phase = {}

        # Tides. (Find the parameters)
        for tide_name in syn_tidesRut:
            print(tide_name)

            omega = 2*np.pi / tide_periods[tide_name] 

            horiz_amp = tidesRut_params[tide_name][0]/100 # velocity
            horiz_phase = np.deg2rad(tidesRut_params[tide_name][1])

            verti_amp = tidesRut_params[tide_name][2]/100 # velocity
            verti_phase = np.deg2rad(tidesRut_params[tide_name][3])

            # East component.
            a_e = horiz_amp * abs(secular_v_e/secular_speed) * secular_speed / ref_speed
            phi_e = np.sign(secular_v_e) * horiz_phase

            # North component.
            a_n = horiz_amp * abs(secular_v_n/secular_speed) * secular_speed / ref_speed
            phi_n = np.sign(secular_v_n) * horiz_phase # rad

            # Up component. 
            a_u = verti_amp * verti_scale * verti_ratio
            phi_u = verti_phase
            
            #(Depending on the flow is horizontal nor not)
            #if horizontal == False:
            #    a_u = verti_amp # m/d
            #    phi_u = verti_phase # rad
            #else:
            #    a_u = 0
            #    phi_u = 0

            # Record the tides.
            tide_amp[(tide_name,'e')] = a_e
            tide_amp[(tide_name,'n')] = a_n
            tide_amp[(tide_name,'u')] = a_u

            tide_phase[(tide_name,'e')] = phi_e
            tide_phase[(tide_name,'n')] = phi_n
            tide_phase[(tide_name,'u')] = phi_u

        method = 'analytical'
        if method == 'numerical' or method == 'both':
            sim = {}
            for tide_name in syn_tidesRut:
            
                t_axis = np.arange(-600,600,0.0005) # unit: day
     
                omega = 2*np.pi / tide_periods[tide_name] 
    
                # East component.
                a_e = tide_amp[(tide_name,'e')]
                phi_e = tide_phase[(tide_name,'e')]
                sim[(tide_name,'e')] =  a_e * (np.sin(omega*t_axis + phi_e))
    
                # North component.
                a_n = tide_amp[(tide_name,'n')]
                phi_n = tide_phase[(tide_name,'n')]
                sim[(tide_name,'n')] = a_n * (np.sin(omega*t_axis + phi_n))
    
                # Up component.
                a_u = tide_amp[(tide_name,'u')]
                phi_u = tide_phase[(tide_name,'u')]
                sim[(tide_name,'u')] = a_u * (np.sin(omega*t_axis + phi_u))
    

            # Find time series ve, vn, vu by addition.
            fig = plt.figure(1,figsize=(16,8))
            ax = fig.add_subplot(211)
    
            p_e = np.zeros(shape=t_axis.shape)
            p_n = np.zeros(shape=t_axis.shape)
            p_u = np.zeros(shape=t_axis.shape)
            for tide_name in syn_tidesRut:
                p_e = p_e + sim[(tide_name,'e')]
                p_n = p_n + sim[(tide_name,'n')]
                p_u = p_u + sim[(tide_name,'u')]
    
                #ax.plot(t_axis,sim[(tide_name,'n')],label=tide_name + '_N')
                ax.plot(t_axis,sim[(tide_name,'u')],label=tide_name + '_U')
    
            ax.legend()
    
            # Add secular velocity.
            v_e = secular_v_e + p_e
            v_n = secular_v_n + p_n
            v_u = secular_v_u + p_u
    
            v = (v_e, v_n, v_u)
    
            # Plot the velocity time series.
            #ax = fig.add_subplot(212)
            #ax.plot(t_axis, v_e)
            #ax.plot(t_axis, v_n)
            #ax.plot(t_axis, v_u)
            #ax.plot(t_axis,sim[('Msf','n')])
            #fig.savefig('1.png',format='png')
        else:
            t_axis = None
            v = None
       
        # Return synthetic velocity time series.
        return (t_axis, secular_v, v, tide_amp, tide_phase)


    def syn_offset_data(self, t_axis, secular_v, v=None, modeling_tides=None, tide_amp=None, tide_phase=None, offsetfields=None, noise_sigma = 0):

        # Obtain offsets from synthetics.
        n_offsets = len(offsetfields)
        print("number of offsetfields:", n_offsets)
        n_rows = n_offsets * 2
        data_vector = np.zeros(shape=(n_rows,1))
        t_origin = self.t_origin.date()

        # Three components.
        if t_axis is not None and v is not None:
            v_e, v_n, v_u = v
            method = 'numerical'
        else:
            method = 'analytical'

        for i in range(n_offsets):

            print('offsetfield: ',i,'/',n_offsets,'\n')

            #print(offsetfields[i])
            vecs = [offsetfields[i][2],offsetfields[i][3]]
            
            t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]

            if method == 'numerical':

                ## Numerical way.
                # Cut the time series.
                if t_a >= t_axis[0] and t_b <= t_axis[-1]:
                    inds = np.argmin(np.abs(t_axis-t_a))
                    inde = np.argmin(np.abs(t_axis-t_b))
                else:
                    raise Exception('Time axis too short!')

                t_interval = t_axis[inds:inde]
                v_e_interval = v_e[inds:inde]
                v_n_interval = v_n[inds:inde]
                v_u_interval = v_u[inds:inde]

                # Plot the cut time series. 
                #if i==1:
                #    fig = plt.figure(1, figsize=(10,6))
                #    plt.clf()
                #    ax = fig.add_subplot(111)
                #    ax.plot(t_interval,v_u_interval)
                #    fig.savefig('interval.png',format='png')

                # Integration of velocity wrt time. 
                offset_e = np.trapz(v_e_interval, t_interval)
                offset_n = np.trapz(v_n_interval, t_interval)
                offset_u = np.trapz(v_u_interval, t_interval)

                # Velocity vector.
                offset_vec_1 = np.zeros(shape=(3,1))
                offset_vec_1[:,0] = [offset_e,offset_n,offset_u]

                print(offset_vec_1)

            elif method == 'analytical':

                offset={}

                offset['e'] = 0
                offset['n'] = 0
                offset['u'] = 0

                n_modelng_tides = len(modeling_tides) # Three components.
                comps = ['e','n','u']

                ii = 0
                for comp in comps:
                    offset[comp] = offset[comp] + secular_v[ii] * (t_b - t_a)
                    ii = ii + 1
                    for tide_name in modeling_tides:
                        omega = 2*np.pi / self.tide_periods[tide_name] 
                        dis_amp = tide_amp[(tide_name,comp)] / omega
                        dis_phase = tide_phase[(tide_name,comp)] + np.pi

                        tide_dis = dis_amp*np.cos(omega*t_b + dis_phase) - dis_amp * np.cos(omega*t_a + dis_phase)
                        offset[comp] = offset[comp] + tide_dis
                
                # Velocity vector.
                offset_vec_2 = np.zeros(shape=(3,1))
                offset_vec_2[:,0] = [offset['e'],offset['n'],offset['u']]

            # Final offset_vec
            offset_vec = offset_vec_2

            # Two observation vectors.
            for j in range(2):
                obs_vec = np.zeros(shape=(3,1))
                obs_vec[:,0] = np.asarray(vecs[j])
                #print("Observation vector:\n", obs_vec)
                
                # Projection onto the observation vectors.
                obs_offset = np.matmul(np.transpose(offset_vec),obs_vec)
                #print(obs_offset)
                
                # Record the data.
                data_vector[2*i+j] = obs_offset

        # Add noise
        np.random.seed(seed=2018)
        data_vector = data_vector + np.random.normal(scale = noise_sigma, size=data_vector.shape)

        print('Data vector Done')

        return data_vector

def main():

    fourD_sim = simulation()

if __name__=='__main__':

    main()
