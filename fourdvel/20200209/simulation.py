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

    def __init__(self):

        super(simulation, self).__init__()

        tide_periods = self.tide_periods

        # Rutford data from from Murray (2007) 
        self.tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']
        tidesRut = self.tidesRut

        self.tidesRut_params = {}
        tidesRut_params = self.tidesRut_params

        ###############################################
        # Actual parameters from Murray (2007)
        # First two columns are horizontal responses. (amplitude cm/d)
        # Last two columns are vertical forcings. (amplitude cm)
        
        #tidesRut_params['K2'] =    [3.91,   163,    29.1,   99  ]
        #tidesRut_params['S2'] =    [4.56,   184,    101.6,  115 ]
        #tidesRut_params['M2'] =    [3.15,   177,    156.3,  70  ]
        #tidesRut_params['K1'] =    [1.22,   79,     49,     73  ]
        #tidesRut_params['P1'] =    [1.48,   77.0,   16.6,   64  ]
        #tidesRut_params['O1'] =    [1.54,   81.0,   43.1,   54  ]
        #tidesRut_params['Mf'] =    [1.17,   250.0,  2.9,    163 ]
        #tidesRut_params['Msf'] =   [5.65,   18.8,   0.3,    164 ]
        #tidesRut_params['Mm'] =    [1.15,   253.0,  1.6,    63  ]
        #tidesRut_params['Ssa'] =   [0.92,   256.0,  1.5,    179 ]
        #tidesRut_params['Sa'] =    [0.33,   273.0,  0.2,    179 ]

        ## Convert displacement to velocity.
        #for tide_name in tidesRut_params.keys():
        #    tidesRut_params[tide_name][2] = self.dis_amp_to_velo_amp(
        #                                            tidesRut_params[tide_name][2],
        #                                            tide_name = tide_name)
        #    # We don't know the definition of phase of vertical motion in displacement domian.
        #    # We directly use this phase.
        #    #tidesRut_params[tide_name][3] = tidesRut_params[tide_name][3]
        
        ###############################################

        #### Convention: This is in displacement domain

        # Disp = A * sin(omega * t + phi)

        # the corresponding Velo is
        # Velo = A * omega * sin(omega * t + phi + pi/2)

        # This conversion is done with 
        # dis_amp_to_velo_phase & dis_phase_velo_phase


        # Horizontal: assume this corresponds to 1 meter /day.
        self.ref_speed = 1
        # Vertical: assume the vertical scale.
        self.verti_scale = 1

        # Models are represented in displacement.
        model_num = 2

        if model_num == 1:

            # Model 1
            # Original
            # Displacement.        
            tidesRut_params['K2'] =    [0.31,  163,    29.1,   99]
            tidesRut_params['S2'] =    [0.363, 184,    101.6,  115]
            tidesRut_params['M2'] =    [0.259, 177,    156.3,  70] # M2

            tidesRut_params['K1'] =    [0.19,  79,     49,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   16.6,   64]

            tidesRut_params['O1'] =    [0.264, 81.0,   43,     54]  # O1
            tidesRut_params['Mf'] =    [2.54,  250.0,  2.9,    163] # Mf
            tidesRut_params['Msf'] =   [13.28, 18.8,   0.3,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  1.6,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  1.5,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0.2,    179]

        if model_num == 2:

            # Model 2.
            # Original
            # 1. Scale the vertical motion.
            coe = 2/3
 
            tidesRut_params['K2'] =    [0.31,  163,    29.1*coe,   99]
            tidesRut_params['S2'] =    [0.363, 184,    101.6*coe,  115]

            tidesRut_params['M2'] =    [0.259, 177,    156.3*coe,  70] # M2

            tidesRut_params['K1'] =    [0.19,  79,     49*coe,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   16.6*coe,   64]

            tidesRut_params['O1'] =    [0.264, 81.0,   43*coe,     54]  # O1
            tidesRut_params['Mf'] =    [2.54,  250.0,  2.9*coe,    163] # Mf
            tidesRut_params['Msf'] =   [13.28, 18.8,   0.3*coe,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  1.6*coe,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  1.5*coe,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0.2*coe,    179]

        if model_num == 3:

            # Model 3, increase the horizontal amplitude of M2 and O1
            # 1. Scale the vertical motion.
            coe = 2/3
            # 2. Add horizontal short_period on ice shelves.
 
            tidesRut_params['K2'] =    [5.00,  163,    29.1*coe,   99]
            tidesRut_params['S2'] =    [5.00, 184,    101.6*coe,  115]

            tidesRut_params['M2'] =    [10.00, 177,    156.3*coe,  70] # M2

            tidesRut_params['K1'] =    [4.00,  79,     49*coe,     73]
            tidesRut_params['P1'] =    [4.00,  77.0,   16.6*coe,   64]

            tidesRut_params['O1'] =    [4.00, 81.0,   43*coe,     54]  # O1
            tidesRut_params['Mf'] =    [2.54,  250.0,  2.9*coe,    163] # Mf
            tidesRut_params['Msf'] =   [13.28, 18.8,   0.3*coe,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  1.6*coe,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  1.5*coe,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0.2*coe,    179]

        if model_num == 4:

            # Model 4, increase the horizontal amplitude of M2 and O1
            # 1. Scale the vertical motion.
            coe = 2/3
            # 2. Add horizontal short_period on ice shelves.
            # 3. Use simple model (only S2 and M2) on ice shelf for periodic grounding.
 
            tidesRut_params['K2'] =    [5.00,  163,    0*coe,   99]
            tidesRut_params['S2'] =    [5.00, 184,     150.0*coe,  115]

            tidesRut_params['M2'] =    [10.00, 177,    150.0*coe,  70] # M2

            tidesRut_params['K1'] =    [4.00,  79,     0*coe,     73]
            tidesRut_params['P1'] =    [4.00,  77.0,   0*coe,   64]

            tidesRut_params['O1'] =    [4.00, 81.0,    0*coe,     54]  # O1

            tidesRut_params['Mf'] =    [2.54,  250.0,  0*coe,    163] # Mf
            tidesRut_params['Msf'] =   [13.28, 18.8,   0*coe,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  0*coe,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  0*coe,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0*coe,    179]

        if model_num == 5:

            # Model 4, increase the horizontal amplitude of M2 and O1
            # 1. Scale the vertical motion.
            coe = 2/3
            # 2. Remove added horizontal short_period on ice shelves.
            # 3. Use simple model (only S2 and M2) on ice shelf for periodic grounding.
 
            tidesRut_params['K2'] =    [0.31,  163,    0*coe,   99]
            tidesRut_params['S2'] =    [0.363, 184,     150.0*coe,  115]

            tidesRut_params['M2'] =    [0.269, 177,    150.0*coe,  70] # M2

            tidesRut_params['K1'] =    [0.19,  79,     0*coe,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   0*coe,   64]

            tidesRut_params['O1'] =    [0.264, 81.0,    0*coe,     54]  # O1

            tidesRut_params['Mf'] =    [2.54,  250.0,  0*coe,    163] # Mf
            tidesRut_params['Msf'] =   [13.28, 18.8,   0*coe,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  0*coe,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  0*coe,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0*coe,    179]



        ###############################################################      
        # Convert from displacement domain to velocity domain.
        for tide_name in tidesRut_params.keys():
            tidesRut_params[tide_name][0] = self.dis_amp_to_velo_amp(
                                                    tidesRut_params[tide_name][0],
                                                    tide_name = tide_name)
 
            tidesRut_params[tide_name][2] = self.dis_amp_to_velo_amp(
                                                    tidesRut_params[tide_name][2],
                                                    tide_name = tide_name)
            
            # Phase of displacement is 90 degree after velocity.
            # Assume the same functional form in velocity and displacement domain.
            tidesRut_params[tide_name][1] = self.dis_phase_to_velo_phase(
                                                    tidesRut_params[tide_name][1],
                                                    deg = True)

            tidesRut_params[tide_name][3] = self.dis_phase_to_velo_phase(
                                                    tidesRut_params[tide_name][3],
                                                    deg = True)
 
        #print(tidesRut_params)

        ##################################################################

        # Included constituents in synthetic data.
        #self.syn_tidesRut = []
        #self.syn_tidesRut = ['M2','O1','Msf']
        #self.syn_tidesRut = ['K2','S2','M2','K1','P1','O1','Mf','Msf']

        # all constituents are included
        self.syn_tidesRut = ['K2','S2','M2','K1','P1','O1','Msf','Mf','Mm','Ssa','Sa']


        # Load timings and design matrix (currently hardcoded)
        timings_pkl = os.path.join(self.pickle_dir, 'timings_csk_20171116_20200701_s1_20170601_20180601.pkl')

        if os.path.exists(timings_pkl):
            with open(timings_pkl,'rb') as f:
                self.timings = pickle.load(f)
        else:
            raise Exception('timing file is missing')

        design_mat_set_pkl = os.path.join(self.pickle_dir, 'design_mat_set_csk_20171116_20200701_s1_20170601_20180601_Rutford_full.pkl')

        if os.path.exists(design_mat_set_pkl):
            with open(design_mat_set_pkl, 'rb') as f:
                self.design_mat_set = pickle.load(f)
        else:
            raise Exception('design matrix file is missing')


    def true_tide_vec_set(self, point_set, secular_v_set, modeling_tides, tide_amp_set, tide_phase_set):
        
        true_tide_vec_set = {}

        for point in point_set:
                true_tide_vec_set[point] = self.true_tide_vec(secular_v_set[point],
                                                modeling_tides,
                                                tide_amp_set[point],
                                                tide_phase_set[point])
        return true_tide_vec_set

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

            # Phases (rad).
            phi_E = self.wrapped(phi_E)
            phi_N = self.wrapped(phi_N)
            phi_U = self.wrapped(phi_U)

            # Put them into the vector.
            param_vec[3 + k*6 : 3 + (k+1)*6] = np.expand_dims(np.asarray([amp_E, amp_N, amp_U, phi_E, phi_N, phi_U]), axis=1)

        return param_vec

    def syn_velocity_set(self, point_set, velo_model_set):
        
        # Initilization
        secular_v_set = {}
        v_set = {}
        tide_amp_set = {}
        tide_phase_set = {}
        
        for point in point_set:

            velo_model = velo_model_set[point]

            #if point==self.test_point:
            #    print(velo_model)
            #    print(stop)
        
            (t_axis, secular_v, v, tide_amp, tide_phase) = self.syn_velocity(
                                                            velo_model = velo_model)

            secular_v_set[point] = secular_v
            v_set[point] = v         
            tide_amp_set[point] = tide_amp
            tide_phase_set[point] = tide_phase
       

        # Saved to class
        self.t_axis = t_axis
        self.v_set = v_set

        # Return parameters.
        return (secular_v_set, tide_amp_set, tide_phase_set)


    def syn_velocity(self, velo_model):
        
        # Tides.
        tide_periods = self.tide_periods

        #print(velo_model)
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

        #print(secular_v_e, secular_v_n)
        secular_speed = np.sqrt(secular_v_e**2 + secular_v_n**2)
        secular_v = (secular_v_e, secular_v_n, secular_v_u)

        # Tides
        tide_amp = {}
        tide_phase = {}

        # Tides. (Find the parameters)
        for tide_name in syn_tidesRut:
            #print(tide_name)

            omega = 2*np.pi / tide_periods[tide_name] 

            horiz_amp = self.cm2m(tidesRut_params[tide_name][0]) # m/d velocity
            horiz_phase = np.deg2rad(tidesRut_params[tide_name][1])

            verti_amp = self.cm2m(tidesRut_params[tide_name][2]) # m/d velocity
            verti_phase = np.deg2rad(tidesRut_params[tide_name][3])

            # East component. (Amp always positive)
            a_e = horiz_amp * abs(secular_v_e/secular_speed) * secular_speed / ref_speed
            # Phase change sign, when amp is negative.
            phi_e = np.sign(secular_v_e) * horiz_phase

            # North component. (Amp always positive)
            a_n = horiz_amp * abs(secular_v_n/secular_speed) * secular_speed / ref_speed
            # Phase change sign, when amp is negative.
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

        # If "numerical" and "both", simulate the real time series
        # Be careful about the length of time series
        # If "analytical", this part is skipped. tide_amp and tide_phase 
        # are used to derive offset directly.

        #self.method = 'analytical'
        self.method = 'analytical'

        # Create synthetic time series
        # "sin" is used

        if self.method == 'numerical' or self.method == 'both':
            sim = {}
            for tide_name in syn_tidesRut:

                # the identifier of using method numerical 
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
            #fig = plt.figure(1,figsize=(16,8))
            #ax = fig.add_subplot(211)
    
            p_e = np.zeros(shape=t_axis.shape)
            p_n = np.zeros(shape=t_axis.shape)
            p_u = np.zeros(shape=t_axis.shape)
            for tide_name in syn_tidesRut:
                p_e = p_e + sim[(tide_name,'e')]
                p_n = p_n + sim[(tide_name,'n')]
                p_u = p_u + sim[(tide_name,'u')]
    
                #ax.plot(t_axis,sim[(tide_name,'n')],label=tide_name + '_N')
                #ax.plot(t_axis,sim[(tide_name,'u')],label=tide_name + '_U')
    
            #ax.legend()
    
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
            #ax.plot(t_axis, v_u)
            #ax.set_xlim([-50,50])

            #fig.savefig('fig_sim/1.png',format='png')

        else:
            t_axis = None
            v = None
       
        # Return synthetic velocity time series.
        return (t_axis, secular_v, v, tide_amp, tide_phase)


    def set_stack_design_mat_set(self, stack_design_mat_set):

        self.stack_design_mat_set = stack_design_mat_set

        return 0

    def set_grounding(self, grounding):

        self.grounding = grounding

        return 0

    def syn_offsets_data_vec_set(self, point_set, secular_v_set, modeling_tides, 
                            tide_amp_set, tide_phase_set, offsetfields_set, noise_sigma_set):

        data_vec_set = {}
        count = 0
        for point in point_set:
            #print('simulation at point: ', count)
            count = count + 1

            # Obtain offsets from synthetics.
            offsetfields = offsetfields_set[point]
            secular_v = secular_v_set[point]
            tide_amp = tide_amp_set[point]
            tide_phase = tide_phase_set[point]
            noise_sigma = noise_sigma_set[point]

            data_vec_set[point] = self.syn_offsets_data_vec( point=point, secular_v = secular_v,
                                                        modeling_tides = modeling_tides,
                                                        tide_amp = tide_amp,
                                                        tide_phase = tide_phase,
                                                        offsetfields = offsetfields,
                                                        noise_sigma = noise_sigma
                                                        )

        return data_vec_set

    def syn_offsets_data_vec(self, point=None, secular_v=None, modeling_tides=None, tide_amp=None, tide_phase=None, offsetfields=None, noise_sigma = None):

        # Obtain offsets from synthetics.
        n_offsets = len(offsetfields)
        #print("number of offsetfields:", n_offsets)
        n_rows = n_offsets * 2
        data_vector = np.zeros(shape=(n_rows,1))
        t_origin = self.t_origin.date()

        method = "with grounding"
        #method = "without grounding"

        # Three components.
        # Numerical method
        if method == "time series provided":
            #print(method)

            #print('numerical')
            t_axis = self.t_axis
            v = self.v_set[point]
            v_e, v_n, v_u = v

            # Find the d_e, d_n, d_u
            #t_axis, v_e, v_n, v_u
            delta = t_axis[1] - t_axis[0]
            d_e = np.cumsum(np.copy(v_e) * delta)
            d_n = np.cumsum(np.copy(v_n) * delta)
            d_u = np.cumsum(np.copy(v_u) * delta)

            d_e = d_e - np.mean(d_e)
            d_n = d_n - np.mean(d_n)
            d_u = d_u - np.mean(d_u)

            # periodic grounding
            d_u[d_u<self.grounding] = self.grounding
            
            # Plot the velocity time series.
            fig = plt.figure(1, figsize=(10,5))
            ax = fig.add_subplot(111)
            #ax.plot(t_axis, v_u/np.max(v_u), 'r')
            ax.plot(t_axis, d_u, 'b')
            ax.set_xlim([-12,12])
            fig.savefig('fig_sim/2.png',format='png')
            print(np.max(d_u))

        # Analytical with grounding
        elif method == "with grounding old way":
            #print(method)

            # secular offset
            secular_off = np.asarray(secular_v)[:,None] * (t_b - t_a)
            print(secular_off)

            # Tidal displacement
            G_a = self.design_mat_set[timing_a]
            G_b = self.design_mat_set[timing_b]

            dis_timing_a = np.matmul(G_a, model_vec)
            dis_timing_b = np.matmul(G_b, model_vec)

            #print('dis_EN_ta: ',dis_EN_ta[0:2])
            #print('dis_EN_tb: ',dis_EN_tb[0:2])
            #print('dis_U_ta: ',dis_U_ta[0])
            #print('dis_U_tb: ',dis_U_tb[0])
        

            #print('dis_timing_a: ',dis_timing_a)
            #print('dis_timing_b: ',dis_timing_b)

            # Grounding
            if dis_timing_b[2] < grounding:
                dis_timing_b[2] = grounding
            if dis_timing_a[2] < grounding:
                dis_timing_a[2] = grounding

            # Tidal offset
            tide_off = dis_timing_b - dis_timing_a

            # Total offset
            offset_vec = secular_off + tide_off

        elif method == "with grounding":
            #print(method)

            ## For a single point
            timings = self.timings
            design_mat_set = self.design_mat_set

            # Convert parameters from velocity to displacement
            tide_dis_amp = {}
            tide_dis_phase = {}

            for comp in ['e','n','u']:
                for tide_name in self.syn_tidesRut:            
                    omega = 2*np.pi / self.tide_periods[tide_name]
                    # amplitude
                    tide_dis_amp[(tide_name,comp)] = tide_amp[(tide_name,comp)] / omega
                    # phase
                    tide_dis_phase[(tide_name,comp)] = tide_phase[(tide_name,comp)]+ np.pi

            # Derive amplitudes for cos and sin terms
            cos_coef = {}
            sin_coef = {}

            for comp in ['e','n','u']:
                for tide_name in self.syn_tidesRut:
                    cos_coef[(tide_name,comp)] = tide_dis_amp[(tide_name, comp)] * np.cos(tide_dis_phase[tide_name,comp])
                    sin_coef[(tide_name, comp)] = (-1) * tide_dis_amp[(tide_name, comp)] * np.sin(tide_dis_phase[tide_name,comp])

            # Construct model vector
            model_vec = []
            for tide_name in self.syn_tidesRut:
                for comp in ['e','n','u']:
                    model_vec.append(cos_coef[(tide_name, comp)])
                for comp in ['e','n','u']:
                    model_vec.append(sin_coef[(tide_name, comp)])

            model_vec = np.asarray(model_vec)[:,None]

            #print(cos_coef[('S2','u')])
            #print(sin_coef[('S2','u')])
            #print(self.grounding)
            #print(stop)

            # Obtain stacked matrix
            stacked_design_mat_EN_ta, stacked_design_mat_EN_tb, stacked_design_mat_U_ta, stacked_design_mat_U_tb = self.stack_design_mat_set[point]

            # Find horizontal displacement at timing_a, timing_b
            dis_EN_ta = np.matmul(stacked_design_mat_EN_ta, model_vec)
            dis_EN_tb = np.matmul(stacked_design_mat_EN_tb, model_vec)

            # Find vertical displacement at timing_a, timing_b
            dis_U_ta = np.matmul(stacked_design_mat_U_ta, model_vec)
            dis_U_tb = np.matmul(stacked_design_mat_U_tb, model_vec)

            # Grounding
            dis_U_ta[dis_U_ta < self.grounding] = self.grounding
            dis_U_tb[dis_U_tb < self.grounding] = self.grounding

            # Find offset
            offset_EN = dis_EN_tb - dis_EN_ta
            offset_U = dis_U_tb - dis_U_ta

            offset_ENU = np.vstack((np.transpose(offset_EN.reshape(n_offsets,2)), np.transpose(offset_U)))

            for i in range(n_offsets):
                t_a = (offsetfields[i][0] - t_origin).days + round(offsetfields[i][4],4)
                t_b = (offsetfields[i][1] - t_origin).days + round(offsetfields[i][4],4)
                
                tmp = np.asarray(secular_v)
                
                offset_ENU[:,i] = offset_ENU[:,i] +  tmp * (t_b - t_a) 

            #print('offset_ENU: ', offset_ENU.shape)

            # Find observed offset
            data_vector1 = np.zeros(shape=(n_offsets*2,1))
            for i in range(n_offsets):
                data_vector1[2*i,0] = np.dot(offsetfields[i][2],offset_ENU[:,i])
                data_vector1[2*i+1,0] = np.dot(offsetfields[i][3],offset_ENU[:,i])

            data_vector = data_vector1

        elif method == "without grounding":

            #print(method)

            ### CONVERT parameters from velocity to displacement ###
            tide_dis_amp = {}
            tide_dis_phase = {}

            for comp in ['e','n','u']:
                for tide_name in self.syn_tidesRut:            
                    omega = 2*np.pi / self.tide_periods[tide_name]
                    # amplitude
                    tide_dis_amp[(tide_name,comp)] = tide_amp[(tide_name,comp)] / omega
                    # phase
                    tide_dis_phase[(tide_name,comp)] = tide_phase[(tide_name,comp)]+ np.pi

            # Generate offsets
            for i in range(n_offsets):
    
                #print('offsetfield: ',i,'/',n_offsets,'\n')
    
                #print(offsetfields[i])
                vecs = [offsetfields[i][2],offsetfields[i][3]]
                
                t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
                t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]
    
                timing_a = (offsetfields[i][0], round(offsetfields[i][4],4))
                timing_b = (offsetfields[i][1], round(offsetfields[i][4],4))

                option = "analytical"
    
                if option == 'numerical_old_way':
    
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
                    offset_vec = np.zeros(shape=(3,1))
                    offset_vec[:,0] = [offset_e,offset_n,offset_u]
    
                    #print(offset_vec)
                
                elif option == "numerical_new_way":
    
                    if t_a >= t_axis[0] and t_b <= t_axis[-1]:
                        inds = np.argmin(np.abs(t_axis-t_a))
                        inde = np.argmin(np.abs(t_axis-t_b))
                    else:
                        raise Exception('Time axis too short!')
    
                    offset_e = d_e[inde] - d_e[inds]
                    offset_n = d_n[inde] - d_n[inds]
                    offset_u = d_u[inde] - d_u[inds]
    
                    # Velocity vector.
                    offset_vec = np.zeros(shape=(3,1))
                    offset_vec[:,0] = [offset_e,offset_n,offset_u]
    
                elif option == 'analytical':
    
                    offset={}
    
                    offset['e'] = 0
                    offset['n'] = 0
                    offset['u'] = 0
    
                    # Three components.
                    comps = ['e','n','u']
    
                    #print(tide_amp)
                    #print(tide_phase)
    
                    ii = 0
                    for comp in comps:
                        offset[comp] = offset[comp] + secular_v[ii] * (t_b - t_a)
                        ii = ii + 1
    
                        # Changed to Rutford tides instead of modeling tides
                        # 2019.07.05
    
                        # Iterative over all tidal components
                        for tide_name in self.syn_tidesRut:
    
                            omega = 2*np.pi / self.tide_periods[tide_name]
                            dis_amp = tide_dis_amp[(tide_name, comp)]
                            dis_phase = tide_dis_phase[(tide_name, comp)]
                            
                            # Displacement difference
                            tide_dis = dis_amp*np.cos(omega*t_b + dis_phase) - dis_amp * np.cos(omega*t_a + dis_phase)
    
                            offset[comp] = offset[comp] + tide_dis
    
                        #print(stop)
                    
                    # Velocity vector.
                    offset_vec = np.zeros(shape=(3,1))
                    offset_vec[:,0] = [offset['e'],offset['n'],offset['u']]
    
    
                ########## End with constructing offset_vec in without grounding #############3
    
                # Project 3d displacement onto observational vectors
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

        ###########  Add noise #####################
        lon, lat = point
        seed_num = int(lon*10+lat) % (2**30-1)
        np.random.seed(seed=seed_num)

        # Range.
        data_vector[0::2] = data_vector[0::2] + np.random.normal(scale = noise_sigma[0], size=data_vector[0::2].shape)

        # Azimuth.
        data_vector[1::2] = data_vector[1::2] + np.random.normal(scale = noise_sigma[1], size=data_vector[1::2].shape)

        return data_vector


def main():

    fourD_sim = simulation()

if __name__=='__main__':

    main()

## Deprecated:
#    def syn_offsets_data_vec_set(self, point_set, secular_v_set, modeling_tides, 
#                            tide_amp_set, tide_phase_set, offsetfields_set, noise_sigma_set):
#
#        data_vec_set = {}
#        # Point by Point.
#        count = 0
#        for point in point_set:
#            print(count)
#            count = count + 1
#
#            # Obtain offsets from synthetics.
#            offsetfields = offsetfields_set[point]
#            secular_v = secular_v_set[point]
#            tide_amp = tide_amp_set[point]
#            tide_phase = tide_phase_set[point]
#            
#            n_offsets = len(offsetfields)
#            #print("number of offsetfields:", n_offsets)
#            n_rows = n_offsets * 2
#            data_vector = np.zeros(shape=(n_rows,1))
#            t_origin = self.t_origin.date()
#   
#            for i in range(n_offsets):
#    
#                #print('offsetfield: ',i,'/',n_offsets,'\n')
#                vecs = [offsetfields[i][2],offsetfields[i][3]]
#                t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
#                t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]
#    
#                offset={}
#                offset['e'] = 0
#                offset['n'] = 0
#                offset['u'] = 0
#
#                n_modelng_tides = len(modeling_tides) # Three components.
#                comps = ['e','n','u']
#    
#                ii = 0
#                for comp in comps:
#                    offset[comp] = offset[comp] + secular_v[ii] * (t_b - t_a)
#                    ii = ii + 1
#                    for tide_name in modeling_tides:
#                        omega = 2*np.pi / self.tide_periods[tide_name] 
#                        dis_amp = tide_amp[(tide_name,comp)] / omega
#                        dis_phase = tide_phase[(tide_name,comp)] + np.pi
#    
#                        tide_dis = dis_amp*np.cos(omega*t_b + dis_phase) - dis_amp * np.cos(omega*t_a + dis_phase)
#                        offset[comp] = offset[comp] + tide_dis
#                
#                # Velocity vector.
#                offset_vec = np.zeros(shape=(3,1))
#                offset_vec[:,0] = [offset['e'],offset['n'],offset['u']]
#    
#                # Project to two observation vectors.
#                for j in range(2):
#                    obs_vec = np.zeros(shape=(3,1))
#                    obs_vec[:,0] = np.asarray(vecs[j])
#                    #print("Observation vector:\n", obs_vec)
#                    
#                    # Projection onto the observation vectors.
#                    obs_offset = np.matmul(np.transpose(offset_vec),obs_vec)
#                    #print(obs_offset)
#                    
#                    # Record the data.
#                    data_vector[2*i+j] = obs_offset
#
#            data_vec_set[point] = data_vector
#    
#        # Add noise
#        np.random.seed(seed=2018)
#        for point in point_set:
#            data_vec_set[point] = data_vec_set[point] +  \
#                np.random.normal(scale = noise_sigma_set[point], size=data_vec_set[point].shape)
#    
#        print('Data vector Done')
#
#        return data_vec_set


#    def syn_velocity_set(self, point_set, velo_model_set):
#        
#        # Tides.
#        tide_periods = self.tide_periods
#
#        # Reference speed to tide amplitudes.
#        ref_speed = self.ref_speed
#        verti_scale = self.verti_scale
#
#        # Rutford tide model.
#        tidesRut_params = self.tidesRut_params
#        syn_tidesRut = self.syn_tidesRut
#
#        # Initilization
#        secular_v_set = {}
#        tide_amp_set = {}
#        tide_phase_set = {}
#        
#        for point in point_set:
#            # Secular velcity.
#            secular_v_e = velo_model_set[point][0]
#            secular_v_n = velo_model_set[point][1]
#            secular_v_u = 0
#            verti_ratio = velo_model_set[point][2]
#
#            secular_speed = np.sqrt(secular_v_e**2 + secular_v_n**2)
#            secular_v_set[point] = (secular_v_e, secular_v_n, secular_v_u)
#
#            # Tides
#            tide_amp = {}
#            tide_phase = {}
#
#            # Tides. (Find the parameters)
#            for tide_name in syn_tidesRut:
#                #print(tide_name)
#
#                omega = 2*np.pi / tide_periods[tide_name] 
#
#                horiz_amp = tidesRut_params[tide_name][0]/100 # velocity
#                horiz_phase = np.deg2rad(tidesRut_params[tide_name][1])
#
#                verti_amp = tidesRut_params[tide_name][2]/100 # velocity
#                verti_phase = np.deg2rad(tidesRut_params[tide_name][3])
#
#                # East component.
#                a_e = horiz_amp * abs(secular_v_e/secular_speed) * secular_speed / ref_speed
#                phi_e = np.sign(secular_v_e) * horiz_phase
#
#                # North component.
#                a_n = horiz_amp * abs(secular_v_n/secular_speed) * secular_speed / ref_speed
#                phi_n = np.sign(secular_v_n) * horiz_phase # rad
#
#                # Up component. 
#                a_u = verti_amp * verti_scale * verti_ratio
#                phi_u = verti_phase
#                
#                # Record the tides.
#                tide_amp[(tide_name,'e')] = a_e
#                tide_amp[(tide_name,'n')] = a_n
#                tide_amp[(tide_name,'u')] = a_u
#
#                tide_phase[(tide_name,'e')] = phi_e
#                tide_phase[(tide_name,'n')] = phi_n
#                tide_phase[(tide_name,'u')] = phi_u
#            
#            tide_amp_set[point] = tide_amp
#            tide_phase_set[point] = tide_phase
#       
#        # Return parameters.
#        return (secular_v_set, tide_amp_set, tide_phase_set)


#    def flow_model(self):
#
#        # Constants for ice flow model.
#        A = 2.4e-24
#        alpha = 0.04
#        g = 9.81
#        h = 1000
#        n_g = 3
#        rho = 900
#        s_v = 0.6
#
#        # Gravitional driving stress.
#        tau_d = rho*g*h*alpha
#
#        # Basal drag.
#        tau_b = 0.8 * tau_d
#
#        w = 50*1000 #m
#        L = 150*1000 #m
#
#        # End of constants.
#
#        # Beginning of simulations.
#        x = np.linspace(0,L,num=np.round(L/500))
#        y = np.linspace(0,2*w,num=np.round(2*w/500))
#
#        yy,xx = np.meshgrid(y,x)
#        #print(xx.shape)
#        #print(tau_d)
#        #print(tau_d * w/h)
#        #print(tau_d * w/h * 0.2)
#
#        #print(2*A*w/(n_g+1))
#
#        v_ideal_center = 2*A*w/(n_g+1) * (tau_d * w / h * 0.2)**n_g
#        print(v_ideal_center)
#
#        v_ideal = v_ideal_center * (1 - (1-yy/w)**(n_g+1))
#
#        k_h=10**(-1*np.abs(np.log10(L)-0.8))
#        gamma = (1 + np.tanh(k_h * (x-0.6*L)))/2
#
#        Gamma = {}
#
#        Gamma_const = v_ideal/v_ideal_center
#
#
#        # Add all tide signals together.
#        p_e = np.zeros(shape=t_axis.shape)
#        p_n = np.zeros(shape=t_axis.shape)
#        p_u = np.zeros(shape=t_axis.shape)
#
#        for key in syn_tidesRut:
#            p_e = p_e + sim[(key,'e')]
#            p_n = p_n + sim[(key,'n')]
#            p_u = p_u + sim[(key,'u')]
#
#        # Full velocity signals.
#        # Constant velocity + tidal signals.
#        # Consider the transition from ice stream to ice shelf.
#        v_e = np.zeros(shape=t_axis.shape)
#
#        v_n = s_v * x_loc/L * (-v_ideal[ind_x,ind_y]*t_axis + p_n)
#        v_n_tides = s_v * x_loc/L * p_n
#
#        v_u = s_v * (x_loc-L)/(10*L) * v_ideal[ind_x,ind_y] + p_u
#        v_u_tides = p_u
#
#        # Plotting.
#        fig = plt.figure(2,figsize=(7,7))
#        ax = fig.add_subplot(111)
#        
#        # Choice 1.
#        #p1 = ax.imshow(v_ideal/v_ideal_center,cmap=plt.cm.coolwarm)
#        #p1 = ax.imshow(v_ideal,cmap=plt.cm.coolwarm)
#        #fig.colorbar(p1)
#
#        # Choice 2. Summed tidal signals.
#        ax.plot(t_axis, v_u)
#
#        # velocity
#        #v_e = np.zeros(shape=xx.shape)
#        #v_n = np.z
#
#        plt.show()
#
#        return


