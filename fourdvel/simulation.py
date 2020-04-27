#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in Aug, 2018

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
import sys
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

class simulation(fourdvel):

    def __init__(self, param_file):

        if param_file is not None:
            super(simulation, self).__init__(param_file)
        elif len(sys.argv)>1:
            super(simulation, self).__init__(sys.argv[1])
        else:
            print("Need parameter file")
            raise Exception()

        # Tide periods
        tide_periods = self.tide_periods

        # Vertical displacement model
        self.get_tidal_model()

        # Set the tidal constituents in the data
        self.tidesRut = ['K2','S2','M2','N2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']
        tidesRut = self.tidesRut

        self.tidesRut_params = {}
        tidesRut_params = self.tidesRut_params

        # Load Rutford vertical and horizontal tide model from from Murray (2007)
        #############################################################
        # Actual parameters from Murray (2007)
        # First two columns are horizontal responses. (amplitude cm/d)
        # Last two columns are vertical forcings. (amplitude cm)
        # N2 is added
        
        #tidesRut_params['K2'] =    [3.91,   163,    29.1,   99  ]
        #tidesRut_params['N2'] =    [3.91,   163,    30.5,   20  ]
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
        #############################################################

        #### Convention: This is in displacement domain ####

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
        model_num = 1

        if model_num == 1:

            # Model 1
            # Original
            # Displacement.        
            tidesRut_params['K2'] =    [0.31,  163,    29.1,   99]
            tidesRut_params['N2'] =    [0.32,   163,    30.5,   20]
            tidesRut_params['S2'] =    [0.363, 184,    101.6,  115]
            tidesRut_params['M2'] =    [0.259, 177,    156.3,  70]  # M2

            tidesRut_params['K1'] =    [0.19,  79,     49,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   16.6,   64]

            tidesRut_params['O1'] =    [0.264, 81.0,   43,     54]  # O1
            tidesRut_params['Mf'] =    [15.00,  250.0,  2.9,    163] # Mf
            #tidesRut_params['Msf'] =   [40.00, 18.8,   0.3,    164] # Msf
            tidesRut_params['Msf'] =   [35.00, 260,   0.3,    164] # New Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  1.6,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  1.5,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0.2,    179]

        if model_num == 2:

            # Model 2.
            # Original
            # 1. Scale the vertical motion.
            coe = 2/3
 
            tidesRut_params['K2'] =    [0.31,  163,    29.1*coe,   99]
            tidesRut_params['N2'] =    [0.32,   163,    30.5*coe,   20]
            tidesRut_params['S2'] =    [0.363, 184,    101.6*coe,  115]
            tidesRut_params['M2'] =    [0.259, 177,    156.3*coe,  70]  # M2

            tidesRut_params['K1'] =    [0.19,  79,     49*coe,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   16.6*coe,   64]
            tidesRut_params['O1'] =    [0.264, 81.0,   43*coe,     54]  # O1
            
            tidesRut_params['Mf'] =    [15.00,  250.0,  2.9*coe,    163] # Mf
            tidesRut_params['Msf'] =   [40.00, 18.8,   0.3*coe,    164] # Msf

            tidesRut_params['Mm'] =    [5.04,  253.0,  1.6*coe,    63]
            tidesRut_params['Ssa'] =   [26.74, 256.0,  1.5*coe,    179]
            tidesRut_params['Sa'] =    [19.18, 273.0,  0.2*coe,    179]

        if model_num == 3:

            # Model 3, increase the horizontal amplitude of M2 and O1
            # 1. Scale the vertical motion.
            coe = 2/3
            # 2. Add horizontal short_period on ice shelves.
 
            tidesRut_params['K2'] =    [5.00,  163,    29.1*coe,   99]
            tidesRut_params['N2'] =    [5.00,   163,    30.5*coe,   20]
            tidesRut_params['S2'] =    [5.00, 184,    101.6*coe,  115]

            tidesRut_params['M2'] =    [10.00, 177,    156.3*coe,  70] # M2

            tidesRut_params['K1'] =    [4.00,  79,     49*coe,     73]
            tidesRut_params['P1'] =    [4.00,  77.0,   16.6*coe,   64]

            tidesRut_params['O1'] =    [4.00, 81.0,   43*coe,     54]  # O1
            tidesRut_params['Mf'] =    [15.00,  250.0,  2.9*coe,    163] # Mf
            tidesRut_params['Msf'] =   [40.00, 18.8,   0.3*coe,    164] # Msf

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
            tidesRut_params['N2'] =    [5.00,   163,    0*coe,   20]
            tidesRut_params['S2'] =    [5.00, 184,     150.0*coe,  115]

            tidesRut_params['M2'] =    [10.00, 177,    150.0*coe,  70] # M2

            tidesRut_params['K1'] =    [4.00,  79,     0*coe,     73]
            tidesRut_params['P1'] =    [4.00,  77.0,   0*coe,   64]

            tidesRut_params['O1'] =    [4.00, 81.0,    0*coe,     54]  # O1

            tidesRut_params['Mf'] =    [15.00,  250.0,  0*coe,    163] # Mf
            tidesRut_params['Msf'] =   [40.00, 18.8,   0*coe,    164] # Msf

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
            tidesRut_params['N2'] =    [0.32,   163,    0*coe,   20]

            tidesRut_params['S2'] =    [0.363, 184,     150.0*coe,  115]

            tidesRut_params['M2'] =    [0.269, 177,    150.0*coe,  70] # M2

            tidesRut_params['K1'] =    [0.19,  79,     0*coe,     73]
            tidesRut_params['P1'] =    [0.24,  77.0,   0*coe,   64]

            tidesRut_params['O1'] =    [0.264, 81.0,    0*coe,     54]  # O1

            tidesRut_params['Mf'] =    [15.00,  250.0,  0*coe,    163] # Mf
            tidesRut_params['Msf'] =   [40.00, 18.8,   0*coe,    164] # Msf

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

        # Set the tides for simulation given by the parameter file
        self.syn_tidesRut = self.simulation_tides

        # Load reference velocity model
        self.get_grid_set_v2()
        self.get_grid_set_velo()


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
            #phi_e = np.sign(secular_v_e) * horiz_phase
            phi_e = horiz_phase
            if secular_v_e < 0: phi_e = self.wrapped(phi_e + np.pi)

            # North component. (Amp always positive)
            a_n = horiz_amp * abs(secular_v_n/secular_speed) * secular_speed / ref_speed
            # Phase change sign, when amp is negative.
            #phi_n = np.sign(secular_v_n) * horiz_phase # rad
            phi_n = horiz_phase
            if secular_v_n < 0: phi_n = self.wrapped(phi_n + np.pi)

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

    def set_up_disp_set(self, up_disp_set):
        self.up_disp_set = up_disp_set
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

        method = self.simulation_method

        # Three components.
        # Numerical method
        if method == "time series provided":
            print("time series is provided")
            print("numerical")
            
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
            d_u[d_u < self.grounding] = self.grounding
            
            # Plot the velocity time series.
            fig = plt.figure(1, figsize=(10,5))
            ax = fig.add_subplot(111)
            #ax.plot(t_axis, v_u/np.max(v_u), 'r')
            ax.plot(t_axis, d_u, 'b')
            ax.set_xlim([-12,12])
            fig.savefig('fig_sim/2.png',format='png')
            print(np.max(d_u))

        elif method == "model_with_grounding":

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


            # Obtain stacked matrix
            stacked_design_mat_EN_ta, stacked_design_mat_EN_tb, stacked_design_mat_U_ta, stacked_design_mat_U_tb = self.stack_design_mat_set[point]

            # Obtain up displacement vector
            tide_height_master_model, tide_height_slave_model = self.up_disp_set[point]
            
            # Scale the up displacement vector
            velo_model = self.grid_set_velo[point]
            tide_height_master = tide_height_master_model * velo_model[2]
            tide_height_slave = tide_height_slave_model * velo_model[2]

            # Note that the stacked design mat/up displacement may be empty 
            # because there is no data.
            # In this case, the stacked_design_mat is a empty list
            # return an empty vector

            if point == self.test_point:
                
                #print(tide_height_master)
                #print(tide_height_slave)
                print("master tide height: ", tide_height_master.shape)
                print("slave tide height: ", tide_height_slave.shape)
                print(velo_model)
                #print(stop)


            if len(stacked_design_mat_EN_ta)==0:

                data_vector = np.asarray([])

            else:

                # Find horizontal displacement at timing_a, timing_b
                dis_EN_ta = np.matmul(stacked_design_mat_EN_ta, model_vec)
                dis_EN_tb = np.matmul(stacked_design_mat_EN_tb, model_vec)

                # Find vertical displacement at timing_a, timing_b
                # Use model
                if not self.external_up_disp: 
                    dis_U_ta = np.matmul(stacked_design_mat_U_ta, model_vec)
                    dis_U_tb = np.matmul(stacked_design_mat_U_tb, model_vec)
                # Use external data
                else:
                    dis_U_ta = tide_height_master.reshape(len(tide_height_master),1)
                    dis_U_tb = tide_height_slave.reshape(len(tide_height_slave),1)

                #print(dis_U_ta_1.shape)
                #print(dis_U_ta_2.shape)
                #print(velo_model)
                #print(stop)
                   
    
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

        elif method == "model_without_grounding":

            ### CONVERT parameters from velocity to displacement ###
            tide_dis_amp = {}
            tide_dis_phase = {}

            for comp in ['e','n','u']:
                for tide_name in self.syn_tidesRut:            
                    omega = 2*np.pi / self.tide_periods[tide_name]

                    # Changed 2020.02.09
                    # amplitude
                    #tide_dis_amp[(tide_name,comp)] = tide_amp[(tide_name,comp)] / omega
                    tide_dis_amp[(tide_name,comp)] = self.velo_amp_to_dis_amp(tide_amp[(tide_name,comp)], tide_name)

                    # phase
                    #tide_dis_phase[(tide_name,comp)] = tide_phase[(tide_name,comp)] + np.pi
                    tide_dis_phase[(tide_name, comp)] = self.velo_phase_to_dis_phase(tide_phase[(tide_name, comp)], deg=False)

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
                            # Changed to use sin instead of cos 2020.02.09
                            #tide_dis = dis_amp * np.cos(omega*t_b + dis_phase) - dis_amp * np.cos(omega*t_a + dis_phase)
                            tide_dis = dis_amp * np.sin(omega*t_b + dis_phase) - dis_amp * np.sin(omega*t_a + dis_phase)
   
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

