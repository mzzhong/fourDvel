#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in July, 2018

import os
import sys
import pickle
import shelve
import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import datetime
from datetime import date

import multiprocessing
from multiprocessing import Value

from configure import configure
from display import display

class estimate(configure):

    def __init__(self, param_file=None):

        if param_file:
            super(estimate,self).__init__(param_file)

            # Do the preparation to load data
            self.preparation()

            # Initializing visualization library
            self.display = display(param_file)

        else:
            raise Exception("Need to provide parameter file to class inversion")

    def estimate(self, point_set, tracks_set, task_name, inversion_method=None):

        ### Get data either from simulation or real data ###

        # Choose the test mode.
        test_id = self.test_id
        test_mode = self.test_mode

        # Num of points.
        n_points = len(point_set)

        # test point
        #print('The test point is: ', self.test_point)
        #print(stop)

        # All variables are dictionary with point_set as the key.
        # Data set formation.
        redo_dataset_formation = 1
        tmp_dataset_pkl_name = "tmp_point_set_dataset.pkl"
        if redo_dataset_formation:

            all_data_set = self.data_set_formation(point_set, tracks_set, test_mode)
            # Save the obtained data set
            with open(tmp_dataset_pkl_name,"wb") as f:
                pickle.dump(all_data_set,f)
        else:
            # Load the obtained data set
            with open(tmp_dataset_pkl_name,"rb") as f:
                all_data_set = pickle.load(f)

        (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set) = all_data_set

        print("Data set formation Done")

        do_linear = True

        if inversion_method == 'Bayesian_Linear' or do_linear == True:

            # Data prior.
            invCd_set = self.real_data_uncertainty_set(point_set, data_vec_set, \
                                                            noise_sigma_set)
            from solvers import Bayesian_Linear

            ### MODEL ###
            # Design matrix.
            linear_design_mat_set = self.build_G_set(point_set, offsetfields_set=offsetfields_set)
    
            #print("Design matrix set (G)\n:", linear_design_mat_set[self.test_point])
            print("Design matrix (obs) set Done")
    
            #design_mat_enu_set = self.build_G_ENU_set(point_set, offsetfields_set=offsetfields_set)
    
            #print("Design matrix set (G)\n:", linear_design_mat_set[self.test_point])
            print("Design matrix (enu) set Done")
     
            # Model prior.
            invCm_set = self.model_prior_set(point_set, horizontal = self.horizontal_prior)
            print("Model prior set Done")
    
            # Model posterior (Singular matrix will come back with nan).
            Cm_p_set = self.model_posterior_set(point_set, linear_design_mat_set, invCd_set, invCm_set)
            #print('Model posterior: ',Cm_p_set[self.test_point])
            print("Model posterior set Done")
    
            # Show the model posterior.
            #self.display.show_model_mat(Cm_p_set[self.test_point])
    
            ### Inversion ###
            # Estimate model params.
            model_vec_set = self.param_estimation_set(point_set, linear_design_mat_set, data_vec_set, invCd_set, invCm_set, Cm_p_set)
            #print('model_vec_set: ', model_vec_set)
            print('Model vec set estimation Done')

            # Output the result of test_point
            print("***Results of Bayesian linear (test point)")
            bl_model_vec = model_vec_set[self.test_point]
            print(bl_model_vec)
    
            # Calculale the residual.
            resid_of_secular_set, resid_of_tides_set = self.resids_set(point_set, linear_design_mat_set, data_vec_set, model_vec_set)
            print('Residual calculation Done')
    
            # Convert to tidal params.
            tide_vec_set = self.model_vec_set_to_tide_vec_set(point_set, model_vec_set)
            #print('tide_vec_set: ',tide_vec_set)
            print('Tide vec set Done')
    
            # Convert model posterior to uncertainty of params.
            # Require: tide_vec and Cm_p
            tide_vec_uq_set = self.model_posterior_to_uncertainty_set(point_set, tide_vec_set, Cm_p_set)
            print('Uncertainty set estimation Done')
    
            print('Point set inversion Done')

            ############ Analyze the results ##############################
            print('Start to analysis...')
            # Get continous displacement time series
            #self.continous_signal(self.test_point, tide_vec_set[self.test_point])
    
            # Display the misfit
            #self.analysis = analysis()
            #self.analysis.test_point = self.test_point
    
           # Save additional info in other_set_1
            other_set_1 = {} 
            #other_set_1 = self.analysis.check_fitting_set(point_set, data_info_set, offsetfields_set, linear_design_mat_set, design_mat_enu_set, data_vec_set, model_vec_set, tide_vec_set)
    
    
            ########### Show the results ########################
            # Stack the true and inverted models.
            # Show on point in the point set.

            self.show_vecs = False
            show_control = False
            if self.show_vecs == True or show_control== True:
            #if self.show_vecs == True and inversion_method=="Bayesian_Linear":
    
                if true_tide_vec_set is not None:
                    stacked_vecs = np.hstack((  true_tide_vec_set[self.test_point], 
                                                tide_vec_set[self.test_point], 
                                                tide_vec_uq_set[self.test_point]))
                    row_names = ['Input','Estimated','Uncertainty']
                    column_names = ['Secular'] + self.modeling_tides
                else:
                    stacked_vecs = np.hstack((  tide_vec_set[self.test_point], 
                                                tide_vec_uq_set[self.test_point]))
                    row_names = ['Estimated','Uncertainty']
                    column_names = ['Secular'] + self.modeling_tides
    
                self.display.display_vecs(stacked_vecs, row_names, column_names, test_id)

            #######################################################

        if inversion_method in ['Bayesian_MCMC', 'Bayesian_MCMC_Linear']:

            from solvers import Bayesian_MCMC

            BMC = Bayesian_MCMC(self.param_file)
            
            # Set the point_set to work on
            BMC.set_point_set(point_set)

            # Find the linear design mat
            linear_design_mat_set = self.build_G_set(point_set, offsetfields_set=offsetfields_set)

            # Set linear tides
            BMC.set_linear_design_mat_set(linear_design_mat_set)

            # Set modeling tides
            BMC.set_modeling_tides(self.modeling_tides)
            
            # Provide model priors
            self.up_lower = -4
            self.up_upper = 0
            BMC.set_model_priors(model_prior_set = true_tide_vec_set, no_secular_up = self.no_secular_up, up_short_period = self.up_short_period, horizontal_long_period = self.horizontal_long_period, up_lower = self.up_lower, up_upper = self.up_upper)

            # Provide data priors
            BMC.set_noise_sigma_set(noise_sigma_set)

            # Provide data
            BMC.set_data_set(data_vec_set)

            # Provide offsetfield info
            BMC.set_offsetfields_set(offsetfields_set)

            # Stack the design matrix modeling tides
            stack_design_mat_set = self.get_stack_design_mat_set(point_set, self.model_design_mat_set, offsetfields_set)

            # Provide the matrix to simulator
            BMC.set_stack_design_mat_set(stack_design_mat_set)

            # Obtain the up displacement
            up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

            # Provide the up displacement to the solver
            BMC.set_up_disp_set(up_disp_set)

            # Obtain true model vec from true tide vec
            if self.test_mode in [1,2]:

                true_model_vec_set = self.tide_vec_set_to_model_vec_set(point_set, true_tide_vec_set)

            else:
                true_model_vec_set = None

            # Run inversion
            est_grounding = None
            if inversion_method=="Bayesian_MCMC":
                model_vec, est_grounding = BMC.run_MCMC(run_point = self.test_point, true_model_vec_set=true_model_vec_set, task_name = task_name, suffix=str(self.test_mode))
                print("*** Result of Bayesian MCMC")
                print(model_vec)

            elif inversion_method=="Bayesian_MCMC_Linear":
                model_vec = BMC.run_MCMC_Linear(run_point = self.test_point, true_model_vec_set = true_model_vec_set, suffix=str(self.test_mode))

                print("*** Result of Bayesian MCMC Linear")
                print(model_vec)

            # Compare model vec and tide_vec
            stacked_model_vecs = []
            stacked_tide_vecs = []
            row_names = []
            # True model
            if true_model_vec_set is not None:
                true_model_vec = true_model_vec_set[self.test_point]
                true_tide_vec = true_tide_vec_set[self.test_point]

                stacked_model_vecs.append(true_model_vec)
                stacked_tide_vecs.append(true_tide_vec)
                row_names.append("Input")

            # Linear model
            if bl_model_vec.shape[0]>1:
                bl_tide_vec = self.model_vec_to_tide_vec(bl_model_vec)

                stacked_model_vecs.append(bl_model_vec)
                stacked_tide_vecs.append(bl_tide_vec)
                row_names.append("Linear")

            # Non-linear model
            tide_vec = self.model_vec_to_tide_vec(model_vec)

            stacked_model_vecs.append(model_vec)
            stacked_tide_vecs.append(tide_vec)
            row_names.append("Nonlinear")

            print("Compare model vecs")
            stacked_model_vecs = np.hstack(stacked_model_vecs)
            print(stacked_model_vecs)

            print("Compare tide vec")
            stacked_tide_vecs = np.hstack((stacked_tide_vecs))
            print(stacked_tide_vecs)

            print("@@ Estimated grounding level @@", est_grounding)

            # Visualize the models in a table
            stacked_vecs = stacked_tide_vecs
            column_names = ['Secular'] + self.modeling_tides
            self.display.display_vecs(stacked_vecs, row_names,column_names, test_id)

        ########### Inversion done ##########################

        # Record and return
        print("Recording...")
        all_sets = {}
        all_sets['true_tide_vec_set'] =  true_tide_vec_set
        all_sets['tide_vec_set'] = tide_vec_set

        all_sets['tide_vec_uq_set'] = tide_vec_uq_set
        all_sets['resid_of_secular_set'] = resid_of_secular_set
        all_sets['resid_of_tides_set'] = resid_of_tides_set
        all_sets['other_set_1'] = other_set_1

        return all_sets

