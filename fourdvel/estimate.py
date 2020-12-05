#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in July, 2018

import os
import sys
import pickle
import shelve
import time
import copy

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
from solvers import solvers

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

    def set_task_name(self, task_name):
    
        self.task_name = task_name

    def estimate(self, point_set, tracks_set):

        ### Get data either from simulation or real data ###

        # Find the info
        test_id = self.test_id

        data_mode = {}
        data_mode['csk'] = self.csk_data_mode
        data_mode['s1'] = self.s1_data_mode

        task_name = self.task_name
        inversion_method = self.inversion_method

        # Num of points.
        n_points = len(point_set)

        # Initialize many sets
        true_tide_vec_set = {}
        tide_vec_set = {}

        tide_vec_uq_set = {}
        resid_of_secular_set = {}
        resid_of_tides_set = {}

        others_set = {}
        for point in point_set:
            others_set[point] = {}
        
        # test point
        #print('The test point is: ', self.test_point)

        # All variables are dictionary with point_set as the key.
        # Data set formation.
        redo_dataset_formation = 1
        tmp_dataset_pkl_name = "tmp_point_set_dataset.pkl"
        if redo_dataset_formation:
            all_data_set = self.data_set_formation(point_set, tracks_set, data_mode)
            # Save the obtained data set
            with open(tmp_dataset_pkl_name,"wb") as f:
                pickle.dump(all_data_set,f)
        else:
            # Load the obtained data set
            with open(tmp_dataset_pkl_name,"rb") as f:
                all_data_set = pickle.load(f)

        (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set) = all_data_set

        print("Data set formation Done")

        # Check task and inversion method
        if task_name in ["tides_1","tides_3"]:
            assert inversion_method == "Bayesian_Linear"

        if task_name == "tides_2":
            assert inversion_method in ["Bayesian_MCMC", "Nonlinear_Optimization"]

        if task_name in ["tides_1", "tides_3"] and inversion_method == 'Bayesian_Linear':
            # Data prior.
            invCd_set = self.real_data_uncertainty_set(point_set, data_vec_set, \
                                                            noise_sigma_set)
            ### MODEL ###

            # Design matrix.
            linear_design_mat_set_orig = self.build_G_set(point_set, offsetfields_set=offsetfields_set)
    
            #print("Design matrix set (G)\n:", linear_design_mat_set[self.test_point])
            print("Design matrix (obs) set is Done")

            if task_name == "tides_3":
                # Enumerate grounding
                # Rutford
                #enum_grounding_level = [-1.8, -1.7, -1.6, -1.5]

                if self.external_grounding_level_file is None:
                    enum_grounding_level = [-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8,-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]

                    #enum_grounding_level = [-1.80, -1.75, -1.70, -1.65, -1.60, -1.55, -1.50]
                    #enum_grounding_level = [-1.78, -1.76, -1.74, -1.72, -1.70, -1.68, -1.66, -1.64]
                    enum_grounding_level = [-1.700, -1.695, -1.690, -1.685, -1.680, -1.675, -1.670, -1.665]
                else:
                    enum_grounding_level = ['external']
                    # Currently the external grounding file is saved in ${id}_grid_set_others.pkl
                    with open(self.external_grounding_level_file,'rb') as f:
                        external_grounding_level = pickle.load(f)
            else:
                enum_grounding_level = [None]

            # Loop through the grounding level
            for ienum, grounding_level in enumerate(enum_grounding_level):
                print("New enumeration: ", ienum, grounding_level)
               
                # Inversion with enforced grounding level ("tides_3")
                if grounding_level is not None:
                    # Make a deep copy of dictionary of design matrix
                    linear_design_mat_set = copy.deepcopy(linear_design_mat_set_orig)

                    if grounding_level == 'external':
                        given_grounding_level = external_grounding_level
                    else:
                        given_grounding_level = grounding_level

                    # Modifying G matrix to add direct modeling of vertical displacement
                    linear_design_mat_set = self.modify_G_set(point_set, linear_design_mat_set, offsetfields_set, grounding_level = given_grounding_level)
                    print("Modified matrix (obs) set is Done")

                # Default inversion ("tides_1")
                else:
                     linear_design_mat_set = linear_design_mat_set_orig
    
                # Model prior.
                invCm_set = self.model_prior_set(point_set)
                print("Model prior set Done")
        
                # Model posterior (Singular matrix will come back with nan).
                Cm_p_set = self.model_posterior_set(point_set, linear_design_mat_set, invCd_set, invCm_set, test_point = self.test_point)
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
                print("Bayesian linear model: \n")
                print(bl_model_vec)
    
                # Calculale the residual.
                resid_of_secular_set, resid_of_tides_set = self.get_resid_set(point_set, linear_design_mat_set, data_vec_set, model_vec_set)
                print('Residual calculation Done')
                resid_of_tides_point = resid_of_tides_set[self.test_point]
                print("Residual at this point: ", resid_of_tides_point)

                # Calculale the model likelihood.
                model_likelihood_set = self.get_model_likelihood_set(point_set, linear_design_mat_set, data_vec_set, model_vec_set, invCd_set)
                print('Model likelihood calculation Done')
                model_likelihood_point = model_likelihood_set[self.test_point]
                print("Model likelihood at this point: ", model_likelihood_point)

                # Convert to tidal params.
                tide_vec_set = self.model_vec_set_to_tide_vec_set(point_set, model_vec_set)
                #print('tide_vec_set: ',tide_vec_set)
                print('Tide vec set Done')
    
                # Convert model posterior to uncertainty of params.
                # Require: tide_vec and Cm_p
                tide_vec_uq_set = self.model_posterior_to_uncertainty_set(point_set, tide_vec_set, Cm_p_set)
                print('Uncertainty set estimation Done')
        
                print('Point set inversion Done')
    
                ############ Some additional work ##############################
                
                print('Some additional work ...')
                if self.task_name == "tides_3":
                    # Put the vertical scaling into other_set_1
                    self.extract_up_scale_set(point_set, model_vec_set, others_set)

                    # Save the residuals
                    self.save_resid_set(point_set, resid_of_tides_set, others_set, grounding_level)

            # Select the optimal grounding level
            # If mode is tides_3 and the enumeration is actually done
            if self.task_name == "tides_3" and enum_grounding_level[0] is not None:
                print(others_set[self.test_point]['grounding_level_resids'][enum_grounding_level[0]])
                self.select_optimal_grounding_level(point_set, others_set)
                print("Optimal grounding level before scaling: ", others_set[self.test_point]["optimal_grounding_level"])
                print("The scaling is: ", model_vec_set[self.test_point][-1][0])
                print("Optimal grounding level after scaling: ", others_set[self.test_point]["optimal_grounding_level"] * model_vec_set[self.test_point][-1][0])

            ########### Show the results ########################
            # Stack the true and inverted models.
            # Show on point in the point set.

            self.show_vecs = False
            show_control = False
            if self.test_point_file:
                show_control = True
            
            if self.show_vecs == True or show_control== True:
            #if self.show_vecs == True and inversion_method=="Bayesian_Linear":
    
                if self.test_point in true_tide_vec_set:
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
       
        if task_name == "tides_2" and inversion_method in ['Bayesian_MCMC', 'Nonlinear_Optimization']:

            ## Step 1: Prepare the solver (named BMC)
            BMC = solvers(self.param_file)
            
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
            BMC.set_model_priors(model_mean_prior_set = true_tide_vec_set, no_secular_up = self.no_secular_up, up_short_period = self.up_short_period, horizontal_long_period = self.horizontal_long_period, up_lower = self.up_lower, up_upper = self.up_upper)

            # Provide data priors
            BMC.set_noise_sigma_set(noise_sigma_set)

            # Provide data
            BMC.set_data_set(data_vec_set)

            # Provide offsetfield info
            BMC.set_offsetfields_set(offsetfields_set)

            # Stack the design matrix modeling tides
            stack_design_mat_set = self.get_stack_design_mat_set(point_set, self.model_design_mat_set, offsetfields_set)

            # Provid the matrix to simulator
            BMC.set_stack_design_mat_set(stack_design_mat_set)

            # Obtain the up displacement
            up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

            # Provide the up displacement to the solver
            BMC.set_up_disp_set(up_disp_set)

            # Obtain true model vec from true tide vec
            if self.simulation_mode:
                true_model_vec_set = self.tide_vec_set_to_model_vec_set(point_set, true_tide_vec_set)
            else:
                true_model_vec_set = None
            BMC.set_true_model_vec_set(true_model_vec_set)

            # Pass the velo model to solver as well, as it contains the up_scale
            BMC.set_grid_set_velo(self.grid_set_velo)

            # Set task name
            BMC.set_task_name(task_name)

            # Run inversion
            est_grounding = None
            suffix = str(data_mode['csk'])+'_' + str(data_mode['s1'])
            if inversion_method=="Bayesian_MCMC":
                model_vec, est_grounding = BMC.run_MCMC(run_point = self.test_point, suffix = suffix)
                print("*** Result of Bayesian MCMC")
                print(model_vec)
            
            elif inversion_method=="Nonlinear_Optimization":
                model_vec_set = BMC.run_optimize(run_point = self.test_point)
                print("*** Result of nonlinear optimization")
                print(model_vec_set[self.test_point])
            else:
                raise Exception()

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
            #if bl_model_vec.shape[0]>1:
            #    bl_tide_vec = self.model_vec_to_tide_vec(bl_model_vec)

            #    stacked_model_vecs.append(bl_model_vec)
            #    stacked_tide_vecs.append(bl_tide_vec)
            #    row_names.append("Linear")

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
        all_sets['others_set'] = others_set

        return all_sets

