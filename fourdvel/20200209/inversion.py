#!/usr/bin/env python3

# Author: Minyan Zhong

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

from matplotlib import cm

import seaborn as sns

import datetime
from datetime import date

import multiprocessing

from multiprocessing import Value

import time

from scipy import linalg

from fourdvel import fourdvel
from display import display
from analysis import analysis

class inversion(fourdvel):

    def __init__(self, param_file="params.in"):

        if len(sys.argv)==2:
            param_file = sys.argv[1]

        self.param_file = param_file
        
        super(inversion,self).__init__(param_file)

        self.display = display(param_file)

        self.preparation()

    def find_track_data_set(self, point_set, vecs_set, track):

        from dense_offset import dense_offset

        track_num = track[0]
        sate = track[1]

        print('Find track data...')
        print(self.test_point, track_num, sate)

        if self.proj == "Evans":

            if sate == 'csk':
                stack = "stripmap"
                #workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"
                workdir = self.csk_workdir
                name='track_' + str(track_num).zfill(2) + str(0)
                runid = 20180712
    
            elif sate == 's1':
                stack = 'tops'
    
                #workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
                workdir = self.s1_workdir
                
                name = 'track_' + str(track_num)
                #runid = 20180703
                runid = 20200101

        elif self.proj == "Rutford":

            if sate == 'csk':
                stack = "stripmap"
                #workdir = "/net/kraken/nobak/mzzhong/CSK-Rutford"
                workdir = self.csk_workdir
                name='track_' + str(track_num).zfill(3) + '_' + str(0)

                # 64 x 128
                runid = 20190921
    
            elif sate == 's1':

                raise Exception("S1 data not ready for Rutford yet")

                stack = 'tops'
    
                #workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
                workdir = self.s1_workdir
                
                name = 'track_' + str(track_num)
                runid = 20180703
        
        else:

            raise Exception("Can't find data for project: ", self.proj)


        # Create dense offset object.
        offset = dense_offset(stack=stack, workdir=workdir, runid=runid)
        offset.initiate(trackname = name)

        print('satellite: ', sate, ' track number: ', track_num)

        # Dates allowed to use.
        if sate == 'csk':
            used_dates = self.csk_data[track_num]
        elif sate == 's1':
            used_dates = self.s1_data[track_num]

        #print('track_number: ',track_num)
        #print('used dates: ', used_dates)
 
        track_pairs_set, track_offsets_set = \
        offset.extract_offset_set_series(point_set = point_set, dates = used_dates, offsetFieldStack = self.offsetFieldStack_all[(sate, track_num)] )


        #track_pairs_set, track_offsets_set = \
        #offset.extract_offset_set_series(point_set = point_set, dates = used_dates)

        #print('obtained pairs: ', track_pairs)
        #print('obtained offsets: ', track_offsets)

        # Add observation vectors and time fraction.
        track_offsetfields_set = {}
        for point in point_set:
            track_offsetfields_set[point] = []

            # vec info is available.
            if point in vecs_set.keys():
                vec1 = vecs_set[point][0]
                vec2 = vecs_set[point][1]
                t_frac = self.track_timefraction[(sate,track_num)]
                tail = [vec1, vec2, t_frac]

                # offsetfields match the obtained offsets.
                for pair in track_pairs_set[point]:
                    offsetfield = pair + tail
                    track_offsetfields_set[point].append(offsetfield)

            # 2019.02.14
            # Cancel the obtained offsets, if no vec info is available for the point.
            else:
                track_pairs_set[point] = []
                track_offsets_set[point] = []

            #if point == (-83.74, -76.02):
            #    print('track_offsetfields at this point: ', track_offsetfields_set[point])

        return track_offsetfields_set, track_offsets_set

    def offsets_set_to_data_vec_set(self, point_set, offsets_set):
        data_vec_set = {}
        for point in point_set:
            data_vec_set[point] = self.offsets_to_data_vec(offsets_set[point])
        
        return data_vec_set

    def offsets_to_data_vec(self,offsets):
        data_vec = np.zeros(shape=(len(offsets)*2,1))
        
        for i in range(len(offsets)):
            data_vec[2*i,0] = offsets[i][0]
            data_vec[2*i+1,0] = offsets[i][1]

        return data_vec

    def load_noise_sigma(self, point):

        if self.grid_set_data_uncert is not None:
            data_uncert = self.grid_set_data_uncert[point]
            noise_sigma = (data_uncert[1], data_uncert[3])
        elif self.data_uncert_const is not None:
            noise_sigma = self.data_uncert_const
        else:
            raise Exception('No measurement error model')

        return noise_sigma

    def data_set_formation(self, point_set, tracks_set, test_mode=None):

        from simulation import simulation

        ### DATA ###
        # Modes:
        # 1. Synthetic data: projected catalog (Done)
        # 2. Synthetic data: real catalog  (Test)
        # 3. True data: real catalog (Not sure)

        # If using synthetic data, need to return true tidal parameters for comparison.

        # Deduce how many independent tracks there are.
        indep_tracks = []
        for point in point_set:
            for track in tracks_set[point]:
                indep_tracks.append((track[0],track[3]))  # only track number and satellite name.

        # Remove the repeated tracks.
        indep_tracks = sorted(list(set(indep_tracks)))

        # Print total number of tracks.
        print("Number of tracks in this point set: ", len(indep_tracks))
        print("They are: ", indep_tracks)

        if test_mode == 1:
            
            test_point = self.test_point

            # Get catalog of all offsetfields.
            data_info_set = {}
            offsetfields_set = {}
            offsets_set = {}
            for point in point_set:
                offsetfields_set[point] = []
                offsets_set[point] = []
                data_info_set[point] = []

            for point in point_set:
                tracks = tracks_set[point]

                # Create all offsetfields
                offsetfields_set[point] = self.tracks_to_full_offsetfields(tracks)
                
                # Not available
                offsets_set[point] = "test_mode is 1"

                # Not avaiable, because all tracks are derived at once
                data_info_set[point] = "test_mode is 1"

            # Synthetic data.
            fourD_sim = simulation()

            # Model parameters.
            velo_model_set = {}
            for point in point_set:
                velo_model_set[point] = self.grid_set_velo[point]

            # Obtain the synthetic ice flow.
            (secular_v_set, tide_amp_set, tide_phase_set) = fourD_sim.syn_velocity_set(
                                                            point_set = point_set, 
                                                            velo_model_set = velo_model_set)
            # Data prior.
            noise_sigma_set = {}
            for point in point_set:
                noise_sigma_set[point] = self.load_noise_sigma(point)

            # Stack the design matrix for Rutford tides
            stack_design_mat_set = self.stack_design_mat_set(point_set, self.rutford_design_mat_set, offsetfields_set)

            # Provide the matrix to simulator
            fourD_sim.set_stack_design_mat_set(stack_design_mat_set)

            # Provide grounding level
            fourD_sim.set_grounding(self.grounding)

            # Get offsets
            data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                point_set = point_set,
                                secular_v_set = secular_v_set, 
                                modeling_tides = self.modeling_tides, 
                                tide_amp_set = tide_amp_set, 
                                tide_phase_set = tide_phase_set, 
                                offsetfields_set = offsetfields_set, 
                                noise_sigma_set = noise_sigma_set)

            # True tidal params. (Every point has the value)
            true_tide_vec_set = fourD_sim.true_tide_vec_set(point_set, secular_v_set, 
                                            self.modeling_tides, tide_amp_set, tide_phase_set)

        if test_mode == 2 or test_mode == 3:

            # Use the first point for testing.
            test_point = self.test_point
            #test_point = (-83.74, -76.02)

            # Get catalog of all offsetfields.
            data_info_set = {}
            offsetfields_set = {}
            offsets_set = {}
            true_tide_vec_set = {}

            for point in point_set:
                offsetfields_set[point] = []
                offsets_set[point] = []
                data_info_set[point] = []

            # Read track one by one.
            for track in indep_tracks:

                track_num = track[0]
                sate_name = track[1]

                if sate_name == "csk" and self.use_csk == False:
                    continue

                if sate_name == "s1" and self.use_s1 == False:
                    continue

                print('=========== NEW TRACK ==============')
                print('Reading: ', track[0], track[1])

                # We know track number so we can find the observation vector set.
                vecs_set = {}
                for point in point_set:
                    all_tracks_at_this_point = tracks_set[point]

                    #if point == (-83.74, -76.02):
                    #    print('all tracks at this point: ', all_tracks_at_this_point)

                    for this_track in all_tracks_at_this_point:
                        # If this track covers this point.
                        if this_track[0] == track_num:
                            vecs_set[point] = (this_track[1], this_track[2])

                    #if point == (-83.74, -76.02):
                    #    print('vecs at this point: ', vecs_set[point])

                        # Note: not all points have vectors.

                #print(vecs_set[test_point])
                #print(stop)

                track_offsetfields_set, track_offsets_set = self.find_track_data_set(point_set, vecs_set, track)
                
                print('Available track offsetfields: ',
                        track_offsetfields_set[test_point])
                print('Obtained offsets: ', 
                            track_offsets_set[test_point])
                print('Length of offsetfields and offsets: ', 
                        len(track_offsetfields_set[test_point]),len(track_offsets_set[test_point]))

                # Point by point addition
                for point in point_set:
                    # List addition.
                    offsetfields_set[point] = offsetfields_set[point] + track_offsetfields_set[point]
                    # List addition.
                    offsets_set[point] = offsets_set[point] + track_offsets_set[point]

                    # Save the information. (# track info)
                    data_info_set[point].append([(track[0], track[1]), len(track_offsets_set[point])])


                #print('offset list length: ', len(offsets_set[(-83.74, -76.02)]))
                #print('offset field list length: ',len(offsetfields_set[(-83.74, -76.02)]))

            print('======= END OF EXTRACTION ========')
            print('Total number of offsetfields: ', len(offsetfields_set[test_point]))
            print('Total length of offsets: ', len(offsets_set[test_point]))

            # Generate synthetic data
            if test_mode == 2:

                # Synthetic data.
                fourD_sim = simulation()

                fourD_sim.test_point = self.test_point
                
                # Model parameters.
                velo_model_set = {}
                for point in point_set:
                    velo_model_set[point] = self.grid_set_velo[point]
                
                # Obtain the synthetic ice flow.
                (secular_v_set, tide_amp_set, tide_phase_set) = fourD_sim.syn_velocity_set(
                                                                point_set = point_set, 
                                                                velo_model_set = velo_model_set)

                # Data prior.
                noise_sigma_set = {}
                for point in point_set:
                    noise_sigma_set[point] = self.load_noise_sigma(point)

                # Stack the design matrix for Rutford tides
                stack_design_mat_set = self.stack_design_mat_set(point_set, self.rutford_design_mat_set, offsetfields_set)

                # Provide the matrix to simulator
                fourD_sim.set_stack_design_mat_set(stack_design_mat_set)

                # Provide grounding
                fourD_sim.set_grounding(self.grounding)

                data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                    point_set = point_set,
                                    secular_v_set = secular_v_set, 
                                    modeling_tides = self.modeling_tides, 
                                    tide_amp_set = tide_amp_set, 
                                    tide_phase_set = tide_phase_set, 
                                    offsetfields_set = offsetfields_set, 
                                    noise_sigma_set = noise_sigma_set)

                # True tidal params. (Every point has the value)
                # velocity domain m/d
                true_tide_vec_set = fourD_sim.true_tide_vec_set(point_set,secular_v_set,                                    self.modeling_tides, tide_amp_set, tide_phase_set)

            # Just use the obtained data
            elif test_mode == 3:

                # Real data
                data_vec_set = self.offsets_set_to_data_vec_set(point_set, offsets_set)

                # Data prior.
                noise_sigma_set = {}
                for point in point_set:
                    noise_sigma_set[point] = self.load_noise_sigma(point)

#                # Get reference velocity
#                n_params = 3 + len(self.modeling_tides) * 6
#                for point in point_set:
#
#                    true_tide_vec_set[point] = np.zeros(shape=(n_params, 1))
#    
#                    true_tide_vec_set[point][0] = secular_v_set[point][0]
#                    true_tide_vec_set[point][1] = secular_v_set[point][1]
#                    true_tide_vec_set[point][2] = secular_v_set[point][2]


        return (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set)

    def point_set_tides(self, point_set, tracks_set, inversion_method=None):

        ### Get data either from simulation or real data ###

        # Choose the test mode.
        test_id = self.test_id
        test_mode = self.test_mode

        # Num of points.
        n_points = len(point_set)

        # test point
        print('The test point is: ', self.test_point)

        # All variables are dictionary with point_set as the key.
        # Data set formation.
        (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set)=\
                                                    self.data_set_formation(point_set, tracks_set, test_mode)

        #print(data_info_set[self.test_point])

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
            print("***Results of Bayesian linear")
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

            #######################################################

        if inversion_method in ['Bayesian_MCMC', 'Bayesian_MCMC_Linear']:

            from solvers import Bayesian_MCMC

            BMC = Bayesian_MCMC(self.param_file)
            # Set the point to work on
            BMC.set_point_set(point_set)

            linear_design_mat_set = self.build_G_set(point_set, offsetfields_set=offsetfields_set)

            # Set linear tides
            BMC.set_linear_design_mat_set(linear_design_mat_set)

            # Set modeling tides
            BMC.set_modeling_tides(self.modeling_tides)
            
            # Provide model priors
            self.up_lower = -10
            self.up_upper = 10
            BMC.set_model_priors(model_prior_set=true_tide_vec_set, no_secular_up = self.no_secular_up, up_short_period = self.up_short_period, horizontal_long_period = self.horizontal_long_period, up_lower = self.up_lower, up_upper = self.up_upper)

            # Provide data priors
            BMC.set_noise_sigma_set(noise_sigma_set)

            # Provide data
            BMC.set_data_set(data_vec_set)

            # Provide offsetfield info
            BMC.set_offsetfields_set(offsetfields_set)

            # Stack the design matrix modeling tides
            stack_design_mat_set = self.stack_design_mat_set(point_set, self.design_mat_set, offsetfields_set)

            # Provide the matrix to simulator
            BMC.set_stack_design_mat_set(stack_design_mat_set)

            # Obtain true model vec from true tide vec
            if self.test_mode in [1,2]:

                true_model_vec_set = self.tide_vec_set_to_model_vec_set(point_set, true_tide_vec_set)
                # For test
                #true_tide_vec_set2 = self.model_vec_set_to_tide_vec_set(point_set, true_model_vec_set)
                #print(true_tide_vec_set[self.test_point])
                #print(true_tide_vec_set2[self.test_point])
                #print(stop)

            else:
                true_model_vec_set = None

            # Run inversion
            est_grounding = None
            if inversion_method=="Bayesian_MCMC":
                model_vec, est_grounding = BMC.run_MCMC(run_point = self.test_point, true_model_vec_set=true_model_vec_set, suffix=str(self.test_mode))
                print("*** Result of Bayesian MCMC")
                print(model_vec)

            elif inversion_method=="Bayesian_MCMC_Linear":
                model_vec = BMC.run_MCMC_Linear(run_point = self.test_point, true_model_vec_set = true_model_vec_set, suffix=str(self.test_mode))

                print("*** Result of Bayesian MCMC Linear")
                print(model_vec)

            print("Compare model vec")
            true_model_vec = true_model_vec_set[self.test_point]
            print(np.hstack((true_model_vec,bl_model_vec, model_vec)), est_grounding)

            print("Compare tide vec")
            true_tide_vec = true_tide_vec_set[self.test_point]
            bl_tide_vec = self.model_vec_to_tide_vec(bl_model_vec)
            tide_vec = self.model_vec_to_tide_vec(model_vec)
            print(np.hstack((true_tide_vec,bl_tide_vec, tide_vec)), est_grounding)

            # Display these two
            stacked_vecs = np.hstack((true_tide_vec, tide_vec))
            row_names = ['Input','Estimated']
            column_names = ['Secular'] + self.modeling_tides

            self.display.display_vecs(stacked_vecs, row_names, column_names, test_id)
            print(stop)

        
        ########### Inversion done ##########################


        ############ Analyze the results ##############################
        print('Start to analysis...')
        # Get continous displacement time series
        #self.continous_signal(self.test_point, tide_vec_set[self.test_point])

        # Display the misfit
        #self.analysis = analysis()
        #self.analysis.test_point = self.test_point

       # Save additional info from analysis in other_set_1
        other_set_1 = {} 
        #other_set_1 = self.analysis.check_fitting_set(point_set, data_info_set, offsetfields_set, linear_design_mat_set, design_mat_enu_set, data_vec_set, model_vec_set, tide_vec_set)


        ########### Show the results ########################
        # Stack the true and inverted models.
        # Show on point in the point set.

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


    def driver_serial_tile(self, start_tile=None, stop_tile=None, use_threading = False, all_grid_sets=None, threadId=None):
        
        # Input information.
        tile_set = self.tile_set
        grid_set = self.grid_set 
        horizontal_prior = self.horizontal_prior
        self.show_vecs = False

        # if not multi-threading:
        if use_threading == False:
            grid_set_true_tide_vec = {}
            grid_set_tide_vec = {}
            grid_set_tide_vec_uq = {}
            grid_set_resid_of_secular = {}
            grid_set_resid_of_tides = {}
            grid_set_other_1 = {}

            start_tile = 0
            stop_tile = 10**10
            self.show_vecs = True

        #print("start tile: ", start_tile, "stop tile: ",stop_tile)

        count_tile = 0
        count_run = 0

        #print("Number of tiles: ",len(tile_set))

        # Find the test point
        test_point = self.test_point

        for tile in sorted(tile_set.keys()):
            
            # Work on a particular tile.
            lon, lat = tile
            
            # Run all in serial.
            #if count_tile % self.nthreads == threadId and count_tile<10**10:

            if (count_tile >= start_tile) and (count_tile < stop_tile):

            #if (count_tile >= start_tile) and (count_tile < stop_tile) and \
            #        (test_point in tile_set[tile]):
            
            #if (count_tile >= start_tile and count_tile < stop_tile and 
            #                                            count_tile % 2 == 1):

            # Debug this tile.
            #if count_tile >= start_tile and count_tile < stop_tile and f_tile == (-81, -79):
            # Debug this tile.
            #if count_tile >= start_tile and count_tile < stop_tile and tile == (-84.0, -76.2):

            # Debug this tile for Rutford
            #if count_tile >= start_tile and count_tile < stop_tile and tile == self.float_lonlat_to_int5d((-83.0, -78.6)):

                #print("Find the tile", tile)

                #print('***  Start a new tile ***')
                #self.print_int5d([lon, lat])

                point_set = tile_set[tile] # List of tuples

                # Set test point 
                if self.test_point is None:
                    self.test_point = point_set[0]
                    test_point = self.test_point
                    #print(self.test_point)

                # Output the location and size of tile. 
                #print('tile coordinates: ', tile)
                #print('Number of points in this tile: ', len(point_set))

                # Find the union of tracks 
                # Only consider track_number and satellite which define the offsetfields.
                tracks_set = {}
                for point in point_set:
                    tracks_set[point] = grid_set[point]

                # Set the test point
                if test_point in point_set:
                    pass
                else:
                    self.test_point = point_set[0]
                    test_point = self.test_point

                # Run it
                # Default is False for recording
                recorded = False

                simple_count = True
                if simple_count == True:

                    all_sets  = self.point_set_tides(point_set = point_set, tracks_set = tracks_set, inversion_method=self.inversion_method)
    
                    # Save the results
                    # Update (only for parallel call)
                    if use_threading and self.inversion_method == 'Bayesian_Linear':
    
                        # Save the results to disk
                        point_result_folder = self.this_result_folder + '/point_result'
    
                        point_name = str(lon) + '_' + str(lat)
    
                        with open(point_result_folder + "/" + point_name + ".pkl","wb") as f:
                            pickle.dump(all_sets, f)

                        # Say that this tile is record
                        recorded = True
                        
                        # Save the results through updating dictionary manager
                        if all_sets['true_tide_vec_set'] is not None:
                            all_grid_sets['grid_set_true_tide_vec'].update(all_sets['true_tide_vec_set'])
                        
                        all_grid_sets['grid_set_tide_vec'].update(all_sets['tide_vec_set'])
    
                        all_grid_sets['grid_set_tide_vec_uq'].update(all_sets['tide_vec_uq_set'])
                   
                        all_grid_sets['grid_set_resid_of_secular'].update(all_sets['resid_of_secular_set'])
                        all_grid_sets['grid_set_resid_of_tides'].update(all_sets['resid_of_tides_set'])
                        all_grid_sets['grid_set_other_1'].update(all_sets['other_set_1'])

                    if recorded == False:
                        print("Problematic unrecorded tile", tile)
                        print(stop)

                # Count the run tiles
                count_run = count_run + 1

            # Count the total tiles
            count_tile = count_tile + 1

        self.f.write("count run: " + str(count_run))
        self.f.write("count tile: " + str(count_tile))
        print("count run: " + str(count_run))
        print("count tile: " + str(count_tile))

        return 0

    def driver_parallel_tile(self):

        # Initialization
        test_id = self.test_id

        result_folder = '/home/mzzhong/insarRoutines/estimations'
        self.result_folder = result_folder

        this_result_folder = os.path.join(result_folder, str(test_id))
        self.this_result_folder = this_result_folder

        if not os.path.exists(this_result_folder + '/point_result'):
            os.mkdir(this_result_folder+'/point_result')

        # Using multi-threads to get map view estimation.
        # make driver serial tile parallel.

        tile_set = self.tile_set

        self.grid_set_true_tide_vec = {}
        self.grid_set_tide_vec = {}
        self.grid_set_tide_vec_uq = {}

        self.grid_set_resid_of_secular = {}
        self.grid_set_resid_of_tides = {}

        self.grid_set_other_1 = {}

        self.f = open("tile_counter.txt","w")

        do_calculation = True

        if do_calculation:

            # Count the total number of tiles
            n_tiles = len(tile_set.keys())
            print('Total number of tiles: ', n_tiles)
    
            # Chop into multiple threads. 
            nthreads = 10
            self.nthreads = nthreads
            total_number = n_tiles
    
            # Only calculate the first half
            # Ad hoc control
            # 2020.01.13
            half_number = total_number//2    
            divide = self.chop_into_threads(half_number, nthreads)
            # First half
            # pass
            # Second half
            divide = divide + half_number
            print("divide: ", divide)

            # Full divide
            divide = self.chop_into_threads(total_number, nthreads)
    
            # Multithreading starts here.
            # The function to run every chunk.
            func = self.driver_serial_tile
    
            # Setup the array.
            manager = multiprocessing.Manager()
            
            all_grid_sets = {}
            all_grid_sets['grid_set_true_tide_vec'] = manager.dict()
            all_grid_sets['grid_set_tide_vec'] = manager.dict()
            all_grid_sets['grid_set_tide_vec_uq'] = manager.dict()
            all_grid_sets['grid_set_resid_of_secular'] = manager.dict()
            all_grid_sets['grid_set_resid_of_tides'] = manager.dict()
            all_grid_sets['grid_set_other_1'] = manager.dict()

            all_grid_sets['counter'] = manager.dict()
    
            jobs=[]
            for ip in range(nthreads):
                start_tile = divide[ip]
                stop_tile = divide[ip+1]
    
                # Use contiguous chunks
                p=multiprocessing.Process(target=func, args=(start_tile, stop_tile, True,
                                                        all_grid_sets, ip))
    
                # Based on modulus
                #p=multiprocessing.Process(target=func, args=(0, n_tiles, True,
                #                                        all_grid_sets, ip))
    
                jobs.append(p)
                p.start()
    
            for ip in range(nthreads):
                jobs[ip].join()

            self.f.close()
    
            # Convert the results to normal dictionary.
            # Save the results to the class.
            print("Saving the results...")
            self.grid_set_true_tide_vec = dict(all_grid_sets['grid_set_true_tide_vec'])
            self.grid_set_tide_vec = dict(all_grid_sets['grid_set_tide_vec'])
            self.grid_set_tide_vec_uq = dict(all_grid_sets['grid_set_tide_vec_uq'])
    
            self.grid_set_resid_of_secular = dict(all_grid_sets['grid_set_resid_of_secular'])
            self.grid_set_resid_of_tides = dict(all_grid_sets['grid_set_resid_of_tides'])
            self.grid_set_other_1 = dict(all_grid_sets['grid_set_other_1'])

            # Stop here
            #print(stop)
    
            ## end of dict.

        else:
            # Load the results from point_result
            print("Loading the results...")
            point_results = os.listdir(self.this_result_folder + '/point_result')

            for ip, point_pkl in enumerate(point_results):
                print(ip)
                pklfile = self.this_result_folder + '/point_result/' + point_pkl
                with open(pklfile,"rb") as f:
                    all_sets = pickle.load(f)

                if all_sets['true_tide_vec_set'] is not None:
                    self.grid_set_true_tide_vec.update(all_sets['true_tide_vec_set'])
                
                self.grid_set_tide_vec.update(all_sets['tide_vec_set'])

                self.grid_set_tide_vec_uq.update(all_sets['tide_vec_uq_set'])
       
                self.grid_set_resid_of_secular.update(all_sets['resid_of_secular_set'])
                self.grid_set_resid_of_tides.update(all_sets['resid_of_tides_set'])
                self.grid_set_other_1.update(all_sets['other_set_1'])

        ## Save the final results in dictionary manager

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_true_tide_vec.pkl', 'wb') as f:
            
            pickle.dump(self.grid_set_true_tide_vec, f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_tide_vec.pkl','wb') as f:
            pickle.dump(self.grid_set_tide_vec, f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_tide_vec_uq.pkl','wb') as f:
            pickle.dump(self.grid_set_tide_vec_uq, f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_resid_of_secular.pkl','wb') as f:
            pickle.dump(self.grid_set_resid_of_secular, f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_resid_of_tides.pkl','wb') as f:
            pickle.dump(self.grid_set_resid_of_tides, f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_other_1.pkl','wb') as f:
            pickle.dump(self.grid_set_other_1, f)

        return 0

def main():

    start_time = time.time()

    fourd_inv = inversion()

    # Tile set.
    #fourd_inv.driver_serial_tile()
    fourd_inv.driver_parallel_tile()

    print('All finished!')

    elapsed_time = time.time() - start_time
    print("Elasped time: ", elapsed_time)

    return 0

if __name__=='__main__':

    main()
