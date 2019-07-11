#!/usr/bin/env python3

# Author: Minyan Zhong

import numpy as np
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
from analysis import analysis

class inversion(fourdvel):

    def __init__(self):
        
        super(inversion,self).__init__()

        self.display = display()

        self.preparation()

    def find_track_data_set(self, point_set, vecs_set, track):

        from dense_offset import dense_offset

        track_num = track[0]
        sate = track[1]

        print('Find track data...')
        print(self.test_point, track_num, sate)

        if sate == 'csk':
            stack = "stripmap"
            workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"
            name='track_' + str(track_num).zfill(2) + str(0)
            runid = 20180712

        elif sate == 's1':
            stack = 'tops'

            workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
            
            #workdir = '/net/jokull/bak/mzzhong/S1-Evans'

            name = 'track_' + str(track_num)
            runid = 20180703

        # Create dense offset object.
        offset = dense_offset(stack=stack, workdir=workdir)
        offset.initiate(trackname = name, runid=runid)

        print('satellite: ', sate, ' track number: ', track_num)

        # Dates allowed to use.
        if sate == 'csk':
            used_dates = self.csk_data[track_num]
        elif sate == 's1':
            used_dates = self.s1_data[track_num]

        #print('track_number: ',track_num)
        #print('used dates: ', used_dates)
 
        track_pairs_set, track_offsets_set = offset.extract_offset_set_series(point_set = point_set, dates = used_dates)

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
            # Cancel the obtained offsets, if no vec info is available.
            else:
                track_pairs_set[point] = []
                track_offsets_set[point] = []

            if point == (-83.74, -76.02):
                print('track_offsetfields at this point: ', track_offsetfields_set[point])

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
        print('Number of tracks in this point set: ', len(indep_tracks))
        print(indep_tracks)

        #print(stop)

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

            #print(noise_sigma_set[self.test_point])
            #print(self.modeling_tides)
            #print(offsetfields_set[self.test_point])
            #print(stop)

            data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                point_set = point_set,
                                secular_v_set = secular_v_set, 
                                modeling_tides = self.modeling_tides, 
                                tide_amp_set = tide_amp_set, 
                                tide_phase_set = tide_phase_set, 
                                offsetfields_set = offsetfields_set, 
                                noise_sigma_set = noise_sigma_set)

            # Data prior.
            invCd_set = self.real_data_uncertainty_set(point_set, data_vec_set, 
                                                            noise_sigma_set)
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
            for point in point_set:
                offsetfields_set[point] = []
                offsets_set[point] = []
                data_info_set[point] = []

            # Read track one by one.
            for track in indep_tracks:

                track_num = track[0]

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

                # Point by point addtion
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

            if test_mode == 2:

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

                #print(noise_sigma_set[test_point])
                #print(stop)
 
                data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                    point_set = point_set,
                                    secular_v_set = secular_v_set, 
                                    modeling_tides = self.modeling_tides, 
                                    tide_amp_set = tide_amp_set, 
                                    tide_phase_set = tide_phase_set, 
                                    offsetfields_set = offsetfields_set, 
                                    noise_sigma_set = noise_sigma_set)

                # Data prior.
                invCd_set = self.real_data_uncertainty_set(point_set, data_vec_set, 
                                                                noise_sigma_set)
                # True tidal params. (Every point has the value)
                true_tide_vec_set = fourD_sim.true_tide_vec_set(point_set,secular_v_set,                                    self.modeling_tides, tide_amp_set, tide_phase_set)

            elif test_mode == 3:

                # Real data
                data_vec_set = self.offsets_set_to_data_vec_set(point_set, offsets_set)

                # Data prior.
                noise_sigma_set = {}
                for point in point_set:
                    noise_sigma_set[point] = self.load_noise_sigma(point)

                #print(noise_sigma_set[test_point])
                #print(stop)
                    
                invCd_set = self.real_data_uncertainty_set(point_set, data_vec_set, 
                                                                noise_sigma_set)

                # Get reference velocity
                n_params = 3 + len(self.modeling_tides) * 6
                for point in point_set:

                    true_tide_vec_set[point] = np.zeros(shape=(n_params, 1))
    
                    true_tide_vec_set[point][0] = secular_v_set[point][0]
                    true_tide_vec_set[point][1] = secular_v_set[point][1]
                    true_tide_vec_set[point][2] = secular_v_set[point][2]


        return (data_info_set, data_vec_set, invCd_set, offsetfields_set, true_tide_vec_set)

    def point_set_tides(self, point_set, tracks_set, inversion_method=None):

        ### Data ###
        # Choose the test mode.
        test_id = self.test_id
        test_mode = self.test_mode

        # Num of points.
        n_points = len(point_set)

        # test point
        print('The test point is: ', self.test_point)

        # All variables are dictionary with point_set as the key.
        # Data set formation.
        (data_info_set, data_vec_set, invCd_set, offsetfields_set, true_tide_vec_set) = \
                                self.data_set_formation(point_set, tracks_set, test_mode)

        #print(data_info_set[self.test_point])

        print("Data set formation Done")


        if inversion_method == 'Bayesian_Linear':

            from solvers import Bayesian_Linear

            BL = Bayesian_Linear()

            ### MODEL ###
            # Design matrix.
            design_mat_set = self.build_G_set(point_set, offsetfields_set=offsetfields_set)
    
            #print("Design matrix set (G)\n:", design_mat_set[self.test_point])
            print("Design matrix (obs) set Done")
    
            #design_mat_enu_set = self.build_G_ENU_set(point_set, offsetfields_set=offsetfields_set)
    
            #print("Design matrix set (G)\n:", design_mat_set[self.test_point])
            print("Design matrix (enu) set Done")
     
            # Model prior.
            invCm_set = self.model_prior_set(point_set, horizontal = self.horizontal_prior)
            print("Model prior set Done")
    
            # Model posterior (Singular matrix will come back with nan).
            Cm_p_set = self.model_posterior_set(point_set, design_mat_set, invCd_set, invCm_set)
            #print('Model posterior: ',Cm_p_set[self.test_point])
            print("Model posterior set Done")
    
            # Show the model posterior.
            #self.display.show_model_mat(Cm_p_set[self.test_point])
    
            ### Inversion ###
            # Estimate model params.
            model_vec_set = self.param_estimation_set(point_set, design_mat_set, data_vec_set, invCd_set, invCm_set, Cm_p_set)
            print('Model vec set estimation Done')
    
            # Calculale the residual.
            resid_of_secular_set, resid_of_tides_set = self.resids_set(point_set, design_mat_set, data_vec_set, model_vec_set)
            print('Residual calculation Done')
    
            # Convert to tidal params.
            tide_vec_set = self.model_vec_set_to_tide_vec_set(point_set, model_vec_set)
            print('Tide vec set Done')
    
            # Convert model posterior to uncertainty of params.
            # Require: tide_vec and Cm_p
            tide_vec_uq_set = self.model_posterior_to_uncertainty_set(point_set, tide_vec_set, Cm_p_set)
            print('Uncertainty set estimation Done')
    
            print('Point set inversion Done')

            print('Start to analysis...')
            # Get continous displacement time series
            #self.continous_signal(self.test_point, tide_vec_set[self.test_point])

            # Display the misfit
            #self.analysis = analysis()
            #self.analysis.test_point = self.test_point

            # Save additional info from analysis in other_set_1
            other_set_1 = {} 
            #other_set_1 = self.analysis.check_fitting_set(point_set, data_info_set, offsetfields_set, design_mat_set, design_mat_enu_set, data_vec_set, model_vec_set, tide_vec_set)

        elif inversion_method=='Bayesian_MCMC':

            from solvers import Bayesian_MCMC

            BMC = Bayesian_MCMC()

            # Set the point to work on
            BMC.set_point_set(point_set)

            # Set modeling tides
            BMC.set_modeling_tides(self.modeling_tides)
            
            # Provide model priors
            BMC.set_model_priors(model_prior=true_tide_vec_set, no_secular_up = self.no_secular_up, up_short_period = self.up_short_period, horizontal_long_period = self.horizontal_long_period)

            # Provide data priors
            BMC.set_data_covariance(invCd_set)

            # Provide data
            BMC.set_data(data_vec_set)

            # Provide offsetfield info
            BMC.set_offsetfields(offsetfields_set)

            # Provide design_mat_set
            BMC.set_design_mat(self.design_mat_set)
            
            # Run inversion
            model_vec_set = BMC.run()

            print(stop)

            # Get the result
            #all_sets = BMC.get_model_vec_set()
            
            #model_vec_set = all_sets['model_vec_set']
            #grounding_set = all_sets['grouding_set']

            # Convert to tidal params.
            #tide_vec_set = self.model_vec_set_to_tide_vec_set(point_set, model_vec_set)
 
        else:
            raise Exception('Please choose a inversion method')


        ##### Show the results ####
        # Stack the true and inverted models.
        # Show on point in the point set.

        if self.show_vecs == True:

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
        all_sets = {}
        all_sets['true_tide_vec_set'] =  true_tide_vec_set
        all_sets['tide_vec_set'] = tide_vec_set

        all_sets['tide_vec_uq_set'] = tide_vec_uq_set
        all_sets['resid_of_secular_set'] = resid_of_secular_set
        all_sets['resid_of_tides_set'] = resid_of_tides_set
        all_sets['other_set_1'] = other_set_1

        return all_sets

    def chop_into_threads(self, total_number, nthreads):

        # Devide chunk size.
        mod = total_number % nthreads        
        if mod > 0:
            chunk_size = (total_number - mod + nthreads) // nthreads
        else:
            chunk_size = total_number // nthreads

        # Deduce divides.
        divide = np.zeros(shape=(nthreads+1,))
        divide[0] = 0

        for it in range(1, nthreads+1):
            divide[it] = chunk_size * it
        divide[nthreads] = total_number

        return divide

    def driver_serial_tile(self, start_tile=None, stop_tile=None, use_threading = False, all_grid_sets=None):
        
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

        print("start tile: ", start_tile, "stop tile: ",stop_tile)

        count_tile = 0
        count_run = 0

        #print(len(tile_set))
        #print(stop)

        for tile in sorted(tile_set.keys()):
            # Work on a particular tile.
            lon, lat = tile
            
            # Run all in serial.
            #if (count_tile >= start_tile and count_tile < stop_tile): 
            
            #if (count_tile >= start_tile and count_tile < stop_tile and 
            #                                            count_tile % 2 == 1):

            # Only run this example tile.
            if count_tile >= start_tile and count_tile < stop_tile and tile == (-77, -76.6):

            # Debug this tile.
            #if count_tile >= start_tile and count_tile < stop_tile and tile == (-84.0, -76.2):

                print('***  Start a new tile ***')
                
                point_set = tile_set[tile] # List of tuples

                # Output the location and size of tile. 
                print('tile coordinates: ', tile)
                print('Number of points in this tile: ', len(point_set))

                # Find the union of tracks 
                # Only consider track_number and satellite which define the offsetfields.
                tracks_set = {}
                for point in point_set:
                    tracks_set[point] = grid_set[point]
                
                self.test_point = point_set[0] 
                
                # Inversion happens here
                self.inversion_method = 'Bayesian_MCMC'
                
                all_sets  = self.point_set_tides(point_set = point_set, tracks_set = tracks_set, inversion_method=self.inversion_method)

                # Save the results
                # Update (only for parallel call)

                if use_threading and self.inversion_method == 'Bayesian_Linear':

                    if all_sets['true_tide_vec_set'] is not None:
                        all_grid_sets['grid_set_true_tide_vec'].update(all_sets['true_tide_vec_set'])
                    
                    all_grid_sets['grid_set_tide_vec'].update(all_sets['tide_vec_set'])

                    all_grid_sets['grid_set_tide_vec_uq'].update(all_sets['tide_vec_uq_set'])
               
                    all_grid_sets['grid_set_resid_of_secular'].update(all_sets['resid_of_secular_set'])
                    all_grid_sets['grid_set_resid_of_tides'].update(all_sets['resid_of_tides_set'])
                    all_grid_sets['grid_set_other_1'].update(all_sets['other_set_1'])

                # Count the run tiles
                count_run = count_run + 1

            # Count the total tiles
            count_tile = count_tile + 1

        return 0

    def driver_parallel_tile(self):
        # Using multi-threads to get map view estimation.
        # make driver serial tile parallel.

        tile_set = self.tile_set

        self.grid_set_true_tide_vec = {}
        self.grid_set_tide_vec = {}
        self.grid_set_tide_vec_uq = {}

        self.grid_set_resid_of_secular = {}
        self.grid_set_resid_of_tides = {}

        self.grid_set_others = {}

        # Count the total number of tiles
        n_tiles = len(tile_set.keys())
        print('Total number of tiles: ', n_tiles)

        # Chop into multiple threads. 
        nthreads = 10
        total_number = n_tiles
        print(total_number)
        print(stop)

        divide = self.chop_into_threads(total_number, nthreads)
        print(divide)

        # Multithreading starts here.
        # The function to run every chunk.
        func = self.driver_serial_tile

        # Setup the array.
        manager = multiprocessing.Manager()
        
        #grid_set_true_tide_vec = manager.dict()
        #grid_set_tide_vec = manager.dict()
        #grid_set_tide_vec_uq = manager.dict()
        #grid_set_resid_of_secular = manager.dict()
        #grid_set_resid_of_tides = manager.dict()
        #grid_set_other_1 = manager.dict()

        all_grid_sets = {}
        all_grid_sets['grid_set_true_tide_vec'] = manager.dict()
        all_grid_sets['grid_set_tide_vec'] = manager.dict()
        all_grid_sets['grid_set_tide_vec_uq'] = manager.dict()
        all_grid_sets['grid_set_resid_of_secular'] = manager.dict()
        all_grid_sets['grid_set_resid_of_tides'] = manager.dict()
        all_grid_sets['grid_set_other_1'] = manager.dict()

        
        jobs=[]
        for ip in range(nthreads):
            start_tile = divide[ip]
            stop_tile = divide[ip+1]
            p=multiprocessing.Process(target=func, args=(start_tile, stop_tile, True,
                                                    all_grid_sets))
            jobs.append(p)
            p.start()

        for ip in range(nthreads):
            jobs[ip].join()

        # Save the results.
        self.grid_set_true_tide_vec = dict(all_grid_sets['grid_set_true_tide_vec'])
        self.grid_set_tide_vec = dict(all_grid_sets['grid_set_tide_vec'])
        self.grid_set_tide_vec_uq = dict(all_grid_sets['grid_set_tide_vec_uq'])

        self.grid_set_resid_of_secular = dict(all_grid_sets['grid_set_resid_of_secular'])
        self.grid_set_resid_of_tides = dict(all_grid_sets['grid_set_resid_of_tides'])
        self.grid_set_other_1 = dict(all_grid_sets['grid_set_other_1'])

        ## end of dict.

        test_id = self.test_id
        result_folder = '/home/mzzhong/insarRoutines/estimations'

        this_result_folder = os.path.join(result_folder, str(test_id))
        if not os.path.exists(this_result_folder):
            os.mkdir(this_result_folder)

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

    fourd_inv = inversion()

    # Tile set.
    fourd_inv.driver_serial_tile()
    #fourd_inv.driver_parallel_tile()

    print('All finished!')

    return 0

if __name__=='__main__':

    main()


#    def point_tides(self, point, tracks):
#
#        ### Data ###
#        # Choose the test mode.
#        test_id = self.test_id
#        test_mode = self.test_mode
#
#        # Data formation.
#        (data_vec, invCd, offsetfields, true_tide_vec) = self.data_formation(point, tracks, test_mode)
#
#        #self.data_vec_mode_1 = data_vec
#
#        ### MODEL ###
#        # Design matrix.
#        design_mat = self.build_G(offsetfields=offsetfields)
#        #print("Design matrix (G)\n:", design_mat)
# 
#        # Model prior.
#        invCm = self.model_prior(horizontal = self.horizontal_prior)
#
#        # Model posterior.
#        Cm_p = self.model_posterior(design_mat, invCd, invCm)
#        #print(Cm_p[0,0])
#        #print(design_mat[0,0])
#        #print(stop)
#
#        #print('Model posterior: ',Cm_p)
#
#        # Show the model posterior.
#        #self.display.show_model_mat(Cm_p)
#
#        #self.model_analysis(design_mat = design_mat, model_prior = invCm)
#
#        ### Inversion ###
#        # Estimate model params.
#        model_vec = self.param_estimation(design_mat, data_vec, invCd, invCm, Cm_p)
#
#        # Calculate and show the residuals.
#        resid_of_secular = self.resid_of_secular(design_mat, data_vec, model_vec)
#        self.display.show_resid_dist(resid_of_secular, label=str(test_id)+'secular')
#
#        resid_of_tides = self.resid_of_tides(design_mat, data_vec, model_vec)
#        self.display.show_resid_dist(resid_of_tides, label=str(test_id)+'tides')
#
#        # Convert to tidal params.
#        tide_vec = self.model_vec_to_tide_vec(model_vec)
#
#        # Convert model posterior to uncertainty of params.
#        # Require: tide_vec and Cm_p
#        tide_vec_uq = self.model_posterior_to_uncertainty(tide_vec, Cm_p)
#
#        #print(tide_vec_uq)
#        #print(stop)
#
#        print('Inversion Done')
#        
#        ##### Show the results ####
#        # Stack the true and inverted models.
#        if true_tide_vec is not None:
#            stacked_vecs = np.hstack((true_tide_vec, tide_vec, tide_vec_uq))
#            row_names = ['Input','Estimated','Uncertainty']
#            column_names = ['Secular'] + self.modeling_tides
#        else:
#            stacked_vecs = np.hstack((tide_vec, tide_vec_uq))
#            row_names = ['Estimated','Uncertainty']
#            column_names = ['Secular'] + self.modeling_tides
#
#        # Always save the results.
#        self.display.display_vecs(stacked_vecs, row_names, column_names, test_id)
#
#        return 0

#    def driver_mp(self):
#        # Using multi-threads to get map view estimation.
#
#        # Calcuate design matrix and do estimations.
#        # Always redo it.
#        redo = 1
#
#        # est_dict pickle file.
#        est_dict_pkl = self.est_dict_name + '.pkl'
#
#        # The minimum number of tracks.
#        min_tracks = 2
#
#        if redo or not os.path.exists(est_dict_pkl):
#
#            # Load the grid point set.
#            grid_set = self.grid_set
#
#            # Remove the bad grid points where not enough tracks are available.
#            bad_keys=[]
#            for key in grid_set.keys():
#                if len(grid_set[key])<min_tracks:
#                    bad_keys.append(key)
#
#            for key in bad_keys:
#                if key in grid_set:
#                    del grid_set[key]
#
#            # Count the total number of grid points.
#            total_number = len(grid_set.keys())
#            print(total_number)
#
#            # Chop into multiple threads. 
#            nthreads = 16
#            
#            mod = total_number % nthreads
#            
#            if mod > 0:
#                chunk_size = (total_number - mod + nthreads) // nthreads
#            else:
#                chunk_size = total_number // nthreads
#
#            divide = np.zeros(shape=(nthreads+1,))
#            divide[0] = 0
#
#            for it in range(1, nthreads+1):
#                divide[it] = chunk_size * it
#            divide[nthreads] = total_number
#
#            print(divide)
#            print(len(divide))
#
#            # Multithreading starts here.
#            # The function to run every chunk.
#            func = self.chunk_run
#
#            manager = multiprocessing.Manager()
#            grid_est = manager.dict()
#
#            jobs=[]
#            for ip in range(nthreads):
#                start = divide[ip]
#                end = divide[ip+1]
#                p=multiprocessing.Process(target=func, args=(start,end,grid_est,))
#                jobs.append(p)
#                p.start()
#
#            for ip in range(nthreads):
#                jobs[ip].join()
#
#            est_dict = dict(grid_est)
#
#            # Save the results.    
#            with open(os.path.join(self.design_mat_folder, est_dict_pkl),'wb') as f:
#                pickle.dump(est_dict,f)
#
#        else:
#
#            # Load the pre-computed results.
#            with open(os.path.join(self.design_mat_folder,est_dict_pkl),'rb') as f:
#                est_dict = pickle.load(f)
#
#        # Show the estimation.
#        self.show_est(est_dict)


#    def chunk_run(self,start,end,grid_est):
#
#        grid_set = self.grid_set
#        
#        count = 0
#        for grid in grid_set.keys():
#            # Output the percentage of completeness.
#            if count % 500 == 0 and start==0 and count<end:
#                print(count/end)
#
#            if count>=start and count<end:
#                est_value = self.model_analysis(point=grid, tracks=grid_set[grid])
#                grid_est[grid] = est_value
#            count = count + 1


#    def driver_serial(self):
#        
#        # Tile by tile.
#        tile_set = self.tile_set
#        grid_set = self.grid_set 
#
#        horizontal_prior = self.horizontal_prior
#
#        for tile in tile_set.keys():
#
#            if tile == (-77, -76.6):
#
#                for point in tile_set[tile]:
#    
#                    # Find the tracks.
#                    tracks = grid_set[point]
#    
#                    # Obtain the real data.
#                    self.point_tides(point = point, tracks = tracks)
#
#                    break


#    def data_formation(self, point, tracks, test_mode=None):
#
#        from simulation import simulation
#
#        ### DATA ###
#        # Modes:
#        # 1. Synthetic data: projected catalog (Done)
#        # 2. Synthetic data: real catalog  (Test)
#        # 3. True data: real catalog (Not sure)
#
#        # If using synthetic data, need to return true tidal parameters for comparison.
#
#        if test_mode == 1:
#
#            offsetfields = self.tracks_to_full_offsetfields(tracks)
#
#            # Now find offsets.
#            # Synthetic data.
#            fourD_sim = simulation()
#            velo_model = self.grid_set_velo[point]
#            # Obtain the synthetic ice flow.
#            (secular_v, tide_amp, tide_phase) = fourD_sim.syn_velocity(velo_model=velo_model)
#
#            # Obtain SAR data.
#            # Data prior.
#            ## Use external noise model.
#            noise_sigma = self.load_noise_sigma(point)
#
#
#            data_vec = fourD_sim.syn_offsets_data_vec(point=point, secular_v = secular_v, tide_amp = tide_amp, tide_phase = tide_phase, modeling_tides = self.modeling_tides, offsetfields = offsetfields, noise_sigma = noise_sigma)
#                
#            #print("Data vector (G)\n", data_vec)
#
#            # Data prior.
#            invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#            # True tidal params.
#            true_tide_vec = fourD_sim.true_tide_vec(secular_v, self.modeling_tides, tide_amp, tide_phase)
#
#        if test_mode == 2 or test_mode == 3:
#
#            # Get catalog of real offsetfields.
#            offsetfields = []
#            offsets = []
#            for track in tracks:
#                track_offsetfields, track_offsets = self.find_track_data(point, track)
#    
#                print('Available track offsetfields: ',track_offsetfields)
#                print('Obtained offsets: ', track_offsets)
#                print('Length of offsetfields and offsets: ', len(track_offsetfields),len(track_offsets))
#    
#                offsetfields = offsetfields + track_offsetfields
#                offsets = offsets + track_offsets
#
#            print('======= END OF EXTRACTION ========')
#            print('Total number of offsetfields: ', len(offsetfields))
#            print('Total length of offsets: ', len(offsets))
#
#            if test_mode == 2:
#
#                # Synthetic data.
#                fourD_sim = simulation()
#                # Obtain the synthetic ice flow.
#                velo_model = self.grid_set_velo[point]
#                (secular_v, tide_amp, tide_phase) = fourD_sim.syn_velocity(velo_model = velo_model)
#
#                # Data prior.
#                ## Use external noise model.
#                noise_sigma = self.load_noise_sigma(point)
#
#
#                data_vec = fourD_sim.syn_offsets_data_vec(point=point, secular_v = secular_v, tide_amp = tide_amp, tide_phase = tide_phase, modeling_tides = self.modeling_tides, offsetfields = offsetfields, noise_sigma = noise_sigma)
#
#                 # Data prior.
#                invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#                # True tidal params.
#                true_tide_vec = fourD_sim.true_tide_vec(secular_v, self.modeling_tides, tide_amp, tide_phase) 
#    
#            elif test_mode == 3:
#
#                # Flatten offsets to be a data vector.
#                data_vec = self.offsets_to_data_vec(offsets)
#    
#                # Data prior.
#                # Pre-assigned.
#
#                ## Use external noise model.
#                noise_sigma = self.load_noise_sigma(point)
#
#                #print(noise_sigma)
#                #print(stop)
#
#                invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#                # No true tidal parameters.
#                true_tide_vec = None
#
#        return (data_vec, invCd, offsetfields, true_tide_vec)

#    def find_track_data(self, point, track):
#
#        from dense_offset import dense_offset
#
#        track_num = track[0]
#        sate = track[3]
#
#        print('Find track data') 
#        print(point, track_num, sate)
#
#        if sate == 'csk':
#            stack = "stripmap"
#            workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"
#            name='track_' + str(track_num).zfill(2) + str(0)
#            runid = 20180712
#
#        elif sate == 's1':
#            stack = 'tops'
#            workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
#            name = 'track_' + str(track_num)
#            runid = 20180703
#
#        # Create dense_offset object
#        offset = dense_offset(stack=stack, workdir=workdir)
#        offset.initiate(trackname = name, runid=runid)
#
#        print('satellite: ', sate, ' track number: ', track_num)
#
#        # Dates allowed to use.
#        if sate == 'csk':
#            used_dates = self.csk_data[track_num]
#        elif sate == 's1':
#            used_dates = self.s1_data[track_num]
#
#        #print('track_number: ',track_num)
#        #print('used dates: ', used_dates)
#        
#        track_pairs, track_offsets = offset.extract_offset_series(point=point, dates = used_dates)
#
#        #print('obtained pairs: ', track_pairs)
#        #print('obtained offsets: ', track_offsets)
#
#        # Add observation vectors and time fraction.
#        track_offsetfields = []
#        
#        vec1 = track[1]
#        vec2 = track[2]
#        t_frac = self.track_timefraction[(sate,track_num)]
#        tail = [vec1, vec2, t_frac]
#
#        for pair in track_pairs:
#            offsetfield = pair + tail
#            track_offsetfields.append(offsetfield)
#
#        return track_offsetfields, track_offsets

