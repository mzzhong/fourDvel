#!/usr/bin/python3

# Author: Minyan Zhong
# Development starts in Feb 2020

# Configuration of the inverse problem: data_vec formation

import os
import sys
import pickle

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import datetime
from datetime import date

from fourdvel import fourdvel

from simulation import simulation

class configure(fourdvel):

    def __init__(self, param_file=None):

        if param_file:
            super(configure,self).__init__(param_file)
        else:
            raise Exception("Need to provide parameter file to class configuration")

        # Create the object for simulation
        print("Initializing simulation...")
        if self.test_mode == 1 or self.test_mode == 2:
            self.fourD_sim = simulation(param_file)
        print("Done with simulation initiation...")

    def find_track_data_set(self, point_set, vecs_set, track):

        from dense_offset import dense_offset

        track_num = track[0]
        sate = track[1]

        print('Find track data...')
        print(self.test_point, track_num, sate)

        if self.proj == "Evans":

            if sate == 'csk':
                stack = "stripmap"
                workdir = self.csk_workdir
                name='track_' + str(track_num).zfill(2) + str(0)
                runid = self.csk_id
    
            elif sate == 's1':
                stack = 'tops'
                workdir = self.s1_workdir
                name = 'track_' + str(track_num)
                runid = self.s1_id

        elif self.proj == "Rutford":

            if sate == 'csk':
                stack = "stripmap"
                workdir = self.csk_workdir
                name='track_' + str(track_num).zfill(3) + '_' + str(0)
                runid = self.csk_id
    
            elif sate == 's1':

                # TODO
                raise Exception("S1 data not ready for Rutford yet")

                stack = 'tops'
                workdir = self.s1_workdir
                name = 'track_' + str(track_num)
                runid = self.s1_id
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
        offset.extract_offset_set_series(point_set = point_set, test_point = self.test_point, dates = used_dates, offsetFieldStack = self.offsetFieldStack_all[(sate, track_num)], sate=sate, track_num=track_num)

        #track_pairs_set, track_offsets_set = \
        #offset.extract_offset_set_series(point_set = point_set, dates = used_dates)

        #print('obtained pairs: ', track_pairs)
        #print('obtained offsets: ', track_offsets)

        # Based on track_pairs_set, add observation vectors and time fractions to get track_offsetfields_set
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

    def load_simulation_noise_sigma(self, point):

        simulation_noise_sigma = self.simulation_data_uncert_const
        return simulation_noise_sigma

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
            fourD_sim = self.fourD_sim

            # Model parameters.
            velo_model_set = {}
            for point in point_set:
                velo_model_set[point] = self.grid_set_velo[point]

            # Obtain the synthetic ice flow.
            (secular_v_set, tide_amp_set, tide_phase_set) = fourD_sim.syn_velocity_set(
                                                            point_set = point_set, 
                                                            velo_model_set = velo_model_set)
            # Data prior.
            simulation_noise_sigma_set = {}
            noise_sigma_set = {}
            for point in point_set:
                simulation_noise_sigma_set[point] = self.load_simulation_noise_sigma(point)
                noise_sigma_set[point] = self.load_noise_sigma(point)

            # Find the stack of design matrix for Simulation (Rutford) tides
            stack_design_mat_set = self.get_stack_design_mat_set(point_set, self.rutford_design_mat_set, offsetfields_set)

            # Provide the matrix to the simulator
            fourD_sim.set_stack_design_mat_set(stack_design_mat_set)

            # Find the stack of up displacement
            # Use it for simulation
            up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

            # Provide the up displacement to the simulator
            fourD_sim.set_up_disp_set(up_disp_set)

            # Get offsets
            data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                point_set = point_set,
                                secular_v_set = secular_v_set, 
                                modeling_tides = self.modeling_tides, 
                                tide_amp_set = tide_amp_set, 
                                tide_phase_set = tide_phase_set, 
                                offsetfields_set = offsetfields_set, 
                                noise_sigma_set = simulation_noise_sigma_set)

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

                # Exclude the data required by the user
                if sate_name == "csk" and self.use_csk == False:
                    continue

                if sate_name == "s1" and self.use_s1 == False:
                    continue

                if sate_name == "csk" and track_num in self.csk_excluded_tracks:
                    continue

                if sate_name == "s1" and track_num in self.s1_excluded_tracks:
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

                # Point by point adding the data from the new track
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
                fourD_sim = self.fourD_sim
                
                # Model parameters.
                velo_model_set = {}
                for point in point_set:
                    velo_model_set[point] = self.grid_set_velo[point]
                
                # Obtain the synthetic ice flow.
                (secular_v_set, tide_amp_set, tide_phase_set) = fourD_sim.syn_velocity_set(
                                                                point_set = point_set, 
                                                                velo_model_set = velo_model_set)

                # Data prior.
                simulation_noise_sigma_set = {}
                noise_sigma_set = {}
                for point in point_set:
                    simulation_noise_sigma_set[point] = self.load_simulation_noise_sigma(point)
                    noise_sigma_set[point] = self.load_noise_sigma(point)

                # Stack the design matrix for Rutford tides
                stack_design_mat_set = self.get_stack_design_mat_set(point_set, self.rutford_design_mat_set, offsetfields_set)

                # Provide the matrix to simulator
                fourD_sim.set_stack_design_mat_set(stack_design_mat_set)

                # Extract the stack of up displacement from the tide model
                up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

                # Provide the up displacement to the simulator
                fourD_sim.set_up_disp_set(up_disp_set)

                # Form data vector
                data_vec_set = fourD_sim.syn_offsets_data_vec_set(
                                    point_set = point_set,
                                    secular_v_set = secular_v_set, 
                                    modeling_tides = self.modeling_tides, 
                                    tide_amp_set = tide_amp_set, 
                                    tide_phase_set = tide_phase_set, 
                                    offsetfields_set = offsetfields_set, 
                                    noise_sigma_set = simulation_noise_sigma_set)

                # True tidal params. (Every point has the value)
                # velocity domain m/d
                true_tide_vec_set = fourD_sim.true_tide_vec_set(point_set,secular_v_set,self.modeling_tides, tide_amp_set, tide_phase_set)

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
            else:
                raise Exception("Test mode error " + str(test_mode))

        return (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set)
