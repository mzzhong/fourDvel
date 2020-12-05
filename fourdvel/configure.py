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
import collections

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
        if self.simulation_mode:
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

    def load_simulation_noise_sigma_const(self):
        self.simulation_noise_sigma_const_dict = {}
        self.simulation_noise_sigma_const_dict['csk']= self.csk_simulation_data_uncert_const
        self.simulation_noise_sigma_const_dict['s1']= self.s1_simulation_data_uncert_const
        
    def load_noise_sigma_const(self):
        self.noise_sigma_const_dict = {}
        self.noise_sigma_const_dict['csk']= self.csk_data_uncert_const
        self.noise_sigma_const_dict['s1']= self.s1_data_uncert_const

    def load_noise_sigma_external(self, point):
        ## TODO ##
        if self.grid_set_data_uncert is not None:
            data_uncert = self.grid_set_data_uncert[point]
            noise_sigma = (data_uncert[1], data_uncert[3])
        else:
            raise Exception('No measurement error model')

    def data_set_formation(self, point_set, tracks_set, data_mode=None):
        ### DATA Modes ###
        # 1. Synthetic data: Based on catalog
        # 2. Synthetic data: Based on real data
        # 3. Real data

        # If using synthetic data, need to return true tidal parameters for comparison.

        # Deduce how many independent tracks there are in the point set.
        indep_tracks = []
        for point in point_set:
            for track in tracks_set[point]:
                indep_tracks.append((track[0],track[3]))  # only track number and satellite name.

        # Remove the repeated tracks.
        indep_tracks = sorted(list(set(indep_tracks)))

        # Show the information of tracks.
        print("Number of tracks in this point set: ", len(indep_tracks))
        print("They are: ", indep_tracks)

        # Find the unique satellite names
        sate_names = [track[1] for track in indep_tracks]
        sate_names = sorted(list(set(sate_names)))
        print("unique satelite: ", sate_names)

        # Load the noise model (only const noise for now)
        if self.simulation_mode:
            self.load_simulation_noise_sigma_const()
        self.load_noise_sigma_const()

        # Prepare the dictionary
        data_info_set_dict = {}
        data_vec_set_dict = {}
        noise_sigma_set_dict = {}
        offsetfields_set_dict = {}
        true_tide_vec_set_dict = {}

        # Look through different satellites: csk, s1
        for sate_name in sate_names:
            this_data_mode = data_mode[sate_name]

            if this_data_mode == 1:
                print("######################################")
                print("Start to create catalog-based data...")

                test_point = self.test_point
    
                # Get catalog of all offsetfields.
                data_info_set = collections.defaultdict(list)
                offsetfields_set = collections.defaultdict(list)
                offsets_set = collections.defaultdict(list)

                test_point_sate_tracks = [track for track in tracks_set[self.test_point] if track[3] == sate_name]
                print("Date mode 1, Test point tracks: ", self.test_point, test_point_sate_tracks)
   
                for point in point_set:
                    # Only choose the tracks of this satellite
                    this_sate_tracks = [track for track in tracks_set[point] if track[3] == sate_name]

                    # Create all offsetfields
                    offsetfields_set[point] = self.tracks_to_full_offsetfields(this_sate_tracks)
                    
                    # Not available
                    offsets_set[point] = "data_mode is 1"
    
                    # Not avaiable, because all tracks are derived at once
                    data_info_set[point] = "data_mode is 1"
    
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
                # Noise model.
                simulation_noise_sigma_set = {}
                noise_sigma_set = {}
                for point in point_set:
                    simulation_noise_sigma_set[point] = self.simulation_noise_sigma_const_dict[sate_name]
                    noise_sigma_set[point] = [self.noise_sigma_const_dict[sate_name]]*len(offsetfields_set[point])
    
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

                print('########### Done with creating catalog-based synthetic data ############')
                print('Total number of offsetfields at test point: ', len(offsetfields_set[test_point]))
                print('Total number of offset measurement at test_point: ', len(data_vec_set[test_point]))
    
            elif this_data_mode == 2 or this_data_mode == 3:
                # Use the first point for testing.
                test_point = self.test_point
                #test_point = (-83.74, -76.02)
    
                # Get catalog of all offsetfields.
                data_info_set = collections.defaultdict(list)
                offsetfields_set = collections.defaultdict(list)
                offsets_set = collections.defaultdict(list)
                offsetsVar_set = collections.defaultdict(list)

                # Default value, because data_mode(3) doesn't have true_tide_vec_set
                true_tide_vec_set = {}
    
                # Read track one by one.
                for track in indep_tracks:
                    # track[0]: track_number
                    # track[1]: sate name

                    # Only process tracks of this satellite
                    if track[1] != sate_name:
                        continue
    
                    # Exclude the data required by the user
                    # The first two may not be necessary, use_csk and use_s1 are used control the available data dates
                    if sate_name == "csk" and self.use_csk == False:
                        continue
    
                    if sate_name == "s1" and self.use_s1 == False:
                        continue
    
                    if sate_name == "csk" and track[0] in self.csk_excluded_tracks:
                        continue
    
                    if sate_name == "s1" and track[0] in self.s1_excluded_tracks:
                        continue
    
                    # Print the track number and satellite name
                    print('=========== NEW TRACK ==============')
                    print('Reading: ', track[0], track[1])
    
                    # Find the observtion vectors using the track number, track[0]
                    vecs_set = {}
                    for point in point_set:
                        all_tracks_at_this_point = tracks_set[point]
                        for this_track in all_tracks_at_this_point:
                            # If this track covers this point.
                            if this_track[0] == track[0]:
                                vecs_set[point] = (this_track[1], this_track[2])
    
                    track_offsetfields_set, track_offsets_set = self.find_track_data_set(point_set, vecs_set, track)
                    
                    print('For this track, available offsetfields at test point: ',
                            track_offsetfields_set[test_point])
                    print('For this track, obtained offsets and var at test point: ', 
                                track_offsets_set[test_point])
                    print('For this track, length of offsetfields and offsets at test point: ', 
                            len(track_offsetfields_set[test_point]),len(track_offsets_set[test_point]))
    
                    # Point by point adding the data from the new track
                    for point in point_set:
                        # List addition.
                        offsetfields_set[point] = offsetfields_set[point] + track_offsetfields_set[point]

                        # List addition
                        # Old
                        #offsets_set[point] = offsets_set[point] + track_offsets_set[point]

                        # New
                        # Structure of every offset: [rng offset, azi offset, rng offset var, azi offset var].
                        offsets_set[point] = offsets_set[point] + [value[:2] for value in track_offsets_set[point]]
                        offsetsVar_set[point] = offsetsVar_set[point] + [value[2:4] for value in track_offsets_set[point]]
    
                        # Save the information. (# track info)
                        data_info_set[point].append([(track[0], track[1]), len(track_offsets_set[point])])
    
                print('======= END OF EXTRACTION For {} ========'.format(sate_name))
                print('Total number of offsetfields at test point: ', len(offsetfields_set[test_point]))
                print('Total length of offsets at test point: ', len(offsets_set[test_point]))
                print('Total length of offsets variance at test point: ', len(offsetsVar_set[test_point]))
   
                # Generate synthetic data
                if this_data_mode == 2:
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
    
                    # Noise model.
                    simulation_noise_sigma_set = {}
                    noise_sigma_set = {}
                    for point in point_set:
                        simulation_noise_sigma_set[point] = self.simulation_noise_sigma_const_dict[sate_name]
                        noise_sigma_set[point] = [self.noise_sigma_const_dict[sate_name]]*len(offsetfields_set[point])
    
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
                elif this_data_mode == 3:
    
                    # Real data
                    data_vec_set = self.offsets_set_to_data_vec_set(point_set, offsets_set)
    
                    # Noise model.
                    real_data_noise_model_option = 1
                    # Set by user
                    if real_data_noise_model_option == 0:
                        noise_sigma_set = {}
                        for point in point_set:
                            noise_sigma_set[point] = [self.noise_sigma_const_dict[sate_name]]*len(offsetfields_set[point])

                    elif real_data_noise_model_option == 1:
                        noise_sigma_set = {}
                        for point in point_set:
                            noise_sigma_set[point] = [ (np.sqrt(value[0])/5, np.sqrt(value[1])/5 ) for value in offsetsVar_set[point] ]
                    else:
                        raise Exception("Not implemented")

                    #print(noise_sigma_set[test_point])

                    ## Get reference velocity
                    #n_params = 3 + len(self.modeling_tides) * 6
                    #for point in point_set:
                    #    true_tide_vec_set[point] = np.zeros(shape=(n_params, 1))
                    #    true_tide_vec_set[point][0] = secular_v_set[point][0]
                    #    true_tide_vec_set[point][1] = secular_v_set[point][1]
                    #    true_tide_vec_set[point][2] = secular_v_set[point][2]
            else:
                raise Exception("Data mode error " + str(this_data_mode))

            print("***************************")
            print("Summary of satellite {} dataset extraction".format(sate_name))
            print("date mode: ", this_data_mode)
            print("data vec shape: ", data_vec_set[test_point].shape)
            print("**************************")
            data_info_set_dict[sate_name] = data_info_set
            data_vec_set_dict[sate_name] = data_vec_set
            noise_sigma_set_dict[sate_name] = noise_sigma_set
            offsetfields_set_dict[sate_name] = offsetfields_set
            true_tide_vec_set_dict[sate_name] = true_tide_vec_set

        # Put together information from different satellites
        # data_info_set(deprecated): For each point a list of (track_num, sate_name, number of data points)
        # data_vec_set: For each point, a vector
        # noise_sigma_set: For each point, a two value tuple (range error and azimuth error)
        # offsetfields_set: For each point, a list of offsetfield info [date1, date2, vec1, vec2, t_frac]
        # true_tide_vev_set is the same for all satellites

        # data_info_set is deprecated
        final_data_info_set = {}

        # The three key result sets
        final_data_vec_set = {}
        final_noise_sigma_set = collections.defaultdict(list)
        final_offsetfields_set = collections.defaultdict(list)

        # Find the true_tide_vec_set from any non-empty satellite result
        final_true_tide_vec_set = {}
        for sate_name in true_tide_vec_set_dict.keys():
            if true_tide_vec_set_dict[sate_name]:
                final_true_tide_vec_set = true_tide_vec_set_dict[sate_name]
                break

        print(noise_sigma_set_dict.keys())
        if 'csk' in noise_sigma_set_dict:
            print("csk noise dict size: ", len(noise_sigma_set_dict['csk']))
        if 's1' in noise_sigma_set_dict:
            print("s1 noise dict size: ", len(noise_sigma_set_dict['s1']))

        #print(data_vec_set_dict['csk'][point].shape)

        for point in point_set:
            for sate_name in sate_names:
                if point not in final_data_vec_set:
                    final_data_vec_set[point] =  data_vec_set_dict[sate_name][point]
                else:
                    if len(final_data_vec_set[point])>0 and len(data_vec_set_dict[sate_name][point])>0:
                        final_data_vec_set[point] = np.vstack((final_data_vec_set[point], data_vec_set_dict[sate_name][point]))
                    elif len(data_vec_set_dict[sate_name][point])>0:
                        final_data_vec_set[point] = data_vec_set_dict[sate_name][point]

                final_noise_sigma_set[point] = final_noise_sigma_set[point] + noise_sigma_set_dict[sate_name][point]
                final_offsetfields_set[point] = final_offsetfields_set[point] + offsetfields_set_dict[sate_name][point]

        print('======= SUMMARY OF DATA SET FORMATION ========')
        print("Summary (include all satellites: )")
        print('Total number of offsetfields at test point: ', len(final_offsetfields_set[test_point]))
        print('Total length of noise_sigma pair at test point: ', len(final_noise_sigma_set[test_point]))
        print('Total length of offset measurement: ', len(final_data_vec_set[test_point]))
        print('==============================================')

        # show the information of test
        #print(final_offsetfields_set[test_point])
        print(final_noise_sigma_set[test_point])
        if test_point in true_tide_vec_set:
            print("true tide vec at test point: ", true_tide_vec_set[test_point])

        return (final_data_info_set, final_data_vec_set, final_noise_sigma_set, final_offsetfields_set, final_true_tide_vec_set)
