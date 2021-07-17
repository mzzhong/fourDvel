#!/usr/bin/env python3
import os
import argparse

import pickle
import numpy as np

import multiprocessing
from fourdvel import fourdvel
from display import display

import matplotlib.pyplot as plt
import collections

def createParser():

    parser = argparse.ArgumentParser( description='driver of fourdvel')
    
    parser.add_argument('-p','--param_file', dest='param_file', type=str, help='parameter file', required=True)

    parser.add_argument('--output_mode', dest='output_mode', type=str, help='output_mode (estimation or analysis)', default='estimation')

    # needs to be consistent with what is used for running 
    parser.add_argument('-t','--task_name', dest='task_name', type=str, help='task_name', required=True)

    # This is for further narrow down the output quantity
    parser.add_argument('--output_name', dest='output_name', type=str, help='output_name', default=None)

    parser.add_argument('-q','--quant_list_name', dest='quant_list_name', type=str, help='quant_list_name', default=None)

    parser.add_argument('--npc','--no_phase_correction', dest='no_phase_correction', action='store_true')
   
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

class output(fourdvel):

    def __init__(self, inps):

        param_file = inps.param_file
        super(output,self).__init__(param_file = param_file)

        self.get_grid_set_v2()
        self.get_grid_set_velo()
        test_id = self.test_id

        self.estimation_dir = os.path.join(self.estimations_dir,str(test_id))

        self.display = display(param_file)

        self.output_mode = inps.output_mode

        self.output_name = inps.output_name

        self.quant_list_name = inps.quant_list_name

        self.task_name = inps.task_name

        self.no_phase_correction = inps.no_phase_correction

    def run_output_residual(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id


        #with open(this_result_folder + '/' 
        #            + str(test_id) + '_' + 'grid_set_resid_of_secular.pkl','rb') as f:
        #    self.grid_set_resid_of_secular = pickle.load(f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_resid_of_tides.pkl','rb') as f:
            self.grid_set_resid_of_tides = pickle.load(f)

        # compare id
        compare_id = 202010565141

        with open(self.estimations_dir + '/' + str(compare_id) + '/' +
                    str(compare_id) + '_' + 'grid_set_resid_of_tides.pkl','rb') as f:
            self.grid_set_resid_of_tides_compare = pickle.load(f)

        # Set the compare set

        grid_sets = {}
        #grid_sets['resid_of_secular'] = self.grid_set_resid_of_secular
        grid_sets['resid_of_tides'] = self.grid_set_resid_of_tides

        state = 'est'
        comps = ['range','azimuth']

        for misfit_name in grid_sets.keys():
            for comp in comps:

                quant_name = '_'.join([misfit_name, comp])

                print('Output quantity name: ', quant_name)
                grid_set_quant = {}
                grid_set_quant_compare = {}

                # The two grid sets
                this_grid_set = grid_sets[misfit_name]
                compare_grid_set = self.grid_set_resid_of_tides_compare

                output_keys = this_grid_set.keys()

                # For all available points in grid_set.
                for point in output_keys:
                
                    # Four entries: range_mean(0), range_rms(1), azimuth_mean(2), azimuth_rms(3)
                    quant = this_grid_set[point]

                    try:
                        quant_compare = compare_grid_set[point]
                    except:
                        quant_compare = [np.nan]*10

                    # Output the rms
                    if comp == 'range':
                        grid_set_quant[point] = quant[1]
                        #print(quant[1])
                        #print(quant_compare[1])
                        grid_set_quant_compare[point] = quant[1] - quant_compare[1]

                    elif comp == 'azimuth':
                        grid_set_quant[point] = quant[3]
                        grid_set_quant_compare[point] = quant[3] - quant_compare[3]

                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

                # Write the compare to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '_compare' + '.xyz')

                self.display.write_dict_to_xyz(grid_set_quant_compare, xyz_name = xyz_name)

        return 0

    def load_master_model(self,num,prefix='est'):

        this_result_folder = self.estimation_dir
        # Load all the results.
        if prefix == 'true':
            filename = self.estimations_dir +'/'+str(num)+'/'+str(num)+'_grid_set_true_tide_vec.pkl'
        else:
            filename = self.estimations_dir +'/'+str(num)+'/'+str(num)+'_grid_set_tide_vec.pkl'
           
        with open(filename,'rb') as f:
            self.grid_set_master_model_tide_vec = pickle.load(f)
        return 0

    def load_slave_model(self,num,prefix='est'):

        this_result_folder = self.estimation_dir
        # Load all the results.

        if prefix == 'true':
            filename = self.estimations_dir + '/' + str(num)+'/'+str(num)+'_grid_set_true_tide_vec.pkl'
        else:
            filename = self.estimations_dir + '/' + str(num)+'/'+str(num)+'_grid_set_tide_vec.pkl'
 
        with open(filename,'rb') as f:
            self.grid_set_slave_model_tide_vec = pickle.load(f)
        return 0

    def load_everything(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        # Load all the results.
        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_true_tide_vec.pkl','rb') as f:
            self.grid_set_true_tide_vec = pickle.load(f)


        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_tide_vec.pkl','rb') as f:
            self.grid_set_tide_vec = pickle.load(f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_tide_vec_uq.pkl','rb') as f:
            self.grid_set_tide_vec_uq = pickle.load(f)

        return 0

    def run_output_difference(self, compare_id, compare_prefix):

        print('Ouput difference...')

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        self.load_master_model(test_id)
        self.load_slave_model(num=compare_id,prefix=compare_prefix)

        #self.load_true_est_uq()

        quant_list = [ 'secular_horizontal_velocity_difference' ]

        for quant_name in quant_list:
            
            print('Output quantity name: ', quant_name)
            grid_set_quant = {}

            if quant_name == 'secular_horizontal_velocity_difference':
                grid_set_slave = self.grid_set_slave_model_tide_vec
                grid_set_master = self.grid_set_master_model_tide_vec

                for point in grid_set_master.keys():
                    if not np.isnan(grid_set_master[point][0,0]) and point in grid_set_slave:
                        quant_master = self.tide_vec_to_quantity(input_tide_vec = grid_set_master[point], quant_name = "secular_horizontal_velocity_EN")
                        quant_slave = self.tide_vec_to_quantity(input_tide_vec = grid_set_slave[point], quant_name = 'secular_horizontal_velocity_EN')
                        grid_set_quant[point] = np.linalg.norm(quant_master - quant_slave, 2)

                # Write to xyz file.
                state = 'est'

                xyz_name = os.path.join(this_result_folder, '_'.join([str(test_id), state, quant_name, str(compare_id), compare_prefix]) + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

                xyz_name_2 = os.path.join(this_result_folder, '_'.join([str(test_id), state, quant_name]) + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name_2)


        return 0

    def run_output_others(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_others.pkl','rb') as f:
            this_grid_set = pickle.load(f)

        # credible interval needs to before grounding level for filtering purposes
        quant_list=["up_scale", "grounding_level_credible_interval", "optimal_grounding_level_prescale", "optimal_grounding_level", "height"]
        #quant_list=["up_scale", "grounding_level_credible_interval", "optimal_grounding_level_prescale", "optimal_grounding_level", "grounding_duration", "height"]

        states = ['true', 'est']

        # Used for save the results from true and est to get bias
        saved_grid_set_quant_results = {}

        for state in states:
            for quant_name in quant_list:
                print('Output quantity name: ', quant_name)
                grid_set_quant = {}
    
                output_keys = this_grid_set.keys()

                for point in output_keys:

                    # The point should be tuple. However, there are exceptions for the key, e.g. current_auto_enum_stage
                    if not isinstance(point, tuple):
                        continue

                    model_up = self.grid_set_velo[point][2] 
    
                    # Ad hoc treatment of grounding level
                    if quant_name.startswith("optimal_grounding_level"):

                        quant_name_orig = 'optimal_grounding_level'
   
                        # If the data doesn't exist 
                        if not quant_name_orig in this_grid_set[point].keys():
                            continue
   
                        # Remove the stagnent points 
                        #if self.grid_set_velo[point][2]<=0.4:
                        #if self.grid_set_velo[point][2]<=0.2:
                        #if self.grid_set_velo[point][2]<=0.1:
                        #if self.grid_set_velo[point][2]<=0.05:
                        if model_up == 0:
                            continue
    
                        if this_grid_set[point][quant_name_orig]=='external':
                            continue
  
                        if state == 'est':
                            optimal_grounding_level = this_grid_set[point][quant_name_orig]/(10**6) 
                            
                            # Remove very low grounding level, note the value is integer
                            if self.proj == 'Rutford' and optimal_grounding_level <= -2.8:
                                continue

                            if self.proj == 'Evans' and optimal_grounding_level <= -3.0:
                                continue

                            #gl_ci_thres = 100
                            #gl_ci_thres = 0.5
                            gl_ci_thres = 1.0
                            #gl_ci_thres = 1.5
                            
                            # Remove based obtained credible level
                            gl_ci = grid_set_gl_ci.get(point, np.nan)
                            if np.isnan(gl_ci) or gl_ci>=gl_ci_thres:
                                continue

                            # Remove points if the lowest_tide_height is too close to the obtained value
                            lowest_tide_height = this_grid_set[point]['lowest_tide_height']
                            print('state: ', state)
                            print('lowest_tide_height: ', lowest_tide_height)
                            print('obtained ogl: ', optimal_grounding_level)
                            #if optimal_grounding_level - lowest_tide_height <= 0.1:
                            #    continue

                            # Remove points without descending track
                            #print(self.grid_set[point])
                            count = 0
                            for track in self.grid_set[point]:
                                print(track)
                                if track[3] == 'csk' and track[0]>=11:
                                    count += 1
                            if count==0:
                                continue
    
                    # Record everything, including np.nan
                    # np.nan is filtered in write_dict_to_xyz

                    # up_scale
                    if quant_name in ["up_scale"]:
                        if state == "true":
                            grid_set_quant[point] = this_grid_set[point].get("true_" + quant_name, np.nan)
                        
                        elif state == 'est':
                            
                            grid_set_quant[point] = this_grid_set[point].get(quant_name, np.nan)

                    # optimal grounding level
                    elif quant_name.startswith("optimal_grounding_level"):

                        #if quant_name.endswith('prescale'):
                        #print(quant_name)
                        #print(stop)

                        if state == "true":
                            grid_set_quant[point] = this_grid_set[point].get("true_" + "optimal_grounding_level", np.nan)

                        elif state == 'est':
                            optimal_grounding_level_int = this_grid_set[point].get("optimal_grounding_level", np.nan)
                            optimal_grounding_level = optimal_grounding_level_int / 10**6

                            crsp_up_scale = this_grid_set[point].get("up_scale",np.nan)

                            # unscaled
                            if quant_name.endswith('prescale'):
                                grid_set_quant[point] = optimal_grounding_level

                            # scaled
                            else:
                                if "grounding_level_up_scale" in this_grid_set[point]: 
                                    grid_set_quant[point] = optimal_grounding_level * crsp_up_scale
                                else:
                                    raise ValueError()

                        else:
                            raise ValueError()

                    elif quant_name in ["grounding_level_credible_interval"]:
                        if state == "true":
                            grid_set_quant[point] = np.nan
                        elif state == 'est':
                            ci = this_grid_set[point].get("grounding_level_credible_interval", (np.nan, np.nan))
                            grid_set_quant[point] = ci[1] - ci[0]

                            # clip every thing outside ice-shelf
                            if model_up == 0:
                                grid_set_quant[point] = np.nan

                        else:
                            raise ValueError()

                    elif quant_name in ["height"]:
                        if state == "true":
                            grid_set_quant[point] = this_grid_set[point].get(quant_name, np.nan)
                        elif state == 'est':
                            grid_set_quant[point] = np.nan

                        else:
                            raise ValueError()
                    else:
                        raise Exception()

                if state == 'est' and quant_name in ["grounding_level_credible_interval"]:
                    grid_set_gl_ci = grid_set_quant

                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + 'others' + '_' + quant_name + '.xyz')
                print("Saving: ", xyz_name)
                
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

                # Save the results
                saved_grid_set_quant_results[(state, quant_name)] = grid_set_quant

        # bias
        state = 'bias'
        for quant_name in  ['optimal_grounding_level', 'up_scale']:
            print('Output quantity name: ', quant_name)
            grid_set_quant = {}
    
            output_keys = this_grid_set.keys()

            true_grid_set_quant = saved_grid_set_quant_results[("true", quant_name)]
            est_grid_set_quant = saved_grid_set_quant_results[("est", quant_name)]

            for point in output_keys:

                if not isinstance(point, tuple):
                    continue
 
                grid_set_quant[point] = est_grid_set_quant.get(point,np.nan) - true_grid_set_quant.get(point,np.nan)

            # Write to xyz file.
            xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + 'others' + '_' + quant_name + '.xyz')

            #print(grid_set_quant)
            #print(sorted(grid_set_quant.keys()))
            #print(len(grid_set_quant.keys()))
            #print(stop)
            
            self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0

    def run_output_analysis(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        # residual_vs_tide_height
        if self.task_name == 'residual_vs_tide_height':
            pkl_name = '_'.join([str(test_id), 'grid_set_analysis', self.task_name, self.analysis_name])  + '.pkl'

            # Load the pickle file
            with open(this_result_folder + '/' + pkl_name,'rb') as f:
                this_grid_set = pickle.load(f)
    
            # Set the quantities for output
            state='est'
    
            quant_names = ['best_slr_results', 'best_slr_data_stats','lowest_tide']
            
            subquant_names = {}
            subquant_names['best_slr_results'] = ['slope','intercept','r_value','p_value','min_proxy_tide','track_num']
            subquant_names['best_slr_data_stats']=['data_mean','data_median','data_std','picked_data_mean','picked_data_median','picked_data_std']
            subquant_names['lowest_tide']=['height','track_num']
    
            for quant_name in quant_names:
    
                for subquant_name in subquant_names[quant_name]:
    
                    print('Output quantity name: ', quant_name +'_' + subquant_name)
    
                    grid_set_quant = {} 
                    output_keys = this_grid_set.keys()
            
                    for point in output_keys:
                    
                        point_values = this_grid_set[point]
                        
                        # Valid result 
                        if len(point_values)<20:
                            #print(point_values)
                            point_quant_values = point_values[quant_name]
    
                            # This is not an empty dictionary
                            if len(point_quant_values)>0:
                                grid_set_quant[point] = point_quant_values[subquant_name]
                            else:
                                grid_set_quant[point] = np.nan
                        else:
                            grid_set_quant[point] = np.nan
            
                    # Write to xyz file.
                    xyz_name = os.path.join(this_result_folder, '_'.join((str(test_id), state, self.analysis_name, quant_name, subquant_name)) + '.xyz')
                    self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        # residual analysis 
        elif self.task_name == 'residual_analysis':
            pkl_name = '_'.join((str(test_id), 'grid_set_analysis', self.task_name))  + '.pkl'

            with open(this_result_folder + '/' + pkl_name,'rb') as f:
                residual_set_by_point = pickle.load(f)
 
            print(residual_set_by_point)

            # Rearrange the dictionary to generate residual_set_by_track
            residual_set_by_obs = collections.defaultdict(dict)

            for point in residual_set_by_point.keys():
                for obs in residual_set_by_point[point]:
                    print(point, obs, residual_set_by_point[point][obs])
                    residual_set_by_obs[obs][point] = residual_set_by_point[point][obs]
            
            # Output the residuals
            print(residual_set_by_obs)
            for obs in residual_set_by_obs.keys():

                (obs_sate, obs_track_num), obs_vec = obs
                xyz_name = os.path.join(this_result_folder, '_'.join(['residual',obs_sate, str(obs_track_num), obs_vec]) + '.xyz')

                grid_set_quant = residual_set_by_obs[obs]

                print(xyz_name)
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        else:
            raise ValueError()


        return 0

    def output_estimations(self, output_states):

        modeling_tides = self.modeling_tides
        n_modeling_tide = self.n_modeling_tides

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        quant_list_name = self.quant_list_name
        task_name = self.task_name

        self.load_everything()

        quant_list = [  'secular_horizontal_speed',
                        'secular_east_velocity',
                        'secular_north_velocity',
                        'Msf_horizontal_displacement_amplitude',
                        'Msf_east_displacement_amplitude',
                        'Msf_north_displacement_amplitude',
                        'M2_up_displacement_amplitude',
                        'O1_up_displacement_amplitude']

        quant_list = [  'secular_horizontal_speed',
                        'secular_east_velocity',
                        'secular_north_velocity',
                        'secular_up_velocity',
                        'secular_horizontal_velocity',
                        'Msf_horizontal_displacement_amplitude',
                        'Msf_north_displacement_amplitude',
                        'Msf_north_displacement_phase',
                        'M2_up_displacement_amplitude',
                        'M2_up_displacement_phase',
                        'O1_up_displacement_amplitude',
                        'O1_up_displacement_phase']

        quant_list = [  'secular_horizontal_speed',
                        'secular_east_velocity',
                        'secular_north_velocity',
                        'secular_up_velocity',
                        'secular_horizontal_velocity',

                        'Msf_horizontal_displacement_amplitude',
                        'Msf_north_displacement_amplitude',
                        'Msf_north_displacement_phase',


                        'Mf_horizontal_displacement_amplitude',
                        'Mf_north_displacement_amplitude',
                        'Mf_north_displacement_phase',

                        'M2_up_displacement_amplitude',
                        'M2_up_displacement_phase',
                        'O1_up_displacement_amplitude',
                        'O1_up_displacement_phase']


        quant_list = [  'secular_horizontal_speed',
                        'secular_east_velocity',
                        'secular_north_velocity',
                        'secular_up_velocity',
                        'secular_horizontal_velocity',

                        'M2_up_displacement_amplitude',
                        'M2_up_displacement_phase',
                        'O1_up_displacement_amplitude',
                        'O1_up_displacement_phase',
                        'N2_up_displacement_amplitude',
                        'N2_up_displacement_phase',
 
                        # Msf
                        "Msf_horizontal_displacement_group",
                        "Msf_up_displacement_amplitude",
                        "Msf_up_displacement_phase",

                        # Msf east and north
                        "Msf_east_displacement_amplitude",
                        "Msf_east_displacement_phase",

                        "Msf_north_displacement_amplitude",
                        "Msf_north_displacement_phase",

                        # Mf
                        'Mf_horizontal_displacement_amplitude',
                        "Mf_up_displacement_amplitude",
                        "Mf_up_displacement_phase",

                        # M2
                        'M2_horizontal_displacement_amplitude',

                        # O1
                        'O1_horizontal_displacement_amplitude'
                        ]


        if quant_list_name == "BM_2017":
            quant_list = [  'secular_horizontal_speed',
                            'secular_east_velocity',
                            'secular_north_velocity',
                            'secular_up_velocity',
                            'secular_horizontal_velocity',
    
                            'M2_up_displacement_amplitude',
                            'M2_up_displacement_phase',
                            'O1_up_displacement_amplitude',
                            'O1_up_displacement_phase',
     
                            # Msf
                            "Msf_horizontal_displacement_group"                            
                            ]

        sub_quant_names_for_groups = {}
        sub_quant_names_for_groups["Msf_horizontal_displacement_group"] = [ "Msf_along_flow_displacement_amplitude",
                                                                            "Msf_along_flow_displacement_phase",
                                                                            "Msf_along_flow_displacement_phase_in_deg",
                                                                            
                                                                            "Msf_cross_flow_displacement_amplitude", 
                                                                            "Msf_cross_flow_displacement_phase",

                                                                            "Msf_horizontal_displacement_amplitude"]


        # Additional parameters
        if task_name == 'tides_3':
            quant_list.append('up_amplitude_scaling')

        if self.est_topo_resid == True:
            quant_list.append('topo_resid')

        print("quant_list: ", quant_list)
        
        ### End with quant list ###

        states = {}
        states['true'] = self.grid_set_true_tide_vec
        states['est'] = self.grid_set_tide_vec

        # This is from model_posterior_to_uncertainty, which converts Cm_p to tide_vec_uq
        states['uq'] = self.grid_set_tide_vec_uq

        # Look through the sets
        phase_center = {}

        # Used for save the results from true and est to get bias
        saved_grid_set_quant_results = {}

        # Create a file to save mean phase
        f_mp  = open(self.estimation_dir + '/' + 'mean_phase.txt','w')

        print("output states: ", output_states)
        for state in output_states:

            print("\n")
            print("###########################")
            print("current state: ", state)

            if state in ["true","est","uq"]:
                this_grid_set = states[state]

            # Loop through the quantities.
            for quant_name in quant_list:

                ## Derive the point set
                # normal states
                if state in ["true", "est", "uq"]:

                    # down sample for velocity vector.
                    if quant_name == 'secular_horizontal_velocity':
                        output_keys = []
                        
                        for point in this_grid_set.keys():
    
                            lon, lat = point
                            lon_ind = np.round(lon/self.lon_step_int)
                            lat_ind = np.round(lat/self.lat_step_int) 
    
                            if self.resolution == 100:
                                downsample = 50
                            elif self.resolution == 500:
                                downsample = 10
                            elif self.resolution == 1000:
                                downsample = 5
                            elif self.resolution == 2000:
                                downsample = 5
                            else:
                                raise Exception()
    
                            if lon_ind % downsample==0 and lat_ind % downsample==0:
                                output_keys.append((lon,lat))
    
                        output_keys = set(output_keys)
    
                    else:
                        output_keys = this_grid_set.keys()
                        print("size of output point set: ", len(output_keys))

                    # Note that: For "true", there is no output_keys in data_mode 3.
                    print('Output quantity name: ', quant_name)

   
                    # Initialization
                    grid_set_quant = {}
                    # no phase correction
                    grid_set_quant_npc = {}
    
                    # Check if this is a single or group quant_name
                    # group mode
                    if quant_name.endswith("group"):
                        print("group name")
                        if quant_name == "Msf_horizontal_displacement_group":
                            sub_quant_names = sub_quant_names_for_groups[quant_name]
                            
                            for sub_quant_name in sub_quant_names:
                                grid_set_quant[sub_quant_name] = {}
                                grid_set_quant_npc[sub_quant_name] = {}
    
                        else:
                            raise Exception("Undefined group name")
                        
                        for point in output_keys:
                            # The vector is not nan
                            if not np.isnan(this_grid_set[point][0,0]):

                                # when processing uq, need to accompany it with est
                                if state in ['true', 'est']:
                                    input_tide_vec = this_grid_set[point]
                                elif state in ['uq']:
                                    input_tide_vec = (states['est'][point], this_grid_set[point])
                                else:
                                    raise Exception()

                                quant_group = self.tide_vec_to_quantity(input_tide_vec = input_tide_vec, quant_name = quant_name, point = point, state=state)
    
                                # Save it into grid_set_quant
                                for sub_quant_name in sub_quant_names:
                                    try:
                                        grid_set_quant[sub_quant_name][point] = quant_group[sub_quant_name]
                                    except:
                                        print(grid_set_quant.keys())
                                        print(stop)
    
    
                    # Normal single mode
                    else:
                        sub_quant_names = [quant_name]
                        grid_set_quant[quant_name] = {}
                        grid_set_quant_npc[quant_name] = {}

                        # Loop through each point in the output point set
                        # If state is true and data_mode is 3, there is simply no key    
                        for point in output_keys:
                        
                            # Only record points where inverse problem can be done, Cm_p exists.
                            if not np.isnan(this_grid_set[point][0,0]):

                                # when processing uq, need to accompany it with est
                                if state in ['true', 'est']:
                                    input_tide_vec = this_grid_set[point]
                                elif state in ['uq']:
                                    input_tide_vec = (states['est'][point], this_grid_set[point])
                                else:
                                    raise Exception()

                                # It is possible that some tides are not in the model. This is taken care of in the called method.
                                quant = self.tide_vec_to_quantity(input_tide_vec = input_tide_vec, quant_name = quant_name, point=point, state=state)
        
                                # Here we record everything, if Cm_p exists, including nan futher filtered by tide_vec_to_quantity.
                                grid_set_quant[quant_name][point] = quant

                    ######################################################
                    # Some additonal work on removing bad values
                    for sub_quant_name in sub_quant_names:
                        for point in grid_set_quant[sub_quant_name].keys():
                            # Need at least two track
                            if len(self.grid_set[point])<=1:
                                grid_set_quant[sub_quant_name][point] = np.nan
 
                    ########    End of extraction   #############
    
                    ##### Do phase correction for mean phase #################
                    do_correction = True
                    if self.no_phase_correction:
                        do_correction = False
    
                    ## Do phase correction with the mean phase of true model
                    do_correction_with_true = False
    
                    for sub_quant_name in sub_quant_names:
                        if (state=='true' or state=='est') and 'phase' in sub_quant_name:
                            ### Some post processing on phase
                            # Remove wrong phase where there is only one track (this is supposed to be done in the lines above)
                            for point in grid_set_quant[sub_quant_name].keys():
                                # Need at least two track
                                if len(self.grid_set[point])<=1:
                                    grid_set_quant[sub_quant_name][point] = np.nan

                            # Make a copy of the quant before phase correction
                            grid_set_quant_npc[(state, sub_quant_name)] = grid_set_quant[sub_quant_name].copy()

                            # Get all the values                        
                            values = np.asarray(list(grid_set_quant[sub_quant_name].values()))

                            # Count the number of non-zero values
                            count = np.count_nonzero(~np.isnan(values))
   
                            # If there are at least one valid value 
                            if count>0:
                                # this is turned off ususally
                                if do_correction_with_true ==True and sub_quant_name in phase_center and state == "est":
                                    print("In phase center: ", sub_quant_name)
                                    center = phase_center[sub_quant_name]
                                # default version
                                else:
                                    print("Calculated the mean phase")
                                    center = np.nansum(values) /count

                                # Fix the center for certain componenets
                                if state == 'est' and self.proj == "Evans" and sub_quant_name == "Msf_along_flow_displacement_phase":
                                    center = -3.647
                                    #print(stop)

                                # Save the mean phase
                                f_mp.write(state + '_' + sub_quant_name + ' ' + str(center)+'\n')

                                # Do correction
                                # Forced no correction for "in_deg" (ad hoc)
                                if do_correction and not "in_deg" in sub_quant_name:
                                    print("Do mean phase shift")
                                    print("The mean phase is: ", center)
                                    for point in grid_set_quant[sub_quant_name].keys():
                                        grid_set_quant[sub_quant_name][point] -= center
                                else:
                                    print("Skip mean phase shift: ", sub_quant_name)
                                    print("The mean phase is: ", center)
    
                                if state=="true":
                                    print("Give the mean phase of true model to phase center dictionary")
                                    phase_center[sub_quant_name] = center

                            # Mean phase is not defined
                            else:
                                # Save the mean phase
                                f_mp.write(state + '_' + sub_quant_name + ' ' + 'NaN' + '\n')
                               
                    ######## End of mean phase correction   #####

                elif state == "bias":
                    # Note that: For "true", there is no output_keys in data_mode 3.
                    print('Output quantity name: ', quant_name)
    
                    # Initialization
                    grid_set_quant = {}
                    # If this is a group 
                    if quant_name.endswith("group"):
                        print("group name")
                        if quant_name == "Msf_horizontal_displacement_group":
                            sub_quant_names = sub_quant_names_for_groups[quant_name]
                            for sub_quant_name in sub_quant_names:
                                grid_set_quant[sub_quant_name] = {}
                        else:
                            raise Exception("Undefined group name")
 
                    # If this is a normal quant_name
                    else:
                        sub_quant_names = [quant_name]
                        grid_set_quant[quant_name] = {}

                    # Find the difference betweeen true and est (without phase correction)
                    for sub_quant_name in sub_quant_names:
                        true_grid_set_quant = saved_grid_set_quant_results[("true", sub_quant_name)]
                        est_grid_set_quant = saved_grid_set_quant_results[("est", sub_quant_name)]

                        for point in true_grid_set_quant.keys():
                            if point in est_grid_set_quant:
                                if isinstance(true_grid_set_quant[point], float):
                                    grid_set_quant[sub_quant_name][point] = est_grid_set_quant[point] - true_grid_set_quant[point]
                                elif isinstance(true_grid_set_quant[point], tuple):
                                    grid_set_quant[sub_quant_name][point] = tuple([est_grid_set_quant[point][j]-true_grid_set_quant[point][j] for j in range(len(true_grid_set_quant[point]))])
                                else:
                                    raise Exception("Unknown type of data")
                else:
                    raise ValueError('Unknown state')

   
                #### Write to xyz file #####
                for sub_quant_name in sub_quant_names:
                    xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + sub_quant_name + '.xyz')
                    self.display.write_dict_to_xyz(grid_set_quant[sub_quant_name], xyz_name = xyz_name)

                    # Save the results
                    # If this is phase, then phase correction may be done.
                    # Record the phase values with no correction
                    if (state=='true' or state=='est') and 'phase' in sub_quant_name:
                        saved_grid_set_quant_results[(state, sub_quant_name)] = grid_set_quant_npc[(state, sub_quant_name)]
                    # In other cases, just record the values
                    else:
                        saved_grid_set_quant_results[(state, sub_quant_name)] = grid_set_quant[sub_quant_name]

        # close the mean phase file
        f_mp.close()

        return 0                

def main(iargs=None):

    inps = cmdLineParse(iargs)

    out = output(inps)

    if out.task_name in ['residual_analysis']:
        out.output_mode = 'analysis'

    # output estimation
    if out.output_mode == 'estimation':

        # If output_name is provided, we need to override the default output names
        if out.output_name is not None:
            # override the settings from param file
            out.output_true = False
            out.output_est = False
            out.output_uq = False
            out.output_resid = False
            out.output_difference = False
            out.output_analysis = False
            out.output_others = False
    
            # Set this output name to be True
            setattr(out, 'output_' + out.output_name, True)

        ### Prepare the outout ###
        # Start to output
        output_states = []
    
        if out.output_true: output_states.append("true")
        if out.output_est:  output_states.append("est")
        if out.output_uq:   output_states.append("uq")
    
        # bias
        if out.output_true and out.output_est:
            output_states.append("bias")
    
        # First output others
        if out.output_others:
            out.run_output_others()
     
        ### Do the output jobs ###
        out.output_estimations(output_states)
    
        # Output residual 
        if out.output_resid: 
            out.run_output_residual()
    
        if out.output_difference:
            if out.proj=="Evans":
                # Evans
                #out.run_output_difference(compare_id=620, compare_prefix='true')
                out.run_output_difference(compare_id=20211514, compare_prefix='true')

            elif out.proj == "Rutford":
                # Rutford
                out.run_output_difference(compare_id=202007042, compare_prefix='true')
            else:
                raise Exception()

    elif out.output_mode == 'analysis':

        out.run_output_analysis()

    else:
        raise ValueError()

    # End of main

if __name__=='__main__':
    main()
