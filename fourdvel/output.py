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
    
    parser.add_argument('-p','--param_file', dest='param_file',type=str,help='parameter file',required=True)

    parser.add_argument('-q','--quant_list_name', dest='quant_list_name',type=str,help='quant_list_name',default=None)
    
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
                    if not np.isnan(grid_set_master[point][0,0]):
                        quant_master = self.tide_vec_to_quantity(tide_vec = grid_set_master[point],quant_name = "secular_horizontal_velocity_EN")
                        quant_slave = self.tide_vec_to_quantity(tide_vec = grid_set_slave[point], quant_name = 'secular_horizontal_velocity_EN')
                        grid_set_quant[point] = np.linalg.norm(quant_master - quant_slave, 2)

                # Write to xyz file.
                state = 'est'

                xyz_name = os.path.join(this_result_folder, '_'.join([str(test_id), state, quant_name, str(compare_id), compare_prefix]) + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0

    def run_output_others(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        state='est'

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_others.pkl','rb') as f:
            this_grid_set = pickle.load(f)

 
        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_others.pkl','rb') as f:
            this_grid_set = pickle.load(f)


        quant_list=["up_scale", "optimal_grounding_level"]

        for quant_name in quant_list:

            print('Output quantity name: ', quant_name)
            grid_set_quant = {}

            output_keys = this_grid_set.keys()
            for point in output_keys:

                #if quant_name.startswith('optimal'):
                #    print(stop)

                # Ad hoc
                if quant_name == "optimal_grounding_level":

                    if not quant_name in this_grid_set[point].keys():
                        continue

                    if self.grid_set_velo[point][2]<=0.4:
                        continue

                    try:
                        if this_grid_set[point][quant_name]=='external':
                            continue
                    except:
                        pass

                    try:
                        if this_grid_set[point][quant_name]<=-2.8:
                            continue
                    except:
                        pass

                #if quant_name.startswith('optimal'):
                #    print(stop)
        
                # Record everything, including np.nan
                # np.nan is filtered in write_dict_to_xyz
                grid_set_quant[point] = this_grid_set[point][quant_name]

            # Write to xyz file.
            xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + 'others' + '_' + quant_name + '.xyz')
            
            self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0

    def output_analysis(self):

        this_result_folder = self.estimation_dir
        test_id = self.test_id

        # Load the pickle file
        pkl_name = '_'.join((str(test_id), 'grid_set_analysis', self.analysis_name)) + '.pkl'

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

        return 0

    def output_estimations(self, output_states, quant_list_name):

        modeling_tides = self.modeling_tides
        n_modeling_tide = self.n_modeling_tides

        this_result_folder = self.estimation_dir
        test_id = self.test_id

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
                        'secular_up_velocity',
                        'secular_horizontal_velocity',

                        'M2_up_displacement_amplitude',
                        'M2_up_displacement_phase',
                        'O1_up_displacement_amplitude',
                        'O1_up_displacement_phase',
                        'N2_up_displacement_amplitude',
                        'N2_up_displacement_phase',
                        #'Q1_up_displacement_amplitude',
                        #'Q1_up_displacement_phase',
 
                        # Msf
                        "Msf_horizontal_displacement_group",
                        "Msf_up_displacement_amplitude",
                        "Msf_up_displacement_phase",


                        # Mf
                        'Mf_horizontal_displacement_amplitude',
                        "Mf_up_displacement_amplitude",
                        "Mf_up_displacement_phase",

                        # M2
                        'M2_horizontal_displacement_amplitude',

                        # O1
                        'O1_horizontal_displacement_amplitude'

                        ]

#        quant_list = [
#                        'M2_up_displacement_amplitude',
#                        'M2_up_displacement_phase',
#                        'O1_up_displacement_amplitude',
#                        'O1_up_displacement_phase']

        if quant_list_name == "BM_2017":
            quant_list = [  'secular_horizontal_speed',
                            'secular_up_velocity',
                            'secular_horizontal_velocity',
    
                            'M2_up_displacement_amplitude',
                            'M2_up_displacement_phase',
                            'O1_up_displacement_amplitude',
                            'O1_up_displacement_phase',
     
                            # Msf
                            "Msf_horizontal_displacement_group"
                            ]



        states = {}
        states['true'] = self.grid_set_true_tide_vec
        states['est'] = self.grid_set_tide_vec
        states['uq'] = self.grid_set_tide_vec_uq


        # Look through the sets
        phase_center = {}

        for state in output_states:
           
            print("current state: ", state)
            
            this_grid_set = states[state]

            # Loop through the quantities.
            for quant_name in quant_list:

                ## Derive the point set
                # Down sample for velocity vector.
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
                    print(len(output_keys))
                    #input("Wait")

                # Note that: For "true", there is no output_keys in test_mode 3.

                print('Output quantity name: ', quant_name)

                # Initialization
                grid_set_quant = {}

                # check if this is a single or group quant_name
                if quant_name.endswith("group"):

                    print("group name")
                    if quant_name == "Msf_horizontal_displacement_group":
                        sub_quant_names = ["Msf_along_flow_displacement_amplitude",
                                           "Msf_along_flow_displacement_phase",
                                           "Msf_along_flow_displacement_phase_in_deg",

                                           "Msf_cross_flow_displacement_amplitude", 
                                           "Msf_cross_flow_displacement_phase",
                                           "Msf_horizontal_displacement_amplitude"]

                        for sub_quant_name in sub_quant_names:
                            grid_set_quant[sub_quant_name] = {}

                    else:
                        raise Exception("Undefined group name")
                    
                    for point in output_keys:

                        # The vector is not nan
                        if not np.isnan(this_grid_set[point][0,0]):

                            quant_group = self.tide_vec_to_quantity(tide_vec = this_grid_set[point], quant_name = quant_name, point = point, state=state)

                            # save it into grid_set_quant
                            for sub_quant_name in sub_quant_names:
                                grid_set_quant[sub_quant_name][point] = quant_group[sub_quant_name]

                # Normal single mode
                else:
                    sub_quant_names = [quant_name]
                    grid_set_quant[quant_name] = {}

                    #print(output_keys)
                    #print(len(output_keys))
                    #print(quant_name)
                    #input("Press Enter to continue...")

                    for point in output_keys:

                    
                        # Only record points where inverse problem can be done, Cm_p exists.
                        if not np.isnan(this_grid_set[point][0,0]):
                            # It is possible that some tides are not in the model. This is taken care of in the called method.

                            quant = self.tide_vec_to_quantity(tide_vec = this_grid_set[point],quant_name = quant_name, point=point, state=state)
    
                            # Here we record everything, if Cm_p exists, including nan futher filtered by tide_vec_to_quantity.
                            grid_set_quant[quant_name][point] = quant

                # Output the result
                #if state=="est" and quant_name == "secular_horizontal_speed":

                #print(grid_set_quant.keys())
                    #for point, v in grid_set_quant[quant_name].items():
                    #    if not np.isnan(v):
                    #        print(point,v)
                
                ########    End of extraction   #############

                ## Do phase correction for mean phase
                do_correction = True
                ## Do phase correction with the mean phase of true model
                do_correction_with_true = False

                for sub_quant_name in sub_quant_names:

                    if (state=='true' or state=='est') and 'phase' in sub_quant_name:

                        values = np.asarray(list(grid_set_quant[sub_quant_name].values()))
                        count = np.count_nonzero(~np.isnan(values))

                        if count>0:
                            if do_correction_with_true ==True and \
                                            sub_quant_name in phase_center and \
                                            state == "est":
                                print("In phase center: ",sub_quant_name)
                                center = phase_center[sub_quant_name]
                            else:
                                print("Calculate th mean phase")
                                center = np.nansum(values) /count

                            # Do correction
                            if do_correction and not "in_deg" in sub_quant_name:

                                print("Do mean phase shift")
                                for point in grid_set_quant[sub_quant_name].keys():
                                    grid_set_quant[sub_quant_name][point] -= center
                            else:
                                print("Skip mean phase shift: ", sub_quant_name)
                                print("The mean phase is: ", center)

                            if state=="true":
                                print("Give the mean phase of true model to phase center dictionary")
                                phase_center[sub_quant_name] = center

                ########    End of mean phase correction   #####

                #### Write to xyz file #####

                for sub_quant_name in sub_quant_names:
                    xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + sub_quant_name + '.xyz')
                    self.display.write_dict_to_xyz(grid_set_quant[sub_quant_name], xyz_name = xyz_name)


        return 0                

def main(iargs=None):

    inps = cmdLineParse(iargs)

    out = output(inps)

    output_states = []
    if out.output_true: output_states.append("true")
    if out.output_est:  output_states.append("est")
    if out.output_uq:   output_states.append("uq")

    if out.output_others: out.run_output_others()

    quant_list_name = inps.quant_list_name

    out.output_estimations(output_states, quant_list_name)
 
    if out.output_resid: out.run_output_residual()

    if out.output_difference:
        if out.proj=="Evans":
            # Evans
            out.run_output_difference(compare_id=620, compare_prefix='true')
        elif out.proj == "Rutford":
            # Rutford
            out.run_output_difference(compare_id=2020104656, compare_prefix='true')
        else:
            raise Exception()

    if out.output_analysis: out.output_analysis()

if __name__=='__main__':
    main()
