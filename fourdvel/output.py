#!/usr/bin/env python3

import os
import sys

import pickle

import numpy as np

import multiprocessing

from fourdvel import fourdvel
from display import display

import utm

import matplotlib.pyplot as plt


class output(fourdvel):

    def __init__(self):

        if len(sys.argv)==1:
            super(output,self).__init__()
        else:
            super(output,self).__init__(param_file = sys.argv[1])

        self.get_grid_set_velo()
        test_id = self.test_id

        result_folder = '/home/mzzhong/insarRoutines/estimations'
        self.this_result_folder = os.path.join(result_folder,str(test_id))

        self.display = display() 

        #with open(self.this_result_folder + '/' 
        #            + str(test_id) + '_' + 'grid_set_tide_vec.pkl','rb') as f:
        #    self.grid_set_tide_vec = pickle.load(f)

    def residual(self):

        this_result_folder = self.this_result_folder
        test_id = self.test_id

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_resid_of_secular.pkl','rb') as f:
            self.grid_set_resid_of_secular = pickle.load(f)

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_resid_of_tides.pkl','rb') as f:
            self.grid_set_resid_of_tides = pickle.load(f)


        grid_sets = {}
        grid_sets['resid_of_secular'] = self.grid_set_resid_of_secular
        grid_sets['resid_of_tides'] = self.grid_set_resid_of_tides

        state = 'est'
        comps = ['range','azimuth']

        for misfit_name in grid_sets.keys():
            for comp in comps:

                quant_name = '_'.join([misfit_name, comp])

                print('Output quantity name: ', quant_name)
                grid_set_quant = {}

                this_grid_set = grid_sets[misfit_name]
                output_keys = this_grid_set.keys()

                # For all available points in grid_set.
                for point in output_keys:
                
                    # Four dim, range_mean, range_std, azimuth_mean, azimuth_std
                    quant = this_grid_set[point]

                    if comp == 'range':
                        grid_set_quant[point] = quant[1] # mean and std
                    elif comp == 'azimuth':
                        grid_set_quant[point] = quant[3]

                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0

    def load_master_model(self,num,prefix='est'):

        this_result_folder = self.this_result_folder
        # Load all the results.
        if prefix == 'true':
            filename = '/home/mzzhong/insarRoutines/estimations/'+str(num)+'/'+str(num)+'_grid_set_true_tide_vec.pkl'
        else:
            filename = '/home/mzzhong/insarRoutines/estimations/'+str(num)+'/'+str(num)+'_grid_set_tide_vec.pkl'
           
        with open(filename,'rb') as f:
            self.grid_set_master_model_tide_vec = pickle.load(f)
        return 0

    def load_slave_model(self,num,prefix='est'):

        this_result_folder = self.this_result_folder
        # Load all the results.

        if prefix == 'true':
            filename = '/home/mzzhong/insarRoutines/estimations/'+str(num)+'/'+str(num)+'_grid_set_true_tide_vec.pkl'
        else:
            filename = '/home/mzzhong/insarRoutines/estimations/'+str(num)+'/'+str(num)+'_grid_set_tide_vec.pkl'
 
        with open(filename,'rb') as f:
            self.grid_set_slave_model_tide_vec = pickle.load(f)
        return 0

    def load_everything(self):

        this_result_folder = self.this_result_folder
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

    def output_differences(self, compare_id, compare_prefix):

        print('Ouput difference...')

        this_result_folder = self.this_result_folder
        test_id = self.test_id

        self.load_master_model(test_id)
        self.load_slave_model(num=compare_id,prefix=compare_prefix)

        #self.load_true_est_uq()

        quant_list = [ 'secular_horizontal_velocity_difference' ]

        for quant_name in quant_list:
            
            print('Output quantity nane: ', quant_name)
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

    def output_others(self):

        this_result_folder = self.this_result_folder
        test_id = self.test_id

        state='est'
        quant_name='other_1'

        with open(this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_other_1.pkl','rb') as f:
            this_grid_set = pickle.load(f)

        print('Output quantity name: ', quant_name)
        grid_set_quant = {}

        output_keys = this_grid_set.keys()

        for point in output_keys:
        
            # Record everything, if Cm_p exists, including nan futher filtered by tide_vec_to_quantity.
            grid_set_quant[point] = this_grid_set[point]

        # Write to xyz file.
        xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
        self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)


    def output_estimations(self):

        modeling_tides = self.modeling_tides
        n_modeling_tide = self.n_modeling_tides

        this_result_folder = self.this_result_folder
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

                        'Msf_horizontal_displacement_amplitude',

                        'Mf_horizontal_displacement_amplitude',

                        'M2_horizontal_displacement_amplitude',

                        'O1_horizontal_displacement_amplitude',

                        'M2_up_displacement_amplitude',
                        'M2_up_displacement_phase',
                        'O1_up_displacement_amplitude',
                        'O1_up_displacement_phase']



#        quant_list = [
#                        'M2_up_displacement_amplitude',
#                        'M2_up_displacement_phase',
#                        'O1_up_displacement_amplitude',
#                        'O1_up_displacement_phase']
#

        #quant_list = ['secular_horizontal_velocity']


        states = {}
        states['true'] = self.grid_set_true_tide_vec
        states['est'] = self.grid_set_tide_vec
        states['uq'] = self.grid_set_tide_vec_uq


        # Look through the sets
        for state in states.keys():
            
            print(state)
            
            this_grid_set = states[state]

            # Loop through the quantities.
            for quant_name in quant_list:

                print('Output quantity name: ', quant_name)
                grid_set_quant = {}

                # Down-sample for velocity vector.
                if quant_name == 'secular_horizontal_velocity':
                    output_keys = []
                    for point in this_grid_set.keys():

                        lon, lat = point
                        lon_ind = np.round(lon/self.lon_step)
                        lat_ind = np.round(lat/self.lat_step) 

                        if lon_ind % 10==0 and lat_ind % 10==0:
                            output_keys.append((lon,lat))

                    output_keys = set(output_keys)

                else:
                    output_keys = this_grid_set.keys()

                # For "true", there is no output_keys in test_mode 3.

                # For all available points in grid_set.
                center = 0
                count = 0
                for point in output_keys:
                
                    # Only record points where inverse problem can be done, Cm_p exists.
                    if not np.isnan(this_grid_set[point][0,0]):
                        # It is possible that some tides are not in the model. This is taken care of in the called method.
                        quant = self.tide_vec_to_quantity(tide_vec = this_grid_set[point],quant_name = quant_name, point=point, state=state)

                        # Record everything, if Cm_p exists, including nan futher filtered by tide_vec_to_quantity.
                        grid_set_quant[point] = quant

                        if (state=='true' or state=='est') and 'phase' in quant_name and not np.isnan(quant):
                            count = count + 1
                            center = center + quant

                # Correction for mean phase.
                if (state == 'true' or state=='est') and 'phase' in quant_name and count > 0:
                    center = center/count
                    for point in grid_set_quant.keys():
                        grid_set_quant[point] = grid_set_quant[point] - center


                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0                

def main():

    out = output()

    # Analysis the results. 
    out.output_estimations()
    out.output_differences(compare_id=620, compare_prefix='true')
    out.residual()

    #out.output_others()

if __name__=='__main__':
    main()
