#!/usr/bin/env python3

import os
import pickle

import numpy as np

import multiprocessing

from fourdvel import fourdvel
from display import display

import utm

class analysis(fourdvel):

    def __init__(self):

        super(analysis,self).__init__()

        test_id = self.test_id
        result_folder = '/home/mzzhong/insarRoutines/estimations'
        self.this_result_folder = os.path.join(result_folder,str(test_id))

        self.display = display() 

        with open(self.this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_tide_vec.pkl','rb') as f:
            self.grid_set_tide_vec = pickle.load(f)

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

    def second_invariant(self,strain):

        I2 = 0.5 * (np.trace(strain)**2 - np.trace(np.matmul(strain,strain)))

        return I2

    def strain_rate(self, start_point, stop_point, grid_set_I2):

        alpha = 1000 # 1 km

        point_set = sorted(self.point_set)

        count_point = 0

        # Record the results.
        I2_set = {}
        epsilon_xy = {}

        for point in point_set:
        #for point in [(-77, -76.8)]:
            count_point = count_point+1
            if count_point >= start_point and count_point < stop_point:
                count_point = count_point + 1

                lon, lat = point

                lon_ref = lon
                lat_ref = lat

                print(lon,lat)

                # Nearby points.
                lon_range = self.round1000(np.arange(lon-0.5, lon+0.5, self.lon_step))
                lat_range = self.round1000(np.arange(lat-0.1, lat+0.1, self.lat_step))

                sens_rows = []
                sens_obs = []

                count = 0
                count_all = 0
                for lon_obs in lon_range:
                    for lat_obs in lat_range:
                        count_all = count_all + 1

                        point_obs = (lon_obs, lat_obs)
                        delta = self.latlon_distance(lon,lat,lon_obs,lat_obs)
                        delta = self.km2m(delta)
                        
                        # Weighted
                        w = np.exp(-delta**2/(2*alpha**2))
                        if  w > 1e-2 and (lon_obs,lat_obs) in point_set:
                            count = count+1
                            delta_x = self.latlon_distance(lon_obs,lat_ref,lon_ref,lat_ref)
                            delta_x = self.km2m(delta_x)
                            delta_y = self.latlon_distance(lon_ref,lat_obs,lon_ref,lat_ref)
                            delta_y = self.km2m(delta_y)

                            row1 = np.zeros(shape=(1,6))
                            row1[0,:] = [1,delta_x,delta_y, 0, 0, 0] #unit: m
                            row2 = np.zeros(shape=(1,6))
                            row2[0,:] = [0,0,0, 1, delta_x, delta_y]
                        
                            ve_obs = self.grid_set_tide_vec[(lon_obs,lat_obs)][0] #unit: m
                            vn_obs = self.grid_set_tide_vec[(lon_obs,lat_obs)][1]

                            w = 1
                            sens_obs.append(ve_obs * w)
                            sens_obs.append(vn_obs * w)
                            sens_rows.append(row1 * w)
                            sens_rows.append(row2 * w)

                print(count_all, count)

                if count > 10:
                    sens_rows = tuple(sens_rows)
                    G = np.vstack(sens_rows)

                    b = np.zeros(shape=(len(sens_obs),1))
                    b[:,0] = sens_obs

                    m = np.matmul(np.linalg.pinv(G), b)

                    strain = np.zeros(shape=(2,2))
                    strain[0,0] = m[1,0]
                    strain[0,1] = m[2,0]
                    strain[1,0] = m[4,0]
                    strain[1,1] = m[5,0]

                    #print(strain)
                    I2_set[point] = self.second_invariant(strain)

        grid_set_I2.update(I2_set)

    def parallel_driver(self):

        point_set = self.grid_set_tide_vec.keys()
        test_id = self.test_id

        n_points = len(point_set)

        n_threads = 10
        total_number = n_points
        divide = self.chop_into_threads(total_number, nthreads)

        func = self.strain

        manager = multiprocessing.Manager()
        grid_set_I2 = manager.dict()
        grid_set_epsilon_xy = manager.dict()

        jobs = []
        for ip in range(nthreads):
            start_point = divide[ip]
            stop_point = divide[ip+1]

            p = multiprocessing.Process(target=func, args=(start_point, stop_point,grid_set_I2))

            jobs.append(p)
            p.start()

        for ip in range(nthreads):
            jobs[ip].join()

        # Save the results.
        grid_set_I2 = dict(grid_set_I2)

        with open(self.this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_I2', 'wb') as f:
            
            pickle.dump(grid_set_I2, f)

        return

    def output_strain_rate(self):

        test_id = self.test_id

        with open(self.this_result_folder + '/' 
                    + str(test_id) + '_' + 'grid_set_I2.pkl','rb') as f:
            self.grid_set = I2 = pickle.load(f)

        xyz_name = os.path.join(self.this_result_folder,str(test_id)+'_'+'grid_set_I2.xyz')

        self.display.write_dict_to_xyz(grid_set_I2, xyz_name = xyz_name)

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
                
                    quant = this_grid_set[point]

                    if comp == 'range':
                        grid_set_quant[point] = quant[1] # mean and std
                    elif comp == 'azimuth':
                        grid_set_quant[point] = quant[3]

                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0


    def output_estimations(self):

        modeling_tides = self.modeling_tides
        n_modeling_tide = self.n_modeling_tides

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


#        quant_list = [  'secular_horizontal_speed',
#                        'secular_east_velocity',
#                        'secular_north_velocity',
#                        'Msf_horizontal_displacement_amplitude',
#                        'Msf_east_displacement_amplitude',
#                        'Msf_north_displacement_amplitude',
#                        'M2_up_displacement_amplitude',
#                        'O1_up_displacement_amplitude']

        quant_list = [  'secular_horizontal_speed',
                        'secular_east_velocity',
                        'secular_north_velocity',
                        'secular_horizontal_velocity',
                        'Msf_horizontal_displacement_amplitude',
                        'M2_up_displacement_amplitude',
                        'O1_up_displacement_amplitude']

        #quant_list = ['secular_horizontal_velocity']


        states = {}
        states['true'] = self.grid_set_true_tide_vec
        states['est'] = self.grid_set_tide_vec
        states['uq'] = self.grid_set_tide_vec_uq

        # Look through the sets
        for state in states.keys():

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

                # For all available points in grid_set.
                for point in output_keys:
                
                    # Only record valid points.
                    if not np.isnan(this_grid_set[point][0,0]):
                        quant = self.tide_vec_to_quantity(tide_vec = this_grid_set[point],quant_name = quant_name)

                        # Record everything, including nan.
                        grid_set_quant[point] = quant

                # Write to xyz file.
                xyz_name = os.path.join(this_result_folder, str(test_id) + '_' + state + '_' + quant_name + '.xyz')
                self.display.write_dict_to_xyz(grid_set_quant, xyz_name = xyz_name)

        return 0

def main():
    runAna = analysis()

    # Analysis the results. 
    #runAna.output_estimations()
    runAna.residual()

    #run_ana.parallel_driver()

if __name__=='__main__':
    main()

