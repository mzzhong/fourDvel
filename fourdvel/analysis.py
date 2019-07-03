#!/usr/bin/env python3

import os
import pickle

import numpy as np

import multiprocessing

from fourdvel import fourdvel
from display import display

import utm

import matplotlib.pyplot as plt

class analysis(fourdvel):

    def __init__(self):

        super(analysis,self).__init__()

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


######################

    def check_fitting_set(self, point_set, data_info_set, offsetfields_set, design_mat_set, design_mat_enu_set, data_vec_set, model_vec_set, tide_vec_set):

        fitting_set={}
        
        for i, point in enumerate(point_set):
            #G = design_mat_set[point]
            m = model_vec_set[point]
            # G is valid
            if not np.isnan(m[0,0]):
                fitting_set[point] = self.check_fitting(point, data_info_set[point], offsetfields_set[point], design_mat_set[point], design_mat_enu_set[point], data_vec_set[point], model_vec_set[point], tide_vec_set[point])
            else:
                fitting_set[point] = np.nan

        return fitting_set

    def check_fitting(self, point, data_info, offsetfields, design_mat, design_mat_enu, data_vec, model_vec, tide_vec):

        G = design_mat
        G_enu = design_mat_enu
        
        m = model_vec
        pred = np.matmul(G,m)
        pred_enu = np.matmul(G_enu,m)
        obs = data_vec

        #print(offsetfields)
        #print(tide_vec)
        #print(self.modeling_tides)
        #print(self.n_modeling_tides)

        # switch
        plot = 0

        if plot:
            #print('continous prediction')
            t_axis = np.arange(-600,600,0.0005)
            con_pred = np.zeros(shape=t_axis.shape)

            for k, tide_name in enumerate(self.modeling_tides):
                for t in range(3):
                    if t==2:
                        ampU = tide_vec[3+k*6+t]
                        phaseU = tide_vec[3+k*6+t+3]
                        omega = 2*np.pi / self.tide_periods[tide_name]
    
                        dis_ampU = self.velo_amp_to_dis_amp(ampU, tide_name)
                        dis_phaseU = self.velo_phase_to_dis_phase(phaseU, tide_name)
    
                        #print(tide_name, dis_ampU)
    
                        con_pred = con_pred + dis_ampU * np.sin(omega*t_axis + dis_phaseU)

        #################
        N = len(pred)
        #print(data_info, N)
        #print(offsetfields, len(offsetfields))

        # Get number of tracks.
        Nt = 0
        valid_tracks = []
        for i, info in enumerate(data_info):
            # Not empty:
            if info[1]>0:
                Nt += 1
                valid_tracks.append(info)


        #print('valid_tracks: ',valid_tracks)
        j=0
        k=0
        val_container = []

        for i, info in enumerate(valid_tracks):
            k += info[1]

            # Only range offset
            N = k-j

            if plot:
                #ax = fig.add_subplot(Nt,1,i+1)
                fig = plt.figure(1,figsize=(15,10))
                # Continous data.
                ax = fig.add_subplot(211)
                ax.plot(t_axis, con_pred,color='k')
                
                max_len = self.tide_periods['Msf'] 
                ax.set_xlim([0, max_len])
                ax = fig.add_subplot(212)

            for i, offsetfield in enumerate(offsetfields[j:k]):

                t_a = (offsetfield[0] - self.t_origin.date()).days + offsetfields[i][4]
                t_b = (offsetfield[1] - self.t_origin.date()).days + offsetfields[i][4]

                ## Vectors.
                vec_range = offsetfield[2]
                vec_azimuth = offsetfield[3]

                # horiz of range vec
                norm = np.sqrt(vec_range[0]**2 + vec_range[1]**2)
                vec_range_horiz = np.asarray((vec_range[0]/norm, vec_range[1]/norm, 0))

                #print(vec_range, vec_azimuth, vec_range_horiz)

                pred_range_offset = pred[(j+i)*2,0]
                obs_range_offset = obs[(j+i)*2,0]

                pred_azimuth_offset = pred[(j+i)*2+1,0]
                obs_azimuth_offset = obs[(j+i)*2+1,0]

                # Offset projected in (E, N, U) components.
                pred_e = pred_enu[(j+i)*3,0]
                pred_n = pred_enu[(j+i)*3+1,0]
                pred_u = pred_enu[(j+i)*3+2,0]

                offset_enu = np.asarray((pred_e,pred_n,pred_u))
                #print('offset_enu: ', offset_enu)

                # Offset along the range_horizontal component
                offset_range_horiz = np.dot(offset_enu,vec_range_horiz)
                #print('offset_range_horiz: ',offset_range_horiz)

                # Calculate vertical offset
                inc_ang = np.arccos(vec_range[2])
                #print(inc_ang/np.pi*180)
                offset_vertical_obs = ( obs_range_offset - np.sin(inc_ang) * offset_range_horiz) / np.cos(inc_ang)

                offset_vertical_pred = ( pred_range_offset - np.sin(inc_ang) * offset_range_horiz) / np.cos(inc_ang)

                #print('############# offset_vertical_obs: ', offset_vertical_obs)

                ###############
                # Pick the value to plot.
                misfit = abs(pred_range_offset - obs_range_offset)
                offset_vertical_diff = offset_vertical_obs - offset_vertical_pred

                show_val = offset_vertical_diff

                val_container.append(show_val)

                ###############
                if plot:
                    t_a = t_a % max_len
                    t_b = t_b % max_len
    
                    if t_a < t_b:
                        ax.plot([t_a,t_b],[show_val,show_val],'k')
                    else:
                        ax.plot([t_a, max_len],[show_val,show_val],'k')
                        ax.plot([0,t_b],[show_val,show_val],'k')
            
            if plot:
                ax.set_xlim([0, max_len])
                #title = str(info[0][0])+'_'+str(info[0][1])
                #ax.set_title(title)
                #fig.savefig(title + '_dif.png')
                #plt.close()

            j+= info[1]

        if plot:
            fig.savefig('dif_all.png')

        #return np.mean(val_container)
        return np.std(val_container)


