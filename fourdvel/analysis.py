#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in Nov, 2019

import os
import sys
import pickle
import numpy as np
import scipy
import multiprocessing
from display import display

import matplotlib.pyplot as plt
import datetime
import functools
from configure import configure

class analysis(configure):

    def __init__(self, param_file=None, simple_mode=False):

        if simple_mode:
            return

        if len(sys.argv)==2:
            param_file = sys.argv[1]

        elif param_file is None:

            print("Parameter file is required")
            raise Exception()

        super(analysis, self).__init__(param_file)

        self.preparation()

        test_id = self.test_id

        result_folder = self.estimations_dir
        self.this_result_folder = os.path.join(result_folder, str(test_id))

        # Load the reuslts
        self.load_everything()

        # Find the slr_name
        analysis_name = self.analysis_name
        if analysis_name.startswith("slr"):
            self.slr_name = analysis_name.split('_',1)[1]

    def set_task_name(self, task_name):

        self.task_name = task_name

    def load_everything(self):

        print("Loading everything...")

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

        self.point_set = self.grid_set_tide_vec.keys()

        print("size of tide_vec", len(self.grid_set_tide_vec.keys()))

        print("Loading done")

        return 0

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

                print("lon lat: ", lon,lat)

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

        nthreads = 10
        total_number = n_points
        
        divide = self.chop_into_threads(total_number, nthreads)

        func = self.strain_rate

        manager = multiprocessing.Manager()
        grid_set_I2 = manager.dict()
        grid_set_epsilon_xy = manager.dict()

        jobs = []
        for ip in range(nthreads):
            start_point = divide[ip]
            stop_point = divide[ip+1]

            p = multiprocessing.Process(target=func, args=(start_point, stop_point, grid_set_I2))

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

    def point_time_series(self):

        this_result_folder = self.this_result_folder
        test_id = self.test_id
        point = self.test_point

        lon, lat = point

        grid_set_true_tide_vec = self.grid_set_true_tide_vec
        grid_set_tide_vec = self.grid_set_tide_vec
        grid_set_tide_vec_uq = self.grid_set_tide_vec_uq

        # Find the result
        tide_vec = grid_set_tide_vec[point]
        print("tide vec")
        print(tide_vec)

        # Plot M2
        # Get M2 parameters
        tide_name = "N2"
        omega = self.tide_omegas[tide_name]
        quant_name = tide_name+ "_up_displacement_amplitude"
        disp_amp = self.tide_vec_to_quantity(tide_vec = tide_vec, quant_name = quant_name,  point=point, state="est")
        print("disp_amp: ", disp_amp)

        quant_name = tide_name+ "_up_displacement_phase_in_deg"
        disp_phase = self.tide_vec_to_quantity(tide_vec = tide_vec, quant_name = quant_name,  point=point, state="est")
        print( "disp_phase: ",disp_phase)

        t1 = (datetime.datetime(2017,6,1) - self.t_origin).days
        t2 = (datetime.datetime(2017,6,2) - self.t_origin).days

        taxis = np.arange(t1,t2,1/24)
        data = disp_amp * np.sin(omega*taxis + disp_phase/180*np.pi)

        plt.plot(taxis,data)
        plt.savefig("data.png")
        
        return 0


    def grounding_demonstration(self):

        tide_taxis = self.tide_taxis
        tide_data = self.tide_data

        tide_taxis = np.asarray(tide_taxis)
        tide_data = np.asarray(tide_data)

        delta = tide_taxis[1] - tide_taxis[0]

        x_left = 8005
        x_right = 8012
        clip = -2.5

        clipped_tide_data = tide_data.copy()
        clipped_tide_data[tide_data < clip] = clip

        fig = plt.figure(1,figsize=(10,10))
        ax = fig.add_subplot(111)

        ax.plot(tide_taxis, tide_data, color="gray",linewidth=8)
        ax.plot(tide_taxis, clipped_tide_data, color="black",linewidth=4)
        ax.plot([x_left, x_right], [clip, clip], color="blue", linestyle="--")

        ax.set_xlim(x_left, x_right)
        ax.set_xlabel("days (from 1992-01-01)", fontsize=15)

        # Find a desired case
        good_idx = np.where(np.logical_and(tide_taxis>8006, tide_data<-3)==True)
        m_idx = good_idx[0][0]
        s_idx = m_idx + int(4/delta)

        tm = tide_taxis[m_idx]
        ts = tide_taxis[s_idx]
        zm = tide_data[m_idx]
        zs = tide_data[s_idx]
        ax.plot(tm,zm,'r.',markersize=20)
        ax.plot(ts,zs,'k.',markersize=20)
        ax.plot(tm, clip, 'b.', markersize=20)

        ax.plot([8006, ts], [zs, zs], color="black", linestyle="--")
        ax.plot([8006, tm], [zm, zm], color="red", linestyle="--")

        eg = 0.05
        ax.plot([8006, 8006],[zm,zs],color="red",linewidth=5)
        ax.plot([8006-eg, 8006+eg],[zm,zm],color="red",linewidth=5)
        ax.plot([8006-eg, 8006+eg],[zs,zs],color="red",linewidth=5)

        ax.plot([8006.5, 8006.5],[clip,zs],color="blue",linewidth=5)
        ax.plot([8006.5-eg, 8006.5+eg],[clip,clip],color="blue",linewidth=5)
        ax.plot([8006.5-eg, 8006.5+eg],[zs,zs],color="blue",linewidth=5)

        fig.savefig("clip_demo.png")

    def point_set_prediction_evaluation(self, point_set, tracks_set):

        test_point = self.test_point
        grid_set = self.grid_set
        test_mode = self.test_mode

        # Find the data set
        (data_info_set, data_vec_set, noise_sigma_set, offsetfields_set, true_tide_vec_set) = self.data_set_formation(point_set, tracks_set, test_mode)

        linear_design_mat_set = self.build_G_set(point_set, offsetfields_set=offsetfields_set)

        # Perform estimation on each point
        analysis_set = {}

        for point in point_set:

            if self.single_point_mode and point!=test_point:
                continue

            # Find data and model for the test point
            data_info = data_info_set[point]
            data_vec = data_vec_set[point]
            offsetfields = offsetfields_set[point]
            design_mat = linear_design_mat_set[point]

            # Find the estimation
            tide_vec = self.grid_set_tide_vec[point]
            model_vec = self.tide_vec_to_model_vec(tide_vec)

            # Valid estimation exists
            if not np.isnan(model_vec[0,0]):
 
                data_vec_pred = np.matmul(design_mat, model_vec)
                data_vec_residual = data_vec - data_vec_pred
                
                # Find range residual
                data_vec_residual_range = data_vec_residual[0::2]

                # Find range residual
                data_vec_residual_azimuth = data_vec_residual[1::2]

                # Find slope of range and bed
                analysis_results = self.point_residual_vs_tidal_height(point, data_info, offsetfields, data_vec, data_vec_pred, data_vec_residual)

                range_residual_stats = [np.mean(data_vec_residual_range),   np.std(data_vec_residual_range)]

                azimuth_residual_stats = [np.mean(data_vec_residual_azimuth),np.std(data_vec_residual_azimuth)]

                analysis_results['range_residual_stats'] = range_residual_stats
                analysis_results['azimuth_residual_stats'] = azimuth_residual_stats
                # Mean of range
                analysis_set[point] = analysis_results

            else:
                analysis_set[point] = [np.nan]*20

        # Record and return
        print("Recording...")
        all_sets = {}
        all_sets['analysis_set'] = analysis_set

        return all_sets

    def slr_data_convert(self, data, label):
        if label.endswith("master_tide") or label.endswith("slave_tide"):
            return data
        else:
            return np.absolute(data)

    def point_residual_vs_tidal_height(self, point, data_info, offsetfields, data_vec, data_vec_pred, data_vec_residual):

        slr_name = self.slr_name

        # Analysis track by track
        data_num_total = 0

        tide_proxy = {}
        low_tide_proxy = {}
        high_tide_proxy = {}
        master_tide_proxy = {}
        slave_tide_proxy = {}

        range_data = {}
        azimuth_data = {}
        range_data_pred = {}
        azimuth_data_pred = {}
        range_residual = {}
        azimuth_residual = {}

        slr_results = {}
        track_name_list = []
        data_num_list = []

        # Find the tidal height data
        tide_taxis = self.tide_taxis
        delta = tide_taxis[1] - tide_taxis[0]
        tide_data = self.tide_data

        for i, track in enumerate(data_info):
            
            track_name, data_num = track
            track_num = track_name[0]

            # There is no available measurement in this track
            if data_num == 0:
                continue

            data_num_list.append(data_num)

            track_name_list.append(track_name)
            
            # Obtain the data of this track
            data_vec_track = data_vec[ data_num_total*2 : (data_num_total + data_num)*2 ]
            data_vec_track_range = data_vec_track[0::2, 0]
            data_vec_track_azimuth = data_vec_track[1::2, 0]

            # Obtain the data_pred of this track
            data_vec_pred_track = data_vec_pred[ data_num_total*2 : (data_num_total + data_num)*2 ]
            data_vec_pred_track_range = data_vec_pred_track[0::2, 0]
            data_vec_pred_track_azimuth = data_vec_pred_track[1::2, 0]

            # Obtain the offsetfields of this track            
            offsetfields_track = offsetfields[ data_num_total: data_num_total + data_num ]
            # Obtain the residual of this track
            data_vec_residual_track = data_vec_residual[ data_num_total*2 : (data_num_total + data_num)*2 ]
            
            range_residual_track = data_vec_residual_track[::2,0]
            azimuth_residual_track = data_vec_residual_track[1::2,0]

            ###################################################
            taxis = []
            tide_proxy_track = []
            for j, offsetfield in enumerate(offsetfields_track):
                t1 = (offsetfield[0] - self.t_origin.date()).days + offsetfield[4]
                t2 = (offsetfield[1] - self.t_origin.date()).days + offsetfield[4]
                taxis.append((t1+t2)/2)

                # The tidal height
                z1 = tide_data[int(np.round((t1 - tide_taxis[0])/delta))]
                z2 = tide_data[int(np.round((t2 - tide_taxis[0])/delta))]

                # Record the low tide 
                tide_proxy_track.append((z1, z2, track_num, offsetfield[0], offsetfield[1]))
            
            # Find the low tide
            low_tide_proxy_track=np.asarray([min(v[0],v[1]) for v in tide_proxy_track])
            high_tide_proxy_track=np.asarray([max(v[0],v[1]) for v in tide_proxy_track])
            master_tide_proxy_track=np.asarray([v[0] for v in tide_proxy_track])
            slave_tide_proxy_track=np.asarray([v[1] for v in tide_proxy_track])

            # Peform SLR for this track
            # Range residual vs low tide
            range_low_tide_slr_results_track = scipy.stats.linregress(low_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            range_high_tide_slr_results_track = scipy.stats.linregress(high_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            range_master_tide_slr_results_track = scipy.stats.linregress(master_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            range_slave_tide_slr_results_track = scipy.stats.linregress(slave_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            azimuth_low_tide_slr_results_track = scipy.stats.linregress(low_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            azimuth_high_tide_slr_results_track = scipy.stats.linregress(high_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            azimuth_master_tide_slr_results_track = scipy.stats.linregress(master_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            azimuth_slave_tide_slr_results_track = scipy.stats.linregress(slave_tide_proxy_track, self.slr_data_convert(range_residual_track, slr_name))

            # Record the results in the dictionaries
            # Tide
            tide_proxy[track_name] = tide_proxy_track
            low_tide_proxy[track_name] = low_tide_proxy_track
            high_tide_proxy[track_name] = high_tide_proxy_track
            master_tide_proxy[track_name] = master_tide_proxy_track
            slave_tide_proxy[track_name] = slave_tide_proxy_track

            # Data
            range_data[track_name] = data_vec_track_range
            azimuth_data[track_name] = data_vec_track_azimuth

            range_data_pred[track_name] = data_vec_pred_track_range
            azimuth_data_pred[track_name] = data_vec_pred_track_azimuth

            range_residual[track_name] = range_residual_track
            azimuth_residual[track_name] = azimuth_residual_track

            # SLR results
            slr_results[(track_name, "range_low_tide")] = range_low_tide_slr_results_track
            slr_results[(track_name, "range_high_tide")] = range_high_tide_slr_results_track
            slr_results[(track_name, "range_master_tide")] = range_master_tide_slr_results_track
            slr_results[(track_name, "range_slave_tide")] = range_slave_tide_slr_results_track

            slr_results[(track_name, "azimuth_low_tide")] = azimuth_low_tide_slr_results_track
            slr_results[(track_name, "azimuth_high_tide")] = azimuth_high_tide_slr_results_track
            slr_results[(track_name, "azimuth_master_tide")] = azimuth_master_tide_slr_results_track
            slr_results[(track_name, "azimuth_slave_tide")] = azimuth_slave_tide_slr_results_track
 
            # Move to next track
            data_num_total += data_num

        # Done with recording/analysis of seperate tracks

        ################ Analysis the results ######################
        # Merge the results from different tracks
        # Convert dict to nested list
        tide_proxy_list = [ tide_proxy[track_name] for track_name in tide_proxy.keys() ]

        low_tide_proxy_list = [ low_tide_proxy[track_name] for track_name in low_tide_proxy.keys() ]

        high_tide_proxy_list = [ high_tide_proxy[track_name] for track_name in high_tide_proxy.keys() ]
 
        master_tide_proxy_list = [ master_tide_proxy[track_name] for track_name in master_tide_proxy.keys() ]
 
        slave_tide_proxy_list = [ slave_tide_proxy[track_name] for track_name in slave_tide_proxy.keys() ]

        range_data_list = [ range_data[track_name] for track_name in range_data.keys() ]
        azimuth_data_list = [ azimuth_data[track_name] for track_name in azimuth_data.keys() ]

        range_data_pred_list = [ range_data_pred[track_name] for track_name in range_data_pred.keys() ]

        azimuth_data_pred_list = [ azimuth_data_pred[track_name] for track_name in azimuth_data_pred.keys() ]

        range_residual_list = [ range_residual[track_name] for track_name in range_residual.keys() ]
        azimuth_residual_list = [ azimuth_residual[track_name] for track_name in azimuth_residual.keys() ]

        # Convert nested list to list
        range_residual_merged = np.hstack(range_residual_list)
        azimuth_residual_merged = np.hstack(azimuth_residual_list)
        tide_proxy_merged = functools.reduce(lambda x,y:x+y,tide_proxy_list)
        low_tide_proxy_merged = np.hstack(low_tide_proxy_list)
        high_tide_proxy_merged = np.hstack(high_tide_proxy_list)
        master_tide_proxy_merged = np.hstack(master_tide_proxy_list)
        slave_tide_proxy_merged = np.hstack(slave_tide_proxy_list)

        range_data_merged = np.hstack(range_data_list)
        azimuth_data_merged = np.hstack(azimuth_data_list)

        range_data_pred_merged = np.hstack(range_data_pred_list)
        azimuth_data_pred_merged = np.hstack(azimuth_data_pred_list)

        range_residual_merged = np.hstack(range_residual_list)
        azimuth_residual_merged = np.hstack(azimuth_residual_list)

        # Set the proxy to regress on
        if slr_name.endswith("low_tide"):
            this_tide_proxy_list = low_tide_proxy_list
        
        elif slr_name.endswith("high_tide"):
            this_tide_proxy_list = high_tide_proxy_list

        elif slr_name.endswith("master_tide"):
            this_tide_proxy_list = master_tide_proxy_list

        elif slr_name.endswith("slave_tide"):
            this_tide_proxy_list = slave_tide_proxy_list

        else:
            print("Unknown proxy_name ", stop)

        this_tide_proxy_merged = np.hstack(this_tide_proxy_list)

        # Set the data to regress for
        if slr_name.startswith("range"):
            this_data_list = range_data_list
            this_data_pred_list = range_data_pred_list
            this_residual_list = range_residual_list

        elif slr_name.startswith("azimuth"):
            this_data_list = azimuth_data_list
            this_data_pred_list = azimuth_data_pred_list
            this_residual_list = azimuth_residual_list

        else:
            print("Unknown data_name ", stop)

        # Merge the data
        this_data_merged = np.hstack(this_data_list)
        this_data_pred_merged = np.hstack(this_data_pred_list)
        this_residual_merged = np.hstack(this_residual_list)

        # Convert list to numpy array
        data_num_merged = np.asarray(data_num_list)

        # track and dates
        td_tide_proxy_merged = np.asarray([(v[2],v[3],v[4]) for v in tide_proxy_merged])

        # SLR on range_merged
#        range_merged_slr_results = scipy.stats.linregress(this_tide_proxy_merged, np.absolute(range_residual_merged))
#        range_merged_slope = range_merged_slr_results[0]
#        range_merged_intercept = range_merged_slr_results[1]
#        range_merged_r_value = range_merged_slr_results[2]
#        range_merged_p_value = range_merged_slr_results[3]
#
#        # SLR on azimuth_merged
#        azimuth_merged_slr_results = scipy.stats.linregress(this_tide_proxy_merged, np.absolute(azimuth_residual_merged))
#        azimuth_merged_slope = azimuth_merged_slr_results[0]
#        azimuth_merged_intercept = azimuth_merged_slr_results[1]
#        azimuth_merged_r_value = azimuth_merged_slr_results[2]
#        azimuth_merged_p_value = azimuth_merged_slr_results[3]

        # Find the unique tracks
        track_tide_proxy_merged = np.asarray([v[2] for v in tide_proxy_merged])
        unique_tracks = np.unique(np.asarray(track_tide_proxy_merged))
        #print("unique tracks: ",unique_tracks)

        # Show the results
        show_analysis = False
        if self.single_point_mode: show_analysis = True

        if show_analysis:

            ### Show results with all tracks merged

            ### 1. Continous time series ####
            fig = plt.figure(100, figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.plot(tide_taxis, tide_data,"black",linewidth=0.5)
            #ax.plot(taxis, np.absolute(data_vec_residual_track[::2]),"r.",markersize=10)
            ax.set_xlim([taxis[0]-1,taxis[-1]+1])

            ax.set_ylim([-4,4])
            ax.set_ylabel("vertical displacement (m)",fontsize=15)

            # 2013.06.01 to 2014.06.01, 7823 days, 8188 days
            ax.set_xlim([7800, 8300])
            ax.set_xlabel("days from 1992/01/01",fontsize=15)
            ax.plot([7823,7823],[-5,5],linewidth=5)
            ax.plot([8188,8188],[-5,5],linewidth=5)
            ax.text(7823,-3.5, "2013/06/01",fontsize=15)
            ax.text(8188,-3.5, "2014/06/01",fontsize=15)

            figname="_".join((self.point2str(point), "vertical_displacement")) + ".png"
            fig.savefig(self.this_result_folder + '/' + figname)

            ### 2. range residual vs tide ###
            #fig = plt.figure(i+10, figsize=(10,10))
            #ax = fig.add_subplot(111)

            #for idx, track_num in enumerate(unique_tracks):

            #    track_name = track_name_list[idx]
            #    this_tide_proxy_track = this_tide_proxy_list[idx]
            #    range_residual_track = range_residual_list[idx]
            #    
            #    ax.plot(this_tide_proxy_track, range_residual_track, color=np.random.rand(3,)/2, marker=".",markersize=15,linestyle="None")

            ## Plot the regression line
            #ax.plot(this_tide_proxy_merged, this_tide_proxy_merged * range_merged_slope + range_merged_intercept,'k',linewidth=10)

            #ax.set_title("R value: " + str(round(range_merged_r_value,3)) + " P value: " + str(round(range_merged_p_value,3)))

            ## Configure the figure
            #ax.set_xlim([-3,3])
            #ax.set_xlim([-3,3])
            #
            ## Save the figure
            #figname = self.this_result_folder + '/' + self.point2str(point)+ '_' + slr_name + '.png'
            #fig.savefig(figname)

            #######  Show tracks seperately ###############
            for idx, current_track in enumerate(unique_tracks):

                track_name = track_name_list[idx]

                fig = plt.figure(idx+1, figsize=(12,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                label_A = "tide A > tide B" 
                label_B = "tide A < tide B"

                for it in range(len(this_tide_proxy_merged)):
    
                    trk = track_tide_proxy_merged[it]
    
                    if trk != current_track:
                        continue
    
                    x = this_tide_proxy_merged[it]
                    data = this_data_merged[it]
                    data_pred = this_data_pred_merged[it]
                    y = this_residual_merged[it]

                    mt = master_tide_proxy_merged[it]
                    st = slave_tide_proxy_merged[it]
    
                    d1 = td_tide_proxy_merged[it][1].strftime("%m%d")
                    d2 = td_tide_proxy_merged[it][2].strftime("%m%d")
    
                    # Left figure
                    
                    #if y>=0:
                    #    ax1.plot(x,y,'r.',markersize=15)
                    #else:
                    #    ax1.plot(x,y,'b.',markersize=15)
    
                    #y_shift = np.random.uniform()/4
                    #if y<0:
                    #    y_shift*=(-1)
                    #rand_color = np.random.rand(3,)/2
                    #ax1.text(x,y+y_shift, "_".join((str(trk),d1,d2,str(mt),str(st))),color=rand_color,fontsize=8)
                    #ax1.plot([x,x],[y, y+y_shift],color=rand_color)

                    ax1.plot(x,self.slr_data_convert(y, slr_name),'k.',markersize=15)
    
                    # Right figure
                    # Data
                    #ax2.plot(x,data,'r.',markersize=15)

                    # Prediction
                    #ax2.plot(x,data_pred,'b.',markersize=15)

                    # Residual
                    together = False
                    # together
                    if together:
                        ax2.plot(x,y,'k.',markersize=15)
                    else:
                        if mt>=st:
                            ax2.plot(x,y,'r.',markersize=15, label=label_A)
                            label_A = None
                        else:
                            ax2.plot(x,y,'b.',markersize=15, label=label_B)
                            label_B = None
                       
    
                    y_shift = np.random.uniform()/4
                    if mt<st:
                        y_shift*=(-1)
                    rand_color = np.random.rand(3,)/2
                
                    # Plot the data logistics    
                    #ax2.text(x,y+y_shift, "_".join((d1,d2,str(mt),str(st))),color=rand_color,fontsize=8, horizontalalignment='left')
                    #ax2.plot([x,x],[y, y+y_shift],color=rand_color)

                # Load the SLR results
                slr_slope = slr_results[(track_name, slr_name)][0]
                slr_intercept = slr_results[(track_name, slr_name)][1]
                slr_r_value = slr_results[(track_name, slr_name)][2]
                slr_p_value = slr_results[(track_name, slr_name)][3]

                # Plot the linear regression line
                ax1.plot(this_tide_proxy_merged, this_tide_proxy_merged * slr_slope + slr_intercept,'k',linewidth=10)
                ax1.set_title("Slope: " + str(round(slr_slope,3)) + " R value: " + str(round(slr_r_value,3)) + " P value: " + str(round(slr_p_value,3)))
    
                # Configure the figure
                ax1.set_xlim([-3,3])
                ax1.set_ylim([-1,1])
                #ax1.set_xlabel("tide_height_at_master_scene (m) ")
                ax1.set_xlabel("min(tide_height_at_master, tide_height_at_slave) (m)")

                ax1.set_ylabel("abs(data - model prediction) (m)")

                ax2.legend()
                ax2.set_xlim([-3,3])
                ax2.set_ylim([-1,1])
                ax2.set_xlabel("min(tide_height_at_master, tide_height_at_slave) (m)")
                #ax2.set_xlabel("tidal_height_at_master_scene (m)")
                ax2.set_ylabel("residual (m)")
   
                # Save the figure 
                figname="_".join((self.point2str(point), slr_name, str(current_track).zfill(3))) + ".png"
                fig.savefig(self.this_result_folder + '/' + figname)


            #######  Show the proxy tide for different tracks ###############

            fig = plt.figure(102, figsize=(7,7))
            ax = fig.add_subplot(111)

            for idx, current_track in enumerate(unique_tracks):

                track_name = track_name_list[idx]
                track_num = track_name[0]

                randcolor = np.random.rand(3,)/2

                ax.plot(np.zeros(shape=(data_num_list[idx],))+idx+1, low_tide_proxy_list[idx], color=randcolor, markersize=15, marker=".", linestyle="None")

                ax.text(idx + 1, 2.8, "track " + str(track_num), fontsize=12, color=randcolor)

            # Configuration
            ax.set_xlim([0,len(unique_tracks)+1])
            ax.set_xlabel("track", fontsize=15)
            ax.set_ylim([-3,3])
            ax.set_ylabel("m", fontsize=15)
            ax.xaxis.set_ticklabels([])

            figname="_".join((self.point2str(point), slr_name, 'proxy_tide_height')) + ".png"
            fig.savefig(self.this_result_folder + '/' + figname)

        # End of showing analysis

        # Key tracks
        # Identify the key track
        best_slr_idx = None

        fix_the_track = True
        if fix_the_track:
            key_tracks = [188, 99, 10, 232, 143]
            for key_track in key_tracks:
                for idx, track_num in enumerate(unique_tracks):
                    if track_num==key_track and data_num_merged[idx]>=20:
                        best_slr_idx = idx
                        break
                if best_slr_idx is not None:
                    break

        if best_slr_idx is None:
            # Pick out the most significant slr regression
            best_slr_p_value = float("inf")
            for idx, track_name in enumerate(track_name_list):
                slr_p_value = slr_results[(track_name, slr_name)][3]
                # At least 20 data points
                if slr_p_value<best_slr_p_value and data_num_merged[idx]>=20:
                    best_slr_p_value = slr_p_value
                    best_slr_idx = idx

        if best_slr_idx is not None:
            best_slr_results = {}
            track_name = track_name_list[best_slr_idx]
            best_slr_results['slope'] = slr_results[(track_name, slr_name)][0]
            best_slr_results['intercept'] = slr_results[(track_name, slr_name)][1]
            best_slr_results['r_value'] = slr_results[(track_name, slr_name)][2]
            best_slr_results['p_value'] = slr_results[(track_name, slr_name)][3]
            best_slr_results['min_proxy_tide'] = np.amin(this_tide_proxy_list[best_slr_idx])
            best_slr_results['track_name'] = track_name
            best_slr_results['track_num'] = track_name[0]

            best_slr_data_stats = {}
            best_slr_data_stats['data_mean'] = np.mean(this_residual_list[best_slr_idx])
            best_slr_data_stats['data_median'] = np.median(this_residual_list[best_slr_idx])
            best_slr_data_stats['data_std'] = np.std(this_residual_list[best_slr_idx])
            
            # Pickout values with proxy tide less than -2m
            picked_idx = this_tide_proxy_list[best_slr_idx]<-2
            picked_data = this_residual_list[best_slr_idx][picked_idx]
            best_slr_data_stats['picked_data_mean'] = np.mean(picked_data)
            best_slr_data_stats['picked_data_median'] = np.median(picked_data)
            best_slr_data_stats['picked_data_std'] = np.std(picked_data)
 
        else: 
            best_slr_results = {}
            best_slr_data_stats = {}            
 
        # Pick out the lowest tide touched at this point (min of low tide)
        lowest_tide = 100
        for idx, track_name in enumerate(track_name_list):
            lowest_tide_track = np.amin(low_tide_proxy[track_name])
            if lowest_tide_track < lowest_tide:
                lowest_tide = lowest_tide_track
                lowest_tide_track_name = track_name

        lowest_tide_results = {}
        lowest_tide_results["height"] = lowest_tide
        lowest_tide_results["track_name"] = lowest_tide_track_name
        lowest_tide_results["track_num"] = lowest_tide_track_name[0]

        # Save the results
        analysis_results = {}
        analysis_results["best_slr_results"] = best_slr_results
        analysis_results["best_slr_data_stats"] = best_slr_data_stats
        analysis_results["lowest_tide"] = lowest_tide_results

        if self.single_point_mode:
            print(analysis_results)
            #print(Done)

        return analysis_results

def main():

    ana = analysis(simple_mode=True)
    #ana.point_time_series()
    ana.get_tidal_model()
    ana.grounding_demonstration()

if __name__=="__main__":
    main()
