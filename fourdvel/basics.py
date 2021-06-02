#!/usr/bin/env python3

# Author: Minyan Zhong
# Aug, 2018
import os
import numpy as np
import pandas as pd
import datetime
import math
import pathlib

class basics():

    def __init__(self):

        # Set NAN value
        self.INT_NAN = -99999

        # Set float to integer convertion
        self.float2int = 10**5

        # Directory of Antarctic data
        self.Ant_Data_dir = '/net/kraken/bak/mzzhong/Ant_Data'

        self.datasets = ["csk","s1"]

        # Time origin (important)
        self.t_origin = datetime.datetime(1992,1,1,0,0,0,0)

        # Tides
        self.tide_periods = {}
        tide_periods = self.tide_periods
        tide_periods['K2'] = 0.49863484
        tide_periods['S2'] = 0.50000000
        tide_periods['M2'] = 0.51752505
        tide_periods['N2'] = 0.52743115

        tide_periods['K1'] = 0.99726967
        tide_periods['P1'] = 1.00274532
        tide_periods['O1'] = 1.07580578
        tide_periods['Q1'] = 1.11951458

        tide_periods['Mf'] = 13.66083078
        tide_periods['Msf'] = 14.76529444
        tide_periods['Mm'] = 27.55463190
        tide_periods['Ssa'] = 182.62818021
        tide_periods['Sa'] = 365.25636042

        tide_periods['M4'] = tide_periods['M2']/2 
        tide_periods['S4'] = tide_periods['S2']/2
        tide_periods['MS4'] = 1/(1/tide_periods['M2'] + 1/tide_periods['S2'])

        self.tide_short_period_members = ["K2","S2","M2","N2","K1","P1","O1","Q1"]
        self.tide_long_period_members = ["Mf","Msf","Mm","SSa","Sa"]

        self.tides = list(tide_periods.keys())

        self.tide_omegas = {}
        for tide_name in self.tides:
            self.tide_omegas[tide_name] = 2*np.pi/tide_periods[tide_name]

    def latlon_distance(self,lon1,lat1,lon2,lat2):
        
        R = 6371
        lat1 = np.deg2rad(lat1)
        lon1 = np.deg2rad(lon1)
        lat2 = np.deg2rad(lat2)
        lon2 = np.deg2rad(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c

        return distance
    
    def km2m(self, value):
        return value*1000

    def deg2km(self,deg):
        return deg/180*np.pi*6371

    def deg2m(self,deg):
        return deg/180*np.pi*6371*1000

    def wrapped(self,phase):
        
        return (phase + np.pi) % (2*np.pi) - np.pi

    def wrapped_deg(self,phase):
        
        return (phase + 180) % (360) - 180

    def round1000(self, value):
        return np.round(value*1000)/1000

    # Make all coordinates integer
    def round_int_5dec(self, value):
        return np.round(value*100000).astype(int)

    def int5d_to_float(self, x):

        if isinstance(x, list) or isinstance(x, tuple):
            return [num/(10**5) for num in x]
        else:
            return x/(10**5)

    def float_to_int5d(self, x):

        if isinstance(x, list) or isinstance(x, tuple):
            return [int(round(num*10**5)) for num in x]
        else:
            return int(round(x*10**5))

    def print_int5d(self,x):
        print(np.asarray(x)/10**5)
        return 0

    def point2str(self, x):
        lon, lat = x
        return str(lon)+'_'+str(lat)

    def float_lonlat_to_int5d(self,x):
        lon, lat = x
        return (int(round(lon*10**5)), int(round(lat*10**5)))

    def round_to_grid_points(self, x, re):
        return np.round(x * re)/re

    def comp_name(self,comp):

        if comp == 0:
            return 'East component'
        elif comp == 1:
            return 'North component'
        elif comp == 2:
            return 'Up component'

    def m2cm(self,value):
        return value * 100

    def rad2deg(self,value):
        return value /np.pi * 180

    def deg2minute(self,value, tide_name):
        # minute
        return value /360 * self.tide_periods[tide_name] * 24* 60

    def deg2day(self,value, tide_name):
        # days
        return value /360 * self.tide_periods[tide_name]

    def cm2m(self,value):
        return value / 100

    def float_rounding(self,value,precision):
        return np.round(value * precision)/precision

    def velo_amp_to_dis_amp(self,velo,tide_name):
        return velo / self.tide_omegas[tide_name]

    def dis_amp_to_velo_amp(self, dis_amp, tide_name):
        return dis_amp * self.tide_omegas[tide_name]

    def velo_phase_to_dis_phase(self, phase, deg = False):
        if deg==True:
            return self.wrapped_deg(phase - 90)
        elif deg==False:
            return self.wrapped(phase - np.pi/2)
        else:
            raise Exception()

    def dis_phase_to_velo_phase(self, phase, deg = False):
        if deg==True:
            return self.wrapped_deg(phase + 90)
        elif deg==False:
            return self.wrapped(phase + np.pi/2)
        else:
            raise Exception()

    def unit_vec(self, v1, v2=None):
        if v2:
            return np.asarray([v1,v2,np.sqrt(1-v1**2-v2**2)])
        else:
            return np.asarray([v1,np.sqrt(1-v1**2),0])

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

    def satellite_constants(self):

        self.track_timefraction = {}
        track_timefraction = self.track_timefraction

        if self.proj == "Rutford":

            # CSK.
            fid = open('/net/kraken/nobak/mzzhong/CSK-Rutford/csk_times_rutford.txt')
            csk_times = fid.readlines()
            fid.close()

            # 32 tracks in total
            tracks = self.csk_tracks
            for it, track_num in enumerate(tracks):
                track_timefraction[('csk',track_num)] = float(csk_times[it])

            # S1AB, only three tracks cover Rutford Ice Stream
            # Time of scene.
            t37 = datetime.time(6,26,45)
            track_timefraction[('s1',37)] = (t37.hour * 3600 + t37.minute*60 + t37.second)/(24*3600)
 
            t65 = datetime.time(4,34,10)
            track_timefraction[('s1',65)] = (t65.hour * 3600 + t65.minute*60 + t65.second)/(24*3600)

            t7 = datetime.time(5,6,30)
            track_timefraction[('s1',7)] = (t7.hour * 3600 + t7.minute*60 + t7.second)/(24*3600)

        elif self.proj == "Evans":

            # CSK.
            fid = open('/net/kraken/nobak/mzzhong/CSK-Evans/csk_times.txt')
            csk_times = fid.readlines()
            fid.close()
    
            # 22 tracks.
            tracks = range(22) 
            for track_num in tracks:
                track_timefraction[('csk',track_num)] = float(csk_times[track_num])
    
            # S1AB.
            # Time of scene.
            t37 = datetime.time(6,26,45)
            track_timefraction[('s1',37)] = (t37.hour * 3600 + t37.minute*60 + t37.second)/(24*3600)
            
            t52 = datetime.time(7,7,30)
            track_timefraction[('s1',52)] = (t52.hour * 3600 + t52.minute*60 + t52.second)/(24*3600)
    
            t169 = datetime.time(7,40,30)
            track_timefraction[('s1',169)] = (t169.hour * 3600 + t169.minute*60 + t169.second)/(24*3600)
    
            t65 = datetime.time(4,34,10)
            track_timefraction[('s1',65)] = (t65.hour * 3600 + t65.minute*60 + t65.second)/(24*3600)
    
            t7 = datetime.time(5,6,30)
            track_timefraction[('s1',7)] = (t7.hour * 3600 + t7.minute*60 + t7.second)/(24*3600)
    
            # New tracks
            t50 = datetime.time(3,53,40)
            track_timefraction[('s1',50)] = (t50.hour * 3600 + t50.minute*60 + t50.second)/(24*3600)
    
            t64 = datetime.time(2,57,0)
            track_timefraction[('s1',64)] = (t64.hour * 3600 + t64.minute*60 + t64.second)/(24*3600)
    
            #t49 = datetime.time(2,16,0)
            #track_timefraction[('s1',49)] = (t49.hour * 3600 + t49.minute*60 + t49.second)/(24*3600)
    
            #print(track_timefraction)

        return 0

    def csk_evans_min_coverage(self):
        min_cov = {}
        for it in range(22):
            min_cov[it] = 0
        min_cov[3] = 6
        min_cov[4] = 6
        min_cov[5] = 5
        min_cov[7] = 6
        min_cov[9] = 5.1
        min_cov[11] = 3
        min_cov[13] = 3.3
        min_cov[14] = 7
        min_cov[16] = 7
        min_cov[17] = 7
        min_cov[18] = 8

        return min_cov

    def read_tides_params(self, params_file=None):

        # Read in tide params exported from TMD MATLAB code
        print("Reading tide model...")
        
        if params_file is None:
            raise Exception()
            #params_file = "/net/kamb/ssd-tmp1/mzzhong/tides_model/TMD_Matlab_Toolbox_v2.5/TMD/results/RIS_tides_params.txt"
        
        f = open(params_file)
        lines = f.readlines()
        f.close()

        n_tides = len(lines)//6
        print("n_tides: ", n_tides)

        ris_tides_params={}
        tide_names = []
        for i in range(n_tides):
            tide_name1 = lines[i*6].rstrip()


            cos_amp = float(lines[i*6+1].split()[1])
            sin_amp = float(lines[i*6+2].split()[1])
            ph = float(lines[i*6+3].split()[1])
            pf = float(lines[i*6+4].split()[1])
            pu = float(lines[i*6+5].split()[1])

            print(tide_name1, cos_amp, sin_amp, ph, pf, pu)

            ris_tides_params[(tide_name1,"cos_x")] = cos_amp
            ris_tides_params[(tide_name1,"sin_x")] = sin_amp
            ris_tides_params[(tide_name1,"ph")] = ph
            ris_tides_params[(tide_name1,"pf")] = pf
            ris_tides_params[(tide_name1,"pu")] = pu

            tide_names.append(tide_name1.capitalize())

        # Convert to standard params
        n_tides = len(tide_names)
        for i, tide_name in enumerate(tide_names):
            tide_name1 = tide_name.lower()
            
            # Get the extracted value
            cos_amp = ris_tides_params[(tide_name1,"cos_x")]
            sin_amp = ris_tides_params[(tide_name1,"sin_x")]
            ph = ris_tides_params[(tide_name1,"ph")]
            pf = ris_tides_params[(tide_name1,"pf")]
            pu = ris_tides_params[(tide_name1,"pu")]

            #exp_sig = pf * cos_amp * np.cos(omega * taxis + ph + pu) \
            #            - pf * sin_amp * np.sin(omega * taxis + ph + pu); 

            # Get normal amplitude and phase
            phi = ph+pu

            cos_coe =   pf * (cos_amp * np.cos(phi) - sin_amp * np.sin(phi))
            sin_coe = - pf * (cos_amp * np.sin(phi) + sin_amp * np.cos(phi))

            #exp_sig = cos_coe * np.cos(omega * taxis) + sin_coe * np.sin(omega * taxis);
            tide_amp = np.sqrt(cos_coe**2 + sin_coe**2)
            tide_phase = np.arctan2(cos_coe, sin_coe)
            tide_phase_deg = tide_phase / np.pi * 180
            tide_phase_min = tide_phase / (2 * np.pi) * self.tide_periods[tide_name] * 60 * 24

            # Save the amplitude and phase
            ris_tides_params[(tide_name1, "tide_amp")] = tide_amp

            ris_tides_params[(tide_name1, "tide_phase")] = tide_phase            
            ris_tides_params[(tide_name1, "tide_phase_deg")] = tide_phase_deg
            ris_tides_params[(tide_name1, "tide_phase_min")] = tide_phase_min

        return (tide_names, ris_tides_params)

    def write_dict_to_xyz(self, show_dict, xyz_name, f2i=None):
        
        # Write to txt file.
        f = open(xyz_name,'w')
        #cap=1000
        cap=10**10
        for key in sorted(show_dict.keys()):
            lon, lat = key

            if f2i is None:
                lon, lat = self.int5d_to_float([lon,lat])
            else:
                lon, lat = lon/f2i, lat/f2i

            value = show_dict[key]

            # More then one value (velocity vector).
            if isinstance(value,tuple):
                if not np.isnan(value[0]):
                    record = str(lon)+' '+str(lat)
                    for irec in range(len(value)):
                        record = record + ' ' + str(value[irec])
                    record = record + '\n'
                    f.write(record)
            else:

                # Only output valid values.
                value = min(value,cap)
                if not np.isnan(value):
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()

        return

    def write_datamat_to_xyz(self, datamat, xaxis, yaxis, xyz_name, f2i=None):
       
        f = open(xyz_name,'w')

        xaxis = np.asarray(xaxis) / f2i
        yaxis = np.asarray(yaxis) / f2i

        for i in range(len(yaxis)):
            for j in range(len(xaxis)):
                lon = xaxis[j]
                lat = yaxis[i]
                value = datamat[i, j]
                if not np.isnan(value):
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()

        return

    def write_UV_to_vec(self, Umat, Vmat, Imat, xaxis, yaxis, vec_name, thres_I, len_scaling, f2i=None):

        # Degree.
        angle = np.rad2deg(np.arctan2(Vmat,Umat))
        # Length.
        length = np.sqrt(Umat**2 + Vmat**2) * len_scaling

        # Mask out small values by intensity mat
        #quant = (angle, length)
        angle[Imat<thres_I] = np.nan
        length[Imat<thres_I] = np.nan
       
        f = open(vec_name, 'w')
        xaxis = np.asarray(xaxis) / f2i
        yaxis = np.asarray(yaxis) / f2i

        for i in range(len(yaxis)):
            for j in range(len(xaxis)):
                lon = xaxis[j]
                lat = yaxis[i]

                aa = angle[i,j]
                ll = length[i,j]

                if not np.isnan(ll):
                    f.write(str(lon)+' '+str(lat)+' '+str(aa)+' '+str(ll)+'\n')

        f.close()

    def read_xyz_into_dict(self, file_name, f2i=None):

        if f2i is None:
            f2i = self.float2int

        data_df = pd.read_csv(file_name, delim_whitespace=True, header=None)
        X = data_df.iloc[:,0] * f2i
        Y = data_df.iloc[:,1] * f2i
        values = data_df.iloc[:,2]
        X = np.round(X).astype(np.int)
        Y = np.round(Y).astype(np.int)

        data_dict = {}
        for i in range(len(X)):
            data_dict[(X[i], Y[i])] = values[i]

        return data_dict

    def read_xyz_into_datamat(self, file_name, f2i=None, add_mean_phase=True):

        if f2i is None:
            f2i = self.float2int

        # Read the data
        xyz_df = pd.read_csv(self.file_name, delim_whitespace=True, header=None)

        X = xyz_df.iloc[:,0] * f2i
        Y = xyz_df.iloc[:,1] * f2i
        data_values = xyz_df.iloc[:,2]

        # Convert to integer
        X = np.round(X).astype(np.int)
        Y = np.round(Y).astype(np.int)

        # Determine delta and range of X and Y
        sort_X = np.sort(np.unique(X))
        delta_X = min(sort_X[1:]-sort_X[:-1])
        min_X = min(sort_X)
        max_X = max(sort_X)

        sort_Y = np.sort(np.unique(Y))
        delta_Y = min(sort_Y[1:]-sort_Y[:-1])
        min_Y = min(sort_Y)
        max_Y = max(sort_Y)

        # Get X and Y axis
        X_axis = np.arange(min_X, max_X+1, delta_X)
        Y_axis = np.arange(min_Y, max_Y+1, delta_Y)

        NX = len(X_axis)
        NY = len(Y_axis)

        # Form the data matrix
        data_mat = np.zeros(shape=(NY, NX)) + np.nan

        # Put values into data_mat
        X_index = (X - min_X) // delta_X
        Y_index = (Y - min_Y) // delta_Y

        for i in range(len(X_index)):
            data_mat[Y_index[i], X_index[i]] = data_values[i]

        # Add mean phase
        if add_mean_phase:

            mean_phase = self.read_mean_phase(file_name)

            if mean_phase is not None:

                data_mat = data_mat + mean_phase

                data_values = data_values + mean_phase

        return (X, Y, data_values, X_axis, Y_axis, data_mat)

    def read_point_data_from_xyz(self, point, file_name, project, winsize=(1,1), add_mean_phase = True):

        local_f2i = 10**4

        data_df = pd.read_csv(file_name, delim_whitespace=True, header=None)
        X = data_df.iloc[:,0] * local_f2i
        Y = data_df.iloc[:,1] * local_f2i
        values = data_df.iloc[:,2]
        X = np.round(X).astype(np.int)
        Y = np.round(Y).astype(np.int)

        # Determine delta and range of X and Y
        sort_X = np.sort(np.unique(X))
        delta_X = min(sort_X[1:]-sort_X[:-1])

        sort_Y = np.sort(np.unique(Y))
        delta_Y = min(sort_Y[1:]-sort_Y[:-1])

        ## Set the window for average
        #if project == 'Rutford':
        #    x_delta = int(round(0.025 * local_f2i))
        #    y_delta = int(round(0.005 * local_f2i))
        #elif project == 'Evans':
        #    x_delta = int(round(0.04 * local_f2i))
        #    y_delta = int(round(0.01 * local_f2i))
        #else:
        #    raise Exception()

        # The point to read
        px, py = point

        px = int(round(px * local_f2i))
        py = int(round(py * local_f2i))

        valid_x = np.absolute(X - px) <= winsize[0]*delta_X
        valid_y = np.absolute(Y - py) <= winsize[1]*delta_Y

        valid_xy = np.logical_and(valid_x, valid_y)
        #print(np.sum(valid_x), np.sum(valid_y), np.sum(valid_xy))

        data_value = np.mean(values[valid_xy])

        if add_mean_phase:
 
            mean_phase = self.read_mean_phase(file_name)

            if mean_phase is not None:

                data_value = data_value + mean_phase

        return data_value

    def read_mean_phase(self, file_name):

        # Look for the mean phase file
        folder = file_name.rsplit('/', maxsplit=1)[0]
        basename = file_name.rsplit('/', maxsplit=1)[1].split('.')[0]

        quant_name = '_'.join(basename.split('_')[1:])

        if quant_name.split('_')[0] in ['true','est'] and quant_name.split('_')[-1] == 'phase':
            mean_phase_file_name = os.path.join(folder, 'mean_phase.txt')

            f = open(mean_phase_file_name)

            lines = f.readlines()

            for line in lines:
                line_quant_name, line_value = line.split()
                if line_quant_name == quant_name:
                    if line_value.lower == 'nan':
                        mean_phase = None
                    else:
                        mean_phase = float(line_value)
            f.close()

        # mean phase is not available
        else:
            mean_phase = None

        return mean_phase 

#####################3
run_basics = 0

if run_basics:
    bs = basics()
    params_file = "/net/kamb/ssd-tmp1/mzzhong/tides_model/TMD_Matlab_Toolbox_v2.5/TMD/results/EIS_tides_params.txt" 
    tide_names, tides_params = bs.read_tides_params(params_file=params_file)
    print(tides_params)
    for tide_name in tide_names:
        tide_name1 = tide_name.lower()
        print(tide_name, round(tides_params[(tide_name1, "tide_amp")],5), round(tides_params[(tide_name1, "tide_phase_deg")],3))
