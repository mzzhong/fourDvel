#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in Aug, 2018

import numpy as np
import datetime
import math
import pathlib

class basics():

    def __init__(self):

        # Set NAN value
        self.INT_NAN = -99999

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
