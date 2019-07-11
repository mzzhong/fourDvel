#!/usr/bin/env python3

import numpy as np
import datetime
import math

class basics():

    def __init__(self):

        self.pickle_dir = '/net/kamb/ssd-tmp1/mzzhong/insarRoutines/pickles'

        # Time origin
        self.t_origin = datetime.datetime(2018,1,1,0,0,0,0)

        # Tides
        self.tides = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa', 'M4', 'S4', 'MS4']

        self.tide_periods = {}
        tide_periods = self.tide_periods
        tide_periods['K2'] = 0.49863484
        tide_periods['S2'] = 0.50000000
        tide_periods['M2'] = 0.51752505
        tide_periods['K1'] = 0.99726967
        tide_periods['P1'] = 1.00274532
        tide_periods['O1'] = 1.07580578
        tide_periods['Mf'] = 13.66083078
        tide_periods['Msf'] = 14.76529444
        tide_periods['Mm'] = 27.55463190
        tide_periods['Ssa'] = 182.62818021
        tide_periods['Sa'] = 365.25636042

        tide_periods['M4'] = tide_periods['M2']/2 
        tide_periods['S4'] = tide_periods['S2']/2
        tide_periods['MS4'] = 1/(1/tide_periods['M2'] + 1/tide_periods['S2'])

        self.tide_omegas = {}
        for tide_name in self.tides:
            self.tide_omegas[tide_name] = 2*np.pi/tide_periods[tide_name]

        # Pixel size
        self.lon_re = 50
        self.lat_re = 200

        self.lon_step = 1/self.lon_re
        self.lat_step = 1/self.lat_re

        # Load the satellite constants
        self.satellite_constants()

    def latlon_distance(self, lon1,lat1,lon2,lat2):
        
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


    def cm2m(self,value):
        return value / 100

    def float_rounding(self,value,precision):
        return np.round(value * precision)/precision

    def velo_amp_to_dis_amp(self,velo,tide_name):
        return velo / self.tide_omegas[tide_name]

    def dis_amp_to_velo_amp(self, dis_amp, tide_name):
        return dis_amp * self.tide_omegas[tide_name]

    def velo_phase_to_dis_phase(self, phase, deg = False):
        if deg:
            return phase - 90
        else:
            return phase - np.pi/2

    def dis_phase_to_velo_phase(self, phase, deg = False):
        if deg:
            return phase + 90
        else:
            return phase + np.pi/2

    def unit_vec(self, v1, v2=None):
        if v2:
            return np.asarray([v1,v2,np.sqrt(1-v1**2-v2**2)])
        else:
            return np.asarray([v1,np.sqrt(1-v1**2),0])

    def satellite_constants(self):

        self.track_timefraction = {}
        track_timefraction = self.track_timefraction

        # CSK.
        fid = open('/net/kamb/ssd-tmp1/mzzhong/insarRoutines/csk_times.txt')
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


        # new tracks
        t50 = datetime.time(3,53,40)
        track_timefraction[('s1',50)] = (t7.hour * 3600 + t7.minute*60 + t7.second)/(24*3600)

        t64 = datetime.time(2,57,0)
        track_timefraction[('s1',64)] = (t7.hour * 3600 + t7.minute*60 + t7.second)/(24*3600)

        #t49 = datetime.time(2,16,0)
        #track_timefraction[('s1',49)] = (t7.hour * 3600 + t7.minute*60 + t7.second)/(24*3600)

        #print(track_timefraction)

        return 0


