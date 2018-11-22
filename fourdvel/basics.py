#!/usr/bin/env python3

import numpy as np
import datetime

class basics():

    def __init__(self):

        # Time origin
        self.t_origin = datetime.datetime(2018,1,1,0,0,0,0)

        # Tides
        self.tides = ['K2','S2','M2','K1','P1','O1','Mf','Msf','Mm','Ssa','Sa']

        self.tide_periods = {}
        tide_periods = self.tide_periods
        tide_periods['K2'] = 0.498
        tide_periods['S2'] = 0.5
        tide_periods['M2'] = 0.52
        tide_periods['K1'] = 1
        tide_periods['P1'] = 1.003
        tide_periods['O1'] = 1.08
        tide_periods['Mf'] = 13.66
        tide_periods['Msf'] = 14.77
        tide_periods['Mm'] = 27.55
        tide_periods['Ssa'] = 182.62
        tide_periods['Sa'] = 365.27

        self.tide_omegas = {}
        for tide_name in self.tides:
            self.tide_omegas[tide_name] = 2*np.pi/tide_periods[tide_name]

        # Pixel size
        self.lon_re = 50
        self.lat_re = 200

        self.lon_step = 1/self.lon_re
        self.lat_step = 1/self.lat_re

    def wrapped(self,phase):
        
        return (phase + np.pi) % (2*np.pi) - np.pi

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

    def cm2m(self,value):
        return value / 100

    def float_rounding(self,value,precision):
        return np.round(value * precision)/precision


    def velo_to_amp(self,velo,tide_name):
        return velo * self.tide_periods[tide_name]/2

    def unit_vec(self, v1, v2=None):
        if v2:
            return np.asarray([v1,v2,np.sqrt(1-v1**2-v2**2)])
        else:
            return np.asarray([v1,np.sqrt(1-v1**2),0])
