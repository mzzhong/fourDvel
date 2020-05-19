#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in July, 2019

from basics import basics
import numpy as np

class forward(basics):
    def design_mat_set(self, timings, modeling_tides):

        mat_set = {}

        n_modeling_tides = len(modeling_tides)
    
        t_origin = self.t_origin.date()
        for timing in timings:
            t = (timing[0] - t_origin).days + timing[1]
            
            # Note that secular displacement is excluded
            design_mat = np.zeros(shape = (3, n_modeling_tides * 6))

            comps = ['e','n','u']
            for i, comp in enumerate(comps):
                for j, tide_name in enumerate(modeling_tides):
                    omega = 2*np.pi / self.tide_periods[tide_name]

                    design_mat[i,3 * (2*j) + i ] = np.cos(omega * t)
                    design_mat[i,3* (2*j+1) + i] = np.sin(omega * t)

            mat_set[timing] = design_mat

        return mat_set
