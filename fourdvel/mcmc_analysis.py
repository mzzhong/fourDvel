#!/usr/bin/env python3
# Author: Minyan Zhong
# Development starts in July, 2019

import os
import sys

import pickle
import numpy as np

import multiprocessing

from fourdvel import fourdvel
from display import display

import matplotlib.pyplot as plt
import collections

import pymc3 as pm

from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev

import sys

class mcmc_analysis(fourdvel):

    def __init__(self):

        if len(sys.argv)==1:
            super(mcmc_analysis,self).__init__()
        else:
            print("param file to read: ", sys.argv[1])
            super(mcmc_analysis,self).__init__(sys.argv[1])

        self.get_grid_set_velo()
        test_id = self.test_id

        result_folder = self.estimations_dir
        self.this_result_folder = os.path.join(result_folder,str(test_id))

        self.display = display(sys.argv[1]) 

        #with open(self.this_result_folder + '/' 
        #            + str(test_id) + '_' + 'grid_set_tide_vec.pkl','rb') as f:
        #    self.grid_set_tide_vec = pickle.load(f)

    def run_map_estimate(self):

        point = self.test_point
        map_estimate_pkl = "_".join(["./estimations/map_estimate_BMC",str(point[0]),str(point[1]),str(self.test_mode)])
        print(map_estimate_pkl)

        with open(map_estimate_pkl + '.pkl', "rb") as f:

            map_estimate = pickle.load(f)

        secular = map_estimate['secular'][0]
        tidal = map_estimate['tidal'][:,0]
        grounding = map_estimate['grounding'][0]

        amp = 0
        for i in range((len(tidal)-4)//2):
            amp += (tidal[2*i]**2 + tidal[2*i+1]**2)**(1/2)
            print(amp)
        
        print(secular, tidal)
        print(amp, grounding)

    def axis_config_secular(self, ax):
        pass

    def axis_config_tidal(self, ax):
        pass

    def run_samples(self):

        if len(sys.argv)>=3:
            true_exist=int(sys.argv[2])
        else:
            true_exist=1

        point = self.test_point
        trace_pkl = "_".join([self.estimation_dir+"/samples_BMC",str(point[0]),str(point[1]),str(self.test_mode)])
        print(trace_pkl)

        with open(trace_pkl + '.pkl', "rb") as f:
            trace = pickle.load(f)

        print(trace)
        print(trace.stat_names)

        if true_exist:  
            true_model_vec_secular = trace.true_model_vec[:3]
            true_model_vec_tidal = trace.true_model_vec[3:]
        
        plt.figure(11, figsize=(10,10))
        plt.plot(trace['step_size_bar'])
        plt.savefig("11.png")

        secular = trace.get_values("secular")
        tidal = trace.get_values('tidal')
        grounding = trace.get_values('grounding')

        print(secular.shape)
        # Display secular parameters
        fig = plt.figure(1, figsize=(15,10))
        #fig, axs = plt.subplots(1,3, sharex=False, sharey=True, figsize=(15,10))

        for i in range(3):
            values = secular[:,0,i]
            ax = fig.add_subplot(1,3, i+1)
            #ax = axs[i]
            
            n, bin_borders, patches = ax.hist(values, bins=40, density=True, fc=(0, 0, 1, 1))

            bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
            bspl = splrep(bin_centers, n, s=1)
            n_smooth = splev(bin_centers, bspl)
            #ax.plot(bin_centers,n_smooth,linewidth=3,color="black")

            if true_exist: 
                ax.plot( [true_model_vec_secular[i],true_model_vec_secular[i]], [0,np.max(n)],linewidth=3, color="red" )

            ax.set_xlabel("m/d",fontsize=15)
            ax.tick_params(labelsize=15)
            ax.get_yaxis().set_ticks([])
            ax.set_ylim(bottom=0)

            # Set ticks
            n_ticks=0
            ratio = 50
            # become more and more accurate
            while n_ticks<4:
                xticks = np.round(np.asarray(bin_centers)*ratio)/ratio
                n_ticks = len(np.unique(xticks))
                ratio+=50
            if i==0:
                #ax.set_xticks([0.66,0.665,0.67,0.675,0.68])
                ax.set_xticks(xticks)
            else:
                ax.set_xticks(xticks)
            
            if i==0:
                ax.set_ylabel("Probability Density", fontsize=15)

            # title
            if i==0:
                ax.set_title("Secular East Velocity",fontsize=15)
            elif i==1:
                ax.set_title("Secular North Velocity",fontsize=15)
            elif i==2:
                ax.set_title("Secular Up Velocity",fontsize=15)


        fig.savefig(self.estimation_dir+"/secular.png", transparent=False, bbox_inches='tight')

        # Display the tidal parameters
        fig = plt.figure(2,figsize=(20,10))
        N, P,_ = tidal.shape
        print(tidal.shape)

        tidal_names = ['M_2', 'S_2', 'K_2', 'O_1', 'K_1', 'P_1']
        comps = ['cos','sin']

        msf_comps = ['cos east','cos north','sin east','sin north']

        for i in range(P):
            values = tidal[:,i,0]
            ax = fig.add_subplot(2,P//2, i+1)
            n, bin_borders, patches = ax.hist(values, bins=40, density=True, fc=(0,1,0,1))

            bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
            bspl = splrep(bin_centers, n, s=1)
            n_smooth = splev(bin_centers, bspl)
            #ax.plot(bin_centers,n_smooth,linewidth=3,color="black")

            if true_exist:
                ax.plot( [true_model_vec_tidal[i],true_model_vec_tidal[i]], [0,np.max(n)],linewidth=3, color="red" )

            if i>=P//2:
                ax.set_xlabel("m",fontsize=15)

            ax.tick_params(labelsize=10)
            ax.get_yaxis().set_ticks([])
            ax.set_ylim(bottom=0)

            if i==0 or i==P//2:
                ax.set_ylabel("Probability Density", fontsize=15)

            # title
            if i<P-4:
                ax.set_title('$'+tidal_names[i//2]+'$' + ' ' + comps[i%2] + ' ' + 'up' ,fontsize=15)
            else:
                ax.set_title('$M_{sf}$' + ' ' + msf_comps[i-(P-4)], fontsize=15)

#            # Set ticks
#            n_ticks=0
#            ratio = 10
#            # become more and more accurate
#            while n_ticks<3:
#                xticks = np.round(np.asarray(bin_centers)*ratio)/ratio
#                n_ticks = len(np.unique(xticks))
#                ratio+=10
#            ax.set_xticks(xticks)
 
            #break
        fig.savefig(self.estimation_dir+"/tidal.png", bbox_inches='tight')


        # Display the grounding parameters
        fig = plt.figure(3,figsize=(10,10))
        print(grounding.shape)
        true_grounding_level = -1.5
        for i in range(1):
            values = grounding[:,0,0]
            ax = fig.add_subplot(111)
            n, bin_borders, patches = ax.hist(values, bins=80, density=True, fc=(0,1,1,1))

            bin_centers = bin_borders[:-1] + np.diff(bin_borders)/2
            bspl = splrep(bin_centers, n, s=1)
            n_smooth = splev(bin_centers, bspl)
            #ax.plot(bin_centers,n_smooth,linewidth=3,color="black")

            if true_exist:
                ax.plot([true_grounding_level,true_grounding_level], [0,np.max(n)],linewidth=3, color="red")

            ax.set_xlim([-2.5,-0.5])
            ax.set_xlabel("m",fontsize=15)
            ax.tick_params(labelsize=15)
            ax.get_yaxis().set_ticks([])
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Probability Density", fontsize=15)

            # title
            ax.set_title('Grounding Level', fontsize=15)

        fig.savefig(self.estimation_dir+"/grounding.png",bbox_inches='tight')

def main():

    mcmc = mcmc_analysis()
    #mcmc.run_map_estimate()
    mcmc.run_samples()

if __name__=="__main__":
    
    main()
