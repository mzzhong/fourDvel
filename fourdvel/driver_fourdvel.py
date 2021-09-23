#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in Dec, 2019

import os
import sys
import pickle
import pathlib
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import datetime
from datetime import date

import multiprocessing
from multiprocessing import Value

import time

# Estimation
from estimate import estimate

# Analysis
from analysis import analysis


def createParser():

    parser = argparse.ArgumentParser( description='driver of fourdvel')
    
    parser.add_argument('-p','--param_file', dest='param_file',type=str,help='parameter file',required=True)

    parser.add_argument('-t','--task_name',dest='task_name',type=str, help='task name', required=True)
    
    parser.add_argument('-n','--nthreads',dest='nthreads',type=int, help='number of threads', required=False, default=None)
    
    parser.add_argument('-m','--mode',dest='mode',type=str, help='mode (calc or load)', required=False, default='calc')
    
    parser.add_argument('-f','--tile_fraction',dest='tile_fraction',type=str, help='the fraction of tile to calculate(e.g., 3,1), default: 1,0', required=False, default='1,0')
    
    parser.add_argument('--no_update',dest='update', help='update (overriding) the existing result (point result), default: True', required=False, action='store_false')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

class driver_fourdvel():
    def __init__(self, inps):

        self.param_file = inps.param_file
        self.task_name = inps.task_name
        self.nthreads = inps.nthreads
        self.mode = inps.mode
        self.tile_fraction = [int(i) for i in inps.tile_fraction.split(',')]

        self.update = inps.update
        print('Update results: ', self.update)
        #print(stop)

        self.estimate_tasks = ["do_nothing", "tides_1", "tides_2", "tides_3", "tides_4", "tides_5"]

        # Explantion:
        # tides_1: conventional linear model (BM 2017). Summation of a family of sinusoidal functions.
        # Introduce modification: introduce grounding/clipping to the time series

        # Inversion method: 
        # (1) Bayesian_Linear (without grounding) 
        # (2) Bayesian_Linear_MCMC (without grounding, same result as (1), but sampling-based) (turned off)
        # (3) Bayesian_MCMC (with grounding introduced, which is difficult to work) (turned off)
        # (4) Nonlinaer optimization (with grounding, which is difficult to workd) (turned off)

        # tides_2: single point MCMC to detect grounding, can take external vertical model to constrain the vertical displacement
        # This is non-linear, as we want to simutaneously infer grounding and tidal params
        # Seems to work on detecting grounding
        # Inversion method: 
        # (1) Bayesian_MCMC method 
        # (2) Nonlinear optimization method

        # tides_3: Try to linearize the problem in tides_2
        # Add vertical data into linear model
        # Estimate the vertical scaling only
        # Enumerate the grounding level
        # Check the residual to select the best grounding level
        # Inversion method: 
        # (1) Bayesian_Linear

        # TODO
        # tides_x: Find the optimal grounding from the enumeration results using tides_3
        # Need to load the results
        # Only grid_set_others needs to be updated

        # tides_y: Using the optimal grounding from tides_x to get the new results
        # Need to load the results
        # All grid_set except than grid_set needs to be updated

        self.analysis_tasks = ["residual_vs_tide_height","residual_analysis"]

        # Set the task
        if self.task_name in self.estimate_tasks:

            # Will change the class tasks to estimate 
            self.tasks = estimate(self.param_file)

            # Set the task name
            self.tasks.set_task_name(task_name = self.task_name)
        
        elif self.task_name in self.analysis_tasks:
            
            self.tasks = analysis(self.param_file)

            # Set the task name
            self.tasks.set_task_name(task_name = self.task_name)

        else:
            print("Undefined task_name")
            raise Exception()

        # Set the basics
        self.tasks.param_file = self.param_file
        
        # Get the basics
        self.estimation_dir = self.tasks.estimation_dir

    def driver_serial_tile(self, start_tile=None, stop_tile=None, use_threading = False, all_grid_sets=None, threadId=None):

        task_name = self.task_name
        tasks = self.tasks
                
        # Input information.
        tile_set = tasks.tile_set
        grid_set = tasks.grid_set

        #print("start tile: ", start_tile, "stop tile: ",stop_tile)
        count_tile = 0
        count_run = 0

        #print("Number of tiles: ",len(tile_set))

        # Find the test point
        test_point = tasks.test_point

        if test_point is None:
            no_test_point = True
        else:
            no_test_point = False

        if start_tile is None or stop_tile is None:
            start_tile = 0
            stop_tile = 10**5

        for count_tile, tile in enumerate(tile_set.keys()):
            
            # Work on a particular tile.
            lon, lat = tile


            # Set the name of the pklfile to save to the disk
            point_result_folder = self.estimation_dir + '/point_result'
        
            point_name = str(lon) + '_' + str(lat)

            point_result_pklname = point_result_folder + "/" + point_name + ".pkl"

            # If this one is calculated then skip it
            if os.path.exists(point_result_pklname) and self.update == False:
                print(point_result_pklname, "is already calculated and update mode is turend off. Skip")
                continue
            else:
                print(point_result_pklname, "is waiting for calculation")
                #continue

            #continue
 
            ############################################################################
            # (1) Run all in serial. # (2) Only run the test point tile

            if (count_tile >= start_tile) and (count_tile < stop_tile):



            # Deprecated
            #if ((count_tile >= start_tile) and (count_tile < stop_tile) and (test_point is None)):
        
            #if (count_tile >= start_tile and count_tile < stop_tile and count_tile % 2 == 1):

            # Debug this tile for Rutford
            #if count_tile >= start_tile and count_tile < stop_tile and tile == self.float_lonlat_to_int5d((-83.0, -78.6)):

            ################################################################################
                #print("Find the tile", tile)
                #print('***  Start a new tile ***')
                #self.print_int5d([lon, lat])

                point_set = tile_set[tile] # A list of tuples

                skip_this_tile = False

                # Set test point 
                if no_test_point:
                    print("No test point is provided")
                    print("Set test point to be the first point in point_set")
                    tasks.test_point = point_set[0]
                    test_point = tasks.test_point

                # Set the tile
                tasks.tile = tile

                # Reset the test point if the test_point is not in the current point set
                if test_point in point_set:
                    print("Test point is in this tile")
                    print("It is either provided in param file or not provided but reset")
                else:
                    print("test point: ",test_point)
                    print("point_set[0]: ",point_set[0])
                    print("Test point is provided but doesn't match this tile")
                    skip_this_tile = True
                    
                if skip_this_tile == False:
                    print("The tile to work on: ", tile)
                    print("start tile and stop tile: ", start_tile, stop_tile)

                # Output the location and size of tile. 
                #print('tile coordinates: ', tile)
                #print('Number of points in this tile: ', len(point_set))

                # Check if the point_set is in the bbox
                if not self.tasks.check_point_set_with_requirements(point_set, kind='bbox', bbox=self.tasks.bbox):
                    print("This point set is not in bbox: ", lon, lat)
                    skip_this_tile = True


                # Check if this tile satisfies the portion requirement
                # default n%1 != 0 is always False, then do not skip
                # e.g., if n%3 != 1 skip. Then only for n=1,4,7,10: n%3 != 1 case is False, and skip is not turned on and they are calculated
                if count_tile % self.tile_fraction[0] != self.tile_fraction[1]:
                    print("This point set is not in the requested fraction: ", lon, lat)
                    skip_this_tile = True

                ## Find tracks_set from point_set
                tracks_set = {}
                for point in point_set:
                    tracks_set[point] = grid_set[point]

                # Run the task
                # Recording is False by default
                simple_count = True
                if simple_count == True and skip_this_tile == False:

                    print("Running tile: ", tile)
                    print("Number of points in this tile: ", len(point_set))

                    ## Estimate tasks ###
                    if task_name in self.estimate_tasks:

                        # do the estimation
                        all_sets = tasks.estimate(point_set = point_set, tracks_set = tracks_set)

                        # save the results
                        recorded = False
                        
                        # update (only for parallel call)
                        if (use_threading and task_name in ["tides_1", "tides_3"] and tasks.inversion_method == 'Bayesian_Linear') or \
                            (use_threading and task_name in ["tides_2"] and tasks.inversion_method == "Nonlinear_Optimization"):
        
       
                            with open(point_result_pklname, "wb") as f:
                                pickle.dump(all_sets, f)
    
                            # Say that this tile is record
                            recorded = True
                            
                            # Save the results through updating dictionary manager
                            if all_sets['true_tide_vec_set'] is not None:
                                all_grid_sets['grid_set_true_tide_vec'].update(all_sets['true_tide_vec_set'])
                            
                            all_grid_sets['grid_set_tide_vec'].update(all_sets['tide_vec_set'])
        
                            all_grid_sets['grid_set_tide_vec_uq'].update(all_sets['tide_vec_uq_set'])
                       
                            all_grid_sets['grid_set_resid_of_secular'].update(all_sets['resid_of_secular_set'])
                            all_grid_sets['grid_set_resid_of_tides'].update(all_sets['resid_of_tides_set'])
                            all_grid_sets['grid_set_others'].update(all_sets['others_set'])

                            all_grid_sets['grid_set_residual_analysis'].update(all_sets['residual_analysis_set'])
    
                        if recorded == False:
                            print("Having problem recording this tile: ", tile)
                            raise Exception()

                    ## Analysis tasks ###
                    elif task_name in self.analysis_tasks:

                        all_sets = tasks.point_set_analysis(point_set = point_set, tracks_set = tracks_set, task_name = task_name)

                        all_grid_sets['grid_set_analysis'].update(all_sets['analysis_set'])

                    elif task_name == "do_nothing":

                        pass

                    else:
                        raise Exception("Undefined task_name", task_name)

                # Count the run tiles
                count_run = count_run + 1

        print("count run: " + str(count_run))
        print("count tile: " + str(count_tile))

        return 0

    def driver_parallel_tile(self):

        task_name = self.task_name
        tasks = self.tasks

        # Initialization
        test_id = tasks.test_id

        estimation_dir = self.estimation_dir

        # Set the place of result for this run
        pathlib.Path(estimation_dir + '/point_result').mkdir(exist_ok=True)

        # Using multi-threads to get map view estimation.
        # make driver serial tile parallel.
        tile_set = tasks.tile_set

        # Count the number of tiles
        if self.mode == 'calc':
            do_calculation = True
        elif self.mode == 'load':
            do_calculation = False
        else:
            raise ValueError()

        if do_calculation:

            # Count the total number of tiles
            n_tiles = len(tile_set.keys())
            print('Total number of tiles: ', n_tiles)
    
            # Chop into multiple threads.
            if self.nthreads is None:
                #nthreads = 10
                #nthreads = 8
                #nthreads = 6
                nthreads = 5
                #nthreads = 4
                #nthreads = 1

            else:
                nthreads = self.nthreads

 
            total_number = n_tiles
    
            # Only calculate the first half
            # Ad hoc control
            # 2020.01.13
            half_number = total_number//2    
            divide = tasks.chop_into_threads(half_number, nthreads)
            # First half
            # pass
            # Second half
            divide = divide + half_number
            print("divide: ", divide)

            # Full divide
            divide = tasks.chop_into_threads(total_number, nthreads)
            print('total number: ', total_number)
            print('nthreads: ', nthreads)
            print("full divide: ", divide)
    
            if nthreads > 1:

                # Multithreading starts here.
                # The function to run every chunk.
                func = self.driver_serial_tile
        
                # Setup the array.
                manager = multiprocessing.Manager()
                
                all_grid_sets = {}
                ## estimate_tasks
                all_grid_sets['grid_set_true_tide_vec'] = manager.dict()
                all_grid_sets['grid_set_tide_vec'] = manager.dict()
                all_grid_sets['grid_set_tide_vec_uq'] = manager.dict()
                all_grid_sets['grid_set_resid_of_secular'] = manager.dict()
                all_grid_sets['grid_set_resid_of_tides'] = manager.dict()
                all_grid_sets['grid_set_others'] = manager.dict()
                all_grid_sets['grid_set_residual_analysis'] = manager.dict()
    
                ## analysis_tasks
                all_grid_sets['grid_set_analysis'] = manager.dict()
    
                jobs=[]
                for ip in range(nthreads):
                    start_tile = divide[ip]
                    stop_tile = divide[ip+1]
        
                    # Use contiguous chunks
                    p=multiprocessing.Process(target=func, args=(start_tile, stop_tile, True,
                                                            all_grid_sets, ip))
        
                    # Based on modulus
                    #p=multiprocessing.Process(target=func, args=(0, n_tiles, True,
                    #                                        all_grid_sets, ip))
        
                    jobs.append(p)
                    p.start()
        
                for ip in range(nthreads):
                    jobs[ip].join()

            else:
                all_grid_sets = {}
                ## estimate_tasks
                all_grid_sets['grid_set_true_tide_vec'] = {}
                all_grid_sets['grid_set_tide_vec'] = {}
                all_grid_sets['grid_set_tide_vec_uq'] = {}
                all_grid_sets['grid_set_resid_of_secular'] = {}
                all_grid_sets['grid_set_resid_of_tides'] = {}
                all_grid_sets['grid_set_others'] = {}
                all_grid_sets['grid_set_residual_analysis'] = {}
    
                ## analysis_tasks
                all_grid_sets['grid_set_analysis'] = {}

                ip = 0
                start_tile = divide[ip]
                stop_tile = divide[ip+1]
 
                self.driver_serial_tile(start_tile, stop_tile, True, all_grid_sets, ip)

            # Convert the results to normal dictionary.
            # Save the results to the class.
            print("Saving the results...")
            tasks.grid_set_true_tide_vec = dict(all_grid_sets['grid_set_true_tide_vec'])
            tasks.grid_set_tide_vec = dict(all_grid_sets['grid_set_tide_vec'])
            tasks.grid_set_tide_vec_uq = dict(all_grid_sets['grid_set_tide_vec_uq'])
    
            tasks.grid_set_resid_of_secular = dict(all_grid_sets['grid_set_resid_of_secular'])
            tasks.grid_set_resid_of_tides = dict(all_grid_sets['grid_set_resid_of_tides'])
            tasks.grid_set_others = dict(all_grid_sets['grid_set_others'])
            tasks.grid_set_residual_analysis = dict(all_grid_sets['grid_set_residual_analysis'])

            # task_name = prediction_evaluation
            tasks.grid_set_analysis = dict(all_grid_sets['grid_set_analysis'])

            ## end of dict.

        else:
            # Load the results from point_result
            print("Loading the results...")
            point_results = os.listdir(self.estimation_dir + '/point_result')

            # Intialization
            tasks.grid_set_true_tide_vec = {}
            tasks.grid_set_tide_vec = {}
            tasks.grid_set_tide_vec_uq = {}
            tasks.grid_set_resid_of_secular = {}
            tasks.grid_set_resid_of_tides = {}
            tasks.grid_set_others = {}
            tasks.grid_set_residual_analysis = {}
 
            # Loop through the results
            for ip, point_pkl in enumerate(point_results):
                if point_pkl.endswith('.pkl'):
                    print(ip)
    
                    pklfile = self.estimation_dir + '/point_result/' + point_pkl
                    
                    with open(pklfile,"rb") as f:
                        all_sets = pickle.load(f)
    
                    if all_sets['true_tide_vec_set'] is not None:
                        tasks.grid_set_true_tide_vec.update(all_sets['true_tide_vec_set'])
                    
                    tasks.grid_set_tide_vec.update(all_sets['tide_vec_set'])
                    tasks.grid_set_tide_vec_uq.update(all_sets['tide_vec_uq_set'])
                    tasks.grid_set_resid_of_secular.update(all_sets['resid_of_secular_set'])
                    tasks.grid_set_resid_of_tides.update(all_sets['resid_of_tides_set'])
                    tasks.grid_set_others.update(all_sets['others_set'])
                    tasks.grid_set_residual_analysis.update(all_sets['residual_analysis_set'])
                    
                    #tasks.grid_set_analysis.update(all_sets['analysis_set'])

        ## Save the final results in dictionary manager
        forceUnSaveTides = False

        forceSaveTides = False

        forceSaveAnalysis = True

        #print(stop)

        # Estimates
        if ((task_name in self.estimate_tasks and tasks.single_point_mode == False) or (task_name in self.estimate_tasks and forceSaveTides == True)) and (forceUnSaveTides==False):

            print("Write results to disk")
            print("The task is estimate")
            print("Single_point_mode is off (So the save is auto on) or forced save is on")
            prefix = str(test_id) + '_'
            with open(self.estimation_dir + '/'+ prefix + 'grid_set_true_tide_vec.pkl', 'wb') as f:
                pickle.dump(tasks.grid_set_true_tide_vec, f)
    
            with open(self.estimation_dir + '/' + prefix + 'grid_set_tide_vec.pkl','wb') as f:
                pickle.dump(tasks.grid_set_tide_vec, f)
    
            with open(self.estimation_dir + '/' + prefix + 'grid_set_tide_vec_uq.pkl','wb') as f:
                pickle.dump(tasks.grid_set_tide_vec_uq, f)
    
            with open(self.estimation_dir + '/' + prefix + 'grid_set_resid_of_secular.pkl','wb') as f:
                pickle.dump(tasks.grid_set_resid_of_secular, f)
    
            with open(self.estimation_dir + '/' + prefix + 'grid_set_resid_of_tides.pkl','wb') as f:
                pickle.dump(tasks.grid_set_resid_of_tides, f)
    
            with open(self.estimation_dir + '/' + prefix+ 'grid_set_others.pkl','wb') as f:
                pickle.dump(tasks.grid_set_others, f)

            with open(self.estimation_dir + '/' + prefix+ 'grid_set_residual_analysis.pkl','wb') as f:
                pickle.dump(tasks.grid_set_residual_analysis, f)

            print("Done")

        # Analysis
        elif (task_name in self.analysis_tasks and tasks.single_point_mode == False) or (task_name in self.analysis_tasks and forceSaveAnalysis == True):

            print("Write results to disk")
            print("The task is prediction evaluation")

            if task_name == 'residual_vs_tide_height':
                pkl_name = '_'.join([str(test_id), 'grid_set_analysis', task_name, tasks.analysis_name])  + '.pkl'

            elif task_name == 'residual_analysis':
                pkl_name = '_'.join((str(test_id), 'grid_set_analysis', task_name))  + '.pkl'
            
            else:
                raise ValueError()

            with open(self.estimation_dir + '/' +  pkl_name ,'wb') as f:
                pickle.dump(tasks.grid_set_analysis, f)

            print("Done")

        else:
            print("@@ Do not save the results to disk @@")
            print("task_name: ", task_name)
            print("single point mode: ", tasks.single_point_mode)

        return 0

def main(iargs=None):
    # Timer starts.
    start_time = time.time()

    # Parse the paramters
    inps = cmdLineParse(iargs)

    # Create a driver
    driver = driver_fourdvel(inps)
    
    # Run with tile set
    #driver.driver_serial_tile()
    driver.driver_parallel_tile()
    print('All finished!')

    # Timer ends
    elapsed_time = time.time() - start_time
    print("Elasped time: ", elapsed_time)

    return 0

if __name__=='__main__':

    main()
