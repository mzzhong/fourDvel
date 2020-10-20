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
import seaborn as sns

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

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

class driver_fourdvel():

    def __init__(self, inps):

        self.param_file = inps.param_file
        self.task_name = inps.task_name

        self.estimate_tasks = ["do_nothing", "tides_1", "tides_2", "tides_3", "tides_4", "tides_5"]

        # Explantion:

        # tides_1: conventional linear (BM 2017)

        # tides_2: single point MCMC to detect grounding, can take external vertical model to constrain the vertical displacement
        # This is non-linear, as we want to simutaneously infer grounding and tidal params
        # Seems to work on detecting grounding

        # tides_3: Try to linearize the problem in tides_2
        # Add vertical data into linear model
        # Estimate the vertical scaling only
        # Enumerate the grounding level
        # Check the residual to seclect the best grounding level

        # TODO
        # tides_4: Find the optimal grounding from the enumeration results using tides_3
        # Need to load the results
        # Only grid_set_others needs to be updated

        # tides_5: Using the optimal grounding from tides_4 to get the new results
        # Need to load the results
        # All grid_set except than grid_set needs to be updated

        self.analysis_tasks = ["prediction_evaluation"]

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

    def check_point_set_with_bbox(self, point_set):

        lons = []
        lats = []
        for point in point_set:
            lons.append(point[0])
            lats.append(point[1])

        lons = np.asarray(lons)
        lats = np.asarray(lats)

        bbox_s, bbox_n, bbox_e, bbox_w = self.tasks.bbox

        if bbox_s is not None:
            if np.nanmax(lats)<bbox_s:
                return False
            
        if bbox_n is not None:
            if np.nanmin(lats)>bbox_n:
                return False
 
        if bbox_e is not None:
            if np.nanmin(lons)>bbox_e:
                return False

        if bbox_w is not None:
            if np.nanmax(lons)<bbox_w:
                return False
 
        return True

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

            ############################################################################
            # (1) Run all in serial. # (2) Only run the test point tile

            if (count_tile >= start_tile) and (count_tile < stop_tile):

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
                if not self.check_point_set_with_bbox(point_set):
                    print("This point set is not in bbox: ", lon, lat)
                    skip_this_tile = True

                ## Find tracks_set from point_set
                tracks_set = {}
                for point in point_set:
                    tracks_set[point] = grid_set[point]

                # Run it
                # Default is False for recording
                simple_count = True
                if simple_count == True and skip_this_tile == False:

                    print("Running tile: ", tile)

                    ## Tides ###
                    if task_name in self.estimate_tasks:

                        recorded = False

                        all_sets = tasks.estimate(point_set = point_set, tracks_set = tracks_set)

                        # Save the results
                        # Update (only for parallel call)
                        if use_threading and tasks.inversion_method == 'Bayesian_Linear':
        
                            # Save the results to disk
                            point_result_folder = self.estimation_dir + '/point_result'
        
                            point_name = str(lon) + '_' + str(lat)
        
                            with open(point_result_folder + "/" + point_name + ".pkl","wb") as f:
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
    
                        if recorded == False:
                            print("Having problem recording this tile: ", tile)
                            raise Exception()

                    ## Prediction evaluiation ###
                    elif task_name in self.analysis_tasks:

                        if task_name == "prediction_evaluation":

                            all_sets = tasks.point_set_prediction_evaluation(point_set = point_set, tracks_set = tracks_set)

                        else:
                            raise Exception()

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
        do_calculation = True

        if do_calculation:

            # Count the total number of tiles
            n_tiles = len(tile_set.keys())
            print('Total number of tiles: ', n_tiles)
    
            # Chop into multiple threads. 
            nthreads = 10
            #nthreads = 1
            self.nthreads = nthreads
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

            # Convert the results to normal dictionary.
            # Save the results to the class.
            print("Saving the results...")
            tasks.grid_set_true_tide_vec = dict(all_grid_sets['grid_set_true_tide_vec'])
            tasks.grid_set_tide_vec = dict(all_grid_sets['grid_set_tide_vec'])
            tasks.grid_set_tide_vec_uq = dict(all_grid_sets['grid_set_tide_vec_uq'])
    
            tasks.grid_set_resid_of_secular = dict(all_grid_sets['grid_set_resid_of_secular'])
            tasks.grid_set_resid_of_tides = dict(all_grid_sets['grid_set_resid_of_tides'])
            tasks.grid_set_others = dict(all_grid_sets['grid_set_others'])

            # task_name = prediction_evaluation
            tasks.grid_set_analysis = dict(all_grid_sets['grid_set_analysis'])

            ## end of dict.

        else:
            # Load the results from point_result
            print("Loading the results...")
            point_results = os.listdir(self.estimation_dir + '/point_result')

            for ip, point_pkl in enumerate(point_results):
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
                
                #tasks.grid_set_analysis.update(all_sets['analysis_set'])

        ## Save the final results in dictionary manager
        forceSaveTides = False
        forceSaveAnalysis = False

        if (task_name in self.estimate_tasks and tasks.single_point_mode == False) or (task_name in self.estimate_tasks and forceSaveTides == True):

            print("Write results to disk")
            print("The task is estimate")
            print("Single_point_mode is off or Force Save is on")
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

            print("Done")

        elif (task_name in self.analysis_tasks and tasks.single_point_mode == False) or (task_name in self.analysis_tasks and forceSaveAnalysis == True):

            print("Write results to disk")
            print("The task is prediction evaluation")

            pkl_name = '_'.join((str(test_id), 'grid_set_analysis', tasks.analysis_name))  + '.pkl'

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
