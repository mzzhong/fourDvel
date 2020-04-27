#!/usr/bin/env python3

import os
import sys
import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import datetime
from datetime import date

import multiprocessing
from multiprocessing import Value

import time

from tasks import tasks
from analysis import analysis

class driver_fourdvel():

    def __init__(self, param_file="params.in"):

        if len(sys.argv)==2:
            param_file = sys.argv[1]
        else:
            raise Exception("A parameter file is required")

        # Set the task
        #self.task = "tides"
        self.task = "prediction_evaluation"

        if self.task == "tides":
            
            self.tasks = tasks(param_file)
        
        elif self.task == "prediction_evaluation":
            
            self.tasks = analysis(param_file)
            
            # Preparation
            self.tasks.get_tidal_model()

        # Set the basics
        self.tasks.param_file = param_file
        
        self.result_folder = '/home/mzzhong/insarRoutines/estimations'

    def driver_serial_tile(self, start_tile=None, stop_tile=None, use_threading = False, all_grid_sets=None, threadId=None):
        
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

                ## Find the union of tracks 
                # Only consider track_number and satellite which define the offsetfields.
                tracks_set = {}
                for point in point_set:
                    tracks_set[point] = grid_set[point]

                # Prepare the task
                task = self.task

                # Run it
                # Default is False for recording
                simple_count = True
                if simple_count == True and skip_this_tile == False:

                    ## Tides ###
                    if task == "tides":

                        recorded = False

                        all_sets = tasks.point_set_tides(point_set = point_set, tracks_set = tracks_set,
                                                        inversion_method = tasks.inversion_method)
    
                        # Save the results
                        # Update (only for parallel call)
                        if use_threading and tasks.inversion_method == 'Bayesian_Linear':
        
                            # Save the results to disk
                            point_result_folder = self.this_result_folder + '/point_result'
        
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
                            all_grid_sets['grid_set_analysis'].update(all_sets['analysis_set'])
    
                        if recorded == False:
                            print("Having problem recording this tile: ", tile)
                            raise Exception()

                    ## Prediction evaluiation ###
                    elif task == "prediction_evaluation":

                        all_sets = tasks.point_set_prediction_evaluation(point_set = point_set, tracks_set = tracks_set)

                        all_grid_sets['grid_set_analysis'].update(all_sets['analysis_set'])

                    else:
                        raise Exception("Undefined task")

                # Count the run tiles
                count_run = count_run + 1

        self.f.write("count run: " + str(count_run))
        self.f.write("count tile: " + str(count_tile))
        print("count run: " + str(count_run))
        print("count tile: " + str(count_tile))

        return 0

    def driver_parallel_tile(self):

        tasks = self.tasks

        # Initialization
        test_id = tasks.test_id

        result_folder = self.result_folder

        # Set the place of result for this run
        this_result_folder = os.path.join(result_folder, str(test_id))
        self.this_result_folder = this_result_folder

        if not os.path.exists(this_result_folder + '/point_result'):
            os.mkdir(this_result_folder+'/point_result')

        # Using multi-threads to get map view estimation.
        # make driver serial tile parallel.
        tile_set = tasks.tile_set

        # Count the number of tiles
        self.f = open("tile_counter.txt","w")

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
    
            # Multithreading starts here.
            # The function to run every chunk.
            func = self.driver_serial_tile
    
            # Setup the array.
            manager = multiprocessing.Manager()
            
            all_grid_sets = {}
            # For tides
            all_grid_sets['grid_set_true_tide_vec'] = manager.dict()
            all_grid_sets['grid_set_tide_vec'] = manager.dict()
            all_grid_sets['grid_set_tide_vec_uq'] = manager.dict()
            all_grid_sets['grid_set_resid_of_secular'] = manager.dict()
            all_grid_sets['grid_set_resid_of_tides'] = manager.dict()
            all_grid_sets['grid_set_other_1'] = manager.dict()
            # For prediction evaluation
            all_grid_sets['grid_set_analysis'] = manager.dict()
            # For others
            all_grid_sets['counter'] = manager.dict()
    
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

            self.f.close()
    
            # Convert the results to normal dictionary.
            # Save the results to the class.
            print("Saving the results...")
            tasks.grid_set_true_tide_vec = dict(all_grid_sets['grid_set_true_tide_vec'])
            tasks.grid_set_tide_vec = dict(all_grid_sets['grid_set_tide_vec'])
            tasks.grid_set_tide_vec_uq = dict(all_grid_sets['grid_set_tide_vec_uq'])
    
            tasks.grid_set_resid_of_secular = dict(all_grid_sets['grid_set_resid_of_secular'])
            tasks.grid_set_resid_of_tides = dict(all_grid_sets['grid_set_resid_of_tides'])
            tasks.grid_set_other_1 = dict(all_grid_sets['grid_set_other_1'])
            tasks.grid_set_analysis = dict(all_grid_sets['grid_set_analysis'])

            ## end of dict.

        else:
            # Load the results from point_result
            print("Loading the results...")
            point_results = os.listdir(self.this_result_folder + '/point_result')

            for ip, point_pkl in enumerate(point_results):
                print(ip)
                pklfile = self.this_result_folder + '/point_result/' + point_pkl
                
                with open(pklfile,"rb") as f:
                    all_sets = pickle.load(f)

                if all_sets['true_tide_vec_set'] is not None:
                    tasks.grid_set_true_tide_vec.update(all_sets['true_tide_vec_set'])
                
                tasks.grid_set_tide_vec.update(all_sets['tide_vec_set'])

                tasks.grid_set_tide_vec_uq.update(all_sets['tide_vec_uq_set'])
       
                tasks.grid_set_resid_of_secular.update(all_sets['resid_of_secular_set'])
                tasks.grid_set_resid_of_tides.update(all_sets['resid_of_tides_set'])
                tasks.grid_set_other_1.update(all_sets['other_set_1'])
                tasks.grid_set_analysis.update(all_sets['analysis_set'])

        ## Save the final results in dictionary manager

        if self.task == "tides" and tasks.single_point_mode == False:

            print("Write results to disk")
            print("The task is tides")

            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_true_tide_vec.pkl', 'wb') as f:
                
                pickle.dump(tasks.grid_set_true_tide_vec, f)
    
            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_tide_vec.pkl','wb') as f:
                pickle.dump(tasks.grid_set_tide_vec, f)
    
            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_tide_vec_uq.pkl','wb') as f:
                pickle.dump(tasks.grid_set_tide_vec_uq, f)
    
            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_resid_of_secular.pkl','wb') as f:
                pickle.dump(tasks.grid_set_resid_of_secular, f)
    
            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_resid_of_tides.pkl','wb') as f:
                pickle.dump(tasks.grid_set_resid_of_tides, f)
    
            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_analysis.pkl','wb') as f:
                pickle.dump(tasks.grid_set_analysis, f)

            print("Done")

        elif self.task == "prediction_evaluation" and tasks.single_point_mode == False:

            print("Write results to disk")
            print("The task is prediction evaluation")

            with open(this_result_folder + '/' 
                        + str(test_id) + '_' + 'grid_set_analysis.pkl','wb') as f:
                pickle.dump(tasks.grid_set_analysis, f)

            print("Done")

        else:

            print("Do not save the results to disk")
            print("task: ", self.task)
            print("single point mode: ", tasks.single_point_mode)

        return 0

def main():

    # Timer starts.
    start_time = time.time()

    # Create a driver
    driver = driver_fourdvel()
    
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
