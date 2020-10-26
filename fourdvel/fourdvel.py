#!/usr/bin/env python3

# Author: Minyan Zhong
# Development starts in June, 2018

# The unit of time is day

import os
import sys
import pickle
import time
import pathlib
import glob

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from datetime import date

import multiprocessing

from basics import basics

class fourdvel(basics):

    def __init__(self, param_file=None):
        print(param_file)
        super(fourdvel,self).__init__()

        if param_file is not None:
            self.read_parameters(param_file)
        elif len(sys.argv)>1:
            self.read_parameters(sys.argv[1])
        else:
            raise Exception("Need to provide parameter file to class fourdvel")

        if self.proj == "Evans":

            # CSK
            # Key is track, values are dates            
            self.csk_data = {}

            self.csk_tracks = range(22)

            for it in self.csk_tracks:
                self.csk_data[it] = []
            self.csk_workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"

            # S1
            # Key is track, values are dates
            self.s1_data = {}

            self.s1_tracks = [37,52,169,65,7,50,64]
            self.s1_workdir = "/net/kraken/nobak/mzzhong/S1-Evans"
        
            for it in self.s1_tracks:
                self.s1_data[it] = []

            self.satellite_constants()

        elif self.proj == "Rutford":

            # CSK
            self.csk_data = {}

            self.csk_tracks = [8,10,23,25,40,52,55,67,69,82,97,99,114,126,128,129,141,143,156,158,171,172,173,186,188,201,203,215,218,230,231,232]

            self.csk_workdir = "/net/kraken/nobak/mzzhong/CSK-Rutford"

            for it in self.csk_tracks:
                self.csk_data[it] = []

            # S1
            self.s1_data = {}

            self.s1_tracks = [37,65,7]

            self.s1_workdir = "/net/kraken/nobak/mzzhong/S1-Evans"
        
            for it in self.s1_tracks:
                self.s1_data[it] = []

            self.satellite_constants()

        # Related folders
        #self.design_mat_folder = './design_mat'

        ########### Control the solution (Rutford) ##########
        # physical size     lat x lon
        #   100m x 100m     0.001 x 0.005
        #   500m x 500m     0.005 x 0.025
        resolution=self.resolution

        if self.proj == "Rutford":
            # lon/lat = 5
            
            if resolution == 1000:
                self.lat_step = 0.01
                self.lon_step = 0.05
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)

            elif resolution == 500:
                self.lat_step = 0.005
                self.lon_step = 0.025
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)

            elif resolution == 100:
                self.lat_step = 0.001
                self.lon_step = 0.005
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)
            else:
                print("Unknown resolution")
                raise Exception()

        elif self.proj == "Evans":
            # lon/lat = 4

            if resolution == 500:
                self.lat_step = 0.005
                self.lon_step = 0.02
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)
            
            elif resolution == 1000:
                self.lat_step = 0.01
                self.lon_step = 0.04
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)

            elif resolution == 2000:
                # ad hoc lon/lat=5
                self.lat_step = 0.02
                self.lon_step = 0.10
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)

            else:
                print("Unknown resolution")
                raise Exception()

        else:
            raise Exception("Unknown project name")
            return

        # The inverse to size
        self.lon_re = np.round(1/self.lon_step).astype(int)
        self.lat_re = np.round(1/self.lat_step).astype(int)

        # Round test point to an existing point
        if self.test_point is not None:
            print("Before rounding: ",self.test_point)
            self.test_point = self.point_rounding(self.test_point)
            print("After rounding: ", self.test_point)

        print("fourdvel initialization done")

    def read_parameters(self, param_file):
        print("Reading: ",param_file)
        f = open(param_file)
        params = f.readlines()
        f.close()

        # Quick and dirty intialization:
        self.data_uncert_const = None
        self.test_point = None
        self.single_point_mode = False
        self.simulation_use_external_up = False
        self.csk_excluded_tracks = []
        self.s1_excluded_tracks = []
        self.external_grounding_level_file = None
        self.simulation_mode = False

        fmt = '%Y%m%d'

        for param in params:
            
            try:    
                name,value = param.split(':')
                name = name.strip()
                value = value.strip()
            except:
                continue

            if name == 'pickle_dir':
                self.pickle_dir = value
                print('pickle_dir',value)
                pathlib.Path(self.pickle_dir).mkdir(exist_ok=True)

            if name == 'estimations_dir':
                self.estimations_dir = value
                print('estimations_dir',value)
                pathlib.Path(self.estimations_dir).mkdir(exist_ok=True)

            if name == 'proj':
                self.proj = value
                print('proj',value)

            if name == 'test_id':
                self.test_id = value
                print('test_id',value)

                self.estimation_dir = self.estimations_dir + "/" + str(self.test_id)
                if not os.path.exists(self.estimation_dir):
                    os.mkdir(self.estimation_dir)

                print("estimation dir: ", self.estimation_dir)

                ## Save the paramerer file
                if param_file=="params.in":
                    cmd = " ".join(["cp", param_file, "./params/"+str(self.test_id)+'_'+param_file])
                    print("copy param file: ",cmd)
                    os.system(cmd)

            if name == "test_point_file":

                test_point_file = value

                f_test_point = open(test_point_file)
                test_point_lines = f_test_point.readlines()
                f_test_point.close()
                
                for test_point_line in test_point_lines:
                    try:    
                        name1,value1 = test_point_line.split(':')
                        name1 = name1.strip()
                        value1 = value1.strip()
                    except:
                        continue

                    if name1=="test_point" and value1!="None":
                        the_point = [float(x) for x in value1.split(",")]
                        self.test_point = self.float_lonlat_to_int5d(the_point)
                    else:
                        continue
                    print("test_point", value1, self.test_point)
                    print("Turn on single point mode")
                    self.single_point_mode = True

            if name == "inversion_method":

                self.inversion_method = value
                print("inversion_method", value)

            if name == "sampling_data_sigma":

                self.sampling_data_sigma = float(value)
                print("sampling_data_sigma", value)

            if name == 'resolution':
                self.resolution = int(value)
                print('resolution: ',value)

            if name == 'grid_set_data_uncert_name':
                self.grid_set_data_uncert_name = value
                print('grid_set_data_uncert_name: ',value)

            if name == "bbox":
                borders = [ s.strip() for s in value.split(",") ]
                self.bbox = [ None if border=="None" else self.float_to_int5d(float(border)) for border in borders]
                self.bbox_s, self.bbox_n, self.bbox_e, self.bbox_w = self.bbox

                print('bbox: ',self.bbox)

            # CSK
            if name == 'use_csk':
                if value == 'True':
                    self.use_csk = True
                elif value == "False":
                    self.use_csk = False
                else:
                    raise Exception("use_csk: " + value)
                print('use_csk: ',value)

            if name == 'csk_data_mode':
                self.csk_data_mode = int(value)
                print('csk_data_mode',value)
                if self.csk_data_mode in [1,2]:
                    self.simulation_mode = True

            if name == 'csk_data_date_option':
                self.csk_data_date_option = value
                print('csk_data_date_option: ', value)

            if name == 'csk_simulation_data_uncert_const':
                self.csk_simulation_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('csk_simulation_data_uncert_const: ',self.csk_simulation_data_uncert_const)

            if name == 'csk_data_uncert_const':
                self.csk_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('csk_data_uncert_const: ',self.csk_data_uncert_const)

            if name == 'csk_id':
                self.csk_id = int(value)
                print('csk_id: ',value)

            if name == 'csk_version':
                self.csk_version = value
                if value == "None":
                    self.csk_version = None
                print('csk_version: ',value)

            if name == 'csk_start':
                self.csk_start = datetime.datetime.strptime(value, fmt).date()
                print('csk_start: ',value)

            if name == 'csk_end':
                self.csk_end = datetime.datetime.strptime(value, fmt).date()
                print('csk_end: ',value)

            if name == 'csk_log':
                self.csk_log = value
                print('csk_log: ',value)

            # S1
            if name == 'use_s1':
                if value == 'True':
                    self.use_s1 = True
                else:
                    self.use_s1 = False
                print('use_s1: ',value)

            if name == 's1_data_mode':
                self.s1_data_mode = int(value)
                print('s1_data_mode',value)
                if self.s1_data_mode in [1,2]:
                    self.simulation_mode = True

            if name == 's1_data_date_option':
                self.s1_data_date_option = value
                print('s1_data_date_option: ', value)

            if name == 's1_simulation_data_uncert_const':
                self.s1_simulation_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('s1_simulation_data_uncert_const: ',self.s1_simulation_data_uncert_const)

            if name == 's1_data_uncert_const':
                self.s1_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('s1_data_uncert_const: ',self.s1_data_uncert_const)

            if name == 's1_id':
                self.s1_id = int(value)
                print('s1_id: ',value)

            if name == 's1_version':
                self.s1_version = value
                if value == "None":
                    self.s1_version = None
                print('s1_version: ',value)

            if name == 's1_start':
                self.s1_start = datetime.datetime.strptime(value, fmt).date()
                print('s1_start: ',value)

            if name == 's1_end':
                self.s1_end = datetime.datetime.strptime(value, fmt).date()
                print('s1_end: ',value)


            if name == 's1_excluded_tracks':
                self.s1_excluded_tracks = [int(x) for x in value.split(',')]
                print('s1_excluded_tracks: ',self.s1_excluded_tracks)

            # Modeling
            if name == 'modeling_tides':
                self.modeling_tides = [x.strip() for x in value.split(',')]
                self.n_modeling_tides = len(self.modeling_tides)
                print('modeling_tides: ', self.modeling_tides)

                if self.modeling_tides[0]=='None':
                    self.modeling_tides = []
                    self.n_modeling_tides = 0

            if name == 'horizontal_prior':
                if value == 'True':
                    self.horizontal_prior = True
                else:
                    self.horizontal_prior = False

            if name == 'no_secular_up':
                if value == 'True':
                    self.no_secular_up = True
                else:
                    self.no_secular_up = False

            if name == 'up_short_period':
                if value == 'True':
                    self.up_short_period = True
                else:
                    self.up_short_period = False

            if name == 'horizontal_long_period':
                if value == 'True':
                    self.horizontal_long_period = True
                else:
                    self.horizontal_long_period = False

            ## Simulation ##
            if name == 'simulation_method':
                self.simulation_method = value
                print("simulation_method: ",self.simulation_method)

            if name == 'simulation_tides':
                self.simulation_tides = [x.strip() for x in value.split(',')]
                self.n_simulation_tides = len(self.simulation_tides)
                print('simulation_tides: ', self.simulation_tides)

                if self.simulation_tides[0]=='None':
                    self.simulation_tides = []
                    self.n_simulation_tides = 0
            
            if name == 'simulation_grounding_level':
                self.simulation_grounding_level = float(value)
                print("simulation_grounding_level: ",self.simulation_grounding_level)

            if name == 'simulation_use_external_up':
                if value == 'True':
                    self.simulation_use_external_up = True
                else:
                    self.simulation_use_external_up = False
                print('simulation_use_external_up: ',value)


            ## External up ##
            if name == 'external_up_disp_file':
                if value == "None":
                    self.external_up_disp_file = None
                else:
                    self.external_up_disp_file = value
                print('external_up_disp_file: ',value)

            ## External grounding level data ##
            if name == 'external_grounding_level_file':
                self.external_grounding_level_file = value
                print('external_grounding_level_file: ',value)

            ## Analysis ##
            if name == 'analysis_name':
                self.analysis_name = value
                print("analysis_name: ",self.analysis_name)

            ## Output ###
            if name == "output_true":
                if value == 'True':
                    self.output_true = True
                else:
                    self.output_true = False
                print('output_true: ',value)

            if name == "output_est":
                if value == 'True':
                    self.output_est = True
                else:
                    self.output_est = False
                print('output_est: ',value)

            if name == "output_uq":
                if value == 'True':
                    self.output_uq = True
                else:
                    self.output_uq = False
                print('output_uq: ',value)

            if name == "output_others":
                if value == 'True':
                    self.output_others = True
                else:
                    self.output_others = False
                print('output_others: ',value)

            if name == "output_resid":
                if value == 'True':
                    self.output_resid = True
                else:
                    self.output_resid = False
                print('output_resid: ',value)

            if name == "output_difference":
                if value == 'True':
                    self.output_difference = True
                else:
                    self.output_difference = False
                print('output_difference: ',value)

            if name == "output_analysis":
                if value == 'True':
                    self.output_analysis = True
                else:
                    self.output_analysis = False
                print('output_analysis: ',value)

        print("Done with reading parameters")
        return 0
            
    def get_CSK_trackDates_from_log(self):
        import csv
        from CSK_Utils import CSK_Utils

        # csk_data[track_number] = [date1, date2, date3,...]
        csk_data = self.csk_data

        csk_start = self.csk_start
        csk_end = self.csk_end

        # Not all data are available, currently, so I read the files exported from E-GEOS. I will switch to real data
        
        #file_folder = self.csk_log
        #data_file = os.path.join(file_folder,'all.csv')
        data_file = self.csk_log

        csk = CSK_Utils()

        tot_products = 0

        num_products = 0
        num_frames = 0

        with open(data_file) as dataset:
            csv_reader = csv.reader(dataset, delimiter=';')
            line = 0
            for row in csv_reader:
                line = line + 1
                if line == 1:
                    continue
                
                # Count as one product.
                tot_products = tot_products + 1

                # Satellite 
                sate = 'CSKS' + row[1][-1]
                # Date
                acq_datefmt = row[5].split(' ')[0]
                # Direction
                direction = row[7][0]

                #print(acq_datefmt)

                # Convert date string to date object
                date_comp = [int(item) for item in acq_datefmt.split('-')]
                theDate = date(date_comp[0],date_comp[1],date_comp[2])

                # If the date is within the range set by user
                if theDate >= csk_start and theDate < csk_end:
    
                    # Find the figure out the track number.                
                    tracks = csk.date2track(day=theDate, sate=sate)[sate]
                    #print(line, tracks, direction)
                    #print(row)
                   
                    if direction == 'A':
                        track = [ i for i in tracks if i<=10 ]
                    else:
                        track = [ i for i in tracks if i>=11 ]

                    #print(track)

                    # Record it.    
                    if track[0] in csk_data.keys():
                        csk_data[track[0]].append(theDate)
                    else:
                        csk_data[track[0]] = [theDate]
    
                    num_frames = num_frames + csk.numOfFrames[track[0]]
                    num_products += 1
    
        print("Total number of products in log: ", tot_products)
        print("Number of products: ", num_products)
        print("Number of frames: ", num_frames)

        # Sort the dates of each track.
        # Output the track info
        for track_num in sorted(csk_data.keys()):
            csk_data[track_num].sort()
            print(track_num)
            print("Number of dates: ", len(csk_data[track_num]))
            #print(csk_data[track_num])
       
        return 0

    def get_CSK_trackDates(self):
        csk_data = self.csk_data
        csk_start = self.csk_start
        csk_end = self.csk_end

        tracklist = self.csk_tracks

        # Set the option for setting available data dates in synthetic test
        # For Evans CSK data is not available, so we make the data log-based
        csk_data_date_option = self.csk_data_date_option

        if csk_data_date_option=="data_based":
            for track_num in tracklist: 
                if self.proj == "Evans": 
                    filefolder = self.csk_workdir + '/track_' + str(track_num).zfill(2) + '0' + '/raw/201*'
                elif self.proj == "Rutford":
                    filefolder = self.csk_workdir + '/track_' + str(track_num).zfill(3) + '_0' + '/raw/201*'
                else:
                    raise Exception()

                filelist = glob.glob(filefolder)
                csk_data[track_num] = []
    
                for rawfile in filelist:
                    datestr = rawfile.split('/')[-1]
                    if len(datestr)==8:
                        theDate = date(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
                        if theDate >= csk_start and theDate < csk_end:
                            csk_data[track_num].append(theDate)
    
                csk_data[track_num] = list(set(csk_data[track_num]))
                csk_data[track_num].sort()

                print("track_num: ",track_num,end=",  ")
                print("Number of dates: ",len(csk_data[track_num]))

        elif csk_data_date_option == "log_based":
            self.get_CSK_trackDates_from_log()

        elif csk_data_date_option == "no_data":
            for track_num in tracklist:
                csk_data[track_num] = []

        else:
            print("csk_data_date_option", csk_data_option, "is not defined yet")
            raise Exception("dates information are not available")

        return 0

    def get_S1_trackDates(self):
        from S1_Utils import S1_Utils

        s1_data = self.s1_data
        s1_start = self.s1_start
        s1_end = self.s1_end

        s1 = S1_Utils()

        tracklist = self.s1_tracks

        s1_data_date_option = self.s1_data_date_option

        if s1_data_date_option=="data_based":
            for track_num in tracklist: 
            
                filefolder = self.s1_workdir + '/data_' + str(track_num) + '/*zip'
                filelist = glob.glob(filefolder)
                s1_data[track_num] = []
    
                for zipfile in filelist:
                    datestr = zipfile.split('_')[6][0:8]
                    theDate = date(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
    
                    if theDate >= s1_start and theDate < s1_end:
                        s1_data[track_num].append(theDate)
    
                s1_data[track_num] = list(set(s1_data[track_num]))
                s1_data[track_num].sort()

                print("track_num: ",track_num)
                print("Number of dates: ", len(s1_data[track_num]))

        elif s1_data_date_option == "projected":
            for track_num in tracklist:
                s1_data[track_num] = []
    
                ref_date = s1.ref_date[track_num]

                day = s1_start
                while day < s1_end:
                    if ((day - ref_date).days % 6==0 and track_num!=7) or \
                        ((day - ref_date).days % 12==0 and track_num==7) :
                        s1_data[track_num].append(day)
                    day = day + datetime.timedelta(days=1)

                print("track_num: ",track_num)
                print("Number of dates: ", len(s1_data[track_num]))
        
        elif s1_data_date_option == "no_data":
            for track_num in tracklist:
                s1_data[track_num] = []
                print("track_num: ",track_num)
                print("Number of dates: ", len(s1_data[track_num]))
        else:
            print("option", s1_data_date_option, "is not defined yet")
            raise Exception("dates are not available")

        return 0

    def get_grid_set_velo_info(self):

        self.grid_set_matched_velo_2d_name = "_".join((self.grid_set_name, "matched_ref_velo_2d"))
        self.grid_set_velo_2d_name = "_".join((self.grid_set_name, "ref_velo_2d"))
        self.grid_set_velo_3d_name = "_".join((self.grid_set_name, "ref_velo_3d"))
    
        self.grid_set_matched_velo_2d_pkl_name = self.pickle_dir + '/' + self.grid_set_matched_velo_2d_name + '.pkl'

        self.grid_set_velo_2d_pkl_name = self.pickle_dir + '/' + self.grid_set_velo_2d_name + '.pkl'

        self.grid_set_velo_3d_pkl_name = self.pickle_dir + '/' + self.grid_set_velo_3d_name + '.pkl'

        return 0

    def get_grid_set_velo(self):

        # Get names
        self.get_grid_set_velo_info()

        if os.path.exists(self.grid_set_velo_3d_pkl_name):
            print('Loading grid_set_velo...')
            with open(self.grid_set_velo_3d_pkl_name,'rb') as f:
                self.grid_set_velo = pickle.load(f)
        else:
            print(self.grid_set_velo_3d_pkl_name)
            raise Exception("Unable to load 3d velocity reference model")

    def get_tile_set_info(self):

        if self.proj=="Rutford":

            # For 1000m resolution
            if self.resolution == 1000:
                self.tile_lon_step = 2
                self.tile_lat_step = 0.4

            # For 500m resolution
            if self.resolution == 500:
                self.tile_lon_step = 1
                self.tile_lat_step = 0.2
    
            # For 100m resolution
            if self.resolution == 100:
                self.tile_lon_step = 0.2
                self.tile_lat_step = 0.04
        
        elif self.proj=="Evans":

            if self.resolution == 500:
                # For 500m resolution
                self.tile_lon_step = 0.5
                self.tile_lat_step = 0.1

            if self.resolution == 1000:
                # For 1000m resolution
                self.tile_lon_step = 1
                self.tile_lat_step = 0.2

            if self.resolution == 2000:
                # For 2000m resolution
                self.tile_lon_step = 2
                self.tile_lat_step = 0.4


        self.tile_set_name = "_".join((self.grid_set_name, "tile_set","lon_step",str(self.tile_lon_step),"lat_step", str(self.tile_lat_step)))

        return (self.pickle_dir + '/' + self.tile_set_name + '.pkl')

    def get_tile_set(self):

        self.tile_set_pkl_name = self.get_tile_set_info()
        if os.path.exists(self.tile_set_pkl_name):
            print('Loading tile_set ...')
            with open(self.tile_set_pkl_name,'rb') as f:
                self.tile_set = pickle.load(f)
            print("total number of tiles: ", len(self.tile_set))
        else:
            print("tile set file is missing: ", self.tile_set_pkl_name)
            raise Exception()

        return 0 

    def get_data_uncert(self):

        self.grid_set_data_uncert = None
        if hasattr(self,'grid_set_data_uncert_name'):
            grid_set_data_uncert_set_pkl = self.grid_set_data_uncert_name + '.pkl'
    
            if os.path.exists(grid_set_data_uncert_set_pkl):
                print('Loading data uncert set ...')
                with open(grid_set_data_uncert_set_pkl,'rb') as f:
                    self.grid_set_data_uncert = pickle.load(f)


    def get_used_datasets(self):

        # Find the datasets
        datasets = self.datasets
        used_datasets = [datasets[i] for i, use in enumerate([self.use_csk, self.use_s1]) if use]

        self.used_datasets = used_datasets

        return 0

    def get_grid_set_sources(self):

        datasets = self.datasets
        # Set the data sources for generating grid_set
        sources = {}
        if self.proj == "Rutford":
            sources = {}
            #sources["csk"] = "20190901_v12"
            #sources["s1"] = "unknown"
            sources["csk"]= "_".join(filter(None,(str(self.csk_id), self.csk_version)))
            sources["s1"] = "_".join(filter(None,(str(self.s1_id), self.s1_version)))
 
        elif self.proj == "Evans":
            sources = {}
            #sources["csk"] = "20180712"
            #sources["s1"] = "20200102_v12"
            sources["csk"]= "_".join(filter(None,(str(self.csk_id), self.csk_version)))
            sources["s1"] = "_".join(filter(None,(str(self.s1_id), self.s1_version)))
        else:
             raise Exception("Unknown project name")

        used_sources = [sources[datasets[i]] for i, use in enumerate([self.use_csk, self.use_s1]) if use]

        self.grid_set_sources = sources
        self.used_grid_set_sources = used_sources

        return 0

    def get_grid_set_info(self):

        self.get_used_datasets()
        self.get_grid_set_sources()

        if self.proj == "Rutford":
            grid_set_prefix = "grid_set_csk-r_point"
        elif self.proj == "Evans":
            grid_set_prefix = "grid_set_csk-e_point"
        else:
            raise Exception("Unknown project name")

        grid_set_datasets = "_".join(self.used_datasets)

        grid_set_sources = "_".join(self.used_grid_set_sources)

        grid_set_resolution = str(self.resolution)

        # Finalize the name
        self.grid_set_name= "_".join((grid_set_prefix, grid_set_datasets, grid_set_sources, grid_set_resolution))

        return (self.pickle_dir + '/' + self.grid_set_name + '.pkl')

    def get_grid_set_v2(self):

        self.grid_set_pkl_name = self.get_grid_set_info()
        
        if os.path.exists(self.grid_set_pkl_name):
            print('Loading grid_set...')

            with open(self.grid_set_pkl_name,'rb') as f:
                self.grid_set = pickle.load(f)

            print('total number of grid points: ', len(self.grid_set))

        else:
            print("grid set file missing: ", self.grid_set_pkl_name)
            raise Exception()

    def coverage(self): 

        # Show the coverage of grid points. 
        
        grid_set = self.grid_set

        fig = plt.figure(1, figsize=(8,8))
        ax = fig.add_subplot(111)
        
        vec_count = {}
        plotx = {}
        ploty = {}

        for key in sorted(grid_set.keys()):
            
            track_count = len(grid_set[key])

            if track_count not in vec_count.keys():
                vec_count[track_count] = 1
                plotx[track_count] = [key[0]]
                ploty[track_count] = [key[1]]
            else:
                vec_count[track_count] += 1
                plotx[track_count].append(key[0])
                ploty[track_count].append(key[1])
 
            #if len(grid_set[key])==4:
                #print(key,grid_set[key])
                #plotx.append(key[0])
                #ploty.append(key[1])

        #print('total number of grids:', len(grid_set.keys()))
        #print('number of vectors: ', vec_count)

        # Make the figure.
        #colors = sns.color_palette('bright',10)
        colors = sns.cubehelix_palette(6, start=.5, rot=-.75)
        
        symbol = ['r.','g.','b.','c.','k.','m.','y.','w.']

        for track_count in sorted(plotx.keys()):
            print(track_count,len(plotx[track_count]))
            ax.plot(plotx[track_count],ploty[track_count], color=colors[track_count-1], marker='.',markersize=0.3, linestyle='None', label=str(track_count) + ' Track: ' + str(vec_count[track_count]))
            #ax.plot(plotx[track_count],ploty[track_count],symbol[track_count-1], markersize=0.3, label=str(track_count) + ' Track: ' + str(vec_count[track_count]))

        ax.legend(markerscale=40)
        ax.set_title('coverage')
        fig.savefig('coverage.png',format='png')

        return 0

    def get_timings_info(self):

        fmt = "%Y%m%d"
        timings_pkl_name = self.pickle_dir + '/' + '_'.join(['timings', 'csk', self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt)]) + '.pkl'

        return timings_pkl_name

    def get_timings(self):

        self.timings_pkl_name = self.get_timings_info()
        timings_pkl_name = self.timings_pkl_name
        print('timing file: ', timings_pkl_name)

        redo = 1
        if os.path.exists(timings_pkl_name) and redo == 0:
            with open(timings_pkl_name, 'rb') as f:
                self.timings = pickle.load(f)
        else:
            # Derive all timings (date + time fraction)
            self.timings = []

            # CSK
            for key in self.csk_data.keys():
                tfrac = self.track_timefraction['csk',key]
                for the_date in self.csk_data[key]:
                    self.timings.append((the_date, round(tfrac,4)))
            # S1
            for key in self.s1_data.keys():
                tfrac = self.track_timefraction['s1',key]
                for the_date in self.s1_data[key]:
                    self.timings.append((the_date, round(tfrac,4)))

            self.timings = sorted(self.timings)

            with open(timings_pkl_name,'wb') as f:
                pickle.dump(self.timings, f)

        return 0

    def get_tidal_model(self):

        print("Get the tidal model ...")

        tide_file = self.external_up_disp_file

        if tide_file:
            if os.path.exists(tide_file):
                fid = open(tide_file)
                lines = fid.readlines()
                taxis = []
                data = []
                for line in lines:
                    t,z = line.split()
                    taxis.append(float(t))
                    data.append(float(z))
        
                self.tide_taxis = taxis
                self.tide_data = data
                self.tide_t_delta = self.tide_taxis[1] - self.tide_taxis[0]
            else:
                raise Exception("Tide file does not exist: " + tide_file)

        return

    def get_timings_tide_heights(self):

        # From (date + time fraction) to tide_heights
        self.timings_tide_heights = {}
        t_origin = self.t_origin.date()
 
        for timing in self.timings:
            
            the_date, t_frac = timing
            relative_time = (the_date - t_origin).days + t_frac

            # Find the index
            idx = int((relative_time - self.tide_taxis[0])/self.tide_t_delta)
            self.timings_tide_heights[timing] = self.tide_data[idx]

        #print(self.timings_tide_heights)
        return 0

    def get_design_mat_set(self):

        from forward import forward
        fwd = forward()
        redo = 1
        fmt = "%Y%m%d"

        # For modeling
        # Use the tides set by the parameter file
        self.model_design_mat_set_pkl = self.pickle_dir +'/' + '_'.join(['model_design_mat_set', 'csk',self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt)] + self.modeling_tides ) + '.pkl'

        model_design_mat_set_pkl = self.model_design_mat_set_pkl

        print('Find model_design_mat_set:', model_design_mat_set_pkl)

        if os.path.exists(model_design_mat_set_pkl) and redo==0:
            with open(model_design_mat_set_pkl,'rb') as f:
                self.model_design_mat_set = pickle.load(f)
        else:
            self.model_design_mat_set = fwd.design_mat_set(self.timings, self.modeling_tides)
            print("Size of design mat set: ",len(self.model_design_mat_set))

            with open(model_design_mat_set_pkl,'wb') as f:
                pickle.dump(self.model_design_mat_set,f)

        # Construct matrix simulation (here "rutford" just means the used model)
        # The tides used in modeling is more than the tides used in inversion
        if self.csk_data_mode in [1,2] or self.s1_data_mode in [1,2]:

            self.rutford_design_mat_set_pkl = self.pickle_dir +'/'+ '_'.join(['rutford_design_mat_set', 'csk',self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt)]) + '.pkl'

            rutford_design_mat_set_pkl = self.rutford_design_mat_set_pkl
    
            if os.path.exists(rutford_design_mat_set_pkl) and redo==0:
                with open(rutford_design_mat_set_pkl, 'rb') as f:
                    self.rutford_design_mat_set = pickle.load(f)
    
            else:
                rutford_tides = self.simulation_tides
                self.rutford_design_mat_set = fwd.design_mat_set(self.timings, rutford_tides)
                print("Size of design mat set: ", len(self.rutford_design_mat_set))
    
                with open(rutford_design_mat_set_pkl,'wb') as f:
                    pickle.dump(self.rutford_design_mat_set,f)
        return 0

    def get_up_disp_set(self, point_set, offsetfields_set):

        up_disp_set = {}
        for point in point_set:
            up_disp_set[point] = self.get_up_disp_for_point(point, offsetfields_set[point])

        return up_disp_set

    def get_up_disp_for_point(self, point, offsetfields):

        tide_height_master = []
        tide_height_slave = []
        for i in range(len(offsetfields)):
            timing_a = (offsetfields[i][0], round(offsetfields[i][4],4))
            timing_b = (offsetfields[i][1], round(offsetfields[i][4],4))

            # Tide height for the two dates
            tide_height_master.append(self.timings_tide_heights[timing_a])
            tide_height_slave.append(self.timings_tide_heights[timing_b])

        tide_height_master = np.asarray(tide_height_master)
        tide_height_slave = np.asarray(tide_height_slave)

        return (tide_height_master, tide_height_slave)


    def get_stack_design_mat_set(self, point_set, design_mat_set, offsetfields_set):

        stack_design_mat_set = {}
        for point in point_set:
            stack_design_mat_set[point] = self.get_stack_design_mat_point(point, design_mat_set, offsetfields_set[point])

        return stack_design_mat_set

    def get_stack_design_mat_point(self, point, design_mat_set, offsetfields):

        # At point level
        # Stack the design matrix for all pairs
        stacked_design_mat_EN_ta = []
        stacked_design_mat_EN_tb = []
        stacked_design_mat_U_ta = []
        stacked_design_mat_U_tb = []

        # Note that it is possible that offsetfields is empty

        for i in range(len(offsetfields)):

            timing_a = (offsetfields[i][0], round(offsetfields[i][4],4))
            timing_b = (offsetfields[i][1], round(offsetfields[i][4],4))

            # design_mat_set shape: 
            # 2 * n_params for EN
            # 1 * n_params for U
            stacked_design_mat_EN_ta.append(design_mat_set[timing_a][:2,:])
            stacked_design_mat_EN_tb.append(design_mat_set[timing_b][:2,:])

            stacked_design_mat_U_ta.append(design_mat_set[timing_a][2,:])
            stacked_design_mat_U_tb.append(design_mat_set[timing_b][2,:])

        # Vertical stack
        if stacked_design_mat_EN_ta:
            stacked_design_mat_EN_ta = np.vstack(tuple(stacked_design_mat_EN_ta))

        if stacked_design_mat_EN_tb:
            stacked_design_mat_EN_tb = np.vstack(tuple(stacked_design_mat_EN_tb))

        if stacked_design_mat_U_ta:
            stacked_design_mat_U_ta = np.vstack(tuple(stacked_design_mat_U_ta))

        if stacked_design_mat_U_tb:
            stacked_design_mat_U_tb = np.vstack(tuple(stacked_design_mat_U_tb))

        return (stacked_design_mat_EN_ta, stacked_design_mat_EN_tb, stacked_design_mat_U_ta, stacked_design_mat_U_tb)

    
    def get_offset_field_stack(self):

        # !!! Ad hoc !!!
        # Turn off csk for Evans project
        #if self.proj == "Evans":
        #    self.use_csk = False 
   
        # Create the dictionary for all offset fields 
        self.offsetFieldStack_all = {}

        # Find the necessary tracks
        skip_this = True
        if self.test_point is not None and skip_this == False:
            print(self.test_point)
            tracks_set = self.grid_set[self.test_point]
            track_num_set = set()
            for track in tracks_set:
                track_num_set.add((track[0],track[3]))
                
        else:
            track_num_set = {}

        track_num_set = sorted(track_num_set)
        print("necessary tracks: ", track_num_set)
        print(self.use_csk)

        if self.use_csk:
            for track_num in self.csk_tracks:
                if track_num_set and not (track_num,"csk") in track_num_set:
                    continue

                print("Hey", track_num)

                if self.proj == "Evans":
                    track_offsetFieldStack_pkl = os.path.join(self.csk_workdir, "track_" + str(track_num).zfill(2) + '0', "cuDenseOffsets", "_".join(filter(None, ["offsetFieldStack", str(self.csk_id), self.csk_version])) +  ".pkl")

                elif self.proj == "Rutford":
                    track_offsetFieldStack_pkl = os.path.join(self.csk_workdir, "track_" + str(track_num).zfill(3) + '_0', "cuDenseOffsets", "_".join(filter(None, ["offsetFieldStack", str(self.csk_id), self.csk_version])) +  ".pkl")
                else:
                    raise Exception()

                if os.path.exists(track_offsetFieldStack_pkl):
                    print("Loading: ", track_offsetFieldStack_pkl)
                    with open(track_offsetFieldStack_pkl,'rb') as f:
                        offsetFieldStack = pickle.load(f)
                        self.offsetFieldStack_all[("csk", track_num)]= offsetFieldStack
                else:
                    print(track_offsetFieldStack_pkl + ' does not exist')
                    assert self.csk_data_mode==1, "Test mode must be 1"

        if self.use_s1:
            for track_num in self.s1_tracks:
                if track_num_set and not (track_num, "s1") in track_num_set:
                    continue

                track_offsetFieldStack_pkl = os.path.join(self.s1_workdir, "track_" + str(track_num), "cuDenseOffsets", "_".join(filter(None, ["offsetFieldStack", str(self.s1_id), self.s1_version])) + ".pkl")

                if os.path.exists(track_offsetFieldStack_pkl):
                    print("Loading: ", track_offsetFieldStack_pkl)
                    with open(track_offsetFieldStack_pkl,'rb') as f:
                        offsetFieldStack = pickle.load(f)
                        self.offsetFieldStack_all[("s1", track_num)] = offsetFieldStack
                else:
                    print(track_offsetFieldStack_pkl + ' does not exist')
                    assert self.s1_data_mode==1, "Test mode must be 1"

        return 0

    def point_rounding(self, point):

        point_lon, point_lat = point

        print(self.lon_step_int)
        print(self.lat_step_int)

        point_lon = point_lon // self.lon_step_int * self.lon_step_int
        point_lat = point_lat // self.lat_step_int * self.lat_step_int

        return (point_lon, point_lat)

    def preparation(self):

        # Get pre-defined grid points and the corresponding tracks and vectors.
        self.get_grid_set_v2()
        self.get_tile_set()
        self.get_data_uncert()

        # Get reference velocity model for synthetic test
        self.get_grid_set_velo()

        # Show the counts.
        print('Number of total grid points: ', len(self.grid_set.keys()))

        # Show the coverage.
        #self.coverage()

        # Get the available dates according to settings.
        if self.use_csk and self.use_s1:
            self.get_CSK_trackDates()
            self.get_S1_trackDates()
        
        elif self.use_csk:
            self.get_CSK_trackDates()
        
        elif self.use_s1:
            self.get_S1_trackDates()


        # Prepartion design matrix libaray for inversion
        self.get_timings()
        
        # Load external tidal model
        self.get_tidal_model()

        # Find the tide height for all
        self.get_timings_tide_heights()

        # Find all the design matrix
        self.get_design_mat_set()

        # Load offset field stack data
        self.get_offset_field_stack()

        return 0

    def tracks_to_full_offsetfields(self, tracks):
        
        # Deduce the available offsetfields from all tracks
        csk_data = self.csk_data
        s1_data = self.s1_data

        track_timefraction = self.track_timefraction

        offsetfields = []

        for it in range(len(tracks)):

            #print(tracks[it])
            
            track_num = tracks[it][0]
            vec1 = tracks[it][1]
            vec2 = tracks[it][2]
            sate = tracks[it][3]

            #print(track_num)
            #print(vec1)
            #print(vec2)

            if sate=='csk':
                dates = csk_data[track_num]
                max_delta=8
                t_frac = track_timefraction[('csk',track_num)]

            elif sate=='s1':
                dates = s1_data[track_num]
                max_delta=12
                t_frac = track_timefraction[('s1',track_num)]

            else:
                raise Exception('unfounded satellite name')

            # the offsetfields
            for d1 in dates:
                for d2 in dates:
                    if d1<d2 and (d2-d1).days<=max_delta:
                        #print(d1,d2)
                        offsetfields.append([d1,d2,vec1,vec2,t_frac])

            #print(len(offsetfields))
            #print(stop)

        return offsetfields

    def build_G_set(self, point_set, offsetfields_set):
        
        linear_design_mat_set = {}
        for point in point_set:
            offsetfields = offsetfields_set[point]
            linear_design_mat_set[point] = self.build_G(point=point, offsetfields=offsetfields)

        return linear_design_mat_set

    def build_G(self, point=None, tracks=None, offsetfields=None):

        if point is not None:
            
            lon,lat = point

        if tracks is None and offsetfields is None:
            print('Please provide data info on this grid point')
            return

        elif tracks is not None:
            # Only track info is provided. Using data catalog.
            offsetfields = self.tracks_to_full_offsetfields(tracks)
        
        elif offsetfields is not None:
            # real offsetfields are provided.
            pass

        # Control the number of offsetfields
        n_offsets = len(offsetfields)

        # Important: accounting for no offsetfield scenario
        if n_offsets ==0:
            G = np.zeros(shape=(1,1)) + np.nan
            return G 

        ###############################################################

        ## Build the G matrix
        modeling_tides = self.modeling_tides
        n_modeling_tides = self.n_modeling_tides
        tide_periods = self.tide_periods

        t_origin = self.t_origin.date()

        # Build up delta_td, delta_cos and delta_sin.
        delta_td = np.zeros(shape=(n_offsets,))
        delta_cos = np.zeros(shape=(n_offsets,n_modeling_tides))
        delta_sin = np.zeros(shape=(n_offsets,n_modeling_tides))
        
        for i in range(n_offsets):

            t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]

            delta_td[i] = (offsetfields[i][1] - offsetfields[i][0]).days

            for j in range(n_modeling_tides):

                tide_name = modeling_tides[j]

                omega = 2 * np.pi / tide_periods[tide_name]
            
                delta_cos[i,j] = np.cos(omega*t_b) - np.cos(omega*t_a)
                delta_sin[i,j] = np.sin(omega*t_b) - np.sin(omega*t_a)

        n_rows = n_offsets * 2 # Each offset corresponds to a vector.

        # E, N, U components.
        n_cols = 3 + n_modeling_tides * 6 # cosE, cosN, cosU and sinE, sinN, sinU.
        
        ## G formation.
        G = np.zeros(shape=(n_rows,n_cols))

        # Iterate over offsetfields
        for i in range(n_offsets):

            # Find the observation vector (los, azi) refering to "create_grid_set"
            vecs = [offsetfields[i][2],offsetfields[i][3]]

            # Two observation vectors
            for j in range(2):

                # Get the vector (represent E,N,U)
                vector = np.asarray(vecs[j])

                # Row entries of the observation.
                row = np.zeros(shape=(n_cols,))

                # Secular component.
                row[0:3] = vector * delta_td[i]
                
                # Tidal components. (Projection)
                for k in range(n_modeling_tides):
                    row[3*(2*k+1):3*(2*k+2)] = vector * delta_cos[i,k]
                    row[3*(2*k+2):3*(2*k+3)] = vector * delta_sin[i,k]

                # Put them in into G.
                G[i*2+j,:] = row

        return G
        # End of building.


    def modify_G_set(self, point_set, G_set, offsetfields_set, grounding_level):

        # Extract the stack of up displacement from the tide model
        up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

        for point in point_set:

            G = G_set[point]
            tide_height_master_model, tide_height_slave_model = up_disp_set[point]

            # Should not do scaling here!
            #velo_model = self.grid_set_velo[point]
            #tide_height_master = tide_height_master_model * velo_model[2]
            #tide_height_slave = tide_height_slave_model * velo_model[2]

            # No scaling
            tide_height_master = tide_height_master_model
            tide_height_slave = tide_height_slave_model

            offsetfields = offsetfields_set[point]

            if self.external_grounding_level_file is None:
                given_grounding_level = grounding_level
            else:

                # Try to get the value
                try:
                    given_grounding_level = grounding_level[point]['optimal_grounding_level']
                    print("optimal grounding level: ", given_grounding_level)
                except:
                    given_grounding_level = -10

                if np.isnan(given_grounding_level):
                    given_grounding_level = -10

            G_set[point] = self.modify_G(point=point, offsetfields=offsetfields, G=G, tide_height_master = tide_height_master, tide_height_slave = tide_height_slave, grounding_level = given_grounding_level)

        return G_set

    def modify_G(self, point, offsetfields, G, tide_height_master, tide_height_slave, grounding_level):

        lon,lat = point

        # Control the number of offsetfields
        n_offsets = len(offsetfields)

        # Important: accounting for the no offsetfield scenario
        if n_offsets ==0:
            G = np.zeros(shape=(1,1)) + np.nan
            return G 

        ###############################################################

        ## Modify the G matrix
        # Find the shape of G
        n_rows, n_cols = G.shape

        # Perform clipping
        #print("shape: ",tide_height_master.shape)
        #print("shape: ",grounding_level)

        tide_height_master[tide_height_master<grounding_level]=grounding_level
        tide_height_slave[tide_height_slave<grounding_level]=grounding_level

        # Find the vertical displacement
        disp_up = (tide_height_slave - tide_height_master).reshape(n_offsets,1)

        # Find the vertical displacement vector
        disp_up_vecs = np.hstack((np.zeros(shape=(n_offsets,2)), disp_up))

        # Add a column to model vertical displacement from external tide model
        G = np.hstack((G, np.zeros(shape=(n_rows,1))))
        # Iterate over offsetfields
        #print("n_offsets: ",n_offsets)
        for i in range(n_offsets):

            # Find the observation vector (los, azi) refering to "create_grid_set"
            vecs = [offsetfields[i][2],offsetfields[i][3]]

            # Find the vertical displacement vector
            disp_up_vec = disp_up_vecs[i,:]

            # Two observation vectors
            for j in range(2):

                # Get the vector (represent E,N,U)
                vector = np.asarray(vecs[j])

                # Find the projection
                G[i*2+j,n_cols] = np.dot(vector, disp_up_vec)

        return G
        # End of modifying G.

    def model_vec_set_to_tide_vec_set(self, point_set, model_vec_set):
        tide_vec_set = {}

        for point in point_set:
            tide_vec_set[point] = self.model_vec_to_tide_vec(model_vec_set[point])

        return tide_vec_set

    def model_vec_to_tide_vec(self,model_vec):

        # Tides.
        tide_periods = self.tide_periods
        n_modeling_tides = self.n_modeling_tides
        modeling_tides = self.modeling_tides

        #num_params = 3 + n_modeling_tides*6
        #param_vec = np.zeros(shape=(num_params,1))
        
        # Copy the model_vec
        param_vec = np.copy(model_vec)

        # If model_vec is invalid.
        if np.isnan(model_vec[0,0]):
            param_vec = param_vec + np.nan
            return param_vec

        # Tides.
        for k in range(n_modeling_tides):
            
            tide_name = modeling_tides[k]

            # Model_vec terms: cosE, cosN, cosU, sinE, sinN, sinU.
            # Tide_vec terms:, ampE, ampN, ampU, phaseE, phaseN, phaseU
            
            # E N U
            for t in range(3):

                ### return value is in velocity domain m/d

                # cos term.
                coe1 = model_vec[3+k*6+t]

                # sin term.
                coe2 = model_vec[3+k*6+t+3]

                # omega
                omega = 2*np.pi / tide_periods[tide_name]

                approach = 2
                # Approach 1
                if approach==1:
                    # disp = a*coswt + b*sinwt
                    # velo = -aw * sinwt + bw*coswt
                    # sqrt(a**2+b**2)*w * sin(wt + phi)
                    # write it in the sin form, to be consistent with synthetic test
                    # tan phi = b/(-a)
    
                    # From displacement to velocity doamin
                    # For amplitide
                    amp = np.sqrt(coe1*coe1+coe2*coe2)*omega
    
                    # Phase.
                    phase = np.arctan2(coe2,-coe1)

                # Approach 2
                elif approach==2:
                    # disp = a*coswt + b*sinwt
                    # disp = amp * sin(wt + phi)
                    # disp_amp = np.sqrt(a**2+b**2)
                    # disp_phi = a/b

                    disp_amp = np.sqrt(coe1**2 + coe2**2)
                    disp_phase = np.arctan2(coe1,coe2)

                    amp = self.dis_amp_to_velo_amp(disp_amp,tide_name)
                    phase = self.dis_phase_to_velo_phase(disp_phase)
                    #print(disp_phase, phase)
                    #print(stop)
                    phase = self.wrapped(phase)

                else:
                    raise Exception()

                param_vec[3+k*6+t,0] = amp
                param_vec[3+k*6+t+3,0] = phase

        return param_vec

    # tide_vec to model_vec
    def tide_vec_set_to_model_vec_set(self, point_set, tide_vec_set):
        model_vec_set = {}

        for point in point_set:
            model_vec_set[point] = self.tide_vec_to_model_vec(tide_vec_set[point])

        return model_vec_set

    def tide_vec_to_model_vec(self, tide_vec):

        # Tides.
        tide_periods = self.tide_periods
        n_modeling_tides = self.n_modeling_tides
        modeling_tides = self.modeling_tides

        #num_params = 3 + n_modeling_tides*6
        #param_vec = np.zeros(shape=(num_params,1))
        
        param_vec = np.copy(tide_vec)

        # If tide_vec is invalid.
        if np.isnan(tide_vec[0,0]):
            param_vec = param_vec + np.nan
            return param_vec

        # Tides.
        for k in range(n_modeling_tides):
            
            tide_name = modeling_tides[k]

            # Model_vec terms: cosE, cosN, cosU, sinE, sinN, sinU. (displacement)
            # Tide_vec terms:, ampE, ampN, ampU, phaseE, phaseN, phaseU (velocity)
            
            # E N U
            for t in range(3):

                # velo = amp*sin(wt + phi)
                # velo = amp*sin(phi)*coswt + amp*cos(phi)*sinwt
                # disp = amp/w*sin(phi)*sinwt - amp/w*cos(phi)*coswt
                # cos term = -amp/w*cos(phi)
                # sin term = amp/w*sin(phi)

                # amp term.
                #print(tide_vec.shape, 3+k*6+t)
                amp = tide_vec[3+k*6+t]

                # phase term.
                phi = tide_vec[3+k*6+t+3]

                omega = 2*np.pi / tide_periods[tide_name]

                cos_term = -amp/omega * np.cos(phi)
                sin_term = amp/omega * np.sin(phi)

                param_vec[3+k*6+t,0] = cos_term
                param_vec[3+k*6+t+3,0] = sin_term
        
        return param_vec

    def model_posterior_to_uncertainty_set(self, point_set, tide_vec_set, Cm_p_set):

        tide_vec_uq_set = {}
        for point in point_set:
            tide_vec_uq_set[point] = self.model_posterior_to_uncertainty(
                                                tide_vec = tide_vec_set[point],
                                                Cm_p = Cm_p_set[point])

        return tide_vec_uq_set

    def model_posterior_to_uncertainty(self, tide_vec, Cm_p):

        tide_periods = self.tide_periods
        modeling_tides = self.modeling_tides
        n_modeling_tides = self.n_modeling_tides

        num_params = 3 + n_modeling_tides*6
        param_uq = np.zeros(shape=(num_params,1))


        # If Cm_p is invalid.
        if np.isnan(Cm_p[0,0]):
            param_uq = param_uq + np.nan
            return param_uq

        # Cm_p is valid, so is tide_vec

        # Secular velocity.
        variance = np.diag(Cm_p)
        param_uq[0:3,0] = variance[0:3]

        # Tide components.
        for k in range(n_modeling_tides):

            # E, N ,U
            for t in range(3):
        
                # cos term var.
                error_c_t = variance[3+k*6+t]

                # sin term var.
                error_s_t = variance[3+k*6+t+3]

                # Amplitude
                amp = tide_vec[3+k*6+t][0]

                # Phase
                phase = tide_vec[3+k*6+t+3][0]

                # Amplitude error
                amp_error = (error_c_t * np.sin(phase)**2 - error_s_t * np.cos(phase)**2) / (np.sin(phase)**4 - np.cos(phase)**4)

                # Phase error
                phase_error = (-error_c_t * np.cos(phase)**2 + error_s_t * np.sin(phase)**2) / (amp**2 * (np.sin(phase)**4 - np.cos(phase)**4))

                #if modeling_tides[k] == 'O1' and t==2:
                #    print(error_c_t)
                #    print(error_s_t)
                #    print(phase)
                #    print(amp_error)
                #    print(phase_error)

                #    print((error_c_t * np.sin(phase)**2 - error_s_t * np.cos(phase)**2) )
                #    print((np.sin(phase)**4 - np.cos(phase)**4))
                #    print(stop)

                param_uq[3+k*6+t,0] = amp_error
                param_uq[3+k*6+t+3,0] = phase_error

        # From variance to standard deviation (Important!).
        param_uq = np.sqrt(param_uq)
        return param_uq

    def simple_data_uncertainty_set(self, point_set, data_vec_set, noise_sigma_set):
        
        invCd_set = {}
        for point in point_set:
            invCd_set[point] = self.simple_data_uncertainty(data_vec_set[point], 
                                                            noise_sigma_set[point])
        return invCd_set

    def simple_data_uncertainty(self,data_vec, sigma):

        n_data = data_vec.shape[0]

        Cd = np.zeros(shape = (n_data,n_data))
        invCd = np.zeros(shape = (n_data,n_data))

        for i in range(n_data):
            Cd[i,i] = sigma**2
            invCd[i,i] = 1/(sigma**2)

        return invCd


    def real_data_uncertainty_set(self, point_set, data_vec_set, noise_sigma_set):
        
        invCd_set = {}
        for point in point_set:
            invCd_set[point] = self.real_data_uncertainty(data_vec_set[point], 
                                                            noise_sigma_set[point])
        return invCd_set

    def real_data_uncertainty(self,data_vec, sigma):

        # Range the azimuth are different.
        n_data = data_vec.shape[0]

        Cd = np.zeros(shape = (n_data,n_data))
        invCd = np.zeros(shape = (n_data,n_data))

        for i in range(n_data//2):
            # Range.
            Cd[2*i,2*i] = sigma[i][0]**2
            # Azimuth.
            Cd[2*i+1,2*i+1] = sigma[i][1]**2

            invCd[2*i,2*i] = 1/Cd[2*i,2*i]
            invCd[2*i+1,2*i+1] = 1/Cd[2*i+1,2*i+1]

        return invCd

    def model_prior_set(self, point_set):

        invCm_set = {}
        for point in point_set:
            invCm_set[point] = self.model_prior()

        return invCm_set

    def model_prior(self):

        n_modeling_tides = self.n_modeling_tides
        modeling_tides = self.modeling_tides

        # Find the number of parameters
        num_params = 3 + n_modeling_tides*6

        # For tides_3, add one param for up_disp scale
        if self.task_name == "tides_3":
            num_params +=1
        
        # Set the model priors
        # Sigmas of model parameters.
        inf_permiss = 0
        inf_restrict = 100000
 
        inv_sigma = np.zeros(shape=(num_params, num_params))

        horizontal = self.horizontal_prior

        # Secular velocity.
        if horizontal == True:
            # Remove upper component.
            inv_sigma[2,2] = inf_restrict

        if hasattr(self,'no_secular_up'):
            if self.no_secular_up == True:
                inv_sigma[2,2] = inf_restrict

        if hasattr(self,'up_short_period'):
            up_short_period = self.up_short_period

        else:
            up_short_period = False

        if hasattr(self,'horizontal_long_period'):
            horizontal_long_period = self.horizontal_long_period
        else:
            horizontal_long_period = False

        # Tides.
        for i in range(n_modeling_tides):
            tide_name = modeling_tides[i]
            #print(tide_name)
            for j in range(6):
                k = 3 + i*6 + j
               
                # only horizontal motion 
                if horizontal==True and (j==2 or j==5):
                    inv_sigma[k,k] = inf_restrict

                # Control the up on be only short period
                if not tide_name in self.tide_short_period_members and up_short_period and (j==2 or j==5):
                    inv_sigma[k,k] = inf_restrict
                    #print(tide_name, j)

                # Control the horizontal to be only long period
                if not tide_name in self.tide_long_period_members and horizontal_long_period and (j==0 or j==1 or j==3 or j==4):
                    inv_sigma[k,k] = inf_restrict
 
        invCm = np.square(inv_sigma)

        return invCm

    def model_posterior_set(self, point_set, linear_design_mat_set, data_prior_set, model_prior_set, test_point=None):

        Cm_p_set = {}
        for point in point_set:

            Cm_p_set[point] = self.model_posterior(linear_design_mat_set[point], 
                                                    data_prior_set[point], 
                                                    model_prior_set[point])
        return Cm_p_set

    def model_posterior(self, design_mat, data_prior, model_prior):

        G = design_mat
        invCd = data_prior
        invCm = model_prior

        # First of all: Consider invalid G
        if np.isnan(G[0,0]):
            Cm_p = np.zeros(shape=(1,1))+np.nan
            return Cm_p

        # OK, G is valid
        #print('G: ', G.shape)
        #print('invCd: ',invCd.shape)

        if G.shape[0] != invCd.shape[0]:
            raise Exception('G and invCd shapes are unmatched')

        invCm_p = np.matmul(np.matmul(np.transpose(G), invCd),G) + invCm

        # If G is singular.
        if np.linalg.cond(invCm_p) < 1/sys.float_info.epsilon:
            #print('normal')
        #if np.linalg.cond(invCm_p) < 10**8:
            Cm_p = np.linalg.pinv(invCm_p)
        else:
            #print('singular')
            Cm_p = np.zeros(shape=invCm_p.shape) + np.nan

        #print(stop)

        return Cm_p


    def model_posterior_analysis_set(self,point_set=None, Cm_p_set=None):

        others_set = {}
        for point in point_set:
            others_set[point] = []
            others_set[point].append(self.model_posterior_analysis(Cm_p = Cm_p[point]))

        return others_set

    def model_posterior_analysis(self,Cm_p):

        # noise sensitivity matrix
        #sensMat = np.linalg.pinv(np.matmul(np.transpose(G),G))
        #error_lumped = np.sqrt(max(np.trace(sensMat),0))

        error_lumped = np.mean(np.sqrt(max(np.trace(Cm_p))))
        out = error_lumped

        return quant

    def extract_up_scale_set(self, point_set, model_vec_set, others_set):

        for point in point_set:
            model_vec = model_vec_set[point]
            if np.isnan(model_vec[0,0]):
                others_set[point]["up_scale"] = np.nan
            else:
                others_set[point]["up_scale"] = self.extract_up_scale(point, model_vec)

        return 0

    def extract_up_scale(self, point, model_vec):

        # The first index after tides
        return model_vec[3 + len(self.modeling_tides) * 6, 0]

    def save_resid_set(self, point_set, resid_set, others_set, grounding_level):

        for point in point_set:
            if not 'grounding_level_resids' in others_set[point].keys():
                others_set[point]['grounding_level_resids'] = {}

            others_set[point]['grounding_level_resids'][grounding_level] = resid_set[point]

        return 0

    def select_optimal_grounding_level(self, point_set, others_set):

        for point in point_set:

            # if range_rmse is np.nan, optimal grounding level is np.nan
            others_set[point]["optimal_grounding_level"] = np.nan
            
            min_resid = float("inf")

            for grounding_level, resids in others_set[point]['grounding_level_resids'].items():

                # Do selection based on full rmse
                # range_rmse can np.nan
                range_rmse = resids[1]
                azimuth_rmse = resids[3]
                full_rmse = np.sqrt(range_rmse**2 + azimuth_rmse**2)

                if full_rmse < min_resid:
                    min_resid = full_rmse
                    others_set[point]["optimal_grounding_level"]  = grounding_level

        return 0

    # Simple version.
    def param_estimation_simple(self, design_mat, data):

        G = design_mat
        d = data

        invG = np.linalg.pinv(G)
        model_vec = np.matmul(invG, d)

        return model_vec

    # Bayesian inversion. (set)
    def param_estimation_set(self, point_set, linear_design_mat_set, data_vec_set,
                        data_prior_set, model_prior_set, model_posterior_set):

        model_vec_set = {}
        for point in point_set:
            model_vec_set[point] = self.param_estimation(linear_design_mat_set[point],
                                            data_vec_set[point], data_prior_set[point],
                                            model_prior_set[point], model_posterior_set[point])

        return model_vec_set

    # Bayesian version. 
    def param_estimation(self, design_mat, data, data_prior, model_prior, model_posterior=None):

        G = design_mat
        d = data
        
        invCd = data_prior
        invCm = model_prior
        Cm_p = model_posterior

        # If not provided, find Cm_p.
        if Cm_p is None:
            Cm_p = self.model_posterior(design_mat=G, data_prior=invCd, model_prior=invCm)

        # Check singularity.
        if np.isnan(Cm_p[0,0]):
            model_vec = np.zeros(shape=(1,1)) + np.nan
            return model_vec

        dd = np.matmul(np.matmul(np.transpose(G),invCd),d)

        model_p = np.matmul(Cm_p, dd)

        model_vec = model_p

        #print(model_vec)
        #print(model_vec.shape)

        return model_vec


    # Calculate residual sets.
    def resids_set(self, point_set, linear_design_mat_set, data_vec_set, model_vec_set):

        resid_of_secular_set = {}
        resid_of_tides_set = {}

        # Only keep the mean and standard deviation due to disk space.
        # There are four entries:
        # range mean, range rms, azimuth mean, azimuth rms

        for point in point_set:

            # secular.
            resid_of_secular = self.resid_of_secular(linear_design_mat_set[point],
                                                data_vec_set[point], model_vec_set[point])

            if not np.isnan(resid_of_secular[0,0]):
                resid_of_secular_set[point] = ( np.mean(resid_of_secular[0::2]),
                                                np.sqrt(np.mean(resid_of_secular[0::2]**2)),
                                                np.mean(resid_of_secular[1::2]),
                                                np.sqrt(np.mean(resid_of_secular[1::2]**2)))

                #print(resid_of_secular_set[point])

            else:
                resid_of_secular_set[point] = (np.nan,np.nan,np.nan,np.nan)

            # tides.
            resid_of_tides = self.resid_of_tides(linear_design_mat_set[point],
                                                data_vec_set[point], model_vec_set[point])

            if not np.isnan(resid_of_tides[0,0]):
                # range and azimuth
                resid_of_tides_set[point] = (   np.mean(resid_of_tides[0::2]),
                                                np.sqrt(np.mean(resid_of_tides[0::2]**2)),
                                                np.mean(resid_of_tides[1::2]),
                                                np.sqrt(np.mean(resid_of_tides[1::2]**2)))

                #print(resid_of_tides_set[point])

            else:
                resid_of_tides_set[point] = (np.nan,np.nan,np.nan,np.nan)

        return (resid_of_secular_set, resid_of_tides_set)

    # Residual of secular velocity.
    def resid_of_secular(self, design_mat, data, model):

        G = design_mat
        d = data
        m = model
        
        # Check singularity.
        if np.isnan(m[0,0]):
            resid_of_secular = np.zeros(shape=(1,1))+np.nan
            return resid_of_secular

        # Only keep secular velocity.
        m_secular = np.zeros(shape=m.shape)
        m_secular[0:2] = m[0:2]
       
        pred = np.matmul(G,m_secular)

        resid_of_secular = d - pred 

        #print("resid of secular")
        #print(np.hstack((d, pred, resid_of_secular)))
 
        return resid_of_secular

    # Residual including all tides.
    def resid_of_tides(self, design_mat, data, model):

        G = design_mat
        d = data
        m = model
        
        # Check singularity.
        if np.isnan(m[0,0]):
            resid_of_tides = np.zeros(shape=(1,1))+np.nan
            return resid_of_tides

        pred = np.matmul(G,m)
        resid_of_tides = d - pred

        return resid_of_tides

    def tide_vec_to_quantity(self, tide_vec, quant_name, point=None, state=None, extra_info=None):

        # modeling tides.
        modeling_tides = self.modeling_tides
        tide_periods = self.tide_periods
        tide_omegas = self.tide_omegas

        t_vec = tide_vec[:,0]

        # Output nan, if not exist.
        item_name = quant_name.split('_')[0]
        if (not item_name == 'secular') and (not item_name in modeling_tides):
            quant = np.nan
            return quant

        # Secular horizontal speed.
        if quant_name == 'secular_horizontal_speed':
            quant = np.sqrt(t_vec[0]**2 + t_vec[1]**2)

        # Secular east.
        elif quant_name == 'secular_east_velocity':
            quant = t_vec[0]

        # Secular north.
        elif quant_name == 'secular_north_velocity':
            quant = t_vec[1]

        # Secular horizontal speed.
        elif quant_name == 'secular_horizontal_velocity':
            
            # Degree.
            angle = np.rad2deg(np.arctan2(t_vec[1],t_vec[0]))
            # Length.
            speed = np.sqrt(t_vec[0]**2 + t_vec[1]**2)

            if not np.isnan(angle) and speed >=0.1:
                length = 0.2
                quant = (angle, length)

            else:
                quant = (0, 0)

            return quant

        elif quant_name == 'secular_horizontal_velocity_EN':

            return np.asarray([t_vec[0], t_vec[1]])
        
        # Secular up.
        elif quant_name == 'secular_up_velocity':
            quant = t_vec[2]

        # Msf horizontal amplitude (speed).        
        elif quant_name == 'Msf_horizontal_velocity_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = t_vec[3+k*6]
                    ampN = t_vec[3+k*6+1]
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Msf lumped horizontal displacement amplitude.
        elif quant_name == 'Msf_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Msf up displacement amplitude.
        elif quant_name == 'Msf_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = ampU
                else:
                    k=k+1

        # Msf Up phase. 
        elif quant_name.startswith('Msf_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            quant = self.rad2deg(phaseU)
                            quant = self.deg2day(quant, tide_name)
                            
                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)
                            quant = self.deg2day(quant, tide_name)

                        else:
                            raise Exception("Unknown state")
                    
                    else:
                        quant = np.nan
                else:
                    k=k+1

        elif quant_name == "Msf_horizontal_displacement_group":

            # Find the size of vertical tides
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

            # End of finding ampU

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6], tide_name)
                    phaseE = self.velo_phase_to_dis_phase(t_vec[3+k*6+3])

                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1], tide_name)
                    phaseN = self.velo_phase_to_dis_phase(t_vec[3+k*6+4])

                    # 2019.11.26: Calculate along flow and cross flow
                    # restore the amplitude representation
                    # The displacement is represented as amp*sin(omega*t+phase)
                    a = ampE * np.sin(phaseE) # amp of cos term
                    b = ampE * np.cos(phaseE) # amp of sin term
                    c = ampN * np.sin(phaseN) # amp of cos term
                    d = ampN * np.cos(phaseN) # amp os sin term

                    # Find ve and vn
                    ve = t_vec[0]
                    vn = t_vec[1]

                    # Along flow angle
                    # between -pi and pi
                    # rotation of coordinates; add minus before
                    theta1 = -np.arctan2(vn,ve)

                    # ALF Rotation
                    amp_cos_alf = a*np.cos(theta1) - c*np.sin(theta1)
                    amp_sin_alf = b*np.cos(theta1) - d*np.sin(theta1)

                    # ALF amplitude % phase
                    # e * coswt + f * sinwt
                    # sqrt(e**2 + f**2) * cos(wt+phi)
                    # artan(phi) = e/f
                    amp_alf = (amp_cos_alf**2 + amp_sin_alf**2)**(1/2)
                    phase_alf = np.arctan2(amp_cos_alf, amp_sin_alf)

                    # crf rotation
                    amp_cos_crf = a*np.sin(theta1) + c*np.cos(theta1)
                    amp_sin_crf = b*np.sin(theta1) + d*np.cos(theta1)

                    # crf amplitude
                    amp_crf = (amp_cos_crf**2 + amp_sin_crf**2)**(1/2)
                    phase_crf = np.arctan2(amp_cos_crf, amp_sin_crf)

                    # Convert unit of phase (use days for Msf)
                    phase_alf = self.rad2deg(phase_alf)
                    phase_alf = self.wrapped_deg(phase_alf)

                    # Record the degree value
                    phase_alf_in_deg = phase_alf

                    phase_alf = self.deg2day(phase_alf, tide_name)

                    phase_crf = self.rad2deg(phase_crf)
                    phase_crf = self.wrapped_deg(phase_crf)

                    # Record the degree value
                    phase_crf_in_deg = self.rad2deg(phase_crf)

                    phase_crf = self.deg2day(phase_crf, tide_name)

                    # Check if the value is valid not
                    ve_model = self.grid_set_velo[point][0]
                    vn_model = self.grid_set_velo[point][1]
                    v_model = (ve_model**2 + vn_model**2)**(1/2)

                    lon, lat = self.int5d_to_float(point)

                    thres_for_v = 0.4
                    thres_for_amp = 0.1

                    amp_full = (amp_alf**2 + amp_crf**2)**0.5

                    # Remove some invalid phase values
                    if v_model>thres_for_v and amp_alf>thres_for_amp:
                        if self.proj == "Rutford" and lat<-77.8:
                            pass

                        elif self.proj == "Evans" and lat<-75.85:
                        #elif self.proj == "Evans":
                            pass

                        else:
                            phase_alf = np.nan
                            phase_alf_in_deg = np.nan
                    else:
                        phase_alf = np.nan
                        phase_alf_in_deg = np.nan


                    if v_model > thres_for_v and amp_crf > thres_for_amp:
                        if self.proj == "Rutford" and lat<-77.8:
                            pass

                        #elif self.proj == "Evans" and ampU>0.5:
                        elif self.proj == "Evans":
                            pass

                        else:
                            phase_crf = np.nan
                            phase_crf_in_deg = np.nan
                    else:
                        phase_crf = np.nan
                        phase_crf_in_deg = np.nan

                    quant = {}
                    quant["Msf_along_flow_displacement_amplitude"] = amp_alf
                    quant["Msf_along_flow_displacement_phase"] = phase_alf
                    quant["Msf_along_flow_displacement_phase_in_deg"] = phase_alf_in_deg


                    quant["Msf_cross_flow_displacement_amplitude"] = amp_crf
                    quant["Msf_cross_flow_displacement_phase"] = phase_crf
                    quant["Msf_cross_flow_displacement_phase"] = phase_crf_in_deg

                    quant["Msf_horizontal_displacement_amplitude"] = amp_full 
                    return quant

                else:
                    k=k+1

        ############################################
        # Msf East amp.
        elif quant_name == 'Msf_east_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    quant = ampE
                else:
                    k=k+1

        # Msf East phase.
        elif quant_name == 'Msf_east_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    phaseE=self.velo_phase_to_dis_phase(t_vec[3+k*6+3])

                    quant = self.rad2deg(phaseE)
                    quant = self.wrapped_deg(quant)

                else:
                    k=k+1

        # Msf North amp.
        elif quant_name == 'Msf_north_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = ampN
                else:
                    k=k+1

        # Msf North phase.
        elif quant_name == 'Msf_north_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)
                    quant = self.rad2deg(phaseN)
                    quant = self.wrapped_deg(quant)

                else:
                    k=k+1

        ############### End of Msf ###############

        # Mf lumped horizontal displacement amplitude.
        elif quant_name == 'Mf_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Mf up displacement amplitude.
        elif quant_name == 'Mf_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = ampU
                else:
                    k=k+1

        # Mf Up phase. 
        elif quant_name.startswith('Mf_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            quant = self.rad2deg(phaseU)
                            quant = self.deg2day(quant, tide_name)
                            
                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)
                            quant = self.deg2day(quant, tide_name)

                        else:
                            raise Exception("Unknown state")
                    
                    else:
                        quant = np.nan
                else:
                    k=k+1

        ################## End of Mf ###########################################

        # M2 lumped horizontal displacement amplitude.
        elif quant_name == 'M2_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # O1 lumped horizontal displacement amplitude.
        elif quant_name == 'O1_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        ##################################################################3
        # Up component

        # O1 Up amp.
        elif quant_name == 'O1_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1
        
        # O1 Up phase. 
        # (only on ice shelves)
        elif quant_name.startswith('O1_up_displacement_phase'):
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    # value in velocity model > 0
                    #if (ampU > thres) or (state=='uq') :

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq') :
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            quant = self.rad2deg(phaseU)
                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant, tide_name)

                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)

                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant,tide_name)
                        else:
                            raise Exception("Unknown state")


                    # set to np.nan
                    else:
                        quant = np.nan
                else:
                    k=k+1

        # M2 Up amplitude.
        elif quant_name == 'M2_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

        # M2 Up phase. 
        # convert to minute
        elif quant_name.startswith('M2_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.3

                    #if ampU >=thres or state == 'uq':
                    if (self.grid_set_velo[point][2]>0 and ampU >= thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            #print("phase radian: ", phaseU)
                            quant = self.rad2deg(phaseU)
                            #print("phase degree: ", quant)
                            
                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant, tide_name)
                                #print("phaseU minute: ", phaseU)

                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)

                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant,tide_name)
                        else:
                            raise Exception("Unknown state")
                    
                    else:
                        quant = np.nan
                else:
                    k=k+1

        # N2 Up amplitude.
        elif quant_name == 'N2_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'N2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1
        
        # N2 Up phase. 
        # convert to minute
        elif quant_name.startswith('N2_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'N2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    if (self.grid_set_velo[point][2]>0 and ampU >= thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            #print("phase radian: ", phaseU)
                            quant = self.rad2deg(phaseU)
                            #print("phase degree: ", quant)
                            
                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant, tide_name)
                                #print("phaseU minute: ", phaseU)

                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)

                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant,tide_name)
                        else:
                            raise Exception("Unknown state")
                    else:
                        quant = np.nan
                else:
                    k=k+1

        # Q1 Up amplitude.
        elif quant_name == 'Q1_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Q1':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

        # Q1 Up phase. 
        # convert to minute
        elif quant_name.startswith('Q1_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Q1':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.03

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value)
                            #print("phase radian: ", phaseU)
                            quant = self.rad2deg(phaseU)
                            #print("phase degree: ", quant)
                            
                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant, tide_name)
                                #print("phaseU minute: ", phaseU)

                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)

                            if quant_name.endswith("in_deg"):
                                pass
                            else:
                                quant = self.deg2minute(quant,tide_name)
                        else:
                            raise Exception("Unknown state")
                    else:
                        quant = np.nan
                else:
                    k=k+1

        else:
            quant = None
            raise Exception(quant_name, ' is not defined yet!')

        return quant
 
def main():

    fourD = fourdvel()
    fourD.preparation()
   
if __name__=='__main__':
    main()
