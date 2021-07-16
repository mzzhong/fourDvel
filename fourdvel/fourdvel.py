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
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

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
            #self.csk_workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"
            self.csk_workdir = "/marmot-nobak/mzzhong/CSK-Evans-v3"

            # S1
            # Key is track, values are dates
            self.s1_data = {}

            self.s1_tracks = [37,52,169,65,7,50,64]
            self.s1_data_dir = "/net/kraken/nobak/mzzhong/S1-Evans"
            self.s1_workdir = "/net/kraken/nobak/mzzhong/S1-Evans-v2"
        
            for it in self.s1_tracks:
                self.s1_data[it] = []

            self.satellite_constants()

        elif self.proj == "Rutford":

            # CSK
            self.csk_data = {}

            self.csk_tracks = [8,10,23,25,40,52,55,67,69,82,97,99,114,126,128,129,141,143,156,158,171,172,173,186,188,201,203,215,218,230,231,232]

            #self.csk_workdir = "/net/kraken/nobak/mzzhong/CSK-Rutford"
            self.csk_workdir = "/net/kraken/nobak/mzzhong/CSK-Rutford-v2"

            for it in self.csk_tracks:
                self.csk_data[it] = []

            # S1
            self.s1_data = {}

            self.s1_tracks = [37,65,7]

            self.s1_workdir = "/net/kraken/nobak/mzzhong/S1-Evans-v2"
        
            for it in self.s1_tracks:
                self.s1_data[it] = []

            self.satellite_constants()


        # Get a map from track_num to track_ind
        # This is more convienet for analysis
        self.track_num_to_track_ind = {}
        self.track_ind_to_track_num = {}
        
        for i, track_num in enumerate(self.csk_tracks):
            self.track_num_to_track_ind[('csk', track_num)] = ('csk',i)
            self.track_ind_to_track_num[('csk',i)] = ('csk',track_num)

        for i, track_num in enumerate(self.s1_tracks):
            self.track_num_to_track_ind[('s1',track_num)] = ('s1', -i-1)
            self.track_ind_to_track_num[('s1',i)] = ('s1',track_num)

        #print(self.track_num_to_track_ind)
        #print(stop)

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

        # Intialize some parameters
        self.test_point = None
        self.single_point_mode = False
        self.simulation_use_external_up = False
        self.csk_excluded_tracks = []
        self.s1_excluded_tracks = []
        self.external_grounding_level_file = None
        self.simulation_mode = False

        self.csk_data_log = None
        self.csk_data_product_ids = None

        # error model
        self.data_error_mode = None
        self.data_uncert_grid_set_pklfile = None

        # up disp
        self.up_disp_mode = None

        # resid topo
        self.est_topo_resid = False

        # params for creating grid set
        self.min_num_of_csk_tracks = 1
        self.min_num_of_s1_tracks = 100

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

            if name == "test_point":
                if value.lower() != "none":

                    test_point_file, test_point_id = value.split('|')
                    test_point_file, test_point_id = test_point_file.strip(), test_point_id.strip()

                    assert os.path.exists(test_point_file), print("Test point file does not exist")

                    f_test_point = open(test_point_file,'r')
                    test_point_lines = f_test_point.readlines()
                    f_test_point.close()
                    
                    for test_point_line in test_point_lines:
                        try:    
                            line_id, line_value = test_point_line.split(':')
                            line_id, line_value = line_id.strip(), line_value.strip()
                        except:
                            continue
    
                        if line_id==test_point_id:
                            try:
                                the_point = [float(x) for x in line_value.split(",")]
                                self.test_point = self.float_lonlat_to_int5d(the_point)
                                print("Turn on single point mode")
                                self.single_point_mode = True
                            except:
                                continue
                    if self.test_point is None:
                        print("Cannot find test point ", test_point_id)
                        raise Exception()
                else:
                    self.test_point = None
                
                print("test_point", value, self.test_point, self.single_point_mode)

            if name == "inversion_method":

                self.inversion_method = value
                print("inversion_method", value)

            if name == "sampling_data_sigma":

                self.sampling_data_sigma = float(value)
                print("sampling_data_sigma", value)

            if name == 'resolution':
                self.resolution = int(value)
                print('resolution: ',value)

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

            if name == 'csk_data_log':
                self.csk_data_log = value
                print('csk_data_log: ',value)

            if name == 'csk_data_product_ids':
                if value.lower() != None:
                    assert os.path.exists(value), print("product id file missing")
                    f = open(value,'r')
                    line = f.readlines()[0]
                    self.csk_data_product_ids = [int(i) for i in line.split(',')]
                    print(self.csk_data_product_ids)
                else:
                    self.csk_data_product_ids = None

                print('csk_data_product_ids: ',value)
                print("Number of products: ", len(set(self.csk_data_product_ids)))

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


            # Params for creating grid set
            if name == 'min_num_of_csk_tracks':
                self.min_num_of_csk_tracks = int(value)
                print('min_num_of_csk_tracks: ',value)

            if name == 'min_num_of_s1_tracks':
                self.min_num_of_s1_tracks = int(value)
                print('min_num_of_s1_tracks: ',value)

            ## Error model ###
            # Simulation error model
            if name == 'csk_simulation_data_uncert_const':
                self.csk_simulation_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('csk_simulation_data_uncert_const: ',self.csk_simulation_data_uncert_const)

            if name == 's1_simulation_data_uncert_const':
                self.s1_simulation_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('s1_simulation_data_uncert_const: ',self.s1_simulation_data_uncert_const)

            # Data error model
            # error model mode
            if name == 'data_error_mode':
                self.data_error_mode = value
                print('data_error_mode', value)

            # const csk
            if name == 'csk_data_uncert_const':
                self.csk_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('csk_data_uncert_const: ',self.csk_data_uncert_const)

            # const s1
            if name == 's1_data_uncert_const':
                self.s1_data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('s1_data_uncert_const: ',self.s1_data_uncert_const)

            # external
            if name == 'data_uncert_grid_set_pklfile':
                self.data_uncert_grid_set_pklfile = value
                print('data_uncert_grid_set_pklfile', value)
            ##

            ## Modeling ##
            if name == 'up_disp_mode':
                self.up_disp_mode = value
                print('up_disp_mode', value)

            if name == 'up_disp_data_folder':
                self.up_disp_data_folder = value
                print('up_disp_data_folder', value)

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

            if name == 'simulation_model_num':
                self.simulation_model_num = int(value)
                print('simulation_model_num: ',value)

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

            ## Topo resid
            if name == 'est_topo_resid':
                if value == 'True':
                    self.est_topo_resid = True
                else:
                    self.est_topo_resid = False
                print('est_topo_resid: ', value)

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


        # Sanity check of parameters
        #assert self.data_error_mode is not None
        if self.data_error_mode is None:
            self.data_error_mode = 'const'

        print("Done with reading parameters")
        return 0

    def check_point_set_with_bbox(self, point_set, bbox):

        if bbox == "ice_shelf":
            if self.proj == "Rutford":
                bbox_s, bbox_n, bbox_e, bbox_w = None, self.float_to_int5d(-78.30), None, None
            elif self.proj == "Evans":
                bbox_s, bbox_n, bbox_e, bbox_w = None, self.float_to_int5d(-75.80), None, None
            else:
                raise Exception()

        else:
            bbox_s, bbox_n, bbox_e, bbox_w = bbox

        lons = []
        lats = []
        for point in point_set:
            lons.append(point[0])
            lats.append(point[1])

        lons = np.asarray(lons)
        lats = np.asarray(lats)

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
            
    def get_CSK_trackDates_from_log(self):
        import csv
        from CSK_Utils import CSK_Utils
        csk = CSK_Utils()

        # csk_data[track_number] = [date1, date2, date3,...]
        csk_data = self.csk_data

        csk_start = self.csk_start
        csk_end = self.csk_end
        
        data_file = self.csk_data_log

        min_cov = self.csk_evans_min_coverage()

        num_products = 0
        num_frames = 0
        with open(data_file) as dataset:
            csv_reader = csv.reader(dataset, delimiter=';')
            line = 0
            for row in csv_reader:
                line = line + 1
                if line == 1:
                    continue

                # Product ID
                product_id = int(row[0])
                # Satellite 
                sate = 'CSKS' + row[1][-1]
                # Date
                acq_datefmt = row[5].split(' ')[0]
                # Direction
                direction = row[7][0]
                # Coverage
                coverage = float(row[19][:4])
                # Time fraction
                t_frac_fmt = row[5].split(' ')[2]
                hour, minute, second = [int(s) for s in t_frac_fmt.split(':')]
                t_frac = (hour*3600 + minute*60 + second)/(24*3600)

                # coverage less 1%, this is not a planned acquisition
                if coverage<1:
                    continue

                print(acq_datefmt, t_frac_fmt, t_frac)

                # Convert date string to date object
                date_comp = [int(item) for item in acq_datefmt.split('-')]
                theDate = date(date_comp[0],date_comp[1],date_comp[2])

                # If the date is within the range set by user
                if theDate >= csk_start and theDate < csk_end:
                    # Find the figure out the track number.                
                    tracks = csk.date2track(day=theDate, sate=sate)[sate]
                   
                    if direction == 'A':
                        track = [ i for i in tracks if i<=10 ]
                    else:
                        track = [ i for i in tracks if i>=11 ]

                    print(sate, track, coverage)
                    assert len(track)==1, print("Fail to derive the track number")

                    if coverage < min_cov[track[0]]:
                        print("Bad acquisition")
                        continue

                    if self.csk_data_product_ids:
                        if not product_id in self.csk_data_product_ids:
                            continue

                    # Record it. 
                    csk_data[track[0]].append(theDate)
    
                    num_frames = num_frames + csk.numOfFrames[track[0]]
                    num_products += 1
    
        print("Number of products: ", num_products)
        print("Number of frames: ", num_frames)

        # Sort the dates of each track.
        # Output the dates for each track
        for track_num in sorted(csk_data.keys()):
            csk_data[track_num].sort()
            print("track number: ", track_num)
            print("Number of dates: ", len(csk_data[track_num]))

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
                    filefolder = self.csk_workdir + '/track_' + str(track_num).zfill(3) + '_0' + '/raw/201*'
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

                        ##############################
                        # For Evans project
                        # ad hoc, remove shortened data
                        # /net/kraken/bak/mzzhong/CSK-Evans/analyze_csk_data/analyze_existing_data.py

                        # track 11
                        if track_num == 11 and theDate >= date(2017,12,26) and theDate <= date(2018,3,4):
                            continue

                        # track 12
                        if track_num == 12:
                            if theDate >= date(2018,1,22) and theDate <= date(2018,3,3):
                                continue
                            #continue

                        # ad hoc, remove all track 12 data
                        #if track_num == 12:
                        # 01: 2019 and 2020 data
                        #if track_num == 12 and theDate <= date(2018,12,31):
                        # 02: 2018 data
                        #if track_num == 12 and (theDate <= date(2018,1,1) or theDate >= date(2018,12,31)):
                        # 03: 2017 data
                        #if track_num == 12 and theDate >= date(2017,12,31):
                        # 04: 2018, 2019, 2020 data
                        #if track_num == 12 and theDate <= date(2017,12,31):
                        #    continue

                        ################################
                        # For Rutford project
                        if theDate >= csk_start and theDate < csk_end:
                            csk_data[track_num].append(theDate)
    
                csk_data[track_num] = list(set(csk_data[track_num]))
                csk_data[track_num].sort()

                print("track_num: ",track_num,end=",  ")
                print("Number of dates: ",len(csk_data[track_num]))
                print("Dates: ", csk_data[track_num])

            #print(stop)

        elif csk_data_date_option == "log_based":
            self.get_CSK_trackDates_from_log()

        elif csk_data_date_option == "no_data":
            for track_num in tracklist:
                csk_data[track_num] = []

        else:
            print("csk_data_date_option", csk_data_option, "is not defined yet")
            raise Exception("dates information are not available")

        #print(stop)
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
            
                filefolder = self.s1_data_dir + '/data_' + str(track_num) + '/*zip'
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
        if self.proj == 'Rutford':
            self.grid_set_name= "_".join((grid_set_prefix, grid_set_datasets, grid_set_sources, grid_set_resolution))
        
        elif self.proj == 'Evans':
            #grid_set_cov_version = 'cov_v0'
            #grid_set_cov_version = 'cov_v1'
            #grid_set_cov_version = 'cov_v2'
            grid_set_cov_version = 'cov_v3'

            self.grid_set_name= "_".join((grid_set_prefix, grid_set_datasets, grid_set_sources, grid_set_resolution, grid_set_cov_version))

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

        fig = plt.figure(1, figsize=(8,8))
        ax = fig.add_subplot(111)
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
                self.tide_t_delta = 0.001
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
            idx = int(round((relative_time - self.tide_taxis[0])/self.tide_t_delta))
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

        if self.up_disp_mode is not None:
            up_disp_mode = self.up_disp_mode

        else:
            # Same as the input time series for vertical motion
            up_disp_mode = "single"
    
            # Use parameterized vertical motion (10 constituents)
            #up_disp_mode = "single_parametric"
    
            # Use self-derived vertical motion
            #up_disp_mode = "parametric_map"
            print("Please check the up disp mode: ", up_disp_mode)

        if up_disp_mode == "single":
            up_disp_set = {}
            for point in point_set:
                up_disp_set[point] = self.get_up_single_disp_for_point(point, offsetfields_set[point])

        elif up_disp_mode == "parametric_single":
            
            if self.proj == "Rutford":
                tide_cons, tide_params = self.read_ris_tides_params()
            else:
                raise ValueError()
            up_disp_set = {}
            for point in point_set:
                up_disp_set[point] = self.get_up_parametric_single_disp_for_point(point, offsetfields_set[point], tide_cons, tide_params)

        elif up_disp_mode == "parametric_map":
            up_disp_data_folder = self.up_disp_data_folder
            
            # load time axis
            pklfilename = 'taxis.pkl'
            with open(up_disp_data_folder + '/' + pklfilename, "rb") as f:
                taxis = pickle.load(f)
            #print(taxis, taxis[1] -taxis[0])
 
            up_disp_set = {}
            for point in point_set:
                check_result = 0
                # Compare the old and new up disp data
                if check_result:
                    (m0, s0) = self.get_up_single_disp_for_point(point, offsetfields_set[point])
                    (m1, s1) = self.get_up_varying_disp_for_point(point, up_disp_data_folder, taxis, offsetfields_set[point])

                    lon_int, lat_int = point
                    pklfilename = str(lon_int) + '_' + str(lat_int) + '.pkl'
                    filepath = up_disp_data_folder + '/' + pklfilename

                    if os.path.exists(filepath):
                        fig = plt.figure(1, figsize=(8,8))

                        with open(filepath, "rb") as f:
                            up_disp = pickle.load(f)

                        ax = fig.add_subplot(411)
                        ax.plot(m0)
                        ax = fig.add_subplot(412)
                        ax.plot(m1 - m0)
                        ax = fig.add_subplot(413)
                        ax.plot(taxis, self.tide_data)
                        ax = fig.add_subplot(414)
                        ax.plot(taxis, up_disp - self.tide_data)
                        fig.savefig("m0_m1.png")
                        print(stop)

                # get the up disp data
                up_disp_set[point] = self.get_up_varying_disp_for_point(point, up_disp_data_folder, taxis, offsetfields_set[point])

        else:
            raise ValueError()

        return up_disp_set

    def get_up_single_disp_for_point(self, point, offsetfields):

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

    def get_up_parametric_single_disp_for_point(self, point, offsetfields, tide_cons, tide_params):

        # timings
        master_timings = []
        slave_timings = []
        for i in range(len(offsetfields)):
            timing_a = (offsetfields[i][0], round(offsetfields[i][4],4))
            timing_b = (offsetfields[i][1], round(offsetfields[i][4],4))

            ta = (timing_a[0] - self.t_origin.date()).days + timing_a[1]
            tb = (timing_b[0] - self.t_origin.date()).days + timing_b[1]

            master_timings.append(ta)
            slave_timings.append(tb)
    
        master_timings = np.asarray(master_timings)
        slave_timings = np.asarray(slave_timings)

        # tide heights
        tide_height_master = 0
        tide_height_slave = 0
        for tide_name in tide_cons:
            amp = tide_params[(tide_name.lower(), "tide_amp")]
            phase = tide_params[(tide_name.lower(), "tide_phase")]
            omega = self.tide_omegas[tide_name]

            tide_height_master = tide_height_master + amp * np.sin(omega * master_timings + phase)
            tide_height_slave = tide_height_slave + amp * np.sin(omega * slave_timings + phase)

        tide_height_master = np.asarray(tide_height_master)
        tide_height_slave = np.asarray(tide_height_slave)

        #if point == self.test_point:
        #    print(master_timings)
        #    print(slave_timings)
        #    print(tide_height_master)
        #    print(tide_height_slave)
        #    print(stop)

        return (tide_height_master, tide_height_slave)

    def get_up_varying_disp_for_point(self, point, up_disp_data_folder, taxis, offsetfields):

        lon_int, lat_int = point
        pklfilename = str(lon_int) + '_' + str(lat_int) + '.pkl'
        filepath = up_disp_data_folder + '/' + pklfilename

        # The tide atomic data doesn't exist
        if not os.path.exists(filepath):
            up_disp = np.asarray(self.tide_data)

        # The tide atomic data exists
        else:
            with open(filepath, "rb") as f:
                up_disp = pickle.load(f)

        # Get timings for master and slave        
        ta_arr = []
        tb_arr = []
        for i in range(len(offsetfields)):
            # (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            ta = (offsetfields[i][0] - self.t_origin.date()).days + offsetfields[i][4]
            tb = (offsetfields[i][1] - self.t_origin.date()).days + offsetfields[i][4]

            ta_arr.append(ta)
            tb_arr.append(tb)

        ta_arr = np.asarray(ta_arr)
        tb_arr = np.asarray(tb_arr)

        # Important! Need to enforce 0.001 delta here. Using taxis[1]-taxis[0] will cause problem due to float precision 
        t_delta = 0.001
        # Assert if this t_delta is correct
        assert abs(taxis[1] - taxis[0] - 0.001) < 1e-3, print("t_delta is wrong")

        # The index of the value
        ta_inds = np.round((ta_arr - taxis[0])/t_delta).astype(np.int)
        tb_inds = np.round((tb_arr - taxis[0])/t_delta).astype(np.int)

        tide_height_master = up_disp[ta_inds]
        tide_height_slave = up_disp[tb_inds]

        # To save the memory
        del up_disp

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

                print("Get offset field stack: csk track ", track_num)

                # if data mode is pure synthetic data not relying on real data
                if self.csk_data_mode == 1:
                    print("csk data mode is 1, skip loading offset field stack")
                    continue

                # ad hoc, 2020.01.02
                special_case = False
                if self.proj == "Evans" and track_num == 12 and special_case:
                    track_offsetFieldStack_pkl = os.path.join(self.csk_workdir, "track_" + str(track_num).zfill(3) + '_0', "cuDenseOffsets", "_".join(filter(None, ["offsetFieldStack", str(self.csk_id), "v13"])) +  ".pkl")

                elif self.proj == "Evans":
                    track_offsetFieldStack_pkl = os.path.join(self.csk_workdir, "track_" + str(track_num).zfill(3) + '_0', "cuDenseOffsets", "_".join(filter(None, ["offsetFieldStack", str(self.csk_id), self.csk_version])) +  ".pkl")

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

                print("Get offset field stack: s1 track ", track_num)

                if self.s1_data_mode == 1:
                    print("s1 data mode is 1, skip loading offset field stack")
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

    def get_pair_baselines(self):

        pair_baselines_dict = {}

        if self.est_topo_resid and self.use_csk:
            pklfile = os.path.join(self.csk_workdir, 'all_pair_const_baseline.pkl')
            
            with open(pklfile,'rb') as f:
                pair_baselines_dict['csk'] = pickle.load(f)

        if self.est_topo_resid and self.use_s1:
            pklfile = os.path.join(self.s1_workdir, 'all_pair_const_baseline.pkl')
            with open(pklfile,'rb') as f:
                pair_baselines_dict['s1'] = pickle.load(f)

        self.pair_baselines_dict = pair_baselines_dict

        return 0

    def get_error_model(self):

        if self.data_uncert_grid_set_pklfile is not None:
            with open(self.data_uncert_grid_set_pklfile, 'rb') as f:
                self.data_uncert_grid_set = pickle.load(f)
        else:
            self.data_uncert_grid_set = None

        #print(self.data_uncert_grid_set)

    def preparation(self):

        # Get pre-defined grid points and the corresponding tracks and vectors.
        self.get_grid_set_v2()
        self.get_tile_set()

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

        # Find the tide height for all, if a single time seris is used
        self.get_timings_tide_heights()

        # Find all the design matrix
        self.get_design_mat_set()

        # Load offset field stack data
        self.get_offset_field_stack()

        # Load the pair baseline data
        self.get_pair_baselines()

        # Load error model
        self.get_error_model()

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

            #print("Available dates for this track: ", dates)

            # the offsetfields
            for d1 in dates:
                for d2 in dates:
                    if d1<d2 and (d2-d1).days<=max_delta:
                        #print(d1,d2)
                        offsetfields.append([d1,d2,vec1,vec2,t_frac])

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
        # End of building G.

    def modify_G_for_topo_resid_set(self, point_set, G_set, offsetfields_set, data_info_set, demfactor_set):

        for point in point_set:

            #if point != self.test_point:
            #    continue
            
            G = G_set[point]

            offsetfields = offsetfields_set[point]

            data_info = data_info_set[point]

            demfactor_dict = demfactor_set[point]

            demfactor_list = [demfactor_dict[data_info[i]] for i in range(len(data_info))]

            B_perp_list = []
            #B_perp_list_2 = []
            for i, offsetfield in enumerate(offsetfields):
                sate, track_num = data_info[i]
                date1, date2 = offsetfield[0:2]
                pairname = date1.strftime('%Y%m%d') + '_' + date2.strftime('%Y%m%d')

                B_perp_list.append(self.pair_baselines_dict[sate][sate, track_num, pairname])
                
                ## Find the B_perp data
                #if sate == 'csk':
                #    track_name = 'track_' + str(track_num).zfill(3) + '_0'
                #    B_perp_pklfile = os.path.join(self.csk_workdir, track_name, 'merged/interp_baselines', pairname+'.pkl')
                #    with open(B_perp_pklfile, "rb") as f:
                #        B_perp_data = pickle.load(f)
                #    
                #    B_perp_list_2.append(B_perp_data[0])

                #elif sate == 's1':
                #    track_name = 'track_' + str(track_num)
                #    B_perp_pklfile = os.path.join(self.s1_workdir, track_name, 'merged/interp_baselines', pairname+'.pkl')
                #    with open(B_perp_pklfile, "rb") as f:
                #        B_perp_data = pickle.load(f)
                #    
                #    B_perp_list_2.append(B_perp_data[0])

                #else:
                #    raise ValueError()
 
            #print('offsetfields: ', len(offsetfields))
            #print('data_info: ', len(data_info))
            #print('demfactor: ', len(demfactor_list))
            #print('B_perp: ', len(B_perp_list))
            #print(B_perp_list, B_perp_list_2)

            # modify G for topo resid
            G_set[point] = self.modify_G_for_topo_resid(point=point, G=G, offsetfields=offsetfields, demfactor_list=demfactor_list, B_perp_list=B_perp_list)

        return G_set

    def modify_G_for_topo_resid(self, point, G, offsetfields, demfactor_list, B_perp_list):

        # Control the number of offsetfields
        n_offsets = len(offsetfields)

        # Important: accounting for the no offsetfield scenario
        if n_offsets ==0:
            G = np.zeros(shape=(1,1)) + np.nan
            return G 

        ## Modify the G matrix
        # Find the shape of G
        n_rows, n_cols = G.shape
        #print("n_rows, n_cols: ", n_rows, n_cols)

        # Add a column to model topo resid
        G = np.hstack((G, np.zeros(shape=(n_rows,1))))

        # Iterate over offsetfields
        #print("n_offsets: ",n_offsets)
        for i in range(n_offsets):

            # Two observation vectors for each offsetfield
            # LOS and AZI
            for j in range(2):
                # Only for LOS
                if j%2==0:
                    phi = B_perp_list[i] * demfactor_list[i]
                    G[i*2+j,n_cols] = phi

        #print(n_offsets)
        #print(i)
        #print(G)
        #print(G.shape)
        #print(stop)

        return G
        # End of modifying G.

    def modify_G_set(self, point_set, G_set, offsetfields_set, up_disp_set, grounding_level, gl_name):

        # Extract the stack of up displacement from the tide model
        # up_disp_set = self.get_up_disp_set(point_set, offsetfields_set)

        for point in point_set:

            #if point == self.test_point:
            #    print(point)
            #    print(grounding_level[point])

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

            if gl_name == "external":
                # Try to get the value from external file
                try:
                    given_grounding_level = grounding_level[point]['optimal_grounding_level']
                    #print("optimal grounding level at this point is: ", given_grounding_level)
                except:
                    given_grounding_level = -10

                if np.isnan(given_grounding_level):
                    given_grounding_level = -10
            
            elif gl_name == "optimal":
                # Try to get the value from the obtained grounding level
                if point in grounding_level:
                    given_grounding_level_int = grounding_level[point]['optimal_grounding_level']
                    given_grounding_level = given_grounding_level_int / (10**6)

                    #if point == self.test_point:
                    #    print('given_grounding_level: ', given_grounding_level)

                    # 2021.04.10: Manually clip value large than zero
                    given_grounding_level = min(given_grounding_level, 0)

                    # 2021.04.12: Manually set the value to be -10
                    #given_grounding_level = -10
                    print("In optimal mode: optimal grounding level at this point is: ", point, given_grounding_level)
                    #print("AAA")

                # Not available, set it to -10
                else:
                    given_grounding_level = -10
                    #print("BBB")

                # If invalid, set it to be -10
                if np.isnan(given_grounding_level):
                    given_grounding_level = -10

            elif gl_name == "float":
                given_grounding_level = grounding_level

            elif gl_name == "auto":
                given_grounding_level = grounding_level[point]

            else:
                raise ValueError()

            # modify G
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
        
        # Copy the model_vec, same length
        # Secular, tidal, nonlinear param and topo resid are copied
        param_vec = np.copy(model_vec)

        # If model_vec is invalid.
        if np.isnan(model_vec[0,0]):
            param_vec = param_vec + np.nan
            return param_vec

        # Loop through the tides.
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

        num_params = len(tide_vec)
        num_tide_params = n_modeling_tides*6

        param_uq = np.zeros(shape=(num_params,1))

        # If Cm_p is invalid.
        if np.isnan(Cm_p[0,0]):
            # Set param_uq to be np.nan
            param_uq = param_uq + np.nan
            return param_uq

        # Cm_p is valid, so is tide_vec

        # Set secular params
        variance = np.diag(Cm_p)
        param_uq[0:3,0] = variance[0:3]
        # Set param_uq params 
        param_uq[3 + num_tide_params:, 0] = variance[3 + num_tide_params:]

        # Tide components. Do the conversion.
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

    #def simple_data_uncertainty_set(self, point_set, data_vec_set, noise_sigma_set):
    #    
    #    invCd_set = {}
    #    for point in point_set:
    #        invCd_set[point] = self.simple_data_uncertainty(data_vec_set[point], 
    #                                                        noise_sigma_set[point])
    #    return invCd_set

    #def simple_data_uncertainty(self,data_vec, sigma):

    #    n_data = data_vec.shape[0]

    #    Cd = np.zeros(shape = (n_data,n_data))
    #    invCd = np.zeros(shape = (n_data,n_data))

    #    for i in range(n_data):
    #        Cd[i,i] = sigma**2
    #        invCd[i,i] = 1/(sigma**2)

    #    return invCd

    def real_data_uncertainty_set(self, point_set, data_vec_set, noise_sigma_set):
        
        invCd_set = {}
        for point in point_set:
            invCd_set[point] = self.real_data_uncertainty(data_vec_set[point], noise_sigma_set[point])
            #print(np.diagonal(invCd_set[point]).tolist())
            #print(stop)

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

        # If est topo resid (the last one)
        if self.est_topo_resid:
            num_params +=1
        
        # Set the model priors
        # Sigmas of model parameters.
        inf_permiss = 0
        inf_restrict = 100000
        #inf_restrict = 10**8

        # Default is the value for permiss 
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
        assert G.shape[0] == invCd.shape[0], print(G.shape, invCd.shape, 'G and invCd shapes do not match')
        assert G.shape[1] == invCm.shape[0], print(G.shape, invCm.shape, 'G and invCm shapes do not match')

        invCm_p = np.matmul(np.matmul(np.transpose(G), invCd),G) + invCm

        # If G is singular.
        if np.linalg.cond(invCm_p) < 1/sys.float_info.epsilon:
            # This step can still have problem in rare case
            try:
                Cm_p = np.linalg.pinv(invCm_p)
            except:
                Cm_p = np.zeros(shape=invCm_p.shape) + np.nan
        else:
            Cm_p = np.zeros(shape=invCm_p.shape) + np.nan

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

    ## For nonlinear model, tides_2 ##
    def extract_grounding_up_scale_set(self, point_set, grounding_set, up_scale_set, others_set):

        for point in point_set:
            # true
            others_set[point]["true_up_scale"] = self.grid_set_velo[point][2]
            others_set[point]["true_optimal_grounding_level"] = self.simulation_grounding_level

            # estimated
            if point in grounding_set:
                others_set[point]["optimal_grounding_level"] = grounding_set[point]
            if point in up_scale_set:
                others_set[point]["up_scale"] = up_scale_set[point]

        return 0 

    def extract_up_scale_set(self, point_set, model_vec_set, others_set):

        for point in point_set:
            model_vec = model_vec_set[point]
            # true
            others_set[point]["true_up_scale"] = self.grid_set_velo[point][2]

            # estimated
            if np.isnan(model_vec[0,0]):
                others_set[point]["up_scale"] = np.nan
            else:
                others_set[point]["up_scale"] = self.extract_up_scale(model_vec)

        return 0

    def extract_up_scale(self, model_vec):
        # up_scale is at the first index after secular and tidal params
        if not np.isnan(model_vec[0,0]):
            return model_vec[3 + len(self.modeling_tides) * 6, 0]
        else:
            return np.nan

    ## For linear model, tides_3 ##
    def export_to_others_set_wrt_gl(self, point_set, grounding_level, model_vec_set, model_likelihood_set, resid_set, others_set):

        if isinstance(grounding_level, int):
            mode = 'const'
        elif isinstance(grounding_level, dict):
            mode = 'point'
        else:
            raise ValueError()

        for point in point_set:
            if mode == 'const':
                grounding_level_point = grounding_level
            elif mode == 'point':
                grounding_level_point = grounding_level[point]
            else:
                raise ValueError()

            # up scale
            if not 'grounding_level_up_scale' in others_set[point].keys():
                others_set[point]['grounding_level_up_scale'] = {}

            others_set[point]['grounding_level_up_scale'][grounding_level_point] = self.extract_up_scale(model_vec_set[point])

            # model likelihood
            if not 'grounding_level_model_likelihood' in others_set[point].keys():
                others_set[point]['grounding_level_model_likelihood'] = {}

            others_set[point]['grounding_level_model_likelihood'][grounding_level_point] = model_likelihood_set[point]

            # residual
            if not 'grounding_level_resids' in others_set[point].keys():
                others_set[point]['grounding_level_resids'] = {}

            others_set[point]['grounding_level_resids'][grounding_level_point] = resid_set[point]

        return 0

    def calc_hpdi(self, x, y, alpha=0.9):

        if np.isnan(y[0]):
            return (np.nan, np.nan)

        if len(x)<2:
            return (np.nan, np.nan)
        
        l = len(x)
        shortest_l = l 
        
        # total mass
        total_mass = np.nansum(y)
        max_mass = 0 
        for i in range(l-1):
            for j in range(i+1,l):
                # calculate mass
                mass = np.nansum(y[i:j+1])
                # print(mass, total_mass)
                if mass/total_mass >= alpha:
                    if (j-i+1<shortest_l) or ((j-i+1==shortest_l) and mass > max_mass):
                        shortest_l = j-i+1
                        max_mass = mass
                        best_i = i 
                        best_j = j
        return (x[best_i], x[best_j])

        #try:
        #    return (x[best_i], x[best_j])
        #except:
        #    print("Unbounded error")
        #    print(x)
        #    print(y)
        #    return (np.nan, np.nan)

    def calc_hpdi_v2(self, x, y, alpha=None):

        if np.isnan(y[0]):
            return (np.nan, np.nan)

        if len(x)<2:
            return (np.nan, np.nan)

        l = len(x)

        if alpha is None:
            if self.proj == 'Rutford':
                alpha = 0.95
            elif self.proj == 'Evans':
                alpha = 0.68
                alpha = 0.3
            else:
                raise ValueError()

        # Set nan to zero
        y[np.isnan(y)] = 0

        max_ind = np.argmax(y)
        left = max_ind
        right = max_ind
        
        total_mass = np.nansum(y)
        
        mass = y[max_ind]

        #print("max_ind: ", max_ind, x[max_ind], y[max_ind])
        #print(y)

        # Expand from the max point
        while mass < total_mass * alpha:
            #print(left, right, mass)
        
            if left > 0:
                left_candi = y[left - 1]
            else:
                left_candi = None
        
            if right < l-1:
                right_candi = y[right+1]
            else:
                right_candi = None
        
            # Both are at the end, which should not be possible
            if left_candi is not None and right_candi is not None:
                if left_candi > right_candi:
                    left -= 1
                    mass += y[left]
                else:
                    right += 1
                    mass += y[right]
        
            elif left_candi is not None:
                left -= 1
                mass += y[left]
        
            elif right_candi is not None:
                right += 1
                mass += y[right]
        
            else:
                raise Exception("hpdi_error")

        return (x[left], x[right])

    def select_optimal_grounding_level(self, point_set, grid_set_velo, others_set):

        select_mode = "likelihood"
 
        for point in point_set:

            #if point != self.test_point:
            #    continue

            velo_model = grid_set_velo[point]

            # Set true value
            others_set[point]["true_optimal_grounding_level"] = self.simulation_grounding_level
            others_set[point]["true_up_scale"] = velo_model[2]

            # Set estimation value
            # default value is nan
            # e.g. if range_rmse is np.nan, optimal grounding level is np.nan
            others_set[point]["optimal_grounding_level"] = np.nan
            others_set[point]["up_scale"] = np.nan

            # resid based (deprecated) 
            if select_mode == "resid":
                min_value = float("inf")

                for grounding_level_int, resids in sorted(others_set[point]['grounding_level_resids'].items()):

                    # Do selection based on the full root mean squre error
                    # range_rmse can be np.nan (np.nan < number is False)
                    range_rmse = resids[1]
                    azimuth_rmse = resids[3]
                    full_rmse = np.sqrt(range_rmse**2 + azimuth_rmse**2)

                    if full_rmse < min_value:
                        min_value = full_rmse
                        others_set[point]["optimal_grounding_level"]  = grounding_level_int

                        # Need to add up_scale here (TODO)
            
            # likelihood based
            elif select_mode == "likelihood":
                min_value = float("inf")

                grounding_levels = []
                likelihoods = []

                # go from small to large gl
                for grounding_level_int, likelihood in sorted(others_set[point]['grounding_level_model_likelihood'].items()):
                    #print(grounding_level_int, likelihood)

                    if likelihood < min_value:
                        min_value = likelihood

                        # Save the optimal grounding level
                        others_set[point]["optimal_grounding_level"]  = grounding_level_int

                        # Save the corresponding up_scale
                        others_set[point]["up_scale"] = others_set[point]['grounding_level_up_scale'][grounding_level_int]

                    grounding_levels.append(grounding_level_int)
                    likelihoods.append(likelihood)

                grounding_levels = np.asarray(grounding_levels)
                likelihoods = np.asarray(likelihoods)

                # calculate credible interval
                if len(grounding_levels)>=2:
                    # Need to consider the case if the enumeration is not even
                    # Do interpolation on [-4,0] spacing = 0.01
                    interp_fun = interp1d(grounding_levels, likelihoods, kind='linear')
   
                    # interpolation every 1cm (10**(-2))
                    gls_interp = np.arange(min(grounding_levels),max(grounding_levels)+1e-6, 10**(-2) * 10**6)
                    likelihoods_interp = interp_fun(gls_interp)
    
                    ## Get the probability ##
                    # normalize the likelihood in log space
                    # Sum up the denominator
                    likelihoods_interp = likelihoods_interp - np.nanmin(likelihoods_interp)
                    prob_sum = np.nansum(np.exp(-likelihoods_interp))
                    gl_probs_interp = np.exp(-likelihoods_interp) / prob_sum
    
                    # Calculate credible interval
                    #if point == self.test_point:
                    #    start_time = time.time()
                    #    gl_ci = self.calc_hpdi(gls_interp/10**6, gl_probs_interp)
                    #    elapsed_time = time.time() - start_time
                    #    print("Elapased time: ", elapsed_time)
                    #    print(gl_ci)
                    #    print(stop)
                    #    start_time = time.time()
                    #    gl_ci_2 = self.calc_hpdi_v2(gls_interp/10**6, gl_probs_interp)
                    #    elapsed_time = time.time() - start_time
                    #    print("Elapased time: ", elapsed_time)
                    #    print(gl_ci_2)

                    #print(gls_interp / 10**6)
                    #print(gl_probs_interp) 
                    gl_ci_2 = self.calc_hpdi_v2(gls_interp/10**6, gl_probs_interp)

                else:
                    gls_interp = grounding_levels
                    gl_probs_interp = np.ones(shape=(1,))
                    gl_ci_2 = (np.nan, np.nan)
                
                others_set[point]["grounding_level_credible_interval"] = gl_ci_2

                # For test point, save and show the result
                if point == self.test_point:
                    show_test_point = False
                    if show_test_point:
                        print('optimal gl: ', others_set[point]["optimal_grounding_level"])
                        #print('interpolating on : ')
                        #print(grounding_levels)
                        #print(likelihoods)

                        print('credible interval: ', others_set[point]["grounding_level_credible_interval"])
                        print('enumerate grounding level: ', grounding_levels)

                        # Plot the probability
                        fig = plt.figure(200, figsize=(7,7))
                        ax = fig.add_subplot(211)
                        ax.plot(grounding_levels, likelihoods, 'k')
                        ax.plot(gls_interp, likelihoods_interp, 'r')
                        ax = fig.add_subplot(212)
                        ax.plot(gls_interp, gl_probs_interp)
                        fig.savefig('200.png')
                        print(stop)

                    # Save the probability 
                    others_set[point]["grounding_level_prob"] = {}
                    for i in range(len(gl_probs_interp)):
                        others_set[point]["grounding_level_prob"][gls_interp[i]] = gl_probs_interp[i]

            else:
                raise ValueError("unknown select mode")

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

        return model_vec

    # Calculate residual sets.
    def get_resid_set(self, point_set, linear_design_mat_set, data_vec_set, model_vec_set):

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

    def point_set_residual_analysis(self, point_set, data_info_set, offsetfields_set, data_vec_set, linear_design_mat_set, model_vec_set):

        test_point = self.test_point
        grid_set = self.grid_set

        # Perform estimation on each point
        residual_analysis_set = {}

        for point in point_set:

            # check if it is single point mode
            #if self.single_point_mode and point!=test_point:
            #    continue

            # Find data and model for the test point
            data_info = data_info_set[point]
            data_vec = data_vec_set[point]
            offsetfields = offsetfields_set[point]
            design_mat = linear_design_mat_set[point]

            # Find the estimation
            model_vec = model_vec_set[point]

            # Valid estimation exists
            if not np.isnan(model_vec[0,0]):

                # prediction 
                data_vec_pred = np.matmul(design_mat, model_vec)

                # residual
                data_vec_residual = data_vec - data_vec_pred
                
                # Find range residual
                data_vec_residual_range = data_vec_residual[0::2]

                # Find azimuth residual
                data_vec_residual_azimuth = data_vec_residual[1::2]

                residual_analysis_point_result = self.point_residual_analysis(point, data_info, offsetfields, data_vec, data_vec_pred, data_vec_residual)

                # Save this point
                residual_analysis_set[point] = residual_analysis_point_result

            else:

                residual_analysis_set[point] = None


        return residual_analysis_set

    def point_residual_analysis(self, point, data_info, offsetfields, data_vec, data_vec_pred, data_vec_residual):

        #print("Work on point: ", point)

        # Partition the residual according to track, center date and time interval
        # Give coords to each residual

        # To save the coordinates for this point
        coords_list = []

        data_num_list = []
        track_name_list = []
        data_num_total = 0

        tide_proxy_list = []

        # Find the tidal height data
        tide_taxis = self.tide_taxis
        tide_taxis_delta = self.tide_t_delta
        tide_data = self.tide_data

        # The residual of each track at this point
        track_residual = {}
        
        # Look through by track
        data_info_summary = self.summarize_data_info(data_info)
        for i, track in enumerate(data_info_summary):
            # check if it is single point mode
            #if self.single_point_mode and point!=self.test_point:
            #    continue

            track_name, data_num = track
            sate_name = track_name[0]
            track_num = track_name[1]

            #print("track_name, data_num: ", track_name, data_num)

            # There is no available measurement in this track
            #if data_num == 0:
            #    continue

            data_num_list.append(data_num)
            track_name_list.append(track_name)
            
            # Obtain the data of this track
            data_vec_track = data_vec[ data_num_total*2 : (data_num_total + data_num)*2 ]
            data_vec_track_range = data_vec_track[0::2, 0]
            data_vec_track_azimuth = data_vec_track[1::2, 0]

            # Obtain the data_pred of this track
            data_vec_pred_track = data_vec_pred[ data_num_total * 2 : (data_num_total + data_num)*2 ]
            # range & azimuth
            data_vec_pred_track_range = data_vec_pred_track[0::2, 0]
            data_vec_pred_track_azimuth = data_vec_pred_track[1::2, 0]

            # Obtain the offsetfields of this track            
            offsetfields_track = offsetfields[ data_num_total: data_num_total + data_num ]

            # Obtain the residual of this track
            data_vec_residual_track = data_vec_residual[ data_num_total*2 : (data_num_total + data_num)*2 ]
            # range & azimuth
            range_residual_track = data_vec_residual_track[::2,0]
            azimuth_residual_track = data_vec_residual_track[1::2,0]

            # Save the residual
            track_residual[((sate_name, track_num), 'range')] = np.nanstd(range_residual_track)
            track_residual[((sate_name, track_num), 'azimuth')] = np.nanstd(azimuth_residual_track)

            # Move to next track
            data_num_total += data_num

            ############ Analyze the each residual point #########
            # Find the index of the track
            _, track_ind = self.track_num_to_track_ind[(sate_name, track_num)]

            # Record the relevant information of each offset field
            for j, offsetfield in enumerate(offsetfields_track):
                t1_day = self.count_days(offsetfield[0])
                t2_day = self.count_days(offsetfield[1])
                t_center = (t1_day + t2_day)/2
                t_center_date = self.t_origin + datetime.timedelta(days=t_center)
                t_center_datestr = t_center_date.strftime('%Y%m%d')
                delta_t = t2_day - t1_day

                coords_list.append( ((sate_name, track_num, track_ind), (t_center, t_center_date, t_center_datestr), delta_t) )

                # Record tide
                # Fractional time
                t1 = t1_day + offsetfield[4]
                t2 = t2_day + offsetfield[4]

                # The tidal height
                z1 = tide_data[int(np.round((t1 - tide_taxis[0])/tide_taxis_delta))]
                z2 = tide_data[int(np.round((t2 - tide_taxis[0])/tide_taxis_delta))]

                # Record the sampled tide by master and slave
                tide_proxy_list.append((z1, z2, track_num, offsetfield[0], offsetfield[1]))

        # Check if the coords are correct: 2 x length of coords == length of residual
        assert(len(coords_list)*2 == len(data_vec_residual)), \
        print("coords_list length {} and data_vec_residual length {} don't match".format(len(coords_list), len(data_vec_residual)))

        # Plot and make analysis
        plot_analysis = False

        if plot_analysis == True and self.single_point_mode == False:
            raise Exception("Cannot plot analysis when single point mode is turned on")

        if plot_analysis:
            if self.proj == 'Rutford':
                analysis_start_day = datetime.datetime(2013,6,1)
                analysis_stop_day = datetime.datetime(2014,10,1)

            elif self.proj == 'Evans':
                analysis_start_day = datetime.datetime(2017,1,1)
                analysis_stop_day = datetime.datetime(2021,6,1)
            else:
                raise ValueError()

            range_residual = data_vec_residual[::2,0]
            azimuth_residual = data_vec_residual[1::2,0]
   
            # Set up the figure 
            fig = plt.figure(1, figsize=(14,11))
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
    
            for i in range(len(coords_list)):
                track_tuple, t_tuple, delta_t = coords_list[i]
                sate_name, track_num, track_ind = track_tuple
                t_center, t_center_date, t_center_datestr = t_tuple
    
                # Get tide height for master and slave
                z1, z2 = tide_proxy_list[i][0:2]

                tide_signature = str(z1) + ',' + str(z2)
                
                # ax1
                #print(track_ind)
                ax1.plot(track_ind, range_residual[i],'r.')
                ax1.plot(track_ind, azimuth_residual[i],'b.')

                if abs(azimuth_residual[i])>0.35:
                    ax1.text(track_ind, azimuth_residual[i], str(t_center_datestr)+' '+str(delta_t) + ' '+tide_signature)

                if abs(range_residual[i])>0.35:
                    ax1.text(track_ind, range_residual[i], str(t_center_datestr)+' '+str(delta_t) + ' '+tide_signature)
    
                #_, track_num = self.track_ind_to_track_num[(sate_name, track_ind)]
                ax1.text(track_ind, 0, str(track_num))
    
                # ax2
                #print(t_center_date)
                #t_center_date = self.t_origin + datetime.timedelta(days=t_center)
                #print(t_center_date, self.t_origin)

                x_off = self.count_days(t_center_date) - self.count_days(analysis_start_day)
                ax2.plot(x_off, range_residual[i],'r.')
                ax2.plot(x_off, azimuth_residual[i],'b.')
                #ax2.text(t_center,0, t_center_datestr)
                # ax3
                #ax3.plot(delta_t, range_residual[i],'r.')
                #ax3.plot(delta_t, azimuth_residual[i],'b.')

            # Configure ax1 axis
            ax1.set_xlabel('Tracks')
            ax1.set_ylabel('Residual (m)')
            ax1.set_title('Residuals by track')
 
            # Configure ax2 axis
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Residual (m)')
            ax2.set_title('Red: range; Blue: azimuth')

            xticks = np.arange(0, self.count_days(analysis_stop_day) - self.count_days(analysis_start_day), 150)
            ax2.set_xticks(xticks)

            xticklabels = [(analysis_start_day + datetime.timedelta(days=int(xtick))).strftime('%Y-%m-%d') for xtick in xticks ]
            ax2.set_xticklabels(xticklabels)
 
            fig.savefig("residual.png")
            plt.show()

        residual_analysis_point_result = track_residual

        return residual_analysis_point_result

    def get_model_likelihood_set(self, point_set, linear_design_mat_set, data_vec_set, model_vec_set, invCd_set):

        model_likelihood_set = {}
        for point in point_set:
            G = linear_design_mat_set[point]
            d = data_vec_set[point]
            m = model_vec_set[point]
            invCd = invCd_set[point]

            if np.isnan(m[0,0]):
                model_likelihood_set[point] = np.nan
            else:
                # calculate model likelihood which is "posterior prob = *  exp(-model_likelihood)"
                model_likelihood = 0.5 * (d - G @ m).T @ invCd @ (d - G @ m)
                model_likelihood_set[point] = model_likelihood[0,0]
       
        return model_likelihood_set

    def tide_vec_to_quantity(self, input_tide_vec, quant_name, point=None, state=None, extra_info=None):

        # modeling tides.
        modeling_tides = self.modeling_tides
        tide_periods = self.tide_periods
        tide_omegas = self.tide_omegas

        # for uq, both est and uq are passed in a tuple
        # The main vec is data_vec
        if state == 'uq':
            # input_tide_vec is tuple
            vec1, vec2 = input_tide_vec

            # reduce the dimension
            estimation_vec = vec1[:,0]
            data_vec = vec2[:,0]
        else:
            # reduce the dimension
            data_vec = input_tide_vec[:,0]

        # Output nan, if not exist.
        extra_list = ['up_amplitude_scaling', 'topo_resid']
        item_name = quant_name.split('_')[0]
        if (not item_name == 'secular') and (not item_name in modeling_tides) and (not quant_name in extra_list):
            quant = np.nan
            return quant

        # Secular horizontal speed.
        if quant_name == 'secular_horizontal_speed':
            quant = np.sqrt(data_vec[0]**2 + data_vec[1]**2)

        # Secular east.
        elif quant_name == 'secular_east_velocity':
            quant = data_vec[0]

        # Secular north.
        elif quant_name == 'secular_north_velocity':
            quant = data_vec[1]

        # Secular horizontal speed.
        elif quant_name == 'secular_horizontal_velocity':
            # Degree.
            angle = np.rad2deg(np.arctan2(data_vec[1],data_vec[0]))
            # Length.
            speed = np.sqrt(data_vec[0]**2 + data_vec[1]**2)

            if not np.isnan(angle) and speed >=0.1:
                length = 0.2
                quant = (angle, length)

            else:
                quant = (0, 0)

            return quant

        elif quant_name == 'secular_horizontal_velocity_EN':

            return np.asarray([data_vec[0], data_vec[1]])
        
        # Secular up.
        elif quant_name == 'secular_up_velocity':
            quant = data_vec[2]

        ################### Msf #################################
        # Msf horizontal amplitude (speed).        
        elif quant_name == 'Msf_horizontal_velocity_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = data_vec[3+k*6]
                    ampN = data_vec[3+k*6+1]
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Msf lumped horizontal displacement amplitude.
        elif quant_name == 'Msf_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(data_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(data_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Msf up displacement amplitude.
        elif quant_name == 'Msf_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = ampU
                else:
                    k=k+1

        # Msf Up phase. 
        elif quant_name.startswith('Msf_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):

                        value = data_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU = self.velo_phase_to_dis_phase(value)
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
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

            # End of finding ampU

            k = 0
            for tide_name in modeling_tides:
                # k +=1 is tide_name is not "Msf"
                if tide_name == 'Msf':

                    if state in ['true','est']:
                        est_vec = data_vec
                    elif state in ['uq']:
                        # state is uq, data_vec stores uq
                        # estimation is passed in additonally by estimation_vec
                        est_vec = estimation_vec 
                    else:
                        raise ValueError()
 
                    ampE = self.velo_amp_to_dis_amp(est_vec[3+k*6], tide_name)
                    phaseE = self.velo_phase_to_dis_phase(est_vec[3+k*6+3])

                    ampN = self.velo_amp_to_dis_amp(est_vec[3+k*6+1], tide_name)
                    phaseN = self.velo_phase_to_dis_phase(est_vec[3+k*6+4])

                    ################## Notes ####################
                    # Calculate along flow and cross flow
                    # restore the amplitude representation
                    # The displacement is represented as amp*sin(omega*t+phase)

                    # rotation matrix
                    # [ cos(theta), -sin(theta),
                    #   sin(theta), cos(theta)]
                    # this rotation matrix rotatex a point counterclock wise for theta

                    # In this application, we want to find the coordinates in rotated coordinate system
                    # The rotation of the coordinate system is theta = arctan2(vn/ve)
                    # This is equivalent to rotate a point counterclock-wise for theta1 = -theta
                    
                    # The calculation is:
                    #   (a*coswt + b*sinwt)*cos(theta1) - (c*coswt + d*sinwt)*sin(theta1)
                    # = (a*cos(theta1) - c*sin(theta1))coswt +  (b*cos(theta1) - d*sin(theta1))*sinwt
                    # = alf_cos_term * coswt + alf_sin_term * sinwt

                    # Define:
                    # cos term: e = a*cos(theta1) - c*sin(theta1)
                    # sin term: f = b*cos(theta1) - d*sin(theta1)

                    # Here, we derive the sigma_e and sigma_f
                    # sigma_e^2 = cos(theta1)^2 * sigma_a^2 + sin(theta1)^2 * simga_c^2
                    # sigma_f^2 = cos(theta1)^2 * sigma_b^2 + sin(theta1)^2 * simga_d^2

                    # model_vec: cosE(a), cosN(c), cosU, sinE(b), sinN(d), sinU
                   
                    a = ampE * np.sin(phaseE) # amp of cos term
                    b = ampE * np.cos(phaseE) # amp of sin term
                    c = ampN * np.sin(phaseN) # amp of cos term
                    d = ampN * np.cos(phaseN) # amp of sin term

                    # Find ve and vn, the secular velocity
                    ve = est_vec[0]
                    vn = est_vec[1]

                    # Along flow angle
                    # between -pi and pi
                    # theta1 is the rotation angle of coordinates; add minus before
                    theta1 = -np.arctan2(vn,ve)

                    # ALF Rotation (based on the notes above)
                    amp_cos_alf = a*np.cos(theta1) - c*np.sin(theta1)
                    amp_sin_alf = b*np.cos(theta1) - d*np.sin(theta1)

                    # ALF amplitude & phase
                    # e * coswt + f * sinwt
                    # sqrt(e**2 + f**2) * sin(wt+phi)
                    # cos(phi) = f
                    # sin(phi) = e
                    # tan(phi) = e/f
                    amp_alf = (amp_cos_alf**2 + amp_sin_alf**2)**(1/2)
                    phase_alf = np.arctan2(amp_cos_alf, amp_sin_alf)

                    # CRF amplitude & phase
                    # theta2 = theta1 - pi/2
                    # cos(theta2) = cos(theta1 - pi/2) =  sin(theta1)
                    # sin(theta2) = sin(theta1 - pi/2) = -cos(theta1)

                    #   (a*coswt + b*sinwt)*cos(theta2) - (c*coswt + d*sinwt)*sin(theta2)
                    # = (a*cos(theta2) - c*sin(theta2))coswt +  (b*cos(theta2) - d*sin(theta2))*sinwt
                    # = (a*sin(theta1) + c*cos(theta1))coswt +  (b*sin(theta1) + d*cos(theta1)) * sinwt
                    # = crf_cos_term * coswt + crf_sin_term * sinwt

                    # cos and sin term after rotation (based on the notes above)
                    amp_cos_crf = a*np.sin(theta1) + c*np.cos(theta1)
                    amp_sin_crf = b*np.sin(theta1) + d*np.cos(theta1)

                    # crf amplitude
                    # e * coswt + f * sinwt
                    # sqrt(e**2 + f**2) * sin(wt+phi)
                    # cos(phi) = f
                    # sin(phi) = e
                    # tan(phi) = e/f
                    amp_crf = (amp_cos_crf**2 + amp_sin_crf**2)**(1/2)
                    # crf phase
                    phase_crf = np.arctan2(amp_cos_crf, amp_sin_crf)

                    ## End of coordinates transform ##

                    # Need to clip some amplitude phases if necessary
                    # Need to clip unrealistic phase values for true and est
                    if state in ['true','est']:
                        # Check if the value is valid not
                        ve_model = self.grid_set_velo[point][0]
                        vn_model = self.grid_set_velo[point][1]

                        # positive value means ice shelf
                        vu_model = self.grid_set_velo[point][2]

                        v_model = (ve_model**2 + vn_model**2)**(1/2)
    
                        lon, lat = self.int5d_to_float(point)
 
                        ############## Phase ####################
                        # Convert unit of phase (use days for Msf)
                        phase_alf = self.rad2deg(phase_alf)
                        phase_alf = self.wrapped_deg(phase_alf)
    
                        # Record the degree value
                        phase_alf_in_deg = phase_alf
    
                        phase_alf = self.deg2day(phase_alf, tide_name)
    
                        phase_crf = self.rad2deg(phase_crf)
                        phase_crf = self.wrapped_deg(phase_crf)
    
                        # Record the degree value
                        phase_crf_in_deg = phase_crf
    
                        phase_crf = self.deg2day(phase_crf, tide_name)

                        # Set criterions for clipping    
                        if self.proj == "Rutford": 
                            thres_for_v = 0.1
                            thres_for_amp_alf = 0.075
                            thres_for_amp_crf = 0.025

                        elif self.proj == "Evans":
                            thres_for_v = 0.1
                            #thres_for_v = 0.5
                            #thres_for_v = 1.0

                            # Along-flow
                            #thres_for_amp_alf = 0.1
                            thres_for_amp_alf = 0.075
                            #thres_for_amp_alf = 0.050
                            #thres_for_amp_alf = 0.025
                            #thres_for_amp_alf = 0.010

                            # Cross-flow
                            #thres_for_amp_crf = 0.025
                            thres_for_amp_crf = 0.010
                            
                            # 2021.05.19
                        else:
                            raise ValueError()
 
   
                        amp_full = (amp_alf**2 + amp_crf**2)**0.5
    
                        # Remove some invalid phase values
                        if v_model > thres_for_v and amp_alf > thres_for_amp_alf:
                            if self.proj == "Rutford":
                            # Remove the northern data
                            #if self.proj == "Rutford" and lat<-77.8:
                                pass
    
                            # clip the northern tributaries
                            #elif self.proj == "Evans" and lat<-75.85:
                            # clip the western tributaries
                            elif self.proj == "Evans" and (not (lon<-77.5 and vu_model==0)):
                                pass
                            else:
                                phase_alf = np.nan
                                phase_alf_in_deg = np.nan
                        else:
                            phase_alf = np.nan
                            phase_alf_in_deg = np.nan
    
    
                        # filter the phase crf
                        if v_model > thres_for_v and amp_crf > thres_for_amp_crf:
                            if self.proj == "Rutford" and lat<-77.8:
                                pass
    
                            #elif self.proj == "Evans" and ampU>0.5:
                            #elif self.proj == "Evans" and lat<-75.85:
                            # only keep the valus on ice shelf
                            elif self.proj == "Evans" and vu_model>0:
                                pass
    
                            else:
                                phase_crf = np.nan
                                phase_crf_in_deg = np.nan
                        else:
                            phase_crf = np.nan
                            phase_crf_in_deg = np.nan

                        ############## Amplitude ################
                        if self.proj == 'Evans':
                            if v_model>0.05:
                                pass
                            else:
                                amp_alf = np.nan
                                amp_crf = np.nan

                        # Save the output
                        quant = {}
                        quant["Msf_along_flow_displacement_amplitude"] = amp_alf
                        quant["Msf_along_flow_displacement_phase"] = phase_alf
                        quant["Msf_along_flow_displacement_phase_in_deg"] = phase_alf_in_deg
    
    
                        quant["Msf_cross_flow_displacement_amplitude"] = amp_crf
                        quant["Msf_cross_flow_displacement_phase"] = phase_crf
                        quant["Msf_cross_flow_displacement_phase_in_deg"] = phase_crf_in_deg
    
                        quant["Msf_horizontal_displacement_amplitude"] = amp_full 

                    # For uncertainty
                    elif state in ['uq']:

                        #amp_alf = 0
                        #phase_alf = 0
                        #phase_alf_in_deg = 0

                        #amp_crf = 0
                        #phase_crf = 0
                        #phase_crf_in_deg = 0
                        #
                        #amp_full = 0

                        # Calculate the uncertainty, we need to first recover the sigma of 

                        # Find the sigma of E&N amplitude and phase
                        sigma_ampE = data_vec[3+k*6]
                        sigma_phaseE = data_vec[3+k*6+3]
    
                        sigma_ampN = data_vec[3+k*6+1]
                        sigma_phaseN = data_vec[3+k*6+4]

                        #a = ampE * np.sin(phaseE) # amp of cos term
                        #b = ampE * np.cos(phaseE) # amp of sin term
                        #c = ampN * np.sin(phaseN) # amp of cos term
                        #d = ampN * np.cos(phaseN) # amp of sin term

                        sigma2_a = (np.sin(phaseE)**2) * (sigma_ampE**2) + ((ampE * np.cos(phaseE))**2) * (sigma_phaseE**2)
                        sigma2_b = (np.cos(phaseE)**2) * (sigma_ampE**2) + ((ampE * np.sin(phaseE))**2) * (sigma_phaseE**2)

                        sigma2_c = (np.sin(phaseN)**2) * (sigma_ampN**2) + ((ampN * np.cos(phaseN))**2) * (sigma_phaseN**2)
                        sigma2_d = (np.cos(phaseN)**2) * (sigma_ampN**2) + ((ampN * np.sin(phaseN))**2) * (sigma_phaseN**2)

                        # For along-flow, calculate the sigma2 of cos term and sin term
                        # Notes:
                        #amp_cos_alf = a*np.cos(theta1) - c*np.sin(theta1)
                        #amp_sin_alf = b*np.cos(theta1) - d*np.sin(theta1)

                        sigma2_amp_cos_alf = sigma2_a * np.cos(theta1)**2 + sigma2_b * np.sin(theta1)**2
                        sigma2_amp_sin_alf = sigma2_b * np.cos(theta1)**2 + sigma2_d * np.sin(theta1)**2


                        # For cross-flow, calculate the sigma2 of cos term and sin term
                        # Notes:
                        #amp_cos_crf = a*np.sin(theta1) + c*np.cos(theta1)
                        #amp_sin_crf = b*np.sin(theta1) + d*np.cos(theta1)

                        sigma2_amp_cos_crf = sigma2_a * np.sin(theta1)**2 + sigma2_c * np.cos(theta1)**2
                        sigma2_amp_sin_crf = sigma2_b * np.sin(theta1)**2 + sigma2_d * np.cos(theta1)**2


                        # Find the error for amplitude and phase
                        # Notes:
                        # Amplitude error
                        #amp_error = (error_c_t * np.sin(phase)**2 - error_s_t * np.cos(phase)**2) / (np.sin(phase)**4 - np.cos(phase)**4)
                        # Phase error
                        #phase_error = (-error_c_t * np.cos(phase)**2 + error_s_t * np.sin(phase)**2) / (amp**2 * (np.sin(phase)**4 - np.cos(phase)**4))

                        # We have amp_alf, phase_alf, amp_crf, phase_crf

                        sigma2_amp_alf = (sigma2_amp_cos_alf * np.sin(phase_alf)**2 - sigma2_amp_sin_alf * np.cos(phase_alf)**2) / (np.sin(phase_alf)**4 - np.cos(phase_alf)**4)

                        sigma2_phase_alf = (-sigma2_amp_cos_alf * np.cos(phase_alf)**2 + sigma2_amp_sin_alf * np.sin(phase_alf)**2) / (amp_alf**2 * (np.sin(phase_alf)**4 - np.cos(phase_alf)**4))

                        sigma2_amp_crf = (sigma2_amp_cos_crf * np.sin(phase_crf)**2 - sigma2_amp_sin_crf * np.cos(phase_crf)**2) / (np.sin(phase_crf)**4 - np.cos(phase_crf)**4)

                        sigma2_phase_crf = (-sigma2_amp_cos_crf * np.cos(phase_crf)**2 + sigma2_amp_sin_crf * np.sin(phase_crf)**2) / (amp_crf**2 * (np.sin(phase_crf)**4 - np.cos(phase_crf)**4)) 

                        #print(amp_alf)
                        #print(amp_crf)
                        #print(phase_alf)
                        #print(phase_crf)
                        #print(stop)

                        # Take the square root
                        #sigma_amp_alf = np.sqrt(sigma2_amp_alf)
                        #sigma_phase_alf = np.sqrt(sigma2_phase_alf)
                        #sigma_amp_crf = np.sqrt(sigma2_amp_crf)
                        #sigma_phase_crf = np.sqrt(sigma2_phase_crf)

                        sigma_amp_alf = np.sqrt(abs(sigma2_amp_alf))
                        sigma_phase_alf = np.sqrt(abs(sigma2_phase_alf))
                        sigma_amp_crf = np.sqrt(abs(sigma2_amp_crf))
                        sigma_phase_crf = np.sqrt(abs(sigma2_phase_crf))

                        # Convert phase to days
                        sigma_phase_alf_deg = self.rad2deg(sigma_phase_alf)
                        sigma_phase_alf_days = self.deg2day(sigma_phase_alf_deg, tide_name)

                        sigma_phase_crf_deg = self.rad2deg(sigma_phase_crf)
                        sigma_phase_crf_days = self.deg2day(sigma_phase_crf_deg, tide_name)

                        # Save the result
                        quant = {}
                        quant["Msf_along_flow_displacement_amplitude"] = sigma_amp_alf
                        quant["Msf_along_flow_displacement_phase"] = sigma_phase_alf_days
                        quant["Msf_along_flow_displacement_phase_in_deg"] = 0
    
    
                        quant["Msf_cross_flow_displacement_amplitude"] = sigma_amp_crf
                        quant["Msf_cross_flow_displacement_phase"] = sigma_phase_crf_days
                        quant["Msf_cross_flow_displacement_phase_in_deg"] = 0
    
                        quant["Msf_horizontal_displacement_amplitude"] = (sigma_amp_alf ** 2 + sigma_amp_crf ** 2)**0.5

                    else:
                        raise Exception("Unknown state")

                    return quant

                else:
                    k=k+1

        ############################################
        # Msf East amp.
        elif quant_name == 'Msf_east_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(data_vec[3+k*6],tide_name)
                    quant = ampE
                else:
                    k=k+1

        # Msf East phase.
        elif quant_name == 'Msf_east_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':

                    value = data_vec[3+k*6+3]
                    if state in [ 'true','est']:
                        phaseE = self.velo_phase_to_dis_phase(value)
                        quant = self.rad2deg(phaseE)
                        quant = self.deg2day(quant, tide_name)
                        
                    elif state in ['uq']:
                        quant = value
                        quant = self.rad2deg(quant)
                        quant = self.deg2day(quant, tide_name)

                    else:
                        raise Exception("Unknown state")
 
                else:
                    k=k+1

        # Msf North amp.
        elif quant_name == 'Msf_north_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampN = self.velo_amp_to_dis_amp(data_vec[3+k*6+1],tide_name)
                    quant = ampN
                else:
                    k=k+1

        # Msf North phase.
        elif quant_name == 'Msf_north_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':

                    value = data_vec[3+k*6+4]

                    if state in [ 'true','est']:
                        phaseE=self.velo_phase_to_dis_phase(value)
                        quant = self.rad2deg(phaseE)
                        quant = self.deg2day(quant, tide_name)
                        
                    elif state in ['uq']:
                        quant = value
                        quant = self.rad2deg(quant)
                        quant = self.deg2day(quant, tide_name)

                    else:
                        raise Exception("Unknown state")
 
                else:
                    k=k+1

        ############### End of Msf ###############

        # Mf lumped horizontal displacement amplitude.
        elif quant_name == 'Mf_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampE = self.velo_amp_to_dis_amp(data_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(data_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # Mf up displacement amplitude.
        elif quant_name == 'Mf_up_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = ampU
                else:
                    k=k+1

        # Mf Up phase. 
        elif quant_name.startswith('Mf_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    value = data_vec[3+k*6+5]

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
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


        ################ Semidirunal and Dirunal ###############################

        # M2 lumped horizontal displacement amplitude.
        elif quant_name == 'M2_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampE = self.velo_amp_to_dis_amp(data_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(data_vec[3+k*6+1],tide_name)
                    quant = np.sqrt(ampE**2 + ampN**2)
                else:
                    k=k+1

        # O1 lumped horizontal displacement amplitude.
        elif quant_name == 'O1_horizontal_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampE = self.velo_amp_to_dis_amp(data_vec[3+k*6],tide_name)
                    ampN = self.velo_amp_to_dis_amp(data_vec[3+k*6+1],tide_name)
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
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1
        
        # O1 Up phase. 
        # (only on ice shelves)
        elif quant_name.startswith('O1_up_displacement_phase'):
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    thres = 0.1
                    # 2021.01.29
                    thres = 0.06


                    model_up = self.grid_set_velo[point][2]>0

                    # value in velocity model > 0
                    #if (ampU > thres) or (state=='uq') :
                    if (ampU >=thres or state == 'uq') and (self.proj == 'Rutford' or (self.proj == 'Evans' and model_up>0)):

                        value = data_vec[3+k*6+5]
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
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

        # M2 Up phase. 
        # convert to minute
        elif quant_name.startswith('M2_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)

                    # old way
                    #thres = 0.3
                    #if (self.grid_set_velo[point][2]>0 and ampU >= thres) or (state=='uq'):

                    # Explanation: I use M2 phase as the proxy for semi-dirunal phase variation
                    # To keep the any many data points as possible, I need to lower the criterion

                    # 2021.01.29
                    thres = 0.1
                    # 2021.02.03
                    #thres = 0.01
                    # 2021.05.17
                    #thres = 0.05

                    model_up = self.grid_set_velo[point][2]
                    
                    #if ampU >=thres or state == 'uq':

                    # clip values outside ice-shelf for Evans
                    if (ampU >=thres or state == 'uq') and (self.proj == 'Rutford' or (self.proj == 'Evans' and model_up>0)):

                        value = data_vec[3+k*6+5]
                        if state in [ 'true','est']:
                            # Find the phase
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
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1
        
        # N2 Up phase. 
        # convert to minute
        elif quant_name.startswith('N2_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'N2':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    thres = 0.1
                    # 2021.01.29
                    thres = 0.04
                    # 2021.04.06
                    thres = 0.03

                    model_up = self.grid_set_velo[point][2]
                    #if (ampU >= thres) or (state=='uq'):
                    if (ampU >=thres or state == 'uq') and (self.proj == 'Rutford' or (self.proj == 'Evans' and model_up>0)):

                        value = data_vec[3+k*6+5]
                        if state in [ 'true','est']:
                            # Find the phase

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
                            # Find the phase

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
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    quant = np.sqrt(ampU**2)
                else:
                    k=k+1

        # Q1 Up phase. 
        # convert to minute
        elif quant_name.startswith('Q1_up_displacement_phase'):

            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Q1':
                    ampU = self.velo_amp_to_dis_amp(data_vec[3+k*6+2],tide_name)
                    thres = 0.03

                    #if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                    if (ampU > thres) or (state=='uq'):

                        value = data_vec[3+k*6+5]
                        if state in [ 'true','est']:
                            # Find the phase
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
                            # Find the phase
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

        # Additonal parameters
        elif quant_name == "up_amplitude_scaling":
            assert len(data_vec)>= 3 + self.n_modeling_tides*6 + 1, print("length of param_uq_vec has problem for up_amplitude scaling ", len(data_vec))

            # the first index after tidal params
            ind = 3 + self.n_modeling_tides * 6

            # TODO, Need to import true up amp scaling here
            if state in ['true']:
                quant = 0
            elif state in ['est']:
                quant = data_vec[ind]
            elif state in ['uq']:
                quant = data_vec[ind]
            else:
                raise ValueError()

        elif quant_name == "topo_resid":
            assert len(data_vec)>= 3 + self.n_modeling_tides*6 + 1, print("length of param_uq_vec has problem for topo resid ", len(data_vec))

            # For now, the last index is topo_resid
            ind = -1

            # TODO, Need to import true topo resid
            if state in ['true']:
                quant = 0
            elif state in ['est']:
                quant = data_vec[ind]
            elif state in ['uq']:
                quant = data_vec[ind]
            else:
                raise ValueError()

        else:
            quant = None
            raise Exception(quant_name, ' is not defined yet!')

        return quant
 
def main():

    fourD = fourdvel()
    fourD.preparation()
   
if __name__=='__main__':
    main()
