#!/usr/bin/env python3

# Author: Minyan Zhong
# Create time: June 2018

###

# All time is in the unit of day.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

from matplotlib import cm

import seaborn as sns

import datetime
from datetime import date

import multiprocessing

import time

from scipy import linalg

from basics import basics

from numba import jit

INT_NAN = -99999

class fourdvel(basics):

    def __init__(self, param_file=None):

        super(fourdvel,self).__init__()

        #print(param_file)

        if param_file is None:
            self.read_parameters(sys.argv[1])
        else:
            self.read_parameters(param_file)

        if self.proj == "Evans":
            # Data.

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
            self.s1_workdir = "/net/jokull/nobak/mzzhong/S1-Evans"
        
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

            self.s1_workdir = "/net/jokull/nobak/mzzhong/S1-Evans"
        
            for it in self.s1_tracks:
                self.s1_data[it] = []

            self.satellite_constants()

        # Related folders
        self.design_mat_folder = './design_mat'


        ########### Control the solution (Rutford) ##########
        # physical size     lat x lon
        #   100m x 100m     0.001 x 0.005
        #   500m x 500m     0.005 x 0.025
        resolution=self.resolution

        if self.proj == "Rutford":
            if resolution == 500:
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
                print(stop)

        elif self.proj == "Evans":
            if resolution == 500:
                self.lat_step = 0.005
                self.lon_step = 0.02
                self.lat_step_int = self.round_int_5dec(self.lat_step)
                self.lon_step_int = self.round_int_5dec(self.lon_step)
            else:
                print("Unknown resolution")
                print(stop)

        else:
            raise Exception("Unknown project name")
            return

        # The inverse to size
        self.lon_re = np.round(1/self.lon_step).astype(int)
        self.lat_re = np.round(1/self.lat_step).astype(int)

        print("fourdvel initialization done")

    def read_parameters(self, param_file):

        print(param_file)
        
        f = open(param_file)

        fmt = '%Y%m%d'

        params = f.readlines()

        # Quick and dirty:
        self.data_uncert_const = None

        for param in params:
            
            try:    
                name,value = param.split(':')
                name = name.strip()
                value = value.strip()
            except:
                continue

            if name == 'proj':
                self.proj = value
                print('proj',value)

            if name == 'test_id':
                self.test_id = value
                print('test_id',value)

                self.estimation_dir = "/net/kamb/ssd-tmp1/mzzhong/insarRoutines/estimations/"+str(self.test_id)
                if not os.path.exists(self.estimation_dir):
                    os.mkdir(self.estimation_dir)

                print("estimation dir: ", self.estimation_dir)

                ## Save the paramerer file
                if param_file=="params.in":
                    cmd = " ".join(["cp", param_file, "./params/"+str(self.test_id)+'_'+param_file])
                    print("copy param file: ",cmd)
                    os.system(cmd)

            if name == "test_point":

                if value!="None":
                    the_point = [float(x) for x in value.split(",")]
                    self.test_point = self.float_lonlat_to_int5d(the_point)
                else:
                    self.test_point = None
                print("test_point", value, self.test_point)

            if name == "inversion_method":

                self.inversion_method = value
                print("inversion_method", value)

            if name == "sampling_data_sigma":

                self.sampling_data_sigma = float(value)
                print("sampling_data_sigma", value)

            if name == 'test_mode':
                self.test_mode = int(value)
                print('test_mode',value)

            if name == 'grid_set_name':
                self.grid_set_name = value
                print('grid_set_name: ',value)

            if name == 'resolution':
                self.resolution = int(value)
                print('resolution: ',value)

            if name == 'grid_set_velo_name':
                self.grid_set_velo_name = value
                print('grid_set_velo_name: ',value)

            if name == 'tile_set_name':
                self.tile_set_name = value
                print('tile_set_name: ',value)

            if name == 'grid_set_data_uncert_name':
                self.grid_set_data_uncert_name = value
                print('grid_set_data_uncert_name: ',value)

            if name == 'data_uncert_const':
                self.data_uncert_const = (float(value.split(',')[0]), float(value.split(',')[1]))
                print('data_uncert_const: ',self.data_uncert_const)

            if name == 'est_dict_name':
                self.est_dict_name = value
                print('est_dict_name: ',value)

            if name == 'use_s1':
                if value == 'True':
                    self.use_s1 = True
                else:
                    self.use_s1 = False

                print('use_s1: ',value)

            if name == 's1_start':
                self.s1_start = datetime.datetime.strptime(value, fmt).date()
                print('s1_start: ',value)

            if name == 's1_end':
                self.s1_end = datetime.datetime.strptime(value, fmt).date()
                print('s1_end: ',value)

            if name == 'use_csk':
                if value == 'True':
                    self.use_csk = True
                else:
                    self.use_csk = False
                print('use_csk: ',value)

            if name == 'csk_start':
                self.csk_start = datetime.datetime.strptime(value, fmt).date()
                print('csk_start: ',value)

            if name == 'csk_end':
                self.csk_end = datetime.datetime.strptime(value, fmt).date()
                print('csk_end: ',value)

            if name == 'csk_log':
                self.csk_log = value
                print('csk_log: ',value)

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

            if name == 'grounding':
                self.grounding = float(value)
                print("grounding: ",self.grounding)

        return 0
            
    def get_CSK_trackDates_from_log(self):

        import csv
        from CSK_Utils import CSK_Utils

        # csk_data[track_number] = [date1, date2, date3,...]
        csk_data = self.csk_data

        csk_start = self.csk_start
        csk_end = self.csk_end

        # Not all data are available, currently, so I read the files exported from E-GEOS. I will switch to real data
        #file_folder = '/home/mzzhong/links/kraken-nobak-net/CSKData/data_20171116_20180630'
        file_folder = self.csk_log

        data_file = os.path.join(file_folder,'all.csv')

        csk = CSK_Utils()

        tot_product = 0
        tot_frames = 0

        with open(data_file) as dataset:
            csv_reader = csv.reader(dataset, delimiter=';')
            line = 0
            for row in csv_reader:
                line = line + 1
                if line == 1:
                    continue
                
                # Count as one product.
                tot_product = tot_product + 1

                # Satellite 
                sate = 'CSKS' + row[1][-1]
                # Date
                acq_datefmt = row[5].split(' ')[0]
                # Direction
                direction = row[7][0]

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

                    # Record it.    
                    if track[0] in csk_data.keys():
                        csk_data[track[0]].append(theDate)
                    else:
                        csk_data[track[0]] = [theDate]
    
                    tot_frames = tot_frames + csk.numOfFrames[track[0]]
    
        
        print("Number of product: ", tot_product)
        print("Number of frames: ", tot_frames)

        # Sort the dates of each track.
        # Output the track info
        for track_num in sorted(csk_data.keys()):
            csk_data[track_num].sort()
            print(track_num)
            print(csk_data[track_num])
        
        return 0

    def get_CSK_trackDates(self):

        import glob

        csk_data = self.csk_data
        csk_start = self.csk_start
        csk_end = self.csk_end

        tracklist = self.csk_tracks

        option = "data_based"

        if option=="data_based":

            for track_num in tracklist: 
            
                filefolder = '/net/kraken/nobak/mzzhong/CSK-Rutford/track_' + str(track_num).zfill(3) + '_0' + '/raw/201*'

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

                print("track_num: ",track_num)
                print(csk_data[track_num])

        else:
            print("option", option, "is not defined yet")
            raise Exception("dates are not available")

        return 0

    def get_S1_trackDates(self):

        from S1_Utils import S1_Utils
        import glob

        s1_data = self.s1_data
        s1_start = self.s1_start
        s1_end = self.s1_end

        s1 = S1_Utils()

        tracklist = self.s1_tracks

        option = "fake"

        if option=="data_based":

            for track_num in tracklist: 
            
                filefolder = '/home/mzzhong/links/jokull-nobak-net/S1-Evans/data_' + str(track_num) + '/*zip'
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
                print(s1_data[track_num])

        elif option == "fake":

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
                print(s1_data[track_num])
        else:
            print("option", option, "is not defined yet")
            raise Exception("dates are not available")

        return 0

    def get_grid_set_velo(self):

        self.grid_set_velo = None

        if hasattr(self, 'grid_set_velo_name'):

            dim = 3
            if dim == 3:
                grid_set_velo_pkl = self.grid_set_velo_name + '_' + str(self.resolution) + '_3d' + '.pkl'
    
            if os.path.exists(grid_set_velo_pkl):
                print('Loading grid_set_velo...')
                with open(grid_set_velo_pkl,'rb') as f:
                    self.grid_set_velo = pickle.load(f)
            else:
                raise Exception("Unable to load velocity model")

    def get_tile_set(self):

        self.tile_set = None
        if hasattr(self,'tile_set_name'):
            self.tile_set_name = self.tile_set_name.replace("$resolution", str(self.resolution))

            tile_set_pkl = self.tile_set_name + '.pkl'
            if os.path.exists(tile_set_pkl):
                print('Loading tile_set ...')
                with open(tile_set_pkl,'rb') as f:
                    self.tile_set = pickle.load(f)

            print("total number of tiles: ", len(self.tile_set))

    def get_data_uncert(self):

        self.grid_set_data_uncert = None
        if hasattr(self,'grid_set_data_uncert_name'):
            grid_set_data_uncert_set_pkl = self.grid_set_data_uncert_name + '.pkl'
    
            if os.path.exists(grid_set_data_uncert_set_pkl):
                print('Loading data uncert set ...')
                with open(grid_set_data_uncert_set_pkl,'rb') as f:
                    self.grid_set_data_uncert = pickle.load(f)

    def round_to_grid_points(self, x, re):
        
        return np.round(x * re)/re


    def get_grid_set_v2(self):
        import gdal

        # Currently only CSK on Rutford
        redo = 0

        ###########################################
        # Generate pickle file
        grid_set_pkl = self.grid_set_name + '_' + str(self.resolution) + '.pkl'

        lat_step_int = self.lat_step_int
        lon_step_int = self.lon_step_int

        print("step_int: ", lon_step_int, lat_step_int)
        print("re size (sampling rate): ", self.lon_re, self.lat_re)

        if os.path.exists(grid_set_pkl) and redo == 0:
            print('Loading grid_set...')

            with open(grid_set_pkl,'rb') as f:
                self.grid_set = pickle.load(f)

            print('total number of grid points: ', len(self.grid_set))
        else:
            print("Cannot find: ", grid_set_pkl, "or redo is True")
            print('Calculating grid_set...')
            grid_set = {}

            # Only use CSK for now
            satellites = ['csk','s1']
            #satellites = ['csk']

            directory = {}
            tracklist = {}
            offset_id = {}

            directory['csk'] = self.csk_workdir

            tracklist['csk'] = self.csk_tracks
            if self.proj == "Rutford":
                offset_id['csk'] = 20190921
            elif self.proj == "Evans":
                offset_id['csk'] = 20180712

            directory['s1'] = self.s1_workdir

            # update 20190702
            tracklist['s1'] = self.s1_tracks
            #offset_id['s1'] = 20180703
            offset_id['s1'] = 20200101

            for sate in satellites:

                for track_num in tracklist[sate]:

                    print(sate,track_num)

                    if sate == 'csk' and self.proj == "Rutford":
                        trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(3)+'_0')
                    elif sate == "csk" and self.proj == "Evans":
                        trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(2)+'0')
                    elif sate == "s1":
                        trackdir = os.path.join(directory[sate],'track_' + str(track_num))
                    else:
                        raise Exception("Undefined")
 
                    gc_losfile = os.path.join(trackdir,'merged','geom_master','gc_los_offset_' + str(offset_id[sate]) + '.rdr')
                    
                    gc_losvrtfile = gc_losfile + '.vrt'
                    dataset = gdal.Open(gc_losvrtfile)
                    geoTransform = dataset.GetGeoTransform()
                    
                    lon0 = geoTransform[0]
                    lon_interval = geoTransform[1]
        
                    lat0 = geoTransform[3]
                    lat_interval = geoTransform[5]
        
                    xsize = dataset.RasterXSize
                    ysize = dataset.RasterYSize
        
                    lon_list = np.linspace(lon0, lon0 + lon_interval*(xsize-1), xsize)
                    lat_list = np.linspace(lat0, lat0 + lat_interval*(ysize-1), ysize)

                    #print(lon_list, lat_list)

                    ## If files not geocoded onto the FULL RESOLUTION grid points,
                    ## do it here.

                    ## This is closed for now, because rutford offset field have
                    ## been geocoded to full resolution
    
                    ## lat 0.001
                    ## lon 0.005

                    #lon_list = self.round_to_grid_points(lon_list, self.lon_re)
                    #lat_list = self.round_to_grid_points(lat_list, self.lat_re)
                    #######################################################

                    #print(lon_list, lat_list)

                    ### Convert to 5 decimal point integer
                    lon_list = self.round_int_5dec(lon_list)
                    lat_list = self.round_int_5dec(lat_list)
        
                    #print(lon_list,len(lon_list),xsize)
                    #print(lat_list,len(lat_list),ysize)

                    # Mesh grid
                    grid_lon, grid_lat = np.meshgrid(lon_list, lat_list)
        
                    # Maskout the invalid
                    los = dataset.GetRasterBand(1).ReadAsArray()
                    azi = dataset.GetRasterBand(2).ReadAsArray()
        
                    #print(grid_lon)
                    #print(grid_lon.shape)
                    #print(los.shape)

                    grid_lon[los == 0] = INT_NAN
                    grid_lat[los == 0] = INT_NAN
        
                    #fig = plt.figure(1)
                    #ax = fig.add_subplot(111)
                    #ax.imshow(grid_lat)
                    #plt.show()

                    # Flatten the grid points        
                    grid_lon_1d = grid_lon.flatten()
                    grid_lat_1d = grid_lat.flatten()
        
                    # Read the observation vectors
                    enu_gc_losfile = os.path.join(trackdir,'merged','geom_master','enu_gc_los_offset_' + str(offset_id[sate]) + '.rdr.vrt')
                    enu_gc_azifile = os.path.join(trackdir,'merged','geom_master','enu_gc_azi_offset_' + str(offset_id[sate]) + '.rdr.vrt')
                    try:
                        dataset = gdal.Open(enu_gc_losfile)
                    except:
                        raise Exception('geometry file not exist')

                    # Los ENU
                    elos = dataset.GetRasterBand(1).ReadAsArray()
                    nlos = dataset.GetRasterBand(2).ReadAsArray()
                    ulos = dataset.GetRasterBand(3).ReadAsArray()
        
                    # Azi ENU
                    dataset = gdal.Open(enu_gc_azifile)
                    eazi = dataset.GetRasterBand(1).ReadAsArray()
                    nazi = dataset.GetRasterBand(2).ReadAsArray()
                    uazi = dataset.GetRasterBand(3).ReadAsArray()
        
                    #print(grid_lon_1d,len(grid_lon_1d))
                    #print(grid_lat_1d,len(grid_lat_1d))


                    # Loop through all grid points   
                    for kk in range(len(grid_lon_1d)):
        
                        ii = kk // xsize
                        jj = kk - ii * xsize
        
                        if grid_lon[ii,jj]==INT_NAN or grid_lat[ii,jj]==INT_NAN:
                            continue

                        ### Add downsampling here #########
                        if grid_lon[ii,jj]%lon_step_int!=0 or grid_lat[ii,jj]% lat_step_int!=0:
                            continue

                        # The element being pushed into the list.
                        # 1. track number; 2. los (three vectors) 3. azi (three vectors) 4. satellite name.
                        info = (track_num,(elos[ii,jj],nlos[ii,jj],ulos[ii,jj]),(eazi[ii,jj],nazi[ii,jj],uazi[ii,jj]),sate)
        
                        # Push into the grid_set, only add new grid when sate is csk.
                        if (grid_lon[ii,jj],grid_lat[ii,jj]) not in grid_set.keys():
                            if sate=='csk':
                                grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])] = [info]
                            else:
                                pass
                        else:
                            grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])].append(info)

            #print(grid_set.keys())
            print("Total number of grid points: ", len(grid_set))

            print("Writing to Pickle file...")
            with open(grid_set_pkl,'wb') as f:
                pickle.dump(grid_set,f)

            self.grid_set = grid_set

            print("Done")

        print("Output a test point")

        if self.proj == "Rutford":
            key = (-8100000,-7900000)
            print("Test point: ",key)
            print(self.grid_set[key])

        elif self.proj == "Evans":
            key = (-7700000, -7680000)
            print("Test point: ",key)
            print(self.grid_set[key])

        return 0

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

    def stack_design_mat_set(self, point_set, design_mat_set, offsetfields_set):

        stack_design_mat_set = {}
        for point in point_set:
            stack_design_mat_set[point] = self.stack_design_mat_for_point(point, design_mat_set, offsetfields_set[point])

        return stack_design_mat_set

    def stack_design_mat_for_point(self, point, design_mat_set, offsetfields):

        # At point level
        # Stack the design matrix for all pairs
        stacked_design_mat_EN_ta = []
        stacked_design_mat_EN_tb = []
        stacked_design_mat_U_ta = []
        stacked_design_mat_U_tb = []

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


    def get_timings(self):

        fmt = "%Y%m%d"
        timings_pkl = self.pickle_dir + '/' + '_'.join(['timings', 'csk', self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt)]) + '.pkl'

        print('timing file: ', timings_pkl)

        redo = 1

        if os.path.exists(timings_pkl) and redo == 0:
            with open(timings_pkl, 'rb') as f:
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
            
            with open(timings_pkl,'wb') as f:
                pickle.dump(self.timings, f)

        #print(self.timings)
        #print(stop)
 
        return 0

    def get_design_mat_set(self):

        from forward import forward
        fwd = forward()

        redo = 1

        fmt = "%Y%m%d"
        design_mat_set_pkl = self.pickle_dir +'/' + '_'.join(['design_mat_set', 'csk',self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt)] + self.modeling_tides ) + '.pkl'

        print('design_mat_set file:', design_mat_set_pkl)

        if os.path.exists(design_mat_set_pkl) and redo==0:
            with open(design_mat_set_pkl,'rb') as f:
                self.design_mat_set = pickle.load(f)
        else:
            self.design_mat_set = fwd.design_mat_set(self.timings, self.modeling_tides)
            print("Size of design mat set: ",len(self.design_mat_set))

            with open(design_mat_set_pkl,'wb') as f:
                pickle.dump(self.design_mat_set,f)

        # For simulation, we need design mat for all tides
        if self.test_mode==1 or self.test_mode==2:

            rutford_design_mat_set_pkl = self.pickle_dir +'/'+ '_'.join(['design_mat_set', 'csk',self.csk_start.strftime(fmt), self.csk_end.strftime(fmt), 's1', self.s1_start.strftime(fmt), self.s1_end.strftime(fmt), 'Rutford_full'] ) + '.pkl'
    
            if os.path.exists(rutford_design_mat_set_pkl) and redo==0:
                with open(rutford_design_mat_set_pkl, 'rb') as f:
                    self.rutford_design_mat_set = pickle.load(f)
    
            else:
                rutford_tides = ['K2','S2','M2','K1','P1','O1','Msf','Mf','Mm','Ssa','Sa']
                self.rutford_design_mat_set = fwd.design_mat_set(self.timings, rutford_tides)
                print("Size of design mat set: ", len(self.rutford_design_mat_set))
    
                with open(rutford_design_mat_set_pkl,'wb') as f:
                    pickle.dump(self.rutford_design_mat_set,f)
        return 0

    def get_offset_field_stack(self):
    
        self.offsetFieldStack_all = {}

        if self.use_csk:

            for track in self.csk_tracks:
                track_offsetFieldStack_pkl = os.path.join(self.csk_workdir, "track_" + str(track).zfill(3) + '_0', \
                                                            "cuDenseOffsets", "offsetFieldStack_20190921_v10.pkl")
                if os.path.exists(track_offsetFieldStack_pkl):
                    print("Loading: ", track_offsetFieldStack_pkl)
                    with open(track_offsetFieldStack_pkl,'rb') as f:
                        offsetFieldStack = pickle.load(f)
                        self.offsetFieldStack_all[("csk", track)] = offsetFieldStack


        if self.use_s1:

            for track in self.s1_tracks:
                track_offsetFieldStack_pkl = os.path.join(self.s1_workdir, "track_" + str(track), \
                                                            "cuDenseOffsets", "offsetFieldStack_20200101_v10.pkl")
                if os.path.exists(track_offsetFieldStack_pkl):
                    print("Loading: ", track_offsetFieldStack_pkl)
                    with open(track_offsetFieldStack_pkl,'rb') as f:
                        offsetFieldStack = pickle.load(f)
                        self.offsetFieldStack_all[("s1", track)] = offsetFieldStack

    def preparation(self):

        # Get pre-defined grid points and the corresponding tracks and vectors.
        self.get_grid_set_v2()

        self.get_grid_set_velo()
        self.get_tile_set()
        self.get_data_uncert()

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

        # Prepartion G matrix libaray for inversion
        self.get_timings()
        self.get_design_mat_set()

        # Load offset field stack data
        self.get_offset_field_stack()

        return 0

    def tracks_to_full_offsetfields(self,tracks):
        
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
            linear_design_mat_set[point] = self.build_G(offsetfields=offsetfields)

        return linear_design_mat_set

    def build_G(self, point=None, tracks=None, offsetfields=None, horizontal = False):

        if point is not None:
            
            lon,lat = point

            #if not (lon>-81.5 and lon<-80.5 and lat<-76.5):
            #    return np.nan
    
            #if not (lon==-81.24 and (lat==-76.59 or lat==-76.595)):
            #    return 0
    
            #if not (lon==-72.14 and lat==-74.845):
            #    return -1
    
            # Downsampling to save time.
            #if not (np.round((lon*100)) % 8 == 0) and (np.round((lat*1000) % 20)==0):
            #    print(point)
            #    return np.nan

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

        # No offsetfield in available.
        if n_offsets ==0:
            G = np.zeros(shape=(1,1)) + np.nan
            return G 

        #print('total number of offsetfield:', n_offsets)
        #n_offsets = 10

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

            #print(offsetfields[i][4])

            t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]

            delta_td[i] = (offsetfields[i][1] - offsetfields[i][0]).days

            for j in range(n_modeling_tides):

                tide_name = modeling_tides[j]

                omega = 2 * np.pi / tide_periods[tide_name]
            
                delta_cos[i,j] = np.cos(omega*t_b) - np.cos(omega*t_a)
                delta_sin[i,j] = np.sin(omega*t_b) - np.sin(omega*t_a)

            #print(stop)

        n_rows = n_offsets * 2 # Each offset corresponds to a vector.

        # E, N, U components.
        n_cols = 3 + n_modeling_tides * 6 # cosE, cosN, cosU and sinE, sinN, sinU.
        
        # Only the E, N components.
        #n_cols = 3 + n_modeling_tides * 4 # cosE, cosN, sinE, sinN, Not finished below.
        
        ## G formation.
        G = np.zeros(shape=(n_rows,n_cols))

        # Iterate over offsetfields
        for i in range(n_offsets):
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


    def build_G_ENU_set(self, point_set, offsetfields_set):
        
        linear_design_mat_set = {}
        for point in point_set:
            offsetfields = offsetfields_set[point]
            linear_design_mat_set[point] = self.build_G_ENU(offsetfields=offsetfields)

        return linear_design_mat_set

    def build_G_ENU(self, point=None, tracks=None, offsetfields=None, horizontal = False):

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

        # No offsetfield in available.
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

            #print(offsetfields[i][4])

            t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]

            delta_td[i] = (offsetfields[i][1] - offsetfields[i][0]).days

            for j in range(n_modeling_tides):

                tide_name = modeling_tides[j]

                omega = 2 * np.pi / tide_periods[tide_name]
            
                delta_cos[i,j] = np.cos(omega*t_b) - np.cos(omega*t_a)
                delta_sin[i,j] = np.sin(omega*t_b) - np.sin(omega*t_a)

        n_rows = n_offsets * 3 # Three components

        if horizontal == False:
            # E, N, U components.
            n_cols = 3 + n_modeling_tides * 6 # cosE, cosN, cosU and sinE, sinN, sinU.
        else:
            # Only the E, N components.
            n_cols = 3 + n_modeling_tides * 4 # cosE, cosN, sinE, sinN, Not finished below.
        
        ## G formation.
        G = np.zeros(shape=(n_rows,n_cols))

        # E, N, U Three components
        for i in range(n_offsets):
            vecs = [(1,0,0),(0,1,0),(0,0,1)]

            # The observation vectors.
            for j in range(3):
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
                G[i*3+j,:] = row

        return G
        # End of building.

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

        num_params = 3 + n_modeling_tides*6
        param_vec = np.zeros(shape=(num_params,1))

        # If model_vec is invalid.
        if np.isnan(model_vec[0,0]):
            param_vec = param_vec + np.nan
            return param_vec

        # Model_vec is valid.
        param_vec[0:3,0] = model_vec[0:3,0]

        # Tides.
        for k in range(n_modeling_tides):
            
            tide_name = modeling_tides[k]

            # Model_vec terms: cosE, cosN, cosU, sinE, sinN, sinU.
            # Tide_vec terms:, ampE, ampN, ampU, phaseE, phaseN, phaseU
            
            # E N U
            for t in range(3):

                ### return value is in velocity domain m/d
                # disp = a*coswt + b*sinwt
                # velo = -aw * sinwt + bw*coswt
                # sqrt(a**2+b**2)*w * sin(wt + phi)
                # write it in the sin form, to be consistent with synthetic test
                # tan phi = b/(-a)

                # cos term.
                coe1 = model_vec[3+k*6+t]

                # sin term.
                coe2 = model_vec[3+k*6+t+3]

                omega = 2*np.pi / tide_periods[tide_name]

                # From displacement to velocity doamin
                # For amplitide
                amp = np.sqrt(coe1*coe1+coe2*coe2)*omega

                # Phase.
                phase = np.arctan2(coe2,-coe1)

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

        num_params = 3 + n_modeling_tides*6
        param_vec = np.zeros(shape=(num_params,1))

        # If tide_vec is invalid.
        if np.isnan(tide_vec[0,0]):
            param_vec = param_vec + np.nan
            return param_vec

        # Model_vec is valid.
        param_vec[0:3,0] = tide_vec[0:3,0]

        # Tides.
        for k in range(n_modeling_tides):
            
            tide_name = modeling_tides[k]

            # Model_vec terms: cosE, cosN, cosU, sinE, sinN, sinU.
            # Tide_vec terms:, ampE, ampN, ampU, phaseE, phaseN, phaseU
            
            # E N U
            for t in range(3):

                ### return value is in velocity domain m/d
                # velo = amp*sin(wt + phi)
                # velo = amp*sin(phi)coswt + amp*cos(phi)sinwt
                # disp = amp/w*sin(phi)*sinwt - amp/w*cos(phi)*coswt
                # cos term = -amp/w*cos(phi)
                # sin term = amp/w*sin(phi)

                # amp term.
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

        for i in range(n_data):
            # Range.
            if i % 2 == 0:
                Cd[i,i] = sigma[0]**2
            # Azimuth.
            else:
                Cd[i,i] = sigma[1]**2

            invCd[i,i] = 1/Cd[i,i]

        return invCd


    def model_prior_set(self, point_set, horizontal = False):

        invCm_set = {}
        for point in point_set:
            invCm_set[point] = self.model_prior(horizontal = horizontal)

        return invCm_set

    def model_prior(self, horizontal = False):

        n_modeling_tides = self.n_modeling_tides
        modeling_tides = self.modeling_tides

        num_params = 3 + n_modeling_tides*6

        # Model priori.
        
        # Sigmas of model parameters.
        inf_permiss = 0
        inf_restrict = 100000
 
        inv_sigma = np.zeros(shape=(num_params, num_params))

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


        #print(horizontal_long_period)
        #print(stop)

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
                if not tide_name in ['M2','S2','O1'] and up_short_period and (j==2 or j==5):
                    inv_sigma[k,k] = inf_restrict
                    #print(tide_name, j)

                # Control the horizontal to be only long period
                if not tide_name in ['Mf','Msf','Mm'] and horizontal_long_period and (j==0 or j==1 or j==3 or j==4):
                    inv_sigma[k,k] = inf_restrict
 
        invCm = np.square(inv_sigma)

        return invCm

    def model_posterior_set(self, point_set, linear_design_mat_set, data_prior_set, model_prior_set):
        Cm_p_set = {}
        for point in point_set:

            #print(point)
            Cm_p_set[point] = self.model_posterior(linear_design_mat_set[point], 
                                                    data_prior_set[point], 
                                                    model_prior_set[point])

        return Cm_p_set

    def model_posterior(self, design_mat, data_prior, model_prior):

        G = design_mat
        invCd = data_prior
        invCm = model_prior

        # No valid G
        if np.isnan(G[0,0]):
            Cm_p = np.zeros(shape=(1,1))+np.nan
            return Cm_p

        # G is valid.
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
        for point in point_set:

            # secular.
            resid_of_secular = self.resid_of_secular(linear_design_mat_set[point],
                                                data_vec_set[point], model_vec_set[point])

            if not np.isnan(resid_of_secular[0,0]):
                resid_of_secular_set[point] = ( np.mean(resid_of_secular[0:-1:2]),
                                                np.std (resid_of_secular[0:-1:2]),
                                                np.mean(resid_of_secular[1:-1:2]),
                                                np.std (resid_of_secular[1:-1:2]))

                #print(resid_of_secular_set[point])

            else:
                resid_of_secular_set[point] = (np.nan,np.nan,np.nan,np.nan)

            # tides.
            resid_of_tides = self.resid_of_tides(linear_design_mat_set[point],
                                                data_vec_set[point], model_vec_set[point])

            # Ouput the residual
            #if point == self.test_point:
                #print(resid_of_tides)
                #print(len(resid_of_tides))

            if not np.isnan(resid_of_tides[0,0]):
                # range and azimuth
                resid_of_tides_set[point] = ( np.mean(resid_of_tides[0:-1:2]),
                                                np.std (resid_of_tides[0:-1:2]),
                                                np.mean(resid_of_tides[1:-1:2]),
                                                np.std (resid_of_tides[1:-1:2]))

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

        #print("resid of tides")
        #print(np.hstack((d, pred, resid_of_tides)))
        
        #plt.figure()
        #plt.plot(resid_of_tides)
        #plt.savefig('./fig_sim/resid.png',format='png')
        #plt.close()
        
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
                    phaseE = self.velo_phase_to_dis_phase(t_vec[3+k*6+3], tide_name)

                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1], tide_name)
                    phaseN = self.velo_phase_to_dis_phase(t_vec[3+k*6+4], tide_name)

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

                    # ACF rotation
                    amp_cos_crf = a*np.sin(theta1) + c*np.cos(theta1)
                    amp_sin_crf = b*np.sin(theta1) + d*np.cos(theta1)

                    # ACF amplitude
                    amp_crf = (amp_cos_crf**2 + amp_sin_crf**2)**(1/2)
                    phase_crf = np.arctan2(amp_cos_crf, amp_sin_crf)


                    # Convert unit of phase (use days for Msf)
                    phase_alf = self.rad2deg(phase_alf)
                    phase_alf = self.wrapped_deg(phase_alf)
                    phase_alf = self.deg2day(phase_alf, tide_name)

                    phase_crf = self.rad2deg(phase_crf)
                    phase_crf = self.wrapped_deg(phase_crf)
                    phase_crf = self.deg2day(phase_crf, tide_name)


                    # Check if the value is valid not
                    ve_model = self.grid_set_velo[point][0]
                    vn_model = self.grid_set_velo[point][1]
                    v_model = (ve_model**2 + vn_model**2)**(1/2)


                    lon, lat = self.int5d_to_float(point)

                    thres_for_v = 0.4
                    thres_for_amp = 0.1

                    amp_full = (amp_alf**2 + amp_crf**2)**0.5

                    if v_model>thres_for_v and amp_alf>thres_for_amp:
                        if self.proj == "Rutford" and lat<-77.8:
                            pass

                        elif self.proj == "Evans" and ampU>0.5:
                            pass

                        else:
                            phase_alf = np.nan
                    else:
                        phase_alf = np.nan

                    if v_model > thres_for_v and amp_crf > thres_for_amp:
                        if self.proj == "Rutford" and lat<-77.8:
                            pass

                        elif self.proj == "Evans" and ampU>0.5:
                            pass

                        else:
                            phase_crf = np.nan
                    else:
                        phase_crf = np.nan


                    quant = {}
                    quant["Msf_along_flow_displacement_amplitude"] = amp_alf
                    quant["Msf_along_flow_displacement_phase"] = phase_alf
                    quant["Msf_cross_flow_displacement_amplitude"] = amp_crf
                    quant["Msf_cross_flow_displacement_phase"] = phase_crf
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
                    phaseE=self.velo_phase_to_dis_phase(t_vec[3+k*6+3],tide_name)

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
                    # estimated value > thres
                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq') :
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value, tide_name)
                            quant = self.rad2deg(phaseU)
                            quant = self.wrapped_deg(quant)

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

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                        # Find the phase
                        value = t_vec[3+k*6+5]
                        print("value: ", value)

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value, tide_name)
                            print("phaseU: ", phaseU)
                            print(stop)


                            quant = self.rad2deg(phaseU)
                            quant = self.wrapped_deg(quant)

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
