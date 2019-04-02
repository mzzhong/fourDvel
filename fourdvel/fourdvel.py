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

class fourdvel(basics):

    def __init__(self):

        super(fourdvel,self).__init__()

        # Constants.
        self.satellite_constants()

        # Data.
        self.csk_data = {}
        self.csk_tracks = range(22)

        for i in self.csk_tracks:
            self.csk_data[i] = []

        self.s1_data = {}
        self.s1_tracks = [37,52,169,65,7]
        
        for i in self.s1_tracks:
            self.s1_data[i] = []

        # Grounding line file.
        self.glfile = '/home/mzzhong/links/jokull-nobak-net/Ant_Data/GL_Points_Evans.txt'
        self.design_mat_folder = './design_mat'

        self.read_parameters()

    def read_parameters(self):
        
        f = open('params.in')

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

            if name == 'test_id':
                self.test_id = value
                print('test_id',value)

            if name == 'test_mode':
                self.test_mode = int(value)
                print('test_mode',value)

            if name == 'grid_set_name':
                self.grid_set_name = value
                print('grid_set_name: ',value)

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

        return 0
            
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

        #print(track_timefraction)

        return 0

    def get_CSK_trackDates(self):

        import csv
        from CSK_Utils import CSK_Utils

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
                
                #print(row)
                sate = 'CSKS' + row[1][-1]
                acq_datefmt = row[5].split(' ')[0]
                direction = row[7][0]
                
                date_comp = [int(item) for item in acq_datefmt.split('-')]
                theDate = date(date_comp[0],date_comp[1],date_comp[2])
                #print(sate, date_comp, direction)

                if theDate >= csk_start and theDate < csk_end:
    
                    # Find the corresponding track.                
                    tracks = csk.date2track(day=theDate, sate=sate)[sate]
    
                    # Ascending or descending.                
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
    
        
        print("number of product: ", tot_product)
        print("number of frames: ", tot_frames)


        # Sort the tracks.
        for track_num in sorted(csk_data.keys()):
            csk_data[track_num].sort()
            #print(track_num)
            #print(csk_data[track_num])

        return 0

    def get_S1_trackDates(self):

        from S1_Utils import S1_Utils
        import glob

        s1_data = self.s1_data
        s1_start = self.s1_start
        s1_end = self.s1_end

        # Currently only track_37 and track_52 are available.
        s1 = S1_Utils()

        tracklist = self.s1_tracks

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

        #print(s1_data)

        return 0

    def get_grid_set_velo(self):

        self.grid_set_velo = None

        if hasattr(self, 'grid_set_velo_name'):
            dim = 3
            if dim == 3:
                grid_set_velo_pkl = self.grid_set_velo_name + '_3d' + '.pkl'
    
            if os.path.exists(grid_set_velo_pkl):
                print('Loading grid_set_velo...')
                with open(grid_set_velo_pkl,'rb') as f:
                    self.grid_set_velo = pickle.load(f)

    def get_tile_set(self):

        self.tile_set = None
        if hasattr(self,'tile_set_name'):
            tile_set_pkl = self.tile_set_name + '.pkl'
            if os.path.exists(tile_set_pkl):
                print('Loading tile_set ...')
                with open(tile_set_pkl,'rb') as f:
                    self.tile_set = pickle.load(f)


    def get_data_uncert(self):

        self.grid_set_data_uncert = None
        if hasattr(self,'grid_set_data_uncert_name'):
            grid_set_data_uncert_set_pkl = self.grid_set_data_uncert_name + '.pkl'
    
            if os.path.exists(grid_set_data_uncert_set_pkl):
                print('Loading data uncert set ...')
                with open(grid_set_data_uncert_set_pkl,'rb') as f:
                    self.grid_set_data_uncert = pickle.load(f)


    def get_grid_set(self):
        import gdal

        # Currently only CSK

        redo = 0
        grid_set_pkl = self.grid_set_name + '.pkl'

        # Step size
        self.lon_step = 0.02
        self.lat_step = 0.005

        if os.path.exists(grid_set_pkl) and redo == 0:
            print('Loading grid_set...')

            with open(grid_set_pkl,'rb') as f:
                self.grid_set = pickle.load(f)

            #print(self.grid_set[-77,-76.8])

        else:
            print('Calculating grid_set...')
            grid_set = {}

            satellites = ['csk','s1']
            #satellites = ['csk']

            directory = {}
            tracklist = {}
            offset_id = {}

            directory['csk'] = '/net/kraken/nobak/mzzhong/CSK-Evans'
            tracklist['csk'] = range(22)
            offset_id['csk'] = 20180712

            directory['s1'] = '/net/jokull/nobak/mzzhong/S1-Evans'
            tracklist['s1'] = [37, 52, 169, 65, 7]
            offset_id['s1'] = 20180703

            for sate in satellites:

                for track_num in tracklist[sate]:

                    print(sate,track_num)

                    if sate == 'csk':
                        trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(2)+'0')
                    else:
                        trackdir = os.path.join(directory[sate],'track_' + str(track_num))
 
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
        
                    #print(lon_list,len(lon_list),xsize)
                    #print(lat_list,len(lat_list),ysize)
        
                    grid_lon, grid_lat = np.meshgrid(lon_list, lat_list)
        
                    # rounding
                    grid_lon = self.round1000(grid_lon)
                    grid_lat = self.round1000(grid_lat)
        
                    # maskout the invalid
                    los = dataset.GetRasterBand(1).ReadAsArray()
                    azi = dataset.GetRasterBand(2).ReadAsArray()
        
                    #print(los)
                    #print(azi)
        
                    grid_lon[los == 0] = np.nan
                    grid_lat[los == 0] = np.nan
        
                    #fig = plt.figure(1)
                    #ax = fig.add_subplot(111)
                    #ax.imshow(grid_lat)
                    #plt.show()
        
                    grid_lon_1d = grid_lon.flatten()
                    grid_lat_1d = grid_lat.flatten()
        
        
                    # read the vectors
                    enu_gc_losfile = os.path.join(trackdir,'merged','geom_master','enu_gc_los_offset_' + str(offset_id[sate]) + '.rdr.vrt')
                    enu_gc_azifile = os.path.join(trackdir,'merged','geom_master','enu_gc_azi_offset_' + str(offset_id[sate]) + '.rdr.vrt')

                    try:
                        dataset = gdal.Open(enu_gc_losfile)
                    except:
                        raise Exception('geometry file not exist')

                    elos = dataset.GetRasterBand(1).ReadAsArray()
                    nlos = dataset.GetRasterBand(2).ReadAsArray()
                    ulos = dataset.GetRasterBand(3).ReadAsArray()
        
                    dataset = gdal.Open(enu_gc_azifile)
                    eazi = dataset.GetRasterBand(1).ReadAsArray()
                    nazi = dataset.GetRasterBand(2).ReadAsArray()
                    uazi = dataset.GetRasterBand(3).ReadAsArray()
        
        
                    #grid_lon_1d = grid_lon_1d[np.logical_not(np.isnan(grid_lon_1d))]
                    #grid_lat_1d = grid_lat_1d[np.logical_not(np.isnan(grid_lat_1d))]
        
                    #print(grid_lon_1d,len(grid_lon_1d))
                    #print(grid_lat_1d,len(grid_lat_1d))
        
                    for kk in range(len(grid_lon_1d)):
        
                        ii = kk // xsize
                        jj = kk - ii * xsize
        
                        if np.isnan(grid_lon[ii,jj]) or np.isnan(grid_lat[ii,jj]):
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
                    
            with open(grid_set_pkl,'wb') as f:
                pickle.dump(grid_set,f)

            self.grid_set = grid_set

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

    def preparation(self):

        # Get pre-defined grid points and the corresponding tracks and vectors.
        self.get_grid_set()

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
                max_delta=16
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
        
        design_mat_set = {}
        for point in point_set:
            offsetfields = offsetfields_set[point]
            design_mat_set[point] = self.build_G(offsetfields=offsetfields)

        return design_mat_set

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

        n_rows = n_offsets * 2 # Each offset corresponds to a vector.

        
        if horizontal == False:
            # E, N, U components.
            n_cols = 3 + n_modeling_tides * 6 # cosE, cosN, cosU and sinE, sinN, sinU.
        else:
            # Only the E, N components.
            n_cols = 3 + n_modeling_tides * 4 # cosE, cosN, sinE, sinN, Not finished below.
        
        ## G formation.
        G = np.zeros(shape=(n_rows,n_cols))

        for i in range(n_offsets):
            vecs = [offsetfields[i][2],offsetfields[i][3]]

            # The observation vectors.
            for j in range(2):
                vector = np.asarray(vecs[j])

                # Row entries of the observation.
                row = np.zeros(shape=(n_cols,))

                # Secular component.
                row[0:3] = vector * delta_td[i]
                
                # Tidal components.
                for k in range(n_modeling_tides):
                    row[3*(2*k+1):3*(2*k+2)] = vector * delta_cos[i,k]
                    row[3*(2*k+2):3*(2*k+3)] = vector * delta_sin[i,k]

                # Put them in into G.
                G[i*2+j,:] = row

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

                # cos term.
                coe1 = model_vec[3+k*6+t]

                # sin term.
                coe2 = model_vec[3+k*6+t+3]

                omega = 2*np.pi / tide_periods[tide_name]

                # Amplitide.
                amp = np.sqrt(coe1*coe1+coe2*coe2)*omega

                # Phase.
                phase = np.arctan2(coe2,-coe1)

                #if t == 1:
                #    print(tide_name,'North component')
                #    print('Two terms:\n',coe1,coe2)

                # Four quadrants.
                #if coe2>0 and -coe1>0:
                #    pass
                #elif coe2>0 and -coe1<0:
                #    phi = phi + np.pi
                #elif coe2<0 and -coe1<0:
                #    phi = phi - np.pi
                #elif coe2<0 and -coe1>0:
                #    pass

                #phi = phi + np.pi

                param_vec[3+k*6+t,0] = amp
                param_vec[3+k*6+t+3,0] = phase
        
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
                
                if horizontal==True and (j==2 or j==5):
                    inv_sigma[k,k] = inf_restrict

                if not tide_name in ['M2','O1'] and up_short_period and (j==2 or j==5):
                    inv_sigma[k,k] = inf_restrict
                    #print(tide_name, j)

                if not tide_name in ['Mf','Msf','Mm'] and horizontal_long_period and (j==0 or j==1 or j==3 or j==4):
                    inv_sigma[k,k] = inf_restrict
 
        #print(stop)

        invCm = np.square(inv_sigma)

        return invCm

    def model_posterior_set(self, point_set, design_mat_set, data_prior_set, model_prior_set):
        Cm_p_set = {}
        for point in point_set:

            #print(point)
            Cm_p_set[point] = self.model_posterior(design_mat_set[point], 
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
        #if np.linalg.cond(invCm_p) < 10**8:
            Cm_p = np.linalg.pinv(invCm_p)
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

    # Simple version.
    def param_estimation_simple(self, design_mat, data):

        G = design_mat
        d = data

        invG = np.linalg.pinv(G)
        model_vec = np.matmul(invG, d)

        return model_vec

    # Bayesian inversion. (set)
    def param_estimation_set(self, point_set, design_mat_set, data_vec_set,
                        data_prior_set, model_prior_set, model_posterior_set):

        model_vec_set = {}
        for point in point_set:
            model_vec_set[point] = self.param_estimation(design_mat_set[point],
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
    def resids_set(self, point_set, design_mat_set, data_vec_set, model_vec_set):

        resid_of_secular_set = {}
        resid_of_tides_set = {}

        # Only keep the mean and standard deviation due to disk space.
        for point in point_set:

            # secular.
            resid_of_secular = self.resid_of_secular(design_mat_set[point],
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
            resid_of_tides = self.resid_of_tides(design_mat_set[point],
                                                data_vec_set[point], model_vec_set[point])

            if point == self.test_point:
                print(resid_of_tides)
                print(len(resid_of_tides))

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

    def tide_vec_to_quantity(self, tide_vec, quant_name, point=None, state=None):

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

        # Msf along flow displacement amplitude.
        elif quant_name == 'Msf_along_flow_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    phaseE = self.velo_phase_to_dis_phase(t_vec[3+k*6+3], tide_name)

                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)

                else:
                    k=k+1

        # Msf across flow displacement amplitude.
        elif quant_name == 'Msf_across_flow_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Msf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    phaseE = self.velo_phase_to_dis_phase(t_vec[3+k*6+3], tide_name)

                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)

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
                    quant = quant.wrapped_deg(quant)

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

        # Mf East amp.
        elif quant_name == 'Mf_east_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    quant = ampE
                else:
                    k=k+1


        # Mf East phase.
        elif quant_name == 'Mf_east_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    phaseE=self.velo_phase_to_dis_phase(t_vec[3+k*6+3],tide_name)

                    quant = self.rad2deg(phaseE)
                    quant = quant.wrapped_deg(quant)

                else:
                    k=k+1

        # Mf North amp.
        elif quant_name == 'Mf_north_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = ampN
                else:
                    k=k+1

        # Mf North phase.
        elif quant_name == 'Mf_north_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'Mf':
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)
                    quant = self.rad2deg(phaseN)
                    quant = self.wrapped_deg(quant)

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

        # M2 East amp.
        elif quant_name == 'M2_east_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    quant = ampE
                else:
                    k=k+1


        # M2 East phase.
        elif quant_name == 'M2_east_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    phaseE=self.velo_phase_to_dis_phase(t_vec[3+k*6+3],tide_name)

                    quant = self.rad2deg(phaseE)
                    quant = quant.wrapped_deg(quant)

                else:
                    k=k+1

        # M2 North amp.
        elif quant_name == 'M2_north_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = ampN
                else:
                    k=k+1

        # M2 North phase.
        elif quant_name == 'M2_north_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)
                    quant = self.rad2deg(phaseN)
                    quant = self.wrapped_deg(quant)

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

        # O1 East amp.
        elif quant_name == 'O1_east_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampE = self.velo_amp_to_dis_amp(t_vec[3+k*6],tide_name)
                    quant = ampE
                else:
                    k=k+1


        # O1 East phase.
        elif quant_name == 'O1_east_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    phaseE=self.velo_phase_to_dis_phase(t_vec[3+k*6+3],tide_name)

                    quant = self.rad2deg(phaseE)
                    quant = quant.wrapped_deg(quant)

                else:
                    k=k+1

        # O1 North amp.
        elif quant_name == 'O1_north_displacement_amplitude':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampN = self.velo_amp_to_dis_amp(t_vec[3+k*6+1],tide_name)
                    quant = ampN
                else:
                    k=k+1

        # O1 North phase.
        elif quant_name == 'O1_north_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    phaseN = self.velo_amp_to_dis_amp(t_vec[3+k*6+4],tide_name)
                    quant = self.rad2deg(phaseN)
                    quant = self.wrapped_deg(quant)

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
        
        # O1 Up phase. (only on ice shelves)
        elif quant_name == 'O1_up_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'O1':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.1

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq') :
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value, tide_name)
                            quant = self.rad2deg(phaseU)
                            quant = self.wrapped_deg(quant)
                            quant = self.deg2minute(quant, tide_name)
                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)
                            quant = self.deg2minute(quant,tide_name)


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
        elif quant_name == 'M2_up_displacement_phase':
            k = 0
            for tide_name in modeling_tides:
                if tide_name == 'M2':
                    ampU = self.velo_amp_to_dis_amp(t_vec[3+k*6+2],tide_name)
                    thres = 0.3

                    if (self.grid_set_velo[point][2]>0 and ampU > thres) or (state=='uq'):
                        value = t_vec[3+k*6+5]

                        if state in [ 'true','est']:
                            phaseU=self.velo_phase_to_dis_phase(value, tide_name)
                            quant = self.rad2deg(phaseU)
                            quant = self.wrapped_deg(quant)
                            quant = self.deg2minute(quant, tide_name)
                        elif state in ['uq']:
                            quant = value
                            quant = self.rad2deg(quant)
                            quant = self.deg2minute(quant,tide_name)
                    
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
