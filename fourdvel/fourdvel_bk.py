#!/usr/bin/env python3

# Author: Minyan Zhong
# Create time: June 2018

###

# All time is in the unit of day.

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from matplotlib import cm

import seaborn as sns

import datetime
from datetime import date

import multiprocessing

import time

from scipy import linalg

class fourdvel():

    def unit_vec(self, v1, v2=None):
        if v2:
            return np.asarray([v1,v2,np.sqrt(1-v1**2-v2**2)])
        else:
            return np.asarray([v1,np.sqrt(1-v1**2),0])

    def read_parameters(self):
        
        f = open('params.in')
        params = f.readlines()
        for param in params:
            
            try:    
                name,value = param.split()

                if name == 'grid_set_name':
                    self.grid_set_name = value
    
                if name == 'est_dict_name':
                    self.est_dict_name = value
    
                if name == 'est_output_name':
                    self.est_output_name = value
    
                if name == 'use_s1':
                    if value == 'True':
                        self.use_s1 = True
                    else:
                        self.use_s1 = False
    
                if name == 'use_csk':
                    if value == 'True':
                        self.use_csk = True
                    else:
                        self.use_csk = False 
    
                if name == 'csk_log':
                    self.csk_log = value
            except:
                pass
            
    def __init__(self):

        # Time origin
        self.t_origin = datetime.datetime(2018,1,1,0,0,0,0)

        # Tides
        self.tides = {}
        tides = self.tides
        tides['K2'] = 0.498
        tides['S2'] = 0.5
        tides['M2'] = 0.52
        tides['K1'] = 1
        tides['P1'] = 1.003
        tides['O1'] = 1.08
        tides['Mf'] = 13.66
        tides['Msf'] = 14.77
        tides['Mm'] = 27.55
        tides['Ssa'] = 182.62
        tides['Sa'] = 365.27
        
        # Modeling tides
        self.modeling_tides = ['M2','O1','Mf','Msf','Mm','K2','S2','K1','P1','Ssa','Sa']
        #self.modeling_tides = ['M2','O1','Mf','Msf','Mm']
        self.modeling_tides = ['M2','O1','Msf']

        self.n_tides = len(self.modeling_tides)

        self.periods = np.asarray([ tides[tide_name] for tide_name in self.modeling_tides ])

        self.omega = 2*np.pi/self.periods

        self.constants()

        # data
        self.csk_data = {}
        self.csk_tracks = range(22)

        for i in self.csk_tracks:
            self.csk_data[i] = []

        self.s1_data = {}
        self.s1_tracks = [37,52]
        
        for i in self.s1_tracks:
            self.s1_data[i] = []

        # grounding line file
        self.glfile = '/home/mzzhong/links/jokull-nobak-net/Ant_Plot/Data/GL_Points_Evans.txt'
        self.design_mat_folder = './design_mat'

        self.read_parameters()

    def constants(self):

        self.track_timefraction = {}
        track_timefraction = self.track_timefraction

        fid = open('csk_times.txt')
        csk_times = fid.readlines()
        fid.close()

        tracks = range(22)
        
        for track_num in tracks:
            track_timefraction[('csk',track_num)] = float(csk_times[track_num])

        t37 = datetime.time(6,26,45)
        track_timefraction[('s1',37)] = (t37.hour * 3600 + t37.minute*60 + t37.second)/(24*3600)
        t52 = datetime.time(7,7,30)
        track_timefraction[('s1',52)] = (t52.hour * 3600 + t52.minute*60 + t52.second)/(24*3600)

        #print(track_timefraction)

    def get_CSK_trackDates(self):

        import csv
        from CSK_Utils import CSK_Utils

        csk_data = self.csk_data

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
                
                tracks = csk.date2track(day=theDate, sate=sate)[sate]

                if direction == 'A':
                    track = [ i for i in tracks if i<=10 ]
                else:
                    track = [ i for i in tracks if i>=11 ]

                if track[0] in csk_data.keys():
                    csk_data[track[0]].append(theDate)
                else:
                    csk_data[track[0]] = [theDate]

                tot_frames = tot_frames + csk.numOfFrames[track[0]]

        
        print("number of product: ", tot_product)
        print("number of frames: ", tot_frames)


        for track_num in sorted(csk_data.keys()):
            csk_data[track_num].sort()
            #print(track_num)
            #print(csk_data[track_num])

        return 0

    def get_S1_trackDates(self):

        from S1_Utils import S1_Utils
        import glob

        s1_data = self.s1_data

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
                s1_data[track_num].append(theDate)

            s1_data[track_num] = list(set(s1_data[track_num]))
            s1_data[track_num].sort()

        #print(s1_data)

        return 0 

    def get_track_latlon(self):
        import gdal

        # currently only CSK

        redo = 0

        grid_set_pkl = self.grid_set_name + '.pkl'

        if os.path.exists(grid_set_pkl) and redo == 0:
            print('loading grid_set...')

            with open(grid_set_pkl,'rb') as f:
                self.grid_set = pickle.load(f)

        else:

            print('calculating grid_set...')

            grid_set = {}

            #satellites = ['csk','s1']
            satellites = ['csk']

            directory = {}
            tracklist = {}
            offset_id = {}

            directory['csk'] = '/net/kraken/nobak/mzzhong/CSK-Evans'
            tracklist['csk'] = range(22)
            offset_id['csk'] = 20180712

            directory['s1'] = '/net/jokull/nobak/mzzhong/S1-Evans'
            tracklist['s1'] = [37,52]
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
                    grid_lon = np.round(grid_lon * 1000)/1000
                    grid_lat = np.round(grid_lat * 1000)/1000
        
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
        
                        # Push into the grid_set, only when sate is csk.
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

    def build_G(self, point, tracks):

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

        if tracks is None:
            print('Please provide track info on this grid point')
            return

        #print(tracks)

        #print(point)
        #print(tracks)

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

        #print(offsetfields)
        #print(len(offsetfields))

        # Control the number of offsetfields
        n_offsets = len(offsetfields)
        #print('total number of offsetfield:', n_offsets)
        #n_offsets = 10

        ###############################################################

        ## Build the G matrix
        modeling_tides = self.modeling_tides
        n_tides = self.n_tides
        periods = self.periods
        omega = self.omega

        t_origin = self.t_origin.date()

        # Build up delta_td, delta_cos and delta_sin.
        delta_td = np.zeros(shape=(n_offsets,))
        delta_cos = np.zeros(shape=(n_offsets,n_tides))
        delta_sin = np.zeros(shape=(n_offsets,n_tides))
        
        for i in range(n_offsets):

            #print(offsetfields[i][4])

            t_a = (offsetfields[i][0] - t_origin).days + offsetfields[i][4]
            t_b = (offsetfields[i][1] - t_origin).days + offsetfields[i][4]

            delta_td[i] = (offsetfields[i][1] - offsetfields[i][0]).days
            
            for j in range(n_tides):
            
                delta_cos[i,j] = np.cos(omega[j]*t_b) - np.cos(omega[j]*t_a)
                delta_sin[i,j] = np.sin(omega[j]*t_b) - np.sin(omega[j]*t_a)

        n_rows = n_offsets * 2 # each offset corresponds to a vector.
        n_cols = 3 + n_tides * 6 # cosE, cosN, cosU and sinE, sinN, sinU.
        
        ## G formation.
        G = np.zeros(shape=(n_rows,n_cols))

        for i in range(n_offsets):
            vecs = [offsetfields[i][2],offsetfields[i][3]]

            # The two displacement components.
            for j in range(2):
                vector = np.asarray(vecs[j])

                # Row entries of the observation.
                row = np.zeros(shape=(n_cols,))

                # Secular component.
                row[0:3] = vector * delta_td[i]
                
                # Tidal components.
                for k in range(n_tides):
                    row[3*(2*k+1):3*(2*k+2)] = vector * delta_cos[i,k]
                    row[3*(2*k+2):3*(2*k+3)] = vector * delta_sin[i,k]

                # Put in into G.
                G[i*2+j,:] = row

        #print(G.shape)
        # end of building


        ########################################################

        ## Analyze G

        # model resolution matrix
        invG = np.linalg.pinv(G)
        model_res = np.matmul(invG, G)
        res_est = np.trace(model_res)/n_cols
        output1 = res_est

        # noise sensitivity matrix
        sensMat = np.linalg.pinv(np.matmul(np.transpose(G),G))
        
        # lumped_error
        error_lumped = np.sqrt(max(np.trace(sensMat),0))
        output2 = error_lumped
        
        # M2, O1, Msf
        # Msf cosine north error
        tide_num = 3
        E_off_cos = 1
        N_off_cos = 2
        U_off_cos = 3
        E_off_sin = 4
        N_off_sin = 5
        U_off_sin = 6

        ind = 2+6*(tide_num-1)+N_off_cos

        error_Msf_cos_N = sensMat[ind,ind]

        output3 = error_Msf_cos_N
        
        return output3

        # display
        print('total number of offsetfields: ',n_offsets)

        #showmat = sensMat

        #if lat == -76.59:
        #    self.G1 = G
        #    print('G1: ',self.G1)
        #    np.save('G1.npy',G)
        #if lat == -76.595:
        #    self.G2 = G
        #    print('G2: ',self.G2)
        #    np.save('G2.npy',G)
        #print('point: ',point)
        
        print('lumped error:', error_lumped)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        im = ax.imshow(showmat, cmap = cm.jet)
        
        fig.colorbar(im,orientation='horizontal',shrink=0.7)

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_cols))

        for it in range(len(modeling_tides)):
            ax.text(n_cols+1, 3+(it+1)*6-3,modeling_tides[it] + ': ' + str(periods[it]), fontsize=10)
            ax.text(n_cols+1, 3+(it+1)*6-2,str(3+it*6) + ' - ' + str(3+(it+1)*6-1),fontsize=10)

            ax.plot(np.asarray(range(n_cols)), (3+it*6-0.5) + np.zeros(shape=(n_cols,)),linestyle='--',color='k')
        
        plt.show()
        
        return 0

    def inversion(self,start,end,grid_est):

        grid_set = self.grid_set

        count = 0
        for grid in grid_set.keys():
            # Output the percentage of completeness.
            if count % 500 == 0 and start==0 and count<end:
                print(count/end)

            if count>=start and count<end:
                est_value = self.build_G(point=grid, tracks=grid_set[grid])
                grid_est[grid] = est_value
            count = count + 1

    def driver(self):

        self.preparation()

        # Calcuate design matrix and do estimations.
        redo = 1

        # est_dict pickle file.
        est_dict_pkl = self.est_dict_name + '.pkl'

        # The minimum number of tracks.
        min_tracks = 2

        if redo or not os.path.exists(est_dict_pkl):

            # Load the grid point set.
            grid_set = self.grid_set

            # Remove the bad grid points where not enough tracks are available.
            bad_keys=[]
            for key in grid_set.keys():
                if len(grid_set[key])<min_tracks:
                    bad_keys.append(key)

            for key in bad_keys:
                if key in grid_set:
                    del grid_set[key]

            # Count the total number of grid points.
            total_number = len(grid_set.keys())
            print(total_number)

            # Chop into multiple threads. 
            nthreads = 16
            
            mod = total_number % nthreads
            
            if mod > 0:
                chunk_size = (total_number - mod + nthreads) // nthreads
            else:
                chunk_size = total_number // nthreads

            divide = np.zeros(shape=(nthreads+1,))
            divide[0] = 0

            for it in range(1, nthreads+1):
                divide[it] = chunk_size * it
            divide[nthreads] = total_number

            print(divide)
            print(len(divide))

            # Multithreading starts here.
            func = self.inversion

            manager = multiprocessing.Manager()
            grid_est = manager.dict()

            jobs=[]
            for ip in range(nthreads):
                start = divide[ip]
                end = divide[ip+1]
                p=multiprocessing.Process(target=func, args=(start,end,grid_est,))
                jobs.append(p)
                p.start()

            for ip in range(nthreads):
                jobs[ip].join()

            est_dict = dict(grid_est)

            # Save the results.    
            with open(os.path.join(self.design_mat_folder, est_dict_pkl),'wb') as f:
                pickle.dump(est_dict,f)

        else:

            # Load the pre-computed results.
            with open(os.path.join(self.design_mat_folder,est_dict_pkl),'rb') as f:
                est_dict = pickle.load(f)

        # Show the estimation.
        self.show_est(est_dict)

    def show_est(self,show_dict):

        # Write to txt file.
        est_xyz = self.est_dict_name + '.xyz'
        f = open(os.path.join(self.design_mat_folder,est_xyz),'w')
        cap=100
        for key in sorted(show_dict.keys()):
            lon, lat = key
            value = show_dict[key]
            value = min(value,cap)
            if value == 0:
                value == 100

            if not np.isnan(value):
                f.write(str(lon)+' '+str(lat)+' '+str(min(value,cap))+'\n')

        f.close()

        return
     
        # Prepare the matrix.

        # First coordinate (width)
        lon0 = -84
        lon_interval = 0.02
        lon1 = -68
        lon_list = np.arange(lon0, lon1, lon_interval)

        # Second coordinate (length) 
        lat0 = -74.5
        lat_interval = -0.005
        lat1 = -77.5
        lat_list = np.arange(lat0, lat1, lat_interval)

        llon, llat = np.meshgrid(lon_list,lat_list)

        showmat = np.zeros(llon.shape)
        showmat[:] = np.nan

        #print(showmat.shape)

        points={}

        for key in show_dict.keys():
            lon,lat = key
            ind_x = int(round((lon-lon0)/lon_interval))
            ind_y = int(round((lat-lat0)/lat_interval))

            if not (ind_y,ind_x) in points.keys():
                points[(ind_y,ind_x)] = (lon,lat)
            else:
                print('Confliction!')
                print(points[(ind_y,ind_x)])
                print(key)
                print(ind_x,ind_y)
                print(stop)

            if ind_x>=0 and ind_x<len(lon_list) and ind_y>=0 and ind_y<len(lat_list):
                showmat[ind_y,ind_x] = show_dict[key]

        print(len(points.keys()))

        # Plot.
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)

        extent=[lon_list[0],lon_list[-1],lat_list[-1],lat_list[0]]

        f1 = ax.imshow(showmat,cmap=plt.cm.coolwarm,vmin=0,vmax=0.15, extent=extent, aspect='auto')
        #f1 = ax.imshow(showmat,cmap=plt.cm.jet,vmin=0,vmax=10)

        # Plot grounding line.
        f = open(self.glfile)
        data = f.readlines()
        print(len(data))

        #subdata = [pair for pair in data if float(pair.split()[0])>=lon0 and float(pair.split()[0])<=lon1 and float(pair.split()[1])<=lat0 and float(pair.split()[1])>=lat1 ]
        #print(len(subdata))
        
        #x = [pair.split()[0] for pair in subdata]
        #y = [pair.split()[1] for pair in subdata]

        x=[]
        y=[]
        for i in range(0,len(data),100):
            pair = data[i]
            lon = float(pair.split()[0])
            lat = float(pair.split()[1])
            x.append(lon)
            y.append(lat)
            
        ax.plot(x,y,'k.',label='grouding line')
        ax.legend(loc=3)

        ax.set_xlim([lon_list[0],lon_list[-1]])
        ax.set_ylim([lat_list[-1],lat_list[0]])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


        #ax.set_title('sensitivity of noise')
        cbar=fig.colorbar(f1,orientation='vertical',shrink=0.8,ticks=np.arange(0,0.201,0.05))
        cbar.set_label('max value in sensitivity matrix', fontsize=15)

        #fig.colorbar(f1,orientation='horizontal',shrink=0.8)
        fig.savefig('temp.png',format='png')
        
        print('Done')


    def preparation(self):

        # Get pre-defined grid points and the corresponding tracks and vectors.
        self.get_track_latlon()
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

def main():

    fourD = fourdvel()
    
    #fourD.get_track_latlon()
    
    #fourD.simulation()

    #fourD.driver()

    # To count the number of products.
    #fourD.get_CSK_trackDates()

if __name__=='__main__':
    main()
