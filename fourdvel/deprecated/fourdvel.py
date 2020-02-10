#!/usr/bin/env python3

# Minyan Zhong June 2018

# All time is in days

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

class fourdvel():

    def unit_vec(self, v1, v2=None):
        if v2:
            return np.asarray([v1,v2,np.sqrt(1-v1**2-v2**2)])
        else:
            return np.asarray([v1,np.sqrt(1-v1**2),0])

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

    def synthetics(self,point=None,tracks=None):

        # Tide periods.
        tides = self.tides

        # Rutford data.
        tidesRut = {}
        
        tidesRut['K2'] = [0.31, 163, 29.1, 99]
        tidesRut['S2'] = [0.363, 184, 101.6, 115]
        tidesRut['M2'] = [0.259, 177, 156.3, 70]
        tidesRut['K1'] = [19, 79, 49, 73]
        tidesRut['P1'] = [0.24, 77.0, 16.6, 64]
        tidesRut['O1'] = [0.264, 81.0, 43, 54]
        tidesRut['Mf'] = [2.54, 250.0, 2.9, 163]
        tidesRut['Msf'] = [7, 13.28, 18.8, 0.3, 164]
        tidesRut['Mm'] = [5.04, 253.0, 1.6, 63]
        tidesRut['Ssa'] = [62, 26.74, 256.0, 1.5, 179]
        tidesRut['Sa'] = [7, 19.18, 273.0, 0.2, 179]

        # Constants for ice flow model.
        A = 2.4e-24
        alpha = 0.04
        g = 9.81
        h = 1000
        n_g = 3
        rho = 900
        s_v = 0.6

        # Gravitional driving stress
        tau_d = rho*g*h*alpha

        # Basal drag
        tau_b = 0.8 * tau_d

        w = 25*1000 #m
        L = 150*1000 #m

        # End of constants

        # Beginning of simulations

        x = np.linspace(0,L,num=np.round(L/500))
        y = np.linspace(0,2*w,num=np.round(2*w/500))

        yy,xx = np.meshgrid(y,x)
        #print(xx.shape)

        print(tau_d)
        print(tau_d * w/h)
        print(tau_d * w/h * 0.2)

        v_ideal_center = 2*A*w/(n_g+1) * (tau_d * w / h * 0.2)**n_g
        print(v_ideal_center)
        print(stop)
        
        v_ideal = v_ideal_center * (1 - (1-yy/w)**(n_g+1))

        k_h=10**(-1*np.abs(np.log10(L)-0.8))
        gamma = (1 + np.tanh(k_h * (x-0.6*L)))/2

        Gamma = {}

        Gamma_const = v_ideal/v_ideal_center

        # Time of origin
        t_origin = self.t_origin

        t_axis = np.arange(-365,365,0.01)
        t_axis = np.arange(0,300,0.01)

        # Signal of every tide at a particular grid point.
        # Fix the grid point for now.
        ind_x = 200
        ind_y = 50

        x_loc = xx[ind_x,ind_y]
        y_loc = yy[ind_x,ind_y]

        sim_const = Gamma_const[ind_x,ind_y]
        sim = {}
        for key in tidesRut.keys():
            print(key)

            a_n = tidesRut[key][0]/100 # meter
            phi_n = tidesRut[key][1]/180 * np.pi # rad
            a_u = tidesRut[key][2]/100 * gamma[ind_x] # meter, modulated by the grounding line.
            phi_u = tidesRut[key][3]/180 * np.pi

            omega = 2*np.pi / tides[key] 

            sim[(key,'e')] = np.zeros(shape=t_axis.shape)

            sim[(key,'n')] = sim_const * a_n * (np.sin(omega*t_axis + phi_n) - np.sin(phi_n))

            sim[(key,'u')] = sim_const * a_u * (np.sin(omega*t_axis + phi_u) - np.sin(phi_u))

        # Add all tide signals together
        p_e = np.zeros(shape=t_axis.shape)
        p_n = np.zeros(shape=t_axis.shape)
        p_u = np.zeros(shape=t_axis.shape)

        for key in tidesRut.keys():
            p_e = p_e + sim[(key,'e')]
            p_n = p_n + sim[(key,'n')]
            p_u = p_u + sim[(key,'u')]

        # Full signals
        d_e = np.zeros(shape=t_axis.shape)

        d_n = s_v * x_loc/L * (-v_ideal[ind_x,ind_y]*t_axis + p_n)
        d_n_tides = s_v * x_loc/L * p_n

        d_u = s_v * (x_loc-L)/(10*L) * v_ideal[ind_x,ind_y] + p_u
        d_u_tides = p_u

        # Plotting.
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        
        # Choice 1
        #p1 = ax.imshow(v_ideal/v_ideal_center,cmap=plt.cm.coolwarm)
        p1 = ax.imshow(v_ideal,cmap=plt.cm.coolwarm)
        fig.colorbar(p1)

        # Choice 2
        #ax.plot(t_axis, d_n_tides)


        # velocity
        #v_e = np.zeros(shape=xx.shape)
        #v_n = np.z

        plt.show()

        return


    def get_CSK_trackDates(self):

        import csv
        from CSK_Utils import CSK_Utils

        csk_data = self.csk_data

        # Not all data are available, currently, so I read the files exported from E-GEOS. I will switch to real data
        file_folder = '/home/mzzhong/links/kraken-nobak-net/CSKData/data_20171116_20180630'
        #file_folder = '/home/mzzhong/links/kraken-nobak-net/CSKData/data_20171116_20171219'


        data_file = os.path.join(file_folder,'all.csv')

        csk = CSK_Utils()

        with open(data_file) as dataset:
            csv_reader = csv.reader(dataset, delimiter=';')
            line = 0
            for row in csv_reader:
                line = line + 1
                if line == 1:
                    continue
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

        if os.path.exists('grid_set.pkl') and redo == 0:
            print('loading grid_set...')

            with open('grid_set.pkl','rb') as f:
                self.grid_set = pickle.load(f)

        else:

            print('calculating grid_set...')

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
                    enu_gc_losfile = os.path.join(trackdir,'merged','geom_master','enu_gc_los_offset_' + str(offset_id[sate]) + '.rdr')
                    enu_gc_azifile = os.path.join(trackdir,'merged','geom_master','enu_gc_azi_offset_' + str(offset_id[sate]) + '.rdr')

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
        
                        info = (track_num,(elos[ii,jj],nlos[ii,jj],ulos[ii,jj]),(eazi[ii,jj],nazi[ii,jj],uazi[ii,jj]),sate)
        
                        if (grid_lon[ii,jj],grid_lat[ii,jj]) not in grid_set.keys():
                            grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])] = [info]
        
                        else:
                            grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])].append(info)
                    
                    #print(grid_set) 
                    #print(stop)
    
   
            with open('grid_set.pkl','wb') as f:
                pickle.dump(grid_set,f)

            self.grid_set = grid_set

    def coverage(self): 
        
        # statistics
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


        # make the figure
        colors = sns.color_palette('bright',10)
        
        symbol = ['r.','g.','b.','c.','k.','m.','y.','w.']

        for track_count in plotx.keys():
            print(track_count)
            ax.plot(plotx[track_count],ploty[track_count], color=colors[track_count-1], marker='.',markersize=0.3, linestyle='None', label=str(track_count) + ' Track: ' + str(vec_count[track_count]))
            #ax.plot(plotx[track_count],ploty[track_count],symbol[track_count-1], markersize=0.3, label=str(track_count) + ' Track: ' + str(vec_count[track_count]))


        ax.legend(markerscale=30)
        ax.set_title('coverage')
        fig.savefig('coverage.png',format='png')

    def build_G(self, point, tracks, grid_error):

        if tracks is None:
            print('Please provide track info on this grid point')
            return

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

        n_rows = n_offsets * 2 # each offset corresponds to a vector
        n_cols = 3 + n_tides * 6 # ENU of cos and ENU of sin
        
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
        #invG = np.linalg.pinv(G)
        #showmat = np.matmul(invG, G)

        # noise sensitivity matrix
        #showmat = np.linalg.pinv(np.matmul(np.transpose(G),G))
        #showmat = np.matmul(np.transpose(G),G)
        #showmat = G

        # model uncertainty matrix
        C_m = np.linalg.pinv(np.matmul(np.transpose(G),G))
        grid_error[point] = np.max(C_m)
        showmat = C_m
        #print(np.max(C_m))

        #time.sleep(5)
        return

        # display
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

    def inversion(self):

        redo=1        
        if redo:

            grid_set = self.grid_set

            manager = multiprocessing.Manager()
            grid_error = manager.dict()

            count = 0
            func = self.build_G
            jobs = []
            count_run = 0
            nproc = 30
            for grid in grid_set.keys():
                count = count + 1
                print(count)
                if len(grid_set[grid])>=3:
                    count_run = count_run + 1
                    p = multiprocessing.Process(target=func, args=(grid,grid_set[grid],grid_error,))
                    jobs.append(p)
                    p.start()

                    if count_run == nproc:
                        for ip in range(nproc):
                            jobs[ip].join()

                        count_run = 0
                        jobs = []
                
                    #self.build_G(point=grid, tracks=grid_set[grid])
                    #print(stop)

            #print(grid_error)
            error_dict = dict(grid_error)
            
            with open('error_dict.pkl','wb') as f:
                pickle.dump(error_dict,f)

        else:
            with open('error_dict.pkl','rb') as f:
                error_dict = pickle.load(f)

        # Show the error
        self.show_error(error_dict)

    def show_error(self,show_dict):
        
        # Prepare the matrix.
        lon0 = -84
        lon_interval = 0.02
        lon1 = -68

        lat0 = -74.5
        lat_interval = -0.005
        lat1 = -77.5

        lon_list = np.arange(lon0, lon1, lon_interval)
        lat_list = np.arange(lat0, lat1, lat_interval)

        llon, llat = np.meshgrid(lon_list,lat_list)

        showmat = np.zeros(llon.shape)
        showmat[:] = np.nan

        for key in show_dict.keys():
            x,y = key
            ind_x = int((x-lon0)/lon_interval)
            ind_y = int((y-lat0)/lat_interval)
            showmat[ind_x,ind_y] = show_dict[key]

        # Plot.
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        f1 = ax.imshow(showmat,cmap=plt.cm.jet)
        fig.colorbar(f1,orientation='horizontal',shrink=0.8)
        plt.show()
        
        print('Done')

    def simulation(self):

        # Different types of simulations.
        # 1. Continuous time series
        # 2. Same information as the real available offsetfields

        self.synthetics()
        return
        grid_set = self.grid_set
        for grid in grid_set.keys():
            if len(grid_set[grid]) == 6:
                self.synthetics(point=grid, tracks=grid_set[grid])
                print(stop)
            
    def preparation(self):

        # Get grid points and the corresponding tracks and vectors
        self.get_track_latlon()
        #self.coverage()


        # Get the available dates
        self.get_CSK_trackDates() 
        self.get_S1_trackDates() 
        
def main():

    fourD = fourdvel()
    
    fourD.preparation()
    
    #fourD.simulation()
    fourD.inversion()

if __name__=='__main__':
    main()


#    def get_grid_set(self):
#        import gdal
#
#        # Currently only CSK
#
#        redo = 0
#        grid_set_pkl = self.grid_set_name + '.pkl'
#
#        # Step size in geocoded files
#        # lon_step = 0.02 degree
#        # lat_step = 0.005 degree
#
#        self.lon_step = 0.02
#        self.lat_step = 0.005
#
#        if os.path.exists(grid_set_pkl) and redo == 0:
#            print('Loading grid_set...')
#
#            with open(grid_set_pkl,'rb') as f:
#                self.grid_set = pickle.load(f)
#
#            #print(self.grid_set[-77,-76.8])
#
#        else:
#            print('Calculating grid_set...')
#            grid_set = {}
#
#            satellites = ['csk','s1']
#            #satellites = ['csk']
#
#            directory = {}
#            tracklist = {}
#            offset_id = {}
#
#            directory['csk'] = '/net/kraken/nobak/mzzhong/CSK-Evans'
#            tracklist['csk'] = range(22)
#            offset_id['csk'] = 20180712
#
#            directory['s1'] = '/net/jokull/nobak/mzzhong/S1-Evans'
#
#            # update 20190702
#            tracklist['s1'] = [37, 52, 169, 65, 7, 50, 64]
#
#            offset_id['s1'] = 20180703
#
#            for sate in satellites:
#
#                for track_num in tracklist[sate]:
#
#                    print(sate,track_num)
#
#                    if sate == 'csk':
#                        trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(2)+'0')
#                    else:
#                        trackdir = os.path.join(directory[sate],'track_' + str(track_num))
# 
#                    gc_losfile = os.path.join(trackdir,'merged','geom_master','gc_los_offset_' + str(offset_id[sate]) + '.rdr')
#                    
#                    gc_losvrtfile = gc_losfile + '.vrt'
#                    dataset = gdal.Open(gc_losvrtfile)
#                    geoTransform = dataset.GetGeoTransform()
#                    
#                    lon0 = geoTransform[0]
#                    lon_interval = geoTransform[1]
#        
#                    lat0 = geoTransform[3]
#                    lat_interval = geoTransform[5]
#        
#                    xsize = dataset.RasterXSize
#                    ysize = dataset.RasterYSize
#        
#                    lon_list = np.linspace(lon0, lon0 + lon_interval*(xsize-1), xsize)
#                    lat_list = np.linspace(lat0, lat0 + lat_interval*(ysize-1), ysize)
#        
#                    #print(lon_list,len(lon_list),xsize)
#                    #print(lat_list,len(lat_list),ysize)
#        
#                    grid_lon, grid_lat = np.meshgrid(lon_list, lat_list)
#        
#                    # rounding
#                    grid_lon = self.round1000(grid_lon)
#                    grid_lat = self.round1000(grid_lat)
#        
#                    # maskout the invalid
#                    los = dataset.GetRasterBand(1).ReadAsArray()
#                    azi = dataset.GetRasterBand(2).ReadAsArray()
#        
#                    #print(los)
#                    #print(azi)
#        
#                    grid_lon[los == 0] = np.nan
#                    grid_lat[los == 0] = np.nan
#        
#                    #fig = plt.figure(1)
#                    #ax = fig.add_subplot(111)
#                    #ax.imshow(grid_lat)
#                    #plt.show()
#        
#                    grid_lon_1d = grid_lon.flatten()
#                    grid_lat_1d = grid_lat.flatten()
#        
#        
#                    # read the vectors
#                    enu_gc_losfile = os.path.join(trackdir,'merged','geom_master','enu_gc_los_offset_' + str(offset_id[sate]) + '.rdr.vrt')
#                    enu_gc_azifile = os.path.join(trackdir,'merged','geom_master','enu_gc_azi_offset_' + str(offset_id[sate]) + '.rdr.vrt')
#
#                    try:
#                        dataset = gdal.Open(enu_gc_losfile)
#                    except:
#                        raise Exception('geometry file not exist')
#
#                    elos = dataset.GetRasterBand(1).ReadAsArray()
#                    nlos = dataset.GetRasterBand(2).ReadAsArray()
#                    ulos = dataset.GetRasterBand(3).ReadAsArray()
#        
#                    dataset = gdal.Open(enu_gc_azifile)
#                    eazi = dataset.GetRasterBand(1).ReadAsArray()
#                    nazi = dataset.GetRasterBand(2).ReadAsArray()
#                    uazi = dataset.GetRasterBand(3).ReadAsArray()
#        
#        
#                    #grid_lon_1d = grid_lon_1d[np.logical_not(np.isnan(grid_lon_1d))]
#                    #grid_lat_1d = grid_lat_1d[np.logical_not(np.isnan(grid_lat_1d))]
#        
#                    #print(grid_lon_1d,len(grid_lon_1d))
#                    #print(grid_lat_1d,len(grid_lat_1d))
#        
#                    for kk in range(len(grid_lon_1d)):
#        
#                        ii = kk // xsize
#                        jj = kk - ii * xsize
#        
#                        if np.isnan(grid_lon[ii,jj]) or np.isnan(grid_lat[ii,jj]):
#                            continue
#
#                        # The element being pushed into the list.
#                        # 1. track number; 2. los (three vectors) 3. azi (three vectors) 4. satellite name.
#                        info = (track_num,(elos[ii,jj],nlos[ii,jj],ulos[ii,jj]),(eazi[ii,jj],nazi[ii,jj],uazi[ii,jj]),sate)
#        
#                        # Push into the grid_set, only add new grid when sate is csk.
#                        if (grid_lon[ii,jj],grid_lat[ii,jj]) not in grid_set.keys():
#                            if sate=='csk':
#                                grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])] = [info]
#                            else:
#                                pass
#                        else:
#                            grid_set[(grid_lon[ii,jj],grid_lat[ii,jj])].append(info)
#
#            print("Total number of grid points: ", len(grid_set))
#
#            print("Save to pickle file...")                    
#            with open(grid_set_pkl,'wb') as f:
#                pickle.dump(grid_set,f)
#
#            self.grid_set = grid_set
#
#        return 0

