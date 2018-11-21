#!/usr/bin/env python3

import os
import numpy as np

import gdal

import matplotlib.pyplot as plt
from matplotlib import cm


from fourdvel import fourdvel

import pickle

from Ant_data import Ant_data


lon_re = 50
lat_re = 200

lon_step = 1/lon_re
lat_step = 1/lat_re

class grouping(fourdvel):

    def __init__(self):
        super(grouping,self).__init__()

        self.doub_diff_file ='/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/doub_diff.off'

        self.preparation()

    def rounding(self,x):
        return np.round(x*1000)/1000

    def match_velo_to_grid_set(self):

        grid_set = self.grid_set

        # Generate matched_velo
        redo = 0

        if redo == 1:
            velo_dir = '/net/jokull/nobak/mzzhong/Ant_Plot/Data/velocity_models'
            npz_filebasename = 'AntVelo_v1.npz'

            npzfile = np.load(os.path.join(velo_dir, npz_filebasename))

            vel_lon = npzfile['vel_lon']
            vel_lat = npzfile['vel_lat']
            ve = npzfile['ve']
            vn = npzfile['vn']
            v_comb = npzfile['v_comb']

            # Rounding latlon to grid points.
            vel_lon = np.round(vel_lon * lon_re)/lon_re
            vel_lat = np.round(vel_lat * lat_re)/lat_re

            # Match velo to grid_set
            matched_velo = {}
            for i in range(vel_lon.shape[0]):
                #print(i)
                for j in range(vel_lat.shape[0]):
                   
                    if (vel_lon[i,j],vel_lat[i,j]) in grid_set.keys():
                        matched_velo[(vel_lon[i,j], vel_lat[i,j])] = (ve[i,j],vn[i,j])
                        #if vel_lon[i,j] > -77.1 and vel_lon[i,j] < -76.9 and vel_lat[i,j]>-76.8 and vel_lat[i,j]<-76.6:
                        #if vel_lon[i,j] == -77 and vel_lat[i,j] == -76.7:

                            #print((vel_lon[i,j], vel_lat[i,j], ve[i,j], vn[i,j]))
 

            with open('./pickles/matched_velo.pkl','wb') as f:
                pickle.dump(matched_velo,f)

        else:
            with open('./pickles/matched_velo.pkl','rb') as f:
                matched_velo = pickle.load(f)

        self.matched_velo = matched_velo

        return 0

    def create_grid_set_velo(self):

        grid_set = self.grid_set
        matched_velo = self.matched_velo

        # Create the velocity value for each grid point.
        grid_set_velo = {}
        # Available.
        for key in grid_set.keys():
            if key in matched_velo.keys():
                # And the value is valid.
                if matched_velo[key] != (0,0):
                    value = matched_velo[key]
                # The value is invalid.
                else:
                    value = (np.nan, np.nan)

                grid_set_velo[key] = value

        # Not Available.
        for key in grid_set.keys():
            if not (key in matched_velo.keys()):
                # Seach for the nearest.
                lon, lat = key
                dist = 0
                found = False
                while not found:
                    dist = dist + 1
                    for ix in range(-dist, dist+1):
                        for iy in range(-dist, dist+1):
                            new_lon = self.rounding(lon + lon_step * ix)
                            new_lat = self.rounding(lat + lat_step * iy)

                            if (new_lon, new_lat) in matched_velo.keys():
                                grid_set_velo[key] = matched_velo[(new_lon, new_lat)]
                                found = True

        print('Number of points matched: ', len(matched_velo))
        print('Number of points in grid set: ', len(grid_set))
        print('Number of points in grid velo set: ',len(grid_set_velo))

        print(grid_set_velo[(-77,-76.7)])

        name = './pickles/grid_set_csk_point_velo'
        with open(name+'.pkl','wb') as f:
            pickle.dump(grid_set_velo,f)

        return 0

    def process_velo(self):

        self.match_velo_to_grid_set()
        self.create_grid_set_velo()

    def coloring_dfs(self, x, y, dep):

        x = np.round(x*1000)/1000
        y = np.round(y*1000)/1000
        print(x,y,dep)
        
        if ((x,y) in self.full_set.keys()) and (not (x, y) in self.colored_set.keys()):
            self.colored_set[(x,y)] = self.full_set[(x,y)]
        else:
            return 0

        self.coloring(x-self.lon_step, y, dep+1)
        self.coloring(x+self.lon_step, y, dep+1)
        self.coloring(x, y-self.lat_step, dep+1)
        self.coloring(x, y+self.lat_step, dep+1)

        return 0

    def fill_in_holes(self, quene):

        new_quene = quene

        for element in quene:

            head_x = element[0]
            head_y = element[1]

            for direction in range(4):
                if direction==0:
                    next_x = head_x - self.lon_step
                    next_y = head_y
                elif direction==1:
                    next_x = head_x + self.lon_step
                    next_y = head_y
                elif direction==2:
                    next_x = head_x
                    next_y = head_y - self.lat_step
                elif direction==3:
                    next_x = head_x
                    next_y = head_y + self.lat_step
            
                next_x = np.round(next_x*1000)/1000
                next_y = np.round(next_y*1000)/1000
        
                if not (next_x, next_y) in quene:
                    new_quene.append((next_x,next_y))
 
        return new_quene

    def coloring_bfs(self, point_set, x, y):

        redo = 0
        if os.path.exists('./pickles/quene.pkl') and redo == 0:
            with open('./pickles/quene.pkl','rb') as f:
                quene = pickle.load(f)
        else:
            x = np.round(x*1000)/1000
            y = np.round(y*1000)/1000
     
            quene = []
            quene.append((x,y))
    
            head = -1
            while head < len(quene)-1:
    
                print(len(quene))
    
                head = head + 1
                head_x = quene[head][0]
                head_y = quene[head][1]
    
                for direction in range(4):
                    if direction==0:
                        next_x = head_x - self.lon_step
                        next_y = head_y
                    elif direction==1:
                        next_x = head_x + self.lon_step
                        next_y = head_y
                    elif direction==2:
                        next_x = head_x
                        next_y = head_y - self.lat_step
                    elif direction==3:
                        next_x = head_x
                        next_y = head_y + self.lat_step
                
                    next_x = np.round(next_x*1000)/1000
                    next_y = np.round(next_y*1000)/1000
            
                    if ((next_x,next_y) in point_set) and (not (next_x, next_y) in quene):
                        quene.append((next_x,next_y))

            # Fill in the holes.
            #quene = self.fill_in_holes(quene)
            
            with open('./pickles/quene.pkl','wb') as f:
                pickle.dump(quene,f)

        return quene

    def shelves(self):

        from dense_offset import dense_offset

        # Load the double difference data. 
        stack = 'tops'
        workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
        track_num = 37
        name = 'track_' + str(track_num)
        runid = 20180703

        offset = dense_offset(stack=stack, workdir=workdir)
        offset.initiate(trackname = name, runid=runid)

        doub_diff_file = self.doub_diff_file

        # Read in the double difference file.
        dataset = gdal.Open(doub_diff_file)
        doub_diff_map = dataset.GetRasterBand(1).ReadAsArray()

        # Remove the invalid.
        doub_diff_map[doub_diff_map==0]=np.nan


        # Find the ice shelves.
        grid_set = self.grid_set
        thres = 0.1
        shelves = {}

        shelves_lon = []
        shelves_lat = []
        for grid in grid_set.keys():
            point = grid
            lon, lat = point
            ind_x, ind_y = offset.point_index(point)

            if ind_x is not None and ind_y is not None and lon>-79:
                doub_diff_value = doub_diff_map[ind_y,ind_x]
                
                # It is on ice shelves
                if doub_diff_value >= thres:
                    shelves[grid] = grid_set[grid]
                    shelves_lon.append(point[0])
                    shelves_lat.append(point[1])
                    
        print(len(grid_set))
        print(len(shelves))

        # Coloring.
        start_lon = -75
        start_lat = -77 
        new_shelves_keys = self.coloring_bfs(shelves.keys(),start_lon, start_lat)

        print(len(shelves.keys()))
        print(len(new_shelves_keys))

        # Update shelves from coloring.
        new_shelves = {}
        new_shelves_lon = []
        new_shelves_lat = []

        for point in new_shelves_keys:
            new_shelves[point] = grid_set[point]
            new_shelves_lon.append(point[0])
            new_shelves_lat.append(point[1])

        # Update ground.
        new_ground_keys = grid_set.keys()-new_shelves_keys

        new_ground = {}
        new_ground_lon = []
        new_ground_lat = []

        for point in new_ground_keys:
            new_ground[point] = grid_set[point]
            new_ground_lon.append(point[0])
            new_ground_lat.append(point[1])


        self.shelves = new_shelves
        self.shelves_lon = new_shelves_lon
        self.shelves_lat = new_shelves_lat

        self.ground = new_ground
        self.ground_lon = new_ground_lon
        self.ground_lat = new_ground_lat

        #self.ax = self.fig.add_subplot(111)
        #im = ax.imshow(doub_diff_map,cmap = cm.coolwarm)
        #fig.colorbar(im)
        #self.ax.plot(new_shelves_lon,new_shelves_lat,'b.')
        #self.ax.plot(new_ground_lon,new_ground_lat,'r.')
 
    def streams(self):

        grid_set = self.grid_set

        ground = self.ground

        shelves = self.shelves
        shelves_lon = self.shelves_lon
        shelves_lat = self.shelves_lat

        Ant = Ant_data()

        # Obtain the data.
        vel, vel_lon, vel_lat = Ant.get_veloData()

        # Rounding coordinates.
        vel_lon  = np.round(vel_lon * lon_re) / lon_re
        vel_lat = np.round(vel_lat* lat_re)/lat_re

        #print(vel_lon)
        #print(vel_lat)

        # Only moving ice.
        thres = 0.1 # m/d
        # Form the set
        moving_ice_set = []
        for i in range(vel.shape[0]):
            for j in range(vel.shape[1]):
                if vel[i,j]>thres:
                    moving_ice_set.append((vel_lon[i,j],vel_lat[i,j]))

        moving_ice_set = set(moving_ice_set)

        # Moving points in Evans.
        streams_keys = moving_ice_set & ground.keys()
        print(len(streams_keys))

        streams = {}
        streams_lon = []
        streams_lat = []
        for point in streams_keys:
            streams[point] = grid_set[point]
            streams_lon.append(point[0])
            streams_lat.append(point[1])

        print(len(streams_lon))

        self.streams = streams

        fig = plt.figure(2,figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.plot(streams_lon, streams_lat, 'r.')
        ax.plot(shelves_lon, shelves_lat, 'b.')
    
        fig.savefig('./fig_sim/streams.png',format='png')

def main():
    
    group = grouping()

    group.process_velo()

    # Using double difference. 
    #group.shelves()

    # Using velocity model.
    #group.streams()
 
if __name__=='__main__':
    main()
