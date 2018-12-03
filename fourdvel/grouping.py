#!/usr/bin/env python3

import os
import numpy as np

import gdal

import matplotlib.pyplot as plt
from matplotlib import cm


from fourdvel import fourdvel

import pickle

from Ant_data import Ant_data


class grouping(fourdvel):

    def __init__(self):
        super(grouping,self).__init__()
        self.get_grid_set()
        self.doub_diff_file ='/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/doub_diff.off'

    def rounding(self,x):
        return np.round(x*1000)/1000

    def match_velo_to_grid_set(self):

        grid_set = self.grid_set

        # Generate matched_velo.
        print('Matching velocity model to grids set')
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
            vel_lon = np.round(vel_lon * self.lon_re)/self.lon_re
            vel_lat = np.round(vel_lat * self.lat_re)/self.lat_re

            # Match velo to grid_set
            matched_velo = {}
            for i in range(vel_lon.shape[0]):
                #print(i)
                for j in range(vel_lat.shape[0]):
                   
                    if (vel_lon[i,j],vel_lat[i,j]) in grid_set.keys():
                        # only save valid values.
                        valid = True

                        # Nan value.
                        if (ve[i,j]==0 and vn[i,j]==0):
                            valid = False

                        # Holes.
                        if vel_lon[i,j] > -75 and vel_lon[i,j] < -74.5 and vel_lat[i,j]<-77 and vel_lat[i,j] > -77.2:
                            valid = False
                        
                        if valid:
                            matched_velo[(vel_lon[i,j], vel_lat[i,j])] = [ve[i,j],vn[i,j]]
 
            with open('./pickles/matched_velo.pkl','wb') as f:
                pickle.dump(matched_velo,f)

        else:
            with open('./pickles/matched_velo.pkl','rb') as f:
                matched_velo = pickle.load(f)

        self.matched_velo = matched_velo

        return 0

    def create_grid_set_velo_2d(self):

        print('Interpolating grid set velocity model...')

        grid_set_velo_name = self.grid_set_velo_name
        redo = 0
        if redo == 1:

            grid_set = self.grid_set
            matched_velo = self.matched_velo
    
            # Create the velocity value for each grid point.
            grid_set_velo_2d = {}
            # Available.
            for key in grid_set.keys():
                if key in matched_velo.keys():
                    # And the value is valid.
                    if matched_velo[key] != (0,0):
                        value = matched_velo[key]
                    # The value is invalid.
                    else:
                        value = (np.nan, np.nan)
    
                    grid_set_velo_2d[key] = value
    
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
                                new_lon = self.rounding(lon + self.lon_step * ix)
                                new_lat = self.rounding(lat + self.lat_step * iy)
    
                                if (new_lon, new_lat) in matched_velo.keys():
                                    grid_set_velo_2d[key] = matched_velo[(new_lon, new_lat)]
                                    found = True
    
            print('Number of points matched: ', len(matched_velo))
            print('Number of points in grid set: ', len(grid_set))
            print('Number of points in grid velo set: ',len(grid_set_velo_2d))
    
            print(grid_set_velo_2d[(-77,-76.7)])
    
            with open(grid_set_velo_name + '_2d'+'.pkl','wb') as f:
                pickle.dump(grid_set_velo_2d,f)

        else:
            with open(grid_set_velo_name + '_2d' + '.pkl','rb') as f:
                grid_set_velo_2d = pickle.load(f)


        self.grid_set_velo_2d = grid_set_velo_2d

        write_to_file = True
        if write_to_file:
            xyz_file = '/home/mzzhong/insarRoutines/estimations/grid_set_velo_2d_speed.xyz'
            f = open(xyz_file,'w')
            for key in sorted(grid_set_velo_2d.keys()):
                lon, lat = key
                value = np.sqrt(grid_set_velo_2d[key][0]**2 + grid_set_velo_2d[key][1]**2)
                if not np.isnan(value) and value>0:
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()
        return 0

    def add_verti(self):

        from dense_offset import dense_offset
        from scipy.signal import  medfilt


        print('Add vertical component to grid set model...')

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
        doub_diff_map[np.isnan(doub_diff_map)] = 0
        #doub_diff_map[doub_diff_map==0]=np.nan

        maxval = 0.4
        doub_diff_map[doub_diff_map<0] = 0
        doub_diff_map[doub_diff_map>maxval] = maxval
        doub_diff_map = doub_diff_map/maxval

        # Remove noise.
        p1 = (160,0)
        p2 = (550,800)
        k = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1] - p1[0]*k

        for y in range(doub_diff_map.shape[0]):
            for x in range(doub_diff_map.shape[1]):
                if y == np.round(k*x+b):
                    doub_diff_map[y,x] = 1
                if y >= np.round(k*x+b):
                    doub_diff_map[y,x] = 0

        p1 = (0,800)
        p2 = (700,0)
        k = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1] - p1[0]*k

        for y in range(doub_diff_map.shape[0]):
            for x in range(doub_diff_map.shape[1]):
                if y == np.round(k*x+b):
                    doub_diff_map[y,x] = 1
                if y <= np.round(k*x+b):
                    doub_diff_map[y,x] = 0

        c = (550,420)
        r = 70
        
        for y in range(doub_diff_map.shape[0]):
            for x in range(doub_diff_map.shape[1]):
                if r == np.round(np.sqrt((x-c[0])**2 + (y-c[1])**2)):
                    doub_diff_map[y,x] = 1
                if r >= np.round(np.sqrt((x-c[0])**2 + (y-c[1])**2)):
                    doub_diff_map[y,x] = 0


        # Median filter
        doub_diff_map = medfilt(doub_diff_map, kernel_size = 7)
       
        # Plot it.
        #fig = plt.figure(figsize=(10,10))
        #ax = fig.add_subplot(111)
        #im = ax.imshow(doub_diff_map)
        #fig.colorbar(im)
        #fig.savefig('./fig_sim/double_diff.png',format='png')

        ######## Provide the third component.
        grid_set_velo_2d = self.grid_set_velo_2d
        grid_set_velo_name = self.grid_set_velo_name

        print(len(grid_set_velo_2d))

        test_point = (-76, -76.8)
        all_points = grid_set_velo_2d.keys()
        grid_set_velo_3d = {}
        for point in all_points:
            ind_x, ind_y = offset.point_index(point)

            #if point == test_point:
            #    print(ind_x,ind_y)
            #    print(grid_set_velo_2d[point])
            #    print(doub_diff_map[ind_y, ind_x])
            #    print(doub_diff_map.shape)
            #    print(type(doub_diff_map))

            velo_2d = grid_set_velo_2d[point]
            if ind_x is not None and ind_y is not None:
                velo_up = [doub_diff_map[ind_y, ind_x]]
            else:
                velo_up = [0]
            velo_3d = velo_2d + velo_up

            grid_set_velo_3d[point] = velo_3d

            #print(grid_set_velo_3d[point])
 

        print(grid_set_velo_3d[test_point])
        print(len(grid_set_velo_3d))

        with open(grid_set_velo_name + '_3d' + '.pkl', 'wb') as f:
            pickle.dump(grid_set_velo_3d,f)

        print('Done with 3d velocity fields')

        ##################
        write_to_file = True
        if write_to_file:
            xyz_file = '/home/mzzhong/insarRoutines/estimations/grid_set_velo_3d_verti.xyz'
            f = open(xyz_file,'w')
            for key in sorted(grid_set_velo_3d.keys()):
                lon, lat = key
                value = np.sqrt(grid_set_velo_3d[key][2]**2)
                if not np.isnan(value) and value>0:
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()

        return

    def velo_model(self):

        # 2D
        self.match_velo_to_grid_set()
        self.create_grid_set_velo_2d()

        # 3D
        self.add_verti()


    ##############################################

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

    def define_shelves(self):

        grid_set = self.grid_set
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
        vel_lon  = np.round(vel_lon * self.lon_re) / self.lon_re
        vel_lat = np.round(vel_lat* self.lat_re)/self.lat_re

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

    ##############################################

    def grid_tiles(self):

        grid_set = self.grid_set

        # Evans bounding box.
        west = self.round1000(-85)
        east = self.round1000(-69)
        north = self.round1000(-74.2)
        south = self.round1000(-77.6)

        tile_lon_size = 1
        tile_lat_size = 0.2

        looks = 1
        tile_lon_num = np.round(self.lon_re * tile_lon_size)
        tile_lat_num = np.round(self.lat_re * tile_lat_size)

        sub_lon_list = np.arange(tile_lon_num) / tile_lon_num * tile_lon_size
        sub_lat_list = np.arange(tile_lat_num) / tile_lat_num * tile_lat_size
    
        count = 0
        tile_set = {}
        
        for tile_lon in self.round1000(np.arange(west, east,tile_lon_size)):
            
            for tile_lat in self.round1000(np.arange(south, north,tile_lat_size)):

                location = (tile_lon,tile_lat)
                tile_set[location] = []

                tile_west = self.round1000(tile_lon)
                tile_east = self.round1000(tile_lon+ tile_lon_size)

                tile_south = self.round1000(tile_lat)
                tile_north = self.round1000(tile_lat + tile_lat_size)

                lon_list = self.round1000(tile_west + sub_lon_list)
                lat_list = self.round1000(tile_south + sub_lat_list)

                for lon in lon_list:
                    for lat in lat_list:
                        
                        point = (lon, lat)
                        if point in grid_set.keys():
                            count = count + 1
                            tile_set[location].append(point)

        print(count)
        print(len(grid_set))
        print(len(tile_set))

        # Remove empty tiles.
        empty_tiles = []
        for tile in tile_set.keys():
            if len(tile_set[tile])==0:
                empty_tiles.append(tile)

        for tile in empty_tiles:
            tile_set.pop(tile)

        print(len(tile_set))
        with open('./pickles/tile_set_csk.pkl','wb') as f:
            pickle.dump(tile_set,f)

def main():
    
    group = grouping()

    # Generate constant velocity.
    # First two component.
    group.velo_model()
    
    # Find tiles.
    #group.grid_tiles()

 
if __name__=='__main__':
    main()
