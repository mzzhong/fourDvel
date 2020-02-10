#!/usr/bin/env python3

import os
import numpy as np

import gdal

import matplotlib.pyplot as plt
from matplotlib import cm

from fourdvel import fourdvel

import pickle

#from Ant_data import Ant_data

# Preparing references for synthetic tests and inversion

class grouping(fourdvel):

    def __init__(self):
        super(grouping,self).__init__()

        # Only need grid set
        self.get_grid_set_v2()

        # for Evans
        self.doub_diff_file ='/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/doub_diff.off'

        # for Ruford
        self.shelf_grid_points = "/net/kraken/bak/mzzhong/Ant_Data/GroundingLines/bedmap2_shelf_latlon.xyz"

    def rounding(self,x):
        return np.round(x*1000)/1000

    def match_velo_v2_to_grid_set(self):

        grid_set = self.grid_set

        # Temporary file
        filename = "./pickles/matched_velo_v2_" + str(self.resolution) +".pkl"

        # Generate matched_velo.
        print('Matching velocity model (v2) to grids set')
        
        redo = 0

        if not os.path.exists(filename) or redo == 1:
            velo_dir = '/net/kraken/bak/mzzhong/Ant_Data/velocity_models'
            npz_filebasename = 'AntVelo_v2.npz'

            npzfile = np.load(os.path.join(velo_dir, npz_filebasename))

            vel_lon = npzfile['vel_lon']
            vel_lat = npzfile['vel_lat']
            ve = npzfile['ve']
            vn = npzfile['vn']
            v_comb = npzfile['v_comb']

            # Convert lon to (-180,180)
            inds = vel_lon > 180
            vel_lon[inds] = vel_lon[inds] - 360


            # Rounding latlon to grid points.
            # Floating grid points
            vel_lon = self.round_to_grid_points(vel_lon, self.lon_re)
            vel_lat = self.round_to_grid_points(vel_lat, self.lat_re)

            # CONVERT TO INTEGER!
            vel_lon = self.round_int_5dec(vel_lon)
            vel_lat = self.round_int_5dec(vel_lat)

            print(vel_lon)
            print(vel_lat)

            #unique_lons = np.unique(vel_lon[(vel_lon>-80) & (vel_lon<-70) & (vel_lat>-77.5) & (vel_lat<-74)])
            #print(unique_lons)
            #print(unique_lons.shape)

            #unique_lats = np.unique(vel_lat[(vel_lon>-80) & (vel_lon<-70) & (vel_lat>-77.5) & (vel_lat<-74)])
            #print(unique_lats)
            #print(unique_lats.shape)

            #print(vel_lon)
            #print(vel_lon.shape)
            #print(vel_lat.shape)

            # Match velo to grid_set
            matched_velo = {}
            count = 0
            # Loop through all grid points in the data
            count_out=0
            for i in range(vel_lon.shape[0]):
                count_out+=1
                if count_out%1000==0:
                    print(count_out, vel_lon.shape[0])
                
                for j in range(vel_lon.shape[0]):

                    #print((vel_lon[i,j],vel_lat[i,j]))
                   
                    if (vel_lon[i,j],vel_lat[i,j]) in grid_set.keys():

                        count = count + 1

                        valid = True

                        # Remove all zero value (invalid)
                        if ve[i,j] == 0 and vn[i,j] == 0:
                            valid = False

                        # Save the two-component velocity.
                        if valid:
                            matched_velo[(vel_lon[i,j], vel_lat[i,j])] = [ve[i,j],vn[i,j]]

            print('Matched point count: ',count)
 
            with open(filename,'wb') as f:
                pickle.dump(matched_velo,f)

        else:
            with open(filename,'rb') as f:
                matched_velo = pickle.load(f)

        self.matched_velo = matched_velo
        print("length of mached velo: ", len(matched_velo))
        print("length of grid_set: ", len(grid_set))

        return 0

    def create_grid_set_velo_2d(self):

        print('Interpolating grid set velocity model...')

        grid_set_velo_name = self.grid_set_velo_name + '_' + str(self.resolution)

        pklname = grid_set_velo_name + '_2d'+'.pkl'

        redo = 0

        if not os.path.exists(pklname) or redo == 1:

            grid_set = self.grid_set
            matched_velo = self.matched_velo

            print('Number of points in grid set: ', len(grid_set))
            
            print('Number of points matched: ', len(matched_velo))
   
            # Create the velocity value for each grid point.
            grid_set_velo_2d = {}

            # Available.
            count = 0
            for key in grid_set.keys():
                count = count + 1
                if count % 10000 == 0:
                    print(count, len(grid_set))
 
                if key in matched_velo.keys():
                    grid_set_velo_2d[key] = matched_velo[key]

            print('Number of matched valid points being put into grid_set_velo_2d before interpolation: ',len(grid_set_velo_2d))

            print("Start to interpolate...") 
            # Interpolation for points not being matched.
            count = 0
            for key in grid_set.keys():
                count = count + 1
                if count % 10000 == 0:
                    print(count, len(grid_set))

                if not (key in matched_velo.keys()):
                    print("unmatched point: ", key)
                    # Seach for the nearest.
                    lon, lat = key
                    dist = 0
                    found = False
                    while not found:
                        dist = dist + 1
                        for ix in range(-dist, dist+1):
                            for iy in range(-dist, dist+1):
                                new_lon = lon + self.lon_step_int * ix
                                new_lat = lat + self.lat_step_int * iy
    
                                if (new_lon, new_lat) in matched_velo.keys() and found == False:
                                    grid_set_velo_2d[key] = matched_velo[(new_lon, new_lat)]
                                    found = True
                                    print("matched velo: ", matched_velo[(new_lon, new_lat)])
    

            # For test 
            #key = (-8100000,-7900000)
            #print('velocity at: ',key, grid_set_velo_2d[key])
   
            # Save it! 
            with open(grid_set_velo_name + '_2d'+'.pkl','wb') as f:
                pickle.dump(grid_set_velo_2d,f)

        else:
            with open(grid_set_velo_name + '_2d' + '.pkl','rb') as f:
                grid_set_velo_2d = pickle.load(f)

        self.grid_set_velo_2d = grid_set_velo_2d

        print('Number of points in grid_set_velo_2d (after interpolation): ',len(grid_set_velo_2d))
        print('Number of points in grid_set: ',len(self.grid_set))

        write_to_file = True
        if write_to_file:
            xyz_file = '/home/mzzhong/insarRoutines/estimations/grid_set_velo_2d_speed.xyz'
            f = open(xyz_file,'w')

            count = 0
            for key in sorted(grid_set_velo_2d.keys()):
                count = count + 1

                lon, lat = key

                # Only output the speed.
                value = np.sqrt(grid_set_velo_2d[key][0]**2 + grid_set_velo_2d[key][1]**2)
                if not np.isnan(value) and value>0:
                    f.write(str(self.int5d_to_float(lon))+' '+str(self.int5d_to_float(lat))+' '+str(value)+'\n')
                else:
                    pass
                    #print(key, grid_set_velo_2d[key], value)

        print(count)

        f.close()
        return 0

    def add_verti_evans(self):

        from dense_offset import dense_offset
        from scipy.signal import  medfilt

        print('Add vertical component to grid set model...')

        # Load the double difference data. 
        stack = 'tops'
        workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
        track_num = 37
        name = 'track_' + str(track_num)
        runid = 20180703

        offset = dense_offset(stack=stack, workdir=workdir, runid=runid)
        offset.initiate(trackname = name)

        doub_diff_file = self.doub_diff_file

        # Read in the double difference file.
        dataset = gdal.Open(doub_diff_file)
        doub_diff_map = dataset.GetRasterBand(1).ReadAsArray()


        # Remove the invalid.
        doub_diff_map[np.isnan(doub_diff_map)] = 0
        #doub_diff_map[doub_diff_map==0]=np.nan

        # Set the maxvalue, and do normalization.
        maxval = 0.4
        doub_diff_map[doub_diff_map<0] = 0
        doub_diff_map[doub_diff_map>maxval] = maxval
        doub_diff_map = doub_diff_map/maxval

        # Remove noise manually.
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

        grid_set_velo_name = self.grid_set_velo_name + '_' + str(self.resolution)

        print(len(grid_set_velo_2d))

        test_point = (-7600000, -7680000)
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

                # Only save values larger than zero.
                if not np.isnan(value) and value>0:
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()

        return

    def add_verti_rutford(self):

        print('Add vertical component to grid set model...')

        ###### Load in the ice shelf map with 100m x 100m resolution ####
        shelf_xyz = "/net/kraken/bak/mzzhong/Ant_Data/GroundingLines/bedmap2_shelf_latlon.xyz"
        f = open(shelf_xyz,"r")
        shelf_points = f.readlines()

        ######## Provide the third component.
        grid_set_velo_2d = self.grid_set_velo_2d
        grid_set_velo_name = self.grid_set_velo_name

        grid_set_velo_name = grid_set_velo_name + '_' + str(self.resolution)

        print(len(grid_set_velo_2d))

        #test_point = (-76, -76.8)
        key = (-8100000,-7900000)

        all_points = grid_set_velo_2d.keys()
        grid_set_velo_3d = {}

        # Make a copy of 2d, and set vert to be 0 by default
        for point in all_points:
            grid_set_velo_3d[point] = grid_set_velo_2d[point]
            grid_set_velo_3d[point].append(0)
        print("total grid points: ", len(grid_set_velo_3d))
    
        # Loop through all point in shelf
        count = 0
        count_out = 0
        for line in shelf_points:
            count_out+=1
            if count_out%10000==0:
                print(count_out, len(shelf_points))

            lon, lat, vert = [float(x) for x in line.split()]

            point = (self.round_int_5dec(lon), self.round_int_5dec(lat))

            # Set the third component
            if point in grid_set_velo_3d.keys():
                grid_set_velo_3d[point][2] = vert
                count +=1

        print("test point: ", key, grid_set_velo_3d[key])
        print("set vertical count: ", count)
        print("total grid points: ", len(grid_set_velo_3d))

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

                # Only save values larger than zero.
                if not np.isnan(value) and value>0:
                    f.write(str(self.int5d_to_float(lon))+' '+str(self.int5d_to_float(lat))+' '+str(value)+'\n')

        f.close()

    def velo_model(self):

        # 2D
        self.match_velo_v2_to_grid_set()
        self.create_grid_set_velo_2d()

        # 3D
        if self.proj == "Evans":
            self.add_verti_evans()

        if self.proj == "Rutford":
            self.add_verti_rutford()

    ##############################################

    def grid_tiles(self):

        grid_set = self.grid_set

        if self.proj=="Rutford":

            ############ Rutford ###############
            # Rutford bounding box
            west = self.round_int_5dec(-88)
            east = self.round_int_5dec(-79)
            north = self.round_int_5dec(-76.2)
            south = self.round_int_5dec(-79.4)
    
            # For 500m resolution
            if self.resolution == 500:
                tile_lon_step = 1
                tile_lat_step = 0.2
    
            # For 100m resolution
            if self.resolution == 100:
                tile_lon_step = 0.2
                tile_lat_step = 0.04
            ##################################
        
        elif self.proj=="Evans":

            ########## Evans ###################
            # Evans bounding box.
            west = self.round_int_5dec(-85)
            east = self.round_int_5dec(-69)
            north = self.round_int_5dec(-74.2)
            south = self.round_int_5dec(-77.6)
    
            # For 500m resolution
            tile_lon_step = 0.5
            tile_lat_step = 0.1

        ################################################

        tile_lon_step_int = self.round_int_5dec(tile_lon_step)
        tile_lat_step_int = self.round_int_5dec(tile_lat_step)

        tile_lon_num = np.round(tile_lon_step/self.lon_step)
        tile_lat_num = np.round(tile_lat_step/self.lat_step)

        # Coordinates within a tile 
        sub_lon_list = np.arange(tile_lon_num) * self.lon_step_int
        sub_lat_list = np.arange(tile_lat_num) * self.lat_step_int

        print(sub_lon_list, len(sub_lon_list))
        print(sub_lat_list, len(sub_lat_list))
    
        count = 0
        tile_set = {}
        
        for tile_lon in np.arange(west, east+1, tile_lon_step_int):
            
            for tile_lat in np.arange(south, north+1, tile_lat_step_int):

                location = (tile_lon,tile_lat)
                #print(location)

                tile_set[location] = []

                tile_west = tile_lon
                tile_south = tile_lat

                lon_list = (tile_west + sub_lon_list).astype(int)
                lat_list = (tile_south + sub_lat_list).astype(int)

                for lon in lon_list:
                    for lat in lat_list:
                        
                        point = (lon, lat)
                        if point in grid_set.keys():
                            count = count + 1
                            tile_set[location].append(point)

        print("total number of points added in tiles: ", count)
        print("total number of points in grid set: ", len(grid_set))
        print("total number of tried tiles: ", len(tile_set))

        # Remove empty tiles.
        empty_tiles = []
        for tile in tile_set.keys():
            if len(tile_set[tile])==0:
                empty_tiles.append(tile)

        for tile in empty_tiles:
            tile_set.pop(tile)

        print("total number of tiles ", len(tile_set))
        
        print("Save the tile set...")
        
        if self.proj == "Rutford":
            prefix = "tile_set_csk-r"
        elif self.proj == "Evans":
            prefix = "tile_set_csk-e"
        else:
            print(stop)

        pkl_name = './pickles/' + prefix + "_" + str(self.resolution) + "_lon_" + str(tile_lon_step) + "_lat_" + str(tile_lat_step) + '.pkl'
        
        with open(pkl_name,'wb') as f:
            pickle.dump(tile_set,f)

def main():
    
    group = grouping()

    # Generate secular velocity model.
    #group.velo_model()
    
    # Find tiles.
    group.grid_tiles()

if __name__=='__main__':
    main()
