#!/usr/bin/env python3
import os
import sys
import numpy as np

import gdal

import matplotlib.pyplot as plt
from matplotlib import cm

from fourdvel import fourdvel

import pickle
class grouping(fourdvel):

    def __init__(self):

        if len(sys.argv)>1:
            param_file = sys.argv[1]
        else:
            print("parameter file is required")
            raise Exception()

        super(grouping,self).__init__(param_file)

        # Find the corresponding grid_set file
        self.grid_set_pkl_name = self.get_grid_set_info()
        print("Looking for: ",self.grid_set_pkl_name)

        if not os.path.exists(self.grid_set_pkl_name):
            self.create_grid_set()
        else:
            self.get_grid_set_v2()

        # Auxiliary files for finding ice shelf
        # For Evans
        self.doub_diff_file ='/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/doub_diff.off'

        # for Ruford
        self.rutford_shelf_grid_points = self.Ant_Data_dir + "/GroundingLines/bedmap2_shelf_latlon.xyz"

        # test_point
        if self.proj == "Rutford":
            self.test_point = (-8100000,-7900000)
        elif self.proj == "Evans":
            self.test_point = (-7700000, -7680000)
        else:
            raise Exception()

        # End of __init__

    # Preparing logistics for fourdvel inversion
    def create_grid_set(self):

        lat_step_int = self.lat_step_int
        lon_step_int = self.lon_step_int

        print("lon & lat step size (int5d): ", lon_step_int, lat_step_int)
        print("lon_re & lat_re (sampling rate): ", self.lon_re, self.lat_re)

        grid_set = {}

        directory = {}
        tracklist = {}

        directory['csk'] = self.csk_workdir
        directory['s1'] = self.s1_workdir

        tracklist['csk'] = self.csk_tracks
        tracklist['s1'] = self.s1_tracks

        sources = self.grid_set_sources
        print(sources)
        print(self.used_datasets)

        for sate in self.used_datasets:

            for track_num in tracklist[sate]:

                print(sate,track_num)

                # Find the track directory
                if sate == 'csk' and self.proj == "Rutford":
                    trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(3)+'_0')
                elif sate == "csk" and self.proj == "Evans":
                    trackdir = os.path.join(directory[sate],'track_' + str(track_num).zfill(2)+'0')
                elif sate == "s1":
                    trackdir = os.path.join(directory[sate],'track_' + str(track_num))
                else:
                    raise Exception("Undefined")

                # Find the geocoded los file 
                gc_losfile = os.path.join(trackdir, 'merged','geom_master','gc_los_offset_' + sources[sate] + '.rdr')
                
                gc_losvrtfile = gc_losfile + '.vrt'

                # Read in geometry info
                dataset = gdal.Open(gc_losvrtfile)
                geoTransform = dataset.GetGeoTransform()
                
                lon0 = geoTransform[0]
                lon_interval = geoTransform[1]
    
                lat0 = geoTransform[3]
                lat_interval = geoTransform[5]
    
                xsize = dataset.RasterXSize
                ysize = dataset.RasterYSize

                # Form the grid points 
                lon_list = np.linspace(lon0, lon0 + lon_interval*(xsize-1), xsize)
                lat_list = np.linspace(lat0, lat0 + lat_interval*(ysize-1), ysize)

                # Read in the data
                los = dataset.GetRasterBand(1).ReadAsArray()
                azi = dataset.GetRasterBand(2).ReadAsArray()

                ######################################################
                ## If files not geocoded onto the FULL RESOLUTION grid points,
                ## do it here.

                ## This is closed for now, because rutford offset field have
                ## been geocoded to full resolution as follows: 
                ## lat_step = 0.001
                ## lon_step = 0.005
                #######################################################

                ### Convert to 5 decimal point integer
                lon_list = self.round_int_5dec(lon_list)
                lat_list = self.round_int_5dec(lat_list)

                # Show the lon & lat lists 
                #print(lon_list,len(lon_list),xsize)
                #print(lat_list,len(lat_list),ysize)

                # Mesh grid
                grid_lon, grid_lat = np.meshgrid(lon_list, lat_list)
   
                INT_NAN = self.INT_NAN
 
                grid_lon[los == 0] = INT_NAN
                grid_lat[los == 0] = INT_NAN

                # Show the grid_lon & grid_lat    
                #fig = plt.figure(1)
                #ax = fig.add_subplot(111)
                #ax.imshow(grid_lat)
                #plt.show()

                # Flatten the grid points        
                grid_lon_1d = grid_lon.flatten()
                grid_lat_1d = grid_lat.flatten()

                #print(grid_lon_1d,len(grid_lon_1d))
                #print(grid_lat_1d,len(grid_lat_1d))
    
                # Read the observation vectors
                enu_gc_losfile = os.path.join(trackdir,'merged','geom_master','enu_gc_los_offset_' + sources[sate] + '.rdr.vrt')
                enu_gc_azifile = os.path.join(trackdir,'merged','geom_master','enu_gc_azi_offset_' + sources[sate] + '.rdr.vrt')
                
                try:
                    dataset = gdal.Open(enu_gc_losfile)
                except:
                    raise Exception('ENU file does not exist')

                # Los ENU
                elos = dataset.GetRasterBand(1).ReadAsArray()
                nlos = dataset.GetRasterBand(2).ReadAsArray()
                ulos = dataset.GetRasterBand(3).ReadAsArray()
    
                # Azi ENU
                dataset = gdal.Open(enu_gc_azifile)
                eazi = dataset.GetRasterBand(1).ReadAsArray()
                nazi = dataset.GetRasterBand(2).ReadAsArray()
                uazi = dataset.GetRasterBand(3).ReadAsArray()
    
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
    
                    # Push into the grid_set, only add new grid when sate is "csk".
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
            with open(self.grid_set_pkl_name,'wb') as f:
                pickle.dump(grid_set,f)

            self.grid_set = grid_set

            print("Done")

        print("Output a test point")
        print("Test point: ",self.test_point)
        print(self.grid_set[self.test_point])

        return 0

    def rounding(self,x):
        return np.round(x*1000)/1000

    def create_grid_set_tiles(self):

        grid_set = self.grid_set

        tile_set_pkl_name = self.get_tile_set_info()

        print(tile_set_pkl_name)
        redo = 0
        if os.path.exists(tile_set_pkl_name) and redo==0:
            print("tile set file exists!")
            return

        if self.proj=="Rutford":

            ############ Rutford ###############
            # Rutford bounding box
            west = self.round_int_5dec(-88)
            east = self.round_int_5dec(-79)
            north = self.round_int_5dec(-76.2)
            south = self.round_int_5dec(-79.4)
    
        elif self.proj=="Evans":

            ########## Evans ###################
            # Evans bounding box.
            west = self.round_int_5dec(-85)
            east = self.round_int_5dec(-69)
            north = self.round_int_5dec(-74.2)
            south = self.round_int_5dec(-77.6)
    
        ################################################

        tile_lon_step = self.tile_lon_step
        tile_lat_step = self.tile_lat_step

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
        
        with open(tile_set_pkl_name,'wb') as f:
            pickle.dump(tile_set,f)

        return 0


    def create_grid_set_velo_2d(self):

        grid_set = self.grid_set

        ### Part 1: Create matched velo 2d #####
        # Temporary file
        filename = self.grid_set_matched_velo_2d_pkl_name

        # Generate matched_velo.
        print('Matching velocity model (v2) to grids set...')
        print("filename: ", filename)
        
        redo = 0
        if not os.path.exists(filename) or redo == 1:
            velo_dir = self.Ant_Data_dir + '/velocity_models'
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

            print("vel_lon: ", vel_lon)
            print("vel_lat: ", vel_lat)

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

        #### Part 2: perform interpoltion to get complete 2d reference model ####

        pklname = self.grid_set_velo_2d_pkl_name

        redo = 0
        if not os.path.exists(pklname) or redo == 1:
            
            print('Interpolating grid set velocity model...')

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
            with open(pklname,'wb') as f:
                pickle.dump(grid_set_velo_2d,f)

        else:
            print(pklname, "exists")
            print("Loading...")
            with open(pklname,'rb') as f:
                grid_set_velo_2d = pickle.load(f)

        self.grid_set_velo_2d = grid_set_velo_2d

        print('Number of points in grid_set_velo_2d (after interpolation): ',len(grid_set_velo_2d))
        print('Number of points in grid_set: ',len(self.grid_set))

        write_to_file = True
        if write_to_file:
            xyz_file = self.estimation_dir + '/grid_set_velo_2d_speed.xyz'
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

    def create_grid_set_velo_3d_evans(self):

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

        test_point = self.test_point
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

    def create_grid_set_velo_3d_rutford(self):

        print('Add vertical component to grid set reference velocity model...')

        redo = 0
        key = self.test_point

        if not os.path.exists(self.grid_set_velo_3d_pkl_name) or redo==1:

            ###### Load in the ice shelf map with 100m x 100m resolution ####
            shelf_xyz = self.rutford_shelf_grid_points
            f = open(shelf_xyz,"r")
            shelf_points = f.readlines()
    
            ######## Provide the third component.
            grid_set_velo_2d = self.grid_set_velo_2d
            print("Length of grid_set_velo_2d: ", len(grid_set_velo_2d))
    
    
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

            self.grid_set_velo_3d = grid_set_velo_3d

            with open(self.grid_set_velo_3d_pkl_name, 'wb') as f:
                pickle.dump(self.grid_set_velo_3d , f)

        else:
            print(self.grid_set_velo_3d_pkl_name, "exists.")
            print("Loading...")
            with open(self.grid_set_velo_3d_pkl_name, 'rb') as f:
                grid_set_velo_3d = pickle.load(f)

            self.grid_set_velo_3d = grid_set_velo_3d

            print("test point: ", key, self.grid_set_velo_3d[key])
            print("total grid points: ", len(self.grid_set_velo_3d))
    
        print('Done with 3d reference velocity model')
    
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

        return 0

    def create_grid_set_ref_velo_model(self):

        self.get_grid_set_velo_info()

        # 2D
        self.create_grid_set_velo_2d()

        # 3D
        if self.proj == "Evans":
            self.create_grid_set_velo_3d_evans()

        if self.proj == "Rutford":
            self.create_grid_set_velo_3d_rutford()

    ##############################################

    def add_signatures_grid_set_ref_velo_model(self):
        
        print('Provide additional signatures to the reference velocity model...')

        redo = 1
        if redo==1:

            grid_set_velo_3d = self.grid_set_velo_3d
            print("Length of grid_set_velo_3d: ", len(grid_set_velo_3d))
    
            all_points = grid_set_velo_3d.keys()
    
            # Set the center of grounding zone
            center = (-82.7956, -78.6127)
            radius = 10 #km
            count = 0
            for point in all_points:
                # add the 4th component
                if len(grid_set_velo_3d[point])==3:
                    grid_set_velo_3d[point].append(0)

                point_float = self.int5d_to_float(point)
                dist = self.latlon_distance(point_float[0], point_float[1], center[0], center[1])

                if dist < radius:
                    grid_set_velo_3d[point][3]=1
                    count+=1
            
            print("total_grid points: ", len(grid_set_velo_3d))
            print("test point: ", self.test_point, grid_set_velo_3d[self.test_point])
            print("set signature count: ", count)
            print("total grid points: ", len(grid_set_velo_3d))

            self.grid_set_velo_3d = grid_set_velo_3d

            with open(self.grid_set_velo_3d_pkl_name, 'wb') as f:
                pickle.dump(self.grid_set_velo_3d , f)

        print('Done with adding signatures to 3d reference velocity model')

        print("test point: ", self.test_point, self.grid_set_velo_3d[self.test_point])
        print("total grid points: ", len(self.grid_set_velo_3d))
 
    
        ##################
        write_to_file = True
        if write_to_file:
            xyz_file = '/home/mzzhong/insarRoutines/estimations/grid_set_velo_3d_signature.xyz'
            f = open(xyz_file,'w')
            for key in sorted(grid_set_velo_3d.keys()):
                lon, lat = key
                value = np.sqrt(grid_set_velo_3d[key][3])

                # Only save values larger than zero.
                if not np.isnan(value):
                    f.write(str(self.int5d_to_float(lon))+' '+str(self.int5d_to_float(lat))+' '+str(value)+'\n')

            f.close()

        return 0

def main():
    
    group = grouping()

    # Generate tiles
    group.create_grid_set_tiles()

    # Generate secular velocity model.
    group.create_grid_set_ref_velo_model()

    # Add additional signatures to the velo model
    #group.add_signatures_grid_set_ref_velo_model()
    

if __name__=='__main__':
    main()
