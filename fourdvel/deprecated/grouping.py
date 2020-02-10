########## Deprecated #####################################
#    def coloring_bfs(self, point_set, x, y):
#
#        redo = 0
#        if os.path.exists('./pickles/quene.pkl') and redo == 0:
#            with open('./pickles/quene.pkl','rb') as f:
#                quene = pickle.load(f)
#        else:
#            x = np.round(x*1000)/1000
#            y = np.round(y*1000)/1000
#     
#            quene = []
#            quene.append((x,y))
#    
#            head = -1
#            while head < len(quene)-1:
#    
#                print(len(quene))
#    
#                head = head + 1
#                head_x = quene[head][0]
#                head_y = quene[head][1]
#    
#                for direction in range(4):
#                    if direction==0:
#                        next_x = head_x - self.lon_step
#                        next_y = head_y
#                    elif direction==1:
#                        next_x = head_x + self.lon_step
#                        next_y = head_y
#                    elif direction==2:
#                        next_x = head_x
#                        next_y = head_y - self.lat_step
#                    elif direction==3:
#                        next_x = head_x
#                        next_y = head_y + self.lat_step
#                
#                    next_x = np.round(next_x*1000)/1000
#                    next_y = np.round(next_y*1000)/1000
#            
#                    if ((next_x,next_y) in point_set) and (not (next_x, next_y) in quene):
#                        quene.append((next_x,next_y))
#
#            # Fill in the holes.
#            #quene = self.fill_in_holes(quene)
#            
#            with open('./pickles/quene.pkl','wb') as f:
#                pickle.dump(quene,f)
#
#        return quene
#
#    def define_shelves(self):
#
#        grid_set = self.grid_set
#        shelves = {}
#
#        shelves_lon = []
#        shelves_lat = []
#        for grid in grid_set.keys():
#            point = grid
#            lon, lat = point
#            ind_x, ind_y = offset.point_index(point)
#
#            if ind_x is not None and ind_y is not None and lon>-79:
#                doub_diff_value = doub_diff_map[ind_y,ind_x]
#                
#                # It is on ice shelves
#                if doub_diff_value >= thres:
#                    shelves[grid] = grid_set[grid]
#                    shelves_lon.append(point[0])
#                    shelves_lat.append(point[1])
#                    
#        print(len(grid_set))
#        print(len(shelves))
#
#        # Coloring.
#        start_lon = -75
#        start_lat = -77 
#        new_shelves_keys = self.coloring_bfs(shelves.keys(),start_lon, start_lat)
#
#        print(len(shelves.keys()))
#        print(len(new_shelves_keys))
#
#        # Update shelves from coloring.
#        new_shelves = {}
#        new_shelves_lon = []
#        new_shelves_lat = []
#
#        for point in new_shelves_keys:
#            new_shelves[point] = grid_set[point]
#            new_shelves_lon.append(point[0])
#            new_shelves_lat.append(point[1])
#
#        # Update ground.
#        new_ground_keys = grid_set.keys()-new_shelves_keys
#
#        new_ground = {}
#        new_ground_lon = []
#        new_ground_lat = []
#
#        for point in new_ground_keys:
#            new_ground[point] = grid_set[point]
#            new_ground_lon.append(point[0])
#            new_ground_lat.append(point[1])
#
#
#        self.shelves = new_shelves
#        self.shelves_lon = new_shelves_lon
#        self.shelves_lat = new_shelves_lat
#
#        self.ground = new_ground
#        self.ground_lon = new_ground_lon
#        self.ground_lat = new_ground_lat
#
#    def streams(self):
#
#        grid_set = self.grid_set
#
#        ground = self.ground
#
#        shelves = self.shelves
#        shelves_lon = self.shelves_lon
#        shelves_lat = self.shelves_lat
#
#        Ant = Ant_data()
#
#        # Obtain the data.
#        vel, vel_lon, vel_lat = Ant.get_veloData()
#
#        # Rounding coordinates.
#        vel_lon  = np.round(vel_lon * self.lon_re) / self.lon_re
#        vel_lat = np.round(vel_lat* self.lat_re)/self.lat_re
#
#        #print(vel_lon)
#        #print(vel_lat)
#
#        # Only moving ice.
#        thres = 0.1 # m/d
#        # Form the set
#        moving_ice_set = []
#        for i in range(vel.shape[0]):
#            for j in range(vel.shape[1]):
#                if vel[i,j]>thres:
#                    moving_ice_set.append((vel_lon[i,j],vel_lat[i,j]))
#
#        moving_ice_set = set(moving_ice_set)
#
#        # Moving points in Evans.
#        streams_keys = moving_ice_set & ground.keys()
#        print(len(streams_keys))
#
#        streams = {}
#        streams_lon = []
#        streams_lat = []
#        for point in streams_keys:
#            streams[point] = grid_set[point]
#            streams_lon.append(point[0])
#            streams_lat.append(point[1])
#
#        print(len(streams_lon))
#
#        self.streams = streams
#
#        fig = plt.figure(2,figsize=(10,10))
#        ax = fig.add_subplot(111)
#        ax.plot(streams_lon, streams_lat, 'r.')
#        ax.plot(shelves_lon, shelves_lat, 'b.')
#    
#        fig.savefig('./fig_sim/streams.png',format='png')
#
#    ##############################################



#    def match_velo_to_grid_set(self):
#
#        grid_set = self.grid_set
#
#        # Generate matched_velo.
#        print('Matching velocity model to grids set')
#        redo = 0
#
#        if redo == 1:
#            velo_dir = '/net/jokull/nobak/mzzhong/Ant_Data/velocity_models'
#            npz_filebasename = 'AntVelo.npz'
#
#            npzfile = np.load(os.path.join(velo_dir, npz_filebasename))
#
#            vel_lon = npzfile['vel_lon']
#            vel_lat = npzfile['vel_lat']
#            ve = npzfile['ve']
#            vn = npzfile['vn']
#            v_comb = npzfile['v_comb']
#
#            # Rounding latlon to grid points.
#            vel_lon = np.round(vel_lon * self.lon_re)/self.lon_re
#            vel_lat = np.round(vel_lat * self.lat_re)/self.lat_re
#
#            # Match velo to grid_set
#            matched_velo = {}
#            for i in range(vel_lon.shape[0]):
#                #print(i)
#                for j in range(vel_lat.shape[0]):
#                   
#                    if (vel_lon[i,j],vel_lat[i,j]) in grid_set.keys():
#                        # only save valid values.
#                        valid = True
#
#                        # Nan value.
#                        if (ve[i,j]==0 and vn[i,j]==0):
#                            valid = False
#
#                        # Holes.
#                        if vel_lon[i,j] > -75 and vel_lon[i,j] < -74.5 and vel_lat[i,j]<-77 and vel_lat[i,j] > -77.2:
#                            valid = False
#                        
#                        if valid:
#                            matched_velo[(vel_lon[i,j], vel_lat[i,j])] = [ve[i,j],vn[i,j]]
# 
#            with open('./pickles/matched_velo.pkl','wb') as f:
#                pickle.dump(matched_velo,f)
#
#        else:
#            with open('./pickles/matched_velo.pkl','rb') as f:
#                matched_velo = pickle.load(f)
#
#        self.matched_velo = matched_velo
#
#        return 0

