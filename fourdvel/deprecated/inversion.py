#    def point_tides(self, point, tracks):
#
#        ### Data ###
#        # Choose the test mode.
#        test_id = self.test_id
#        test_mode = self.test_mode
#
#        # Data formation.
#        (data_vec, invCd, offsetfields, true_tide_vec) = self.data_formation(point, tracks, test_mode)
#
#        #self.data_vec_mode_1 = data_vec
#
#        ### MODEL ###
#        # Design matrix.
#        design_mat = self.build_G(offsetfields=offsetfields)
#        #print("Design matrix (G)\n:", design_mat)
# 
#        # Model prior.
#        invCm = self.model_prior(horizontal = self.horizontal_prior)
#
#        # Model posterior.
#        Cm_p = self.model_posterior(design_mat, invCd, invCm)
#        #print(Cm_p[0,0])
#        #print(design_mat[0,0])
#        #print(stop)
#
#        #print('Model posterior: ',Cm_p)
#
#        # Show the model posterior.
#        #self.display.show_model_mat(Cm_p)
#
#        #self.model_analysis(design_mat = design_mat, model_prior = invCm)
#
#        ### Inversion ###
#        # Estimate model params.
#        model_vec = self.param_estimation(design_mat, data_vec, invCd, invCm, Cm_p)
#
#        # Calculate and show the residuals.
#        resid_of_secular = self.resid_of_secular(design_mat, data_vec, model_vec)
#        self.display.show_resid_dist(resid_of_secular, label=str(test_id)+'secular')
#
#        resid_of_tides = self.resid_of_tides(design_mat, data_vec, model_vec)
#        self.display.show_resid_dist(resid_of_tides, label=str(test_id)+'tides')
#
#        # Convert to tidal params.
#        tide_vec = self.model_vec_to_tide_vec(model_vec)
#
#        # Convert model posterior to uncertainty of params.
#        # Require: tide_vec and Cm_p
#        tide_vec_uq = self.model_posterior_to_uncertainty(tide_vec, Cm_p)
#
#        #print(tide_vec_uq)
#        #print(stop)
#
#        print('Inversion Done')
#        
#        ##### Show the results ####
#        # Stack the true and inverted models.
#        if true_tide_vec is not None:
#            stacked_vecs = np.hstack((true_tide_vec, tide_vec, tide_vec_uq))
#            row_names = ['Input','Estimated','Uncertainty']
#            column_names = ['Secular'] + self.modeling_tides
#        else:
#            stacked_vecs = np.hstack((tide_vec, tide_vec_uq))
#            row_names = ['Estimated','Uncertainty']
#            column_names = ['Secular'] + self.modeling_tides
#
#        # Always save the results.
#        self.display.display_vecs(stacked_vecs, row_names, column_names, test_id)
#
#        return 0

#    def driver_mp(self):
#        # Using multi-threads to get map view estimation.
#
#        # Calcuate design matrix and do estimations.
#        # Always redo it.
#        redo = 1
#
#        # est_dict pickle file.
#        est_dict_pkl = self.est_dict_name + '.pkl'
#
#        # The minimum number of tracks.
#        min_tracks = 2
#
#        if redo or not os.path.exists(est_dict_pkl):
#
#            # Load the grid point set.
#            grid_set = self.grid_set
#
#            # Remove the bad grid points where not enough tracks are available.
#            bad_keys=[]
#            for key in grid_set.keys():
#                if len(grid_set[key])<min_tracks:
#                    bad_keys.append(key)
#
#            for key in bad_keys:
#                if key in grid_set:
#                    del grid_set[key]
#
#            # Count the total number of grid points.
#            total_number = len(grid_set.keys())
#            print(total_number)
#
#            # Chop into multiple threads. 
#            nthreads = 16
#            
#            mod = total_number % nthreads
#            
#            if mod > 0:
#                chunk_size = (total_number - mod + nthreads) // nthreads
#            else:
#                chunk_size = total_number // nthreads
#
#            divide = np.zeros(shape=(nthreads+1,))
#            divide[0] = 0
#
#            for it in range(1, nthreads+1):
#                divide[it] = chunk_size * it
#            divide[nthreads] = total_number
#
#            print(divide)
#            print(len(divide))
#
#            # Multithreading starts here.
#            # The function to run every chunk.
#            func = self.chunk_run
#
#            manager = multiprocessing.Manager()
#            grid_est = manager.dict()
#
#            jobs=[]
#            for ip in range(nthreads):
#                start = divide[ip]
#                end = divide[ip+1]
#                p=multiprocessing.Process(target=func, args=(start,end,grid_est,))
#                jobs.append(p)
#                p.start()
#
#            for ip in range(nthreads):
#                jobs[ip].join()
#
#            est_dict = dict(grid_est)
#
#            # Save the results.    
#            with open(os.path.join(self.design_mat_folder, est_dict_pkl),'wb') as f:
#                pickle.dump(est_dict,f)
#
#        else:
#
#            # Load the pre-computed results.
#            with open(os.path.join(self.design_mat_folder,est_dict_pkl),'rb') as f:
#                est_dict = pickle.load(f)
#
#        # Show the estimation.
#        self.show_est(est_dict)


#    def chunk_run(self,start,end,grid_est):
#
#        grid_set = self.grid_set
#        
#        count = 0
#        for grid in grid_set.keys():
#            # Output the percentage of completeness.
#            if count % 500 == 0 and start==0 and count<end:
#                print(count/end)
#
#            if count>=start and count<end:
#                est_value = self.model_analysis(point=grid, tracks=grid_set[grid])
#                grid_est[grid] = est_value
#            count = count + 1


#    def driver_serial(self):
#        
#        # Tile by tile.
#        tile_set = self.tile_set
#        grid_set = self.grid_set 
#
#        horizontal_prior = self.horizontal_prior
#
#        for tile in tile_set.keys():
#
#            if tile == (-77, -76.6):
#
#                for point in tile_set[tile]:
#    
#                    # Find the tracks.
#                    tracks = grid_set[point]
#    
#                    # Obtain the real data.
#                    self.point_tides(point = point, tracks = tracks)
#
#                    break


#    def data_formation(self, point, tracks, test_mode=None):
#
#        from simulation import simulation
#
#        ### DATA ###
#        # Modes:
#        # 1. Synthetic data: projected catalog (Done)
#        # 2. Synthetic data: real catalog  (Test)
#        # 3. True data: real catalog (Not sure)
#
#        # If using synthetic data, need to return true tidal parameters for comparison.
#
#        if test_mode == 1:
#
#            offsetfields = self.tracks_to_full_offsetfields(tracks)
#
#            # Now find offsets.
#            # Synthetic data.
#            fourD_sim = simulation()
#            velo_model = self.grid_set_velo[point]
#            # Obtain the synthetic ice flow.
#            (secular_v, tide_amp, tide_phase) = fourD_sim.syn_velocity(velo_model=velo_model)
#
#            # Obtain SAR data.
#            # Data prior.
#            ## Use external noise model.
#            noise_sigma = self.load_noise_sigma(point)
#
#
#            data_vec = fourD_sim.syn_offsets_data_vec(point=point, secular_v = secular_v, tide_amp = tide_amp, tide_phase = tide_phase, modeling_tides = self.modeling_tides, offsetfields = offsetfields, noise_sigma = noise_sigma)
#                
#            #print("Data vector (G)\n", data_vec)
#
#            # Data prior.
#            invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#            # True tidal params.
#            true_tide_vec = fourD_sim.true_tide_vec(secular_v, self.modeling_tides, tide_amp, tide_phase)
#
#        if test_mode == 2 or test_mode == 3:
#
#            # Get catalog of real offsetfields.
#            offsetfields = []
#            offsets = []
#            for track in tracks:
#                track_offsetfields, track_offsets = self.find_track_data(point, track)
#    
#                print('Available track offsetfields: ',track_offsetfields)
#                print('Obtained offsets: ', track_offsets)
#                print('Length of offsetfields and offsets: ', len(track_offsetfields),len(track_offsets))
#    
#                offsetfields = offsetfields + track_offsetfields
#                offsets = offsets + track_offsets
#
#            print('======= END OF EXTRACTION ========')
#            print('Total number of offsetfields: ', len(offsetfields))
#            print('Total length of offsets: ', len(offsets))
#
#            if test_mode == 2:
#
#                # Synthetic data.
#                fourD_sim = simulation()
#                # Obtain the synthetic ice flow.
#                velo_model = self.grid_set_velo[point]
#                (secular_v, tide_amp, tide_phase) = fourD_sim.syn_velocity(velo_model = velo_model)
#
#                # Data prior.
#                ## Use external noise model.
#                noise_sigma = self.load_noise_sigma(point)
#
#
#                data_vec = fourD_sim.syn_offsets_data_vec(point=point, secular_v = secular_v, tide_amp = tide_amp, tide_phase = tide_phase, modeling_tides = self.modeling_tides, offsetfields = offsetfields, noise_sigma = noise_sigma)
#
#                 # Data prior.
#                invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#                # True tidal params.
#                true_tide_vec = fourD_sim.true_tide_vec(secular_v, self.modeling_tides, tide_amp, tide_phase) 
#    
#            elif test_mode == 3:
#
#                # Flatten offsets to be a data vector.
#                data_vec = self.offsets_to_data_vec(offsets)
#    
#                # Data prior.
#                # Pre-assigned.
#
#                ## Use external noise model.
#                noise_sigma = self.load_noise_sigma(point)
#
#                #print(noise_sigma)
#                #print(stop)
#
#                invCd = self.real_data_uncertainty(data_vec, noise_sigma)
#
#                # No true tidal parameters.
#                true_tide_vec = None
#
#        return (data_vec, invCd, offsetfields, true_tide_vec)

#    def find_track_data(self, point, track):
#
#        from dense_offset import dense_offset
#
#        track_num = track[0]
#        sate = track[3]
#
#        print('Find track data') 
#        print(point, track_num, sate)
#
#        if sate == 'csk':
#            stack = "stripmap"
#            workdir = "/net/kraken/nobak/mzzhong/CSK-Evans"
#            name='track_' + str(track_num).zfill(2) + str(0)
#            runid = 20180712
#
#        elif sate == 's1':
#            stack = 'tops'
#            workdir = '/net/jokull/nobak/mzzhong/S1-Evans'
#            name = 'track_' + str(track_num)
#            runid = 20180703
#
#        # Create dense_offset object
#        offset = dense_offset(stack=stack, workdir=workdir)
#        offset.initiate(trackname = name, runid=runid)
#
#        print('satellite: ', sate, ' track number: ', track_num)
#
#        # Dates allowed to use.
#        if sate == 'csk':
#            used_dates = self.csk_data[track_num]
#        elif sate == 's1':
#            used_dates = self.s1_data[track_num]
#
#        #print('track_number: ',track_num)
#        #print('used dates: ', used_dates)
#        
#        track_pairs, track_offsets = offset.extract_offset_series(point=point, dates = used_dates)
#
#        #print('obtained pairs: ', track_pairs)
#        #print('obtained offsets: ', track_offsets)
#
#        # Add observation vectors and time fraction.
#        track_offsetfields = []
#        
#        vec1 = track[1]
#        vec2 = track[2]
#        t_frac = self.track_timefraction[(sate,track_num)]
#        tail = [vec1, vec2, t_frac]
#
#        for pair in track_pairs:
#            offsetfield = pair + tail
#            track_offsetfields.append(offsetfield)
#
#        return track_offsetfields, track_offsets

