#!/usr/bin/env python3
# Author: Minyan Zhong
# Development started in June, 2019

import numpy as np
import matplotlib.pyplot as plt
import datetime
from basics import basics
from fourdvel import fourdvel

import theano
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import pickle

plt.style.use('seaborn-darkgrid')

class Bayesian_Linear(fourdvel):

    def __init__(self):

        pass

class Bayesian_MCMC(fourdvel):

    def __init__(self, param_file):

        super(Bayesian_MCMC, self).__init__(param_file)

    def set_point_set(self, point_set):

        self.point_set = point_set

    def set_modeling_tides(self, modeling_tides):
    
        self.modeling_tides = modeling_tides
        self.n_params = 3 + 6 * len(modeling_tides)

    def set_model_priors(self, model_prior_set=None, no_secular_up=False, up_short_period=False, horizontal_long_period=False, up_lower=None, up_upper=None):

        self.model_prior_set = model_prior_set
        self.no_secular_up = no_secular_up
        self.up_short_period = up_short_period
        self.horizontal_long_period = horizontal_long_period

        self.up_lower = up_lower
        self.up_upper = up_upper

    def set_noise_sigma_set(self, noise_sigma_set):

        self.noise_sigma_set = noise_sigma_set

    def set_data_set(self, data_vec_set):

        self.data_vec_set = data_vec_set

    def set_offsetfields_set(self, offsetfields_set):

        self.offsetfields_set = offsetfields_set

    def set_stack_design_mat_set(self, stack_design_mat_set):

        self.stack_design_mat_set = stack_design_mat_set

    def set_linear_design_mat_set(self, linear_design_mat_set):

        self.linear_design_mat_set = linear_design_mat_set

    def set_up_disp_set(self, up_disp_set):
        self.up_disp_set = up_disp_set

    def run_test0(self):

        # Create data
        size = 2000
        true_intercept = 1
        true_slope = 2

        x = np.linspace(0,10,size)

        G = np.zeros(shape=(size,2))
        G[:,0] = x
        G[:,1] = 1

        true_model = np.asarray([2,1])[:,None]

        true_regression_line = np.matmul(G, true_model)

        y = true_regression_line + np.random.normal(scale=0.5, size=size).reshape(size,1)

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, xlabel='x',ylabel='y',title = "Generated data and underlying model")
        ax.plot(x,y, 'x', label='sampled data')
        ax.plot(x, true_regression_line, label="true regression line",lw=2)
        plt.legend(loc=0)
        plt.savefig('1.png')

        self.linear_model = pm.Model()
        tt_G = theano.shared(G)
        with self.linear_model as model:

            # Non-informative prior
            intercept = pm.Normal("Intercept", 0, sigma=20, shape=(1,1))
            x_coeff = pm.Normal('x',0, sigma=20, shape=(1,1))

            model_vec = pm.math.concatenate([x_coeff, intercept], axis=0)

            pred = tt_G.dot(model_vec)
            
            likelihood = pm.Normal('y', mu=pred, sigma = 0.5, observed=y)
            
            # Inference
            trace = pm.sample(3000, cores=4)

        plt.figure()
        pm.traceplot(trace)
        plt.savefig('2.png')

    def run_test1(self):

        # Linear regression
        P = 4
        N = 2000
        model = np.arange(P).reshape(P,1)
        np.random.seed(20190717)
        G = np.random.randn(N*P).reshape(N,P)
        true_data = np.matmul(G,model)

        real_data = true_data + np.random.randn(N).reshape(N,1)

        # Solve for model (psudo-inverse)
        est_model_true_data = np.matmul(np.linalg.pinv(G), true_data)
        print(est_model_true_data.T)

        est_model_real_data = np.matmul(np.linalg.pinv(G), real_data)
        print(est_model_real_data.T)

        # Solver for model using Bayesian programming
        self.linear_model = pm.Model()
        tt_G = theano.shared(G)
        with self.linear_model as model:
            rvs = []
            for i in range(P):
                rv = pm.Normal('x_'+str(i), mu=0, sigma=100, shape=(1,1))
                rvs.append(rv)
            model_vec = pm.math.concatenate(rvs, axis=0)
            pred = pm.math.dot(tt_G, model_vec)
            pm.Normal('obs',mu=pred, sigma=1, observed=real_data)

            n_draws = 500
            step = pm.Metropolis()
            trace1 = pm.sample(draws = n_draws, step=step, cores=4, tune=500, discard_tuned_samples=False)
            fname1 = pm.save_trace(trace1)

            trace2 = pm.sample(draws = n_draws, step=step, cores=4, tune=500, trace=trace1, discard_tuned_samples=False)
            fname2 = pm.save_trace(trace2)

            a = trace1.get_values('x_3')
            print(a.shape)
            a2 = trace2.get_values('x_3')
            print(a2.shape)
            return 0
           
            print(dir(trace))
            print('a')
            print(type(a))
            print(a.shape)
            print(a[:100,0,0])
            plt.plot(a[:,0,0])
            plt.savefig('4.png')
            b = trace.get_values('x_3',chains=[3])
            print('b')
            print(type(b))
            print(b.shape)
            print(b[:100,0,0])

        with self.linear_model:
            trace2 = pm.load_trace(fname)
            a = trace2.get_values('x_3')
            plt.plot(a[:,0,0]) 
            plt.savefig('5.png')

        return 0

    def run_test2(self):

        N = 10
        G = np.zeros(shape=(20,N))+1

        G = theano.shared(G)
        #print(G.shape.eval())
        #print(stop)

        self.bmc_model = pm.Model()

        with self.bmc_model as model:
            #secular_mean = [1,2,3]
            #secular_cov = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
            #secular = pm.MvNormal('secular',mu=[1,2,3], )

            # Data
            obs_data = np.random.randn(4,40)

            # RV prior
            #mu = pm.Normal('mu',mu=0, sigma=100)
            #sigma = pm.HalfNormal('sigma',sigma=10)

            # Method 1
            # Create a list of variables
            #rv = []
            #for i in range(N):
            #    # Create a random variable
            #    mu=pm.Normal('mu'+str(i), sigma=100,shape=(1,1))
            #    rv.append(mu)

            # Method 2
            mu = pm.Normal('mu', 0, sigma=1,shape=(4,1))
            theta = pm.Normal('theta', 0, sigma=1,shape=(6,1))
            gamma = pm.math.concatenate([mu, theta], axis=0)

            #gamma = pm.math.concatenate(rv, axis=0)

            #gamma2 = G.dot(gamma)

            # Method 3
            # Multivariate
            #mu = np.zeros(4,)
            #cov = np.eye(4)
            #eta = pm.MvNormal('eta', mu=mu, cov=cov, shape=(1,4))
            #print(eta.random().shape)

            #beta0 = pm.Normal('beta0', 0, sigma=100, shape=(1,1))
            #print(beta0.random())

            #beta = pm.math.concatenate([beta0, eta[0]], axis=0)
            #print(beta.random().shape)
           
            for i in range(4):
                pm.Normal('data'+str(i), mu=gamma[i],sigma=1, observed=obs_data[:,i])
            
            step = pm.Metropolis()
            trace = pm.sample(500, step=step, cores = 8, discard_tuned_samples=True)

            pm.traceplot(trace)
            plt.savefig('bayes_1.png')

            #value0 = trace.get_values('mu',chains=0)
            #value1 = trace.get_values('mu',chains=1)
            #value2 = trace.get_values('mu',chains=2)

            #print(len(value0),len(value1))
            #print(value0, value1)

            #print(dir(trace))
            #print(trace.chains)
            #print(trace.nchains)
            #print(trace.varnames)
            #print(trace.point(50,chain=0))
            #print(trace.points)

    def construct_model_vec(self, point, grounding=False):

        RVs_secular = []
        RVs_tidal = []

        comps = ['e','n','u']

        model_prior = self.model_prior_set[point]

        no_secular_up = self.no_secular_up
        up_short_period = self.up_short_period
        horizontal_long_period = self.horizontal_long_period

        up_lower = self.up_lower
        up_upper = self.up_upper

        with self.bmc_model as model:

            # Secular component
            for i, comp in enumerate(comps):
                #rv = pm.Normal('Secular_'+ comp, mu=model_prior[i], sigma=max(0.1*abs(model_prior[i]),0.0001), shape=(1,1), testval=model_prior[i])
                rv = pm.Normal('Secular_'+ comp, mu=0, sigma=10, shape=(1,1), testval=model_prior[i])
                RVs_secular.append(rv)

            self.model_vec_secular = pm.math.concatenate(RVs_secular, axis=0)


            # Tidal componens
            sigma_permiss = 10
            sigma_restrict = 0.0001
            #sigma_restrict = 10

            comp_name = ['cosE','cosN','cosU','sinE','sinN','sinU']
            for i, tide_name in enumerate(self.modeling_tides):
                for j in range(6):
                    k = 3 + i*6 + j
                    if up_short_period and not tide_name in ['M2','S2','K2','O1','K1','P1'] and (j==2 or j==5):
                        sigma = sigma_restrict
    
                    elif horizontal_long_period and not tide_name in ['Mf','Msf','Mm'] and (j==0 or j==1 or j==3 or j==4):
                        sigma = sigma_restrict
                    else:
                        sigma = sigma_permiss
    
                    rv_name = tide_name + '_' + comp_name[j]
                    rv = pm.Normal(rv_name, mu=0, sigma=sigma, shape=(1,1),testval=0)
                    RVs_tidal.append(rv)

            self.model_vec_tidal = pm.math.concatenate(RVs_tidal, axis=0)

            # Grouding component
            if grounding:
                self.grounding = pm.Uniform('grounding', lower= up_lower, upper = up_upper, testval=-1)


        print('secular parameter vector length: ', len(RVs_secular))
        print('tidal parameter vector length: ', len(RVs_tidal))

        return 0

    def remove_design_mat_cols(self, G, secular_included, remove_tidal_up=False):

        # MCMC_linear: secular_included = True
        # MCMC: secular_included = False

        # Record the parameters that go into the inversion
        self.kept_model_vec_entries = []
        cols = []
        secular_index_offset = 3

        # Secular E and N are included in G
        if secular_included:
            cols.append(G[:,0:secular_index_offset])
            self.kept_model_vec_entries = [0,1,2]
            heading_offset = secular_index_offset
        else:
            heading_offset = 0

        for i, tide_name in enumerate(self.modeling_tides):
            for j in range(6):

                # the index in complete model_vec being subset
                k = heading_offset + i*6 + j

                # Constrain long period up to be small
                if self.up_short_period and tide_name in self.tide_long_period_members and (j==2 or j==5):
                    # Not include this parameter
                    pass

                # Constrain short period horizontal to be small
                elif self.horizontal_long_period and tide_name in self.tide_short_period_members and (j==0 or j==1 or j==3 or j==4):

                    # Not include this parameter
                    pass

                elif remove_tidal_up and (j==2 or j==5):

                    # Not include this parameter
                    pass

                else:
                    #print(tide_name, j)
                    cols.append(G[:,k][:,None])
                    # Record the index of the parameter
                    if secular_included:
                        self.kept_model_vec_entries.append(k)
                    else:
                        self.kept_model_vec_entries.append(k + secular_index_offset)

        new_G = np.hstack(cols)

        return new_G
    
    def run_MCMC_Linear(self, run_point, true_model_vec_set=None, suffix=None):

        print('Running MCMC on linear model...')

        for point in self.point_set:

            if point!=run_point:
                continue

            print('The grid point is: ', point)

            # Design matrix
            linear_design_mat = self.linear_design_mat_set[point]

            # Subset the design matrix to remove some parameters
            linear_design_mat = self.remove_design_mat_cols(G=linear_design_mat, secular_included=True)

            # Print
            #print(linear_design_mat[:,2].tolist())
            #print(np.linalg.cond(linear_design_mat))
            #print(np.linalg.matrix_rank(linear_design_mat))
            #print(stop)

            # Data
            data_vec = self.data_vec_set[point]

            # Pseudo-inverese
            model_est = np.matmul(np.linalg.pinv(linear_design_mat), data_vec)
            print('pseudo-inverse solution of orgingal matrix')
            print(model_est.T)

            #### Rescale the matrix ###
#            # Rescale of mean and std
#            col_mean = np.nanmean(linear_design_mat, axis=0)[None,:]
#            col_std = np.nanstd(linear_design_mat, axis=0)[None,:]
#            #print(col_mean)
#            print("col_std")
#            print(col_std)
#            
#            #normalized_linear_design_mat = (linear_design_mat - col_mean)/col_std
#            normalized_linear_design_mat = linear_design_mat / col_std
#
#            # Pseudo-inverese
#            normalized_model_est = np.matmul(np.linalg.pinv(normalized_linear_design_mat), data_vec)
#            print('pseudo-inverse solution of normalized matrix')
#            print(normalized_model_est.T)
#
#            # Scale back
#            #recovered_model_est = np.linalg.pinv(linear_design_mat).dot(normalized_linear_design_mat).dot(normalized_model_est)
#            recovered_model_est = normalized_model_est / col_std.T
#
#            print('pseudo-inverse solution of normalized matrix after scaling back')
#            print(recovered_model_est.T)
#
#            print('G shape: ', linear_design_mat.shape)
#            print('data shape: ', data_vec.shape)

            # PyMC
            N, P = linear_design_mat.shape
            tt_linear_G = theano.shared(linear_design_mat)
            #tt_linear_G = theano.shared(normalized_linear_design_mat)

            #plt.imshow(linear_design_mat, aspect='auto')
            #plt.colorbar()
            #plt.savefig('G.png')
            #print(stop)

            # Construct the parameter vector
            self.bmc_model = pm.Model()
            #self.construct_model_vec(point = point, grounding = False)

            with self.bmc_model as model:
                # option 1
                #model_vec = pm.math.concatenate([self.model_vec_secular, self.model_vec_tidal], axis=0)

                # option 2
                #rvs = []
                #print('total number of parameters: ', P)
                #for i in range(P):
                #    rv = pm.Normal('x_'+str(i), mu=0, sigma=10, shape=(1,1))
                #    rvs.append(rv)
                #    model_vec = pm.math.concatenate(rvs, axis=0)
        
                # option 3
                model_vec = pm.Normal('x', mu=0, sigma=10, shape=(P,1))
                    
                pred = pm.math.dot(tt_linear_G, model_vec)
                pm.Normal('obs', mu = pred, sigma=0.2, observed = data_vec)

                # Find the MAP solution
                map_estimate = pm.find_MAP(model=model)

                # Get model vec that satisfy to original definition
                model_vec = self.pad_to_orig_model_vec(map_estimate, mode="linear")
                
                # Save the MAP solution
                pkl_name = "_".join([self.estimation_dir+"/map_estimate_BMC_linear",str(point[0]),str(point[1]),suffix])
                with open(pkl_name + ".pkl","wb") as f:
                    pickle.dump(map_estimate,f)

                # MCMC Samping
                N_draws = 4000
                trace = pm.sample(draws=N_draws, tune=4000)

                # Find the true model vec
                if true_model_vec_set is not None:

                    # Include both secular & tidal components
                    true_model_vec = true_model_vec_set[point]

                    # Get the compressed model_vec by removing the components
                    # constrained to be small
                    compressed_true_model_vec = [] 
                    for ind in self.kept_model_vec_entries:
                        compressed_true_model_vec.append(true_model_vec[ind,0])

                    trace.true_model_vec = np.asarray(compressed_true_model_vec)
                else:
                    trace.true_model_vec = None

                # Save the trace to disk
                pkl_name = "_".join([self.estimation_dir+"/samples_BMC_linear",str(point[0]),str(point[1]),suffix])
                with open(pkl_name + ".pkl","wb") as f:
                    pickle.dump(trace,f)

                pm.traceplot(trace)
                plt.savefig('MCMC_linear.png')
                
                # Show the true model vec
                if true_model_vec_set is not None:
                    print("true model vec:")
                    print(true_model_vec_set[point])

        return model_vec

    def pad_to_orig_model_vec(self, map_estimate, mode):

        if not mode in ["linear","nonlinear"]:
            raise Exception("Undefined mode")

        if mode=="nonlinear":
            secular = map_estimate['secular'][0]
            tidal = map_estimate['tidal'][:,0]
            grounding = map_estimate['grounding'][0][0]
    
            compressed_model_vec = np.hstack((secular, tidal))
    
            model_vec = np.zeros(shape=(self.n_params,1))
            model_vec[0][0] = secular[0]
            model_vec[1][0] = secular[1]
            model_vec[2][0] = secular[2]
    
            i=0
            print("kept entries: ", self.kept_model_vec_entries)
            for j in self.kept_model_vec_entries:
                model_vec[j] = compressed_model_vec[i]
                i+=1
    
            return (model_vec, grounding)
        
        elif mode=="linear":

            compressed_model_vec = map_estimate['x']
            model_vec = np.zeros(shape=(self.n_params,1))

            #print(compressed_model_vec.shape)
            #print(model_vec.shape)

            i=0
            print("kept entries: ", self.kept_model_vec_entries)
            for j in self.kept_model_vec_entries:
                model_vec[j] = compressed_model_vec[i]
                i+=1

            return model_vec
        
    def run_MCMC(self, run_point, true_model_vec_set=None, task_name=None, suffix=None):

        print("Running Bayesian MCMC...")
        print("task_name: ", task_name)

        for point in self.point_set:

            #if point != self.float_lonlat_to_int5d((-82.5, -78.6)):
            if point != run_point:
                continue
            
            print('The grid point is: ',point)

            # Obtain design matrix
            # row size: 
            # EN: 2 * num of offset fields
            # U:  1 * num of offset fields
            d_mat_EN_ta, d_mat_EN_tb, d_mat_U_ta, d_mat_U_tb = self.stack_design_mat_set[point]

            if task_name == "tides_1":
                remove_tidal_up = False
            elif task_name == "tides_2":
                remove_tidal_up = True
            else:
                raise Exception()

            # Subset the design matrix to remove some parameters according to prior 
            d_mat_EN_ta = self.remove_design_mat_cols(G=d_mat_EN_ta, secular_included=False, remove_tidal_up = remove_tidal_up)
            d_mat_EN_tb = self.remove_design_mat_cols(G=d_mat_EN_tb, secular_included=False, remove_tidal_up = remove_tidal_up)
            d_mat_U_ta = self.remove_design_mat_cols(G=d_mat_U_ta, secular_included=False, remove_tidal_up = remove_tidal_up)
            d_mat_U_tb = self.remove_design_mat_cols(G=d_mat_U_tb, secular_included=False, remove_tidal_up = remove_tidal_up)

            # Obtain data vector
            data_vec = self.data_vec_set[point]
            print('data_vec shape',data_vec.shape)

            N_data = len(data_vec)
            N_offsets = N_data//2

            # Obtain noise sigma
            noise_sigma = self.noise_sigma_set[point]

            # Obtain offsetfields
            offsetfields = self.offsetfields_set[point]

            # Form the necessary vectors
            vecs = np.zeros(shape=(N_data, 3))
            delta_t = np.zeros(shape=(N_offsets,1))
            t_origin = self.t_origin.date()
            
            for i in range(N_offsets):
                vecs[2*i,:] = np.asarray(offsetfields[i][2])
                vecs[2*i+1,:] = np.asarray(offsetfields[i][3])
                t_a = (offsetfields[i][0] - t_origin).days + round(offsetfields[i][4],4)
                t_b = (offsetfields[i][1] - t_origin).days + round(offsetfields[i][4],4)
                delta_t[i,0] = t_b - t_a

            # shape = (N_offsets ,3)
            delta_t = np.repeat(delta_t, 3, axis=1)

            tt_vecs = theano.shared(vecs)
            tt_delta_t = theano.shared(delta_t)

            # Form the observation vector matrix
            # shape: N_data x (N_offsets*3)
            vec_mat = np.zeros(shape=(N_data, N_offsets*3))
            for i in range(N_offsets):
                vec1 = np.asarray(offsetfields[i][2])
                vec2 = np.asarray(offsetfields[i][3])
                vec_mat[2*i,    3*i:3*(i+1)] = vec1
                vec_mat[2*i+1,  3*i:3*(i+1)] = vec2

            tt_vec_mat = theano.shared(vec_mat)
                
            #print('delta_t: ',delta_t)

            # Make the design matrix shared
            tt_d_mat_EN_ta = theano.shared(d_mat_EN_ta)
            tt_d_mat_EN_tb = theano.shared(d_mat_EN_tb)
            tt_d_mat_U_ta = theano.shared(d_mat_U_ta)
            tt_d_mat_U_tb = theano.shared(d_mat_U_tb)

            # Up displacement model
            up_scale = 1
            tide_height_master_model, tide_height_slave_model = self.up_disp_set[point]

            tide_height_master = tide_height_master_model * up_scale
            tide_height_slave = tide_height_slave_model * up_scale

            dis_U_ta = tide_height_master.reshape(len(tide_height_master),1)
            dis_U_tb = tide_height_slave.reshape(len(tide_height_slave),1)

            tt_dis_U_ta = theano.shared(dis_U_ta)
            tt_dis_U_tb = theano.shared(dis_U_tb)

            # Construct the parameter vector
            self.bmc_model = pm.Model()
            with self.bmc_model as model:

                # Find the number of tidal model parameters
                _ , P = d_mat_EN_ta.shape
    
                # E, N, U
                self.model_vec_secular = pm.Normal('secular', mu=0, sigma=2, shape=(1,3))
    
                # Tidal components
                self.model_vec_tidal = pm.Normal('tidal', mu=0, sigma=2, shape=(P,1))

                # Tidal components
                self.grounding = pm.Normal('grounding', mu=-1, sigma=2, shape=(1,1))

                dis_EN_ta = tt_d_mat_EN_ta.dot(self.model_vec_tidal)
                dis_EN_tb = tt_d_mat_EN_tb.dot(self.model_vec_tidal)

                if task_name == "tides_1":
                    # Based on parameters
                    dis_U_ta = tt_d_mat_U_ta.dot(self.model_vec_tidal)
                    dis_U_tb = tt_d_mat_U_tb.dot(self.model_vec_tidal)
                elif task_name == "tides_2":
                    # Based on external time series
                    dis_U_ta = tt_dis_U_ta 
                    dis_U_tb = tt_dis_U_tb
                else:
                    raise Exception()

                # Clipping U here
                dis_U_ta = tt.clip(dis_U_ta, self.grounding, 100)
                dis_U_tb = tt.clip(dis_U_tb, self.grounding, 100)

                # Find E & N offset
                offset_EN = dis_EN_tb - dis_EN_ta

                # Find U offset
                offset_U = dis_U_tb - dis_U_ta

                # Form 2d displacement
                # Reshape EN from vector to matrix
                # (N_offsets * 2, 1) -> (N_offsets, 2)
                offset_EN = offset_EN.reshape(shape=(N_offsets,2))

                # Form 3d displacement 
                # ENU: shape = (N_offsets, 3)
                offset_ENU = pm.math.concatenate([offset_EN, offset_U], axis=1)

                # Find secular displacement
                # model_vec_secular (1,3)
                # (N_offsets, 3) * (N_offsets, 3)
                offset_secular = self.model_vec_secular.repeat(N_offsets, axis=0) * tt_delta_t

                # Add secular displacement
                offset_total = offset_ENU + offset_secular

                # Flatten it to a vector
                offset_total_flatten = offset_total.reshape(shape=(N_offsets*3,1))

                # Multiply to observation
                # N_offsets * 3 -> N_data
                pred_vec = tt_vec_mat.dot(offset_total_flatten)
                
                # Observation
                obs = pm.Normal('obs', mu=pred_vec, sigma=self.sampling_data_sigma, observed=data_vec)

                print('Model compilaion is done')
                
                MAP_or_Sample = "Sample"
                
                # Find MAP solution
                map_estimate = pm.find_MAP(model=model)

                # Get model vec that satisfy to original definition
                model_vec, grounding = self.pad_to_orig_model_vec(map_estimate, mode="nonlinear")

                # Save the MAP solution
                pkl_name = "_".join([self.estimation_dir+"/map_estimate_BMC",str(point[0]),str(point[1]),suffix])
                with open(pkl_name + ".pkl","wb") as f:
                    pickle.dump(map_estimate,f)

                # Perform MCMC smapling
                n_steps = 6000
                #n_steps = 100
                if MAP_or_Sample == 'Sample':
                    #step = pm.NUTS()
                    trace = pm.sample(n_steps, tune=n_steps,chains=3)

                    # Save the true model vec to the trace object
                    if true_model_vec_set is not None:
                        true_model_vec = true_model_vec_set[point]

                        # secular
                        compressed_true_model_vec = true_model_vec[:3,0].tolist()

                        # tidal
                        for ind in self.kept_model_vec_entries:
                            compressed_true_model_vec.append(true_model_vec[ind,0])

                        trace.true_model_vec = np.asarray(compressed_true_model_vec)
                    else:
                        trace.true_model_vec = None

                    # Save the trace to disk
                    pkl_name = "_".join([self.estimation_dir+"/samples_BMC",str(point[0]),str(point[1]),suffix])
                    with open(pkl_name + ".pkl","wb") as f:
                        pickle.dump(trace,f)

                    # Plot the trace
                    pm.traceplot(trace)
                    plt.savefig(self.estimation_dir+'/MCMC_trace.png')

            # Only do the test point in the point set
            return (model_vec, grounding)

    #### Below are pure optimization using scipy ############
    def construct_bounds(self, point):

        self.bmc_model = pm.Model()
        RVs_secular = []
        RVs_tidal = []

        comps = ['e','n','u']

        model_prior = self.model_prior_set[point]

        no_secular_up = self.no_secular_up
        up_short_period = self.up_short_period
        horizontal_long_period = self.horizontal_long_period

        up_lower = self.up_lower
        up_upper = self.up_upper


        bounds = []
        # Secular component
        for i, comp in enumerate(comps):
            bounds.append( (model_prior[i,0] - max(abs(model_prior[i,0])*0.1, 1e-6), model_prior[i,0]+ max(abs(model_prior[i,0])*0.1,1e-6)) )

        # Tidal componens
        bound_permiss = 1
        bound_restrict = 1e-6
        comp_name = ['cosE','cosN','cosU','sinE','sinN','sinU']

        for i, tide_name in enumerate(self.modeling_tides):
            for j in range(6):
                k = 3 + i*6 + j
                if up_short_period and not tide_name in ['M2','S2','K2','O1','K1','P1'] and (j==2 or j==5):
                    bound = bound_restrict

                elif horizontal_long_period and not tide_name in ['Mf','Msf','Mm'] and (j==0 or j==1 or j==3 or j==4):
                    bound = bound_restrict
                else:
                    bound = bound_permiss

                bounds.append((-bound, bound))

        # Grouding component
        bounds.append((-2,0))

        return bounds

    def run_optimize(self):

        import scipy
        from scipy.optimize import shgo, differential_evolution, dual_annealing

        model_vec_set = {}

        for point in self.point_set:
            
            print(point)
            # Create bounds
            bounds = self.construct_bounds(point)

            # This is used to determine the number of parameters
            print('bounds: ', len(bounds))

            # Obtain design matrix
            # row size: N_offsets
            d_mat_EN_ta, d_mat_EN_tb, d_mat_U_ta, d_mat_U_tb = self.stack_design_mat_set[point]

            # Obtain data vector
            data_vec = self.data_vec_set[point]
            N_data = len(data_vec)
            N_offsets = N_data//2

            # Obtain offsetfields
            offsetfields = self.offsetfields_set[point]

            # Form the vectors
            vecs = np.zeros(shape=(N_data, 3))
            delta_t = np.zeros(shape=(N_offsets,1))
            t_origin = self.t_origin.date()
            for i in range(N_offsets):
                vecs[2*i,:] = np.asarray(offsetfields[i][2])
                vecs[2*i+1,:] = np.asarray(offsetfields[i][3])
                t_a = (offsetfields[i][0] - t_origin).days + round(offsetfields[i][4],4)
                t_b = (offsetfields[i][1] - t_origin).days + round(offsetfields[i][4],4)
                delta_t[i,0] = t_b - t_a

            delta_t = np.repeat(delta_t, 3, axis=1)
            
            # Form the observation vector matrix
            vec_mat = np.zeros(shape=(N_data, N_offsets*3))
            for i in range(N_offsets):
                vec1 = np.asarray(offsetfields[i][2])
                vec2 = np.asarray(offsetfields[i][3])
                vec_mat[2*i,    3*i:3*(i+1)] = vec1
                vec_mat[2*i+1,  3*i:3*(i+1)] = vec2
                
            pred_vec = np.zeros(shape=(N_data,1))
            args_1 = (d_mat_EN_ta, d_mat_EN_tb, d_mat_U_ta, d_mat_U_tb, data_vec, N_data, N_offsets, vecs, delta_t, vec_mat, pred_vec) 
        
            
            linear_design_mat = self.linear_design_mat_set[point]
            args_2 = (linear_design_mat, data_vec)

            #np.random.seed(20190711)
            #x_test = np.random.randn(len(bounds))

            #res1 = forward(x_test, *args_1)
            #res2 = forward_linear(x_test, *args_2)

            #print('comparison...')
            #print(res1, res2)

            #if (abs(res1 - res2<0.01)):
            #    print('successful')
            #else:
            #    raise Exception('forward problem has problem')

            res = differential_evolution(forward, bounds, args_1)
            #res = differential_evolution(forward_linear, bounds, args_2)

            print(res)

            model_vec_set[point] = res.x

        return model_vec_set


# Forward problem calculation used by non-linear optimization

def forward_linear(x, *T):

    G,d = T

    #print(G.shape)
    #print(x.shape)
    #print(d.shape)

    print(x)

    pred = np.matmul(G, x[:,None])

    return np.linalg.norm(pred - d)


#def forward(x, A1, A2, A3, A4, B1, B2, B3, B4, C1):
def forward(x, *T):

    d_mat_EN_ta, d_mat_EN_tb, d_mat_U_ta, d_mat_U_tb, data_vec, N_data, N_offsets, vecs, delta_t, vec_mat, pred_vec = T

    print(x)

    dis_EN_ta = np.matmul(d_mat_EN_ta, x[3:-1,None])
    dis_EN_tb = np.matmul(d_mat_EN_tb, x[3:-1,None])
    dis_U_ta = np.matmul(d_mat_U_ta, x[3:-1,None])
    dis_U_tb = np.matmul(d_mat_U_tb, x[3:-1,None])

    # Clipping
    dis_U_ta[dis_U_ta < x[-1]] = x[-1]
    dis_U_tb[dis_U_tb < x[-1]] = x[-1]

    offset_EN = dis_EN_tb - dis_EN_ta
    offset_U = dis_U_tb - dis_U_ta

    offset_ENU = np.hstack((offset_EN.reshape(N_offsets,2), offset_U))

    # Broadcast
    secular_ENU = np.repeat(x[None,0:3], N_offsets, axis=0) * delta_t

    # Total ENU
    total_ENU = offset_ENU + secular_ENU

    #for i in range(N_offsets):
    #    pred_vec[2*i] = np.dot(vecs[2*i], total_ENU[i])
    #    pred_vec[2*i+1] = np.dot(vecs[2*i+1], total_ENU[i])

    # Flatten
    total_ENU_flatten = total_ENU.reshape(N_offsets*3, 1)

    # Prediction
    pred_vec = np.matmul(vec_mat, total_ENU_flatten)

    return np.linalg.norm(pred_vec - data_vec)

if __name__=='__main__':
    BMC = Bayesian_MCMC()
    BMC.run_test1() 
