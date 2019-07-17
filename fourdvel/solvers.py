#!/usr/bin/env python3

import numpy as np

import theano
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt

import datetime

from basics import basics

plt.style.use('seaborn-darkgrid')

class Bayesian_Linear(basics):

    pass


class Bayesian_MCMC(basics):

    def __init__(self):

        super(Bayesian_MCMC, self).__init__()

    def set_point_set(self, point_set):

        self.point_set = point_set

    def set_modeling_tides(self, modeling_tides):
    
        self.modeling_tides = modeling_tides
        self.n_params = 3 + 6 * len(modeling_tides)

    def set_model_priors(self, model_prior_set=None, no_secular_up=False, up_short_period=False, horizontal_long_period=False, up_lower=None, up_upper=None):

        #with self.bmc_model as model:
        #    secular_v = 

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


    def run_test(self):

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

    def construct_model_vec(self, point):

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

        with self.bmc_model as model:

            # Secular component
            for i, comp in enumerate(comps):
                rv = pm.Normal('Secular_'+ comp, mu=model_prior[i], sigma=max(0.1*abs(model_prior[i]),0.0001), shape=(1,1), testval=model_prior[i])
                RVs_secular.append(rv)

            self.model_vec_secular = pm.math.concatenate(RVs_secular, axis=0)


            # Tidal componens
            sigma_permiss = 10
            sigma_restrict = 0.0001
            comp_name = ['cosE','cosN','cosU','sinE','sinN','sinU']
            for i, tide_name in enumerate(self.modeling_tides):
                for j in range(6):
                    k = 3 + i*6 + j
                    if up_short_period and not tide_name in ['M2','S2','O1'] and (j==2 or j==5):
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
            self.grounding = pm.Uniform('grounding', lower= up_lower, upper = up_upper, testval=-1)


        print('secular parameter vector length: ', len(RVs_secular))
        print('tidal parameter vector length: ', len(RVs_tidal))

        return 0

    def run_MCMC(self):

        print('Running Bayesian MCMC...')

        for point in self.point_set:
            print('The grid point is: ',point)

            # Construct the parameter vector
            self.construct_model_vec(point)

            # Obtain design matrix
            d_mat_EN_ta, d_mat_EN_tb, d_mat_U_ta, d_mat_U_tb = self.stack_design_mat_set[point]

            # Obtain data vector
            data_vec = self.data_vec_set[point]
            print('data_vec shape',data_vec.shape)

            N_data = len(data_vec)
            N_offsets = N_data//2

            # Obtain noise sigma
            noise_sigma = self.noise_sigma_set[point]

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
 
            tt_vecs = theano.shared(vecs)
            tt_delta_t = theano.shared(delta_t)

            # Form the observation vector matrix
            vec_mat = np.zeros(shape=(N_data, N_offsets*3))
            for i in range(N_offsets):
                vec1 = np.asarray(offsetfields[i][2])
                vec2 = np.asarray(offsetfields[i][3])
                vec_mat[2*i,    3*i:3*(i+1)] = vec1
                vec_mat[2*i+1,  3*i:3*(i+1)] = vec2

            tt_vec_mat = theano.shared(vec_mat)
                
            #print('delta_t: ',delta_t)

            # Make design matrix shared
            tt_d_mat_EN_ta = theano.shared(d_mat_EN_ta)
            tt_d_mat_EN_tb = theano.shared(d_mat_EN_tb)
            tt_d_mat_U_ta = theano.shared(d_mat_U_ta)
            tt_d_mat_U_tb = theano.shared(d_mat_U_tb)

            self.model_vec_secular = self.model_vec_secular.T

            with self.bmc_model as model:
                
                dis_EN_ta = tt_d_mat_EN_ta.dot(self.model_vec_tidal)
                dis_EN_tb = tt_d_mat_EN_tb.dot(self.model_vec_tidal)

                dis_U_ta = tt_d_mat_U_ta.dot(self.model_vec_tidal)
                dis_U_tb = tt_d_mat_U_tb.dot(self.model_vec_tidal)

                # Clipping here ...
                #dis_U_ta = tt.clip(dis_U_ta, self.grounding, 100)
                #dis_U_tb = tt.clip(dis_U_tb, self.grounding, 100)

                # offset
                offset_EN = dis_EN_tb - dis_EN_ta
                offset_U = dis_U_tb - dis_U_ta

                # Form 2d displacement
                offset_EN = offset_EN.reshape(shape=(N_offsets,2))

                # Form 3d displacement
                offset_ENU = pm.math.concatenate([offset_EN, offset_U], axis=1)

                # secular velocity
                offset_secular = self.model_vec_secular.repeat(N_offsets, axis=0) * tt_delta_t

                # Add secular velocity
                offset_total = offset_ENU + offset_secular

                # Flatten it to a vector
                offset_total_flatten = offset_total.reshape(shape=(N_offsets*3,1))

                # Multiply to observation

                pred_vec = tt_vec_mat.dot(offset_total_flatten)
                
                # Observation
                obs = pm.Normal('obs', mu=pred_vec, sigma=0.02, observed=data_vec)

                print('Model is built')
                
                MAP_or_sample = "sample"
                if MAP_or_sample == 'MAP':
                    map_estimate = pm.find_MAP(model=model)
                    print(map_estimate)
                elif MAP_or_sample == 'sample':
                    #step = pm.Metropolis()
                    #step = pm.HamiltonianMC()
                    #step = pm.NUTS()
                    trace = pm.sample(1000, cores = 4)

                    pm.traceplot(trace)
                    plt.savefig('bayes_2.png')
                    print(trace)

            return 0

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
                if up_short_period and not tide_name in ['M2','S2','O1'] and (j==2 or j==5):
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
    BMC.run_test0() 
