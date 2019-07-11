#!/usr/bin/env python3

import numpy as np

from basics import basics

class Bayesian_Linear(basics):

    pass


class Bayesian_MCMC(basics):

    def __init__(self):

        import theano.tensor as tt
        import pymc3 as pm

        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')

        self.bmc_model = pm.Model()

    def set_point_set(self, point_set):

        self.point_set = point_set

    def set_modeling_tides(self, modeling_tides):
    
        self.modeling_tides = modeling_tides

        self.n_params = 3 + 6 * len(modeling_tides)

    def set_model_priors(self, model_prior=None, no_secular_up=False, up_short_period=False, horizontal_long_period=False):

        self.model_cov

    def set_grounding_range(self, lower, upper):

        self.grounding_lower = lower
        self.grounding_upper = upper


    def set_data_covariance(self, invCd):

        self.invCd = invCd

    def set_data(self, data_vec_set):

        self.data_vec_set = data_vec_set

    def set_offsetfields(self, offsetfields_set):

        self.offsetfields_set = offsetfields_set

    def set_design_mat(self, design_mat_set):

        self.design_mat_set = design_mat_set


    def run(self):

        with self.bmc_model as bmc_model:

            # Construct data (using Data container)
            example_point = point_set[0]

            data = pm.Data('data', self.data_vec_set[example_point])

            mu = pm.Normal('mu', mu=0, sigma=10)

            y = pm.Normal('y', mu=mu, sigma=1, observed=data)

            # Construct model

            # Tidal
            cov = self.model_covariance
            mu = self.model_prior
            theta = pm.MvNormal('theta', mu=mu, cov=cov)

            # Grounding
            grounding_point = pm.Uniform(lower = self.grounding_lower, upper = self.grounding_upper)

            # Matrix product
            dis_ta = pm.math.dot(design_mat_ta, theta)
            dis_tb = pm.math.dot(design_mat_tb, theta)

            # Clipping
            pm.math.switch(dis_ta)

            # 

        # Using data container variables to fit the same model to several datasets
        traces = []
        for point in self.point_set:
            with bmc_model:
                pm.set_data({'data': self.data_vec_set[point]})
                traces.append(pm.sample())

        pass
