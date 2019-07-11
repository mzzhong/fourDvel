#!/usr/bin/env python3

import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def test1():

    np.random.seed(123)
    
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of data set
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

    fig, axes = plt.subplots(1, 2, sharex = True, figsize=(10,4))
    axes[0].scatter(X1,Y)
    axes[1].scatter(X2,Y)

    fig.savefig('1.png')

    print('Running on PyMC3 v{}'.format(pm.__version__))

    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha',mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal('sigma',sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs',mu=mu, sigma=sigma, observed=Y)

        x = pm.Normal('x', mu=0, sigma=1)
        obs = pm.Normal('obs', mu=x, sigma=1, observed=np.random.randn(1000))

        #map_estimate = pm.find_MAP(model = basic_model)
        trace = pm.sample(500)

        print(len(trace['alpha']))
        pm.traceplot(trace)
        print(pm.summary(trace).round(2))
    
    return 0

def test2():

    gp_model = pm.Model()

    with gp_model:

        cov = np.array([[1., 0.5], [0.5, 2]])
        mu = np.zeros(2)
        vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=(5, 2))

        print(vals.size)

if __name__=="__main__":
    test2()
