#!/usr/bin/env python
# This simulation example compares RHALE vs PDP in a simple simulation example

import numpy as np
import effector
import matplotlib.pyplot as plt

def generate_dataset_uncorrelated(N):
    x1 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x3 = np.random.uniform(-1, 1, size=N)
    return np.stack((x1, x2, x3), axis=-1)

def generate_dataset_correlated(N):
    x1 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x3 = x1
    return np.stack((x1, x2, x3), axis=-1)

def model(x):
    f = np.where(x[:,2] > 0, 3*x[:,0] + x[:,2], -3*x[:,0] + x[:,2])
    return f

def model_jac(x):
    dy_dx = np.zeros_like(x)
    ind1 = x[:, 2] > 0
    ind2 = x[:, 2] <= 0
    dy_dx[ind1, 0] = 3
    dy_dx[ind2, 0] = -3
    dy_dx[:, 2] = 1
    return dy_dx


N = 10_000
X_uncor = generate_dataset_uncorrelated(N)
X_cor = generate_dataset_correlated(N)
Y_cor = model(X_cor)
Y_uncor = model(X_uncor)

# uncorrelated case
# global RHALE
rhale = effector.RHALE(
    data=X_uncor,
    model=model,
    model_jac=model_jac,
    feature_names=['x1','x2','x3'],
    target_name="Y"
)

binning_method = effector.binning_methods.Fixed(10)
rhale.fit(features="all", binning_method=binning_method, centering=True)
rhale.plot(feature=0, centering=True, heterogeneity="std", show_avg_output=False, y_limits=[-5, 5], dy_limits=[-5, 5])
# store without border
plt.savefig('./../latex/figures/simulation_1/uncor_global_rhale.png',
            bbox_inches='tight')

# regional RHALE
regional_rhale = effector.RegionalRHALE(
    data=X_uncor,
    model=model,
    model_jac= model_jac,
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)

binning_method = effector.binning_methods.Fixed(10)
regional_rhale.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    binning_method=binning_method,
    nof_candidate_splits_for_numerical=11
)

regional_rhale.show_partitioning(features=0)
regional_rhale.plot(feature=0, node_idx=1, heterogeneity="std", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/uncor_regional_rhale_1.png',
            bbox_inches='tight')
regional_rhale.plot(feature=0, node_idx=2, heterogeneity="std", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/uncor_regional_rhale_2.png',
            bbox_inches='tight')


# correlated case
rhale = effector.RHALE(
    data=X_cor,
    model=model,
    model_jac=model_jac, 
    feature_names=['x1','x2','x3'],
    target_name="Y",
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)
binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)

rhale.plot(
    feature=0,
    centering=True,
    heterogeneity="std",
    show_avg_output=False,
    y_limits=[-5, 5],
    dy_limits=[-5, 5]
)
plt.savefig('./../latex/figures/simulation_1/cor_global_rhale.png',
            bbox_inches='tight')


# Regional RHALE
regional_rhale = effector.RegionalRHALE(
    data=X_cor, 
    model=model, 
    model_jac= model_jac, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
regional_rhale.fit(
    features="all",
    heter_pcg_drop_thres=0.6,
    binning_method=binning_method,
    nof_candidate_splits_for_numerical=10
)

regional_rhale.show_partitioning(features=0)
# regional_rhale.plot(feature=0, node_idx=1, heterogeneity="std", centering=True, y_limits=[-5, 5])


