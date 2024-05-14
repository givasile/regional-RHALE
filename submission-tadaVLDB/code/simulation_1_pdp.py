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
# global PDP
pdp = effector.PDP(data=X_uncor, model=model, feature_names=['x1','x2','x3'], target_name="Y")
pdp.plot(feature=0, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/uncor_global_pdp.png',
            bbox_inches='tight')

# regional PDP
regional_pdp = effector.RegionalPDP(data=X_uncor, model=model, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)
regional_pdp.fit(features="all", heter_pcg_drop_thres=0.3, nof_candidate_splits_for_numerical=11, centering=True)

regional_pdp.show_partitioning(features=0)

regional_pdp.plot(feature=0, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/uncor_regional_pdp_1.png',
            bbox_inches='tight')
regional_pdp.plot(feature=0, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/uncor_regional_pdp_2.png',
            bbox_inches='tight')


# Correlated setting
# Global PDP
pdp = effector.PDP(data=X_cor, model=model, feature_names=['x1','x2','x3'], target_name="Y")
pdp.plot(feature=0, centering=True, show_avg_output=False, heterogeneity="ice", y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/cor_global_pdp.png',
            bbox_inches='tight')

# Regional-PDP
regional_pdp = effector.RegionalPDP(data=X_cor, model=model, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)
regional_pdp.fit(features="all", heter_pcg_drop_thres=0.4, nof_candidate_splits_for_numerical=11)
regional_pdp.show_partitioning(features=0)


regional_pdp.plot(feature=0, node_idx=1, heterogeneity="ice", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/cor_regional_pdp_1.png',
            bbox_inches='tight')
regional_pdp.plot(feature=0, node_idx=2, heterogeneity="ice", centering=True, y_limits=[-5, 5])
plt.savefig('./../latex/figures/simulation_1/cor_regional_pdp_2.png',
            bbox_inches='tight')
