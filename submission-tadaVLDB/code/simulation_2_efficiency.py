import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import numpy as np
import effector
import tensorflow as tf
from tensorflow import keras
import time


def generate_dataset_uncorrelated(N, D):
    X = np.random.uniform(-1, 1, size=(N,D))
    return X

def create_model(D, L):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=(D,)))
    for _ in range(L):
        model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_model_jac(model):
    def model_jac(x):
        x = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
        return tape.gradient(y, x).numpy()# .squeeze()
    return model_jac


# N = 10_000
# D = 10
# X = generate_dataset_uncorrelated(N, D)

# exec_time_layers = {"rhale":[], "pdp":[], "ale":[]}
# l_list = [3, 5, 10, 20]
# for l in l_list:
#     print(f"Number of layers: {l}")
#     model = create_model(D, l)
#     model_jac = create_model_jac(model)
#     dY_dX = model_jac(X)

#     # rhale
#     print("RHALE")
#     reg_rhale = effector.RegionalRHALE(
#         data=X,
#         model=model,
#         model_jac=model_jac,
#         instance_effects=dY_dX,
#         nof_instances="all",
#         axis_limits=np.array([[-1,1]]*D).T,
#         feature_types=["numerical"]*D,
#     )
#     tic = time.time()
#     reg_rhale.fit(0, binning_method="fixed")
#     toc = time.time()
#     exec_time_layers["rhale"].append(toc-tic)
#     print(f"Regional RHALE: {toc-tic:.2f} seconds")

#     print("ALE")
#     reg_ale = effector.RegionalALE(
#         data=X,
#         model=model,
#         nof_instances="all",
#         axis_limits=np.array([[-1,1]]*D).T,
#         feature_types=["numerical"]*D,
#     )
#     tic = time.time()
#     reg_ale.fit(0, binning_method="fixed")
#     toc = time.time()
#     exec_time_layers["ale"].append(toc-tic)
#     print(f"Regional ALE: {toc-tic:.2f} seconds")

#     # pdp
#     print("PDP")
#     reg_pdp = effector.RegionalPDP(
#         data=X,
#         model=model,
#         nof_instances="all",
#         axis_limits=np.array([[-1,1]]*D).T,
#         feature_types=["numerical"]*D,
#     )

#     # measure execution time
#     tic = time.time()
#     reg_pdp.fit(0, use_vectorized=False)
#     toc = time.time()
#     exec_time_layers["pdp"].append(toc-tic)
#     print(f"Regional PDP: {toc-tic:.2f} seconds")

# plt.figure()
# plt.plot(l_list, exec_time_layers["rhale"], "--o", color="dodgerblue", label="Regional RHALE")
# plt.plot(l_list, exec_time_layers["ale"], "--o", color="red", label="Regional ALE")
# plt.plot(l_list, exec_time_layers["pdp"], "--o", color="black", label="Regional PDP")
# plt.xlabel("Model Complexity (Number of layers)")
# plt.ylabel("Time (seconds)")
# plt.legend()
# plt.title("Execution time (per feature)")
# plt.savefig("./../latex/figures/simulation_2/efficiency_layers.png")
# plt.show(block=False)

D = 10
L = 5
exec_time_N = {"rhale":[], "ale":[], "pdp":[]}
for N in [100, 1000, 10_000, 100_000]:
    print(f"Number of samples: {N}")
    X = generate_dataset_uncorrelated(N, D)
    model = create_model(D, L)
    model_jac = create_model_jac(model)

    # rhale
    dY_dX = model_jac(X)

    print("RHALE")
    reg_rhale = effector.RegionalRHALE(
        data=X,
        model=model,
        model_jac=model_jac,
        instance_effects=dY_dX,
        nof_instances="all",
        axis_limits=np.array([[-1,1]]*D).T,
        feature_types=["numerical"]*D,
    )

    tic = time.time()
    reg_rhale.fit(0, binning_method="fixed")
    toc = time.time()
    print(f"Regional RHALE: {toc-tic:.2f} seconds")
    exec_time_N["rhale"].append(toc-tic)

    print("ALE")
    reg_ale = effector.RegionalALE(
        data=X,
        model=model,
        nof_instances="all",
        axis_limits=np.array([[-1,1]]*D).T,
        feature_types=["numerical"]*D,
    )
    tic = time.time()
    reg_ale.fit(0, binning_method="fixed")
    toc = time.time()
    print(f"Regional ALE: {toc-tic:.2f} seconds")
    exec_time_N["ale"].append(toc-tic)

    # pdp
    reg_pdp = effector.RegionalPDP(
        data=X,
        model=model,
        nof_instances="all",
        axis_limits=np.array([[-1,1]]*D).T,
        feature_types=["numerical"]*D,
    )

    # measure execution time
    tic = time.time()
    reg_pdp.fit(0, use_vectorized=False)
    toc = time.time()
    print(f"Regional PDP: {toc-tic:.2f} seconds")
    exec_time_N["pdp"].append(toc-tic)

plt.figure()
plt.plot([100, 1000, 10_000, 100_000], exec_time_N["rhale"], "--o", color="dodgerblue", label="Regional RHALE")
plt.plot([100, 1000, 10_000, 100_000], exec_time_N["ale"], "--o", color="red", label="Regional ALE")
plt.plot([100, 1000, 10_000, 100_000], exec_time_N["pdp"], "--o", color="black", label="Regional PDP")
plt.xscale("log")
plt.xlabel("Number of instances")
plt.ylabel("Time (seconds)")
plt.legend()
plt.title("Execution time (per feature)")
plt.savefig("./../latex/figures/simulation_2/efficiency_samples.png")
plt.show(block=False)
