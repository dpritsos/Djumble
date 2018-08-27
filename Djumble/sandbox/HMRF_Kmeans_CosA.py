#! /usr/bin/env python
# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import matplotlib.pyplot as plt
import scipy.special as special
import multiprocessing as mp
from multiprocessing import sharedctypes as mp_sct
import time as tm
import sys
import os
import copy
import warnings


import pstats
import cProfile
import StringIO


sys.path.append('../')
import djumble.semisupkmeans as sskm


test_dims = 10

print "Creating Sample"
x_data_2d_arr1 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))
x_data_2d_arr2 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))
x_data_2d_arr3 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))

x_data_2d_arr1 = x_data_2d_arr1 / np.max(x_data_2d_arr1, axis=1).reshape(500, 1)
x_data_2d_arr2 = x_data_2d_arr2 / np.max(x_data_2d_arr2, axis=1).reshape(500, 1)
x_data_2d_arr3 = x_data_2d_arr3 / np.max(x_data_2d_arr3, axis=1).reshape(500, 1)

# print x_data_2d_arr1

# (0.7, 0.2, 0.7, 0.2, 0.6, 0.6, 0.1, 0.3, 0.8, 0.5)
# (0.6, 0.6, 0.7, 0.2, 0.6, 0.6, 0.8, 0.3, 0.9, 0.1)
# (0.2, 0.3, 0.7, 0.2, 0.6, 0.6, 0.2, 0.3, 0.6, 0.4)

# tuple(np.random.normal(0.0, 10.0, size=2))
# x_data_2d_arr1 = np.random.vonmises(0.5, 100, size=(20, 2))
# x_data_2d_arr2 = np.random.vonmises(0.5, 1000, size=(20, 2))
# x_data_2d_arr3 = np.random.vonmises(0.5, 10000, size=(20, 2))

x_data_2d_arr = np.vstack((x_data_2d_arr1, x_data_2d_arr2, x_data_2d_arr3))
print x_data_2d_arr

for xy in x_data_2d_arr1:
    plt.text(xy[0], xy[1], str(1),  color="black", fontsize=20)
for xy in x_data_2d_arr2:
    plt.text(xy[0], xy[1], str(2),  color="green", fontsize=20)
for xy in x_data_2d_arr3:
    plt.text(xy[0], xy[1], str(3),  color="blue", fontsize=20)
# plt.text(x_data_2d_arr2[:, 0], x_data_2d_arr2[:, 1], str(2),  color="red", fontsize=12)
# plt.text(x_data_2d_arr3[:, 0], x_data_2d_arr3[:, 1], str(3),  color="red", fontsize=12)
# plt.show()
# 0/0
# plt.show()

must_lnk_con = [
    set([1, 5]), set([1, 3]), set([1, 6]), set([1, 8]), set([7, 3]), set([521, 525]),
    set([521, 528]), set([521, 539]), set([535, 525]), set([537, 539]), set([1037, 1238]),
    set([1057, 1358]), set([1039, 1438]), set([1045, 1138]), set([1098, 1038]), set([1019, 1138]),
    set([1087, 1338])
]

cannot_lnk_con = [
    set([1, 521]), set([1, 525]), set([1, 528]), set([1, 535]), set([1, 537]), set([1, 539]),
    set([5, 521]), set([5, 525]), set([5, 528]), set([5, 535]), set([8, 521]), set([8, 525]),
    set([8, 528]), set([8, 535]), set([8, 537]), set([8, 539]), set([3, 521]), set([3, 535]),
    set([3, 537]), set([3, 539]), set([6, 521]), set([6, 525]), set([6, 528]), set([6, 535]),
    set([6, 537]), set([6, 539]), set([7, 521]), set([7, 525]), set([7, 528]), set([7, 535]),
    set([7, 537]), set([7, 539]), set([538, 1237]), set([548, 1357]), set([558, 1437]),
    set([738, 1137]), set([938, 1037]), set([838, 1039]), set([555, 1337])
]

must_lnk_con_arr = np.array(
    [
        [1, 5], [1, 3], [1, 6], [1, 8], [7, 3], [521, 525], [521, 528],
        [521, 539], [535, 525], [537, 539], [1037, 1238], [1057, 1358],
        [1039, 1438], [1045, 1138], [1098, 1038], [1019, 1138], [1087, 1338]
    ], dtype=np.int
)

cannot_lnk_con_arr = np.array(
    [
        [1, 521], [1, 525], [1, 528], [1, 535], [1, 537], [1, 539], [5, 521], [5, 525],
        [5, 528], [5, 500], [8, 521], [8, 525], [8, 528], [8, 535], [8, 537], [8, 539],
        [3, 521], [3, 535], [3, 537], [3, 539], [6, 521], [6, 525], [6, 528], [6, 535],
        [6, 537], [6, 539], [7, 521], [7, 525], [7, 528], [7, 535], [7, 537], [7, 539],
        [538, 1237], [548, 1357], [558, 1437], [738, 1137], [938, 1037], [838, 1039],
        [555, 1337]
    ], dtype=np.int
)


k_clusters = 3

init_centrs_arr = [0, 550, 1100]

# Prifilling
# pr = cProfile.Profile()

# Prifilling - Starts
# pr.enable()

print "Running HMRF Kmeans"

hkmeans = sskm.HMRFKmeansSemiSup(
    k_clusters, must_lnk_con_arr, cannot_lnk_con_arr, init_centroids=init_centrs_arr,
    ml_wg=1.0, cl_wg=1.0, max_iter=50, cvg=0.001, lrn_rate=0.03, ray_sigma=2.5, d_params=None,
    icm_max_i=1000, enable_norm=False
)

res = hkmeans.fit(copy.deepcopy(x_data_2d_arr))

print res[1]
print hkmeans.get_params()

"""
ssEM = sskm.StochSemisupEM(
    k_clusters, must_lnk_con_arr, cannot_lnk_con_arr, init_centroids=init_centrs_arr,
    ml_wg=1.0, cl_wg=1.0, max_iter=20, cvg=0.001, icm_max_i=10,
    min_efl=0.5, max_efl=1.5, step_efl=0.1, eft_per_i=10
)

res = ssEM.fit(copy.deepcopy(x_data_2d_arr))

print res[1]
print ssEM.get_params()

"""


# Prifilling - Ends
# pr.disable()

# Prifilling - Stats
# s = StringIO.StringIO()
# ps = pstats.Stats(pr, stream=s)
# ps.sort_stats("time").print_stats()  # cumtime

# print s.getvalue()


clstr_neigh = dict()
for idx, mu_idx in enumerate(res[1]):

    if mu_idx in clstr_neigh:
        clstr_neigh[mu_idx].append(idx)
    else:
        clstr_neigh[mu_idx] = [idx]

    print mu_idx, idx

print clstr_neigh

"""

    for xy in x_data_2d_arr[list(clstr_idxs)]:
        plt.text(xy[0], xy[1], str(mu_idx), color='red', fontsize=15)
    # plt.plot(x_data_2d_arr2, '^')
    # plt.plot(x_data_2d_arr3, '>')

plt.show()


for mu_idx, mu in enumerate(res[0]):

    clstr_idxs = np.where(res[1] == mu_idx)[0]

    for xy in x_data_2d_arr[clstr_idxs]:
        plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)
    # plt.plot(x_data_2d_arr2, '^')
    # plt.plot(x_data_2d_arr3, '>')

plt.show()
"""
