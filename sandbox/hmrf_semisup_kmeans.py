# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import matplotlib.pyplot as plt
import sys
import os
import copy
import pstats
import cProfile
import StringIO

sys.path.append('../djumble/')
# from djumble.hmrf_semisup_km import HMRFKmeans as HMRFKmeans
from hmrf_semisup_km_narray_cy import HMRFKmeans as HMRFKmeans_arr

test_dims = 1000

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
# print x_data_2d_arr

for xy in x_data_2d_arr1:
    plt.text(xy[0], xy[1], str(1),  color="black", fontsize=10)
for xy in x_data_2d_arr2:
    plt.text(xy[0], xy[1], str(2),  color="green", fontsize=10)
for xy in x_data_2d_arr3:
    plt.text(xy[0], xy[1], str(3),  color="blue", fontsize=10)
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
    [[1, 1, 1, 1, 7, 521, 521, 521, 535, 537, 1037, 1057, 1039, 1045, 1098, 1019, 1087],
     [5, 3, 6, 8, 3, 525, 528, 539, 525, 539,  1238,  1358, 1438, 1138, 1038, 1138, 1338]],
    dtype=np.int
)

cannot_lnk_con_arr = np.array(
    [[1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
      7, 7, 538, 548, 558, 738, 938, 838, 555],
     [521,  525,  528,  535,  537,  539,  521,  525,  528,  500,  521,  525,  528,  535,  537,
      539,  521,  535,  537,  539,  521,  525,  528,  535,  537,  539,  521,  525,  528,  535,
      537,  539, 1237, 1357, 1437, 1137, 1037, 1039, 1337]],
    dtype=np.int
)


k_clusters = 3

init_centrs = [set([0]), set([550]), set([1100])]
init_centrs_lst = np.array([0, 550, 1100])

print "Running HMRF Kmeans"
"""
hkmeans = HMRFKmeans(
    k_clusters, must_lnk_con, cannot_lnk_con,
    init_centroids=init_centrs, ml_wg=1.0, cl_wg=1.0, max_iter=300, cvg=0.001, lrn_rate=0.0003,
    ray_sigma=0.5, d_params=None, norm_part=False, globj='non-normed'
)

res = hkmeans.fit(copy.deepcopy(x_data_2d_arr))  # , set([50]))

# print res[1]
"""

# Prifilling
pr = cProfile.Profile()

# Prifilling - Starts
pr.enable()

hkmeans_arr = HMRFKmeans_arr(
    k_clusters, must_lnk_con_arr, cannot_lnk_con_arr,
    init_centroids=init_centrs_lst, ml_wg=1.0, cl_wg=1.0, max_iter=300, cvg=0.001, lrn_rate=0.0003,
    ray_sigma=0.5, d_params=None, norm_part=False, globj_norm=False
)

res = hkmeans_arr.fit(x_data_2d_arr, np.array([], dtype=np.int))

# print res[1]

# Prifilling - Ends
pr.disable()

# Prifilling - Stats
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats("time").print_stats()  # cumtime
print s.getvalue()



"""
for mu_idx in np.unique(res[1]):

    print mu_idx

    clstr_idxs = np.where(res[1] == mu_idx)[0]

    print clstr_idxs

    for xy in x_data_2d_arr[list(clstr_idxs)]:
        plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)

plt.show()

# for mu_idx, mu in enumerate(res[0]):
#
#     clstr_idxs = np.where(res[1] == mu_idx)[0]
#
#     for xy in x_data_2d_arr[clstr_idxs]:
#         plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)
#     # plt.plot(x_data_2d_arr2, '^')
#     # plt.plot(x_data_2d_arr3, '>')
#
# plt.show()
"""
