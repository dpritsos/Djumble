# The following comment enables the use of utf-8 within the script.
# coding=utf-8

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd

import matplotlib.pyplot as plt


def HMRFKmeans(x_arr, constr, const_violat_w, dist_measure):
    """HMRF Kmeans: A Semi-supervised clustering algorithm based on Hidden Markov Random Fields
        Clustering model optimised by Expectation Maximisation (EM) algorithm with Hard clustering
        constraints, i.e. a Kmeans Semi-supervised clustering variant.
    """

    k_centroids = init_cluster()


def ICM():
    """ICM: Iterated Conditional Modes (for the E-Step)
    """

    pass


def FarFirstWeighted(x_data_arr, k_expect, must_lnk_con, cannnot_lnk_con, CosDist):
    """
    """
    pass


def FarFirstCosntraint(x_data_arr, k_expect, must_lnk_cons, cannnot_lnk_cons, distor_measure):
    """
        pick any z ∈ S and set T = {z}
        while |T| < k:
            z = arg maxx∈S ρ(x, T)
            T = T ∪ {z}

        Here ρ(x, T) is the distance from point x to the closest point in set T,
        that is to say, infz∈T ρ(x, z).

    """

    # Initiating the list of array indices for all forthcoming neighbourhoods Np.
    neibs_lsts = [[]]

    data_num = x_data_arr.shape[0]

    # Adding a random point in the neighbourhood N0.
    rnd_idx = np.random.randint(0, data_num)

    neibs_lsts[0].append(rnd_idx)
    neib_c = 1

    farthest_x_idx = data_num + 99  # Not sure for this initialization.

    # Initialising for finding the farthest x array index form all N neighbourhoods.

    all_neibs = []

    while neib_c < k_expect and len(all_neibs) < data_num:

        max_dist = 0

        # Getting the farthest x from all neighbourhoods.
        for i, x in enumerate(x_data_arr):

            all_neibs = [idx for neib in neibs_lsts for idx in neib]

            for neib_x_idx in all_neibs:

                    if i not in all_neibs:

                        dist = distor_measure(x_data_arr[neib_x_idx], x)

                        if dist > max_dist:
                            max_dist = dist
                            farthest_x_idx = i

        # Looking for Must-Link
        must_link_neib_indx = None
        if farthest_x_idx in must_lnk_cons.keys():
            for ml_idx in must_lnk_cons[farthest_x_idx]:
                for n_idx, neib in enumerate(neibs_lsts):
                    if ml_idx in neib:
                        must_link_neib_indx = n_idx

        # Looking for Cannot-Link
        cannot_link = False
        if farthest_x_idx in cannnot_lnk_cons.keys():
            for cl_idx in cannnot_lnk_cons[farthest_x_idx]:
                for neib in neibs_lsts:
                    if cl_idx in neib:
                        cannot_link = True

        # Putting the x in the proper N neighbourhood.
        if must_link_neib_indx:

            neibs_lsts[must_link_neib_indx].append(farthest_x_idx)

        elif cannot_link:

            neib_c += 1
            neibs_lsts.append([farthest_x_idx])

        else:
            neibs_lsts[neib_c-1].append(farthest_x_idx)

    return neibs_lsts


def ConsolidateAL(neibs_lsts, x_data_arr, must_lnk_cons, distor_measure):
    """
    """

    #
    data_num = x_data_arr.shape[0]

    # Estimating centroids.

    # print np.mean(x_data_arr[[1,2,3], :], axis=0)
    neibs_mu = [np.mean(x_data_arr[neib, :], axis=0) for neib in neibs_lsts]

    cnt = 0
    for rnd_idx in range(data_num):

        cnt += 1

        # rnd_idx = np.random.randint(0, data_num)
        # Ascending order.
        srted_dists_neib_idx = np.argsort(
            [distor_measure(mu, x_data_arr[rnd_idx, :])[0, 0] for mu in neibs_mu],
            axis=0
        )

        for neib_idx in srted_dists_neib_idx:
            if rnd_idx in must_lnk_cons.keys():
                for ml_idx in must_lnk_cons[rnd_idx]:
                    if ml_idx in neibs_lsts[neib_idx] and rnd_idx not in neibs_lsts[neib_idx]:
                        neibs_lsts[neib_idx].append(rnd_idx)

    print neibs_lsts


def InitClustering():
    pass


def CosDistPar(x1, x2, distor_params):
    """CosDistPar: Cosine Distance with distortion parameters based on 'Soft Cosine Measure' where
        a weighting schema is the distortion parameters diagonal matrix A. Note that A matrix
        (diagonal) is expected as vector argument in this function.
    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)
    A = sp.diag(distor_params)

    return 1 - (x1 * A * x2.T / (np.sqrt(x1 * A * x1.T) * np.sqrt(x2 * A * x2.T)))


def CosDist(x1, x2):
    """
        Note: I the above function is equivalent if A is set to be the I identity matrix.

    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    return 1 - (x1 * x2.T / (np.sqrt(x1 * x1.T) * np.sqrt(x2 * x2.T)))


def JObjCosDM(x_idx, x_data_arr, mu, mu_neib_idxs_lst,
              must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params):
    """JObjCosDM: J Objective function for Cosine Distortion Measure. It cannot very generic
        because the gradient decent (partial derivative) calculation should be applied which they
        are totally dependent on the distortion measure, here Cosine Distance.

    """

    "Phi_max depends on the distortion measure"

    # #### NOT BEEN DEBUGED YET! #### #

    d = CosDistPar(x_data_arr[x_idx, :], mu, distor_params)

    # Calculating Must-Link violation cost.
    ml_cost = 0
    if x_idx in must_lnk_cons.keys():
        for x_muneib_idx in mu_neib_idxs_lst:
            if x_muneib_idx in must_lnk_cons[x_idx]:
                ml_cost += w_constr_viol_mtrx[x_idx, x_muneib_idx] *\
                           CosDistPar(x_data_arr[x_idx, :], x_data_arr[x_muneib_idx, :],
                                      distor_params)

    # Calculating Cannot-Link violation cost.
    cl_cost = 0.0
    if x_idx in cannot_lnk_cons.keys():
        for x_muneib_idx in mu_neib_idxs_lst:
            if x_muneib_idx in cannot_lnk_cons[x_idx]:
                ml_cost += w_constr_viol_mtrx[x_idx, x_muneib_idx] *\
                           (1 - CosDistPar(x_data_arr[x_idx, :], x_data_arr[x_muneib_idx, :],
                                           distor_params))

    return d + ml_cost + cl_cost


def MuCosDMPar(x_data_arr, neibs_idxs_lsts, distor_params):
    """
    """
    A = sp.diag(distor_params)

    mu_lst = list()
    for neibs_idxlst in neibs_idxs_lsts:

        xi_neib_sum = np.sum(x_data_arr[neibs_idxlst, :], axis=0)
        xi_neib_sum = sp.matrix(xi_neib_sum)

        # Calculating denominator ||Σ xi||(A)
        parametrized_norm_xi = np.sqrt(xi_neib_sum * A * xi_neib_sum.T)

        mu_lst.append(xi_neib_sum / parametrized_norm_xi)

    return mu_lst


def UpdateDistorParams(distor_params, chang_rate, x_data_arr, mu_lst,
                       neib_idxs_lst, must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx,):
    """
    """
    # #### HERE IS THE TRICKY THING #### #
    # I think this is the last hard thing to figure out!
    # i is for x
    # j is for μ of the neib where x is into

    for a_idx, a in distor_params:

        # Calculating Partial Derivative of D(xi, mu).
        xm_pderiv = 0.0
        for mu, neib_idxs in zip(mu_lst, neib_idxs_lst):
            for x_neib_idx in neib_idxs:
                xm_pderiv += PartialDerivative(a_idx, x_data_arr[x_neib_idx], mu, distor_params)

        # Calculating Partial Derivative of D(xi, xj) of Must-Link Constraints.
        mlcost_pderiv = 0.0
        for x_idx in range(x_data_arr.shape[0]):
            if x_idx in must_lnk_cons.keys():
                for x_neib_idx in [idx for neib in neib_idxs_lst for idx in neib]:
                    if x_neib_idx in must_lnk_cons[x_idx]:
                        mlcost_pderiv += w_constr_viol_mtrx[x_idx, x_neib_idx] *\
                            PartialDerivative(a_idx, x_data_arr[x_idx],
                                              x_data_arr[x_neib_idx], distor_params)

        # Calculating Partial Derivative of D(xi, xj) of Cannot-Link Constraints.
        clcost_pderiv = 0.0
        for x_idx in range(x_data_arr.shape[0]):
            if x_idx in cannot_lnk_cons.keys():
                for x_neib_idx in [idx for neib in neib_idxs_lst for idx in neib]:
                    if x_neib_idx in cannot_lnk_cons[x_idx]:
                        clcost_pderiv += w_constr_viol_mtrx[x_idx, x_neib_idx] *\
                            PartialDerivative(a_idx, x_data_arr[x_idx],
                                              x_data_arr[x_neib_idx], distor_params)

        # Changing the a dimension of A = np.diag(distortions-measure-parameters)
        distor_params[a_idx] = a + chang_rate * (xm_pderiv + mlcost_pderiv + clcost_pderiv)

    return distor_params


def PartialDerivative(a_idx, x1, x2, distor_params):
    """
    """

    A = sp.diag(distor_params)
    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = np.sqrt(x1 * A * x1.T)
    x2_pnorm = np.sqrt(x1 * A * x1.T)

    return ((x1[a_idx] * x2[a_idx] * x1_pnorm * x1_pnorm) - x1 * A * x2.T *
            ((np.square(x1[a_idx]) * x2_pnorm + np.square(x2[a_idx]) * x1_pnorm) /
            (2 * x1_pnorm * x2_pnorm))) / (np.square(x1_pnorm) * np.square(x2_pnorm))


if __name__ == '__main__':

    # x1 = np.array([0.1, 0.7, 0.2, 0.8], dtype=np.float32)
    # x2 = np.array([0.2, 0.5, 0.2, 0.2], dtype=np.float32)
    # dA = np.array([0.9, 0.1, 0.3, 1], dtype=np.float32)
    # print CosDistPar(x1, x2, dA)

    # sps.vonmises.rvs(kappa, loc=0, scale=1, size=1)

    x_data_2d_arr1 = np.random.vonmises(0.1, 1000, size=(20, 2))
    x_data_2d_arr2 = np.random.vonmises(0.5, 1000, size=(20, 2))
    x_data_2d_arr3 = np.random.vonmises(0.3, 1000, size=(20, 2))

    # x_data_2d_arr3 = np.random.vonmises(0.9, 0.3, size=(20, 2))

    x_data_2d_arr = np.vstack((x_data_2d_arr1, x_data_2d_arr2, x_data_2d_arr3))

    plt.plot(x_data_2d_arr1[:, 0], x_data_2d_arr1[:, 1], '*')
    plt.plot(x_data_2d_arr2[:, 0], x_data_2d_arr2[:, 1], '>')
    plt.plot(x_data_2d_arr3[:, 0], x_data_2d_arr2[:, 1], '>')
    # plt.plot(x_data_2d_arr2, '^')
    # plt.plot(x_data_2d_arr3, '>')
    plt.show()

    must_lnk_con = {
        1: [5, 3, 6, 8],
        5: [1, 6, 8],
        8: [1, 5],
        3: [1, 7],
        6: [1, 5],
        7: [3],
        21: [25, 28, 39],
        25: [21, 35, 28],
        28: [21, 25],
        35: [25],
        37: [39],
        39: [21, 37]
    }

    cannnot_lnk_con = {
        1: [21, 25, 28, 35, 37, 39],
        5: [21, 25, 28, 35],
        8: [21, 25, 28, 35, 37, 39],
        3: [21, 35, 37, 39],
        6: [21, 25, 28, 35, 37, 39],
        7: [21, 25, 28, 35, 37, 39],
        21: [1, 5, 8, 3, 6, 7],
        25: [1, 3, 6, 7],
        28: [1, 5, 8, 3, 6, 7],
        35: [1, 5, 8, 3, 6, 7],
        37: [1, 5, 8, 3],
        39: [1, 5, 8, 3, 6, 7]
    }

    k_expect = 3

    neibs_lsts = FarFirstCosntraint(x_data_2d_arr, k_expect, must_lnk_con, cannnot_lnk_con, CosDist)

    print neibs_lsts

    print ConsolidateAL(neibs_lsts, x_data_2d_arr, must_lnk_con, CosDist)

