# The following comment enables the use of utf-8 within the script.
# coding=utf-8

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd

import matplotlib.pyplot as plt


def HMRFKmeans(k_expect, x_data_arr, must_lnk_cons, cannot_lnk_cons, dmeasure_noparam,
               distor_measure, distor_params, w_constr_viol_mtrx, dparmas_chang_rate):
    """HMRF Kmeans: A Semi-supervised clustering algorithm based on Hidden Markov Random Fields
        Clustering model optimised by Expectation Maximisation (EM) algorithm with Hard clustering
        constraints, i.e. a Kmeans Semi-supervised clustering variant.
    """

    # Initializing clustering
    # neibs_sets = FarFirstCosntraint(x_data_arr, k_expect, must_lnk_cons,
    #                                           cannot_lnk_cons, dmeasure_noparam)
    # mu_neib_idxs_set_lst = ConsolidateAL(neibs_sets, x_data_arr,
    #                                      must_lnk_cons, dmeasure_noparam)
    # print mu_neib_idxs_set_lst.
    # 0/0
    mu_neib_idxs_set_lst = [set([0]),
                            set([550]),
                            set([1100])]
    mu_lst = MuCosDMPar(x_data_arr, mu_neib_idxs_set_lst, distor_params)

    # EM algorithm execution.
    # While no convergence yet or X times.
    for conv_step in range(30):

        # The E-Step ######
        no_change_cnt = 0
        while no_change_cnt < 2:

            # Calculating the new Neighbourhoods/Clusters.
            for x_idx in np.random.randint(0, x_data_arr.shape[0], size=x_data_arr.shape[0]):

                mu_neib_idx = ICM(x_idx, x_data_arr, mu_lst, mu_neib_idxs_set_lst,
                                  must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx,
                                  distor_params)

                if x_idx not in mu_neib_idxs_set_lst[mu_neib_idx]:

                    # Remove x form all Clusters.
                    for midx, mu_neib_idxs_set in enumerate(mu_neib_idxs_set_lst):
                        # mu_neib_idxs_set.discard(x_idx)
                        mu_neib_idxs_set_lst[midx].discard(x_idx)

                    mu_neib_idxs_set_lst[mu_neib_idx].add(x_idx)

                    no_change = False

                else:
                    no_change = True

            if no_change:
                no_change_cnt += 1

        # The M-Step #######

        # Recalculating centroids.
        mu_lst = MuCosDMPar(x_data_arr, mu_neib_idxs_set_lst, distor_params)
        print mu_lst

        # Re-estimating distortion measure parameters.
        distor_params = UpdateDistorParams(distor_params, dparmas_chang_rate, x_data_arr, mu_lst,
                                           mu_neib_idxs_set_lst, must_lnk_cons,
                                           cannot_lnk_cons, w_constr_viol_mtrx,)

    # Returning the Centroids, Clusters/Neighbourhoods, distortion parameters,
    # constraint violations matrix.
    return mu_lst, mu_neib_idxs_set_lst, distor_params, w_constr_viol_mtrx


def ICM(x_idx, x_data_arr, mu_lst, mu_neib_idxs_set_lst,
        must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params):
    """ ICM: Iterated Conditional Modes (for the E-Step)

        After all points are assigned, they are randomly re-ordered, and
        the assignment process is repeated. This process proceeds until no
        point changes its cluster assignment between two successive iterations.

    """
    last_jobj = 999999.0

    for i, (mu, mu_neib_idxs_set) in enumerate(zip(mu_lst, mu_neib_idxs_set_lst)):

        j_obj = JObjCosDM(x_idx, x_data_arr, mu, mu_neib_idxs_set,
                          must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params)

        if j_obj < last_jobj:
            last_jobj = j_obj
            neib_idx = i

    # Returning the i index, i.e. the neighbourhood/cluster where x point should be assigned into.
    return neib_idx


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
    neibs_sets = [set([])]

    data_num = x_data_arr.shape[0]

    # Adding a random point in the neighbourhood N0.
    rnd_idx = np.random.randint(0, data_num)

    neibs_sets[0].add(rnd_idx)
    neib_c = 1

    farthest_x_idx = data_num + 99  # Not sure for this initialization.

    # Initialising for finding the farthest x array index form all N neighbourhoods.

    all_neibs = []
    while neib_c < k_expect and len(all_neibs) < data_num:

        max_dist = 0
        # Getting the farthest x from all neighbourhoods.
        for i in np.random.randint(0, x_data_arr.shape[0], size=x_data_arr.shape[0]/10):

            all_neibs = [idx for neib in neibs_sets for idx in neib]

            for neib_x_idx in all_neibs:

                    if i not in all_neibs:

                        dist = distor_measure(x_data_arr[neib_x_idx], x_data_arr[i])

                        if dist > max_dist:
                            max_dist = dist
                            farthest_x_idx = i

        # Looking for Must-Link
        must_link_neib_indx = None
        if farthest_x_idx in must_lnk_cons:
            for ml_idx in must_lnk_cons[farthest_x_idx]:
                print "ML", ml_idx
                for n_idx, neib in enumerate(neibs_sets):
                    if ml_idx in neib:
                        must_link_neib_indx = n_idx

        # Looking for Cannot-Link
        cannot_link = False
        if farthest_x_idx in cannnot_lnk_cons:
            for cl_idx in cannnot_lnk_cons[farthest_x_idx]:
                print "CL", cl_idx
                for neib in neibs_sets:
                    if cl_idx in neib:
                        cannot_link = True

        # Putting the x in the proper N neighbourhood.
        if must_link_neib_indx:

            neibs_sets[must_link_neib_indx].add(farthest_x_idx)

        elif cannot_link:

            neib_c += 1
            neibs_sets.append(set([farthest_x_idx]))

        else:
            neibs_sets[neib_c-1].add(farthest_x_idx)

    return neibs_sets


def ConsolidateAL(neibs_sets, x_data_arr, must_lnk_cons, distor_measure):
    """
    """
    # Estimating centroids.
    # print np.mean(x_data_arr[[1,2,3], :], axis=0)
    neibs_mu = [np.mean(x_data_arr[neib, :], axis=0) for neib in neibs_sets]

    cnt = 0

    # I think that randomization factor is required  replacing --> # range(data_num):
    for rnd_idx in np.random.randint(0, x_data_arr.shape[0], size=x_data_arr.shape[0]):

        cnt += 1

        # Ascending order.
        srted_dists_neib_idx = np.argsort(
            [distor_measure(mu, x_data_arr[rnd_idx, :])[0, 0] for mu in neibs_mu],
            axis=0
        )

        for neib_idx in srted_dists_neib_idx:
            if rnd_idx in must_lnk_cons:
                for ml_idx in must_lnk_cons[rnd_idx]:
                    if ml_idx in neibs_sets[neib_idx] and rnd_idx not in neibs_sets[neib_idx]:
                        neibs_sets[neib_idx].append(rnd_idx)

    return neibs_sets


def CosDistPar(x1, x2, distor_params):
    """CosDistPar: Cosine Distance with distortion parameters based on 'Soft Cosine Measure' where
        a weighting schema is the distortion parameters diagonal matrix A. Note that A matrix
        (diagonal) is expected as vector argument in this function.
    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)
    A = sp.diag(distor_params)

    return 1 - (x1 * A * x2.T / (np.sqrt(np.abs(x1 * A * x1.T)) * np.sqrt(np.abs(x2 * A * x2.T))))


def CosDist(x1, x2):
    """
        Note: I the above function is equivalent if A is set to be the I identity matrix.

    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    return 1 - (x1 * x2.T / (np.sqrt(np.abs(x1 * x1.T)) * np.sqrt(np.abs(x2 * x2.T))))


def JObjCosDM(x_idx, x_data_arr, mu, mu_neib_idxs_set,
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
    if x_idx in must_lnk_cons:
        for x_muneib_idx in mu_neib_idxs_set:
            if x_muneib_idx in must_lnk_cons[x_idx]:
                ml_cost += w_constr_viol_mtrx[x_idx, x_muneib_idx] *\
                           CosDistPar(x_data_arr[x_idx, :], x_data_arr[x_muneib_idx, :],
                                      distor_params)

    # Calculating Cannot-Link violation cost.
    cl_cost = 0.0
    if x_idx in cannot_lnk_cons:
        for x_muneib_idx in mu_neib_idxs_set:
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

        xi_neib_sum = np.sum(x_data_arr[list(neibs_idxlst), :], axis=0)
        xi_neib_sum = sp.matrix(xi_neib_sum)

        # Calculating denominator ||Σ xi||(A)
        parametrized_norm_xi = np.sqrt(np.abs(xi_neib_sum * A * xi_neib_sum.T))

        mu_lst.append(xi_neib_sum / parametrized_norm_xi)

    return mu_lst


def MuCos(x_data_arr, neibs_idxs_lsts):
    """
    """
    mu_lst = list()
    for neibs_idxlst in neibs_idxs_lsts:

        xi_neib_sum = np.sum(x_data_arr[neibs_idxlst, :], axis=0)
        xi_neib_sum = sp.matrix(xi_neib_sum)

        # Calculating denominator ||Σ xi||
        parametrized_norm_xi = np.sqrt(np.abs(xi_neib_sum * xi_neib_sum.T))

        mu_lst.append(xi_neib_sum / parametrized_norm_xi)

    return mu_lst


def UpdateDistorParams(distor_params, chang_rate, x_data_arr, mu_lst,
                       neib_idxs_lst, must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx):
    """
    """
    # #### HERE IS THE TRICKY THING #### #
    # I think this is the last hard thing to figure out!
    # i is for x
    # j is for μ of the neib where x is into

    for a_idx, a in enumerate(distor_params):

        # Calculating Partial Derivative of D(xi, mu).
        xm_pderiv = 0.0
        for mu, neib_idxs in zip(mu_lst, neib_idxs_lst):
            for x_neib_idx in neib_idxs:
                xm_pderiv += PartialDerivative(a_idx, x_data_arr[x_neib_idx], mu, distor_params)

        # Calculating Partial Derivative of D(xi, xj) of Must-Link Constraints.
        mlcost_pderiv = 0.0
        for x_idx in range(x_data_arr.shape[0]):
            if x_idx in must_lnk_cons:
                for x_neib_idx in [idx for neib in neib_idxs_lst for idx in neib]:
                    if x_neib_idx in must_lnk_cons[x_idx]:
                        mlcost_pderiv += w_constr_viol_mtrx[x_idx, x_neib_idx] *\
                            PartialDerivative(a_idx, x_data_arr[x_idx],
                                              x_data_arr[x_neib_idx], distor_params)

        # Calculating Partial Derivative of D(xi, xj) of Cannot-Link Constraints.
        clcost_pderiv = 0.0
        for x_idx in range(x_data_arr.shape[0]):
            if x_idx in cannot_lnk_cons:
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

    #print 'A', A

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = np.sqrt(np.abs(x1 * A * x1.T))
    x2_pnorm = np.sqrt(np.abs(x2 * A * x2.T))

    #print 'Derivative:'
    # print (x1[0, a_idx] * x2[0, a_idx] * x1_pnorm * x1_pnorm)
    # print x1 * A * x2.T
    # print (np.square(x1[0, a_idx]) * x2_pnorm + np.square(x2[0, a_idx]) * x1_pnorm)
    # print (2 * x1_pnorm * x2_pnorm)
    # print np.square(x1_pnorm) * np.square(x2_pnorm)
    # print ((x1[0, a_idx] * x2[0, a_idx] * x1_pnorm * x1_pnorm) - x1 * A * x2.T *
    #       ((np.square(x1[0, a_idx]) * x2_pnorm + np.square(x2[0, a_idx]) * x1_pnorm) /
    #       (2 * x1_pnorm * x2_pnorm))) / (np.square(x1_pnorm) * np.square(x2_pnorm))

    return ((x1[0, a_idx] * x2[0, a_idx] * x1_pnorm * x1_pnorm) - x1 * A * x2.T *
            ((np.square(x1[0, a_idx]) * x2_pnorm + np.square(x2[0, a_idx]) * x1_pnorm) /
            (2 * x1_pnorm * x2_pnorm))) / (np.square(x1_pnorm) * np.square(x2_pnorm))


if __name__ == '__main__':

    # x1 = np.array([0.1, 0.7, 0.2, 0.8], dtype=np.float32)
    # x2 = np.array([0.2, 0.5, 0.2, 0.2], dtype=np.float32)
    # dA = np.array([0.9, 0.1, 0.3, 1], dtype=np.float32)
    # print CosDistPar(x1, x2, dA)

    print "Creating Sample"
    x_data_2d_arr1 = sps.vonmises.rvs(1200.489, loc=tuple(np.random.uniform(0.0, 1.0, size=10000)), scale=1, size=(500, 10000))
    x_data_2d_arr2 = sps.vonmises.rvs(1200.489, loc=tuple(np.random.uniform(0.0, 1.0, size=10000)), scale=1, size=(500, 10000))
    x_data_2d_arr3 = sps.vonmises.rvs(1200.489, loc=tuple(np.random.uniform(0.0, 1.0, size=10000)), scale=1, size=(500, 10000))

    # x_data_2d_arr1 = np.random.vonmises(0.5, 100, size=(20, 2))
    # x_data_2d_arr2 = np.random.vonmises(0.5, 1000, size=(20, 2))
    # x_data_2d_arr3 = np.random.vonmises(0.5, 10000, size=(20, 2))

    x_data_2d_arr = np.vstack((x_data_2d_arr1, x_data_2d_arr2, x_data_2d_arr3))

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

    must_lnk_con = {
        1: [5, 3, 6, 8],
        5: [1, 6, 8],
        8: [1, 5],
        3: [1, 7],
        6: [1, 5],
        7: [3],
        521: [525, 528, 539],
        525: [521, 535, 528],
        528: [521, 525],
        535: [525],
        537: [539],
        539: [521, 537]
    }

    cannot_lnk_con = {
        1: [521, 525, 528, 535, 537, 539],
        5: [521, 525, 528, 35],
        8: [521, 525, 528, 535, 537, 539],
        3: [521, 535, 537, 539],
        6: [521, 525, 528, 535, 537, 539],
        7: [521, 525, 528, 535, 537, 539],
        521: [1, 5, 8, 3, 6, 7],
        525: [1, 3, 6, 7],
        528: [1, 5, 8, 3, 6, 7],
        535: [1, 5, 8, 3, 6, 7],
        537: [1, 5, 8, 3],
        539: [1, 5, 8, 3, 6, 7]
    }

    k_expect = 3
    print "Runnint Kmeans"
    res = HMRFKmeans(k_expect, x_data_2d_arr, must_lnk_con, cannot_lnk_con, CosDist,
                     CosDistPar, np.random.uniform(1.0, 1.0, size=10000),
                     np.random.uniform(1.0, 1.0, size=(1500, 1500)),
                     dparmas_chang_rate=0.01)

    for mu_idx, neib_idxs in enumerate(res[1]):
        # print res[0][mu_idx][:, 0], res[0][mu_idx][:, 1]
        # plt.plot(res[0][mu_idx][:, 0], res[0][mu_idx][:, 1], '*', markersize=30)
        #  if mu_idx == 2:
        #    break
        print mu_idx+1, len(neib_idxs), np.sort(neib_idxs)
        for xy in x_data_2d_arr[list(neib_idxs)]:
            plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)
        # plt.plot(x_data_2d_arr2, '^')
        # plt.plot(x_data_2d_arr3, '>')

    plt.show()
