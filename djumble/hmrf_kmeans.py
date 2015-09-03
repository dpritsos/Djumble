# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import matplotlib.pyplot as plt


def HMRFKmeans(k_expect, x_data_arr, must_lnk_cons, cannot_lnk_cons, dmeasure_noparam,
               distor_measure, distor_params, w_constr_viol_mtrx, dparmas_chang_rate):
    """HMRF Kmeans: A Semi-supervised clustering algorithm based on Hidden Markov Random Fields
        Clustering model optimized by Expectation Maximization (EM) algorithm with Hard clustering
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
    last_gobj = 999999999999.0
    for conv_step in range(100):

        print conv_step

        # The E-Step ######

        # Assigning every data-set point to the proper cluster/neighbourhood upon distortion...
        # ...parameters and centroids of the current iteration.
        mu_neib_idxs_set_lst = ICM(x_data_arr, mu_lst, mu_neib_idxs_set_lst,
                                   must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx,
                                   distor_params)

        # The M-Step #######

        # Recalculating centroids upon the new clusters set-up.
        mu_lst = MuCosDMPar(x_data_arr, mu_neib_idxs_set_lst, distor_params)
        # print mu_lst

        # Re-estimating distortion measure parameters upon the new clusters set-up.
        distor_params = UpdateDistorParams(distor_params, dparmas_chang_rate, x_data_arr, mu_lst,
                                           mu_neib_idxs_set_lst, must_lnk_cons,
                                           cannot_lnk_cons, w_constr_viol_mtrx,)

        # Calculating Global JObjective function.
        glob_jobj = GlobJObjCosDM(x_data_arr, mu_lst, mu_neib_idxs_set_lst,
                                  must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx,
                                  distor_params)

        # Terminating upon Global JObej on condition.
        # It suppose that global JObjective monotonically decreases, am I right?
        # if last_gobj - glob_jobj < 0.000:
        #    raise Exception("Global JObjective difference returned a negative value.")

        if glob_jobj < 0.001:
            print "Global Objective", glob_jobj
            break
        # else:
        #     # last_gobj = glob_jobj

        print "Global Objective", glob_jobj

    # Returning the Centroids, Clusters/Neighbourhoods, distortion parameters,
    # constraint violations matrix.
    return mu_lst, mu_neib_idxs_set_lst, distor_params, w_constr_viol_mtrx


def ICM(x_data_arr, mu_lst, mu_neib_idxs_set_lst, must_lnk_cons, cannot_lnk_cons,
        w_constr_viol_mtrx, distor_params):
    """ ICM: Iterated Conditional Modes (for the E-Step)

        After all points are assigned, they are randomly re-ordered, and
        the assignment process is repeated. This process proceeds until no
        point changes its cluster assignment between two successive iterations.

    """
    no_change_cnt = 0
    while no_change_cnt < 2:

        # Calculating the new Neighbourhoods/Clusters.
        for x_idx in np.random.randint(0, x_data_arr.shape[0], size=x_data_arr.shape[0]):

            last_jobj = 999999999999999999999999999999999999999.0

            for i, (mu, mu_neib_idxs_set) in enumerate(zip(mu_lst, mu_neib_idxs_set_lst)):

                j_obj = JObjCosDM(x_idx, x_data_arr, mu, mu_neib_idxs_set,
                                  must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx,
                                  distor_params)
                # print j_obj
                if j_obj < last_jobj:
                    last_jobj = j_obj
                    mu_neib_idx = i
                else:
                    pass
                    # print "else J_Obj", j_obj

            if x_idx not in mu_neib_idxs_set_lst[mu_neib_idx]:

                # Remove x form all Clusters.
                for mu_neib_idxs_set in mu_neib_idxs_set_lst:
                    mu_neib_idxs_set.discard(x_idx)
                    # mu_neib_idxs_set_lst[midx].discard(x_idx)

                mu_neib_idxs_set_lst[mu_neib_idx].add(x_idx)

                no_change = False

            else:
                no_change = True

        if no_change:
            no_change_cnt += 1

    # Returning mu_neib_idxs_set_lst.
    return mu_neib_idxs_set_lst


def CosDistPar(x1, x2, distor_params):
    """CosDistPar: Cosine Distance with distortion parameters based on 'Soft Cosine Measure' where
        a weighting schema is the distortion parameters diagonal matrix A. Note that A matrix
        (diagonal) is expected as vector argument in this function.
    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)
    A = sp.sparse.dia_matrix((distor_params, [0]),
                             shape=(distor_params.shape[0], distor_params.shape[0]))
    # A = sp.diag(distor_params)

    return 1 - (x1 * A * x2.T / (np.sqrt(np.abs(x1 * A * x1.T)) * np.sqrt(np.abs(x2 * A * x2.T))))


def CosDist(x1, x2):
    """
        Note: I the above function is equivalent if A is set to be the I identity matrix.

    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    return 1 - (x1 * x2.T / (np.sqrt(np.abs(x1 * x1.T)) * np.sqrt(np.abs(x2 * x2.T))))


def MuCosDMPar(x_data_arr, neibs_idxs_lsts, distor_params):
    """
    """
    A = sp.sparse.dia_matrix((distor_params, [0]),
                             shape=(distor_params.shape[0], distor_params.shape[0]))
    # A = sp.diag(distor_params)

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


def JObjCosDM(x_idx, x_data_arr, mu, mu_neib_idxs_set,
              must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params):
    """JObjCosDM: J Objective function for Cosine Distortion Measure. It cannot very generic
        because the gradient decent (partial derivative) calculation should be applied which they
        are totally dependent on the distortion measure, here Cosine Distance.

    """

    "Phi_max depends on the distortion measure"

    d = CosDistPar(x_data_arr[x_idx, :], mu, distor_params)

    # Calculating Must-Link violation cost.
    ml_cost = 0.0
    for x_cons in must_lnk_cons:

        if x_idx in x_cons:

            if not (x_cons <= mu_neib_idxs_set):

                x = list(x_cons)

                ml_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                    CosDistPar(x_data_arr[x[0], :], x_data_arr[x[1], :], distor_params)

    # Calculating Cannot-Link violation cost.
    cl_cost = 0.0
    for x_cons in cannot_lnk_cons:

        if x_idx in x_cons:

            if x_cons <= mu_neib_idxs_set:

                x = list(x_cons)

                cl_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                    (1 - CosDistPar(x_data_arr[x[0], :], x_data_arr[x[1], :], distor_params))

    # ### Calculating the Rayleigh's PDF contribution.
    sum1 = 0.0
    sum2 = 0.0

    for a in distor_params:
        sum1 += a / 2 * np.square(0.5)
        sum2 += np.log(a)

    params_pdf = sum2 - sum1 - distor_params.shape[0] * np.log(np.square(0.5))
    # print params_pdf
    # if ml_cost > 0.0 or cl_cost > 0.0:
    #    print d, ml_cost, cl_cost, params_pdf
    return d + ml_cost + cl_cost + params_pdf


def GlobJObjCosDM(x_data_arr, mu_lst, mu_neib_idxs_set_lst,
                  must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params):
    """
    """

    sum_d = 0.0
    for mu, neib_idxs in zip(mu_lst, mu_neib_idxs_set_lst):
        for x_neib_idx in neib_idxs:
            sum_d += CosDistPar(x_data_arr[x_neib_idx], mu, distor_params)

    # Calculating Must-Link violation cost.
    ml_cost = 0.0
    for mu_neib_idxs_set in mu_neib_idxs_set_lst:

        for x_cons in must_lnk_cons:

            if not (x_cons <= mu_neib_idxs_set):

                x = list(x_cons)

                ml_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                    CosDistPar(x_data_arr[x[0], :], x_data_arr[x[1], :], distor_params)

    # Calculating Cannot-Link violation cost.
    cl_cost = 0.0
    for mu_neib_idxs_set in mu_neib_idxs_set_lst:

        for x_cons in cannot_lnk_cons:

            if x_cons <= mu_neib_idxs_set:

                x = list(x_cons)

                cl_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                    (1 - CosDistPar(x_data_arr[x[0], :], x_data_arr[x[1], :], distor_params))

    # ### Calculating the Rayleigh's PDF contribution.
    sum1 = 0.0
    sum2 = 0.0

    for a in distor_params:
        sum1 += a / 2 * np.square(0.5)
        sum2 += np.log(a)

    params_pdf = sum2 - sum1 - distor_params.shape[0] * np.log(np.square(0.5))

    # print sum_d, ml_cost, cl_cost, params_pdf

    # print "In Global Params PDF", params_pdf

    return sum_d + ml_cost + cl_cost + params_pdf


def UpdateDistorParams(dparams, chang_rate, x_data_arr, mu_lst,
                       neib_idxs_lst, must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx):
    """
    """
    # #### HERE IS THE TRICKY THING #### #
    # I think this is the last hard thing to figure out!
    # i is for x
    # j is for μ of the neib where x is into

    # print "OLD Params", dparams

    for a_idx, a in enumerate(dparams):

        # Calculating Partial Derivative of D(xi, mu).
        xm_pderiv = 0.0
        for mu, neib_idxs in zip(mu_lst, neib_idxs_lst):
            for x_neib_idx in neib_idxs:
                xm_pderiv += PartialDerivative(a_idx, x_data_arr[x_neib_idx], mu, dparams)
        # print "Partial Distance", xm_pderiv

        # [idx for neib in neib_idxs_lst for idx in neib]
        # Calculating the Partial Derivative of D(xi, xj) of Must-Link Constraints.
        mlcost_pderiv = 0.0
        for mu_neib_idxs_set in neib_idxs_lst:

            for x_cons in must_lnk_cons:

                if not (x_cons <= mu_neib_idxs_set):

                    x = list(x_cons)

                    mlcost_pderiv += w_constr_viol_mtrx[x[0], x[1]] *\
                        PartialDerivative(a_idx, x_data_arr[x[0], :], x_data_arr[x[1], :], dparams)
        # print "Partial MustLink", mlcost_pderiv

        # Calculating the Partial Derivative of D(xi, xj) of Cannot-Link Constraints.
        clcost_pderiv = 0.0
        for mu_neib_idxs_set in neib_idxs_lst:

            for x_cons in cannot_lnk_cons:

                if x_cons <= mu_neib_idxs_set:

                    x = list(x_cons)

                    clcost_pderiv += w_constr_viol_mtrx[x[0], x[1]] *\
                        PartialDerivative(a_idx, x_data_arr[x[0], :], x_data_arr[x[1], :], dparams)
        # print "Partial MustLink", clcost_pderiv

        # ### Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
        a_pderiv = (1 / a) - (a / 2 * np.square(0.5))

        # Changing the a dimension of A = np.diag(distortions-measure-parameters)
        dparams[a_idx] = a + chang_rate * (xm_pderiv + mlcost_pderiv + clcost_pderiv - a_pderiv)

    # print "Params", dparams

    return dparams


def PartialDerivative(a_idx, x1, x2, distor_params):
    """
    """
    A = sp.sparse.dia_matrix((distor_params, [0]),
                             shape=(distor_params.shape[0], distor_params.shape[0]))
    # A = sp.diag(distor_params)
    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = np.sqrt(np.abs(x1 * A * x1.T))
    x2_pnorm = np.sqrt(np.abs(x2 * A * x2.T))

    res_a = (
                (x1[0, a_idx] * x2[0, a_idx] * x1_pnorm * x1_pnorm) -
                (
                    x1 * A * x2.T *
                    (
                        (
                            np.square(x1[0, a_idx]) * np.square(x2_pnorm) +
                            np.square(x2[0, a_idx]) * np.square(x1_pnorm)
                        ) / (2 * x1_pnorm * x2_pnorm)
                    )
                )
            ) / (np.square(x1_pnorm) * np.square(x2_pnorm))

    # if res_a < 0:
    # print res_a

    return res_a

def FarFirstCosntraint(x_data_arr, k_expect, must_lnk_cons, cannnot_lnk_cons, distor_measure):

    # ########### NOT PROPERLY IMPLEMENTED FOR THIS GIT COMMIT ###
    """
        pick any z ∈ S and set T = {z}
        while |T| < k:
            z = arg maxx∈S ρ(x, T)
            T = T ∪ {z}

        Here ρ(x, T) is the distance from point x to the closest point in set T,
        that is to say, infz∈T ρ(x, z).

    """

    # Initiating the list of array indices for all forthcoming neighborhoods Np.
    neibs_sets = [set([])]

    data_num = x_data_arr.shape[0]

    # Adding a random point in the neighborhood N0.
    rnd_idx = np.random.randint(0, data_num)

    neibs_sets[0].add(rnd_idx)
    neib_c = 1

    farthest_x_idx = data_num + 99  # Not sure for this initialization.

    # Initializing for finding the farthest x array index form all N neighborhoods.

    all_neibs = []
    while neib_c < k_expect and len(all_neibs) < data_num:

        max_dist = 0
        # Getting the farthest x from all neighborhoods.
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
                for n_idx, neib in enumerate(neibs_sets):
                    if ml_idx in neib:
                        must_link_neib_indx = n_idx

        # Looking for Cannot-Link
        cannot_link = False
        if farthest_x_idx in cannnot_lnk_cons:
            for cl_idx in cannnot_lnk_cons[farthest_x_idx]:
                for neib in neibs_sets:
                    if cl_idx in neib:
                        cannot_link = True

        # Putting the x in the proper N neighborhood.
        if must_link_neib_indx:

            neibs_sets[must_link_neib_indx].add(farthest_x_idx)

        elif cannot_link:

            neib_c += 1
            neibs_sets.append(set([farthest_x_idx]))

        else:
            neibs_sets[neib_c-1].add(farthest_x_idx)

    return neibs_sets


def ConsolidateAL(neibs_sets, x_data_arr, must_lnk_cons, distor_measure):

    # ########### NOT PROPERLY IMPLEMENTED FOR THIS GIT COMMIT ###

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


def FarFirstWeighted(x_data_arr, k_expect, must_lnk_con, cannnot_lnk_con, CosDist):
    """
    """
    pass


if __name__ == '__main__':

    print "Creating Sample"
    x_data_2d_arr1 = sps.vonmises.rvs(1200.0, loc=(0.7, 0.2), scale=1, size=(500, 2))
    x_data_2d_arr2 = sps.vonmises.rvs(1200.0, loc=(0.6, 0.6), scale=1, size=(500, 2))
    x_data_2d_arr3 = sps.vonmises.rvs(1200.0, loc=(0.2, 0.3), scale=1, size=(500, 2))

    # tuple(np.random.normal(0.0, 10.0, size=2))
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

    must_lnk_con = [
        set([1, 5]),
        set([1, 3]),
        set([1, 6]),
        set([1, 8]),
        set([7, 3]),
        set([521, 525]),
        set([521, 528]),
        set([521, 539]),
        set([535, 525]),
        set([537, 539]),
        set([1037, 1238]),
        set([1057, 1358]),
        set([1039, 1438]),
        set([1045, 1138]),
        set([1098, 1038]),
        set([1019, 1138]),
        set([1087, 1338])
    ]

    cannot_lnk_con = [
        set([1, 521]),
        set([1, 525]),
        set([1, 528]),
        set([1, 535]),
        set([1, 537]),
        set([1, 539]),
        set([5, 521]),
        set([5, 525]),
        set([5, 528]),
        set([5, 35]),
        set([8, 521]),
        set([8, 525]),
        set([8, 528]),
        set([8, 535]),
        set([8, 537]),
        set([8, 539]),
        set([3, 521]),
        set([3, 535]),
        set([3, 537]),
        set([3, 539]),
        set([6, 521]),
        set([6, 525]),
        set([6, 528]),
        set([6, 535]),
        set([6, 537]),
        set([6, 539]),
        set([7, 521]),
        set([7, 525]),
        set([7, 528]),
        set([7, 535]),
        set([7, 537]),
        set([7, 539]),
        set([538, 1237]),
        set([548, 1357]),
        set([558, 1437]),
        set([738, 1137]),
        set([938, 1037]),
        set([838, 1039]),
        set([555, 1337])
    ]

    k_expect = 3
    print "Running HMRF Kmeans"
    res = HMRFKmeans(k_expect, x_data_2d_arr, must_lnk_con, cannot_lnk_con, CosDist,
                     CosDistPar, np.random.uniform(0.0, 20.0, size=2),
                     np.random.uniform(0.9, 0.9, size=(1500, 1500)),
                     dparmas_chang_rate=0.025)

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
