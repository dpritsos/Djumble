# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import matplotlib.pyplot as plt
import scipy.special as special


class HMRFKmeans(object):
    """ HMRF Kmeans: A Semi-supervised clustering algorithm based on Hidden Markov Random Fields
        Clustering model optimized by Expectation Maximization (EM) algorithm with Hard clustering
        constraints, i.e. a Kmeans Semi-supervised clustering variant.

        Initialization arguments
        ------------------------
            k_clusters: The number of clusters we expect in the data-set/corpus.
            must_lnk: A list of paired Sets of the must-link constraints known a priori.
            cannot_lnk: A list of paired Sets of the cannot-link constraints known a priori.
            init_centroids: An optional set..... of initial clusters centroids.
            max_iter: The maximum number of iterations in case the convergence criterion has not
                been reached.
            cvg: Convergence value. The maximum difference of two sequential J-Objective values
                that must be reached until the algorithms convergence.
            lrn_rate: Learning rate value or the rate with which the distortion parameters must
                be changing in every iteration.
            ray_simga: Rayleigh's distribution sigma-window parameter. This distribution is
                applied over the distortion parameters values for enforcing the J-Objective
                function to decrease.
            w_violations: A weights matrix for must-link and cannot-link violations in the
                clustering process.
            d_params: Distortion parameters vector. It is actually a compressed form of the
                distortion parameters matrix of N x N size. Where N is the size of the vector
                (or feature) space, exactly as it is recommended in bibliography for making the
                calculation easier and faster.

        More details later...

    """

    def __init__(self, k_clusters, must_lnk, cannot_lnk, init_centroids=None, max_iter=300,
                 cvg=0.0001, lrn_rate=0.03, ray_sigma=0.5, w_violations=None, d_params=None):

        self.k_clusters = k_clusters
        self.must_lnk = must_lnk
        self.cannot_lnk = cannot_lnk
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.cvg = cvg
        self.lrn_rate = lrn_rate
        self.ray_sigma = ray_sigma
        self.w_violations = w_violations
        self.d_params = d_params

    def Fit(self, x_data):
        """ Fit method: The HMRF-Kmeans algorithm is running in this method in order to fit the
            data in the Mixture of the von Misses Fisher (vMF) distributions. However, the vMF(s)
            are considered to have the same shape at the end of the process. That is, Kmeans and
            not EM clustering. The similarity measure (a.k.a distortion paramters) is a
            parametrized cosine similarity.

            Arguments
            ---------
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.

        """

        # Initializing clustering

        # Setting up distortion parameters if not have been passed as class argument.
        if not self.d_params:
            self.d_params = np.random.uniform(0.50, 100.0, size=x_data.shape[1])

        # A is the same as d_params but in a diagonal matrix form required for the...
        # ...calculations in the functions bellow. It will save array to matrix transformations.
        self.A = sp.sparse.dia_matrix((distor_params, [0]),
                                      shape=(distor_params.shape[0], distor_params.shape[0]))

        # Setting up the violation weights matrix if not have been passed as class argument.
        if not self.w_violations:
            self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        # ### I am not sure what kind of values this weights should actually have.

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint(x_data, k_clusters, must_lnk_cons,
        #                                           cannot_lnk_cons, dmeasure_noparam)
        # init_clstr_sets_lst = ConsolidateAL(neibs_sets, x_data,
        #                                      must_lnk_cons, dmeasure_noparam)
        init_clstr_sets_lst = list()

        # If initial centroids arguments has been passed.
        init_clstr_sets_lst.extend(self.init_centroids)
        # ### Maybe this should be changed to a numpy vector of indices.

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_lst = MeanCosA(x_data, init_clstr_sets_lst)

        # EM algorithm execution.
        # While no convergence yet or i times.
        for conv_step in range(self.max_iter):

            print conv_step

            # The E-Step.

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            clstr_idxs_set_lst = ICM(x_data, mu_lst, clstr_idxs_sets_lst)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_lst = MeanCosA(x_data, clstr_idxs_set_lst)
            # print mu_lst

            # Re-estimating distortion measure parameters upon the new clusters set-up.
            self.d_params = UpdateDistorParams(self.d_params, x_data, mu_lst, clstr_idxs_set_lst)

            # Calculating Global JObjective function.
            glob_jobj = GlobJObjCosA(x_data, mu_lst, clstr_idxs_set_lst)

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
        return mu_lst, clstr_idxs_set_lst, distor_params, w_constr_viol_mtrx

    def ICM(self, x_data, mu_lst, clstr_idxs_sets_lst):
        """ ICM: Iterated Conditional Modes (for the E-Step).
            After all points are assigned, they are randomly re-ordered, and the assignment process
            is repeated. This process proceeds until no point changes its cluster assignment
            between two successive iterations.

            Arguments
            ---------
            x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
            mu_lst: The list of centroids of the clusters.
            clstr_idxs_sets_lst: The list of sets of the

        """
        no_change_cnt = 0
        while no_change_cnt < 2:

            # Calculating the new Clusters.
            for x_idx in np.random.randint(0, x_data.shape[0], size=x_data.shape[0]):

                last_jobj = 999999999999999999999999999999999999999.0

                for i, (mu, clstr_idxs_set) in enumerate(zip(mu_lst, clstr_idxs_sets_lst)):

                    j_obj = JObjCosA(x_idx, x_data, mu, clstr_idxs_set)
                    # print j_obj
                    if j_obj < last_jobj:
                        last_jobj = j_obj
                        mu_neib_idx = i
                    else:
                        pass
                        # print "else J_Obj", j_obj

                if x_idx not in clstr_idxs_sets_lst[mu_neib_idx]:

                    # Remove x form all Clusters.
                    for clstr_idxs_set in clstr_idxs_sets_lst:
                        clstr_idxs_set.discard(x_idx)
                        # clstr_idxs_sets_lst[midx].discard(x_idx)

                    clstr_idxs_sets_lst[mu_neib_idx].add(x_idx)

                    no_change = False

                else:
                    no_change = True

            if no_change:
                no_change_cnt += 1

        # Returning clstr_idxs_sets_lst.
        return clstr_idxs_sets_lst

    def CosDistA(self, x1, x2):
        """ CosDistA: Cosine Distance with distortion parameters based on 'Soft Cosine Measure'
            where a weighting schema is the distortion parameters diagonal matrix A. Note that
            A matrix (diagonal) is expected as vector argument in this function.

            Arguments
            ---------
                x1, x2: The numpy.array vectors, their (parameterized) cosine distance will
                        be measured.

        """

        # Converting vectors x1 and x2 to 1D matrices.
        x1 = sp.matrix(x1)
        x2 = sp.matrix(x2)

        # Calculating and returning the parameterized cosine distance.
        return 1 - (x1 * A * x2.T /
                    (np.sqrt(np.abs(x1 * self.A * x1.T)) * np.sqrt(np.abs(x2 * self.A * x2.T)))
                    )

    def MeanCosA(self, x_data, clstr_idxs_lsts):
        """ MeanCosA method: It is calculating the centroids of the hyper-spherical clusters.
            Using the parametrized cosine mean as explained in the documentation.

            Arguments
            ---------
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
                clstr_idxs_lsts: The lists of indices for each cluster.

        """

        mu_lst = list()
        for clstr_ilst in clstr_idxs_lsts:

            # Summing up all the X data points for the current cluster.
            xi_sum = np.sum(x_data[list(clstr_ilst), :], axis=0)
            xi_sum = sp.matrix(xi_neib_sum)

            # Calculating denominator ||Σ xi||(A)
            parametrized_norm_xi = np.sqrt(np.abs(xi_sum * self.A * xi_sum.T))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            mu_lst.append(xi_sum / parametrized_norm_xi)

        return mu_lst

    def JObjCosA(self, x_idx, x_data, mu, clstr_idxs_set,
                 must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params, s):
        """ JObjCosA: J Objective function for Cosine Distortion Measure. It cannot very generic
            because the gradient decent (partial derivative) calculation should be applied which they
            are totally dependent on the distortion measure, here Cosine Distance.

        """

        "Phi_max depends on the distortion measure"

        d = CosDistA(x_data[x_idx, :], mu, distor_params)

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        for x_cons in must_lnk_cons:

            if x_idx in x_cons:

                if not (x_cons <= clstr_idxs_set):

                    x = list(x_cons)

                    ml_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                        CosDistA(x_data[x[0], :], x_data[x[1], :], distor_params)

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        for x_cons in cannot_lnk_cons:

            if x_idx in x_cons:

                if x_cons <= clstr_idxs_set:

                    x = list(x_cons)

                    cl_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                        (1 - CosDistA(x_data[x[0], :], x_data[x[1], :], distor_params))

        # ### Calculating the Rayleigh's PDF contribution.
        sum1 = 0.0
        sum2 = 0.0

        for a in distor_params:
            sum1 += np.log(a)
            sum2 += a / 2 * np.square(s)

        params_pdf = sum1 - sum2 - 2 * distor_params.shape[0] * np.log(s)
        # print params_pdf
        # if ml_cost > 0.0 or cl_cost > 0.0:
        #    print d, ml_cost, cl_cost, params_pdf

        # Vector space dimensions
        d = x_data.shape[1]

        # Concentration approximation
        r = np.linalg.norm(x_data)
        k = (r*d - np.power(r, 3)) / (1 - np.power(r, 2))

        # Calculating Bessel Function for the first kind for order equals to vector space dimensions.
        bessel = special.jv((d/2.0)-1.0, k)

        return d + ml_cost + cl_cost + params_pdf + np.log(bessel)*x_data.shape[0]

    def GlobJObjCosA(self, x_data, mu_lst, clstr_idxs_set_lst,
                     must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, distor_params, s):
        """
        """

        sum_d = 0.0
        for mu, neib_idxs in zip(mu_lst, clstr_idxs_set_lst):
            for x_neib_idx in neib_idxs:
                sum_d += CosDistA(x_data[x_neib_idx], mu, distor_params)

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in must_lnk_cons:

                if not (x_cons <= clstr_idxs_set):

                    x = list(x_cons)

                    ml_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                        CosDistA(x_data[x[0], :], x_data[x[1], :], distor_params)

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in cannot_lnk_cons:

                if x_cons <= clstr_idxs_set:

                    x = list(x_cons)

                    cl_cost += w_constr_viol_mtrx[x[0], x[1]] *\
                        (1 - CosDistA(x_data[x[0], :], x_data[x[1], :], distor_params))

        # ### Calculating the Rayleigh's PDF contribution.
        sum1 = 0.0
        sum2 = 0.0

        for a in distor_params:
            sum1 += np.log(a)
            sum2 += a / 2 * np.square(s)

        params_pdf = sum1 - sum2 - 2 * distor_params.shape[0] * np.log(s)

        # print sum_d, ml_cost, cl_cost, params_pdf

        # print "In Global Params PDF", params_pdf

        # Vector space dimensions
        d = x_data.shape[1]
        print d

        # Concentration approximation
        r = np.linalg.norm(x_data)
        k = (r*d - np.power(r, 3)) / (1 - np.power(r, 2))
        print r
        print k

        # Calculating Bessel Function for the first kind for order equals to vector space dimensions.
        bessel = special.jv((d/2.0)-1.0, k)
        print bessel

        print 'np.log(bessel)*N', np.log(bessel)*x_data.shape[0]

        print x_data.shape[0]

        return sum_d + ml_cost + cl_cost + params_pdf - np.log(bessel)*x_data.shape[0]

    def UpdateDistorParams(self, dparams, chang_rate, x_data, mu_lst,
                           neib_idxs_lst, must_lnk_cons, cannot_lnk_cons, w_constr_viol_mtrx, s):
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
                    xm_pderiv += PartialDerivative(a_idx, x_data[x_neib_idx], mu, dparams)
            # print "Partial Distance", xm_pderiv

            # [idx for neib in neib_idxs_lst for idx in neib]
            # Calculating the Partial Derivative of D(xi, xj) of Must-Link Constraints.
            mlcost_pderiv = 0.0
            for clstr_idxs_set in neib_idxs_lst:

                for x_cons in must_lnk_cons:

                    if not (x_cons <= clstr_idxs_set):

                        x = list(x_cons)

                        mlcost_pderiv += w_constr_viol_mtrx[x[0], x[1]] *\
                            PartialDerivative(a_idx, x_data[x[0], :], x_data[x[1], :], dparams)
            # print "Partial MustLink", mlcost_pderiv

            # Calculating the Partial Derivative of D(xi, xj) of Cannot-Link Constraints.
            clcost_pderiv = 0.0
            for clstr_idxs_set in neib_idxs_lst:

                for x_cons in cannot_lnk_cons:

                    if x_cons <= clstr_idxs_set:

                        x = list(x_cons)

                        clcost_pderiv += w_constr_viol_mtrx[x[0], x[1]] *\
                            PartialDerivative(a_idx, x_data[x[0], :], x_data[x[1], :], dparams)
            # print "Partial MustLink", clcost_pderiv

            # ### Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
            a_pderiv = (1 / a) - (a / 2 * np.square(s))

            # Changing the a dimension of A = np.diag(distortions-measure-parameters)
            dparams[a_idx] = a + chang_rate * (xm_pderiv + mlcost_pderiv + clcost_pderiv - a_pderiv)

        # print "Params", dparams

        return dparams

    def PartialDerivative(self, a_idx, x1, x2, distor_params):
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

    def FarFirstCosntraint(self, x_data, k_clusters, must_lnk_cons, cannnot_lnk_cons, distor_measure):

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

        data_num = x_data.shape[0]

        # Adding a random point in the neighborhood N0.
        rnd_idx = np.random.randint(0, data_num)

        neibs_sets[0].add(rnd_idx)
        neib_c = 1

        farthest_x_idx = data_num + 99  # Not sure for this initialization.

        # Initializing for finding the farthest x array index form all N neighborhoods.

        all_neibs = []
        while neib_c < k_clusters and len(all_neibs) < data_num:

            max_dist = 0
            # Getting the farthest x from all neighborhoods.
            for i in np.random.randint(0, x_data.shape[0], size=x_data.shape[0]/10):

                all_neibs = [idx for neib in neibs_sets for idx in neib]

                for neib_x_idx in all_neibs:

                        if i not in all_neibs:

                            dist = distor_measure(x_data[neib_x_idx], x_data[i])

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

    def ConsolidateAL(self, neibs_sets, x_data, must_lnk_cons, distor_measure):

        # ########### NOT PROPERLY IMPLEMENTED FOR THIS GIT COMMIT ###

        """
        """
        # Estimating centroids.
        # print np.mean(x_data[[1,2,3], :], axis=0)
        neibs_mu = [np.mean(x_data[neib, :], axis=0) for neib in neibs_sets]

        cnt = 0

        # I think that randomization factor is required  replacing --> # range(data_num):
        for rnd_idx in np.random.randint(0, x_data.shape[0], size=x_data.shape[0]):

            cnt += 1

            # Ascending order.
            srted_dists_neib_idx = np.argsort(
                [distor_measure(mu, x_data[rnd_idx, :])[0, 0] for mu in neibs_mu],
                axis=0
            )

            for neib_idx in srted_dists_neib_idx:
                if rnd_idx in must_lnk_cons:
                    for ml_idx in must_lnk_cons[rnd_idx]:
                        if ml_idx in neibs_sets[neib_idx] and rnd_idx not in neibs_sets[neib_idx]:
                            neibs_sets[neib_idx].append(rnd_idx)

        return neibs_sets


# The following function most probably won't be needed
def FarFirstWeighted(x_data, k_clusters, must_lnk_con, cannnot_lnk_con, CosDist):
    pass


def MuCos(x_data, neibs_idxs_lsts):
    mu_lst = list()
    for neibs_idxlst in neibs_idxs_lsts:

        xi_neib_sum = np.sum(x_data[neibs_idxlst, :], axis=0)
        xi_neib_sum = sp.matrix(xi_neib_sum)

        # Calculating denominator ||Σ xi||
        parametrized_norm_xi = np.sqrt(np.abs(xi_neib_sum * xi_neib_sum.T))

        mu_lst.append(xi_neib_sum / parametrized_norm_xi)

    return mu_lst


def CosDist(x1, x2):
    """
        Note: I the above function is equivalent if A is set to be the I identity matrix.

    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)

    return x1 * x2.T / (np.sqrt(np.abs(x1 * x1.T)) * np.sqrt(np.abs(x2 * x2.T)))


if __name__ == '__main__':

    test_dims = 1000

    print "Creating Sample"
    x_data_2d_arr1 = sps.vonmises.rvs(1200.0, loc=np.random.uniform(0.0, 0.6, size=(1, test_dims)), scale=1, size=(500, test_dims))
    x_data_2d_arr2 = sps.vonmises.rvs(1200.0, loc=np.random.uniform(0.3, 0.7, size=(1, test_dims)), scale=1, size=(500, test_dims))
    x_data_2d_arr3 = sps.vonmises.rvs(1200.0, loc=np.random.uniform(0.6, 0.9, size=(1, test_dims)), scale=1, size=(500, test_dims))


# (0.7, 0.2, 0.7, 0.2, 0.6, 0.6, 0.1, 0.3, 0.8, 0.5)
# (0.6, 0.6, 0.7, 0.2, 0.6, 0.6, 0.8, 0.3, 0.9, 0.1)
# (0.2, 0.3, 0.7, 0.2, 0.6, 0.6, 0.2, 0.3, 0.6, 0.4)

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

    k_clusters = 3
    print "Running HMRF Kmeans"
    res = HMRFKmeans(k_clusters, x_data_2d_arr, must_lnk_con, cannot_lnk_con, CosDist,
                     CosDistA, np.random.uniform(0.50, 100.0, size=test_dims),
                     np.random.uniform(0.9, 0.9, size=(1500, 1500)),
                     dparmas_chang_rate=0.3, s=0.5)

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
