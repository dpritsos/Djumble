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

sys.path.append('../../synergeticprocessing')
# import synergeticprocessing.synergeticpool as mymp


class CosineKmeans(object):
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
                 cvg=0.001):

        self.k_clusters = k_clusters
        self.must_lnk = must_lnk
        self.cannot_lnk = cannot_lnk
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.cvg = cvg

    def fit(self, x_data, neg_idxs4clstring=np.array([], dtype=np.int)):
        """ Fit method: The HMRF-Kmeans algorithm is running in this method in order to fit the
            data in the Mixture of the von Misses Fisher (vMF) distributions. However, the vMF(s)
            are considered to have the same shape at the end of the process. That is, Kmeans and
            not EM clustering. The similarity measure (a.k.a distortion paramters) is a
            parametrized cosine similarity.

            Arguments
            ---------
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.

            Output
            ------
                mu_lst: The list of N cluster centroids.
                clstr_idxs_set_lst: The list of sets of x_data array indices for each of the N
                    clusters.
                self.A.data: The values of the (hyper-)parametes for the cosine distance after the
                    final model fit.

        """

        # Initializing clustering

        # Setting up the violation weights matrix if not have been passed as class argument.
        # if self.w_violations is None:
        #     self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        #     # ### I am not sure what kind of values this weights should actually have.

        # Converting the weights violation matrix into a sparse matrix.
        # non_zero = np.where(self.ml_cl_cons != 0, 1, 0)
        # self.w_violations = sp.sparse.csr_matrix(np.multiply(self.w_violations, non_zero))
        # self.w_violations = np.multiply(self.w_violations, non_zero)

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint()
        # init_clstr_sets_lst = ConsolidateAL()
        clstr_tags_arr = np.empty(x_data.shape[0], dtype=np.int)
        clstr_tags_arr[:] = np.Inf

        # Selecting a random set of centroids if not any given as class initialization argument.
        if self.init_centroids is None:
            # Pick k random vector from the x_data set as initial centroids. Where k is equals...
            # ...the number of self.k_clusters.
            self.init_centroids = np.random.randint(0, self.k_clusters, size=x_data.shape[0])

        # Setting one index of the x_data set per expected cluster as the initialization centroid...
        # ...for this cluster.
        for i, idx in enumerate(self.init_centroids):
            clstr_tags_arr[idx] = i

        # ########
        # # The above might actually change based on the initialization set-up will be used.
        # ########

        # Set of indices not participating in clustering. NOTE: For particular experiments only.
        self.neg_idxs4clstring = neg_idxs4clstring

        if self.neg_idxs4clstring.shape[0]:
            for idx in self.neg_idxs4clstring:
                clstr_tags_arr[idx] = -9


        # NOTE: Initially normalizing the samples under the distortion parameters values in...
        # ...order to reducing the cosine distance calculation to a simple dot products...
        # ...calculation in the rest of the EM (sub)steps.
        x_data = np.divide(
            x_data,
            np.sqrt(
                np.diag(np.dot(x_data, np.transpose(x_data))),
                dtype=np.float
            ).reshape(x_data.shape[0], 1)
        )

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_arr = self.MeanCosA(x_data, clstr_tags_arr)

        # EM algorithm execution.

        # This values is the last global objective. Thus it should be the highest possible...
        # ...number initially, i.e. Inf.
        last_gobj = np.Inf

        # While no convergence yet repeat for at least i times.
        for conv_step in range(self.max_iter):

            print
            print conv_step

            # The E-Step.

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            clstr_tgs_arr = self.ICM(x_data, mu_arr, clstr_tags_arr)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_arr = self.MeanCosA(x_data, clstr_tgs_arr)
            # print mu_lst

            # NOTE: Normalizing the samples under the new parameters values in order to reducing...
            # ...the cosine distance calculation to a simple dot products calculation in the...
            # ...rest of the EM (sub)steps.
            x_data = np.divide(
                x_data,
                np.sqrt(
                    np.diag(np.dot(x_data, np.transpose(x_data))),
                    dtype=np.float
                ).reshape(x_data.shape[0], 1)
            )

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_arr, clstr_tags_arr)

            # Terminating upon the difference of the last two Global JObej values.
            if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:
                print 'last_gobj - glob_jobj', last_gobj - glob_jobj
                print "Global Objective (Narray)", glob_jobj
                break
            else:
                last_gobj = glob_jobj

            print "Global Objective (Narray)", glob_jobj

        # Storing the amount of iterations until convergence.
        self.conv_step = conv_step

        # Returning the Centroids and the Clusters,i.e. the set of indeces for each cluster.
        return mu_arr, clstr_tags_arr

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'final_iter': self.conv_step,
            'convg_diff': self.cvg
        }

    def ICM(self, x_data, mu_arr, clstr_tags_arr):
        """ ICM: Iterated Conditional Modes (for the E-Step).
            After all points are assigned, they are randomly re-ordered, and the assignment process
            is repeated. This process proceeds until no point changes its cluster assignment
            between two successive iterations.

            Arguments
            ---------
            x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
            mu_lst: The list of centroids of the clusters.
            clstr_idxs_sets_lst: The list of sets of the

            Output
            ------
                clstr_idxs_sets_lst: Returning a python list of python sets of the x_data array
                    row indices for the vectors belonging to each cluster.

        """

        print "In ICM..."

        no_change_cnt = 0
        while no_change_cnt < 2:

            # Calculating the new Clusters.
            for x_idx in np.random.permutation(np.arange(x_data.shape[0])):

                # Skipping the indices should not participate in clustering.
                if np.in1d(x_idx, self.neg_idxs4clstring):
                    continue

                # Setting the initial value for the previews J-Objective value.
                last_jobj = np.Inf

                # Calculating the J-Objective for every x_i vector of the x_data set.
                for i, mu in enumerate(mu_arr):

                    # Getting the indeces for this cluster.
                    clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

                    # Calculating the J-Objective.
                    j_obj = self.JObjCosA(x_idx, x_data, mu, clstr_idxs_arr)

                    if j_obj < last_jobj:
                        last_jobj = j_obj
                        new_clstr_tag = i

                # Checking if the cluster where the data-point belongs into has been changed.
                if clstr_tags_arr[x_idx] != new_clstr_tag:

                    no_change = False

                    # Re-assinging the x_i vector to the new cluster if not already.
                    clstr_tags_arr[x_idx] = new_clstr_tag

                else:
                    no_change = True

            # Counting Non-Changes, i.e. if no change happens for two (2) iteration the...
            # ...re-assingment process stops.
            if no_change:
                no_change_cnt += 1

        # Returning clstr_tags_arr.
        return clstr_tags_arr

    def MeanCosA(self, x_data, clstr_tags_arr):
        """ MeanCosA method: It is calculating the centroids of the hyper-spherical clusters.
            Using the parametrized cosine mean as explained in the documentation.

            Arguments
            ---------
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
                clstr_idxs_lsts: The lists of indices for each cluster.

            Output
            ------
                mu_lst: The list of N centroids(mu_i), one for each of the N expected clusters.

        """

        print "In MeanCosA..."

        mu_lst = list()
        for i in np.arange(self.k_clusters):

            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

            # Summing up all the X data points for the current cluster.
            xi_sum = x_data[clstr_idxs_arr, :].sum(axis=0)

            # Calculating denominator ||Σ xi||
            parametrized_norm_xi = np.sqrt(np.dot(xi_sum, np.transpose(xi_sum)))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            mu_lst.append(xi_sum / parametrized_norm_xi)

        return np.array(mu_lst, dtype=np.float)

    def JObjCosA(self, x_idx, x_data, mu, clstr_idx_arr):
        """ JObjCosA: J-Objective function for parametrized Cosine Distortion Measure. It cannot
            be very generic because the gradient decent (partial derivative) calculations should be
            applied, they are totally dependent on the distortion measure (here is Cosine Distance).

            It is calculating the J-Obective for the specific X data point upon the cosine distance
            plus the must-link and cannot-link constraints.

            Arguments
            ---------
                x_idx: The row index of the x_data array for the specific data-point.
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
                    clstr_idxs_lsts: The lists of indices for each cluster.
                mu: The centroid vector of the current cluster.
                clstr_idxs_set: The set of row indices (from the x_data array) which are assembling
                    the current cluster.

            Output
            ------
                Returning the J-Objective values for the specific x_i in the specific cluster.

        """

        # Calculating the cosine distance of the specific x_i from the cluster's centroid.
        # --------------------------------------------------------------------------------
        dist = 1.0 - np.dot(mu, np.transpose(x_data[x_idx, :]))

        # Calculating Must-Link violation cost.
        # -------------------------------------
        ml_cost = 0.0

        # Getting the index(s) of the must-link-constraints index-table of this data sample.
        idxzof_mli4smpli = np.where(self.must_lnk == x_idx)

        if idxzof_mli4smpli[0].shape[0]:

            # Getting the must-link, with current, data points indeces which they should be in...
            # the same cluster.
            mliz_with_smpli = self.must_lnk[~idxzof_mli4smpli[0], idxzof_mli4smpli[1]]

            # Getting the indeces of must-link than are not in the cluster as they should have been.
            viol_idxs = self.must_lnk[:, ~np.in1d(mliz_with_smpli, clstr_idx_arr)]

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for must-link constraints.
                # NOTE: The violation cost is equivalent to the parametrized Cosine distance...
                # ...which here is equivalent to the (1 - dot product) because the data points...
                # ...assumed to be normalized by the parametrized Norm of the vectors.
                viol_costs = 1.0 - np.dot(x_data[x_idx], np.transpose(x_data[viol_idxs[1]]))

                # Sum-ing up Weighted violations costs.
                ml_cost = np.sum(viol_costs)

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        cl_cost = 0.0

        # Getting the index(s) of the cannot-link-constraints index-table of this data sample.
        idxzof_cli4smpli = np.where(self.cannot_lnk == x_idx)

        if idxzof_cli4smpli[0].shape[0]:

            # Getting the cannot-link, with current, data points indeces which they should not...
            # ...be in the same cluster.
            cliz_with_smpli = self.cannot_lnk[~idxzof_cli4smpli[0], idxzof_cli4smpli[1]]

            # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
            # ...have been.
            viol_idxs = self.cannot_lnk[:, np.in1d(cliz_with_smpli, clstr_idx_arr)]

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance minus the...
                # ...parametrized Cosine distance of the vectors. Since MaxCosine is 1 then...
                # ...maxCosineDistance - CosineDistance == CosineSimilarty of the vectors....
                # ...Again the data points assumed to be normalized.
                viol_costs = np.dot(x_data[x_idx], np.transpose(x_data[viol_idxs[1]]))
                # viol_costs = np.ones_like(viol_costs) - viol_costs

                # Sum-ing up Weighted violations costs.
                cl_cost = np.sum(viol_costs)

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost

    def GlobJObjCosA(self, x_data, mu_arr, clstr_tags_arr):
        """
        """

        print "In GlobalJObjCosA..."

        # Calculating the distance of all vectors, the must-link and cannot-link violations scores.
        sum_d, ml_cost, cl_cost, norm_part_value, params_pdf = 0.0, 0.0, 0.0, 0.0, 0.0
        smlps_cnt, ml_cnt, cl_cnt = 0.0, 0.0, 0.0

        for i, mu in enumerate(mu_arr):

            # Getting the indeces for the i cluster.
            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

            #
            smlps_cnt += clstr_idxs_arr.shape[0]

            # Calculating the cosine distances and add the to the total sum of distances.
            # ---------------------------------------------------------------------------
            sum_d += np.sum(1.0 - np.dot(mu, np.transpose(x_data[clstr_idxs_arr])))

            # Calculating Must-Link violation cost.
            # -------------------------------------

            # Getting the must-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_ml_rows = np.in1d(self.must_lnk[0], clstr_idxs_arr)

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.

            ml_clstr_comn_idxs = np.in1d(
                self.must_lnk, clstr_idxs_arr
            ).reshape(2, self.must_lnk.shape[1])

            ml_viol_columns = np.intersect1d(
                np.where(ml_clstr_comn_idxs[0] != ml_clstr_comn_idxs[1])[0],
                np.hstack((ml_clstr_comn_idxs[0].nonzero()[0], ml_clstr_comn_idxs[1].nonzero()[0]))
            )

            viol_ipairs = self.must_lnk[:, ml_viol_columns]

            #
            ml_cnt += float(viol_ipairs.shape[0])

            if viol_ipairs.shape[0]:

                    # Calculating all pairs of violation costs for must-link constraints.
                    # NOTE: The violation cost is equivalent to the maxCosine distance.
                    viol_costs = 1.0 - np.dot(x_data[viol_ipairs[0]], np.transpose(x_data[viol_ipairs[1]]))

                    if viol_costs.shape[0] > 1:
                        viol_costs_onetime = np.tril(viol_costs, -1)
                    else:
                        viol_costs_onetime = viol_costs

                    # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
                    # ...should be element-by-element.

                    ml_cost += np.sum(viol_costs_onetime)

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_cl_rows = np.in1d(self.cannot_lnk[0], clstr_idxs_arr)

            # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
            # ...have been.

            cl_clstr_comn_idxs = np.in1d(
                self.cannot_lnk, clstr_idxs_arr
            ).reshape(2, self.cannot_lnk.shape[1])

            cl_viol_columns = np.intersect1d(
                np.where(cl_clstr_comn_idxs[0] == cl_clstr_comn_idxs[1])[0],
                cl_clstr_comn_idxs[0].nonzero()[0]
            )

            viol_ipairs = self.cannot_lnk[:, cl_viol_columns]

            #
            cl_cnt += float(viol_ipairs.shape[0])

            if viol_ipairs.shape[0]:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance
                viol_costs = np.dot(x_data[viol_ipairs[0]], np.transpose(x_data[viol_ipairs[1]]))

                if viol_costs.shape[0] > 1:
                    viol_costs_onetime = np.tril(viol_costs, -1)
                else:
                    viol_costs_onetime = viol_costs

                # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
                # ...should be element-by-element. NOTE#2: We are getting only the lower...
                # ...triangle because we need the cosine distance of the constraints pairs...
                # ...only ones.
                cl_cost += np.sum(viol_costs_onetime)

        # Averaging EVERYTHING.
        sum_d = sum_d / smlps_cnt

        if ml_cnt:
            ml_cost = ml_cost / ml_cnt

        if cl_cnt:
            cl_cost = cl_cost / cl_cnt

        print 'dims', x_data.shape[1]
        print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + ml_cost + cl_cost
