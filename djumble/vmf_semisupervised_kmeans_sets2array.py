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

    def __init__(self, k_clusters, ml_cl_cons, init_centroids=None, max_iter=300,
                 cvg=0.001, lrn_rate=0.0003, ray_sigma=0.5, d_params=None,
                 norm_part=False, globj='non-normed', da_pool=None):

        self.da_pool = da_pool
        self.k_clusters = k_clusters
        self.ml_cl_cons = ml_cl_cons
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.cvg = cvg
        self.lrn_rate = lrn_rate
        self.ray_sigma = ray_sigma
        self.A = d_params
        self.norm_part = norm_part

        # This option enables or disables the normalizations values to be included in the...
        # ...calculation of the total values, other than the total cosine distances, the...
        # ...total must-link and cannot-link violation scores.
        if globj == 'non-normed':
            self.globj = False
        elif globj == 'proper':
            self.globj = True
        else:
            raise Exception("globj: can be either 'proper' or 'non-normed'.")

    def fit(self, x_data):
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

        # Setting up distortion parameters if not have been passed as class argument.
        if self.A is None:
            self.A = np.random.uniform(0.50, 100.0, size=x_data.shape[1])
        # A should be a diagonal matrix form for the calculations in the functions bellow. The...
        # ...sparse form will save space and the csr_matrix will make the dia_matrix write-able.
        # self.A = sp.sparse.dia_matrix((self.A, [0]), shape=(self.A.shape[0], self.A.shape[0]))
        # self.A = sp.sparse.coo_matrix(self.A)
        self.A = np.diag(self.A)

        # Setting up the violation weights matrix if not have been passed as class argument.
        # if self.w_violations is None:
        #     self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        #     # ### I am not sure what kind of values this weights should actually have.

        # Converting the weights violation matrix into a sparse matrix.
        non_zero = np.where(self.ml_cl_cons != 0, 1, 0)
        # self.w_violations = sp.sparse.csr_matrix(np.multiply(self.w_violations, non_zero))
        # self.w_violations = np.multiply(self.w_violations, non_zero)

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint()
        # init_clstr_sets_lst = ConsolidateAL()
        clstr_tags_arr = np.empty(x_data.shape[0], dtype=np.int)
        clstr_tags_arr[:] = np.Inf

        # Selecting a random set of centroids if not any given as class initialization argument.
        if not self.init_centroids:
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

        # NOTE: Initially normalizing the samples under the distortion parameters values in...
        # ...order to reducing the cosine distance calculation to a simple dot products...
        # ...calculation in the rest of the EM (sub)steps.
        x_data = np.divide(
            x_data,
            np.sqrt(
                np.diag(np.dot(np.dot(x_data, self.A), x_data.T)),
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
            start_tm = tm.time()

            # The E-Step.

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            clstr_tgs_arr = self.ICM(x_data, mu_arr, clstr_tags_arr)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_arr = self.MeanCosA(x_data, clstr_tgs_arr)
            # print mu_lst

            # Re-estimating distortion measure parameters upon the new clusters set-up.
            self.A = self.UpdateDistorParams(self.A, x_data, mu_arr, clstr_tags_arr)

            # NOTE: Normalizing the samples under the new parameters values in order to reducing...
            # ...the cosine distance calculation to a simple dot products calculation in the...
            # ...rest of the EM (sub)steps.
            x_data = np.divide(
                x_data,
                np.sqrt(
                    np.diag(np.dot(np.dot(x_data, self.A), x_data.T)),
                    dtype=np.float
                ).reshape(x_data.shape[0], 1)
            )

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_arr, clstr_tags_arr)

            timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            print "Time elapsed : %d:%d:%d:%d" % timel

            # Terminating upon the difference of the last two Global JObej values.
            if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:
                # second condition is TEMP!
                print 'last_gobj - glob_jobj', last_gobj - glob_jobj
                print "Global Objective", glob_jobj
                break
            else:
                last_gobj = glob_jobj

            print "Global Objective", glob_jobj

        # Storing the amount of iterations until convergence.
        self.conv_step = conv_step

        # Closing the internal process pool.
        # self.da_pool.close()
        # self.da_pool.join()

        # Returning the Centroids and the Clusters,i.e. the set of indeces for each cluster.
        return mu_arr, clstr_tags_arr

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'final_iter': self.conv_step,
            'convg_diff': self.cvg,
            'lrn_rate': self.lrn_rate,
            'ray_sigma': self.ray_sigma,
            'dist_msur_params': self.A,
            'norm_part': self.norm_part
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
        start_tm = tm.time()

        no_change_cnt = 0
        while no_change_cnt < 2:

            # Calculating the new Clusters.
            for x_idx in np.random.permutation(np.arange(x_data.shape[0])):

                # Setting the initial value for the previews J-Objective value.
                last_jobj = np.Inf

                # Calculating the J-Objective for every x_i vector of the x_data set.
                for i, mu in enumerate(mu_arr):

                    # Getting the indeces for this cluster.
                    clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

                    # Calculating the J-Objective.
                    j_obj = np.round(self.JObjCosA(x_idx, x_data, mu, clstr_idxs_arr), 3)

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

        timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
        print "ICM time: %d:%d:%d:%d" % timel

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

            # Calculating denominator ||Σ xi||(A)
            parametrized_norm_xi = np.sqrt(np.dot(np.dot(xi_sum, self.A), xi_sum.T))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            mu_lst.append(xi_sum / parametrized_norm_xi)

        return np.array(mu_lst, dtype=np.float)

    def NormPart(self, x_data_subset):
        """ The von Mises and von Mises - Fisher Logarithmic Normalization partition function:...
            is calculated in this method. For the 2D data the function is simplified for faster
            calculation.

            *** This function has been build after great research on the subject. However, some
            things are not very clear this is always in a revision state until theoretically proven
            to be correctly used.

            Arguments
            ---------
                x_data_subset: The subset of the data point are included in the von Mises-Fisher
                    distribution.

            Output
            ------
                The logarithmic value of the partition normalization function.

        """

        # Calculating the r.
        # The r it suppose to be the norm of the data points of the current cluster, not the...
        # ...whole mixture as it is in the global objective function. ## Might this need to be...
        # ...revised.
        r = np.linalg.norm(x_data_subset)

        # Calculating the Von Misses Fisher's k concentration is approximated as seen..
        # ....in Banerjee et al. 2003.
        dim = x_data_subset.shape[1]

        if dim == 1:
            raise Exception("Data points cannot have less than two(2) dimension.")

        if dim > 100:
            # Returning a heuristically found constant value as normalizer because when the...
            # ...dimentions are very high Bessel function equals Zero.
            return dim * x_data_subset.shape[0]

        # Calculating the partition function depending on the vector dimensions.
        k = (r*dim - np.power(r, 3.0)) / (1 - np.power(r, 2.0))
        # k=0.001 only for the case where the r is too small. Usually at the 1-2 first iterations...
        # ...of the EM/Kmeans.
        if k < 0.0:
            k = 0.001

        if dim > 3 and dim <= 100:

            # This is the proper way for calculating the von Misses Fisher normalization factor.
            bessel = np.abs(special.jv((dim/2.0)-1.0, k))
            # bessel = np.abs(self.Jd((dim/2.0)-1.0, k))
            cdk = np.power(k, (dim/2.0)-1) / (np.power(2*np.pi, dim/2)*bessel)

        elif dim == 2:

            # This is the proper way for calculating the vMF normalization factor for 2D vectors.
            bessel = np.abs(special.jv(0, k))
            # bessel = np.abs(self.Jd(0, k))
            cdk = 1.0 / (2*np.pi*bessel)

        # Returning the log of the normalization function plus the log of the k that is used in...
        # ...the von Mises Fisher PDF, which is separated from the Cosine Distance due to the log.
        # The normalizers are multiplied by the number of all the X data subset, because it is...
        # the global normalizer after the whole summations has been completed.
        # Still this need to be revised.
        return (np.log(cdk) + np.log(k)) * x_data_subset.shape[0]

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
        start_tm = tm.time()

        # Calculating the cosine distance of the specific x_i from the cluster's centroid.
        # --------------------------------------------------------------------------------
        dist = 1.0 - np.dot(np.dot(mu, self.A), x_data[x_idx, :].T)

        # Calculating Must-Link violation cost.
        # -------------------------------------
        ml_cost = 0.0

        # Getting the must-link (if any) indeces for this index (i.e. data sample).
        mst_lnk_idxs = np.where(self.ml_cl_cons[0, x_idx] == 1)[0]

        # Getting the indeces of must-link than are not in the cluster as they should have been.
        viol_idxs = mst_lnk_idxs[~np.in1d(mst_lnk_idxs, clstr_idx_arr)]

        if viol_idxs.shape[0]:

            # Calculating all pairs of violation costs for must-link constraints.
            # NOTE: The violation cost is equivalent to the parametrized Cosine distance which...
            # ...here is equivalent to the (1 - dot product) because the data points assumed...
            # ...to be normalized by the parametrized Norm of the vectors.
            viol_costs = 1.0 - np.dot(
                np.dot(x_data[x_idx], self.A),
                x_data[viol_idxs].T
            )

            # Sum-ing up Weighted violations costs.
            ml_cost = np.sum(viol_costs)
            # np.multiply(self.w_violations[x_idx, viol_idxs], viol_costs)

            # Equivalent to: (in a for-loop implementation)
            # cl_cost += self.w_violations[x[0], x[1]] *\
            #  self.CosDistA(x_data[x[0], :], x_data[x[1], :])

            ml_cost = ml_cost / float(len(viol_idxs))

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        cl_cost = 0.0

        # Getting the cannot-link (if any) indeces for this index (i.e. data sample).
        cnt_lnk_idxs = np.where(self.ml_cl_cons[0, x_idx] == -1)[0]

        # Getting the indeces of cannot-link than are in the cluster as they shouldn't have been.
        viol_idxs = cnt_lnk_idxs[np.in1d(cnt_lnk_idxs, clstr_idx_arr)]

        if viol_idxs.shape[0]:

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance minus the...
            # ...parametrized Cosine distance of the vectors. Since MaxCosine is 1 then...
            # ...maxCosineDistance - CosineDistance == CosineSimilarty of the vectors....
            # ...Again the data points assumed to be normalized.
            viol_costs = np.dot(np.dot(x_data[x_idx], self.A), x_data[viol_idxs].T)
            # viol_costs = np.ones_like(viol_costs) - viol_costs

            # Sum-ing up Weighted violations costs.
            cl_cost = np.sum(viol_costs)
            # np.multiply(self.w_violations[x_idx, viol_idxs], viol_costs)

            # Equivalent to: (in a for-loop implementation)
            # cl_cost += self.w_violations[x[0], x[1]] *\
            # (1 - self.CosDistA(x_data[x[0], :], x_data[x[1], :]))

            cl_cost = cl_cost / float(len(viol_idxs))

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        sum1, sum2 = 0.0, 0.0
        for a in np.diag(self.A):
            sum1 += np.log(a)
            sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
        params_pdf = sum1 - sum2 - (2 * np.diag(self.A).shape[0] * np.log(self.ray_sigma))
        params_pdf = 0.0

        # Calculating the log normalization function of the von Mises-Fisher distribution...
        # ...NOTE: Only for this cluster i.e. this vMF of the whole PDF mixture.
        norm_part_value = 0.0
        if self.norm_part:
            norm_part_value = self.NormPart(x_data[clstr_idx_arr])

        # print "In JObjCosA...", dist, ml_cost, cl_cost, params_pdf, norm_part_value
        # print "Params are: ", self.A

        #timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
        #print "Jobj time: %d:%d:%d:%d" % timel

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost - params_pdf + norm_part_value

    def GlobJObjCosA(self, x_data, mu_arr, clstr_tags_arr):
        """
        """

        print "In GlobalJObjCosA..."

        # Getting all the must-link (if any) indeces.
        mst_lnk_idxs = np.where(self.ml_cl_cons == 1)

        # Getting all the cannot-link (if any) indeces.
        cnt_lnk_idxs = np.where(self.ml_cl_cons == -1)

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
            sum_d += np.sum(1.0 - np.dot(np.dot(mu, self.A), x_data[clstr_idxs_arr].T))

            # Calculating Must-Link violation cost.
            # -------------------------------------

            # Getting the must-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_ml_rows = np.in1d(mst_lnk_idxs[0], clstr_idxs_arr)

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.
            viol_idxs = mst_lnk_idxs[1][
                ~np.in1d(mst_lnk_idxs[1][in_clstr_ml_rows], clstr_idxs_arr)
            ]

            viol_pair_idx = mst_lnk_idxs[0][
                ~np.in1d(mst_lnk_idxs[1][in_clstr_ml_rows], clstr_idxs_arr)
            ]

            #
            ml_cnt += float(len(viol_idxs))

            if viol_idxs.shape[0]:

                    # Calculating all pairs of violation costs for must-link constraints.
                    # NOTE: The violation cost is equivalent to the maxCosine distance.
                    viol_costs = 1.0 - np.dot(
                        np.dot(x_data[viol_pair_idx], self.A),
                        x_data[viol_idxs].T
                    )

                    if viol_costs.shape[0] > 1:
                        viol_costs_onetime = np.tril(viol_costs, -1)
                    else:
                        viol_costs_onetime = viol_costs

                    # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
                    # ...should be element-by-element.

                    ml_cost += np.sum(viol_costs_onetime)
                    #     np.multiply(
                    #         self.w_violations[viol_pair_idx, viol_idxs],
                    #         viol_costs_onetime
                    #     )
                    # )

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_cl_rows = np.in1d(cnt_lnk_idxs[0], clstr_idxs_arr)

            # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
            # ...have been.
            viol_idxs = cnt_lnk_idxs[1][
                np.in1d(cnt_lnk_idxs[1][in_clstr_cl_rows], clstr_idxs_arr)
            ]

            viol_pair_idx = cnt_lnk_idxs[0][
                np.in1d(cnt_lnk_idxs[1][in_clstr_cl_rows], clstr_idxs_arr)
            ]

            #
            cl_cnt += float(len(viol_idxs))

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance
                viol_costs = np.dot(
                    np.dot(x_data[viol_pair_idx], self.A),
                    x_data[viol_idxs].T
                )

                if viol_costs.shape[0] > 1:
                    viol_costs_onetime = np.tril(viol_costs, -1)
                else:
                    viol_costs_onetime = viol_costs

                # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
                # ...should be element-by-element. NOTE#2: We are getting only the lower...
                # ...triangle because we need the cosine distance of the constraints pairs...
                # ...only ones.
                cl_cost += np.sum(viol_idxs)
                #     np.multiply(
                #         self.w_violations[viol_pair_idx, viol_idxs],
                #         viol_costs_onetime
                #     )
                # )

        # Averaging EVERYTHING.
        smlps_cnt = sum_d / smlps_cnt
        if ml_cnt:
            ml_cost = ml_cost / ml_cnt
        if cl_cnt:
            cl_cost = cl_cost / cl_cnt

        # Calculating the log normalization function of the von Mises-Fisher...
        # ...distribution of the whole mixture.
        if self.norm_part and self.globj:
            norm_part_value += self.NormPart(x_data[clstr_idxs_arr])

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        if self.globj:
            sum1, sum2 = 0.0, 0.0
            for a in np.diag(self.A):
                sum1 += np.log(a)
                sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
            params_pdf = sum1 - sum2 - (2 * np.diag(self.A).shape[0] * np.log(self.ray_sigma))

        print 'dims', x_data.shape[1]
        print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost
        print 'np.log(Rayleigh)', params_pdf
        print 'N*(np.log(cdk) + np.log(k))', norm_part_value

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + ml_cost + cl_cost - params_pdf + norm_part_value

    def UpdateDistorParams(self, A, x_data, mu_arr, clstr_tags_arr):
        """ Update Distortion Parameters: This function is updating the whole set of the distortion
            parameters. In particular the parameters for the Cosine Distance in this implementation
            of the HMRF Kmeans.

            Arguments
            ---------
                A: The diagonal sparse matrix of the cosine distance parameters. This is actually
                    not necessary to be passed as argument but it is here for coding constancy
                    reasons.
                x_data: A numpy.array with X rows of data points and N rows of features/dimensions.
                    clstr_idxs_lsts: The lists of indices for each cluster.
                mu_lst: The list of N centroids(mu_i), one for each of the N expected clusters.
                clstr_idxs_set_lst: The list of sets of x_data array indices for each of the N
                    clusters.

            Output
            ------
                Returning the updated A paramters diagonal (sparse) matrix. Again this is a
                redundant step just for coding constancy reasons.

        """

        print "In UpdateDistorParams..."

        start_tm = tm.time()

        # Getting all the must-link (if any) indeces.
        mst_lnk_idxs = np.where(self.ml_cl_cons == 1)

        # Getting all the cannot-link (if any) indeces.
        cnt_lnk_idxs = np.where(self.ml_cl_cons == -1)

        print "Dispatching"

        process_lst = list()
        update_tups_lst = mp.Manager().list()

        tmp = np.ctypeslib.as_ctypes(x_data)
        x_data_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        x_data_shape = x_data.shape
        print 'Shape', x_data_shape

        tmp = np.ctypeslib.as_ctypes(A)
        A_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        A_shape = A.shape

        tmp = np.ctypeslib.as_ctypes(mu_arr)
        mu_arr_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        mu_arr_shape = mu_arr.shape

        tmp = np.ctypeslib.as_ctypes(clstr_tags_arr)
        clstr_tags_arr_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        clstr_tags_arr_shape = clstr_tags_arr.shape

        mst_lnk_idxs0 = np.zeros_like(mst_lnk_idxs[0])
        mst_lnk_idxs0[:] = mst_lnk_idxs[0][:]
        tmp = np.ctypeslib.as_ctypes(mst_lnk_idxs0)
        mst_lnk_idxs0_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        mst_lnk_idxs0_shape = mst_lnk_idxs0.shape

        mst_lnk_idxs1 = np.zeros_like(mst_lnk_idxs[1])
        mst_lnk_idxs1[:] = mst_lnk_idxs[1][:]
        tmp = np.ctypeslib.as_ctypes(mst_lnk_idxs1)
        mst_lnk_idxs1_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        mst_lnk_idxs1_shape = mst_lnk_idxs1.shape

        cnt_lnk_idxs0 = np.zeros_like(cnt_lnk_idxs[0])
        cnt_lnk_idxs0[:] = cnt_lnk_idxs[0][:]
        tmp = np.ctypeslib.as_ctypes(cnt_lnk_idxs0)
        cnt_lnk_idxs0_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        cnt_lnk_idxs0_shape = cnt_lnk_idxs0.shape

        cnt_lnk_idxs1 = np.zeros_like(cnt_lnk_idxs[1])
        cnt_lnk_idxs1[:] = cnt_lnk_idxs[1][:]
        tmp = np.ctypeslib.as_ctypes(cnt_lnk_idxs1)
        cnt_lnk_idxs1_share = mp_sct.Array(tmp._type_, tmp, lock=False)
        cnt_lnk_idxs1_shape = cnt_lnk_idxs1.shape

        for i, a_idx_range in enumerate([(0, 350), (350, 700)]):

            print a_idx_range

            process_lst.append(
                mp.Process(
                    target=UpdateParam,
                    args=(
                        update_tups_lst, a_idx_range,
                        x_data_share, mu_arr_share, clstr_tags_arr_share,
                        mst_lnk_idxs0_share, mst_lnk_idxs1_share,
                        cnt_lnk_idxs0_share, cnt_lnk_idxs1_share,
                        A_share, x_data_shape, mu_arr_shape, clstr_tags_arr_shape,
                        mst_lnk_idxs0_shape, mst_lnk_idxs1_shape,
                        cnt_lnk_idxs0_shape, cnt_lnk_idxs1_shape,
                        A_shape,
                        self.lrn_rate, self.ray_sigma
                    )
                )
            )
            process_lst[i].start()

        timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
        print "Dispatch time: %d:%d:%d:%d" % timel

        for p in process_lst:
            p.join()

        start_tm = tm.time()

        print "Collecting/Updating"

        for a_idx, a in update_tups_lst:
            # print a_idx, a
            A[a_idx, a_idx] = a

        timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
        print "Collect/Update time: %d:%d:%d:%d" % timel

        # Returning the A parameters. This is actually a dump return for coding constance reasons.
        return A


def UpdateParam(update_tups_lst, a_idx_range, x_data_share, mu_arr_share, clstr_tags_arr_share,
                mst_lnk_idxs0_share, mst_lnk_idxs1_share, cnt_lnk_idxs0_share, cnt_lnk_idxs1_share,
                A_share, x_data_shape, mu_arr_shape, clstr_tags_arr_shape,
                mst_lnk_idxs0_shape, mst_lnk_idxs1_shape,
                cnt_lnk_idxs0_shape, cnt_lnk_idxs1_shape,
                A_shape, lrn_rate, ray_sigma):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        print 'IN'
        x_data = np.ctypeslib.as_array(x_data_share, x_data_shape)
        mu_arr = np.ctypeslib.as_array(mu_arr_share, mu_arr_shape)
        clstr_tags_arr = np.ctypeslib.as_array(clstr_tags_arr_share, clstr_tags_arr_shape)
        mst_lnk_idxs0 = np.ctypeslib.as_array(mst_lnk_idxs0_share, mst_lnk_idxs0_shape)
        cnt_lnk_idxs0 = np.ctypeslib.as_array(cnt_lnk_idxs0_share, cnt_lnk_idxs0_shape)
        mst_lnk_idxs1 = np.ctypeslib.as_array(mst_lnk_idxs1_share, mst_lnk_idxs1_shape)
        cnt_lnk_idxs1 = np.ctypeslib.as_array(cnt_lnk_idxs1_share, cnt_lnk_idxs1_shape)
        mst_lnk_idxs = (mst_lnk_idxs0, mst_lnk_idxs1)
        cnt_lnk_idxs = (cnt_lnk_idxs0, cnt_lnk_idxs1)
        A = np.ctypeslib.as_array(A_share, A_shape)

    # update_tups_lst = list()

    for a_idx, a in zip(range(a_idx_range[0], a_idx_range[1]), np.diag(A)[a_idx_range[0]:a_idx_range[1]]):

        print 'IDX', a_idx

        # Initializing...
        xm_pderiv, mlcost_pderiv, clcost_pderiv = 0.0, 0.0, 0.0
        smpls_cnt, ml_cnt, cl_cnt = 0.0, 0.0, 0.0

        for i, mu in enumerate(mu_arr):

            # Getting the indeces for the i cluster.
            clstr_idxs = np.where(clstr_tags_arr == i)[0]

            smpls_cnt += float(clstr_idxs.shape[0])

            # Calculating the partial derivatives of each parameter for all cluster's member...
            # ...for each cluster.
            # ---------------------------------------------------------------------------------
            for x_clstr_idx in clstr_idxs:
                xm_pderiv += PartialDerivative(a_idx, x_data[x_clstr_idx, :], mu, A)

            # Calculating Must-Link violation cost.
            # -------------------------------------

            # Getting the must-link left side of the pair constraints, i.e. the row...
            # ...indeces of the constraints matrix that are in the cluster's set of...
            # ...indeces.
            in_clstr_ml_rows = np.in1d(mst_lnk_idxs[0], clstr_idxs)

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.
            viol_idxs = mst_lnk_idxs[1][
                ~np.in1d(mst_lnk_idxs[1][in_clstr_ml_rows], clstr_idxs)
            ]

            ml_cnt += float(viol_idxs.shape[0])

            if viol_idxs.shape[0]:

                # Calculating the partial derivatives of all pairs of violations for...
                # ...must-link constraints.
                for x in zip(mst_lnk_idxs[0][in_clstr_ml_rows], viol_idxs):
                    mlcost_pderiv -= PartialDerivative(
                        a_idx, x_data[x[0], :], x_data[x[1], :], A
                    )

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------

            # Getting the cannot-link left side of the pair constraints, i.e. the row...
            # ...indeces of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_cl_rows = np.in1d(cnt_lnk_idxs[0], clstr_idxs)

            # Getting the indeces of cannot-link than are in the cluster as they...
            # ...shouldn't have been.
            viol_idxs = cnt_lnk_idxs[1][np.in1d(cnt_lnk_idxs[1][in_clstr_cl_rows], clstr_idxs)]

            cl_cnt += float(viol_idxs.shape[0])

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance
                for x in zip(cnt_lnk_idxs[0][in_clstr_cl_rows], viol_idxs):
                    clcost_pderiv += PartialDerivative(
                        a_idx, x_data[x[0], :], x_data[x[1], :], A
                    )

        # Averaging EVERYTHING
        xm_pderiv = xm_pderiv / smpls_cnt
        if ml_cnt:
            mlcost_pderiv = mlcost_pderiv / ml_cnt
        if cl_cnt:
            clcost_pderiv = clcost_pderiv / cl_cnt

        # Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
        a_pderiv = (1 / a) - (a / np.square(ray_sigma))
        # print 'Rayleigh Partial', a_pderiv
        a_pderiv = 0.0
        # print 'a', a
        # print 'xm', xm_pderiv
        # print 'ml', mlcost_pderiv
        # print 'cl', clcost_pderiv

        A[a_idx, a_idx] = a + (lrn_rate * (xm_pderiv + mlcost_pderiv + clcost_pderiv - a_pderiv))
        # print new_val
        # print (xm_pderiv[0] + mlcost_pderiv + clcost_pderiv - a_pderiv)
        # print 'ln', lrn_rate
        # print 'new_val', new_val
        #update_tups_lst.append((a_idx, new_val))

    # Changing a diagonal value of the A cosine similarity parameters measure.
    # return update_tups_lst
    # q.put(update_tups_lst)


def PartialDerivative(a_idx, x1, x2, A):
    """ Partial Derivative: This method is calculating the partial derivative of a specific
        parameter given the proper vectors. That is, for the cosine distance is a x_i with the
        centroid vector (mu) of the cluster where x_i is belonging into. As for the constraint
        violations is the x_1 and x_2 of a specific pair of constraints each time this method
        is called.
        **for detail see documentation.

        Arguments
        ---------
            a_idx: The index of the parameter on the diagonal of the A diagonal sparse
                parameters matrix.
            x1, x2: The vectors will be used for the partial derivative calculation.

        Output
        ------
            res_a: The partial derivative's value.

    """

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = np.sqrt(np.dot(np.dot(x1, A), x1.reshape(x1.shape[0], 1)))
    x2_pnorm = np.sqrt(np.dot(np.dot(x2, A), x1.reshape(x2.shape[0], 1)))

    res_a = (
                (x1[a_idx] * x2[a_idx] * x1_pnorm * x2_pnorm) -
                (
                    np.dot(np.dot(x1, A), x2.reshape(x2.shape[0], 1)) *
                    (
                        (
                            np.square(x1[a_idx]) * np.square(x2_pnorm) +
                            np.square(x2[a_idx]) * np.square(x1_pnorm)
                        ) / (2 * x1_pnorm * x2_pnorm)
                    )
                )
            ) / (np.square(x1_pnorm) * np.square(x2_pnorm))

    return res_a


if __name__ == '__main__':

    test_dims = 700

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
        [1, 5],
        [1, 3],
        [1, 6],
        [1, 8],
        [7, 3],
        [521, 525],
        [521, 528],
        [521, 539],
        [535, 525],
        [537, 539],
        [1037, 1238],
        [1057, 1358],
        [1039, 1438],
        [1045, 1138],
        [1098, 1038],
        [1019, 1138],
        [1087, 1338]
    ]

    cannot_lnk_con = [
        [1, 521],
        [1, 525],
        [1, 528],
        [1, 535],
        [1, 537],
        [1, 539],
        [5, 521],
        [5, 525],
        [5, 528],
        [5, 35],
        [8, 521],
        [8, 525],
        [8, 528],
        [8, 535],
        [8, 537],
        [8, 539],
        [3, 521],
        [3, 535],
        [3, 537],
        [3, 539],
        [6, 521],
        [6, 525],
        [6, 528],
        [6, 535],
        [6, 537],
        [6, 539],
        [7, 521],
        [7, 525],
        [7, 528],
        [7, 535],
        [7, 537],
        [7, 539],
        [538, 1237],
        [548, 1357],
        [558, 1437],
        [738, 1137],
        [938, 1037],
        [838, 1039],
        [555, 1337]
    ]

    k_clusters = 3
    init_centrs = [0, 550, 1100]

    # ml_cl_cons = sp.sparse.csr_matrix(np.zeros((x_data_2d_arr.shape[0], x_data_2d_arr.shape[0]), dtype=np.int))
    ml_cl_cons = np.zeros((x_data_2d_arr.shape[0], x_data_2d_arr.shape[0]), dtype=np.int)
    for ml1, ml2 in must_lnk_con:
        ml_cl_cons[ml1, ml2] = 1
    for cl1, cl2 in cannot_lnk_con:
        ml_cl_cons[cl1, cl2] = -1

    # ml_cl_cons = sp.sparse.coo_matrix(ml_cl_cons)

    # os.system("taskset -p 0xff %d" % os.getpid())
    # os.system("OPENBLAS_MAIN_FREE=1")

    print 'CPUs', mp.cpu_count()
    da_pool = None # mp.Pool(2)
    # mymp.SynergeticPool(synergetic_servers=None, local_workers=mp.cpu_count())

    print "Running HMRF Kmeans"
    hkmeans = HMRFKmeans(k_clusters,  ml_cl_cons, init_centroids=init_centrs,
                         max_iter=300, cvg=0.001, lrn_rate=0.0003, ray_sigma=1.0,
                         d_params=np.random.uniform(1.0, 1.0, size=test_dims), norm_part=False,
                         globj='non-normed', da_pool=da_pool)
    res = hkmeans.fit(x_data_2d_arr)

    print list(res[1])

    for mu_idx, mu in enumerate(res[0]):

        clstr_idxs = np.where(res[1] == mu_idx)[0]

        for xy in x_data_2d_arr[clstr_idxs]:
            plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)
        # plt.plot(x_data_2d_arr2, '^')
        # plt.plot(x_data_2d_arr3, '>')

    plt.show()
