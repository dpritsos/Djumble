# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import scipy.special as special
import time as tm

cimport numpy as cnp

cdef extern from "math.h":
    cdef double sqrt(double x)
    cdef double pow(double x, double y)
    cdef double log (double x)

cdef class HMRFKmeans:
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

    # Attributes (Here used for perfornace accelaration).
    cdef cnp.intp_t k_clusters
    cdef cnp.intp_t [:, ::1] must_lnk
    cdef cnp.intp_t [:, ::1] cannot_lnk
    cdef cnp.intp_t [:, :] ml_sorted
    cdef cnp.intp_t [:, :] cl_sorted
    cdef cnp.intp_t ml_size
    cdef cnp.intp_t cl_size
    cdef cnp.intp_t cl_sorted_size
    cdef cnp.intp_t ml_sorted_size
    cdef cnp.intp_t [::1] init_centroids
    cdef double ml_wg
    cdef double cl_wg
    cdef cnp.intp_t max_iter
    cdef double cvg
    cdef double lrn_rate
    cdef double ray_sigma
    cdef double [::1] d_params
    cdef bint norm_part
    cdef bint globj_norm
    cdef double [::1] A
    cdef cnp.intp_t A_size
    cdef cnp.intp_t xdata_size
    cdef cnp.intp_t conv_step
    cdef cnp.intp_t [::1] neg_idxs4clstring
    cdef cnp.intp_t neg_i4c_size

    def __init__(self, cnp.intp_t k_clusters, cnp.intp_t [:, ::1] must_lnk,
                 cnp.intp_t [:, ::1] cannot_lnk, cnp.intp_t [::1] init_centroids=None,
                 double ml_wg=1.0, double cl_wg=1.0, int max_iter=300,
                 double cvg=0.001, double lrn_rate=0.003, double ray_sigma=0.5,
                 double [::1] d_params=None, bint norm_part=False, bint globj_norm=False):

        self.k_clusters = k_clusters

        self.must_lnk = must_lnk
        tmp_arr = np.hstack((must_lnk, must_lnk[::-1, :]))
        self.ml_sorted = tmp_arr[:, np.argsort(tmp_arr)[0]]

        self.cannot_lnk = cannot_lnk
        tmp_arr = np.hstack((cannot_lnk, cannot_lnk[::-1, :]))
        self.cl_sorted = tmp_arr[:, np.argsort(tmp_arr)[0]]

        self.ml_size = self.must_lnk.shape[1]
        self.cl_size = self.cannot_lnk.shape[1]

        self.cl_sorted_size = self.cl_sorted.shape[1]
        self.ml_sorted_size = self.ml_sorted.shape[1]

        self.init_centroids = init_centroids
        self.ml_wg = ml_wg
        self.cl_wg = cl_wg
        self.max_iter = max_iter
        self.cvg = cvg
        self.lrn_rate = lrn_rate
        self.ray_sigma = ray_sigma
        self.A = d_params
        if d_params != None:
            self.A_size = d_params.shape[0]
        self.norm_part = norm_part

        # This option enables or disables the normalizations values to be included in the...
        # ...calculation of the total values, other than the total cosine distances, the...
        # ...total must-link and cannot-link violation scores.
        self.globj_norm = globj_norm


    def fit(self, double [:, ::1] x_data, cnp.intp_t [::1] neg_idxs4clstring):
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
        cdef cnp.intp_t i, idx, conv_step

        self.xdata_size = x_data.shape[0]

        # Setting up distortion parameters if not have been passed as class argument.
        if self.A == None:
            self.A = np.ones((x_data.shape[1]), dtype=np.float)
            self.A_size = self.A.shape[0]
        elif self.A.shape[0] != x_data.shape[1]:
            raise Exception(
                "Dimension mismutch amogst distortion params A[] vector and x_data features size."
            )

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint()
        # init_clstr_sets_lst = ConsolidateAL()
        clstr_tags_arr = np.empty(x_data.shape[0], dtype=np.int)
        clstr_tags_arr[:] = np.Inf

        # Selecting a random set of centroids if not any given as class initialization argument.
        if self.init_centroids == None:
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
        self.neg_i4c_size = self.neg_idxs4clstring.shape[0]

        if self.neg_i4c_size > 0:
            for i_idx in range(self.neg_i4c_size):
                clstr_tags_arr[self.neg_idxs4clstring[i_idx]] = -9

        # NOTE: Initially normalizing the samples under the distortion parameters values in...
        # ...order to reducing the cosine distance calculation to a simple dot products...
        # ...calculation in the rest of the EM (sub)steps.
        # print np.array(x_data[0, 0:100])
        x_data = np.divide(
            x_data,
            np.sqrt(
                np.diag(np.dot(self.dot2d_ds(x_data, self.A), x_data.T)),
                dtype=np.float
            ).reshape(x_data.shape[0], 1)
        )
        # print np.array(x_data[0, 0:100])

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
            # start_tm = tm.time()

            # The E-Step.

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            clstr_tags_arr = self.ICM(x_data, mu_arr, clstr_tags_arr)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_arr = self.MeanCosA(x_data, clstr_tags_arr)

            # Re-estimating distortion measure parameters upon the new clusters set-up.
            self.A = self.UpdateDistorParams(self.A, x_data, mu_arr, clstr_tags_arr)

            # NOTE: Normalizing the samples under the new parameters values in order to reducing...
            # ...the cosine distance calculation to a simple dot products calculation in the...
            # ...rest of the EM (sub)steps.
            x_data = np.divide(
                x_data,
                np.sqrt(
                    np.diag(np.dot(self.dot2d_ds(x_data, self.A), x_data.T)),
                    dtype=np.float
                ).reshape(x_data.shape[0], 1)
            )

            # ##################### KOLPAKI - Recalculating centroids upon the new clusters set-up.
            mu_arr = self.MeanCosA(x_data, clstr_tags_arr)

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_arr, clstr_tags_arr)

            # timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            # print "Time elapsed : %d:%d:%d:%d" % timel

            # Terminating upon the difference of the last two Global JObej values.
            if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:
                # second condition is TEMP!
                print 'last_gobj - glob_jobj', last_gobj - glob_jobj
                print "Global Objective", glob_jobj
                break
            else:
                last_gobj = glob_jobj

            print "Global Objective (narray)", glob_jobj

        # Storing the amount of iterations until convergence.
        self.conv_step = conv_step

        # Returning the Centroids and the Clusters,i.e. the set of indeces for each cluster.
        return mu_arr, clstr_tags_arr

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'final_iter': self.conv_step,
            'ml_wg': self.ml_wg,
            'cl_wg': self.cl_wg,
            'convg_diff': self.cvg,
            'lrn_rate': self.lrn_rate,
            'ray_sigma': self.ray_sigma,
            'dist_msur_params': self.A,
            'norm_part': self.norm_part
        }

    cdef ICM(self, double [:, ::1] x_data, double [:, ::1] mu_arr, object clstr_tags_arr):
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

        cdef cnp.intp_t no_change_cnt = 0
        cdef cnp.intp_t x_idx
        cdef cnp.intp_t i
        cdef double [::1] mu
        cdef double j_obj
        cdef double last_jobj
        cdef bint no_change
        cdef cnp.intp_t [::1] clstr_idxs_arr
        cdef cnp.intp_t new_clstr_tag
        cdef bint skip_smpl = False

        while no_change_cnt < 2:

            # Calculating the new Clusters.
            for x_idx in np.random.permutation(np.arange(x_data.shape[0])):

                # Looking for skipping indices that should not participate in clustering.
                for i in range(self.neg_i4c_size):
                    if self.neg_idxs4clstring[i] == x_idx:
                        skip_smpl = True

                # Skipping the indices should not participate in clustering.
                if not skip_smpl:

                    # Setting the initial value for the previews J-Objective value.
                    last_jobj = np.Inf

                    # Calculating the J-Objective for every x_i vector of the x_data set.
                    for i, mu in enumerate(mu_arr):

                        # Getting the indeces for this cluster.
                        clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

                        # Calculating the J-Objective.
                        j_obj = <double>self.JObjCosA(x_idx, x_data, mu, clstr_idxs_arr)

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

                else:
                    # Reseting skipping samples flag.
                    skip_smpl = False

            # Counting Non-Changes, i.e. if no change happens for two (2) iteration the...
            # ...re-assingment process stops.
            if no_change:
                no_change_cnt += 1

        # Returning clstr_tags_arr.
        return clstr_tags_arr

    cdef MeanCosA(self, double [:, ::1] x_data, object clstr_tags_arr):
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

        cdef cnp.intp_t i
        mu_arr = np.zeros((self.k_clusters, x_data.shape[1]))

        for i in range(self.k_clusters):

            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

            # Summing up all the X data points for the current cluster.
            if len(clstr_idxs_arr):
                # This line should be chaged with a pure c-like code for increading perfornace.
                xi_sum = np.sum(np.asarray(x_data)[clstr_idxs_arr, :], axis=0)
                # ####
            else:
                print "Zero Mean for a clucter triggered!!!"
                zero_vect = np.zeros_like(x_data[0, :])
                zero_vect[:] = 1e-15
                xi_sum[:] = zero_vect[:]

            # Calculating denominator ||Σ xi||(A)
            parametrized_norm_xi = sqrt(self.vdot(self.dot1d_ds(xi_sum, self.A), xi_sum))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            mu_arr[i, :] = xi_sum / parametrized_norm_xi

        return mu_arr

    cdef NormPart(self, double [:, ::1] x_data_subset):
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

    cdef JObjCosA(self, cnp.intp_t x_idx, double [:, ::1] x_data,
                  double [::1] mu, cnp.intp_t [::1] clstr_tags_arr):
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

        # Setting and initalizing local variables.
        cdef double dist = 0.0
        cdef double ml_cost = 0.0
        cdef double cl_cost = 0.0
        cdef cnp.intp_t i
        cdef double sum1 = 0.0
        cdef double sum2 = 0.0
        cdef double params_pdf = 0.0
        cdef double norm_part_value = 0.0

        cdef double tmp = 0.0

        # Calculating the cosine distance of the specific x_i from the cluster's centroid.
        # --------------------------------------------------------------------------------
        tmp = self.vdot(self.dot1d_ds(mu, self.A), x_data[x_idx, :])

        if tmp > 1:
            print "OVER ONE", tmp
        if tmp == 1:
            print "ONE", tmp

        dist = 1.0 - tmp

        # Calculating Must-Link violation cost.
        # -------------------------------------
        for i in range(self.ml_sorted_size):

            if x_idx < self.ml_sorted[0, i]:
                break

            if x_idx == self.ml_sorted[0, i] and\
                clstr_tags_arr[x_idx] != clstr_tags_arr[self.ml_sorted[1, i]]:
                ml_cost += self.ml_wg * (
                    1.0 - self.vdot(
                        self.dot1d_ds(x_data[x_idx, :], self.A),
                        x_data[self.ml_sorted[1, i], :]
                    )
                )

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        for i in range(self.cl_sorted_size):

            if x_idx < self.cl_sorted[0, i]:
                break

            if x_idx == self.cl_sorted[0, i] and\
                clstr_tags_arr[x_idx] == clstr_tags_arr[self.cl_sorted[1, i]]:

                cl_cost += self.cl_wg * (
                    self.vdot(
                        self.dot1d_ds(x_data[x_idx, :], self.A),
                        x_data[self.cl_sorted[1, i], :]
                    )
                )

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        for i in range(self.A_size):
            sum1 += log(self.A[i])
            sum2 += pow(self.A[i], 2.0) / (2 * pow(self.ray_sigma, 2.0))
        params_pdf = sum1 - sum2 - (2 * self.A_size * log(self.ray_sigma))
        # NOTE!
        # params_pdf = 0.0

        # ######## NEEDS to be changed in Cython
        # Calculating the log normalization function of the von Mises-Fisher distribution...
        # ...NOTE: Only for this cluster i.e. this vMF of the whole PDF mixture.
        # if self.norm_part:
        #     norm_part_value = self.NormPart(x_data[clstr_idx_arr])
        # else:
        # norm_part_value = 0.0

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost - params_pdf + norm_part_value

    cdef GlobJObjCosA(self, double [:, ::1] x_data, double [:, ::1] mu_arr,
                      cnp.intp_t [::1] clstr_tags_arr):
        """
        """
        print "In GlobalJObjCosA..."

        # Calculating the distance of all vectors, the must-link and cannot-link violations scores.
        cdef double sum_d = 0.0
        cdef double ml_cost = 0.0
        cdef double cl_cost = 0.0
        cdef double norm_part_value = 0.0
        cdef double params_pdf = 0.0
        cdef double smlps_cnt = 0.0
        cdef double ml_cnt = 0.0
        cdef double cl_cnt = 0.0
        cdef cnp.intp_t k, i, a_idx
        cdef double sum1 = 0.0
        cdef double sum2 = 0.0

        for k in range(self.k_clusters):

            # Calculating the cosine distances and add the to the total sum of distances.
            # ---------------------------------------------------------------------------
            for i in range(self.xdata_size):

                if clstr_tags_arr[i] == k:

                    sum_d += 1.0 - self.vdot(self.dot1d_ds(mu_arr[k, :], self.A), x_data[i, :])

                    smlps_cnt += 1.0

            # Calculating Must-Link violation cost.
            # -------------------------------------
            for i in range(self.ml_size):

                if clstr_tags_arr[self.must_lnk[0, i]] != clstr_tags_arr[self.must_lnk[1, i]] and\
                    (clstr_tags_arr[self.must_lnk[0, i]] == k or\
                                clstr_tags_arr[self.must_lnk[1, i]] == k):

                    ml_cost += self.ml_wg * (
                        1.0 - self.vdot(
                            self.dot1d_ds(x_data[self.must_lnk[0, i], :], self.A),
                            x_data[self.must_lnk[1, i], :]
                        )
                    )

                    ml_cnt += 1.0

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------
            for i in range(self.cl_size):

                if clstr_tags_arr[<int>self.cannot_lnk[0, i]] == k and\
                    clstr_tags_arr[self.cannot_lnk[0, i]] == clstr_tags_arr[self.cannot_lnk[1, i]]:

                    cl_cost += self.cl_wg * (
                        self.vdot(
                            self.dot1d_ds(x_data[self.cannot_lnk[0, i], :], self.A),
                            x_data[<int>self.cannot_lnk[1, i], :]
                        )
                    )

                    cl_cnt += 1.0

        # Averaging EVERYTHING.
        sum_d = sum_d / smlps_cnt

        if ml_cnt:
            ml_cost = ml_cost / ml_cnt

        if cl_cnt:
            cl_cost = cl_cost / cl_cnt

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        if self.globj_norm:
            for a_idx in self.A_size:
                sum1 += log(self.A[a_idx])
                sum2 += pow(self.A[a_idx], 2.0) / (2 * pow(self.ray_sigma, 2.0))
            params_pdf = sum1 - sum2 - (2 * self.A_size * log(self.ray_sigma))
        else:
            params_pdf = 0.0

        # Calculating the log normalization function of the von Mises-Fisher distribution...
        # ...of the whole mixture.
        # if self.norm_part and self.globj_norm:
        #     norm_part_value = 0.0
        #     for i in enumerate(mu_arr.shape[0]):
        #         clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]
        #         norm_part_value += self.NormPart(x_data[clstr_idxs_arr])
        # else:
        norm_part_value = 0.0

        print 'dims', x_data.shape[1]
        print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost
        print 'np.log(Rayleigh)', params_pdf
        print 'N*(np.log(cdk) + np.log(k))', norm_part_value

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + ml_cost + cl_cost - params_pdf + norm_part_value

    cdef UpdateDistorParams(self, double [::1] A,
                            double [:, ::1] x_data, mu_arr, cnp.intp_t [::1] clstr_tags_arr):
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

        # Initializing...
        cdef double xm_pdv = 0.0
        cdef double ml_pdv = 0.0
        cdef double cl_pdv = 0.0
        cdef double smpls_cnt = 0.0
        cdef double ml_cnt = 0.0
        cdef double cl_cnt = 0.0
        cdef cnp.intp_t a_idx, k
        cdef double a_pdv = 0.0
        cdef double [::1] new_A = np.zeros((self.A_size), dtype=np.float)

        for a_idx in range(self.A_size):

            for k in range(self.k_clusters):

                # Calculating the partial derivatives of each parameter for all cluster's member...
                # ...for each cluster.
                # ---------------------------------------------------------------------------------
                for i in range(self.xdata_size):

                    if clstr_tags_arr[i] == k:

                        xm_pdv += self.PartialDerivative(a_idx, x_data[i, :], mu_arr[k, :], A)

                # Calculating Must-Link violation cost.
                # -------------------------------------
                for i in range(self.ml_size):

                    if clstr_tags_arr[self.must_lnk[0, i]] != clstr_tags_arr[self.must_lnk[1, i]] and\
                        (clstr_tags_arr[self.must_lnk[0, i]] == k or\
                                    clstr_tags_arr[self.must_lnk[1, i]] == k):

                        ml_pdv += self.ml_wg*self.PartialDerivative(
                            a_idx,
                            x_data[self.must_lnk[0, i], :],
                            x_data[self.must_lnk[1, i], :],
                            A
                        )

                # Calculating Cannot-Link violation cost.
                # ---------------------------------------
                for i in range(self.cl_size):

                    if clstr_tags_arr[self.cannot_lnk[0, i]] == k and\
                     clstr_tags_arr[self.cannot_lnk[0, i]] == clstr_tags_arr[self.cannot_lnk[1, i]]:

                        cl_pdv -= self.cl_wg*self.PartialDerivative(
                            a_idx,
                            x_data[self.cannot_lnk[0, i], :],
                            x_data[self.cannot_lnk[1, i], :],
                            A
                        )

            # Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
            # Should be on the new_a or on the last iteration A[a_idx] ???
            # new_a = self.A[a_idx] + (self.lrn_rate * (xm_pdv + ml_pdv + cl_pdv))
            # Now it is applied on the previews Α[a_idx]!
            a_pdv = (1 / self.A[a_idx]) - (self.A[a_idx] / pow(self.ray_sigma, 2.0))
            # NOTE!
            # a_pdv = 0.0

            # print 'Rayleigh Partial', a_pderiv

            # if np.abs(a_pderiv) == np.inf:
            #     print "Invalid patch for Rayleighs P'(A) triggered: (+/-)INF P'(A)=", a_pderiv
            #     a_pderiv = 1e-15

            # elif a_pderiv == np.nan:
            #     print "Invalid patch for Rayleighs P(A) triggered: NaN P'(A)=", a_pderiv
            #     a_pderiv = 1e-15

            # Changing a diagonal value of the A cosine similarity parameters measure.
            #print A[a_idx]
            #print A[a_idx] + (self.lrn_rate * (xm_pdv + ml_pdv + cl_pdv - a_pdv))

            new_A[a_idx] = A[a_idx] + (self.lrn_rate * (xm_pdv + ml_pdv + cl_pdv - a_pdv))

            if new_A[a_idx] < 0.0:
                print self.lrn_rate
                print xm_pdv
                print ml_pdv
                print cl_pdv
                print a_pdv
                0/0

            # ΝΟΤΕ: Invalid patch for let the experiments to be completed.
            """
            if new_A[a_idx] < 0.0:
                print "Invalid patch for A triggered: (-) Negative A=", new_A[a_idx], a_pderiv
                new_A[a_idx] = 1e-15

            elif new_A[a_idx] == 0.0:
                print "Invalid patch for A triggered: (0) Zero A=", new_A[a_idx], a_pderiv
                new_A[a_idx] = 1e-15

            elif np.abs(new_A[a_idx]) == np.Inf:
                print "Invalid patch for A triggered: (+/-)INF A=", new_A[a_idx], a_pderiv
                new_A[a_idx] = 1e-15

            elif new_A[a_idx] == np.NaN:
                print "Invalid patch for A triggered: NaN A=", new_A[a_idx], a_pderiv
                new_A[a_idx] = 1e-15
            """

        A[:] = new_A[:]

        # Returning the A parameters. This is actually a dump return for coding constance reasons.
        return A

    cdef inline double PartialDerivative(self, cnp.intp_t a_idx,
                                         double [::1] x1, double [::1] x2, double [::1] A):
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
        cdef double res_a = 0.0

        # Calculating parametrized Norms ||Σ xi||(A)
        x1_pnorm = sqrt(self.vdot(self.dot1d_ds(x1, A), x1))
        x2_pnorm = sqrt(self.vdot(self.dot1d_ds(x2, A), x2))

        res_a = (
                    (x1[a_idx] * x2[a_idx] * x1_pnorm * x2_pnorm) -
                    (
                        self.vdot(self.dot1d_ds(x1, A), x2) *
                        (
                            (
                                pow(x1[a_idx], 2.0) * pow(x2_pnorm, 2.0) +
                                pow(x2[a_idx], 2.0) * pow(x1_pnorm, 2.0)
                            ) / (2 * x1_pnorm * x2_pnorm)
                        )
                    )
                ) / (pow(x1_pnorm, 2.0) * pow(x2_pnorm, 2.0))

        return res_a

    cdef inline double [:, ::1] dot2d(self, double [:, ::1] m1, double [:, ::1] m2):

        if m1.shape[1] != m2.shape[0]:
            raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

        # Matrix index variables.
        cdef unsigned int i, j, k
        cdef unsigned int I = m1.shape[0]
        cdef unsigned int J = m2.shape[1]
        cdef unsigned int K = m1.shape[1]

        # Creating the numpy.array for results and its memory view
        cdef double [:, ::1] res = np.zeros((I, J), dtype=np.float)

        # Calculating the dot product.
        with nogil:
            for i in range(I):
                for j in range(J):
                    for k in range(K):
                        res[i, j] += m1[i, k] * m2[k, j]

        return res

    cdef inline double vdot(self, double [::1] v1, double [::1] v2):

        if v1.shape[0] != v2.shape[0]:
            raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

        # Matrix index variables.
        cdef unsigned int i
        cdef unsigned int I = v1.shape[0]

        # Initializing the result variable.
        cdef double res = <double>0.0

        # Calculating the dot product.
        with nogil:
            for i in range(I):
                res += v1[i] * v2[i]

        return res

    cdef inline double [:, ::1] dot2d_ds(self, double [:, ::1] m1, double [::1] m2):

        if m1.shape[1] != m2.shape[0]:
            raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

        # Matrix index variables.
        cdef unsigned int i, j
        cdef unsigned int I = m1.shape[0]
        cdef unsigned int J = m2.shape[0]

        # Creating the numpy.array for results and its memory view
        cdef double [:, ::1] res = np.zeros((I, J), dtype=np.float)

        # Calculating the dot product.
        with nogil:
            for i in range(I):
                for j in range(J):
                    res[i, j] = m1[i, j] * m2[j]

        return res

    cdef inline double [::1] dot1d_ds(self, double [::1] v, double [::1] m):

        if v.shape[0] != m.shape[0]:
            raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

        # Matrix index variables.
        cdef unsigned int i
        cdef unsigned int I = v.shape[0]

        # Creating the numpy.array for results and its memory view
        cdef double [::1] res = np.zeros((I), dtype=np.float)

        # Calculating the dot product.
        with nogil:
            for i in range(I):
                res[i] = v[i] * m[i]

        return res
