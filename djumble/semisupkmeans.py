# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import scipy.special as special
import warnings

from .voperators import cy as vop


class HMRFKmeansSemiSup(object):
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

    def __init__(self, k_clusters, must_lnk_con, cannot_lnk_con, init_centroids=None,
                 ml_wg=1.0, cl_wg=1.0, max_iter=300, cvg=0.001, lrn_rate=0.0003, ray_sigma=0.5,
                 d_params=None, icm_max_i=10, enable_norm=False):

        if ray_sigma < 1.0 and enable_norm is True:
            raise Exception(
                "Ray's Sigma cannot be less than 1 for this algorithm for being stable."
            )

        if enable_norm is False:
            wrn_msg = "Ray's Sigma is ignored when normalization factor are desabled, " +\
                "i.e. enable_norm=False which is the default value for this argument."
            warnings.warn(wrn_msg)

        self.enable_norm = enable_norm
        self.ray_sigma = ray_sigma
        self.k_clusters = k_clusters
        self.ml_pair_idxs = must_lnk_con
        self.cl_pair_idxs = cannot_lnk_con
        self.init_centroids = init_centroids
        self.ml_wg = ml_wg
        self.cl_wg = cl_wg
        self.max_iter = max_iter
        self.cvg = cvg
        self.lrn_rate = lrn_rate
        self.dv = d_params
        self.icm_max_i = icm_max_i

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
                self.dv.data: The values of the (hyper-)parametes for the cosine distance after the
                    final model fit.

        """

        # Initializing clustering

        # Setting up distortion parameters if not have been passed as class argument.
        if self.dv is None:
            self.dv = np.ones((x_data.shape[1]), dtype=np.float)
        # A should be a diagonal matrix form for the calculations in the functions bellow. The...
        # ...sparse form will save space and the csr_matrix will make the dia_matrix write-able.
        # self.dv = self.dv
        # self.dv = sp.sparse.lil_matrix(self.dv)

        # Setting up the violation weights matrix if not have been passed as class argument.
        # if self.w_violations is None:
        #   self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        #   # ### I am not sure what kind of values this weights should actually have.

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint()
        # init_clstr_sets_lst = ConsolidateAL()
        clstr_tags_arr = np.empty(x_data.shape[0], dtype=np.int)
        clstr_tags_arr[:] = 999999

        best_clstr_tags_arr = np.zeros_like(clstr_tags_arr)

        # Selecting a random set of centroids if not any given as class initialization argument.

        # Pick k random vector from the x_data set as initial centroids. Where k is equals...
        # ...the number of self.k_clusters.
        if self.init_centroids is None:
            self.init_centroids = np.random.randint(0, x_data.shape[0], size=self.k_clusters)
        else:
            # Setting one index of the x_data set per expected cluster as the initialization...
            # ...centroid for this cluster.
            for i, idx in enumerate(self.init_centroids):
                clstr_tags_arr[idx] = i

        print self.init_centroids

        # ########
        # # The above might actually change based on the initialization set-up will be used.
        # ########

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_arr = vop.mean_cosA(x_data, clstr_tags_arr, self.dv, self.k_clusters)
        best_mu_arr = np.zeros_like(mu_arr)

        # EM algorithm execution.

        # This values is the last global objective. Thus it should be the highest possible...
        # ...number initially, i.e. Inf.
        last_gobj = np.Inf

        # While no convergence yet repeat for at least i times.
        for c_step in range(self.max_iter):

            print
            print c_step

            # Storing the amount of iterations until convergence.
            self.conv_step = c_step

            # ########### #
            # The E-Step. #
            # ########### #

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            clstr_tags_arr = self.ICM(x_data, mu_arr, clstr_tags_arr)

            # ########### #
            # The M-Step. #
            # ########### #

            # Recalculating centroids upon the new clusters set-up.
            mu_arr = vop.mean_cosA(x_data, clstr_tags_arr, self.dv, self.k_clusters)

            # Re-estimating distortion measure parameters upon the new clusters set-up.
            self.dv = self.UpdateDistorParams(self.dv, x_data, mu_arr, clstr_tags_arr)

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_arr, clstr_tags_arr)

            # Terminating upon the difference of the last two Global JObej values.
            if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:
                # second condition is TEMP!
                print 'last_gobj - glob_jobj', np.abs(last_gobj - glob_jobj)
                print "Global Objective", glob_jobj

                # When finding a better Clusters set-up than the last best set-up then stopping...
                # ...the EM and returning the current results. Here the EM is stopping at a...
                # ...globla minima given that the cvg precision is relativelly small.
                return mu_arr, clstr_tags_arr

            else:
                if glob_jobj < last_gobj:
                    last_gobj = glob_jobj
                    best_clstr_tags_arr[:] = clstr_tags_arr[:]
                    best_mu_arr[:, :] = mu_arr[:, :]

            print "Global: jObj-val = ", glob_jobj, ", last jObj-val = ", last_gobj

        # Returning the Centroids and the Clusters best set-up found in one of the EM iterations...
        # ..., however, the EM here stops at a local minima not the global.
        return best_mu_arr, best_clstr_tags_arr

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'ICM_Max_Iter': self.icm_max_i,
            'final_iter': self.conv_step,
            'ml_wg': self.ml_wg,
            'cl_wg': self.cl_wg,
            'convg_diff': self.cvg,
            'lrn_rate': self.lrn_rate,
            'ray_sigma': self.ray_sigma,
            'dist_msur_params': self.dv,
            'enable_norm': self.enable_norm,
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
        # while no_change_cnt < 2:
        for dump in range(self.icm_max_i):

            # Stopping the rearrangment of the clusters when at least 2 times nothing changed...
            # ...thus, most-likelly the arrangmet is optimal.
            if no_change_cnt == 2:
                break

            # Calculating the new Clusters.
            for x_idx in np.random.permutation(x_data.shape[0]):

                """
                # Looking for skipping indices that should not participate in clustering.
                for i in range(self.neg_i4c_size):
                    if self.neg_idxs4clstring[i] == x_idx:
                        skip_smpl = True

                # Skipping the indices should not participate in clustering.
                if not skip_smpl:
                """

                # Setting the initial value for the previews J-Objective value.
                last_jobj = np.Inf

                # Calculating the J-Objective for every x_i vector of the x_data set.
                for mu_i in np.arange(mu_arr.shape[0]):

                    # Getting the indeces for this cluster.
                    clstr_idxs_arr = np.where(clstr_tags_arr == mu_i)[0]

                    # Calculating the J-Objective.
                    j_obj = self.JObjCosA(x_idx, x_data, mu_i, mu_arr, clstr_idxs_arr)

                    if j_obj < last_jobj:
                        last_jobj = j_obj
                        new_clstr_tag = mu_i

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

    def JObjCosA(self, x_idx, x_data, mu_i, mu_arr, clstr_idx_arr):
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
        dist = vop.cosDa_vect(mu_arr[mu_i], x_data[x_idx], self.dv)

        # print np.array(mu_arr), x_data[x_idx]

        # Calculating Must-Link violation cost.
        # -------------------------------------
        ml_cost = 0.0

        # Selecting ml_pairs containing the x_idx.
        mlvpairs_wth_xidx = self.ml_pair_idxs[np.where((self.ml_pair_idxs == x_idx))[0]]

        # Selecting must-link vionlation where the x_idx is included
        if x_idx not in clstr_idx_arr:
            ml_voil_tests = np.isin(mlvpairs_wth_xidx, np.hstack((x_idx, clstr_idx_arr)))
        else:
            ml_voil_tests = np.isin(mlvpairs_wth_xidx, clstr_idx_arr)

        # Getting the must-link-constraints violations.
        mlv_pair_rows = np.where(
            (np.logical_xor(ml_voil_tests[:, 0], ml_voil_tests[:, 1]) == True)
        )[0]

        mlv_cnts = np.size(mlv_pair_rows)

        if mlv_cnts:

            # Calculating all pairs of violation costs for must-link constraints.
            # NOTE: The violation cost is equivalent to the parametrized Cosine distance...
            # ...which here is equivalent to the (1 - dot product) because the data points...
            # ...assumed to be normalized by the parametrized Norm of the vectors.
            viol_costs = vop.cosDa_rpairs(
                x_data, self.dv, self.ml_pair_idxs, mlv_pair_rows
            )

            # Sum-ing up Weighted violations costs.
            ml_cost = self.ml_wg * np.sum(viol_costs)  # / float(mlv_cnts)

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        cl_cost = 0.0

        # Selecting cl_pairs containing the x_idx.
        clvpairs_wth_xidx = self.cl_pair_idxs[np.where((self.cl_pair_idxs == x_idx))[0]]

        # Getting the index(s) of the cannot-link-constraints index-table of this data sample.
        if x_idx not in clstr_idx_arr:
            cl_voil_tests = np.isin(clvpairs_wth_xidx, np.hstack((x_idx, clstr_idx_arr)))
        else:
            cl_voil_tests = np.isin(clvpairs_wth_xidx, clstr_idx_arr)

        clv_pair_rows = np.where(
            (np.logical_and(cl_voil_tests[:, 0], cl_voil_tests[:, 1]) == True)
        )[0]

        clv_cnts = np.size(clv_pair_rows)

        if clv_cnts:

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance minus the...
            # ...parametrized Cosine distance of the vectors. Since MaxCosine is 1 then...
            # ...maxCosineDistance - CosineDistance == CosineSimilarty of the vectors....
            # ...Again the data points assumed to be normalized.
            viol_costs = 1.0 - np.array(
                vop.cosDa_rpairs(
                    x_data, self.dv, self.cl_pair_idxs, clv_pair_rows
                )
            )

            cl_cost = self.cl_wg * np.sum(viol_costs)  # / float(clv_cnts)

        if self.enable_norm:
            # Calculating the cosine distance parameters PDF. In fact the log-form...
            # ...of Rayleigh's PDF.
            sum1, sum2 = 0.0, 0.0
            for a in self.dv:
                sum1 += np.log(a)
                sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
            params_pdf = sum1 - sum2 -\
                (2 * self.dv.shape[0] * np.log(self.ray_sigma))

            # Calculating the log normalization function of the von Mises-Fisher distribution...
            # ...NOTE: Only for this cluster i.e. this vMF of the whole PDF mixture.
            sum1, sum2 = 0.0, 0.0
            for a in self.dv:
                sum1 += np.log(a)
                sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
            params_pdf = sum1 - sum2 - (2 * self.dv.shape[0] * np.log(self.ray_sigma))

            # Calculating the log normalization function of the von Mises-Fisher distribution...
            # ...of the whole mixture.
            if mu_arr.shape[1] == 1:
                norm_part_value = 0.0
            elif mu_arr.shape[1] >= 100:
                norm_part_value = self.ray_sigma * mu_arr.shape[1]
            else:
                norm_part_value = 0.0
                norm_part_value = self.NormPart(x_data[clstr_idx_arr])

            # if np.size(clv_pair_rows):
            # sum0 = dist + ml_cost + cl_cost + params_pdf + norm_part_value
            # if np.min(self.dv) < 0 or np.max(self.dv) > 1.0 or sum0 == np.NaN:
            #     print self.dv
            # print np.array(dist), ml_cost, cl_cost, params_pdf, norm_part_value, sum0

            return dist + ml_cost + cl_cost + params_pdf + norm_part_value

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost

    def GlobJObjCosA(self, x_data, mu_arr, clstr_tags_arr):
        """
        """

        print "In GlobalJObjCosA..."

        # Calculating the distance of all vectors, the must-link and cannot-link violations scores.
        sum_d, ml_cost, cl_cost, norm_part_value, params_pdf = 0.0, 0.0, 0.0, 0.0, 0.0
        # ml_cnt, cl_cnt = 0.0, 0.0

        # xd_rows = np.arange(x_data.shape[0])
        # mu_rows = np.arange(mu_arr.shape[0])

        # Calculating the cosine distances and add the to the total sum of distances.
        # ---------------------------------------------------------------------------
        sum_d = np.sum(vop.cosDa(mu_arr, x_data, self.dv))

        # Calculating Must-Link violation cost.
        # -------------------------------------
        for i in range(mu_arr.shape[0]):

            # Getting the indeces for the i cluster.
            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.
            ml_voil_tests = np.isin(self.ml_pair_idxs, clstr_idxs_arr)
            mlv_pair_rows = np.where(
                (np.logical_xor(ml_voil_tests[:, 0], ml_voil_tests[:, 1]) == True)
            )[0]

            ml_cnt = np.size(mlv_pair_rows)

            if ml_cnt:

                # Calculating all pairs of violation costs for must-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance.
                viol_costs = np.sum(
                    vop.cosDa_rpairs(
                        x_data, self.dv, self.ml_pair_idxs, mlv_pair_rows
                    )
                )

                ml_cost += viol_costs

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            cl_voil_tests = np.isin(self.cl_pair_idxs, clstr_idxs_arr)
            clv_pair_rows = np.where(
                (np.logical_and(cl_voil_tests[:, 0], cl_voil_tests[:, 1]) == True)
            )[0]

            cl_cnt = np.size(clv_pair_rows)

            if cl_cnt:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance
                viol_costs = np.array(
                    vop.cosDa_rpairs(
                        x_data, self.dv, self.cl_pair_idxs, clv_pair_rows
                    )
                )

                max_vcost = 1.0  # np.max(viol_costs)
                cl_cost += np.sum(max_vcost - viol_costs)

        if self.enable_norm:
            # Calculating the cosine distance parameters PDF. In fact the log-form...
            # ...of Rayleigh's PDF.
            sum1, sum2 = 0.0, 0.0
            for a in self.dv:
                sum1 += np.log(a)
                sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
            params_pdf = sum1 - sum2 - (2 * self.dv.shape[0] * np.log(self.ray_sigma))

            # Calculating the log normalization function of the von Mises-Fisher distribution...
            # ...of the whole mixture.
            if mu_arr.shape[1] == 1:

                wrn_msg = "Data points cannot have less than two(2) dimension. Thus" +\
                    " normalization partision function (Bezzel) will become Zero (0)"
                warnings.warn(wrn_msg)

                norm_part_value = 0.0

            elif mu_arr.shape[1] >= 100:

                wrn_msg = "Dimentions are very high (i.e. more than 100) high, thus Bessel" +\
                    " function equals Zero (0). A heuristic will be used that is Data matrix" +\
                    " dimensions multiplication"
                warnings.warn(wrn_msg)

                norm_part_value = self.ray_sigma * mu_arr.shape[1]
            else:

                norm_part_value = 0.0

                for i in np.arange(mu_arr.shape[0]):
                    clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]
                    norm_part_value += self.NormPart(x_data[clstr_idxs_arr])

            print 'dims', x_data.shape[1]
            print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
            print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost
            print 'np.log(Rayleigh)', params_pdf
            print 'N*(np.log(cdk) + np.log(k))', norm_part_value

            return sum_d + (self.ml_wg * ml_cost) + (self.cl_wg * cl_cost) +\
                params_pdf + norm_part_value

        else:

            # Averaging all-total distance costs.
            sum_d = sum_d / (x_data.shape[0] * mu_arr.shape[0])
            if ml_cnt:
                ml_cost = ml_cost / np.float(ml_cnt)
            if cl_cnt:
                cl_cost = cl_cost / np.flaot(cl_cnt)

        print 'dims', x_data.shape[1]
        print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + (self.ml_wg * ml_cost) + (self.cl_wg * cl_cost)

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

        # Initializing...
        new_A = np.zeros_like(A, dtype=np.float)
        # xm_pderiv, mlcost_pderiv, clcost_pderiv = 0.0, 0.0, 0.0
        # smpls_cnt, ml_cnt, cl_cnt = 0.0, 0.0, 0.0
        mlcost_pderiv = np.zeros_like(A.data, dtype=np.float)
        clcost_pderiv = np.zeros_like(A.data, dtype=np.float)

        ml_viol_pairs = list()
        cl_viol_pairs = list()
        clstr_idxs_arrz_lst = list()

        # Collecting Violation Pairs.
        for i in range(mu_arr.shape[0]):

            # Getting the indeces for the i cluster.
            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]
            clstr_idxs_arrz_lst.append(clstr_idxs_arr)
            # smpls_cnt += float(clstr_idxs_arr.shape[0])

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.
            ml_voil_tests = np.isin(self.ml_pair_idxs, clstr_idxs_arr)
            mlv_pair_rows = np.where(
                (np.logical_or(ml_voil_tests[:, 0], ml_voil_tests[:, 1]) == False)
            )[0]

            ml_viol_pairs.append(mlv_pair_rows)

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            cl_voil_tests = np.isin(self.cl_pair_idxs, clstr_idxs_arr)
            clv_pair_rows = np.where(
                (np.logical_and(cl_voil_tests[:, 0], cl_voil_tests[:, 1]) == True)
            )[0]

            cl_viol_pairs.append(clv_pair_rows)

        # Calculating the partial derivatives of each parameter for all cluster's member...
        # ...for each cluster.
        # ---------------------------------------------------------------------------------
        xm_pderiv = vop. pDerivative_seq_mk2mr(
            A, mu_arr, x_data, mu_arr.shape[0], clstr_tags_arr
        )

        # Calculating Must-Link violation cost.
        # -------------------------------------

        ml_viol_pairs = np.hstack(ml_viol_pairs)

        if np.size(ml_viol_pairs):

            # Calculating the partial derivatives of all pairs of violations for...
            # ...must-link constraints.
            mlcost_pderiv = vop.pDerivative_seq_rpairs(
                A, x_data, self.ml_pair_idxs, ml_viol_pairs
            )

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------

        cl_viol_pairs = np.hstack(cl_viol_pairs)

        if np.size(cl_viol_pairs):

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance
            clcost_pderiv = vop.pDerivative_seq_rpairs(
                A, x_data, self.cl_pair_idxs, cl_viol_pairs
            )

        # Averaging EVERYTHING
        # xm_pderiv = xm_pderiv / smpls_cnt
        # if ml_cnt:
        #     mlcost_pderiv = mlcost_pderiv / ml_cnt
        # if cl_cnt:
        #     clcost_pderiv = clcost_pderiv / cl_cnt

        # Getting the Max CL cost distance partial derivative.
        max_clc_pder = np.max(clcost_pderiv)

        # Updating Process.
        for i, a in enumerate(A):

            if self.enable_norm:
                # Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
                a_pderiv = (1 / a) - (a / np.square(self.ray_sigma))
            else:
                a_pderiv = 0.0

            # Changing a diagonal value of the A cosine similarity parameters measure.
            new_A[i] = a +\
                (
                    self.lrn_rate *
                    (
                        xm_pderiv[i]
                        + (self.ml_wg * mlcost_pderiv[i])
                        + (self.cl_wg * (max_clc_pder - clcost_pderiv[i]))
                        - a_pderiv
                    )
                )

            # print self.lrn_rate, xm_pderiv[i], mlcost_pderiv[i]
            # print (max_clc_pder - clcost_pderiv[i]), a_pderiv

        # NOTE: Fale-safe operation when a value of distortion paramters vector becames negative.
        # Chaning the A distortion parameters vector to a vector of 0.9.
        if np.min(new_A) < 0.0:

            wrn_msg = "Negative value found in the distortion paramters vector. " +\
                "A vector is replacing it automatically with 0.9 values."
            warnings.warn(wrn_msg)

            A = np.zeros_like(A)
            A[:] = 0.9

        elif np.max(new_A) > 1.0 and self.enable_norm is True:

            wrn_msg = "Over 1.0 value found in the distortion paramters vector. " +\
                "A vector is replacing it automatically with 0.5 values."
            warnings.warn(wrn_msg)

            A = np.zeros_like(A)
            A[:] = 0.5

        else:

            A[:] = new_A[:]

        # Returning the A parameters.
        return A

    def NormPart(self, Xsub):
        """
         The von Mises and von Mises - Fisher Logarithmic Normalization partition function:...
            is calculated in this method. For the 2D data the function is simplified for faster
            calculation.

            *** This function has been build after great research on the subject. However, some
            things are not very clear this is always in a revision state until theoretically proven
            to be correctly used.

            Arguments
            ---------
                Xsub: The subset of the data point are included in the von Mises-Fisher
                    distribution.

            Output
            ------
                The logarithmic value of the partition normalization function.

        """

        # Calculating the r.
        # The r it suppose to be the norm of the data points of the current cluster, not the...
        # ...whole mixture as it is in the global objective function. ## Might this need to be...
        # ...revised.
        r = np.linalg.norm(Xsub)

        # Calculating the Von Misses Fisher's k concentration is approximated as seen..
        # ....in Banerjee et al. 2003.
        dim = Xsub.shape[1]

        # Calculating the partition function depending on the vector dimensions.
        k = (r * dim - np.power(r, 3.0)) / (1 - np.power(r, 2.0))

        # k=0.001 only for the case where the r is too small. Usually at the 1-2 first iterations...
        # ...of the EM/Kmeans.
        if k < 0.0:
            k = 0.001

        if dim > 3:

            # This is the proper way for calculating the von Misses Fisher normalization factor.
            bessel = np.abs(special.jv((dim / 2.0) - 1.0, k))
            # bessel = np.abs(self.Jd((dim/2.0)-1.0, k))
            cdk = np.power(k, (dim / 2.0) - 1) / (np.power(2 * np.pi, dim / 2) * bessel)

        else:

            # This is the proper way for calculating the vMF normalization factor for 2D vectors.
            bessel = np.abs(special.jv(0, k))
            # bessel = np.abs(self.Jd(0, k))
            cdk = 1.0 / (2 * np.pi * bessel)

        # Returning the log of the normalization function plus the log of the k that is used in...
        # ...the von Mises Fisher PDF, which is separated from the Cosine Distance due to the log.
        # The normalizers are multiplied by the number of all the X data subset, because it is...
        # the global normalizer after the whole summations has been completed.
        # NOTE: Still this need to be revised.
        return np.log(cdk) + np.log(k)  # * Xsub.shape[0]


class StochSemisupEM(object):
    """ Stochastic Semi-supervised EM: A Semi-supervised clustering algorithm based
        on stochastic distortion parameters calculation for Clustering model optimized by
        Expectation Maximization (EM) algorithm with Hard clustering constraints,
        i.e. a Expectation Mazimization Semi-supervised clustering variant.

        More details later...

    """

    def __init__(self, k_clusters, must_lnk_con, cannot_lnk_con, init_centroids=None,
                 ml_wg=1.0, cl_wg=1.0, max_iter=300, cvg=0.001, icm_max_i=1000,
                 min_efl=0.5, max_efl=1.5, step_efl=0.1, eft_per_i=100):

        self.k_clusters = k_clusters
        self.ml_pair_idxs = must_lnk_con
        self.cl_pair_idxs = cannot_lnk_con
        self.init_centroids = init_centroids
        self.ml_wg = ml_wg
        self.cl_wg = cl_wg
        self.max_iter = max_iter
        self.cvg = cvg
        self.icm_max_i = icm_max_i
        self.eft_per_i = eft_per_i

        # It is keeping the exponential fuction L paramter value which in selected...
        # ...randomly in every EM interation.
        self.efl_vals = np.arange(min_efl, max_efl, step_efl)
        self.efl_usd_vals = [min_efl]

    def fit(self, x_data):
        """ Fit method: more detail later...

        """

        # Initializing clustering

        # Creating distortion parameters stochastic set for the initial ICM run.
        dvz = np.ones((self.eft_per_i, x_data.shape[1]), dtype=np.float)
        for i in np.arange(self.eft_per_i - 1):
            dvz[i + 1, :] = np.random.exponential(1 / self.efl_usd_vals[0], size=x_data.shape[1])

        # Initialising the best stochastic distortion vector with the first of all eft_per_i cases.
        best_dv = np.zeros((x_data.shape[1]), dtype=np.float)
        best_dv[:] = dvz[0, :]
        best_efl = self.efl_usd_vals[0]
        efl = best_efl

        clstr_tags_arr = np.empty(x_data.shape[0], dtype=np.int)
        clstr_tags_arr[:] = 999999

        best_clstr_tags_arr = np.zeros_like(clstr_tags_arr)

        # Selecting a random set of centroids if not any given as class initialization argument.

        # Pick k random vector from the x_data set as initial centroids. Where k is equals...
        # ...the number of self.k_clusters.
        if self.init_centroids is None:
            self.init_centroids = np.random.randint(0, x_data.shape[0], size=self.k_clusters)
        else:
            # Setting one index of the x_data set per expected cluster as the initialization...
            # ...centroid for this cluster.
            for i, idx in enumerate(self.init_centroids):
                clstr_tags_arr[idx] = i

        print self.init_centroids

        # ########
        # The above might actually change based on the initialization set-up will be used.
        # ########

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_arr = vop.mean_cosA(x_data, clstr_tags_arr, best_dv, self.k_clusters)
        best_mu_arr = np.zeros_like(mu_arr)

        # EM algorithm execution.

        # This values is the last global objective. Thus it should be the highest possible...
        # ...number initially, i.e. Inf.
        last_gobj = np.Inf

        # While no convergence yet repeat for at least i times.
        for c_step in range(self.max_iter):

            print
            print c_step

            # Storing the amount of iterations until convergence.
            self.conv_step = c_step

            # ########### #
            # The E-Step. #
            # ########### #

            # Assigning every data-set point to the proper cluster upon distortion parameters...
            # ...and centroids for the current iteration.
            ctags_per_dv = self.ICM(dvz, x_data, mu_arr, clstr_tags_arr)

            # ########### #
            # The M-Step. #
            # ########### #
            for set_dv, ctags_set in zip(dvz, ctags_per_dv):

                self.dv = set_dv

                # Recalculating centroids upon the new clusters set-up.
                mu_arr_dv = vop.mean_cosA(x_data, ctags_set, set_dv, self.k_clusters)

                # Calculating Global JObjective function.
                glob_jobj = self.GlobJObjCosA(x_data, mu_arr, ctags_set)

                # Terminating upon the difference of the last two Global JObej values.
                if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:
                    # second condition is TEMP!
                    print 'last_gobj - glob_jobj', np.abs(last_gobj - glob_jobj)
                    print "Global Objective", glob_jobj

                    return mu_arr_dv, ctags_set

                else:
                    # Keeping the DV vector if the JObjective value is lower.
                    if glob_jobj < last_gobj:
                        last_gobj = glob_jobj
                        best_dv[:] = set_dv
                        best_efl = efl
                        best_clstr_tags_arr[:] = ctags_set[:]
                        best_mu_arr[:, :] = mu_arr[:, :]

                    clstr_tags_arr[:] = ctags_set[:]
                    mu_arr = mu_arr_dv

            print "Global: jObj-val = ", glob_jobj, ", last jObj-val = ", last_gobj

            # Creating the new distortion vectros tocastic set to be tested in the next EM step.
            while True:
                # Selecting randomly a value from the range of L values...
                # ...for the exponential distribution function below.
                efl = np.random.permutation(self.efl_vals)[0]
                if efl not in self.efl_usd_vals:
                    self.efl_usd_vals.append(efl)
                    break

                if len(self.efl_usd_vals) == len(self.efl_vals):
                    efl = best_efl
                    break

            for i in np.arange(self.eft_per_i - 1):
                dvz[i, :] = np.random.exponential(1 / efl, size=x_data.shape[1])
            dvz[-1, :] = best_dv[:]

        # Returning the Centroids and the Clusters best set-up found in one of the EM iterations...
        # ..., however, the EM here stops at a local minima not the global.
        return best_mu_arr, best_clstr_tags_arr

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'ICM_Max_Iter': self.icm_max_i,
            'final_iter': self.conv_step,
            'ml_wg': self.ml_wg,
            'cl_wg': self.cl_wg,
            'convg_diff': self.cvg,
            'stoch_dv': self.dv,
            'exp_l_usd_vals': self.efl_usd_vals
        }

    def ICM(self, dvz, x_data, mu_arr, clstr_tags_arr):
        """ ICM: Iterated Conditional Modes (for the E-Step).
            After all points are assigned, they are randomly re-ordered, and the assignment process
            is repeated. This process proceeds until no point changes its cluster assignment
            between two successive iterations.
        """

        print "In ICM..."

        ctags_stage0 = np.vstack([clstr_tags_arr] * dvz.shape[0])
        ctags_stage1 = np.zeros_like(ctags_stage0)
        stg_change_cnt = np.zeros(dvz.shape[0])

        no_change_cnt = 0
        # while no_change_cnt < 2:
        for dump in range(self.icm_max_i):

            # Stopping the rearrangment of the clusters when at least 2 times nothing changed...
            # ...thus, most-likelly the arrangmet is optimal.
            # if np.all(np.equal(stg_change_cnt, 2)):
            if np.max(stg_change_cnt) == 2:
                break
            else:
                print 'not yet'

            # Calculating the new Clusters.
            for x_idx in np.random.permutation(x_data.shape[0]):
                # print "index", x_idx

                # Creating a New
                for j, dv in enumerate(dvz):

                    # print "dv"

                    self.dv = dv

                    # Setting the initial value for the previews J-Objective value.
                    last_jobj = np.Inf

                    # Calculating the J-Objective for every x_i vector of the x_data set.
                    for mu_i in np.arange(mu_arr.shape[0]):

                        # Getting the indeces for this cluster....based on the previews stage.
                        clstr_idxs_arr = np.where(ctags_stage0[j] == mu_i)[0]

                        # Calculating the J-Objective.
                        j_obj = self.JObjCosA(x_idx, x_data, mu_i, mu_arr, clstr_idxs_arr)

                        if j_obj < last_jobj:
                            last_jobj = j_obj
                            new_clstr_tag = mu_i

                    # Re-assinging....
                    ctags_stage1[j, x_idx] = new_clstr_tag

            # Checking if the clusters where for every DV and add 1 in the stage-change counter.
            stg_change_cnt[np.where(np.all(np.equal(ctags_stage0, ctags_stage1), axis=1))] += 1

        # Returning clstr_tags_arr.
        return ctags_stage1

    def JObjCosA(self, x_idx, x_data, mu_i, mu_arr, clstr_idx_arr):
        """ JObjCosA: J-Objective function for parametrized Cosine Stochastic Distortion Measure.
            details later...
        """

        # Calculating the cosine distance of the specific x_i from the cluster's centroid.
        # --------------------------------------------------------------------------------
        dist = vop.cosDa_vect(mu_arr[mu_i], x_data[x_idx], self.dv)

        # print np.array(mu_arr), x_data[x_idx]

        # Calculating Must-Link violation cost.
        # -------------------------------------
        ml_cost = 0.0

        # Selecting ml_pairs containing the x_idx.
        mlvpairs_wth_xidx = self.ml_pair_idxs[np.where((self.ml_pair_idxs == x_idx))[0]]

        # Selecting must-link vionlation where the x_idx is included
        if x_idx not in clstr_idx_arr:
            ml_voil_tests = np.isin(mlvpairs_wth_xidx, np.hstack((x_idx, clstr_idx_arr)))
        else:
            ml_voil_tests = np.isin(mlvpairs_wth_xidx, clstr_idx_arr)

        # Getting the must-link-constraints violations.
        mlv_pair_rows = np.where(
            (np.logical_xor(ml_voil_tests[:, 0], ml_voil_tests[:, 1]) == True)
        )[0]

        mlv_cnts = np.size(mlv_pair_rows)

        if mlv_cnts:

            # Calculating all pairs of violation costs for must-link constraints.
            # NOTE: The violation cost is equivalent to the parametrized Cosine distance...
            # ...which here is equivalent to the (1 - dot product) because the data points...
            # ...assumed to be normalized by the parametrized Norm of the vectors.
            viol_costs = vop.cosDa_rpairs(
                x_data, self.dv, self.ml_pair_idxs, mlv_pair_rows
            )

            # Sum-ing up Weighted violations costs.
            ml_cost = self.ml_wg * np.sum(viol_costs)  # / float(mlv_cnts)

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        cl_cost = 0.0

        # Selecting cl_pairs containing the x_idx.
        clvpairs_wth_xidx = self.cl_pair_idxs[np.where((self.cl_pair_idxs == x_idx))[0]]

        # Getting the index(s) of the cannot-link-constraints index-table of this data sample.
        if x_idx not in clstr_idx_arr:
            cl_voil_tests = np.isin(clvpairs_wth_xidx, np.hstack((x_idx, clstr_idx_arr)))
        else:
            cl_voil_tests = np.isin(clvpairs_wth_xidx, clstr_idx_arr)

        clv_pair_rows = np.where(
            (np.logical_and(cl_voil_tests[:, 0], cl_voil_tests[:, 1]) == True)
        )[0]

        clv_cnts = np.size(clv_pair_rows)

        if clv_cnts:

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance minus the...
            # ...parametrized Cosine distance of the vectors. Since MaxCosine is 1 then...
            # ...maxCosineDistance - CosineDistance == CosineSimilarty of the vectors....
            # ...Again the data points assumed to be normalized.
            viol_costs = 1.0 - np.array(
                vop.cosDa_rpairs(
                    x_data, self.dv, self.cl_pair_idxs, clv_pair_rows
                )
            )

            cl_cost = self.cl_wg * np.sum(viol_costs)  # / float(clv_cnts)

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost

    def GlobJObjCosA(self, x_data, mu_arr, clstr_tags_arr):
        """
        """
        print "In GlobalJObjCosA..."

        # Calculating the distance of all vectors, the must-link and cannot-link violations scores.
        sum_d, ml_cost, cl_cost, norm_part_value, params_pdf = 0.0, 0.0, 0.0, 0.0, 0.0
        # ml_cnt, cl_cnt = 0.0, 0.0

        # xd_rows = np.arange(x_data.shape[0])
        # mu_rows = np.arange(mu_arr.shape[0])

        # Calculating the cosine distances and add the to the total sum of distances.
        # ---------------------------------------------------------------------------
        sum_d = np.sum(vop.cosDa(mu_arr, x_data, self.dv))

        # Calculating Must-Link violation cost.
        # -------------------------------------
        for i in range(mu_arr.shape[0]):

            # Getting the indeces for the i cluster.
            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.
            ml_voil_tests = np.isin(self.ml_pair_idxs, clstr_idxs_arr)
            mlv_pair_rows = np.where(
                (np.logical_xor(ml_voil_tests[:, 0], ml_voil_tests[:, 1]) == True)
            )[0]

            ml_cnt = np.size(mlv_pair_rows)

            if ml_cnt:

                # Calculating all pairs of violation costs for must-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance.
                viol_costs = np.sum(
                    vop.cosDa_rpairs(
                        x_data, self.dv, self.ml_pair_idxs, mlv_pair_rows
                    )
                )

                ml_cost += viol_costs

            # Calculating Cannot-Link violation cost.
            # ---------------------------------------

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            cl_voil_tests = np.isin(self.cl_pair_idxs, clstr_idxs_arr)
            clv_pair_rows = np.where(
                (np.logical_and(cl_voil_tests[:, 0], cl_voil_tests[:, 1]) == True)
            )[0]

            cl_cnt = np.size(clv_pair_rows)

            if cl_cnt:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance
                viol_costs = np.array(
                    vop.cosDa_rpairs(
                        x_data, self.dv, self.cl_pair_idxs, clv_pair_rows
                    )
                )

                max_vcost = 1.0  # np.max(viol_costs)
                cl_cost += np.sum(max_vcost - viol_costs)

        # Averaging all-total distance costs.
        sum_d = sum_d / (x_data.shape[0] * mu_arr.shape[0])
        if ml_cnt:
            ml_cost = ml_cost / np.float(ml_cnt)
        if cl_cnt:
            cl_cost = cl_cost / np.float(cl_cnt)

        # print 'dims', x_data.shape[1]
        # print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost=', sum_d + ml_cost + cl_cost

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + (self.ml_wg * ml_cost) + (self.cl_wg * cl_cost)
