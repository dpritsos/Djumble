# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import scipy.special as special

import ..dsmeasure.cy as vop


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

    def __init__(self, k_clusters, must_lnk_con, cannot_lnk_con, init_centroids=None,
                 ml_wg=1.0, cl_wg=1.0, max_iter=300, cvg=0.001, lrn_rate=0.0003, ray_sigma=0.5,
                 d_params=None, norm_part=False, globj_norm=False):

        self.k_clusters = k_clusters
        self.mst_lnk_idxs = must_lnk_con
        self.cnt_lnk_idxs = cannot_lnk_con
        self.init_centroids = init_centroids
        self.ml_wg = ml_wg
        self.cl_wg = cl_wg
        self.max_iter = max_iter
        self.cvg = cvg
        self.lrn_rate = lrn_rate
        self.ray_sigma = ray_sigma
        self.A = d_params
        self.norm_part = norm_part

        # This option enables or disables the normalizations values to be included in the...
        # ...calculation of the total values, other than the total cosine distances, the...
        # ...total must-link and cannot-link violation scores.
        self.globj_norm = globj_norm

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
            self.A = np.ones((x_data.shape[1]), dtype=np.float)
        # A should be a diagonal matrix form for the calculations in the functions bellow. The...
        # ...sparse form will save space and the csr_matrix will make the dia_matrix write-able.
        self.A = np.diag(self.A)
        self.A = sp.sparse.lil_matrix(self.A)

        # Setting up the violation weights matrix if not have been passed as class argument.
        # if self.w_violations is None:
        #   self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        #   # ### I am not sure what kind of values this weights should actually have.

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

        # NOTE: Initially normalizing the samples under the distortion parameters values in...
        # ...order to reducing the cosine distance calculation to a simple dot products...
        # ...calculation in the rest of the EM (sub)steps.
        x_data = np.divide(
            x_data,
            np.sqrt(
                np.diag(np.dot(np.dot(x_data, self.A[:, :].toarray()), x_data.T)),
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
            clstr_tags_arr = self.ICM(x_data, mu_arr, clstr_tags_arr)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_arr = self.MeanCosA(x_data, clstr_tags_arr)
            # print mu_lst

            # Re-estimating distortion measure parameters upon the new clusters set-up.
            self.A = self.UpdateDistorParams(self.A, x_data, mu_arr, clstr_tags_arr)

            # NOTE: Normalizing the samples under the new parameters values in order to reducing...
            # ...the cosine distance calculation to a simple dot products calculation in the...
            # ...rest of the EM (sub)steps.
            x_data = np.divide(
                x_data,
                np.sqrt(
                    np.diag(np.dot(np.dot(x_data, self.A[:, :].toarray()), x_data.T)),
                    dtype=np.float
                ).reshape(x_data.shape[0], 1)
            )

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_arr, clstr_tags_arr)

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
            'ml_wg': self.ml_wg,
            'cl_wg': self.cl_wg,
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
        dist = cos2Da(mu, x_data[x_idx, :], self.A)

        # Calculating Must-Link violation cost.
        # -------------------------------------
        ml_cost = 0.0

        # Getting the index(s) of the must-link-constraints index-table of this data sample.
        idxzof_mli4smpli = np.where(self.mst_lnk_idxs == x_idx)

        if idxzof_mli4smpli[0].shape[0]:

            # Getting the must-link, with current, data points indeces which they should be in...
            # ...the same cluster.
            mliz_with_smpli = self.mst_lnk_idxs[~idxzof_mli4smpli[0], idxzof_mli4smpli[1]]

            # Getting the indeces of must-link than are not in the cluster as they should have been.
            viol_idxs = self.mst_lnk_idxs[:, ~np.in1d(mliz_with_smpli, clstr_idx_arr)]

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for must-link constraints.
                # NOTE: The violation cost is equivalent to the parametrized Cosine distance...
                # ...which here is equivalent to the (1 - dot product) because the data points...
                # ...assumed to be normalized by the parametrized Norm of the vectors.
                viol_costs = cos2Da_rows(x_data, x_data, self.A, x_idx, viol_idxs[1])

                # Sum-ing up Weighted violations costs.
                ml_cost = np.sum(viol_costs)
                # ml_cost = np.sum(np.multiply(self.ml_wg, viol_costs)) <--- PORPER

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        cl_cost = 0.0

        # Getting the index(s) of the cannot-link-constraints index-table of this data sample.
        idxzof_cli4smpli = np.where(self.cnt_lnk_idxs == x_idx)

        if idxzof_cli4smpli[0].shape[0]:

            # Getting the cannot-link, with current, data points indeces which they should not...
            # ...be in the same cluster.
            cliz_with_smpli = self.cnt_lnk_idxs[~idxzof_cli4smpli[0], idxzof_cli4smpli[1]]

            # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
            # ...have been.
            viol_idxs = self.cnt_lnk_idxs[:, np.in1d(cliz_with_smpli, clstr_idx_arr)]

            if viol_idxs.shape[0]:

                # Calculating all pairs of violation costs for cannot-link constraints.
                # NOTE: The violation cost is equivalent to the maxCosine distance minus the...
                # ...parametrized Cosine distance of the vectors. Since MaxCosine is 1 then...
                # ...maxCosineDistance - CosineDistance == CosineSimilarty of the vectors....
                # ...Again the data points assumed to be normalized.
                viol_costs = cos2Da_rows(x_data, x_data, self.A, x_idx, viol_idxs[1])
                # viol_costs = np.ones_like(viol_costs) - viol_costs

                # Sum-ing up Weighted violations costs.
                cl_cost = np.sum(viol_costs)
                # cl_cost = np.sum(np.multiply(self.cl_wg, viol_costs)) <--- PORPER

                # Equivalent to: (in a for-loop implementation)
                # cl_cost += self.w_violations[x[0], x[1]] *\
                # (1 - self.CosDistA(x_data[x[0], :], x_data[x[1], :]))

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        sum1, sum2 = 0.0, 0.0
        for a in np.diag(self.A[:, :].toarray()):
            sum1 += np.log(a)
            sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
        params_pdf = sum1 - sum2 -\
            (2 * np.diag(self.A[:, :].toarray()).shape[0] * np.log(self.ray_sigma))

        # NOTE!
        params_pdf = 0.0

        # Calculating the log normalization function of the von Mises-Fisher distribution...
        # ...NOTE: Only for this cluster i.e. this vMF of the whole PDF mixture.
        if self.norm_part:
            norm_part_value = self.NormPart(x_data[clstr_idx_arr])
        else:
            norm_part_value = 0.0

        # print "In JObjCosA...", dist, ml_cost, cl_cost, params_pdf, norm_part_value
        # print "Params are: ", self.A

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost - params_pdf + norm_part_value

    def GlobJObjCosA(self, x_data, mu_arr, clstr_tags_arr):
        """
        """

        print "In GlobalJObjCosA..."

        # Calculating the distance of all vectors, the must-link and cannot-link violations scores.
        sum_d, ml_cost, cl_cost, norm_part_value, params_pdf = 0.0, 0.0, 0.0, 0.0, 0.0
        smlps_cnt, ml_cnt, cl_cnt = 0.0, 0.0, 0.0

        xd_rows = np.arrange(x_data.shape[0])
        mu_rows = np.arrange(mu.shape[0])

        # Calculating the cosine distances and add the to the total sum of distances.
        # ---------------------------------------------------------------------------
        sum_d = np.sum(vop.cos2Da(mu_arr, x_data, self.A))

        # Calculating Must-Link violation cost.
        # -------------------------------------

        # Getting the must-link left side of the pair constraints, i.e. the row indeces...
        # ...of the constraints matrix that are in the cluster's set of indeces.
        # in_clstr_ml_rows = np.in1d(self.mst_lnk_idxs[0], clstr_idxs_arr)

        # Getting the indeces of must-link than are not in the cluster as they should...
        # ...have been.

        ml_clstr_comn_idxs = np.in1d(
            self.mst_lnk_idxs, clstr_idxs_arr
        ).reshape(2, self.mst_lnk_idxs.shape[1])

        ml_viol_columns = np.intersect1d(
            np.where(ml_clstr_comn_idxs[0] != ml_clstr_comn_idxs[1])[0],
            np.hstack((ml_clstr_comn_idxs[0].nonzero()[0], ml_clstr_comn_idxs[1].nonzero()[0]))
        )

        viol_ipairs = self.mst_lnk_idxs[:, ml_viol_columns]

        #
        ml_cnt += float(viol_ipairs.shape[0])

        if viol_ipairs.shape[0]:

            # Calculating all pairs of violation costs for must-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance.
            viol_costs = np.sum(
                vop.cos2Da_rows(x_data, x_data, self.A, viol_ipairs[0], viol_ipairs[1])
            )

            # if viol_costs.shape[0] > 1:
            #     viol_costs_onetime = np.tril(viol_costs, -1)
            # else:
            #     viol_costs_onetime = viol_costs

            # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
            # ...should be element-by-element.

            ml_cost += np.sum(np.multiply(self.ml_wg, viol_costs))

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------

        # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
        # ...of the constraints matrix that are in the cluster's set of indeces.
        in_clstr_cl_rows = np.in1d(self.cnt_lnk_idxs[0], clstr_idxs_arr)

        # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
        # ...have been.

        cl_clstr_comn_idxs = np.in1d(
            self.cnt_lnk_idxs, clstr_idxs_arr
        ).reshape(2, self.cnt_lnk_idxs.shape[1])

        cl_viol_columns = np.intersect1d(
            np.where(cl_clstr_comn_idxs[0] == cl_clstr_comn_idxs[1])[0],
            cl_clstr_comn_idxs[0].nonzero()[0]
        )

        viol_ipairs = self.cnt_lnk_idxs[:, cl_viol_columns]

        #
        cl_cnt += float(viol_ipairs.shape[0])

        if viol_ipairs.shape[0]:

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance
            viol_costs = np.sum(
                vop.cos2Da_rows(x_data, x_data, self.A, viol_ipairs[0], viol_ipairs[1])
            )

            # if viol_costs.shape[0] > 1:
            #     viol_costs_onetime = np.tril(viol_costs, -1)
            # else:
            #     viol_costs_onetime = viol_costs

            # Sum-ing up Weighted violations costs. NOTE: Here the matrix multiply...
            # ...should be element-by-element. NOTE#2: We are getting only the lower...
            # ...triangle because we need the cosine distance of the constraints pairs...
            # ...only ones.
            cl_cost += np.sum(np.multiply(self.cl_wg, viol_costs))

        """
        # Averaging EVERYTHING.
        sum_d = sum_d / smlps_cnt

        if ml_cnt:
            ml_cost = ml_cost / ml_cnt

        if cl_cnt:
            cl_cost = cl_cost / cl_cnt
        """

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        if self.globj_norm:
            sum1, sum2 = 0.0, 0.0
            for a in np.diag(self.A[:, :].toarray()):
                sum1 += np.log(a)
                sum2 += np.square(a) / (2 * np.square(self.ray_sigma))
            params_pdf = sum1 - sum2 -\
                (2 * np.diag(self.A[:, :].toarray()).shape[0] * np.log(self.ray_sigma))
        else:
            params_pdf = 0.0

        # Calculating the log normalization function of the von Mises-Fisher distribution...
        # ...of the whole mixture.
        if self.norm_part and self.globj_norm:
            norm_part_value = 0.0
            for i in enumerate(mu_arr.shape[0]):
                clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]
                norm_part_value += self.NormPart(x_data[clstr_idxs_arr])
        else:
            norm_part_value = 0.0

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

        new_A = np.zeros_like(A.data, dtype=np.float)

        # Initializing...
        # xm_pderiv, mlcost_pderiv, clcost_pderiv = 0.0, 0.0, 0.0
        # smpls_cnt, ml_cnt, cl_cnt = 0.0, 0.0, 0.0

        ml_viol_pairs = list()
        cl_viol_pairs = list()
        clstr_idxs_arrz_lst = list()

        # Collecting Violation Pairs.
        for i in range(mu_arr.shape[0]):

            # Getting the indeces for the i cluster.
            clstr_idxs_arr = np.where(clstr_tags_arr == i)[0]
            clstr_idxs_arrz_lst.append(clstr_idxs_arr)

            #
            # smpls_cnt += float(clstr_idxs_arr.shape[0])

            # Getting the must-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_ml_rows = np.in1d(self.mst_lnk_idxs[0], clstr_idxs_arr)

            # Getting the indeces of must-link than are not in the cluster as they should...
            # ...have been.

            ml_clstr_comn_idxs = np.in1d(
                self.mst_lnk_idxs, clstr_idxs_arr
            ).reshape(2, self.mst_lnk_idxs.shape[1])

            ml_viol_columns = np.intersect1d(
                np.where(ml_clstr_comn_idxs[0] != ml_clstr_comn_idxs[1])[0],
                np.hstack(
                    (ml_clstr_comn_idxs[0].nonzero()[0], ml_clstr_comn_idxs[1].nonzero()[0])
                )
            )

            ml_viol_pairs.append(self.mst_lnk_idxs[:, ml_viol_columns])

            #
            # ml_cnt += float(viol_ipairs.shape[0])

            # Getting the cannot-link left side of the pair constraints, i.e. the row indeces...
            # ...of the constraints matrix that are in the cluster's set of indeces.
            in_clstr_cl_rows = np.in1d(self.cnt_lnk_idxs[0], clstr_idxs_arr)

            # Getting the indeces of cannot-link than are in the cluster as they shouldn't...
            # ...have been.

            cl_clstr_comn_idxs = np.in1d(
                self.cnt_lnk_idxs, clstr_idxs_arr
            ).reshape(2, self.cnt_lnk_idxs.shape[1])

            cl_viol_columns = np.intersect1d(
                np.where(cl_clstr_comn_idxs[0] == cl_clstr_comn_idxs[1])[0],
                cl_clstr_comn_idxs[0].nonzero()[0]
            )

            cl_viol_pairs.append(elf.cnt_lnk_idxs[:, cl_viol_columns])

            #
            # cl_cnt += float(viol_ipairs.shape[0])

        # Calculating the partial derivatives of each parameter for all cluster's member...
        # ...for each cluster.
        # ---------------------------------------------------------------------------------
        for k, clstr_idxs_arr in enumerate(clstr_idxs_arrz_lst):
            xm_pderiv = vop.pDerivative_seq_rows(A, mu_arr, x_data, )

            vop.pDerivative_seq_one2many()

        # Calculating Must-Link violation cost.
        # -------------------------------------
        if ml_viol_pairs:

            ml_viol_pairs = np.hstack(ml_viol_pairs)

            # Calculating the partial derivatives of all pairs of violations for...
            # ...must-link constraints.
            mlcost_pderiv = vop.pDerivative_seq_rows(
                A, x_data, x_data, ml_viol_pairs[0], ml_viol_pairs[1]
            )

        # Calculating Cannot-Link violation cost.
        # ---------------------------------------
        if cl_viol_pairs:

            cl_viol_pairs = np.hstack(cl_viol_pairs)

            # Calculating all pairs of violation costs for cannot-link constraints.
            # NOTE: The violation cost is equivalent to the maxCosine distance
            clcost_pderiv = vop.pDerivative_seq_rows(
                A, x_data, x_data, cl_viol_pairs[0], cl_viol_pairs[1]
            )

        # Averaging EVERYTHING
        # xm_pderiv = xm_pderiv / smpls_cnt
        # if ml_cnt:
        #     mlcost_pderiv = mlcost_pderiv / ml_cnt
        # if cl_cnt:
        #     clcost_pderiv = clcost_pderiv / cl_cnt

        # Updating Process.
        for i, a in enumerate(A):

            # Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
            a_pderiv = (1 / a) - (a / np.square(self.ray_sigma))
            # NOTE: a_pderiv = 0.0
            # print 'Rayleigh Partial', a_pderiv

            if np.abs(a_pderiv) == np.inf:
                print "Invalid patch for Rayleighs P'(A) triggered: (+/-)INF P'(A)=", a_pderiv
                a_pderiv = 1e-15
            elif a_pderiv == np.nan:
                print "Invalid patch for Rayleighs P(A) triggered: NaN P'(A)=", a_pderiv
                a_pderiv = 1e-15

            # Changing a diagonal value of the A cosine similarity parameters measure.
            A[i] = a +\
                (
                    self.lrn_rate *
                    (
                        xm_pderiv[i]
                        + (self.ml_wg * mlcost_pderiv[i])
                        - (self.cl_wg * clcost_pderiv[i])
                        - a_pderiv
                    )
                )


        """
        for a_idx, a in enumerate(np.array([a[0] for a in A.data])):


            # print self.lrn_rate * (xm_pderiv + mlcost_pderiv + clcost_pderiv - a_pderiv)
            # print xm_pderiv, mlcost_pderiv, clcost_pderiv, a_pderiv
            if new_A[a_idx] < 0.0:
                print self.lrn_rate
                print xm_pderiv
                print mlcost_pderiv
                print clcost_pderiv
                print a_pderiv
                0/0

            # ΝΟΤΕ: Invalid patch for let the experiments to be completed.
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

        # A[:, :] = sp.sparse.lil_matrix(np.diag(new_A))

        # Returning the A parameters.
        return A