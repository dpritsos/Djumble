# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import matplotlib.pyplot as plt
import scipy.special as special
import time as tm


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
                clustering process. (NOT IN USE FOR NOW)
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

    def fit(self, x_data, neg_idxs4clstring=set([])):
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
        #if self.w_violations is None:
        #    self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        # ### I am not sure what kind of values this weights should actually have.

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint(x_data, k_clusters, self.must_lnk,
        #                                           self.cannot_lnk, dmeasure_noparam)
        # init_clstr_sets_lst = ConsolidateAL(neibs_sets, x_data,
        #                                      self.must_lnk, dmeasure_noparam)
        clstr_idxs_set_lst = list()

        # If initial centroids arguments has been passed.
        if self.init_centroids:
            clstr_idxs_set_lst.extend(self.init_centroids)
            # ### Maybe this should be changed to a numpy vector of indices.
        else:

            # ######### This might actually change based on the initialization above.

            # Pick k random vector from the x_data set as initial centroids. Where k is equals...
            # ...the number of self.k_clusters.
            k_rand_idx = np.random.randint(0, self.k_clusters, size=x_data.shape[0])
            clstr_idxs_set_lst.extend([set(idx) for idx in k_rand_idx])

        # Set of indices not participating in clustering. NOTE: For particular experiments only.
        self.neg_idxs4clstring = neg_idxs4clstring

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_lst = self.MeanCosA(x_data, clstr_idxs_set_lst)

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
            clstr_idxs_set_lst = self.ICM(x_data, mu_lst, clstr_idxs_set_lst)

            # The M-Step.

            # Recalculating centroids upon the new clusters set-up.
            mu_lst = self.MeanCosA(x_data, clstr_idxs_set_lst)
            # print mu_lst

            # Calculating Global JObjective function.
            glob_jobj = self.GlobJObjCosA(x_data, mu_lst, clstr_idxs_set_lst)

            timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            print "Time elapsed : %d:%d:%d:%d" % timel

            # Terminating upon difference of the last two Global JObej values.
            if np.abs(last_gobj - glob_jobj) < self.cvg or glob_jobj < self.cvg:

                print 'last_gobj - glob_jobj', last_gobj - glob_jobj
                print "Global Objective", glob_jobj
                break
            else:
                last_gobj = glob_jobj

            print "Global Objective", glob_jobj

        # Storing the amount of iterations until convergence.
        self.conv_step = conv_step

        # Returning the Centroids and the Clusters,i.e. the set of indeces for each cluster.
        return mu_lst, clstr_idxs_set_lst

    def get_params(self):
        return {
            'k_clusters': self.k_clusters,
            'max_iter': self.max_iter,
            'final_iter': self.conv_step,
            'convg_diff': self.cvg
        }

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

                # Skipping the indices should not participate in clustering.
                if x_idx in self.neg_idxs4clstring:
                    continue

                # print "Sample:", x_idx

                # Setting the initial value for the previews J-Objective value.
                last_jobj = np.Inf

                # Calculating the J-Objective for every x_i vector of the x_data set.
                for i, (mu, clstr_idxs_set) in enumerate(zip(mu_lst, clstr_idxs_sets_lst)):

                    # Calculating the J-Objective.
                    j_obj = self.JObjCosA(x_idx, x_data, mu, clstr_idxs_set)

                    if j_obj < last_jobj:
                        last_jobj = j_obj
                        mu_neib_idx = i

                # Re-assinging the x_i vector to the new cluster if not already.
                if x_idx not in clstr_idxs_sets_lst[mu_neib_idx]:

                    # Remove x form all Clusters.
                    for clstr_idxs_set in clstr_idxs_sets_lst:
                        clstr_idxs_set.discard(x_idx)
                        # clstr_idxs_sets_lst[midx].discard(x_idx)

                    clstr_idxs_sets_lst[mu_neib_idx].add(x_idx)

                    no_change = False

                else:
                    no_change = True

            # Counting Non-Changes, i.e. if no change happens for two (2) iteration the...
            # ...re-assingment process stops.
            if no_change:
                no_change_cnt += 1

        timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
        print "ICM elapsed : %d:%d:%d:%d" % timel

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

            Output
            ------
                Returning the parameterized cosine distance between two vectors.

        """

        # Converting vectors x1 and x2 to 1D matrices.
        if sp.sparse.issparse(x1):
            x1 = sp.matrix(x1.todense())
        else:
            x1 = sp.matrix(x1)

        if sp.sparse.issparse(x2):
            x2 = sp.matrix(x2.todense())
        else:
            x2 = sp.matrix(x2)

        if not np.any(x1):
            print x1
        if not np.any(x2):
            print x2

        # Calculating and returning the parameterized cosine distance.
        # np.sqrt(np.abs(x1 * self.A[:, :] * x1.T)) *
        # np.sqrt(np.abs(x2 * self.A[:, :] * x2.T))
        return (
            1 - (
                 x1 * x2.T /
                 (
                  np.sqrt(x1 * x1.T) *
                  np.sqrt(x2 * x2.T)
                  )
                )
        )

    def MeanCosA(self, x_data, clstr_idxs_lsts):
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
        for clstr_ilst in clstr_idxs_lsts:

            # Summing up all the X data points for the current cluster.
            if len(clstr_ilst):
                xi_sum = x_data[list(clstr_ilst), :].sum(axis=0)
                xi_sum = sp.matrix(xi_sum)
            else:
                print "Zero Mean for a clucter triggered!!!"
                zero_vect = np.zeros_like(x_data[0, :])
                zero_vect[:] = 1e-15
                xi_sum = sp.matrix(zero_vect)

            # Calculating denominator ||Σ xi||(A)
            parametrized_norm_xi = np.sqrt(xi_sum * xi_sum.T)
            # parametrized_norm_xi = np.sqrt(np.abs(xi_sum * self.A[:, :] * xi_sum.T))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            mu_lst.append(xi_sum / parametrized_norm_xi)

        return mu_lst

    def JObjCosA(self, x_idx, x_data, mu, clstr_idxs_set):
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
        dist = self.CosDistA(x_data[x_idx, :], mu)

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        for x_cons in self.must_lnk:

            x = list(x_cons)

            if x_idx in x_cons:

                if (x[0] in clstr_idxs_set or x[1] in clstr_idxs_set) and not (x_cons <= clstr_idxs_set):

                    ml_cost += self.CosDistA(x_data[x[0], :], x_data[x[1], :])

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        for x_cons in self.cannot_lnk:

            if x_idx in x_cons:

                if x_cons <= clstr_idxs_set:

                    x = list(x_cons)

                    cl_cost += (1 - self.CosDistA(x_data[x[0], :], x_data[x[1], :]))

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost

    def GlobJObjCosA(self, x_data, mu_lst, clstr_idxs_set_lst):
        """
        """

        print "In GlobalJObjCosA..."

        sum_d = 0.0
        smpls_cnt = 0.0
        for mu, clstr_idxs in zip(mu_lst, clstr_idxs_set_lst):

            for x_clstr_idx in clstr_idxs:

                if x_clstr_idx not in self.neg_idxs4clstring:  # <---NOTE

                    smpls_cnt += 1.0

                    sum_d += self.CosDistA(mu, x_data[x_clstr_idx])

        # Averaging dividing be the number of samples.
        print 'Samples:', smpls_cnt
        sum_d = sum_d / smpls_cnt

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        ml_cnt = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in self.must_lnk:

                if x_cons not in self.neg_idxs4clstring:  # <---NOTE

                    x = list(x_cons)

                    if (x[0] in clstr_idxs_set or x[1] in clstr_idxs_set) and not (x_cons <= clstr_idxs_set):

                        ml_cnt += 1.0

                        ml_cost += self.CosDistA(x_data[x[0], :], x_data[x[1], :])

        # Averaging dividing be the number of must-link constrains.
        if ml_cnt:
            ml_cost = ml_cost / ml_cnt

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        cl_cnt = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in self.cannot_lnk:

                if x_cons not in self.neg_idxs4clstring:  # <---NOTE

                    if x_cons <= clstr_idxs_set:

                        cl_cnt += 1.0

                        x = list(x_cons)

                        cl_cost += (1 - self.CosDistA(x_data[x[0], :], x_data[x[1], :]))

        # Averaging dividing be the number of cannot-link constrains.
        if cl_cnt:
            cl_cost = cl_cost / cl_cnt

        print 'dims', x_data.shape[1]
        print 'sum_d, ml_cost, cl_cost', sum_d, ml_cost, cl_cost
        print 'sum_d + ml_cost + cl_cost', sum_d + ml_cost + cl_cost

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + ml_cost + cl_cost


if __name__ == '__main__':

    test_dims = 1000

    print "Creating Sample"
    x_data_2d_arr1 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))
    x_data_2d_arr2 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))
    x_data_2d_arr3 = sps.vonmises.rvs(5.0, loc=np.random.uniform(0.0, 1400.0, size=(1, test_dims)), scale=1, size=(500, test_dims))

    x_data_2d_arr1 = x_data_2d_arr1 / np.max(x_data_2d_arr1, axis=1).reshape(500, 1)
    x_data_2d_arr2 = x_data_2d_arr2 / np.max(x_data_2d_arr2, axis=1).reshape(500, 1)
    x_data_2d_arr3 = x_data_2d_arr3 / np.max(x_data_2d_arr3, axis=1).reshape(500, 1)

    # print x_data_2d_arr1

    # 0/0

# (0.7, 0.2, 0.7, 0.2, 0.6, 0.6, 0.1, 0.3, 0.8, 0.5)
# (0.6, 0.6, 0.7, 0.2, 0.6, 0.6, 0.8, 0.3, 0.9, 0.1)
# (0.2, 0.3, 0.7, 0.2, 0.6, 0.6, 0.2, 0.3, 0.6, 0.4)

    # tuple(np.random.normal(0.0, 10.0, size=2))
    # x_data_2d_arr1 = np.random.vonmises(0.5, 100, size=(20, 2))
    # x_data_2d_arr2 = np.random.vonmises(0.5, 1000, size=(20, 2))
    # x_data_2d_arr3 = np.random.vonmises(0.5, 10000, size=(20, 2))

    x_data_2d_arr = np.vstack((x_data_2d_arr1, x_data_2d_arr2, x_data_2d_arr3))
    print x_data_2d_arr

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
    #plt.show()

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
        set([5, 535]),
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
    init_centrs = [set([0]), set([550]), set([1100])]
    print "Running HMRF Kmeans"
    ckmeans = CosineKmeans(k_clusters,  must_lnk_con, cannot_lnk_con, init_centroids=init_centrs,
                           max_iter=300, cvg=0.0001)

    res = ckmeans.fit(x_data_2d_arr)  # , set([50]))

    for mu_idx, clstr_idxs in enumerate(res[1]):

        print mu_idx+1, len(clstr_idxs), np.sort(clstr_idxs)

        for xy in x_data_2d_arr[list(clstr_idxs)]:
            plt.text(xy[0], xy[1], str(mu_idx+1), color='red', fontsize=15)
        # plt.plot(x_data_2d_arr2, '^')
        # plt.plot(x_data_2d_arr3, '>')

    plt.show()
