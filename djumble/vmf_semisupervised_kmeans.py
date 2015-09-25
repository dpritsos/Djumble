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
        self.A = d_params

    def Fit(self, x_data):
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
        if not self.A:
            self.A = np.random.uniform(0.50, 100.0, size=x_data.shape[1])
        # A should be a diagonal matrix form for the calculations in the functions bellow. The...
        # ...sparse form will save space and the csr_matrix will make the dia_matrix write-able.
        self.A = sp.sparse.dia_matrix((self.A, [0]), shape=(self.A.shape[0], self.A.shape[0]))
        self.A = sp.sparse.csr_matrix(self.A)

        # Setting up the violation weights matrix if not have been passed as class argument.
        if not self.w_violations:
            self.w_violations = np.random.uniform(0.9, 0.9, size=(x_data.shape[0], x_data.shape[0]))
        # ### I am not sure what kind of values this weights should actually have.

        # Deriving initial centroids lists from the must-link an cannot-link constraints.
        # Not ready yet...
        # init_clstr_sets_lst = FarFirstCosntraint(x_data, k_clusters, self.must_lnk,
        #                                           self.cannot_lnk, dmeasure_noparam)
        # init_clstr_sets_lst = ConsolidateAL(neibs_sets, x_data,
        #                                      self.must_lnk, dmeasure_noparam)
        init_clstr_sets_lst = list()

        # If initial centroids arguments has been passed.
        init_clstr_sets_lst.extend(self.init_centroids)
        # ### Maybe this should be changed to a numpy vector of indices.

        # Calculating the initial Centroids of the assumed hyper-shperical clusters.
        mu_lst = MeanCosA(x_data, init_clstr_sets_lst)

        # EM algorithm execution.

        # This values is the last global objective. Thus it should be the highest possible...
        # ...number initially, i.e. Inf.
        last_gobj = np.Inf

        # While no convergence yet repeat for at least i times.
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
            self.A = UpdateDistorParams(self.A, x_data, mu_lst, clstr_idxs_set_lst)

            # Calculating Global JObjective function.
            glob_jobj = GlobJObjCosA(x_data, mu_lst, clstr_idxs_set_lst)

            # Terminating upon difference of the last two Global JObej values.
            if (last_gobj - glob_jobj) < self.cvg:
                print "Global Objective", glob_jobj
                break
            else:
                last_gobj = glob_jobj

            print "Global Objective", glob_jobj

        # Returning the Centroids, Clusters/Neighbourhoods, distortion parameters,
        # constraint violations matrix.
        return mu_lst, clstr_idxs_set_lst, self.A.data

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
        no_change_cnt = 0
        while no_change_cnt < 2:

            # Calculating the new Clusters.
            for x_idx in np.random.randint(0, x_data.shape[0], size=x_data.shape[0]):

                # Setting the initial value for the previews J-Objective value.
                last_jobj = np.Inf

                # Calculating the J-Objective for every x_i vector of the x_data set.
                for i, (mu, clstr_idxs_set) in enumerate(zip(mu_lst, clstr_idxs_sets_lst)):

                    # Calculating the J-Objective.
                    j_obj = JObjCosA(x_idx, x_data, mu, clstr_idxs_set)

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

            Output
            ------
                mu_lst: The list of N centroids(mu_i), one for each of the N expected clusters.

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

        # Calculating the cosine distance of the specific x_i from the cluster's centroid.
        dist = CosDistA(x_data[x_idx, :], mu)

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        for x_cons in self.must_lnk:

            if x_idx in x_cons:

                if not (x_cons <= clstr_idxs_set):

                    x = list(x_cons)

                    ml_cost += self.w_violations[x[0], x[1]] *\
                        CosDistA(x_data[x[0], :], x_data[x[1], :])

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        for x_cons in self.cannot_lnk:

            if x_idx in x_cons:

                if x_cons <= clstr_idxs_set:

                    x = list(x_cons)

                    cl_cost += self.w_violations[x[0], x[1]] *\
                        (1 - CosDistA(x_data[x[0], :], x_data[x[1], :]))

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        sum1, sum2 = 0.0, 0.0
        for a in self.d_params:
            sum1 += np.log(a)
            sum2 += a / 2 * np.square(self.ray_sigma)
        params_pdf = sum1 - sum2 - 2 * self.d_params.shape[0] * np.log(self.ray_sigma)

        # Calculating Bessel Function for the first kind for order equals to 1/2 of the...
        # vector-space dimensions. Von Misses Fisher's k concentration is approximated as seen..
        # ....in Banerjee et al. 2003.
        dim = x_data.shape[1]
        # Concentration approximation
        r = np.linalg.norm(x_data)
        k = (r*dim - np.power(r, 3)) / (1 - np.power(r, 2))
        # The bessel function calculation using the above approximation for k Concentration.
        bessel = special.jv((dim/2.0)-1.0, k)

        # Calculating and returning the J-Objective value for this cluster's set-up.
        return dist + ml_cost + cl_cost + params_pdf - np.log(bessel)*x_data.shape[0]

    def GlobJObjCosA(self, x_data, mu_lst, clstr_idxs_set_lst):
        """
        """

        sum_d = 0.0
        for mu, neib_idxs in zip(mu_lst, clstr_idxs_set_lst):
            for x_neib_idx in neib_idxs:
                sum_d += CosDistA(x_data[x_neib_idx], mu)

        # Calculating Must-Link violation cost.
        ml_cost = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in self.must_lnk:

                if not (x_cons <= clstr_idxs_set):

                    x = list(x_cons)

                    ml_cost += self.w_violations[x[0], x[1]] *\
                        CosDistA(x_data[x[0], :], x_data[x[1], :])

        # Calculating Cannot-Link violation cost.
        cl_cost = 0.0
        for clstr_idxs_set in clstr_idxs_set_lst:

            for x_cons in self.cannot_lnk:

                if x_cons <= clstr_idxs_set:

                    x = list(x_cons)

                    cl_cost += self.w_violations[x[0], x[1]] *\
                        (1 - CosDistA(x_data[x[0], :], x_data[x[1], :]))

        # Calculating the cosine distance parameters PDF. In fact the log-form of Rayleigh's PDF.
        sum1, sum2 = 0.0, 0.0
        for a in self.d_params:
            sum1 += np.log(a)
            sum2 += a / 2 * np.square(self.ray_sigma)
        params_pdf = sum1 - sum2 - 2 * self.d_params.shape[0] * np.log(self.ray_sigma)

        # Calculating Bessel Function for the first kind for order equals to 1/2 of the...
        # vector-space dimensions. Von Misses Fisher's k concentration is approximated as seen..
        # ....in Banerjee et al. 2003.
        dim = x_data.shape[1]
        # Concentration approximation
        r = np.linalg.norm(x_data)
        k = (r*dim - np.power(r, 3)) / (1 - np.power(r, 2))
        # The bessel function calculation using the above approximation for k Concentration.
        bessel = special.jv((dim/2.0)-1.0, k)
        print bessel
        print 'np.log(bessel)*N', np.log(bessel)*x_data.shape[0]
        print x_data.shape[0]

        # Calculating and returning the Global J-Objective value for the current Spherical...
        # ...vMF-Mixture set-up.
        return sum_d + ml_cost + cl_cost + params_pdf - np.log(bessel)*x_data.shape[0]

    def UpdateDistorParams(self, A, x_data, mu_lst, clstr_idxs_lst):
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

        # Updating every parameter's value one-by-one.
        for a_idx, a in enumerate(A.data):

            # Calculating Partial Derivative of D(xi, mu).
            xm_pderiv = 0.0
            for mu, neib_idxs in zip(mu_lst, clstr_idxs_lst):
                for x_neib_idx in neib_idxs:
                    xm_pderiv += PartialDerivative(a_idx, x_data[x_neib_idx], mu, A)
            # print "Partial Distance", xm_pderiv

            # [idx for neib in clstr_idxs_lst for idx in neib]
            # Calculating the Partial Derivative of D(xi, xj) of Must-Link Constraints.
            mlcost_pderiv = 0.0
            for clstr_idxs_set in clstr_idxs_lst:

                for x_cons in self.must_lnk:

                    if not (x_cons <= clstr_idxs_set):

                        x = list(x_cons)

                        mlcost_pderiv += self.w_violations[x[0], x[1]] *\
                            PartialDerivative(a_idx, x_data[x[0], :], x_data[x[1], :], A)
            # print "Partial MustLink", mlcost_pderiv

            # Calculating the Partial Derivative of D(xi, xj) of Cannot-Link Constraints.
            clcost_pderiv = 0.0
            for clstr_idxs_set in clstr_idxs_lst:

                for x_cons in self.cannot_lnk:

                    if x_cons <= clstr_idxs_set:

                        x = list(x_cons)

                        clcost_pderiv += self.w_violations[x[0], x[1]] *\
                            PartialDerivative(a_idx, x_data[x[0], :], x_data[x[1], :], A)
            # print "Partial MustLink", clcost_pderiv

            # ### Calculating the Partial Derivative of Rayleigh's PDF over A parameters.
            a_pderiv = (1 / a) - (a / 2 * np.square(self.ray_sigma))

            # Changing a diagonal value of the A cosine similarity parameters measure.
            A[a_idx, a_idx] = (a + self.lrn_rate *
                               (xm_pderiv + mlcost_pderiv + clcost_pderiv - a_pderiv)
                               )
        # Returning the A parameters. This is actually a dump return for coding constance reasons.
        return A

    def PartialDerivative(self, a_idx, x1, x2, A):
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

        return res_a

    def FarFirstCosntraint(self, x_data, k_clusters, self.must_lnk,
                           cannnot_lnk_cons, distor_measure):

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
            if farthest_x_idx in self.must_lnk:
                for ml_idx in self.must_lnk[farthest_x_idx]:
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

    def ConsolidateAL(self, neibs_sets, x_data, self.must_lnk, distor_measure):

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
                if rnd_idx in self.must_lnk:
                    for ml_idx in self.must_lnk[rnd_idx]:
                        if ml_idx in neibs_sets[neib_idx] and rnd_idx not in neibs_sets[neib_idx]:
                            neibs_sets[neib_idx].append(rnd_idx)

        return neibs_sets


# The following function most probably won't be needed.
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
