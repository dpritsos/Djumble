

import numpy as np
import scipy as sp


def HMRFKmeans(x_arr, costr, const_violat_w, dist_measure):
    """HMRF Kmeans: A Semi-supervised clustering algorithm based on Hidden Markov Random Fields
        Clustering model optimised by Expectation Maximisation (EM) algorithm with Hard clustering
        constraints, i.e. a Kmeans Semi-supervised clustering variant.
    """

    k_centroids = init_cluster()


def ICM():
    """ICM: Iterated Conditional Modes (for the E-Step)
    """

    pass


def ExploreAL():
    """Explore Step for the Active Learning Phase
    """
    pass


def ConsolidateAL():
    """Consolidate Step for the Active Learning Phase
    """
    pass


def InitClustering():
    pass


def CosDistPar(x1, x2, distor_params):
    """CosDistPar: Cosine Distance with distortion parameters based on 'Soft Cosine Measure' where
        a weighting schema is the distortion parameters diagonal matrix A. Note that A matrix
        (diagonal) is expected as vector argument in this function.
    """

    x1 = sp.matrix(x1)
    x2 = sp.matrix(x2)
    A = sp.diag(distor_params)

    return 1 - (x1 * A * x2.T / (np.linalg.norm(x1) * np.linalg.norm(x2)))


def JObjCosDM(x1, x2, constr_pnlty_weig, distor_params, learning_rate):
    """JObjCosDM: J Objective function for Cosine distortion measure. It cannot very generic
        because the gradient decent (partial derivative) calculation should be applied which they
        are totally dependent on the distortion measure, here Cosine Distance.

    """

    "Phi_max depends on the distortion measure"
    

    pass


if __name__ == '__main__':

    x1 = np.array([1, 1, 1, 1], dtype=np.float32)
    x2 = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    dA = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    print CosDistPar(x1, x2, dA)