# The following comment enables the use of utf-8 within the script.
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats as sps
import random as rnd
import scipy.special as special
import warnings

# from .voperators import cy as vop


class DSSVEM(object):
    """ Distance Space Semi-Supervised Expectation Maximization: A Semi-supervised
        clustering algorithm. That is, a small set the given data is known. The algorithm is 
        transforming the problem to the distance space and then is, optimizing the the 
        offest-coefficients for forming the clusters. 
        
        It can work also in an open-set framework by setting the number of clusters/classes 
        explicitly by adding at least one more to the number of the expected k-clusters.
        That is, when some samples found to be far apart form all the known sets are assumed to be 
        unknown or outages. Thus we expect the algorithm to let them in a different cluster.

        Initialization arguments
        ------------------------
        More details later...

    """

    def __init__(self,
                 # feat_size,  # The feature space size.
                 # sim_lst=['cos', 'eucl', 'minmax', 'jac', 'hamm', 'corr'], # Similarities type list.
                 # offset_dcoefs=None,  # Vector of offset distortion coefficients. None is implicit.
                 # ml_wg=1.0,  # Must-link weighting coefficient.
                 # cl_wg=1.0,   # Cannot-link weighting coefficient.
                 # fs_rnd_itrs=10, # Number of random selection iterations.
                 # max_iter=10,  # Max number of clustering iterations.
                 # lrn_rate=0.001,  # Learning rate of the EM algorithm.
                 # sray_sigma=0.5,  # Not sure to be used 
                 # icm_max_i=10, 
                 # enable_norm=False  # Allowing normalization.
                 ):
        
        self.W = np.array([np.random.rand()]*10)


    def fit(self):

        ctr = 12

        self.trg = (ctr - 1.0) / np.float32(ctr)

    def GradientDescent(self, X):

        self.W

        df = lambda x: -(h * (1/(1+np.exp())) * () * () *  )

        next_x = 6  # We start the search at x=6
        gamma = 0.01  # Step size multiplier
        precision = 0.0001  # Desired precision of result
        max_iters = 10000  # Maximum number of iterations

        # Derivative function
        df = lambda x: 4 * x**3 - 9 * x**2

        for i in range(max_iters):
            current_x = next_x
            next_x = current_x - gamma * df(current_x)
            step = next_x - current_x
            print next_x
            if abs(step) <= precision:
                break

        print("Minimum at", next_x)

        # The output for the above will be something like
        # "Minimum at 2.2499646074278457"

if __name__ == '__main__':

    print "OK"

    dss = DSSVEM()

    dss.GradientDescent()