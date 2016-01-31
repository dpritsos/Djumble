# -*- coding: utf-8 -*-
# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
cimport numpy as cnp

# Creating the numpy.array for results and its memory view
# res = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float)

def dot(double [:, ::1] m1, double [:, ::1] m2, double [:, ::1] res):

    # if m1.shape[1] != m2.shape[0]:
    #    raise MemoryError()

    # Matrix index variables.
    cdef unsigned int i, j, k
    cdef unsigned int I = res.shape[0]
    cdef unsigned int J = res.shape[1]
    cdef unsigned int K = m1.shape[1]

    # Matrix values variables.
    cdef double m1v, m2v

    # Calculating the dot product.
    with nogil:
        for i in xrange(I):
            for j in xrange(J):
                for k in xrange(K):
                    res[i, j] += m1[i, k] * m2[k, j]

    return res


def dot_sparse_diag(double [:, ::1] m1, double [:, ::1] m2,
                    cnp.intp_t [::1] m1dims, cnp.intp_t [::1] m2dims):
    pass
