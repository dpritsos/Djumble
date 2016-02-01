# -*- coding: utf-8 -*-

# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
cimport numpy as cnp


def dot(double [:, ::1] m1, double [:, ::1] m2):

    if m1.shape[1] != m2.shape[0]:
        raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Creating the numpy.array for results and its memory view
    cdef double [:, ::1] res = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float)

    # Matrix index variables.
    cdef unsigned int i, j, k
    cdef unsigned int I = res.shape[0]
    cdef unsigned int J = res.shape[1]
    cdef unsigned int K = m1.shape[1]

    # Calculating the dot product.
    with nogil:
        for i in xrange(I):
            for j in xrange(J):
                for k in xrange(K):
                    res[i, j] += m1[i, k] * m2[k, j]

    return res


def dot_sparse_diag(double [:, ::1] m1, double [:, ::1] m2,
                    cnp.intp_t [::1] m1dims, cnp.intp_t [::1] m2dims):

    if m1dims[1] != m2dims[0]:
        raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")
