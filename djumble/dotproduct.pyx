# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
cimport numpy as cnp
cimport cython


cpdef double [:, ::1] dot2d(double [:, ::1] m1, double [:, ::1] m2):

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


cpdef double [::1] vdot(double [::1] v1, double [::1] v2):

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


cpdef double [:, ::1] dot2d_ds(double [:, ::1] m1, double [::1] m2):

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


cpdef double [::1] dot1d_ds(double [::1] v, double [::1] m):

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


cpdef double [::1] sum_axs0(double [:, ::1] m, cnp.intp_t [::1] idxs):

    # Matrix index variables.
    cdef unsigned int i, j
    cdef unsigned int I = idxs.shape[0]
    cdef unsigned int J = m.shape[1]

    # Creating the numpy.array for results and its memory view
    cdef double [::1] res = np.zeros((J), dtype=np.float)

    # Calculating the dot product.
    with nogil:
        for j in range(J):
            for i in range(I):
                # The idxs array is giving the actual row index of the data matrix...
                # ...to be summed up.
                res[j] += m[idxs[i], j]

    return res

cpdef double [::1] get_diag(double [:, ::1] m):

    # Matrix index variables.
    cdef unsigned int i
    cdef unsigned int I = m.shape[0]

    # Creating the numpy.array for results and its memory view
    cdef double [::1] res = np.zeros((I), dtype=np.float)

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            # The idxs array is giving the actual row index of the data matrix...
            # ...to be summed up.
            res[i] += m[i, i]

    return res
