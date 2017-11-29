# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

from cython.parallel import parallel, prange
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as cnp
cimport cython

cdef extern from "math.h":
    cdef double sqrt(double x) nogil
    cdef double acos(double x) nogil
    cdef double pow(double x, double y) nogil
    cdef double log (double x) nogil


cpdef double [:, ::1] cos2Da_rows(double [:, ::1] m1,
                                  double [:, ::1] m2,
                                  double[::1] A,
                                  int [::1] m1r,
                                  int [::1] m2r):

    cdef:
        # Matrix index variables.
        Py_ssize_t im1, im2, i, j, i2, j2, i3, j3, k

        # Matrices dimentions intilized variables.
        Py_ssize_t m1r_I = m1r.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2r_I = m2r.shape[0]
        Py_ssize_t m2_J = m2.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1r_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2r_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(m1r_I, m2r_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for im1 in range(m1r_I):
            m1_norms[im1] = 0.0

        for im2 in range(m2r_I):
            m2_norms[im2] = 0.0

        for im1 in range(m1r_I):
            for im2 in range(m2r_I):
                csdis_vect[im1, im2] = 0.0

        # Calculating the Norms for the first matrix.
        for i in prange(m1r_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[m1r[i], j] * m1[m1r[i], j] * A[j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

            # Preventing Division by Zero.
            if m1_norms[i] == 0.0:
                m1_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for i2 in prange(m2r_I, schedule='guided'):

            # Calculating Sum.
            for j2 in range(m2_J):
                m2_norms[i2] += m2[m2r[i2], j2] * m2[m2r[i2], j2] * A[j2]

            # Calculating the Square root of the sum
            m2_norms[i2] = sqrt(m2_norms[i2])

            # Preventing Division by Zero.
            if m2_norms[i2] == 0.0:
                m2_norms[i2] = 0.000001


        # Calculating the cosine similarity.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i3 in prange(m1r_I, schedule='guided'):

            for j3 in range(m2r_I):

                # Calculating the elemnt-wise sum of products distorted by A.
                for k in range(m1_J):
                    csdis_vect[i3, j3] += m1[m1r[i3], k] * m2[m2r[j3], k] * A[k]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i3, j3] = csdis_vect[i3, j3] / (m1_norms[i3] * m2_norms[j3])

                # Getting Cosine Distance.
                csdis_vect[i3, j3] =  acos(csdis_vect[i3, j3]) / pi

    return csdis_vect


cpdef double [:, ::1] cos2Da(double [:, ::1] m1, double [:, ::1] m2, double[::1] A):

    cdef:
        # Matrix index variables.
        Py_ssize_t i, j, i2, j2, i3, j3, k, im1, im2

        # Matrices dimentions intilized variables.
        Py_ssize_t m1_I = m1.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]
        Py_ssize_t m2_J = m2.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(m1_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for im1 in range(m1_I):
            m1_norms[im1] = 0.0

        for im2 in range(m2_I):
            m2_norms[im2] = 0.0

        for im1 in range(m1_I):
            for im2 in range(m2_I):
                csdis_vect[im1, im2] = 0.0

        # Calculating the Norms for the first matrix.
        for i in prange(m1_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[i, j] * m1[i, j] * A[j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

            # Preventing Division by Zero.
            if m1_norms[i] == 0.0:
                m1_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for i2 in prange(m2_I, schedule='guided'):

            # Calculating Sum.
            for j2 in range(m2_J):
                m2_norms[i2] += m2[i2, j2] * m2[i2, j2] * A[j2]

            # Calculating the Square root of the sum
            m2_norms[i2] = sqrt(m2_norms[i2])

            # Preventing Division by Zero.
            if m2_norms[i2] == 0.0:
                m2_norms[i2] = 0.000001


        # Calculating the cosine similarity.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i3 in prange(m1_I, schedule='guided'):

            for j3 in range(m2_I):

                # Calculating the elemnt-wise sum of products distorted by A.
                for k in range(m1_J):
                    csdis_vect[i3, j3] += m1[i3, k] * m2[j3, k] * A[k]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i3, j3] = csdis_vect[i3, j3] / (m1_norms[i3] * m2_norms[j3])

                # Getting Cosine Distance.
                csdis_vect[i3, j3] =  acos(csdis_vect[i3, j3]) / pi

    return csdis_vect


cpdef double [:, ::1] cosdis_2d(double [:, ::1] m1, double [:, ::1] m2):

    cdef:
        # Matrix index variables.
        Py_ssize_t i, j, k, iz, jz

        # Matrices dimentions intilized variables.
        Py_ssize_t m1_I = m1.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]
        Py_ssize_t m2_J = m2.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(m1_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for iz in range(m1_I):
            m1_norms[iz] = 0.0

        for iz in range(m2_I):
            m2_norms[iz] = 0.0

        for iz in range(m1_I):
            for jz in range(m2_I):
                csdis_vect[iz, jz] = 0.0

        # Calculating the Norms for the first matrix.
        for i in prange(m1_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[i, j] * m1[i, j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

            # Preventing Division by Zero.
            if m1_norms[i] == 0.0:
                m1_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for i in prange(m2_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m2_J):
                m2_norms[i] += m2[i, j] * m2[i, j]

            # Calculating the Square root of the sum
            m2_norms[i] = sqrt(m2_norms[i])

            # Preventing Division by Zero.
            if m2_norms[i] == 0.0:
                m2_norms[i] = 0.000001

        # Calculating the cosine distances product.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i in prange(m1_I, schedule='guided'):

            for j in range(m2_I):

                # Calculating the elemnt-wise sum of products.
                for k in range(m1_J):
                    csdis_vect[i, j] += m1[i, k] * m2[j, k]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i, j] = csdis_vect[i, j] / (m1_norms[i] * m2_norms[j])

                # Getting Cosine Distance.
                csdis_vect[i, j] =  acos(csdis_vect[i, j]) / pi

    return csdis_vect


cpdef double [:, ::1] eudis_2d(double [:, ::1] m1, double [:, ::1] m2):

    cdef:
        # Matrix index variables.
        Py_ssize_t i, j, k, iz, jz

        # Matrices dimentions intilized variables.
        Py_ssize_t m1_I = m1.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] eudis_vect

    # Creating the temporary cython arrays.
    eudis_vect = cvarray(shape=(m1_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for iz in range(m1_I):
            for jz in range(m2_I):
                eudis_vect[iz, jz] = 0.0

        # Calculating the euclidian distances amogst all vectros of both matrices.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i in prange(m1_I, schedule='guided'):

            for j in range(m2_I):

                # Calculating the elemnt-wise sum of products.
                for k in range(m1_J):
                    eudis_vect[i, j] += (m1[i, k] - m2[j, k]) * (m1[i, k] - m2[j, k])

                # Normalizing with the products of the respective vector norms.
                eudis_vect[i, j] = sqrt(eudis_vect[i, j])

    return eudis_vect


# Note: For interal usage in cython.
cdef inline double vdot(double [::1] v1, double [::1] v2):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v1.shape[0]

        # Initializing the result variable.
        double res = 0.0

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            res += v1[i] * v2[i]

    return res

# Note: For interal usage in cython.
cdef inline double [::1] dot1d_ds(double [::1] v, double [::1] m):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] res

    # Creating the numpy.array for results and its memory view
    res = cvarray(shape=(I,), itemsize=sizeof(double), format="d")

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            res[i] = v[i] * m[i]

    return res

"""
# Note: Make it cdef if only for interal usage in cython.
cpdef double [:, ::1] dot2d(double [:, ::1] m1, double [:, ::1] m2):

    # if m1.shape[1] != m2.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Matrix index variables.
    cdef:
        unsigned int i, j, k
        unsigned int I = m1.shape[0]
        unsigned int J = m2.shape[1]
        unsigned int K = m1.shape[1]

    # Creating the array for results and its memory view
    double [:, ::1] res = np.zeros((I, J), dtype=np.float)

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                for k in range(K):
                    res[i, j] += m1[i, k] * m2[k, j]

    return res
"""

# Note: For interal usage in cython.
cdef double [:, ::1] dot2d_2d(double [:, ::1] m1, double [:, ::1] m2, int [::1] m2r):

    # Matrix index variables.
    cdef:
        Py_ssize_t i, j, ir
        Py_ssize_t I = m1.shape[0]
        Py_ssize_t J = m2.shape[1]
        # Py_ssize_t K = m1.shape[1]
        Py_ssize_t IR = m2r.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] res

    # Creating the numpy.array for results and its memory view
    res = cvarray(shape=(I, J), itemsize=sizeof(double), format="d")

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                for ir in range(IR):
                    res[i, j] += m1[i, ir] * m2[m2r[ir], j]

    return res


# Note: For interal usage in cython.
cdef double [:, ::1] dot2d_ds(double [:, ::1] m1, double [::1] m2):

    # if m1.shape[1] != m2.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Matrix index variables.
    cdef:
        Py_ssize_t i, j
        Py_ssize_t I = m1.shape[0]
        Py_ssize_t J = m2.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] res

    # Creating the array for results and its memory view
    res = cvarray(shape=(I, J), itemsize=sizeof(double), format="d")

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                res[i, j] = m1[i, j] * m2[j]

    return res


# Note: For interal usage in cython.
cdef inline void sum_axs0(double [::1] res, double [:, ::1] m,
                          cnp.intp_t [::1] clust_tags,
                          cnp.intp_t k,
                          double zero_val) nogil:

    # Matrix index variables.
    cdef:
        Py_ssize_t i, j, iz
        Py_ssize_t ct_I = clust_tags.shape[0]
        Py_ssize_t J = m.shape[1]

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for iz in range(J):
            res[iz] = zero_val

        for j in prange(J, schedule='guided'):
            for i in range(ct_I):
                # The i vector has k cluster-tag equal to the requested k the sum it up.
                if clust_tags[i] == k:
                    res[j] += m[i, j]


# Note: Make it cdef if only for interal usage in cython.
cpdef double [::1] get_diag(double [:, ::1] m):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = m.shape[0]

        # Creating the numpy.array for results and its memory view
        double [::1] res = np.zeros((I), dtype=np.float)

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            # The idxs array is giving the actual row index of the data matrix...
            # ...to be summed up.
            res[i] += m[i, i]

    return res

"""
# Note: Make it cdef if only for interal usage in cython.
cpdef double [:, ::1] div2d_vv(double [:, ::1] m, double [::1] v):

    # _vv stands for Vertical Vector.

    # if m.shape[0] != v.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. 2D matrix cannot be dived by the vertical vector")

    # Matrix index variables.
    cdef:
        unsigned int i0, i1, j
        unsigned int I0 = m.shape[0]
        unsigned int I1 = m.shape[1]
        unsigned int J = v.shape[0]

    # Creating the numpy.array for results and its memory view
    double [:, ::1] res = np.zeros((I0, I1), dtype=np.float)

    #
    with nogil, parallel():
        for i0 in prange(I0, schedule='guided'):
            for i1 in range(I1):
                res[i0, i1] = m[i0, i1] / v[i0]

    return res
"""

# Note: Make it cdef if only for interal usage in cython.
cpdef double [::1] vdiv_num(double [::1] v, double num):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v.shape[0]

        # Creating the numpy.array for results and its memory view
        double [::1] res = np.zeros((I), dtype=np.float)

    #
    with nogil:
        for i in range(I):
            res[i] = v[i] / num

    return res


# Note: Make it cdef if only for interal usage in cython.
cpdef double [::1] vsqrt(double [::1] v):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v.shape[0]

        # Creating the numpy.array for results and its memory view
        double [::1] res = np.zeros((I), dtype=np.float)

    #
    with nogil:
        for i in range(I):
            res[i] = sqrt(v[i])

    return res


cpdef double partial_derivative(cnp.intp_t a_idx, double [::1] x1, double [::1] x2, double [::1] A):
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
    cdef:
        double x1_pnorm
        double x2_pnorm
        double x1x2dota

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = sqrt(vdot(dot1d_ds(x1, A), x1))
    x2_pnorm = sqrt(vdot(dot1d_ds(x2, A), x2))

    # Claculating dot_A product of x1 and x2.
    x1x2dota = vdot(dot1d_ds(x1, A), x2)

    return pDerivative(x1[a_idx], x2[a_idx], x1_pnorm, x2_pnorm, x1x2dota)


cdef inline double pDerivative(double x1_ai,
                               double x2_ai,
                               double x1_aipn,
                               double x2_aipn,
                               double x1x2dota) nogil:
    """ Partial Derivative: This method is calculating the partial derivative of a specific
        parameter given the proper vectors. That is, for the cosine distance is a x_i with the
        centroid vector (mu) of the cluster where x_i is belonging into. As for the constraint
        violations is the x_1 and x_2 of a specific pair of constraints each time this method
        is called.
        **for detail see documentation.
    """
    cdef double res_a = 0.0

    # Note: x1x2dota = vdot(dot1d_ds(x1, A), x2)
    res_a = (
                (x1_ai * x2_ai * x1_aipn * x2_aipn) -
                (
                    x1x2dota *
                    (
                        (
                            pow(x1_ai, 2.0) * pow(x2_aipn, 2.0) +
                            pow(x2_ai, 2.0) * pow(x1_aipn, 2.0)
                        ) / (2 * x1_aipn * x2_aipn)
                    )
                )
            ) / (pow(x1_aipn, 2.0) * pow(x2_aipn, 2.0))

    return res_a


cpdef double [::1] pDerivative_seq_rows(double[::1] A,
                                        double [:, ::1] m1,
                                        double [:, ::1] m2,
                                        int [::1] m1r,
                                        int [::1] m2r):
    cdef:
        # Matrices dimentions intilized variables.
        Py_ssize_t a_I = A.shape[0]
        Py_ssize_t m1r_I = m1r.shape[0]
        Py_ssize_t m2r_I = m2r.shape[0]

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1r_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2r_I,), itemsize=sizeof(double), format="d")
    a_pDz_vect = cvarray(shape=(a_I,), itemsize=sizeof(double), format="d")

    return _pDerivative_seq_rows(A, m1, m2, m1r, m2r, m1_norms, m2_norms, a_pDz_vect)


cdef double [::1] _pDerivative_seq_rows(double[::1] A,
                                        double [:, ::1] m1,
                                        double [:, ::1] m2,
                                        int [::1] m1r,
                                        int [::1] m2r,
                                        double [::1] m1_norms,
                                        double [::1] m2_norms,
                                        double [::1] a_pDz_vect) nogil:

    cdef:
        # Matrix index variables.
        Py_ssize_t im1, im2, a1, i, j, i2, j2, i3, j3, k, k2

        # Matrices dimentions intilized variables.
        Py_ssize_t a_I = A.shape[0]
        Py_ssize_t m1r_I = m1r.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2r_I = m2r.shape[0]
        Py_ssize_t m2_J = m2.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        # double [::1] m1_norms
        # double [::1] m2_norms
        # double [::1] a_pDz_vect
        double x1x2dota

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for im1 in range(m1r_I):
            m1_norms[im1] = 0.0

        for im2 in range(m2r_I):
            m2_norms[im2] = 0.0

        for a1 in range(m1r_I):
            a_pDz_vect[a1] = 0.0

        # Calculating the Norms for the first matrix.
        for i in prange(m1r_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[m1r[i], j] * m1[m1r[i], j] * A[j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

            # Preventing Division by Zero.
            if m1_norms[i] == 0.0:
                m1_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for i2 in prange(m2r_I, schedule='guided'):

            # Calculating distorted dot product.
            for j2 in range(m2_J):
                m2_norms[i2] += m2[m2r[i2], j2] * m2[m2r[i2], j2] * A[j2]

            # Calculating the Square root of the sum
            m2_norms[i2] = sqrt(m2_norms[i2])

            # Preventing Division by Zero.
            if m2_norms[i2] == 0.0:
                m2_norms[i2] = 0.000001


        # Calculating the Sequence of partial derivatives for every A array elements.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for j3 in prange(a_I, schedule='guided'):

            for i3 in range(m1r_I):

                for k in range(m2r_I):

                    # Calculating the elemnt-wise sum of products distorted by A.
                    # Note: x1x2dota = vdot(dot1d_ds(x1, A), x2)
                    x1x2dota = 0.0
                    for k2 in range(m1_J):
                        x1x2dota = x1x2dota + m1[m1r[i3], k2] * A[k2] * m2[m2r[k], k2]

                    # Calulating partial derivative for elemnt a_i of A array.
                    a_pDz_vect[j3] = pDerivative(
                        m1[i3, j3], m2[k, j3], m1_norms[i3], m2_norms[k], x1x2dota
                    )

    return a_pDz_vect

cpdef double [::1] pDerivative_seq_one2many(double[::1] A,
                                        double [:, ::1] m1,
                                        double [:, ::1] m2,
                                        int [::1] m1r,
                                        int [::1] m2r):

        cdef:
            # Matrix index variables.
            Py_ssize_t i1, i2, i

            # Matrices dimentions intilized variables.
            Py_ssize_t a_I = A.shape[0]
            Py_ssize_t m1r_I = m1r.shape[0]
            Py_ssize_t m2r_I = m2r.shape[0]

            # MemoryViews for the cython arrays used for sotring the temporary and...
            # ...to be retured results.
            double [::1] a_pDz_vect
            double [::1] tmp_vect
            int [::1] one_elem_cp
            double x1x2dota = 0.0

        # Creating the temporary cython arrays.
        a_pDz_vect = cvarray(shape=(a_I,), itemsize=sizeof(double), format="d")
        m1_norms = cvarray(shape=(m1r_I,), itemsize=sizeof(double), format="d")
        m2_norms = cvarray(shape=(m2r_I,), itemsize=sizeof(double), format="d")
        tmp_a_pDz_vect = cvarray(shape=(a_I,), itemsize=sizeof(double), format="d")

        # The following operatsion taking place in the non-gil and parallel...
        # ...openmp emviroment.
        with nogil:

            # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
            # ...in C garbage values can case floating point overflow, thus, peculiar results...
            # ...like NaN or incorrect calculatons.
            for i in range(a_I):
                a_pDz_vect[i] = 0.0

            for i1 in range(m1r_I):

                # Calculating the partial derivatives form each centroid.

                # NOTE: For some reason this copy is required for passing one elemnt memory...
                # ...view slice to the cython funtion.
                one_elem_cp[:] = m1r[i1]

                tmp_vect = _pDerivative_seq_rows(
                    A, m1, m2, one_elem_cp, m2r, m1_norms, m2_norms, tmp_a_pDz_vect
                )

                # Summing up the.
                for i2 in range(a_I):
                    a_pDz_vect[i2] += tmp_vect[i2]

        return a_pDz_vect


cpdef double [:, ::1] mean_cosA(double [:, ::1] X,
                                cnp.intp_t [::1] clust_tags,
                                double [::1] A,
                                int k_clustz):
    """  mean_cosA method: It is calculating the centroids of the hyper-spherical clusters.
        Using the parametrized cosine mean as explained in the documentation.

    """

    cdef:
        double zero_val = 1e-15
        Py_ssize_t k, i, j, ip, jp
        Py_ssize_t X_J = X.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] mu_arr
        double [::1] xk_sum
        double xk_pnorm

    mu_arr = cvarray(shape=(k_clustz, X_J), itemsize=sizeof(double), format="d")
    xk_sum = cvarray(shape=(X_J,), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil:

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for i in range(k_clustz):
            for j in range(X_J):
                mu_arr[i, j] = 0.0

        for k in range(k_clustz):

            # Summing up all the X data points for the current cluster.
            # NOTE: Returning results buffer is given as argument for enabling NoGIL parrallel...
            sum_axs0(xk_sum, X, clust_tags, k, zero_val)

            # Calculating denominator xk_pnorm(parametrized-norm) == ||Σ xi||(A).
            for ip in range(X_J):
                xk_pnorm += sqrt(xk_sum[ip] * A[ip] * xk_sum[ip])

            # Calculating the Centroid of the (assumed) hyper-sphear.
            for jp in range(X_J):
                mu_arr[k, jp] = xk_sum[jp] / xk_pnorm

    return mu_arr

"""
cpdef NormPart(self, double [:, ::1] x_data_subset):
     The von Mises and von Mises - Fisher Logarithmic Normalization partition function:...
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
"""
