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


# Note: Make it cdef if only for interal usage in cython.
cpdef double vdot(double [::1] v1, double [::1] v2):

    # if v1.shape[0] != v2.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

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

        # Creating the numpy.array for results and its memory view
        double [:, ::1] res = np.zeros((I, J), dtype=np.float)

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                for k in range(K):
                    res[i, j] += m1[i, k] * m2[k, j]

    return res


# Note: Make it cdef in only for Cython interal usage.
cpdef double [:, ::1] dot2d_2d(double [:, :] m1, double [:, :] m2):

    # if m1.shape[1] != m2.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Matrix index variables.
    cdef:
        unsigned int i0, j0, i1, j1
        unsigned int I0 = m1.shape[0]
        unsigned int I1 = m1.shape[1]
        # cdef unsigned int J0 = m2.shape[0]
        unsigned int J1 = m2.shape[1]

    # Creating the numpy.array for results and its memory view
    cdef double [:, ::1] res = np.zeros((I0, J1), dtype=np.float)

    # Calculating the dot product.
    with nogil, parallel():
        for i0 in prange(I0, schedule='guided'):
            for j1 in range(J1):
                for i1 in range(I1):
                    res[i0, j1] += m1[i0, i1] * m2[i1, j1]

    return res


# Note: Make it cdef if only for interal usage in cython.
cpdef double [:, ::1] dot2d_ds(double [:, ::1] m1, double [::1] m2):

    # if m1.shape[1] != m2.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Matrix index variables.
    cdef:
        unsigned int i, j
        unsigned int I = m1.shape[0]
        unsigned int J = m2.shape[0]

        # Creating the numpy.array for results and its memory view
        double [:, ::1] res = np.zeros((I, J), dtype=np.float)

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                res[i, j] = m1[i, j] * m2[j]

    return res


# Note: Make it cdef if only for interal usage in cython.
cpdef double [::1] dot1d_ds(double [::1] v, double [::1] m):

    # if v.shape[0] != m.shape[0]:
    #     raise Exception("Matrix dimensions mismatch. Dot product cannot be computed.")

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v.shape[0]

        # Creating the numpy.array for results and its memory view
        double [::1] res = np.zeros((I), dtype=np.float)

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            res[i] = v[i] * m[i]

    return res


# Note: Make it cdef if only for interal usage in cython.
cpdef double [::1] sum_axs0(double [:, ::1] m, cnp.intp_t [::1] idxs):

    # Matrix index variables.
    cdef:
        unsigned int i, j
        unsigned int I = idxs.shape[0]
        unsigned int J = m.shape[1]

        # Creating the numpy.array for results and its memory view
        double [::1] res = np.zeros((J), dtype=np.float)

    # Calculating the dot product.
    with nogil:
        for j in range(J):
            for i in range(I):
                # The idxs array is giving the actual row index of the data matrix...
                # ...to be summed up.
                res[j] += m[idxs[i], j]

    return res


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


cpdef double PartialDerivative(cnp.intp_t a_idx, double [::1] x1, double [::1] x2, double [::1] A):
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
    cdef double res_a = 0.0

    # Calculating parametrized Norms ||Σ xi||(A)
    x1_pnorm = sqrt(vdot(dot1d_ds(x1, A), x1))
    x2_pnorm = sqrt(vdot(dot1d_ds(x2, A), x2))

    res_a = (
                (x1[a_idx] * x2[a_idx] * x1_pnorm * x2_pnorm) -
                (
                    vdot(dot1d_ds(x1, A), x2) *
                    (
                        (
                            pow(x1[a_idx], 2.0) * pow(x2_pnorm, 2.0) +
                            pow(x2[a_idx], 2.0) * pow(x1_pnorm, 2.0)
                        ) / (2 * x1_pnorm * x2_pnorm)
                    )
                )
            ) / (pow(x1_pnorm, 2.0) * pow(x2_pnorm, 2.0))

    return res_a
