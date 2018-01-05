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


# ##################################################################################### #
# Basic Distance Measures operator. Concarrent programming enamled wherever is possible #
# ##################################################################################### #

cpdef double [:, ::1] cosDa(double [:, ::1] m1, double [:, ::1] m2, double[::1] A):

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
                csdis_vect[i3, j3] =  2 * acos(csdis_vect[i3, j3]) / pi

    return csdis_vect


cpdef double [::1] cosDa_rpairs(double [:, ::1] m,
                                double[::1] A,
                                cnp.intp_t [:, ::1] mrp,
                                cnp.intp_t [::1] mrp_r):

    cdef:
        # Matrix index variables.
        Py_ssize_t im, i, j, i2, j2

        # Matrices dimentions intilized variables.
        Py_ssize_t mrpr_I = mrp_r.shape[0]
        Py_ssize_t m_J = m.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] v0_norms
        double [::1] v1_norms
        double [::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    v0_norms = cvarray(shape=(mrpr_I,), itemsize=sizeof(double), format="d")
    v1_norms = cvarray(shape=(mrpr_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(mrpr_I,), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for im in range(mrpr_I):
            v0_norms[im] = 0.0
            v1_norms[im] = 0.0
            csdis_vect[im] = 0.0

        # Calculating the Norms for the Vector Pairs.
        for i in prange(mrpr_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m_J):
                v0_norms[i] += m[mrp[mrp_r[i], 0], j] * m[mrp[mrp_r[i], 0], j] * A[j]
                v1_norms[i] += m[mrp[mrp_r[i], 1], j] * m[mrp[mrp_r[i], 1], j] * A[j]

            # Calculating the Square root of the sum
            v0_norms[i] = sqrt(v0_norms[i])
            v1_norms[i] = sqrt(v1_norms[i])

            # Preventing Division by Zero.
            if v0_norms[i] == 0.0:
                v0_norms[i] = 0.000001
            if v1_norms[i] == 0.0:
                v1_norms[i] = 0.000001


        # Calculating the cosine similarity.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i2 in prange(mrpr_I, schedule='guided'):

            for j2 in range(m_J):

                # Calculating the elemnt-wise sum of products distorted by A.
                csdis_vect[i2] += m[mrp[mrp_r[i2], 0], j2] * m[mrp[mrp_r[i2], 1], j2] * A[j2]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i2] = csdis_vect[i2] / (v0_norms[i2] * v1_norms[i2])

                # Getting Cosine Distance.
                csdis_vect[i2] =  2 * acos(csdis_vect[i2]) / pi

    return csdis_vect


cpdef double [::1] cosDa_v2r(double [::1] v,
                            double [:, ::1] m,
                            double[::1] A,
                            cnp.intp_t [::1] mr):

    cdef:
        # Matrix index variables.
        Py_ssize_t vi, mi, i, j, i2, j2

        # Matrices dimentions intilized variables.
        Py_ssize_t v_I = v.shape[0]
        Py_ssize_t mr_I = mr.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double v_norm = 0.0
        double [::1] m_norms
        double [::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    m_norms = cvarray(shape=(mr_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(mr_I,), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for mi in range(mr_I):
            m_norms[mi] = 0.0
            csdis_vect[mi] = 0.0

        # Calculating the Norms for the 2D matrix.
        for i in prange(mr_I, schedule='guided'):

            # Calculating Sum.
            for j in range(v_I):
                m_norms[i] += m[mr[i], j] * m[mr[i], j] * A[j]

            # Calculating the Square root of the sum
            m_norms[i] = sqrt(m_norms[i])

            # Preventing Division by Zero.
            if m_norms[i] == 0.0:
                m_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for vi in range(v_I):
            v_norm = v_norm + v[vi] * v[vi] * A[vi]

        # Preventing Division by Zero.
        if v_norm == 0.0:
            v_norm = 0.000001


        # Calculating the cosine similarity.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i2 in prange(mr_I, schedule='guided'):

            for j2 in range(v_I):
                csdis_vect[i2] += m[mr[i2], j2] * v[j2] * A[j2]

            # Normalizing with the products of the respective vector norms.
            csdis_vect[i2] = csdis_vect[i2] / (m_norms[i2] * v_norm)

            # Getting Cosine Distance.
            csdis_vect[i2] =  2 * acos(csdis_vect[i2]) / pi

    return csdis_vect


cpdef double cosDa_vect(double [::1] v1, double [::1] v2, double[::1] A):

    cdef:
        # Matrix index variables.
        Py_ssize_t i

        # Matrices dimentions intilized variables.
        Py_ssize_t v_I = v1.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double v1_norm = 0.0
        double v2_norm = 0.0
        double csdis = 0.0

        # Definding Pi constant.
        double pi = 3.14159265

    # The following operatsion taking place in the non-gil.
    with nogil:

        # Calculating the cosine similarity.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i in range(v_I):

            # Calculating the elemnt-wise sum of products distorted by A.
            csdis = csdis + v1[i] * v2[i] * A[i]

            # Calculating the Norms for the v1 and v2 vectors.
            v1_norm = v1_norm + v1[i] * v1[i] * A[i]
            v1_norm = sqrt(v1_norm)

            v2_norm = v2_norm + v2[i] * v2[i] * A[i]
            v2_norm = sqrt(v2_norm)

        # Preventing Division by Zero.
        if v1_norm == 0.0:
            v1_norm = 0.000001

        if v2_norm == 0.0:
            v2_norm = 0.000001

        # Normalizing with the products of the respective vector norms.
        csdis = csdis / (v1_norm * v2_norm)

        # Getting Cosine Distance.
        csdis =  2 * acos(csdis) / pi

    return csdis


cpdef double [:, ::1] cosD(double [:, ::1] m1, double [:, ::1] m2):

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
                csdis_vect[i, j] =  2 * acos(csdis_vect[i, j]) / pi

    return csdis_vect


cpdef double [:, ::1] mean_cosA(double [:, ::1] X,
                                cnp.intp_t [::1] clust_tags,
                                double [::1] A,
                                int k_clustz):
    """  mean_cosA method: It is calculating the centroids of the hyper-spherical clusters.
        Using the parametrized cosine mean as explained in the documentation.

    """

    cdef:
        double zero_val = 1e-15
        Py_ssize_t k, i, j, ip, jp, ip2, jp2
        Py_ssize_t X_J = X.shape[1]
        Py_ssize_t ct_I = clust_tags.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] mu_arr
        double [::1] xk_sum
        double [::1] xk_pnorms

    mu_arr = cvarray(shape=(k_clustz, X_J), itemsize=sizeof(double), format="d")
    xk_sum = cvarray(shape=(X_J,), itemsize=sizeof(double), format="d")
    xk_pnorms = cvarray(shape=(k_clustz,), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for i in range(k_clustz):
            for j in range(X_J):
                mu_arr[i, j] = 0.0

        for i in range(X_J):
            xk_sum[i] = 0.0

        for k in prange(k_clustz, schedule='guided'):

            # Summing up all the X data points for the current cluster. Equivalent to sum in axis-0.
            for jp in prange(X_J, schedule='guided'):
                for ip in range(ct_I):
                    # The i vector has k cluster-tag equal to the requested k the sum it up.
                    if clust_tags[ip] == k:
                        xk_sum[jp] += X[ip, jp]

            # Calculating denominator xk_pnorm(parametrized-norm) == ||Σ xi||(A).
            for ip2 in range(X_J):
                xk_pnorms[k] += sqrt(xk_sum[ip2] * A[ip2] * xk_sum[ip2])

            # Preveting division by 0.0
            if xk_pnorms[k] == 0.0:
                xk_pnorms[k] = 0.000001

            # Calculating the Centroid of the (assumed) hyper-sphear.
            for jp2 in range(X_J):
                mu_arr[k, jp2] = xk_sum[jp2] / xk_pnorms[k]

    return mu_arr


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


# ##################################################################################### #
# Basic Distance Measure operator. Concarrent programming enamled wherever is possible. #
# ##################################################################################### #

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


cpdef double [::1] pDerivative_seq_rpairs(double[::1] A,
                                          double [:, ::1] m,
                                          cnp.intp_t [:, ::1] mrp,
                                          cnp.intp_t [::1] mrp_r):
    cdef:
        # Matrix index variables.
        Py_ssize_t im, i, j, i2, i3, ai, j2

        # Matrices dimentions intilized variables.
        Py_ssize_t a_I = A.shape[0]
        Py_ssize_t mrpr_I = mrp_r.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] v0_norms
        double [::1] v1_norms
        double [::1] a_pDz_vect
        double x1x2dota

    # Creating the temporary cython arrays.
    v0_norms = cvarray(shape=(mrpr_I,), itemsize=sizeof(double), format="d")
    v1_norms = cvarray(shape=(mrpr_I,), itemsize=sizeof(double), format="d")
    a_pDz_vect = cvarray(shape=(a_I,), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for im in range(mrpr_I):
            v0_norms[im] = 0.0
            v1_norms[im] = 0.0

        for im in range(a_I):
            a_pDz_vect[im] = 0.0

        # Calculating the Norms for the Vector Pairs.
        for i in prange(mrpr_I, schedule='guided'):

            # Calculating Sum.
            for j in range(a_I):
                v0_norms[i] += m[mrp[mrp_r[i], 0], j] * m[mrp[mrp_r[i], 0], j] * A[j]
                v1_norms[i] += m[mrp[mrp_r[i], 1], j] * m[mrp[mrp_r[i], 1], j] * A[j]

            # Calculating the Square root of the sum
            v0_norms[i] = sqrt(v0_norms[i])
            v1_norms[i] = sqrt(v1_norms[i])

            # Preventing Division by Zero.
            if v0_norms[i] == 0.0:
                v0_norms[i] = 0.000001
            if v1_norms[i] == 0.0:
                v1_norms[i] = 0.000001


        # Calculating the Sequence of partial derivatives for every A array elements.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for j2 in prange(a_I, schedule='guided'):

            for i2 in range(mrpr_I):

                # Calculating the elemnt-wise sum of products distorted by A.
                # Note: x1x2dota = vdot(dot1d_ds(x1, A), x2)
                x1x2dota = 0.0
                for ai in range(a_I):
                    x1x2dota = x1x2dota + m[mrp[mrp_r[i2], 0], ai] * A[ai] * m[mrp[mrp_r[i2], 1], ai]

                # Calulating partial derivative for elemnt a_i of A array.
                a_pDz_vect[j2] = pDerivative(
                    m[mrp[mrp_r[i2], 0], j2], m[mrp[mrp_r[i2], 0], j2],
                    v0_norms[i2], v1_norms[i2], x1x2dota
                )

    return a_pDz_vect


cpdef double [::1] pDerivative_seq_mk2mr(double[::1] A,
                                         double [:, ::1] m1,
                                         double [:, ::1] m2,
                                         cnp.intp_t m1k,
                                         cnp.intp_t [::1] m2r):

        cdef:
            # Matrix index variables.
            Py_ssize_t im1, im2, a1, mi, mj, mi2, mj2, k, j, i, mj3

            # Matrices dimentions intilized variables.
            Py_ssize_t a_I = A.shape[0]
            Py_ssize_t m2r_I = m2r.shape[0]
            # Py_ssize_t m_J = m2r.shape[1]

            # MemoryViews for the cython arrays used for sotring the temporary and...
            # ...to be retured results.
            double [::1] a_pDz_vect
            double [::1] m1_norms
            double [::1] m2_norms
            double x1x2dota

        # Creating the temporary cython arrays.
        a_pDz_vect = cvarray(shape=(a_I,), itemsize=sizeof(double), format="d")
        m1_norms = cvarray(shape=(m1k,), itemsize=sizeof(double), format="d")
        m2_norms = cvarray(shape=(m2r_I,), itemsize=sizeof(double), format="d")

        # The following operatsion taking place in the non-gil and parallel...
        # ...openmp emviroment.
        with nogil, parallel():

            # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
            # ...in C garbage values can case floating point overflow, thus, peculiar results...
            # ...like NaN or incorrect calculatons.
            for im1 in range(m1k):
                m1_norms[im1] = 0.0

            for im2 in range(m2r_I):
                m2_norms[im2] = 0.0

            for a1 in range(a_I):
                a_pDz_vect[a1] = 0.0

            # Calculating the Norms for the first matrix.
            for mi in prange(m1k, schedule='guided'):

                # Calculating Sum.
                for mj in range(a_I):
                    m1_norms[mi] += m1[mi, mj] * m1[mi, mj] * A[mj]

                # Calculating the Square root of the sum
                m1_norms[mi] = sqrt(m1_norms[mi])

                # Preventing Division by Zero.
                if m1_norms[mi] == 0.0:
                    m1_norms[mi] = 0.000001


            # Calculating the Norms for the second matrix.
            for mi2 in prange(m2r_I, schedule='guided'):

                # Calculating distorted dot product.
                for mj2 in range(a_I):
                    m2_norms[mi2] += m2[m2r[mi2], mj2] * m2[m2r[mi2], mj2] * A[mj2]

                # Calculating the Square root of the sum
                m2_norms[mi2] = sqrt(m2_norms[mi2])

                # Preventing Division by Zero.
                if m2_norms[mi2] == 0.0:
                    m2_norms[mi2] = 0.000001


            # Calculating the Sequence of partial derivatives for every A array elements.
            # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
            for j in prange(a_I, schedule='guided'):

                for k in range(m1k):

                    for i in range(m2r_I):

                        # Calculating the elemnt-wise sum of products distorted by A.
                        # Note: x1x2dota = vdot(dot1d_ds(x1, A), x2)
                        x1x2dota = 0.0
                        for mj3 in range(a_I):
                            x1x2dota = x1x2dota + m1[k, mj3] * A[mj3] * m2[m2r[i], mj3]

                        # Calulating partial derivative for elemnt a_i of A array.
                        a_pDz_vect[j] = pDerivative(
                            m1[k, j], m2[m2r[i], j], m1_norms[k], m2_norms[i], x1x2dota
                        )

        return a_pDz_vect


# ######################################################################### #
# Below this line the Vector Operators are not used in this implementation. #
# ######################################################################### #

# Note: For interal usage in cython.
cdef double [:, ::1] dot2d_2d(double [:, ::1] m1, double [:, ::1] m2, cnp.intp_t [::1] m2r):

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
