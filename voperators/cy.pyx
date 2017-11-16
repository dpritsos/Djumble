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
        Py_ssize_t i, j
        Py_ssize_t I = m1.shape[0]
        Py_ssize_t J = m2.shape[0]

        # Creating the numpy.array for results and its memory view
        double [:, ::1] res = np.zeros((I, J), dtype=np.float)

    # Calculating the dot product.
    with nogil, parallel():
        for i in prange(I, schedule='guided'):
            for j in range(J):
                res[i, j] = m1[i, j] * m2[j]

    return res


# Note: For interal usage in cython.
cdef inline double [::1] dot1d_ds(double [::1] v, double [::1] m):

    # Matrix index variables.
    cdef:
        unsigned int i
        unsigned int I = v.shape[0]

    # Creating the numpy.array for results and its memory view
    res = cvarray(shape=(I,), itemsize=sizeof(double), format="d")

    # Calculating the dot product.
    with nogil:
        for i in range(I):
            res[i] = v[i] * m[i]

    return res


# Note: For interal usage in cython.
cdef inline double [::1] sum_axs0(double [:, ::1] m,
                           cnp.intp_t [::1] clust_tags,
                           cnp.intp_t k,
                           double zero_val):

    # Matrix index variables.
    cdef:
        Py_ssize_t i, jm iz
        Py_ssize_t ct_I = clust_tags.shape[0]
        Py_ssize_t J = m.shape[1]

        # Creating the numpy.array for results and its memory view
        res = cvarray(shape=(J,), itemsize=sizeof(double), format="d")

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


cpdef double [:, ::1] mean_cosA(double [:, ::1] X,
                                double[::1] clust_tags,
                                double[::1] A
                                int k_clustz):
    """  mean_cosA method: It is calculating the centroids of the hyper-spherical clusters.
        Using the parametrized cosine mean as explained in the documentation.

    """

    cdef:
        double zero_val = 1e-15
        Py_ssize_t i, k, imu, jmu
        Py_ssize_t X_J = X.shape[1]
        double k_pnorm
        double [::1] xk_sum

    mu_arr = cvarray(shape=(k_clustz, X_J), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for imu in range(k_clustz):
            for jmu in range(xdata_J):
                mu_arr[imu, jmu] = 0.0


        for k in prange(k_clustz, schedule='guided'):

            # Summing up all the X data points for the current cluster.
            xk_sum = sum_axs0(X, clust_tags, k, zero_val)

            # Calculating denominator xk_pnorm(parametrized-norm) == ||Σ xi||(A).
            xk_pnorm = sqrt(vdot(dot1d_ds(xk_sum, A), xk_sum))

            # Calculating the Centroid of the (assumed) hyper-sphear. Then appended to the mu list.
            for j in range(X_J):
                mu_arr = xk_sum[i] / xk_pnorm


    return mu_arr


cpdef NormPart(self, double [:, ::1] x_data_subset):
    """ The von Mises and von Mises - Fisher Logarithmic Normalization partition function:...
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

    """

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
