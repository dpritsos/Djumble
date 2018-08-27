import numpy as np
import sys
import pstats
import cProfile
import StringIO

sys.path.append('../djumble/')
import dsmeasures as dop

x1 = np.array([1.0, 0.4, 0.6, 0.5, 0.6, 0.7, 0.6], dtype=np.float)
x2 = np.array([0.0, 1.0, 0.0, 0.9, 0.1, 0.2, 0.3], dtype=np.float)
a = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float)


A = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float)

x1_norm = x1[:] / np.max(x1)
x2_norm = x2[:] / np.max(x2)
# print x1_norm
# print x2_norm

x1_cosnorm = x1 / np.sqrt(dop.vdot(dop.dot1d_ds(x1, A), x1))
x2_cosnorm = x2 / np.sqrt(dop.vdot(dop.dot1d_ds(x2, A), x2))
# print x1_cosnorm
# print x2_cosnorm

xsum = np.sum(np.vstack((x1, x2)), axis=0)
print xsum
xmean = xsum / np.sqrt(dop.vdot(dop.dot1d_ds(xsum, A), xsum))
print xmean

print dop.vdot(dop.dot1d_ds(xmean, A), x2_cosnorm)
print dop.vdot(
    dop.dot1d_ds(xmean, A), x2_cosnorm) /
    (
        np.sqrt(dop.vdot(dop.dot1d_ds(xmean, A), xmean)) *
        np.sqrt(dop.vdot(dop.dot1d_ds(x2_cosnorm, A), x2_cosnorm)
    )
)
# print np.dot(x1_cosnorm, np.matrix(x2_cosnorm).T)

print np.asarray(dop.sum_axs0(a, np.array([0, 1, 2]))) / np.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0])
res_a_x_aT = dop.dot2d_2d(a, a.T)
print np.asarray(res_a_x_aT)
print np.asarray(dop.get_diag(res_a_x_aT))
print np.asarray(dop.div2d_vv(a, dop.get_diag(res_a_x_aT)))
print np.asarray(dop.div2d_vv(a, np.array([1., 2., 3.])))
print np.asarray(dop.vdiv_num(np.array([2., 2., 2.]), 4.0))
print np.asarray(dop.vsqrt(np.array([4., 4., 4.])))
