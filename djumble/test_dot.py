import numpy as np
import scipy.sparse as sp
import dotproduct as dp
import StringIO
import cProfile
import pstats

# Prifilling
pr = cProfile.Profile()


a = np.array([[1, 0, 3], [1, 6, 3]], dtype=np.float)
b = np.array([[1, 2], [1, 2], [3, 4]], dtype=np.float)
c = np.array([[1, 0, 3, 4, 5], [9, 0, 3, 8, 0]], dtype=np.float)
s = np.array([1, 2, 3, 4, 5], dtype=np.float)
s_sp = sp.csr_matrix(np.diag(s))

res = np.zeros((2, 2), dtype=np.float)

# Prifilling - Starts
pr.enable()
"""
for i in range(10000):
    # print np.dot(a, b)
    np.dot(a, b)

for i in range(10000):
    # print np.array(dp.dot2d(a, b))
    dp.dot2d(a, b)

# print

for i in range(10000):
    # print np.array(dp.dot2d_ds(c, s))
    dp.dot2d_ds(c, s)

for i in range(10000):
    # print np.dot(c, s_sp[:, :].toarray())
    np.dot(c, s_sp[:, :].toarray())

# print
"""
for i in range(10000):
    # print np.dot(s, s)
    np.dot(s, s)

for i in range(10000):
    # print np.array(dp.vdot(s, s))
    dp.vdot(s, s)

# print
"""
for i in range(10000):
    # print np.dot(s, s_sp[:, :].toarray())
    np.dot(s, s_sp[:, :].toarray())

for i in range(10000):
    # print np.array(dp.dot1d_ds(s, s))
    dp.dot1d_ds(s, s)
"""


# Prifilling - Ends
pr.disable()

# Prifilling - Stats
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats("cumtime").print_stats()  # cumtime
print s.getvalue()
