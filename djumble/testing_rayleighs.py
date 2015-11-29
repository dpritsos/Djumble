
import numpy as np

s = 0.6

A = np.array([1.55999999]*500)


sum1, sum2 = 0.0, 0.0
for a in A:
    sum1 += np.log(a)
    sum2 += np.square(a) / (2 * np.square(s))
params_pdf = sum1 - sum2 - (2 * A.shape[0] * np.log(s))

print params_pdf


a_pderiv = (1 / A[0]) - (A[0] / np.square(s))

print a_pderiv
