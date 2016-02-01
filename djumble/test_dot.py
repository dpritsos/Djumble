import numpy as np
import dotproduct as dp
import StringIO
import cProfile
import pstats

# Prifilling
pr = cProfile.Profile()


a = np.array([[1, 0, 3], [1, 6, 3]], dtype=np.float)
b = np.array([[1, 2], [1, 2], [3, 4]], dtype=np.float)

res = np.zeros((2, 2), dtype=np.float)

# Prifilling - Starts
pr.enable()

for i in xrange(1000):
    np.dot(a, b)
    dp.dot(a, b)


# Prifilling - Ends
pr.disable()

# Prifilling - Stats
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats("time").print_stats()  # cumtime
print s.getvalue()
