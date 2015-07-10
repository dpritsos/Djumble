

import timeit
import numpy as np

print "For dict keys()", np.mean(timeit.repeat(setup="d = dict([(k, v) for k, v in zip(range(10000), range(10000))])", stmt="for i in d.keys(): pass", repeat=15, number=10000))
print "For dict", np.mean(timeit.repeat(setup="d = dict([(k, v) for k, v in zip(range(10000), range(10000))])", stmt="for i in d: pass", repeat=15, number=10000))
print "For set", np.mean(timeit.repeat(setup="d = set(range(10000))", stmt="for i in d: pass", repeat=15, number=10000))
print "For list", np.mean(timeit.repeat(setup="d = list(range(10000))", stmt="for i in d: pass", repeat=15, number=10000))
print "In dict keys()", np.mean(timeit.repeat(setup="d = dict([(k, v) for k, v in zip(range(10000), range(10000))])", stmt="1500 in d.keys()", repeat=3, number=10000))
print "In dict", np.mean(timeit.repeat(setup="d = dict([(k, v) for k, v in zip(range(10000), range(10000))])", stmt="1500 in d", repeat=15, number=10000))
print "In set", np.mean(timeit.repeat(setup="d = set(range(10000))", stmt="1500 in d", repeat=15, number=10000))
print "In list", np.mean(timeit.repeat(setup="d = list(range(10000))", stmt="1500 in d", repeat=15, number=10000))