# -*- coding: utf-8 -*-

from vbuffer import VBuffer
import numpy as np


m = VBuffer(1, 10)

m.add()
m.add()

a = np.asarray(m)

a[:] = 1.0



a = np.asarray(m)

print m, a, a.shape
