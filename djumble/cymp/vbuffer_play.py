# -*- coding: utf-8 -*-

from vbuffer import VBuffer
import numpy as np


m = VBuffer(1, 10)

a = np.asarray(m)

m.add()

a[:, :] = 1.0

print a, a.shape
