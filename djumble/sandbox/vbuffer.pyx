# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
cimport cython as cy
cimport numpy as cnp
import cython.parallel as cyp
import scipy.special as special
from cpython cimport Py_buffer
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, realloc, free
# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class VBuffer:
    cdef Py_ssize_t buff_size
    cdef Py_ssize_t vbz_size
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef vector[float] vbuffer
    cdef int view_count

    def __cinit__(self, Py_ssize_t vbz_size, Py_ssize_t buff_size):
        self.buff_size = buff_size
        self.vbz_size = vbz_size
        self.view_count = 0

    def add(self):
        if self.view_count > 0:
            raise ValueError("can't add row while being viewed")
        self.vbuffer.resize(self.vbuffer.size() + self.buff_size)

    def __getbuffer__(self, Py_buffer *buffer, int flags):

        cdef Py_ssize_t itemsize = sizeof(self.vbuffer[0])

        self.shape[0] = self.vbuffer.size() / self.buff_size
        self.shape[1] = self.buff_size

        # Adding vbz_size rows, initially zero-filled.

        # Stride 1 is the distance, in bytes, between two items in a row;
        # this is the distance between two adjacent items in the vector.
        # Stride 0 is the distance between the first elements of adjacent rows.
        self.strides[1] = <Py_ssize_t>(  <char*>&(self.vbuffer[1])
                                       - <char*>&(self.vbuffer[0]))
        self.strides[0] = self.buff_size * self.strides[1]

        buffer.buf = <char*>&(self.vbuffer[0])
        buffer.format = 'f'                     # float
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self.vbuffer.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self.view_count -= 1
