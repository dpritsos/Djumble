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
    cdef vector[float] buffer

    def __cinit__(self, Py_ssize_t vbz_size, Py_ssize_t buff_size):
        self.buff_size = buff_size
        self.vbz_size = vbz_size
        cdef int rows
        #for rows in range(self.vbz_size):

        print 'OK'

    def add(self):
        self.buffer.extend(self.buff_size)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        print 'OK'
        cdef Py_ssize_t itemsize = sizeof(self.buffer[0])
        print 'OK'
        self.shape[0] = self.buffer.size() / self.buff_size
        self.shape[1] = self.buff_size
        print 'OK'
        # Adding vbz_size rows, initially zero-filled.

        print 'OK'

        # Stride 1 is the distance, in bytes, between two items in a row;
        # this is the distance between two adjacent items in the vector.
        # Stride 0 is the distance between the first elements of adjacent rows.
        self.strides[1] = <Py_ssize_t>(  <char*>&(self.buffer[1])
                                       - <char*>&(self.buffer[0]))
        self.strides[0] = self.buff_size * self.strides[1]

        print 'OK'
        print self.shape
        print self.strides

        buffer.buf = <char*>&(self.buffer[0])
        buffer.format = 'f'                     # float
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self.buffer.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass
