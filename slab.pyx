import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def slab(DTYPE_t D,np.float_t L,np.float_t c_L,np.float_t c_0,int num_elements,int stepmax):
    cdef np.ndarray[DTYPE_t,ndim=1] uptake = np.zeros(stepmax,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] c = np.zeros(num_elements,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] c_old = np.zeros(num_elements,dtype=DTYPE)
    cdef int step = 0
    cdef int i = 0
    cdef DTYPE_t tally = 0.
    c[0] = c_0
    c[num_elements-1] = c_L
    while step < stepmax:
        for i in range(num_elements-2):
            c[i+1] += D*(c_old[i]-2*c_old[i+1]+c_old[i+2])
        tally = 0.
        for i in range(num_elements):
            tally += c[i]
        uptake[step] = tally/2.
        for i in range(num_elements):
            c_old[i] = c[i]
        step += 1
    return uptake
