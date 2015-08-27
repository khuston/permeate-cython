"""
    cython functions for fast transport modeling

    TODO: Verify forward Euler results with analytical solution
          and with gproms results.

    Written by Kyle J. Huston on Thursday, August 27, 2015
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def leak(double V, double A, double K, double c_in,
                      double D, double L, double c_L,
                      int num_elements,double num_steps):
    """
        Leak from a container
        (Dirichlet boundary condition at x=L)
        Returns the uptake and solution concentration

        Contents <--> Container

        Args:
            V (double): Volume of contents
            A (double): Surface area inside container
            K (double): Container-contents partition coefficient
            c_in (double): Initial concentration in contents
            D (double): Diffusion coefficient in container
            L (double): Thickness of container
            c_L (double): Concentration on exterior
            num_elements (int): Number of discretization elements
            num_steps (int): Number of time steps

        Yields:
            uptake (list): List of uptake at each time step
            c_in (list): List of c_in at each time step

        Developer Notes:
            Diffusion seems stable when dt*diff_coeff < 0.4, where
                 dt*diff_coeff = dt*(num_elements**2 * D)/L**2
            which implies dt should satisfy
                dt < 0.4 * L**2/(D * num_elements**2)

            Another problem might occur if the solution depletes
            too rapidly.
                dc_in/c_in = AD/Vh ( c(1)/c_in - c(0)/c_in ) dt
            dc_in/c_in should be kept small. At least less than 0.1.
            Then dt should be limited below
                dt < 0.1 (V h c_in)/(A D) * 1/(c(1)-c(0))
            Approaching steady state, c(1)-c(0) diverges, which means
            depletion no longer limits the size of our timestep.
    """
    # Counter variables used for timestepping, indexing positions,
    # and trapezoidal integration
    cdef int step = 0
    cdef int i
    cdef double tally = 0.
    cdef double time = 0.

    # Pre-calculate some coefficients
    cdef double max_dt = 0.4 * L**2/(D * num_elements**2)
    cdef double dt
    cdef double diff_factor = num_elements**2 * D/L**2
    cdef double depl_factor = (A*D*num_elements)/(V*L)

    # Timeseries recorder arrays
    cdef double *times = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *uptake = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *leaked = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *c_in_traj = <double *>PyMem_Malloc(num_steps * sizeof(double))

    # Upcoming and previous concentrations
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))

    if not uptake:
        raise MemoryError()
    if not c:
        raise MemoryError()
    if not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = K * c_in         # Boundary condition 1: Equilibrium with c_in
        c[num_elements-1] = c_L # Boundary condition 2: Fixed concentration
        for i in range(num_elements):
            c_old[i] = c[i]

        # Inner loop with static types
        while step < num_steps:
            c_old[0] = K * c_in # Boundary condition 1: Equilibrium with c_in
            # Forward Euler integration of diffusion equation
            dt = abs(double_min(max_dt,0.1/depl_factor*c_in/(c_old[0]-c_old[1])))
            time += dt
            for i in range(num_elements-2):
                c[i+1] += dt*diff_factor*(c_old[i]-2*c_old[i+1]+c_old[i+2])
            c_in -= dt*depl_factor*(c_old[0] - c_old[1])

            # Analysis + c becomes c_old
            tally = 0.
            for i in range(num_elements):
                c_old[i] = c[i]
                tally += c[i]
            uptake[step] = tally/2.
            c_in_traj[step] = c_in
            times[step] = time
            step += 1

        return [ times[i] for i in range(num_steps) ],  \
               [ uptake[i] for i in range(num_steps) ], \
               [ c_in_traj[i] for i in range(num_steps) ]
    finally:
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)


@cython.boundscheck(False)
@cython.wraparound(False)
def dogbone(double V, double A, double K, double c_in,
                      double D, double L,
                      int num_elements,int num_steps):
    """
        Penetration into a dogbone
        (No-flux boundary condition at x=L)
        Returns the uptake and solution concentration

        Solution --> Sample

        Args:
            V (double): Volume of solution
            A (double): Surface area inside sample
            K (double): Container-solution partition coefficient
            c_in (double): Initial concentration in solution
            D (double): Diffusion coefficient in sample
            L (double): Thickness of sample
            num_elements (int): Number of discretization elements
            num_steps (int): Number of time steps

        Yields:
            uptake (list): List of uptake at each time step
            c_in (list): List of c_in at each time step

        Developer Notes:
            Diffusion seems stable when dt*diff_coeff < 0.4, where
                 dt*diff_coeff = dt*(num_elements**2 * D)/L**2
            which implies dt should satisfy
                dt < 0.4 * L**2/(D * num_elements**2)

            Another problem might occur if the solution depletes
            too rapidly.
                dc_in/c_in = AD/Vh ( c(1)/c_in - c(0)/c_in ) dt
            dc_in/c_in should be kept small. At least less than 0.1.
            Then dt should be limited below
                dt < 0.1 (V h c_in)/(A D) * 1/(c(1)-c(0))
            Approaching steady state, c(1)-c(0) diverges, which means
            depletion no longer limits the size of our timestep.
    """
    # Counter variables used for timestepping, indexing positions,
    # and trapezoidal integration
    cdef int step = 0
    cdef int i = 0
    cdef double tally = 0.
    cdef double time = 0.

    # Pre-calculate some coefficients
    cdef double max_dt = 0.4 * L**2/(D * num_elements**2)
    cdef double dt = max_dt
    cdef double diff_factor = num_elements**2 * D/L**2
    cdef double depl_factor = (A*D*num_elements)/(V*L)

    # Timeseries recorder arrays
    cdef double *times = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *uptake = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *leaked = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *c_in_traj = <double *>PyMem_Malloc(num_steps * sizeof(double))

    # Upcoming and previous concentrations
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))

    if not uptake:
        raise MemoryError()
    if not c:
        raise MemoryError()
    if not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = K * c_in                       # Boundary condition 1: Equilibrium with c_in
        c[num_elements-1] = c[num_elements-2] # Boundary condition 2: No flux
        for i in range(num_elements):
            c_old[i] = c[i]

        # Inner loop with static types
        while step < num_steps:
            c_old[0] = K * c_in                           # BC 1: Equilibrium with c_in
            c_old[num_elements-1] = c_old[num_elements-2] # BC 2: No flux
            # Forward Euler integration of diffusion equation
            dt = abs(double_min(max_dt,0.1/depl_factor*c_in/(c_old[0]-c_old[1])))
            time += dt
            for i in range(num_elements-2):
                c[i+1] += dt*diff_factor*(c_old[i]-2*c_old[i+1]+c_old[i+2])
            c_in -= dt*depl_factor*(c_old[0] - c_old[1])

            # Analysis + c becomes c_old
            tally = 0.
            for i in range(num_elements):
                c_old[i] = c[i]
                tally += c[i]
            uptake[step] = tally/2.
            c_in_traj[step] = c_in
            times[step] = time
            step += 1

        return [ times[i] for i in range(num_steps) ],  \
               [ uptake[i] for i in range(num_steps) ], \
               [ c_in_traj[i] for i in range(num_steps) ]
    finally:
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)



@cython.boundscheck(False)
@cython.wraparound(False)
def slab(double D,double L,double c_L,double c_0,int num_elements,int num_steps):
    """
        Diffusion seems stable when coeff < 0.4, where
        coefficient is
            (dt * num_elements**2 * D)/L**2
        which implies dt should satisfy
            dt < 0.4 * L**2/(D * num_elements**2)

        Another problem might occur if the solution depletes
        too rapidly.
            dc_in/c_in = AD/Vh ( c(1)/c_in - c(0)/c_in ) dt
        dc_in/c_in should be kept small. At least less than 0.1.
        Then dt should be limited below
            dt < 0.1 (V h c_in)/(A D) * 1/(c(1)-c(0))
        Approaching steady state, c(1)-c(0) diverges, which means
        depletion no longer limits the size of our timestep.
    """
    cdef int step = 0
    cdef int i = 0
    cdef double tally = 0.

    cdef double dt = 0.4 * L**2/(D * num_elements**2)
    cdef double coeff = dt * num_elements**2 * D/L**2

    cdef double *uptake = <double *>PyMem_Malloc(num_steps * sizeof(double))
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))
    if not uptake:
        raise MemoryError()
    if not c:
        raise MemoryError()
    if not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = c_0
        c[num_elements-1] = c_L
        for i in range(num_elements):
            c_old[i] = c[i]

        while step < num_steps:
            for i in range(num_elements-2):
                c[i+1] += D*(c_old[i]-2*c_old[i+1]+c_old[i+2])
            tally = 0.
            for i in range(num_elements):
                tally += c[i]
            uptake[step] = tally/2.
            for i in range(num_elements):
                c_old[i] = c[i]
            step += 1
        return [ uptake[i] for i in range(num_steps) ]
    finally:
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)
