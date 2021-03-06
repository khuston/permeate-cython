"""
    cython functions for fast 1D transport modeling

    dogbone is the model function right now. It has been verified
    against the analytical solution.

    Written by Kyle J. Huston on Thursday, August 27, 2015
"""

import numpy as np
cimport numpy as np
cimport cython
from cython cimport view
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def leak(double V, double A, double K, double c_in, double c_out,
                      double D, double L,
                      int num_elements=200,double maxt=0, double outputperiod=0,
                      save_profile_flag=False, times_to_save=None
                      ):
    """
        Leak through polymer packaging
        (No-flux boundary condition at x=L)
        Returns the uptake and solution concentration

        Required Args:
            V (double): Volume of solution
            A (double): Surface area inside sample
            K (double): Container-solution partition coefficient
            c_in (double): Initial concentration in solution
            c_out (double): Boundary concentration (outside polymer)
            D (double): Fickian diffusion coefficient in sample
            L (double): Thickness of sample

        Optional Args:
            num_elements (int): Number of discretization elements
            save_profile_flag (True/False): Whether to include 
            maxt (double): Maximum time, i.e. how long to run
            outputperiod (double): Period between outputs
            times_to_save (list of doubles): List of times to give output for

        Units:
            No unit requirement yet... equations are still
            linear, so any units can be used so long as they
            are consistent.

        Yields:
            time (list): List of times at each save
            uptake (list): List of uptake at each save
            c_in (list): List of c_in at each save
            c_profile (list of lists): List of concentration profiles at each save

        Usage examples:
            time, uptake, c_in = leak(V,A,K,c_in,c_out,D,L,maxt=2000,outputperiod=10)

            time, uptake, c_in, c_profile = 
                    dogbone(V,A,K,c_in,c_out,D,L,maxt=2000,
                            outputperiod=10,save_profile_flag=True)

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
    if times_to_save is None:
        times_to_save = np.arange(maxt,-outputperiod/2.,-outputperiod).tolist()
    else:
        times_to_save=times_to_save[::-1]
        try:
            times_to_save = times_to_save.tolist()
        except:
            pass
        maxt = max(times_to_save)
    cdef:
        int i = 0
        int savingdata = 0
        int save_i = 0
        int num_saves = len(times_to_save)
        double tally = 0.
        double time = 0.

    # Pre-calculate some coefficients
    cdef:
        double max_dt = 0.4 * L**2/(D * num_elements**2)
        double dt = max_dt
        double diff_factor = num_elements**2 * D/L**2
        double depl_factor = (A*D*num_elements)/(V*L)

    # Timeseries recorder arrays
    cdef:
        double *times = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *uptake = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *leaked = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_in_traj = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_profile = <double *>PyMem_Malloc(num_saves * num_elements * sizeof(double))
        int save_profile = 0
    if save_profile_flag == True:
        save_profile = 1

    # Upcoming and previous concentrations
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))

    cdef double next_save_time = times_to_save.pop()

    if not times or not uptake or not leaked or not c or not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = K * c_in                       # Boundary condition 1: Equilibrium with c_in
        c[num_elements-1] = c[num_elements-2] # Boundary condition 2: No flux
        for i in range(num_elements):
            c_old[i] = c[i]

        savingdata = 0

        # Inner loop with static types
        while time < maxt and save_i < num_saves:
            # Forward Euler integration of diffusion equation
            dt = abs(double_min(max_dt,0.1/depl_factor*c_in/(c_old[0]-c_old[1])))
            if dt >= (next_save_time - time):
                dt = next_save_time - time
                try:
                    next_save_time = times_to_save.pop()
                except:
                    pass
                savingdata = 1
            time += dt
            for i in range(num_elements-2):
                c[i+1] += dt*diff_factor*(c_old[i]-2*c_old[i+1]+c_old[i+2])
            c_in -= dt*depl_factor*(c_old[0] - c_old[1])

            c[0] = K * c_in                          # BC 1: Equilibrium with c_in
            c[num_elements-1] = c_out                # BC 2: Constant concentration

            # Analysis + c becomes c_old + Save
            tally = 0.
            for i in range(num_elements):
                c_old[i] = c[i]
            # Exact copy of initial save code above!
            if savingdata:
                for i in range(num_elements-2):
                    tally += c[i+1]
                tally += c[0]/2.
                tally += c[num_elements-1]/2.
                if (save_profile):
                    for i in range(num_elements):
                        c_profile[save_i*num_elements + i] = c[i]

                uptake[save_i] = tally/num_elements*L*A
                c_in_traj[save_i] = c_in
                times[save_i] = time
                save_i += 1
                savingdata = 0

        if save_profile:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ], \
                   [ [c_profile[j*num_elements + i] for i in range(num_elements)] for j in range(num_saves) ]
        else:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ]
    finally:
        PyMem_Free(times)
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)
        PyMem_Free(leaked)
        PyMem_Free(c_in_traj)
        PyMem_Free(c_profile)



@cython.boundscheck(False)
@cython.wraparound(False)
def dogbone(double V, double A, double K, double c_in,
                      double D, double L,
                      int num_elements=200,double maxt=0, double outputperiod=0,
                      save_profile_flag=False, times_to_save=None
                      ):
    """
        Penetration into a dogbone
        (No-flux boundary condition at x=L)
        Returns the uptake and solution concentration

        Required Args:
            V (double): Volume of solution
            A (double): Surface area inside sample
            K (double): Container-solution partition coefficient
            c_in (double): Initial concentration in solution
            D (double): Fickian diffusion coefficient in sample
            L (double): Thickness of sample

        Optional Args:
            num_elements (int): Number of discretization elements
            save_profile_flag (True/False): Whether to include 
            maxt (double): Maximum time, i.e. how long to run
            outputperiod (double): Period between outputs
            times_to_save (list of doubles): List of times to give output for

        Units:
            No unit requirement yet... equations are still
            linear, so any units can be used so long as they
            are consistent.

        Yields:
            time (list): List of times at each save
            uptake (list): List of uptake at each save
            c_in (list): List of c_in at each save
            c_profile (list of lists): List of concentration profiles at each save

        Usage examples:
            time, uptake, c_in = dogbone(V,A,K,c_in,D,L,maxt=2000,outputperiod=10)

            time, uptake, c_in, c_profile = 
                    dogbone(V,A,K,c_in,D,L,maxt=2000,
                            outputperiod=10,save_profile_flag=True)

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
    if times_to_save is None:
        times_to_save = np.arange(maxt,-outputperiod/2.,-outputperiod).tolist()
    else:
        times_to_save=times_to_save[::-1]
        try:
            times_to_save = times_to_save.tolist()
        except:
            pass
        maxt = max(times_to_save)
    cdef:
        int i = 0
        int savingdata = 0
        int save_i = 0
        int num_saves = len(times_to_save)
        double tally = 0.
        double time = 0.

    # Pre-calculate some coefficients
    cdef:
        double max_dt = 0.4 * L**2/(D * num_elements**2)
        double dt = max_dt
        double diff_factor = num_elements**2 * D/L**2
        double depl_factor = (A*D*num_elements)/(V*L)

    # Timeseries recorder arrays
    cdef:
        double *times = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *uptake = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *leaked = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_in_traj = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_profile = <double *>PyMem_Malloc(num_saves * num_elements * sizeof(double))
        int save_profile = 0
    if save_profile_flag == True:
        save_profile = 1

    # Upcoming and previous concentrations
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))

    cdef double next_save_time = times_to_save.pop()

    if not times or not uptake or not leaked or not c or not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = K * c_in                       # Boundary condition 1: Equilibrium with c_in
        c[num_elements-1] = c[num_elements-2] # Boundary condition 2: No flux
        for i in range(num_elements):
            c_old[i] = c[i]

        savingdata = 0

        # Inner loop with static types
        while time < maxt and save_i < num_saves:
            # Forward Euler integration of diffusion equation
            dt = abs(double_min(max_dt,0.1/depl_factor*c_in/(c_old[0]-c_old[1])))
            if dt >= (next_save_time - time):
                dt = next_save_time - time
                try:
                    next_save_time = times_to_save.pop()
                except:
                    pass
                savingdata = 1
            time += dt
            for i in range(num_elements-2):
                c[i+1] += dt*diff_factor*(c_old[i]-2*c_old[i+1]+c_old[i+2])
            c_in -= dt*depl_factor*(c_old[0] - c_old[1])

            c[0] = K * c_in                          # BC 1: Equilibrium with c_in
            c[num_elements-1] = c[num_elements-2]    # BC 2: No flux

            # Analysis + c becomes c_old + Save
            tally = 0.
            for i in range(num_elements):
                c_old[i] = c[i]
            # Exact copy of initial save code above!
            if savingdata:
                for i in range(num_elements-2):
                    tally += c[i+1]
                tally += c[0]/2.
                tally += c[num_elements-1]/2.
                if (save_profile):
                    for i in range(num_elements):
                        c_profile[save_i*num_elements + i] = c[i]

                uptake[save_i] = tally/num_elements*L*A
                c_in_traj[save_i] = c_in
                times[save_i] = time
                save_i += 1
                savingdata = 0

        if save_profile:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ], \
                   [ [c_profile[j*num_elements + i] for i in range(num_elements)] for j in range(num_saves) ]
        else:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ]
    finally:
        PyMem_Free(times)
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)
        PyMem_Free(leaked)
        PyMem_Free(c_in_traj)
        PyMem_Free(c_profile)

@cython.boundscheck(False)
@cython.wraparound(False)
def crystal_dogbone(double V, double A, double K, double c_in,
                      double D, double L, double phi, double nu,
                      int num_elements=200,double maxt=0, double outputperiod=0,
                      save_profile_flag=False, times_to_save=None
                      ):
    """
        Penetration into a dogbone
        (No-flux boundary condition at x=L)
        Returns the uptake and solution concentration

        Required Args:
            V (double): Volume of solution
            A (double): Surface area inside sample
            K (double): Container-solution partition coefficient
            c_in (double): Initial concentration in solution
            D (double): Fickian diffusion coefficient in accessible volume of sample
            L (double): Thickness of sample
            phi (double): Accessible volume fraction of sample
            nu (double): Tortuosity exponent
            maxt (double): Maximum time, i.e. how long to run
            outputperiod (double): Period between outputs

        Optional Args:
            num_elements (int): Number of discretization elements
            save_profile_flag (True/False): Whether to include 

        Units:
            No unit requirement yet... equations are still
            linear, so any units can be used so long as they
            are consistent.

        Yields:
            time (list): List of times at each save
            uptake (list): List of uptake at each save
            c_in (list): List of c_in at each save
            c_profile (list of lists): List of concentration profiles at each save

        Usage examples:
            time, uptake, c_in = dogbone(V,A,K,c_in,D,L,maxt=2000,outputperiod=10)

            time, uptake, c_in, c_profile = 
                    dogbone(V,A,K,c_in,D,L,maxt=2000,
                            outputperiod=10,save_profile_flag=True)

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
    if times_to_save is None:
        times_to_save = np.arange(maxt,-outputperiod/2.,-outputperiod).tolist()
    else:
        times_to_save=times_to_save[::-1]
        try:
            times_to_save = times_to_save.tolist()
        except:
            pass
        maxt = max(times_to_save)
    cdef:
        int num_ts = 0
        int i = 0
        int savingdata = 0
        int save_i = 0
        int num_saves = len(times_to_save)
        double tally = 0.
        double time = 0.

    # Pre-calculate some coefficients
    cdef:
        double max_dt = 0.4 * L**2/(D*phi**(nu-1.) * num_elements**2)
        double dt = max_dt
        double diff_factor = num_elements**2 * D/L**2 * phi**(nu-1.)
        double depl_factor = (phi**nu*A*D*num_elements)/(V*L)

    # Timeseries recorder arrays
    cdef:
        double *times = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *uptake = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *leaked = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_in_traj = <double *>PyMem_Malloc(num_saves * sizeof(double))
        double *c_profile = <double *>PyMem_Malloc(num_saves * num_elements * sizeof(double))
        int save_profile = 0
    if save_profile_flag == True:
        save_profile = 1

    # Upcoming and previous concentrations
    cdef double *c = <double *>PyMem_Malloc(num_elements * sizeof(double))
    cdef double *c_old = <double *>PyMem_Malloc(num_elements * sizeof(double))

    cdef double next_save_time = times_to_save.pop()

    if not times or not uptake or not leaked or not c or not c_old:
        raise MemoryError()

    try:
        for i in range(num_elements):
            c[i] = 0
        c[0] = K * c_in                       # Boundary condition 1: Equilibrium with c_in
        c[num_elements-1] = c[num_elements-2] # Boundary condition 2: No flux
        for i in range(num_elements):
            c_old[i] = c[i]

        savingdata = 0

        # Inner loop with static types
        while time < maxt or save_i < num_saves:
            # Forward Euler integration of diffusion equation
            dt = abs(double_min(max_dt,0.1/depl_factor*c_in/(c_old[0]-c_old[1])))
            if dt >= (next_save_time - time):
                dt = next_save_time - time
                try:
                    next_save_time = times_to_save.pop()
                except:
                    pass
                savingdata = 1
            time += dt
            for i in range(num_elements-2):
                c[i+1] += dt*diff_factor*(c_old[i]-2*c_old[i+1]+c_old[i+2])
                num_ts += 1
            c_in -= dt*depl_factor*(c_old[0] - c_old[1])

            c[0] = K * c_in                          # BC 1: Equilibrium with c_in
            c[num_elements-1] = c[num_elements-2]    # BC 2: No flux

            # Analysis + c becomes c_old + Save
            for i in range(num_elements):
                c_old[i] = c[i]
            # Exact copy of initial save code above!
            if savingdata:
                tally = 0.
                for i in range(num_elements-2):
                    tally += c[i+1]
                tally += c[0]/2.
                tally += c[num_elements-1]/2.
                if (save_profile):
                    for i in range(num_elements):
                        c_profile[save_i*num_elements + i] = c[i]

                uptake[save_i] = tally/num_elements*L*A*phi
                c_in_traj[save_i] = c_in
                times[save_i] = time
                save_i += 1
                savingdata = 0
        print num_ts

        if save_profile:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ], \
                   [ [c_profile[j*num_elements + i] for i in range(num_elements)] for j in range(num_saves) ]
        else:
            return [ times[i] for i in range(num_saves) ],  \
                   [ uptake[i] for i in range(num_saves) ], \
                   [ c_in_traj[i] for i in range(num_saves) ]
    finally:
        PyMem_Free(times)
        PyMem_Free(uptake)
        PyMem_Free(c)
        PyMem_Free(c_old)
        PyMem_Free(leaked)
        PyMem_Free(c_in_traj)
        PyMem_Free(c_profile)
