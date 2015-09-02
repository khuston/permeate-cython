import matplotlib.pyplot as plt
import numpy as np
from permeate import dogbone
from slab import Slab

V = 1000000.
A = 0.001

Bi  = 1e-12
L   = 1.
D   = 1e-5
c_L = 1.
c_inf = 0.

maxt = 100000
dt = 1000

num_elements = 200

times, numeric_uptake, _ = dogbone(V,A,1.,c_L,D,L,num_elements,maxt,dt)

x = np.linspace(0.,L,num_elements)

analytic_uptake = []
for t in times:
    analytic = Slab(Bi,L,D,c_L,c_inf)
    analytic_c = analytic.evaluate(x,t)
    analytic_uptake.append(np.trapz(analytic_c,x))

print A*np.array(analytic_uptake)
print numeric_uptake
plt.plot(times,A*np.array(analytic_uptake),'-')
plt.plot(times,np.array(numeric_uptake),'.')
plt.show()
