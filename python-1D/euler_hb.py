import numpy as np
import matplotlib.pyplot as plt
import time
import math
from euler_utils import *

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

# Pre-defined test cases
from euler_tests import *

"""
# Custom job parameters (overwrites pre-defined test case)
flux = LxF # LxF, Osher, LxW
inter = WENO_Roe # DoNone, WENO, WENO_Roe
integ = RK4 # RK1, RK3, RK4 <!> RK4 requires smaller time-steps by a factor 2/3 (cf. CFL)
BC = OutgoingBC # OutgoingBC, PeriodicBC
u0, pb = Riemann(rhoJ,uJ,pJ) # Density(), Riemann(rhoJ,uJ,pJ)
xlims = np.array([-0.5, 0.5]) # Physical domain
Tf = 0.16 # Final time
"""

# Graphics
plots = 0 # 0, 1, 2, 3

# Mesh size
Nx = 512
Co = 0.6 # CFL

# ----------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------

# Data initialisation
t = 0
n = 0
dx = np.abs( (xlims[1]-xlims[0])/Nx )
x = np.linspace(xlims[0]-2.5*dx, xlims[1]+2.5*dx, Nx+6)
Nx = Nx+6

# Initial condition averaging
u = BC(cellav(u0, x, dx))

# CFL
vals, _ = EigA(u)
amax = np.max(np.max(np.abs(vals)))
dt = Co * dx/amax
l_time = np.array([t])
l_speed = np.array([amax])

# Evolution right-hand side
L = RHS(flux, inter, BC, dt, dx)

# Graphics initialisation
if plots:
    figure, ax = plt.subplots(figsize=(10, 8))
    M = 1.1 * (np.max(np.abs(u[plots-1,:])) + 0.02)
    plt.axis([x[0], x[Nx-1], -M, M])
    line1, = ax.plot(x,u[plots-1,:],'b.-')
    plt.xlabel('x')
    plt.ylabel('u_'+str(plots))
    plt.title('t = '+str(t))
    plt.draw()

# Main loop
print('Entering loop (gamma = '+str(gam)+').')
tStart = time.time()
tPlot = time.time()
while t<Tf:
    # Iteration
    u = integ(L, u, dt)
    t = t + dt
    n = n + 1

    # CFL
    vals, _ = EigA(u)
    amax = np.max(np.max(np.abs(vals)))
    dt = Co * dx/amax
    l_time = np.append(l_time, [t])
    l_speed = np.append(l_speed, [amax])

    # Graphics update
    if (plots > 0) and (time.time() - tPlot > 1.5):
        # intermediate solution
        line1.set_ydata(u[plots-1,:])
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.title('t = '+str(t))
        plt.pause(0.05)
        tPlot = time.time()
tEnd = time.time()
print('Elapsed time is '+str(tEnd-tStart)+' seconds.')
print('Terminated in '+str(n)+' iterations.')
print('Max speed '+str(np.max(l_speed))+', growth ratio '+str(np.max(l_speed)/np.min(l_speed))+'.')

# Exact solution
if pb == 'Density':
    utheo = lambda x: u0(x - t)
elif pb == 'Riemann':
    UJ = Prim2Cons(np.array([rhoJ, uJ, pJ]))
    utheo = RiemannExact(UJ, gam, t)

# Plot final solution
if plots:
    # numerical solution
    line1.set_ydata(u[plots-1,:])
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.title('t = '+str(t))
    # exact solution
    xplot = np.linspace(x[0], x[Nx-1], math.floor(np.max([2 * Nx, 1e3])))
    uth = utheo(xplot)
    plt.plot(xplot, uth[plots-1, :], 'k-')
    plt.draw()
    plt.show()

# Error wrt. exact solution
if pb == 'Density':
    uth = cellav(utheo, x, dx)
    ierr = (x>xlims[0])*(x<xlims[1])
    derr = u[:,ierr] - uth[:,ierr]
    one_err = np.array([np.linalg.norm(derr[0,:]*dx,1), np.linalg.norm(derr[1,:]*dx,1), np.linalg.norm(derr[2,:]*dx,1)])
    two_err = np.array([np.linalg.norm(derr[0,:]*dx,2), np.linalg.norm(derr[1,:]*dx,2), np.linalg.norm(derr[2,:]*dx,2)])
    inf_err = np.array([np.linalg.norm(derr[0,:],np.inf), np.linalg.norm(derr[1,:],np.inf), np.linalg.norm(derr[2,:],np.inf)])
    print('L1, L2, Linf errors')
    print(np.array([one_err, two_err, inf_err]))