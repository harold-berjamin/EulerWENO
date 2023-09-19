import numpy as np
from euler_utils import *

# Common parameters
flux = LxF # LxF, LxW
inter = WENO_Roe # DoNone, WENO, WENO_Roe
integ = RK4 # RK1, RK3, RK4 <!> RK4 requires smaller time-steps by a factor 2/3 (cf. CFL)

# Test selection
test = 2 # 1, 2

# Job parameters
if test==1:
    # Density fluctuation
    BC = PeriodicBC # OutgoingBC, PeriodicBC
    u0, pb = Density() # Density(), Riemann(rhoJ,uJ,pJ)
    xlims = np.array([0, 2]) # Physical domain
    Tf = 2 # Final time
elif test==2:
    # Riemann data (test 2)
    # Lax
    rhoJ = np.array([0.445, 0.5]) # rhoJ = np.array([0.445, 0.5])
    uJ = np.array([0.698, 0])     # uJ = np.array([0.698, 0])
    pJ = np.array([3.528, 0.571]) # pJ = np.array([3.528, 0.571])
    # Sod
    rhoJ = np.array([1, 0.125]) # rhoJ = np.array([1, 0.125])
    uJ = np.array([0, 0])       # uJ = np.array([0, 0])
    pJ = np.array([1, 0.1])     # pJ = np.array([1, 0.1])
    # Custom
    rhoJ = np.array([1, 0.125])
    uJ = np.array([0, 0])
    pJ = np.array([1, 0.1])
    # Problem settings
    BC = OutgoingBC # OutgoingBC, PeriodicBC
    u0, pb = Riemann(rhoJ,uJ,pJ) # Density(), Riemann(rhoJ,uJ,pJ)
    xlims = np.array([-0.5, 0.5]) # Physical domain
    Tf = 0.16 # Final time