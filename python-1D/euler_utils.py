import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import fsolve

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------

# Common parameters
gam = 1.4  # 1.4 (heat capacity ratio for diatomic ideal gas ~ standard air)

# ----------------------------------------------------------------
# Euler equations physics
# physical flux u_t + f(u)_x = 0     (Chap. 3 p. 87, Toro, 2009)
# ----------------------------------------------------------------

def f(u):
    v = u[1,:] / u[0,:]
    flx = np.array([u[1,:], 0.5*(3-gam)*u[0,:]*v**2 + (gam-1)*u[2,:], gam*u[2,:]*v - 0.5*(gam-1)*u[0,:]*v**3])
    return flx

def A(u):
    v = u[1,:] / u[0,:]
    p = (gam-1) * (u[2,:] - 0.5*u[0,:]*v**2)
    a = np.sqrt(gam*p/u[0,:])
    N = len(u[0,:])
    jac = np.array([[0]*N, [1]*N, [0]*N, 
                    0.5*(gam-3)*v**2, (3-gam)*v, (gam-1)*np.ones(N), 
                    0.5*(gam-2)*v**3 - a**2*v/(gam-1), 0.5*(3-2*gam)*v**2 + a**2/(gam-1), gam*v])
    return jac

def EigA(u):
    v = u[1,:] / u[0,:]
    p = (gam-1) * (u[2,:] - 0.5*u[0,:]*v**2)
    a = np.sqrt(gam*p/u[0,:])
    vals = np.array([v-a, v, v+a])
    H = 0.5*v**2 + a**2/(gam-1)
    N = len(u[0,:])
    vecs = np.array([[1]*N, v-a, H-v*a, 
                     [1]*N, v, 0.5*v**2, 
                     [1]*N, v+a, H+v*a])
    return vals, vecs

def aA(u):
    vals, vecs = EigA(u)
    N = len(u[0,:])
    mat = np.zeros((9,N))
    for i in range(N):
        aLamb = np.diag(np.absolute(vals[:,i]))
        R = np.reshape(vecs[:,i], (3,3), order='F')
        aA = np.dot(np.dot(R, aLamb), np.linalg.inv(R))
        mat[:,i] = np.reshape(aA, (9,), order='F')
    return mat

# ----------------------------------------------------------------
# Initial conditions
# ----------------------------------------------------------------

def Density():
    def my_func(x):
        rho = 1 + 0.2*np.sin(np.pi*x)
        if isinstance(x, float):
            u = 1
            p = 1
        else:
            Nx = len(x)
            u = np.ones(Nx)
            p = np.ones(Nx)
        u0 = Prim2Cons(np.array([rho,u,p]))
        return u0
    return my_func, 'Density'

def Riemann(rhoJ,uJ,pJ):
    # Initial data
    UJ = Prim2Cons(np.array([rhoJ,uJ,pJ]))
    def my_func(x):
        if isinstance(x, float):
            u0 = UJ[:,0]*(x<0) + UJ[:,1]*(x>=0)
        else:
            Nx = len(x)
            u0 = np.zeros((3,Nx))
            for i in range(Nx):
                u0[:,i] = UJ[:,0]*(x[i]<0) + UJ[:,1]*(x[i]>=0)
        return u0
    return my_func, 'Riemann'

def cellav(u0, x, dx):
    u = u0(x)
    uav0 = lambda x: quad_vec(u0, x-0.5*dx, x+0.5*dx)[0] / dx
    for i in range(len(x)):
        u[:,i] = uav0(x[i])
    return u

# ----------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------

def PeriodicBC(u):
    Nx = len(u[0,:])
    v = u
    v[:,0] = u[:,Nx-6]
    v[:,1] = u[:,Nx-5]
    v[:,2] = u[:,Nx-4]
    v[:,Nx-3] = u[:,3]
    v[:,Nx-2] = u[:,4]
    v[:,Nx-1] = u[:,5]
    return v

def OutgoingBC(u):
    Nx = len(u[0,:])
    v = u
    v[:,0] = u[:,3]
    v[:,1] = u[:,3]
    v[:,2] = u[:,3]
    v[:,Nx-3] = u[:,Nx-4]
    v[:,Nx-2] = u[:,Nx-4]
    v[:,Nx-1] = u[:,Nx-4]
    return v

# ----------------------------------------------------------------
# Low-order schemes
# ----------------------------------------------------------------

def DoNone(u):
    up05m = u
    up05p = np.roll(u, -1, axis=1)
    return up05m, up05p

def LxF(u, inter, dt, dx): # LLF flux
    up05m, up05p = inter(u)
    vals, _ = EigA(u)
    amax = np.max(np.abs(vals[2,:] - vals[1,:]) + np.abs(vals[1,:]))
    fp05 = 0.5 * (f(up05m) + f(up05p) - amax * (up05p - up05m))
    return fp05

def Osher(u, inter, dt, dx): # Osher flux
    Nx = len(u[0,:])
    up05m, up05p = inter(u)
    du = up05p - up05m
    u1 = up05m + (0.5-np.sqrt(15)/10)*du
    u2 = up05m + 0.5*du
    u3 = up05m + (0.5+np.sqrt(15)/10)*du
    iA = (5*aA(u1) + 8*aA(u2) + 5*aA(u3))/18
    fp05 = 0.5*(f(up05m) + f(up05p))
    for i in range(Nx):
        iAr = np.reshape(iA[:,i],(3,3),order='F')
        fp05[:,i] = fp05[:,i] - 0.5* iAr @ du[:,i]
    return fp05

def LxW(u, inter, dt, dx): # LxW flux
    Nx = len(u[0,:])
    up05m, up05p = inter(u)
    fp05 = 0.5 * (f(up05m) + f(up05p))
    df05 = 0.5 * (f(up05p) - f(up05m))
    Ap05 = A(0.5*(up05m + up05p))
    for i in range(Nx):
        Aloc = np.reshape(Ap05[:,i], (3, 3))
        fp05[:,i] = fp05[:,i] - dt/dx * Aloc @ df05[:,i]
    return fp05

def RHS(flux, inter, BC, dt, dx):
    def my_func(u):
        fp05 = flux(BC(u), inter, dt, dx)
        fm05 = np.roll(fp05, 1, axis=1)
        df = BC(-(fp05 - fm05)/dx)
        return df
    return my_func

def RK1(L, u, dt):
    up = u + dt*L(u)
    return up

# ----------------------------------------------------------------
# High-order WENO schemes
# ----------------------------------------------------------------

def reconstructWENO(uim2, uim1, ui, uip1, uip2):
    # polynomial approx
    up051 = (2*uim2 - 7*uim1 + 11*ui )/6
    up052 = ( -uim1 + 5*ui   + 2*uip1)/6
    up053 = (2*ui   + 5*uip1 -   uip2)/6
    # smoothness indicators
    b1 = 13/12*(uim2 - 2*uim1 + ui  )**2 + 0.25*(uim2 - 4*uim1 + 3*ui)**2
    b2 = 13/12*(uim1 - 2*ui   + uip1)**2 + 0.25*(uim1 - uip1)**2
    b3 = 13/12*(ui   - 2*uip1 + uip2)**2 + 0.25*(3*ui - 4*uip1 + uip2)**2
    # weights
    w1 = 0.1 / (1e-6 + b1)**2
    w2 = 0.6 / (1e-6 + b2)**2
    w3 = 0.3 / (1e-6 + b3)**2
    ws = w1 + w2 + w3
    # reconstructed cell-interface value
    up05m = (w1*up051 + w2*up052 + w3*up053)/ws
    return up05m

def WENO(u):
    Nx = len(u[0,:])
    up05m = u.copy()
    up05p = np.roll(u, -1, axis=1)
    for i in range(3):
        for j in range(2, Nx-3):
            up05m[i,j] = reconstructWENO(u[i,j-2],u[i,j-1],u[i,j],u[i,j+1],u[i,j+2])
            up05p[i,j] = reconstructWENO(u[i,j+3],u[i,j+2],u[i,j+1],u[i,j],u[i,j-1])
    return up05m, up05p

def Roe(u, up1):
    sRhoL = np.sqrt(u[0])
    sRhoR = np.sqrt(up1[0])
    vL = u[1] / u[0]
    vR = up1[1] / up1[0]
    v = (sRhoL * vL + sRhoR * vR) / (sRhoL + sRhoR)
    pL = (gam - 1) * (u[2] - 0.5 * u[0] * vL**2)
    pR = (gam - 1) * (up1[2] - 0.5 * up1[0] * vR**2)
    aL = np.sqrt(gam * pL / u[0])
    aR = np.sqrt(gam * pR / up1[0])
    HL = 0.5 * vL**2 + aL**2 / (gam - 1)
    HR = 0.5 * vR**2 + aR**2 / (gam - 1)
    H = (sRhoL * HL + sRhoR * HR) / (sRhoL + sRhoR)
    h = H - 0.5 * v**2
    a = np.sqrt((gam - 1) * h)
    vecsR = np.array([1, 1, 1, v - a, v, v + a, H - v * a, 0.5 * v**2, H + v * a])
    vecsL = 0.5 / h * np.array([0.5 * v**2 + v * h / a, -h / a - v, 1, 2 * h - v**2, 2 * v, -2, 0.5 * v**2 - v * h / a, h / a - v, 1])
    P = np.reshape(vecsR, (3, 3))
    Pm = np.reshape(vecsL, (3, 3))
    return P, Pm

def WENO_Roe(u):
    Nx = len(u[0,:])
    up05m = u.copy()
    up05p = np.roll(u, -1, axis=1)
    w = u.copy()
    for j in range(2, Nx - 3):
        # Roe average
        P, Pm = Roe(u[:, j], u[:, j + 1])
        # local characteristic variables
        for k in range(j - 2, j + 4):
            w[:, k] = np.dot(Pm, u[:, k])
        wp05m = w[:, j].copy()
        wp05p = w[:, j + 1].copy()
        for i in range(3):
            wp05m[i] = reconstructWENO(w[i, j - 2], w[i, j - 1], w[i, j], w[i, j + 1], w[i, j + 2])
            wp05p[i] = reconstructWENO(w[i, j + 3], w[i, j + 2], w[i, j + 1], w[i, j], w[i, j - 1])
        up05m[:, j] = np.dot(P, wp05m)
        up05p[:, j] = np.dot(P, wp05p)
    return up05m, up05p

def RK3(L, u, dt):
    # 1
    u1 = u + dt*L(u)
    # 2
    u2 = 0.25*(3*u + u1 + dt*L(u1))
    # 3
    up = (u + 2*u2 + 2*dt*L(u2))/3
    return up

def RK4(L, u, dt):
    # 1
    u1 = u + 0.5*dt*L(u)
    # 2
    u2 = u + 0.5*dt*L(u1)
    # 3
    u3 = u + dt*L(u2)
    #4
    up = (-u + u1 + 2*u2 + u3 + 0.5*dt*L(u3))/3
    return up

"""
def RK1D(L, u, dt): # clone of RK4 with 1D data format
    # reshaping input data
    Nx = len(u[0,:])
    u1D = np.reshape(u, (1,3*Nx))[0,:]
    def L1D(u1D):
        u = np.reshape(u1D, (3,Nx))
        rhs = np.reshape(L(u), (1,3*Nx))[0,:]
        return rhs
    # RK4 iterations
    up = RK4(L1D, u1D, dt)
    # reshaping output data
    return np.reshape(up, (3,Nx))
"""
# ----------------------------------------------------------------
# Conservative - primitive conversion
# ----------------------------------------------------------------

def Cons2Prim(u):
    if isinstance(u[0], float):
        rho = u[0]
        v   = u[1]/u[0]
        p   = (gam-1)*(u[2] - 0.5*u[1]**2/u[0])
    else:
        rho = u[0,:]
        v   = u[1,:]/u[0,:]
        p   = (gam-1)*(u[2,:] - 0.5*u[1,:]**2/u[0,:])
    return np.array([rho, v, p])

def Prim2Cons(u):
    if isinstance(u[0], float):
        rho  = u[0]
        rhov = u[0]*u[1]
        E    = 0.5*u[0]*u[1]**2 + u[2]/(gam-1)
    else:
        rho  = u[0,:]
        rhov = u[0,:]*u[1,:]
        E    = 0.5*u[0,:]*u[1,:]**2 + u[2,:]/(gam-1)
    return np.array([rho, rhov, E])

# ----------------------------------------------------------------
# Exact Riemann solution
# ----------------------------------------------------------------

def RiemannExact(UJ, gam, t):
    # Initial function arguments
    uprimL = Cons2Prim(UJ[:,0])
    rhoL = uprimL[0]
    uL = uprimL[1]
    pL = uprimL[2]
    uprimR = Cons2Prim(UJ[:,1])
    rhoR = uprimR[0]
    uR = uprimR[1]
    pR = uprimR[2]
    
    # useful quantities (cf. Toro p. 119)
    aL = np.sqrt(gam*pL/rhoL)
    aR = np.sqrt(gam*pR/rhoR)
    AL = 2/((gam+1)*rhoL)
    AR = 2/((gam+1)*rhoR)
    BL = (gam-1)/(gam+1)*pL
    BR = (gam-1)/(gam+1)*pR
    du = uR - uL
    
    def fL(p):
        return (p-pL)*np.sqrt(AL/(p+BL))*(p>pL) + 2*aL/(gam-1)*((p/pL)**((gam-1)/(2*gam)) - 1)*(p<=pL)
    
    def fR(p):
        return (p-pR)*np.sqrt(AR/(p+BR))*(p>pR) + 2*aR/(gam-1)*((p/pR)**((gam-1)/(2*gam)) - 1)*(p<=pR)
    
    def f(p):
        return fL(p) + fR(p) + du
    
    # Solution output
    if uR-uL > 2*(aL+aR)/(gam-1):
        print('Warning: Vacuum is created, i.e., pressure positivity is violated!')
        us = 0
        rholeft = lambda x: 0
        uleft = lambda x: 0
        pleft = lambda x: 0
        rhoright = lambda x: 0
        uright = lambda x: 0
        pright = lambda x: 0
    else:
        # star region
        p0 = max([np.finfo(float).eps, 0.5*(pL+pR)- du*(rhoL+rhoR)*(aL+aR)/8])
        ps = fsolve(f, p0)
        us = 0.5*(uR+uL) + 0.5*(fR(ps)-fL(ps))
        # solution types
        if ps>pL:
            # print('Left-going shock')
            rhoLs = rhoL * ( (gam-1)/(gam+1) + ps/pL )/( (gam-1)/(gam+1)*ps/pL + 1 )
            S1 = uL - aL*np.sqrt( (gam+1)/(2*gam)*ps/pL + (gam-1)/(2*gam) )
            rholeft = lambda x: rhoL*(x<S1*t) + rhoLs*(x>=S1*t)
            uleft = lambda x: uL*(x<S1*t) + us*(x>=S1*t)
            pleft = lambda x: pL*(x<S1*t) + ps*(x>=S1*t)
        else:
            # print('Left-going rarefaction')
            aLs = aL + (uL-us)*(gam-1)/2
            rhoLs = gam*ps/aLs**2
            rholeft = lambda x: rhoL*(x<(uL-aL)*t) + rhoLs*(x>=(us-aLs)*t) + rhoL*np.abs(2/(gam+1) + (gam-1)/((gam+1)*aL)*(uL-x/t))**(2/(gam-1))*(x>=(uL-aL)*t)*(x<(us-aLs)*t)
            uleft = lambda x: uL*(x<(uL-aL)*t) + us*(x>=(us-aLs)*t) + 2/(gam+1)*(aL + (gam-1)/2*uL + x/t)*(x>=(uL-aL)*t)*(x<(us-aLs)*t)
            pleft = lambda x: pL*(x<(uL-aL)*t) + ps*(x>=(us-aLs)*t) + pL*np.abs(2/(gam+1) + (gam-1)/((gam+1)*aL)*(uL-x/t))**(2*gam/(gam-1))*(x>=(uL-aL)*t)*(x<(us-aLs)*t)
        if ps>pR:
            # print('Right-going shock')
            rhoRs = rhoR * ( (gam-1)/(gam+1) + ps/pR )/( (gam-1)/(gam+1)*ps/pR + 1 )
            S3 = uR + aR*np.sqrt( (gam+1)/(2*gam)*ps/pR + (gam-1)/(2*gam) )
            rhoright = lambda x: rhoR*(x>S3*t) + rhoRs*(x<=S3*t)
            uright = lambda x: uR*(x>S3*t) + us*(x<=S3*t)
            pright = lambda x: pR*(x>S3*t) + ps*(x<=S3*t)
        else:
            # print('Right-going rarefaction')
            aRs = aR + (us-uR)*(gam-1)/2
            rhoRs = gam*ps/aRs**2
            rhoright = lambda x: rhoR*(x>(uR+aR)*t) + rhoRs*(x<=(us+aRs)*t) + rhoR*np.abs(2/(gam+1) - (gam-1)/((gam+1)*aR)*(uR-x/t))**(2/(gam-1))*(x<=(uR+aR)*t)*(x>(us+aRs)*t)
            uright = lambda x: uR*(x>(uR+aR)*t) + us*(x<=(us+aRs)*t) + 2/(gam+1)*(-aR + (gam-1)/2*uR + x/t)*(x<=(uR+aR)*t)*(x>(us+aRs)*t)
            pright = lambda x: pR*(x>(uR+aR)*t) + ps*(x<=(us+aRs)*t) + pR*np.abs(2/(gam+1) - (gam-1)/((gam+1)*aR)*(uR-x/t))**(2*gam/(gam-1))*(x<=(uR+aR)*t)*(x>(us+aRs)*t)
    
    def sol(x):
        if isinstance(x, float):
            UL = Prim2Cons(np.array([rholeft(x),uleft(x),pleft(x)]))
            UR = Prim2Cons(np.array([rhoright(x),uright(x),pright(x)]))
            u = UL[:,0]*(x<us*t) + UR[:,0]*(x>=us*t)
        else:
            Nx = len(x)
            u = np.zeros((3,Nx))
            for i in range(Nx):
                UL = Prim2Cons(np.array([rholeft(x[i]),uleft(x[i]),pleft(x[i])]))
                UR = Prim2Cons(np.array([rhoright(x[i]),uright(x[i]),pright(x[i])]))
                u[:,i] = UL[:,0]*(x[i]<us*t) + UR[:,0]*(x[i]>=us*t)
        return u
    return sol