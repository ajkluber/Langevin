""" Numerically solve the Langevin equation

References:
1. Tuckerman, M.E.; Statistical Mechanics: Theory and Molecular Simulation; 2010

"""


import numpy as np
import matplotlib.pyplot as plt

def Harmonic_V(x,omega):
    return 0.5*(omega**2)*(x**2)

def Harmonic_f(mu,omega):
    return lambda x: (-1./mu)*(omega**2)*x

def Double_well_V(x,A,x0):
    return (A/(x0**4))*(x**2 - x0**2)**2

def Double_well_f(mu,A,x0):
    return lambda x: (-4.*A/(mu*(x0**4)))*x*(x**2 - x0**2)

if __name__ == "__main__":
    nsteps = 1000000
    kbT = 1.
    mu = 1.
    gamma = 1.
    omega = 1.
    dt = 0.01
    sigma = np.sqrt(2.*kbT*gamma/mu)

    x = np.zeros(nsteps,float)
    v = np.zeros(nsteps,float)

    xi = np.random.normal(0,1,nsteps)
    theta = np.random.normal(0,1,nsteps)

    flag = 1

    if flag == 1:
        f = Harmonic_f(mu,omega)
        V = lambda x: Harmonic_V(x,omega)
    elif flag == 2:
        x0 = 5.
        Amp = 1.
        f = Double_well_f(mu,Amp,x0)
        V = lambda x: Double_well_V(x,Amp,x0)

    
    for i in xrange(nsteps - 1):
        A = 0.5*(dt**2)*(f(x[i]) - gamma*v[i]) + sigma*(dt**(3./2))*(0.5*xi[i] + np.sqrt(3./36.)*theta[i])

        x[i + 1] = x[i] + dt*v[i] + A
        v[i + 1] = v[i] + 0.5*dt*(f(x[i + 1]) + f(x[i])) - dt*gamma*v[i] + sigma*np.sqrt(dt)*xi[i] - gamma*A

    plt.figure()
    plt.plot(x)

    plt.figure()
    x -= np.mean(x)
    n,bins = np.histogram(x,bins=50)
    bin_avg = 0.5*(bins[1:] + bins[:-1])
    pmf = -np.log(n)
    pmf -= min(pmf)
    plt.plot(bin_avg,pmf)
    plt.plot(bin_avg,V(bin_avg),'g',lw=2)
    plt.show()


