# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 19:26:50 2021

@author: GroH Von Hilfiger
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.linalg import eigh_tridiagonal


## Define N(number of eigenstates) and dy (step size of solver)

## we know that N*dy = 1 ==> dy = 1/N

## boundary conditions are zero ==> psi(0)=psi(N)=0

N = 2000
dy = 1/N

y = np.linspace(0, 1, N+1)

#define dimensionless potential m*2*V

def mL2V_quadratic(y):
    return 1000*(y-1/2)**2

#potential with a gaussian spike

def mL2V_gaussian(y):
    return 1000 * np.exp(-(y-0.7)**2 / (2*0.05**2))

#plt.plot(y,mL2V_gaussian(y))
# V = mL2V(y)
# plt.plot(y,V)

diag = 1/dy**2 + mL2V_gaussian(y[1:-1])
e = -1/(2*dy**2) * np.ones(len(diag)-1)

#this function returns the eigenstates and eigenfunctions of a matrix defined by
# diag and e (Toeplitz matrix)

w, v = eigh_tridiagonal(diag, e)

#plot first few eigenfunctions
plt.figure()
for i in range(4):
    #plt.plot(y[1:-1],v[:,i], label=r'$\psi_{}$'.format(i)) #to show wavefunctions
    plt.plot(y[1:-1],v[:,i]**2, label=r'$\psi_{}^2$'.format(i)) #to show probabilities of finding the particle
    
plt.legend(loc=0)    

##Plot the energies
plt.figure()

plt.bar(np.arange(0,10,1),w[0:10])
plt.ylabel(r'$\frac{mL^2E}{\hbar^2}$',fontsize=15)
plt.xlabel(r'eigenstates')