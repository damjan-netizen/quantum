# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:16:32 2022

@author: GroH Von Hilfiger
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy import sparse as sparse
from scipy.sparse.linalg import eigs

### DEFINE CONSTANTS ####

hbar = const.hbar
e = const.e
m_e = const.m_e
pi = const.pi
epsilon_0 = const.epsilon_0
joul_to_eV = e


### discreatization parameters
N = 2000
l = 0

r = np.linspace(2e-9, 0.0, N, endpoint=False)

#### define matrices


def calculate_potential(r):
    potential = e**2/4/pi/epsilon_0 * (1/r)
    potential_mat = sparse.diags((potential))
    return potential_mat

def calculate_angular_term(r):
    angular = l*(l+1)/r**2
    angular_mat = sparse.diags((angular))
    return angular_mat

def calculate_laplacian_1D(r):
    h = r[1]-r[0]
    main_d = -2/h**2 * np.ones(N)
    off_d = 1/h**2 * np.ones(N-1)
    laplacian_mat = sparse.diags([main_d, off_d, off_d],(0,-1,1))
    return laplacian_mat

def build_hamiltonian(r):
    laplace_term = calculate_laplacian_1D(r)
    angular_term = calculate_angular_term(r)
    potential_term = calculate_potential(r)
    
    hamiltonian = -hbar**2 / (2.0 * m_e) * (laplace_term - angular_term) - potential_term
    return hamiltonian

def plot(r, densities, eigenvalues):
    plt.xlabel('x ($\\mathrm{\AA}$)')
    plt.ylabel('probability density ($\\mathrm{\AA}^{-1}$)')
     
    energies = ['E = {: >5.2f} eV'.format(eigenvalues[i].real / e) for i in range(3)]
    plt.plot(r * 1e+10, densities[0], color='blue',  label=energies[0])
    plt.plot(r * 1e+10, densities[1], color='green', label=energies[1])
    plt.plot(r * 1e+10, densities[2], color='red',   label=energies[2])
     
    plt.legend()
    plt.show()
    return


hamiltonian = build_hamiltonian(r)
 
""" solve eigenproblem """
number_of_eigenvalues = 30
eigenvalues, eigenvectors = eigs(hamiltonian, k=number_of_eigenvalues, which='SM')
 
""" sort eigenvalue and eigenvectors """
eigenvectors = np.array([x for _, x in sorted(zip(eigenvalues, eigenvectors.T), key=lambda pair: pair[0])])
eigenvalues = np.sort(eigenvalues)
 
""" compute probability density for each eigenvector """
densities = [np.absolute(eigenvectors[i, :])**2 for i in range(len(eigenvalues))]
 
""" plot results """
plot(r, densities, eigenvalues)