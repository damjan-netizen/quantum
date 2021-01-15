# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:37:51 2021

@author: Network Damjan
"""

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import linalg as ln
from scipy import sparse as sparse
import matplotlib.animation as animation


class Wave_packet:
    def __init__(self, n_points, dt, sigma0=0.5, k0=1.0, x0=-150.0,
                 x_begin=200.0, x_end=200.0, barrier_height=1.0, barrier_width=3.0):
        self.n_points = n_points
        self.sigma0 = sigma0
        self.k0 = k0
        self.x0 = x0
        self.dt = dt
        self.prob = sp.zeros(n_points)
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        
        #space discretization
        self.x, self.dx = sp.linspace(x_begin, x_end, n_points, retstep=True)
        
        #Initialiyation of the wave function to Gaussian wave packet
        
        norm = (2.0*sp.pi*sigma0**2)**(-0.25)
        self.psi = sp.exp(-(self.x - x0)**2/(4.0*sigma0**2))
        self.psi *= sp.exp(1.0j * k0 *self.x)
        self.psi *= (2.0*sp.pi*sigma0**2)**(-0.25)
        
        #setting up the potential barrier
        
        self.potential = sp.array(
            [barrier_height if 0.0 < x < barrier_width else 0.0 for x in self.x])
        
        # Creating the Hamiltonian
        
        h_diag = sp.ones(n_points)
        h_non_diag = sp.ones(n_points -1)*(-0.5/self.dx**2)
        hamiltonian = sparse.diags([h_diag, h_non_diag, h_non_diag], [0,1,-1])
        
        #computing the Crank-Nicolson time evolution matrix
        implicit = (sparse.eye(self.n_points) - dt / 2.0j * hamiltonian).tocsc()
        explicit = (sparse.eye(self.n_points) + dt / 2.0j * hamiltonian).tocsc() 
        self.evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()
        
wave_packet = Wave_packet(n_points=500, dt=0.5, barrier_width=10, barrier_height=1)