# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:53:47 2022

@author: GroH Von Hilfiger
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

plt.style.use(['science', 'notebook'])

from scipy import sparse

N = 150

X , Y = np.meshgrid(np.linspace(0,1,N,dtype=float),
                    np.linspace(0,1,N,dtype=float))





##### Define potential funciton ########

def gaussian_bump(x,y):
    return np.exp(-(x-0.3)**2/(2*0.1**2))*np.exp(-(y-0.3)**2/(2*0.1**2))



V = gaussian_bump(X,Y)

diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)

#kinetic energy matrix 
T = -1/2 * sparse.kronsum(D,D)

#potential energy matrix
U = sparse.diags(V.reshape(N**2),0) #this means that we put the elements of V on the main diag of U

#Hamiltonian

H = T + U


# Now we can solve this eigenvalue problem

w, v = eigsh(H, k=10, which='SM') #SM == smallest eigenvalues in magnitude

def get_e(n):
    return v.T[n].reshape(N,N) #to get this back into matrix NxN form


##### get user input

energy_level = input("Please enter the energy level : ")
n = int(energy_level)


plt.figure(figsize=(10,8))
plt.contourf(X,Y,get_e(n)**2,20)
plt.colorbar(label='probability')


### Plot in 3D with plotly

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

data_wave = go.Surface(x = X, y = Y, z = get_e(n)**2)

data_potential = go.Surface(x = X, y = Y, z = gaussian_bump(X, Y),colorbar_x=-0.07)



layout = go.Layout(scene=dict(xaxis_title=r'x',
          yaxis_title=r'y',
          zaxis_title=r'z',
          aspectratio=dict(x=1, y=1, z=1),
          camera_eye=dict(x=1.2, y=1.2, z=1.2)))


fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Potetntial function in 2D space', 
                                    'Wavefunction squared\n n = {}'.format(n)],
                    )
fig.add_trace(data_wave,1,2)
fig.add_trace(data_potential,1,1)
fig.update_layout(title='Wavefunctions and potentials', width=1000, height=750)

plot(fig, auto_open=True, filename=r'plotly/eigenfunction n=10, gaussian potential.html')