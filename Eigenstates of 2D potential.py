# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:18:20 2022

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

#### D is an NxN matrix with -2 on the main diagonal and 1 on off diagonals

def get_potential(x,y):
    return 0*x

V = get_potential(X,Y)

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

plt.figure(figsize=(10,8))
plt.contourf(X,Y,get_e(5)**2,20)
plt.colorbar(label='probability')


## animate the solution


# plt.style.use(['default'])
# fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

# def init():
#     ax.plot_surface(X, Y, get_e(5)**2, cmap='coolwarm',
#                     linewidth=0, antialiased=False)
#     ax.set_xlabel(r'$x/a$')
#     ax.set_ylabel(r'$y/a$')
#     ax.set_zlabel(r'$\propto|\psi|^2$')
#     return fig

# def animate3D(i):
#     ax.view_init(elev=20, azim = 4*i)
#     return fig

# ani = animation.FuncAnimation(fig, animate3D, init_func = init,
#                               frames=90, interval=50)
# ani.save('eigenfunction in 2D rotation.gif', writer='pillow',fps=20)

import plotly.graph_objects as go
from plotly.offline import plot

data = go.Surface(x = X, y = Y, z = get_e(5)**2)



layout = go.Layout(scene=dict(xaxis_title=r'x',
          yaxis_title=r'y',
          zaxis_title=r'z',
          aspectratio=dict(x=1, y=1, z=1),
          camera_eye=dict(x=1.2, y=1.2, z=1.2)))


fig = go.Figure(data=data, layout = layout)

fig.update_layout(title='$\psi^2$', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

plot(fig, auto_open=True, filename=r'plotly/eigenfunstions.html')
 