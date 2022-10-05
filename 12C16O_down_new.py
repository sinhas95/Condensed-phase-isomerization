#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:16:15 2022

@author: shreya sinha
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline


au_to_fs=2.41888e-2
hour_to_fs = 3.6e+18 
au_to_eV=27.2114
eV_to_cm = 8065.5
au_to_cm=219474.63068
atomictime_to_s = 2.4188843e-17
hbar2 = 5.3087e-12 # cm-1*s
hbar = 2.4188843e-17 # hartree*s
np_grid = 1800
nz_grid = 1400
bath_grid = 100
n_dim = np_grid * nz_grid

evals = np.loadtxt('vals.dat', usecols=(2,), dtype=float)
Z_grid = np.loadtxt('Z_grid_1400.dat', usecols=(0,), dtype=float)
theta_grid = np.loadtxt('theta_grid_1800.dat', usecols=(0,), dtype=float)
vdos_nacl_grid = np.loadtxt('real_vdos_3CO_nacl.dat', usecols=(0,), dtype=float)
vdos_nacl_orig = np.loadtxt('real_vdos_3CO_nacl.dat', usecols=(1,), dtype=float) 
debye_freq = 222 #cm-1
for ix in range(len(vdos_nacl_orig)):
    vdos_nacl_orig[ix] = np.abs(vdos_nacl_orig[ix])
    
spline_integrand = UnivariateSpline(vdos_nacl_grid, vdos_nacl_orig, k=3, s=0)

for ix in range(len(Z_grid)):
    Z_grid[ix] = Z_grid[ix] * 1.8897261
#------------------------------------------------------------------------------------------
def boltz_eins_fac(inv_temp, omega):
    s = 1./(np.exp(omega*inv_temp) - 1.)
    return s
#------------------------------------------------------------------------------

Nz = nz_grid
Np = np_grid
zmax = 11.338401
zmin = 4.3464835
d_Z = (zmax-zmin)/Nz
d_theta = 2.*np.pi/Np

epsilon = 1e-12 # scaling factor

r_eq = 2.1567444 # bohrradius , C-O equilibrium bond lengeth

v_int = np.zeros([n_dim], dtype=float)
for ix in range(0, np_grid):
    for iy in range(0, nz_grid):
        p = ix*nz_grid + iy
        v_int[p] = (Z_grid[iy]-5.8619304) + ((theta_grid[ix]+np.pi) * r_eq)

temp = [19., 20., 21., 25., 26., 27., 28., 30., 34., 36., 38., 40.,  5., 7., 10., 12., 14., 16.]
t = [19., 20., 21., 25., 26., 27., 28., 30., 34., 36., 38., 40.,  5., 7., 10., 12., 14., 16.]
for it in range(0, len(temp)):
    temp[it] = temp[it] * 0.69503078  #convert into thermal energies cm-1
    
inv_temperature = np.zeros([len(temp)], dtype = float)
for it in range(0, len(temp)):
    inv_temperature[it] = 1./temp[it]

evals_i = [22, 27, 30, 34, 36, 39, 42, 46, 47, 48, 51, 54, 58, 59, 60, 63, 66, 69, 71, 73, 74, 75, 77, 81, 84, 85, 87, 89, 90, \
           91, 94, 96, 99, 102, 103, 104, 105, 106, 109, 110, 113, 118, 119, 121, 122, 123, 125, 126, 127, 129, 132, 134,\
           138, 139, 141, 142, 144, 145, 146, 147]    # list of "O-down" right well states
evals_f = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 37, 38, 40, 41, 43, 44, 45, \
         49, 50, 52, 53, 55, 56, 57, 61, 62, 64, 65, 67, 68, 70, 72, 76, 78, 79, 80, 82, 83, 86, 88, 92, 93, 95, 97, 98, 100, \
         101, 107, 108, 111, 112, 114, 115, 116, 117, 120, 124, 128, 130, 131, 133, 135, 136, \
         137, 140, 143] # list of "C-down" left well states
gs = evals_i[0]

trans_matrix = np.zeros([len(evals_i), len(evals_f)], dtype = float)
for ix in range(len(evals_i)):
      a = evals_i[ix]
      eig_i = np.loadtxt('Eig_%d.dat'%(a), usecols=(2,), dtype=float)
      mag_i = 0.
      for ip in range(0, n_dim):
        mag_i += eig_i[ip] * eig_i[ip] * d_Z * d_theta
      eig_i = eig_i/np.sqrt(mag_i)
      for iy in range(len(evals_f)):
       b = evals_f[iy]
       if a > b and (evals[a] - evals[b]) <= 222. :
           eig_f = np.loadtxt('Eig_%d.dat'%(b), usecols=(2,), dtype=float) 
           mag_f = 0.
           for ir in range(0, n_dim):
             mag_f += eig_f[ir] * eig_f[ir] * d_Z * d_theta
           eig_f = eig_f/np.sqrt(mag_f)
           print mag_i, mag_f
           for il in range(0, n_dim):
                   trans_matrix[ix, iy] += (eig_f[il] * v_int[il] * eig_i[il])* d_Z * d_theta 

np.savetxt('12C16O_trans_matrix_down.dat', trans_matrix)

for im in range(0,len(temp)): 
  td_rate = np.zeros([len(evals_i), len(evals_f)], dtype = float)
  m = t[im]
  for ix in range(len(evals_i)):
      a = evals_i[ix]
      for iy in range(len(evals_f)):
       b = evals_f[iy]
       if a > b and (evals[a] - evals[b]) <= 222. :
           E_i = evals[a]
           E_f = evals[b]
           print E_i, E_f
           omega_k = E_i-E_f
           vdos_k = spline_integrand(omega_k)
           print omega_k, vdos_k

           td_rate[ix, iy] +=  trans_matrix[ix, iy]**2 * \
                                (boltz_eins_fac(inv_temperature[im], omega_k) + 1.)\
                                  * (2. * epsilon / hbar) * vdos_k \
                                  * np.exp(-(E_i-evals[gs]) * inv_temperature[im]) 
               
          
  #print td_rate 
  np.savetxt('12C16O_matrix_rate_%i_K_alldown.dat'%m, td_rate)
  fig, axes = plt.subplots(1,1,figsize = (100,100))
  im = axes.imshow(td_rate, cmap=plt.cm.Blues)
#plt.imshow(grid, interpolation='none')
  plt.xticks(range(len(evals_f)), evals_f, fontsize=40)
  plt.yticks(range(len(evals_i)), evals_i, fontsize=40)
  axes.grid()
  fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
  fmt.set_powerlimits((0, 0))
#plt.imshow(np.random.random((50,50)))
  cbar = plt.colorbar(im, format=fmt, orientation='horizontal')
 # for t in cbar.ax.get_yticklabels():
     #t.set_fontsize(20)
  cbar.ax.tick_params(labelsize=70)
  cbar.ax.xaxis.get_offset_text().set_fontsize(80)
  #plt.show()
  plt.savefig('12C16O_matrix_rate_%i_K_alldown.png'%m, format='png')
  
