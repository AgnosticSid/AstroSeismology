import numpy as np
import matplotlib.pyplot as plt
import pygyre
from scipy.linalg import eig, eigvals
import scipy.integrate
import os
import pandas as pd
from scipy.interpolate import CubicSpline
import rebound
import reboundx
import datetime
from matplotlib import animation
from scipy.special import sph_harm
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from scipy.signal import find_peaks

def Eigenf_spec(N,l):
    M = 4/np.pi
    G = 1
    x_arr = (1/2)*(1-np.cos(np.pi*np.arange(0,N,1)/(N-1)))

    dPhi_0_dr = (np.sin(np.pi*x_arr)/(np.pi*x_arr**(2))-np.cos(np.pi*x_arr)/x_arr)
    M_phi = np.diag(dPhi_0_dr)
    M_phi[0,0] = 0

    dlnrho_dr = np.pi/np.tan(np.pi*x_arr)-1/x_arr
    dlnrho_dr[0] = 0
    dlnrho_dr[-1] = CubicSpline(x_arr[:-1],dlnrho_dr[:-1],extrapolate=True)(1.)*2
    M_lnrho = np.diag(dlnrho_dr)
    M_lnrho[0,0] = 0

    M_d_dr = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            p_i = 2 if i in [0,N-1] else 1

            p_j = 2 if j in [0,N-1] else 1
            
            if i==0 and j==0:
                M_d_dr[i,j]=-(1/3)*(1+2*(N-1)**2)
            elif i==N-1 and j==N-1:
                M_d_dr[i,j]=(1/3)*(1+2*(N-1)**2)
            elif i==j and 0<j<N-1:
                M_d_dr[i,j] = np.cos(np.pi*j/(N-1))/(1-(np.cos(np.pi*j/(N-1)))**2)
            elif i!=j:
                M_d_dr[i,j] = (2*(-1)**(i+j+1)*p_i)/(p_j*(np.cos(np.pi*i/(N-1))-np.cos(np.pi*j/(N-1)))) 

    M_d2_dr2 = np.matmul(M_d_dr,M_d_dr)

    Mat_r = np.diag(1/x_arr)
    Mat_r[0,0] = 0

    rho_0 = np.sinc(x_arr)/M
    M_rho_0 = np.diag(rho_0)

    P0_rho0 = np.sinc(x_arr)
    M_P0_rho0 = np.diag(P0_rho0)

    M_nabla = M_d2_dr2+2*np.matmul(Mat_r,M_d_dr)

    M_zero = np.zeros((N,N))
    M_B1 = np.diag(np.ones(N))
    M_B = np.block([[M_B1,M_zero,M_zero,M_zero,M_zero],
                    [M_zero,M_B1,M_zero,M_zero,M_zero],
                    [M_zero,M_zero,M_B1,M_zero,M_zero],
                    [M_zero,M_zero,M_zero,M_B1,M_zero],
                    [M_zero,M_zero,M_zero,M_zero,M_zero]
                    ])

    M_L = np.block([[M_zero,M_zero,M_phi,M_d_dr+M_lnrho,M_d_dr],
                    [M_zero,M_zero,M_zero,Mat_r,Mat_r],
                    [-M_lnrho-M_d_dr-2*Mat_r,l*(l+1)*Mat_r,M_zero,M_zero,M_zero],
                    [M_phi-np.matmul(M_P0_rho0,M_d_dr)-np.matmul(M_P0_rho0,2*Mat_r),np.matmul(M_P0_rho0,l*(l+1)*Mat_r),M_zero,M_zero,M_zero],
                    [M_zero,M_zero,-4*np.pi*M_rho_0,M_zero,M_nabla-l*(l+1)*np.matmul(Mat_r,Mat_r)]
                    ])

    for i in range(5):
        M_L[i*N,:] = 0
        M_B[i*N,:] = 0
        M_L[i*N,i*N] = 1

    M_L[4*N-1,:] = 0
    M_B[4*N-1,:] = 0
    M_L[4*N-1,N-1] = dPhi_0_dr[-1]
    M_B[4*N-1,4*N-1] = 1

    M_L[5*N-1,:] = 0
    M_B[5*N-1,:] = 0
    M_L[5*N-1,4*N:5*N] = M_d_dr[-1,:]
    M_L[5*N-1,5*N-1] += (l+1)/x_arr[-1]

    return eig(M_L,M_B)

def v(f,v_bias,v_max,f_max,sigma_f):
    return v_bias+(v_max-v_bias)*np.exp(-1/2*((f-f_max)/sigma_f)**(2))

def amplitudes(l_max,N):
    ynorm_spec = []
    xnorm_spec = []
    for l in range(2,l_max):
        R_J = 71492e+5 #cm
        M_J = 1898.13e+27 #g
        G = 6.6726e-8 #cgs units
        omega_J = np.sqrt(G*M_J/R_J**(3))
        M = 4/np.pi
        R = 1
        eigenval2 = Eigenf_spec(N,l)[0]
        eigenfunc2 = Eigenf_spec(N,l)[1]
        x_arr = (1/2)*(1-np.cos(np.pi*np.arange(0,N,1)/(N-1)))
        rho_0 = np.sinc(x_arr)/M
        eig_y = eigenval2[(eigenval2>1.0)*(eigenval2<5.0)]
        eig_y = np.sort(eig_y)
        c = 4*np.pi*R**(3)
        amp_Spec = []
        x_Spec = []
        v_bias = 30 #cm/s
        v_max = 50 #cm/s
        f_max = 1210 #uHz
        sigma_f = 300 #uHz
        for j in eig_y:
            input_Spec = j/(2*np.pi)*omega_J*10**(6)
            E = c*scipy.integrate.simps(((eigenfunc2[0*N:1*N,np.argmin(np.abs(eigenval2-j))]/j)**(2)+l*(l+1)*(eigenfunc2[1*N:2*N,np.argmin(np.abs(eigenval2-j))]/j)**(2))*rho_0*x_arr**(2),x_arr)
            x_Spec.append(l)
            amp_Spec.append(v(input_Spec,v_bias,v_max,f_max,sigma_f)/(np.abs(eigenfunc2[0*N:1*N,np.argmin(np.abs(eigenval2-j))][-1]/(np.sqrt(E)))*R_J*omega_J))

        amp_Spec = np.array(amp_Spec)
        x_Spec = np.array(x_Spec)

        y_plot_spec = []
        for i,j in zip(range(29),eig_y):
            E = c*scipy.integrate.simps(((eigenfunc2[0*N:1*N,np.argmin(np.abs(eigenval2-j))]/j)**(2)+l*(l+1)*(eigenfunc2[1*N:2*N,np.argmin(np.abs(eigenval2-j))]/j)**(2))*rho_0*x_arr**(2),x_arr)
            y_plot_spec.append(np.abs(amp_Spec[i]*eigenfunc2[4*N:5*N,np.argmin(np.abs(eigenval2-j))][-1]/np.sqrt(E)))

        y_plot_spec = np.array(y_plot_spec)

        ynorm_spec.append(y_plot_spec)
        xnorm_spec.append(x_Spec)

    return ynorm_spec

def delta(l,m,eig_final):
    delta_val = 0
    for n in range(len(eig_final[l-2])):
        delta_val += np.random.random()*2*np.pi/eig_final[l-2][n]
    return delta_val

def delta_neg(l,m,eig_final_neg):
    delta_val = 0
    for n in range(len(eig_final_neg[l-2])):
        delta_val += np.random.random()*2*np.pi/np.abs(eig_final_neg[l-2][n])
    return delta_val

if __name__ == "__main__": 
    N = 200
    l_max = 3
    eig_final = []
    eig_final_neg = []
    delta_arr = []
    delta_arr_neg = []
    for l in range(2,l_max):
        eigenval2 = Eigenf_spec(N,l)[0]
        eig_y = eigenval2[(eigenval2>1.0)*(eigenval2<5.0)]
        eig_y = np.sort(eig_y)
        eig_final.append(eig_y)
        eig_final_neg.append(-eig_y)
        delta_vals = []
        delta_vals_neg = [] 
        for m in range(-l,l+1):
            delta_vals.append(delta(l,m,eig_final))
            delta_vals_neg.append(delta_neg(l,m,eig_final_neg))
        delta_arr.append(delta_vals)
        delta_arr_neg.append(delta_vals_neg)
    ynorm_spec = amplitudes(l_max,N)
    home = os.path.expanduser("~")
    logs_dir = os.path.join(home,'Downloads','gyre-7.0','Github_upload')
    np.save(os.path.join(logs_dir,'ynorm_spec_file'),ynorm_spec)
    np.save(os.path.join(logs_dir,'eig_final_file'),eig_final)
    np.save(os.path.join(logs_dir,'eig_final_neg_file'),eig_final_neg)
    np.save(os.path.join(logs_dir,'delta_arr_file'),delta_arr)
    np.save(os.path.join(logs_dir,'delta_arr_neg_file'),delta_arr_neg)