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

def coeff(t,l,ynorm_spec,eig_final):
    val = 0
    for n in range(len(eig_final[l-2])):
        phi_val = ynorm_spec[l-2][n]
        val += phi_val*np.exp(-1j*eig_final[l-2][n]*t)
    return val

def coeff_neg(t,l,ynorm_spec,eig_final_neg):
    val = 0
    for n in range(len(eig_final_neg[l-2])):
        phi_val = ynorm_spec[l-2][n]
        val += phi_val*np.exp(-1j*eig_final_neg[l-2][n]*t)
    return val

if __name__ == "__main__":
    l_max = 3
    ynorm_spec = np.load('ynorm_spec_file.npy')
    eig_final = np.load('eig_final_file.npy')
    eig_final_neg = np.load('eig_final_neg_file.npy')
    delta_arr = np.load('delta_arr_file.npy') 
    delta_arr_neg = np.load('delta_arr_neg_file.npy')
    J2planet = 14696.5063e-6 # J2 of Jupiter
    J4planet = -586.6085e-6 # J4 of Jupiter
    Mplanet = 1. #0.000954588 # Mass of Jupiter in solar masses
    Rplanet = 1. #0.00046732617 # Radius of Jupiter in AU
    ObliquityPlanet = 0. # Obliquity of the Planet
    tmax = 5.7e2 # Maximum integration time
    e = 0.95
    inc = np.pi/2
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.add(m=Mplanet, hash="Jupiter")
    sim.add(a=20*Rplanet,e=e,inc=inc,hash="Juno")
    sim.move_to_com() 
    ps = sim.particles

    def PHI_FORCE(reb_sim,rebx_force,particles,N):
        sim = reb_sim.contents
        phi_f = rebx_force.contents
        R_eq = 1
        x_pos = particles[1].x
        y_pos = particles[1].y
        z_pos = particles[1].z
        r = np.sqrt(x_pos**2+y_pos**2+z_pos**2)
        theta = np.abs(np.arctan2(np.sqrt(x_pos**2+y_pos**2),z_pos))
        phi = np.abs(np.arctan2(y_pos,x_pos))
        t = sim.t
        Phi_f_rad = 0
        Phi_f_theta = 0
        Phi_f_phi = 0
        for l in range(2,l_max):
            l_plus = l+1
            l_minus = l-1
            for m in range(-l,l+1):
                Phi_f_rad += -np.real((-l-1)*(r/R_eq)**(-l-2)*sph_harm(m,l,phi,theta)*(coeff(t,l,ynorm_spec,eig_final)*np.exp(1j*delta_arr[l-2][m+l])+coeff_neg(t,l,ynorm_spec,eig_final_neg)*np.exp(1j*delta_arr_neg[l-2][m+l]))/R_eq)

                if l_minus < np.abs(m):
                    Phi_f_theta += -np.real((r/R_eq)**(-l-1)*(l*(((l+1)**2-m**2)/(4*(l+1)**2-1))**(1/2)*sph_harm(m,l_plus,phi,theta))*(coeff(t,l,ynorm_spec,eig_final)*np.exp(1j*delta_arr[l-2][m+l])+coeff_neg(t,l,ynorm_spec,eig_final_neg)*np.exp(1j*delta_arr_neg[l-2][m+l]))/np.sin(theta)/r)

                else:
                    Phi_f_theta += -np.real((r/R_eq)**(-l-1)*(l*(((l+1)**2-m**2)/(4*(l+1)**2-1))**(1/2)*sph_harm(m,l_plus,phi,theta)-(l+1)*((l**2-m**2)/(4*l**2-1))**(1/2)*sph_harm(m,l_minus,phi,theta))*(coeff(t,l,ynorm_spec,eig_final)*np.exp(1j*delta_arr[l-2][m+l])+coeff_neg(t,l,ynorm_spec,eig_final_neg)*np.exp(1j*delta_arr_neg[l-2][m+l]))/np.sin(theta)/r)

                Phi_f_phi += -np.real((r/R_eq)**(-l-1)*1j*m*sph_harm(m,l,phi,theta)*(coeff(t,l,ynorm_spec,eig_final)*np.exp(1j*delta_arr[l-2][m+l])+coeff_neg(t,l,ynorm_spec,eig_final_neg)*np.exp(1j*delta_arr_neg[l-2][m+l]))/np.sin(theta)/r)

        particles[1].ax += Phi_f_rad*np.sin(theta)*np.cos(phi)+Phi_f_theta*np.cos(theta)*np.cos(phi)-Phi_f_phi*np.sin(phi)
        particles[1].ay += Phi_f_rad*np.sin(theta)*np.sin(phi)+Phi_f_theta*np.cos(theta)*np.sin(phi)+Phi_f_phi*np.cos(phi)
        particles[1].az += Phi_f_rad*np.cos(theta)-Phi_f_theta*np.sin(theta)

    Noutputs = 10**5
    rebx = reboundx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    # rebx.add_force(gh)
    ps["Jupiter"].params["J2"] = J2planet
    ps["Jupiter"].params["J4"] = J4planet
    ps["Jupiter"].params["R_eq"] = Rplanet

    myforce = rebx.create_force("Phi_f")
    myforce.force_type = "pos"
    myforce.update_accelerations = PHI_FORCE
    rebx.add_force(myforce)
 
    times = np.linspace(0,1*tmax,Noutputs) 
    t_arr = np.zeros(Noutputs)
    filename = 'simulation_without_J2J4.bin'
    for i,time in enumerate(times):
        sim.integrate(time,exact_finish_time=False)
        sim.simulationarchive_snapshot(filename)
        t_arr[i] = sim.t
        print(sim.t/times[-1]*100)
   
    # sa = rebound.SimulationArchive(filename)
   
    J2planet = 14696.5063e-6 # J2 of Jupiter
    J4planet = -586.6085e-6 # J4 of Jupiter
    Mplanet = 1. #0.000954588 # Mass of Jupiter in solar masses
    Rplanet = 1. #0.00046732617 # Radius of Jupiter in AU
    ObliquityPlanet = 0. # Obliquity of the Planet
    tmax = 5.7e2 # Maximum integration time
    e = 0.95
    inc = np.pi/2
    sim = rebound.Simulation()
    # sim.units = ('yr','AU','Msun')
    # G = sim.G
    sim.integrator = "whfast"
    sim.add(m=Mplanet, hash="Jupiter")
    sim.add(a=20*Rplanet,e=e,inc=inc,hash="Juno")
    sim.move_to_com() 
    ps = sim.particles

    Noutputs = 10**5
    rebx = reboundx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    # rebx.add_force(gh)
    ps["Jupiter"].params["J2"] = J2planet
    ps["Jupiter"].params["J4"] = J4planet
    ps["Jupiter"].params["R_eq"] = Rplanet

    times = t_arr 
    filename = 'base_simulation_without_J2J4.bin'
    for i,time in enumerate(times):
        sim.integrate(time,exact_finish_time=True)
        sim.simulationarchive_snapshot(filename)
        print(sim.t/times[-1]*100)

    # sa = rebound.SimulationArchive(filename)