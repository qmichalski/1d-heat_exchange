# -*- coding: utf-8 -*-
"""
Created on Friday 19 2023

@author: Dr Quentin Michalski
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def calc_drhodP_H(fluid):
    '''
    Calculate derivative drho/dP at constant H

    Parameters
    ----------
    fluid : TYPE
        Cantera fluid.

    Returns
    -------
    drhodP_H.

    '''
    H0,P0 = fluid.HP
    density0 = fluid.density
    P1 = P0*(1+1e-7)
    H1 = H0
    fluid.HP = H1,P1
    density1 = fluid.density
    drhodP_H = (density1-density0)/(P1-P0)
    return(drhodP_H)

def calc_drhodH_P(fluid):
    '''
    Calculate derivative drho/dH at constant P

    Parameters
    ----------
    fluid : TYPE
        Cantera fluid.

    Returns
    -------
    drhodH_P.

    '''
    H0,P0 = fluid.HP
    density0 = fluid.density
    H1 = H0*(1+1e-7)
    P1 = P0
    fluid.HP = H1,P1
    density1 = fluid.density
    drhodH_P = (density1-density0)/(H1-H0)
    return(drhodH_P)

def dzdx_flow(x,z,parameters,fullOutput=False):
    P = z[0][0]
    h = z[1][0]
    U = z[2][0]
    fluid = parameters['fluidObject']
    fluid.HP = h,P # set fluid at state values
    T = fluid.T # get temperature
    fluid.PQ = P,0 # get saturation liquid state
    TBoil = fluid.T_sat # get boiling temperature
    # test so calculation can go through even though it's boiling at the wall
    # if that's the case, calculation is no good!
    if T > TBoil - 1:
        T = TBoil + 1
    if T < parameters['T_inf']:
        T = parameters['T_inf']
        
    fluid.TP = T, P
    density = fluid.density
    mu = fluid.viscosity
    
    drhodP_H = calc_drhodH_P(fluid)
    drhodh_P = calc_drhodP_H(fluid)
    
    M = [[1                   , 0                   , density * U],
         [U / density * drhodP_H, U / density * drhodh_P, 1        ],
         [0                   , 1                   , U        ]]
    
    Re = density*parameters['geometry']['hydraulicDiameter']*U/mu # Reynolds number
    
    f_D = ( -1.8 * np.log10( 6.9 / Re + ((parameters['geometry']['roughness'] / parameters['geometry']['hydraulicDiameter']) / 3.7) ** 1.11 ))**(-2)
    # Flow gradient formulation
    dSdx = parameters['geometry']['heatExchangePerimeter']
    dQdx = dSdx * parameters['flux']
    dAdx = 0
    
    A = [[-density * f_D / parameters['geometry']['hydraulicDiameter'] * (U**2) /2],
         [-U / parameters['geometry']['hydraulicSection'] * dAdx],
         [1 / parameters['massFlow'] * dQdx]]
    
    mInv = np.linalg.inv( M )
    dzdx = np.matmul( mInv, A )
    
    if fullOutput:
        return(density,T)
    
    return dzdx

def calc_z0xs(parameters):
    fluid = parameters['fluidObject']
    fluid.TP = parameters['T_inf'],parameters['P_inf']
    density = fluid.density
    U = massflow/(density*parameters['geometry']['hydraulicSection'])
    h = fluid.enthalpy_mass
    P = fluid.P
    z0 = np.array([P,h,U])
    xs = np.linspace(0,parameters['geometry']['length'],parameters['N'])
    return(z0,xs)

def calc_output(sol,parameters):
    xs = sol['t']
    zs = sol['y']
    Ps = zs[0]
    hs = zs[1]
    Us = zs[2]
    Ts = np.zeros(parameters['N'])
    rhos = np.zeros(parameters['N'])
    for ii,(x,z) in enumerate(zip(xs,np.transpose(zs))):
        rhos[ii],Ts[ii] = dzdx_flow(x,np.reshape(z,[len(z),1]),parameters,fullOutput=True)
    return(xs,Ps,hs,Us,Ts,rhos)

channelX = 1e-3 # m
channelY = 1e-3 # m
roughness = 5e-6 # m
length = 50e-3 # m
flux = 0.5e6 # W/m2
massflow = 0.01 # kg/s
T_inf = 300 # initial temperature, K
P_inf = 5e5 # initial pressure, K
N = 50 # number of point for export

parameters = {
    'flux': flux,
    'massFlow': massflow, # kg/s
    'T_inf':T_inf, # [K] coolant inlet temperature
    'P_inf':P_inf, # [bar] coolant inlet pressure
    'N' : N
}
parameters['fluidObject'] = ct.Water()
geometry = {
    'roughness' : roughness,
    'hydraulicSection' : channelX*channelY,
    'hydraulicPerimeter' : 2*channelX+2*channelY,
    'length' : length
    }

geometry['heatExchangePerimeter'] = geometry['hydraulicPerimeter'] 
hydraulicDiameter=(4*geometry['hydraulicSection'] / geometry['hydraulicPerimeter'])
geometry['hydraulicDiameter'] = hydraulicDiameter

parameters['geometry'] = geometry

z0,xs = calc_z0xs(parameters)

sol=solve_ivp(fun=lambda x,z: dzdx_flow(x,z,parameters),
              method='LSODA',
              y0=z0,
              t_span=[0.,xs[-1]],
              t_eval=xs,vectorized=True)

xs,Ps,hs,Us,Ts,rhos = calc_output(sol,parameters)

inchToMM = 25.4
combustionAndFlameMax = 88 # mm
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['mathtext.rm'] = "cm"
plt.rcParams['mathtext.it'] = "cm"
plt.rcParams['mathtext.bf'] = "cm"

fig = plt.figure(figsize=(combustionAndFlameMax/inchToMM*1.3, combustionAndFlameMax/inchToMM*1),dpi=400)
gs = fig.add_gridspec(1, 1, hspace=0.05,wspace=0.05)
ax = gs.subplots(sharex=True) # sharey=True
ax2 = ax.twinx()
ax.plot(xs,Ps/1e5,color='blue')
ax2.plot(xs,Ts,color='red')
