#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:34:34 2024

@author: marvink
"""

import numpy as np
from HARMONIE import *
# how to use it
help(call_all)
#%% the plot about impact on CH 
show_impact_CH()

#%% Example: compare REF vs CARRA1, REF = T2MFIX for T2M not for H
PTA    = np.array([268, 266, 267, 266, 268, 262, 265]) # Atmospheric temp
PTS    = np.array([265, 255, 253, 251, 250, 249,248])  # Surface Temp
PVMOD  = np.full(len(PTA), 5)       # wind speed
PQA    = np.full(len(PTA), 0.0005)  # specific humidity atm
PQS    = np.full(len(PTA), 0.001)   # specific humidity surface
PH     = np.full(len(PTA), 2)       # height to be interpolated towards
PHT    = np.full(len(PTA), 11)      # model level height
PZ0H   = np.full(len(PTA), 0.1)  # roughness length for heat, snown
PZ0    = np.full(len(PTA), 0.001)   # roughness length for momentum, snow
PS     = np.full(len(PTA), 101100)     # surface Pressure

# calculate all coeffs and variables
# first for REF, REF = T2MFIX for T2M
emulate = 'REF'
PTNM1, H1, PRI1, PQNM1, PHUNM1 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0, PZ0H, PS, emulate)

emulate = 'AA'
# second as in AA/CARRA1 model (XRIMAX=0)
PTNM2, H2, PRI2, PQNM2, PHUNM2 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0, PZ0H, PS, emulate)

# plot the data, CARE rename figure if needed!
emulate1 = 'REF'
emulate2 = 'CARRA1'
compare_T2M_H(PTA,PTS,PTNM1,PTNM2,emulate1,emulate2,PRI1,PRI2,H1,H2,
              showRi=True)

#%% example of FORCE, plot_single_T2M_H
PTA    = np.array([268, 266, 267, 266, 268, 262, 265]) # Atmospheric temp
PTS    = np.array([265, 255, 253, 251, 250, 249,248])  # Surface Temp
PRI_F  = np.array([0, 0.5, 0.11, 0.01, 0.3, 0,0.05])# Ri of my liking   
PVMOD  = np.full(len(PTA), 5)       # wind speed
PQA    = np.full(len(PTA), 0.0005)  # specific humidity atm
PQS    = np.full(len(PTA), 0.001)   # specific humidity surface
PH     = np.full(len(PTA), 2)       # height to be interpolated towards
PHT    = np.full(len(PTA), 11)      # model level height
PZ0H   = np.full(len(PTA), 0.0001)  # roughness length for heat, snown
PZ0    = np.full(len(PTA), 0.001)   # roughness length for momentum, snow
PS     = np.full(len(PTA), 101100)  # surface Pressure

emulate = 'FORCE'
PTNM1, H1, PRI1, PQNM, PHUNM = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0, PZ0H, PS, emulate,
                           PRI_FORCE=PRI_F)
plot_single_T2M_H(PTA,PTS,PTNM1,emulate,PRI1,H1,showRi=True)

#%% extracting from thredds
from extract_thredds import *
#help(extract_timeseries)
#help(extract_entire_domain)
#%% extract time series from Thredds
dat, PTA,PTS,PQA,PQS,PZ0H,PZ0EFF,T2M_archive,H_archive, PS, PA, PVMOD = \
extract_timeseries(2024, 2, 10, 23.5375, 79.8747,model='AA')

# calculate all coeffs and variables
emulate1 = 'REF'
emulate2 = 'AA'
PTNM1, H1, PRI1, PQNM1, PHUNM1 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0EFF, PZ0H, PS, emulate1)
PTNM2, H2, PRI2, PQNM2, PHUNM2 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0EFF, PZ0H, PS, emulate2)

# plot and include T2M / H from the archive
compare_T2M_H(PTA,PTS,PTNM1,PTNM2,emulate1,emulate2,PRI1,PRI2,H1,H2,
           T2M_archive=T2M_archive,H_archive=H_archive)

#%% extract entire domain from Thredds
dat, PTA,PTS,PQA,PQS,PZ0H,PZ0EFF,T2M_archive,H_archive, PS, PA, PVMOD = \
extract_entire_domain(2024,1,10,25,'MEPS')


emulate1 = 'REF'
emulate2 = 'AA'
PTNM1, H1, PRI1, PQNM1, PHUNM1 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0EFF, PZ0H, PS, emulate1)
PTNM2, H2, PRI2, PQNM2, PHUNM2 = call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0EFF, PZ0H, PS, emulate2)

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.pcolormesh(PTNM1,vmin=240,vmax=280)
plt.colorbar(label='T2M (K)')
plt.title(emulate1)
plt.subplot(1,2,2)
plt.pcolormesh(PTNM2,vmin=240,vmax=280)
plt.colorbar(label='T2M (K)')
plt.title(emulate2)
plt.show()

plt.figure(figsize=(12,8))
plt.pcolormesh(PTNM2-PTNM1,colormap=plt.cm.coolwarm,vmin=-2,vmax=2)
plt.colorbar(label='difference T2M (K)')
plt.title(emulate2+'-'+emulate1)
plt.show()

