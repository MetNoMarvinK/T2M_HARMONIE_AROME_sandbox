#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:32:28 2024

@author: marvink
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from netCDF4 import Dataset
sns.set(style='whitegrid')

#%%
def calculate_Richardsons_number(PTA, PQA, PTG, PQS, PVMOD, PS, PA, PZREF = 11,
                                 PUREF = 11, XRIMAX=None, PDIRCOSZW=np.cos(0),
                                 XVMODFAC=0.1):
    # define constans
    XMD       = 28.9644E-3
    XMV       = 18.0153E-3
    XBOLTZ    = 1.380658E-23
    XAVOGADRO = 6.0221367E+23
    XRD       = XAVOGADRO * XBOLTZ / XMD
    XRV       = XAVOGADRO * XBOLTZ / XMV
    XG        = 9.80665
    XCPD      = 7.* XRD /2.
    p0        = 101315
    # Calculate ZTHVA
    EXNA = (PA/p0)**(XRD/XCPD)
    ZTHVA = PTA / EXNA * (1. + (XRV / XRD - 1.) * PQA)
    
    # Calculate ZTHVS
    EXNS = (PS/p0)**(XRD/XCPD)
    ZTHVS = PTG / EXNS * (1. + (XRV / XRD - 1.) * PQS)
    
    
    # Calculate ZVMOD, sets minimum wind speed to be used in Ri calc
    ZVMOD = wind_threshold(PVMOD,XVMODFAC,UREF=PZREF)
    
    # Calculate Richardson's number PRI
    PRI = XG * PDIRCOSZW * PUREF * PUREF * (ZTHVA - ZTHVS) \
          / (0.5 * (ZTHVA + ZTHVS)) / (ZVMOD * ZVMOD) / PZREF
    
    if XRIMAX is not None:
        PRI = np.minimum(PRI, XRIMAX)
    
    return PRI
#%% calculate CH as in surfex documentation/ recreate THE plot
def stability_functions_CH(Ri1, z, z0, fac=10):
    z0h = z0 / fac
    my = np.log(z0 / z0h)
    Ch_ast = 3.2165 + 4.3431 * my + 0.5360 * my**2 - 0.0781 * my**3
    ph = 0.5802 - 0.1571 * my + 0.0327 * my**2 - 0.0026 * my**3
    C_DN = 0.4**2 / (np.log(z / z0)**2)
    coef = 15
    Ch = (coef * Ch_ast * C_DN * (z / z0h)**ph) * (np.log(z / z0) / np.log(z / z0h))

    REF = np.zeros_like(Ri1)
    AA = np.zeros_like(Ri1)
    MEPS = np.zeros_like(Ri1)

    for i, Ri in enumerate(Ri1):
        if Ri <= 0:
            REF[i] = C_DN * (1 - (coef * Ri / (1 + Ch * np.sqrt(np.abs(Ri))))) * (np.log(z / z0) / np.log(z / z0h))
        if Ri > 0:
            REF[i] = C_DN * (1 / (1 + coef * Ri * np.sqrt(1 + 5 * Ri))) * (np.log(z / z0) / np.log(z / z0h))

    for i, Ri in enumerate(Ri1):
        if Ri <= 0:
            AA[i] = C_DN * (1 - (15 * Ri / (1 + Ch * np.sqrt(np.abs(Ri))))) * (np.log(z / z0) / np.log(z / z0h))
        if Ri > 0:
            Ri = 0
            AA[i] = C_DN * (1 / (1 + 15 * Ri * np.sqrt(1 + 5 * Ri))) * (np.log(z / z0) / np.log(z / z0h))

    for i, Ri in enumerate(Ri1):
        if Ri <= 0:
            MEPS[i] = C_DN * (1 - (15 * Ri / (1 + Ch * np.sqrt(np.abs(Ri))))) * (np.log(z / z0) / np.log(z / z0h))
        if Ri > 0:
            if Ri <= 0.1:
                Ri = 0
            if 0.1 < Ri < 0.4:
                Ri = Ri - 0.1
            if Ri >= 0.4:
                Ri = 0.3
            MEPS[i] = C_DN * (1 / (1 + 15 * Ri * np.sqrt(1 + 5 * Ri))) * (np.log(z / z0) / np.log(z / z0h))

    return REF, AA, MEPS

def show_impact_CH():
    Ri1 = np.linspace(-0.5, 0.6, 300)
    z = 1
    z0 = 0.0003
    result_REF, result_AA, result_MEPS = stability_functions_CH(Ri1, z, z0)

    plt.figure(figsize=(10,8))
    plt.plot(Ri1,result_AA,c='royalblue',lw=3,label='XRIMAX=0')
    plt.plot(Ri1,result_MEPS,lw=3,c='crimson',label='XRIMAX=0.4, XRISHIFT=0.1')
    plt.plot(Ri1,result_REF,lw=3,c='k',label='Louis 1979')
    plt.ylabel('$C_H$',fontsize=18)
    plt.xlabel('$Ri$',fontsize=18)
    plt.title('$z_0$ = 0.0003',fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('XRIMAX_on_CH.png',bbox_inches = 'tight',
                dpi=200)

#%% direct replica of surface_aero_cond.F90 for CH calculation
def surface_aero_cond(PRI, PZ0, PZ0H, PVMOD, PZREF=11, PUREF=11, XKARMAN=0.4,
                      XRISHIFT=0.1, XCH_COEFF1 = 15, RISHIFT=False):
    # Function definitions for CHSTAR and PH
    def CHSTAR(x):
        return 3.2165 + 4.3431 * x + 0.5360 * x**2 - 0.0781 * x**3

    def PH(x):
        return 0.5802 - 0.1571 * x + 0.0327 * x**2 - 0.0026 * x**3

    # Initialize arrays, This is just for safety, whould never activate, I hope
    ZZ0 = np.minimum(PZ0, PUREF * 0.5)
    ZZ0H = np.minimum(ZZ0, PZ0H)
    ZZ0H = np.minimum(ZZ0H, PZREF * 0.5)

    ZWORK1 = np.log(PUREF / ZZ0)
    ZWORK2 = PZREF / ZZ0H
    ZWORK3 = PVMOD**2

    ZMU = np.maximum(np.log(ZZ0 / ZZ0H), 0.0)
    ZFH = ZWORK1 / np.log(ZWORK2)

    ZCHSTAR = CHSTAR(ZMU)
    ZPH = PH(ZMU)

    ZCDN = (XKARMAN / ZWORK1)**2
    ZSTA = PRI * ZWORK3

    ZDI = np.zeros_like(PRI)
    PAC = np.zeros_like(PRI)
    PRA = np.zeros_like(PRI)
    PCH = np.zeros_like(PRI)

    for JJ in range(len(PRI)):
        # unstabe - neutral case
        if PRI[JJ] <= 0.0:
            ZDI[JJ] = 1. / (PVMOD[JJ] + ZCHSTAR[JJ] * ZCDN[JJ] * 15. * ZWORK2[JJ]**ZPH[JJ] * ZFH[JJ] * np.sqrt(-ZSTA[JJ]))
            PAC[JJ] = ZCDN[JJ] * (PVMOD[JJ] - 15. * ZSTA[JJ] * ZDI[JJ]) * ZFH[JJ]
        else:
            # stable case
            ZRIMOD  = np.maximum(PRI[JJ] - XRISHIFT, 0.) if RISHIFT else PRI[JJ]
            ZSTAMOD = ZRIMOD * ZWORK3[JJ]
            ZDI[JJ] = np.sqrt(ZWORK3[JJ] + 5. * ZSTAMOD)
            PAC[JJ] = ZCDN[JJ] * PVMOD[JJ] / (1. + XCH_COEFF1 * ZSTAMOD * ZDI[JJ] / ZWORK3[JJ] / PVMOD[JJ]) * ZFH[JJ]

        PRA[JJ] = 1. / PAC[JJ]
        PCH[JJ] = 1. / (PRA[JJ] * PVMOD[JJ])

    return PRA, PCH

#%% calculate CD, replicates surface_cd.F90, XCD_COEFF1 / XCD_COEFF2 from model

def turbulence_coefficients(PRI, PZ0EFF, PZ0H, PZREF =11, PUREF = 11, XCD_COEFF1=10.0,
                            XCD_COEFF2=5.0, XRISHIFT = 0.1, RISHIFT=False):
    def CMSTAR(X):
        return 6.8741 + 2.6933*X - 0.3601*X**2 + 0.0154*X**3

    def PM(X):
        return 0.5233 - 0.0815*X + 0.0135*X**2 - 0.0010*X**3
    
    # include XRISHIFT if chosen (MEPS, FORCE)
    if RISHIFT:
        ZRIMOD = np.zeros_like(PRI)
        for JJ in range(len(PRI)): 
            if PRI[JJ] <=0:
                ZRIMOD[JJ] = PRI[JJ]
            else:
                ZRIMOD[JJ] = np.maximum(PRI[JJ]-XRISHIFT,0) 
    else:
        ZRIMOD = PRI
    
    ZZ0EFF = np.minimum(PZ0EFF, PUREF * 0.5)
    ZZ0H = np.minimum(ZZ0EFF, PZ0H)

    ZMU = np.log(np.minimum(ZZ0EFF / ZZ0H, 200.))

    PCDN = (0.4 / np.log(PUREF / ZZ0EFF))**2

    ZCMSTAR = CMSTAR(ZMU)
    ZPM = PM(ZMU)

    ZCM = 10. * ZCMSTAR * PCDN * (PUREF / ZZ0EFF)**ZPM

    ZFM = np.zeros_like(ZRIMOD)
    condition = (ZRIMOD > 0.0)
    ZFM[condition] = 1. + XCD_COEFF1 * ZRIMOD[condition] / \
                     np.sqrt(1. + XCD_COEFF2 * ZRIMOD[condition])

    ZFM[condition] = 1. / ZFM[condition]

    ZFM[~condition] = 1. - 10. * ZRIMOD[~condition] / \
                     (1. + ZCM[~condition] * np.sqrt(-ZRIMOD[~condition]))

    PCD = PCDN * ZFM

    PCD = PCDN * (1. / ZFM)

    return PCD
#%% calculate T2M and Q2M as in model (cls_tq.F90)
def CLS_TQ(PTA,PTS, PCD, PCH, PRI, PH, PHT, PZ0H, PS, PA, PQS, PQA, PLMO=None, ZACLS_HS=None):
    XKARMAN = 0.4  # Assuming XKARMAN is a predefined constant
    ZEPS2 = np.sqrt(np.finfo(float).eps)  # Equivalent to Fortran's EPSILON(1.0)

    # 1. Preparatory calculations
    ZBNH = np.log(PHT / PZ0H)
    ZBH = XKARMAN * np.sqrt(PCD) / PCH
    ZRS = np.minimum(PH / PHT, 1.0)
    ZLOGS = np.log(1.0 + ZRS * (np.exp(ZBNH) - 1.0))

    # 2. Stability effects
    if PLMO is not None and ZACLS_HS is not None:  # Check if PLMO is provided
        # Stable case: revised Kullmann 2009 solution
        ZAUX = np.maximum(ZEPS2, PH * ZACLS_HS / (ZACLS_HS * PZ0H + PLMO))
        ZCORS = np.where(PRI >= 0, (ZBNH - ZBH) * np.log(1.0 + ZAUX * ZRS) / np.log(1.0 + ZAUX), 0)
    else:
        # Stable case: Geleyn 1988 solution
        ZCORS = np.where(PRI >= 0, ZRS * (ZBNH - ZBH), 0)

    # Handling PRI < 0 case
    ZCORS = np.where(PRI < 0, np.log(1.0 + ZRS * (np.exp(np.maximum(0.0, ZBNH - ZBH)) - 1.0)), ZCORS)

    # 3. Interpolation of thermodynamical variables
    ZIV = np.maximum(0.0, np.minimum(1.0, (ZLOGS - ZCORS) / ZBH))
    PTNM = PTS + ZIV * (PTA - PTS)
    
    
    ############### CALCULATE Q2M ####################################3
    # calculation of Q2M following case YHUMIDITY=='Q ' in code
    ZPNM = PS + PH/PHT * (PA-PS)
    # Refer to QSATW routine, i.e. saturation humidity over water
    # some constants
    XAVOGADRO = 6.0221367E+23
    XBOLTZ    = 1.380658E-23
    XMD    = 28.9644E-3
    XMV    = 18.0153E-3
    XRD    = XAVOGADRO * XBOLTZ / XMD
    XRV    = XAVOGADRO * XBOLTZ / XMV
    XCPD   = 7.* XRD /2.
    XCPV   = 4.* XRV
    XRHOLW = 1000.
    XRHOLI = 917.
    XCONDI = 2.22
    XCL    = 4.218E+3
    XCI    = 2.106E+3
    XTT    = 273.16
    XTTSI  = XTT - 1.8
    XICEC  = 0.5
    XTTS   = XTT*(1-XICEC) + XTTSI*XICEC
    XLVTT  = 2.5008E+6
    XLSTT  = 2.8345E+6
    XLMTT  = XLSTT - XLVTT
    XESTT  = 611.14
    XGAMW  = (XCL - XCPV) / XRV
    XBETAW = (XLVTT/XRV) + (XGAMW * XTT)
    XALPW  = np.log(XESTT) + (XBETAW /XTT) + (XGAMW *np.log(XTT))
    XGAMI  = (XCI - XCPV) / XRV
    XBETAI = (XLSTT/XRV) + (XGAMI * XTT)
    XALPI  = np.log(XESTT) + (XBETAI /XTT) + (XGAMI *np.log(XTT))

    # COMPUTE SATURATION VAPOR PRESSURE
    ZALP  = np.log(XESTT) + (XBETAW /XTT) + (XGAMW *np.log(XTT))
    ZBETA = XBETAW
    ZGAM  = XGAMW
    ZFOES = np.exp( ZALP - ZBETA/PTNM - ZGAM*np.log(PTNM))

    ZWORK1 = ZFOES/ZPNM
    ZWORK2 = XRD/XRV
    ZQSATNM= ZWORK2*ZWORK1 / (1.+(ZWORK2-1.)*ZWORK1)
          
    PQNM   = PQS+ZIV*(PQA-PQS)
    PQNM   = np.minimum(ZQSATNM,PQNM) # must be below saturation
    PHUNM  = PQNM / ZQSATNM
    


    return PTNM, PQNM, PHUNM
#%% heatflux as in literature/model
def heat_flux(PVMOD,CH,PTS,PTA):
    # constanst
    rho_a = 1.3 
    cp    = 1004.7088578330674
    
    H = rho_a * cp * PVMOD * CH * (PTS-PTA)
    
    return H

#%% latent heat flux (not yet fully included)
def e_sat(PT,PP):
    XAVOGADRO = 6.0221367E+23
    XBOLTZ    = 1.380658E-23
    XMD    = 28.9644E-3
    XMV    = 18.0153E-3
    XRD    = XAVOGADRO * XBOLTZ / XMD
    XRV    = XAVOGADRO * XBOLTZ / XMV
    XCPD   = 7.* XRD /2.
    XCPV   = 4.* XRV
    XRHOLW = 1000.
    XRHOLI = 917.
    XCONDI = 2.22
    XCL    = 4.218E+3
    XCI    = 2.106E+3
    XTT    = 273.16
    XTTSI  = XTT - 1.8
    XICEC  = 0.5
    XTTS   = XTT*(1-XICEC) + XTTSI*XICEC
    XLVTT  = 2.5008E+6
    XLSTT  = 2.8345E+6
    XLMTT  = XLSTT - XLVTT
    XESTT  = 611.14
    XGAMW  = (XCL - XCPV) / XRV
    XBETAW = (XLVTT/XRV) + (XGAMW * XTT)
    XALPW  = np.log(XESTT) + (XBETAW /XTT) + (XGAMW *np.log(XTT))
    XGAMI  = (XCI - XCPV) / XRV
    XBETAI = (XLSTT/XRV) + (XGAMI * XTT)
    XALPI  = np.log(XESTT) + (XBETAI /XTT) + (XGAMI *np.log(XTT))
    
    
    ZALP  = np.log(XESTT) + (XBETAW /XTT) + (XGAMW *np.log(XTT))
    ZBETA = XBETAW
    ZGAM  = XGAMW
    ZFOES = np.exp( ZALP - ZBETA/PT - ZGAM*np.log(PT))

    ZWORK1 = ZFOES/PP
    ZWORK2 = XRD/XRV
    ZQSAT = ZWORK2*ZWORK1 / (1.+(ZWORK2-1.)*ZWORK1)
    
    return ZQSAT

def hu_hui(wg,wfc):
    if wg<wfc:
        hu = 0.5*(1-np.cos(wg/wfc *np.pi))
    else:
        hu = 1
        
    return hu

# a bit too much external info needed for now latent heat flux currently
# def latent_heat_flux(PVMOD,CH,PTS,PTA,PQS,PQA,PS,PA):
#     # constanst
#     rho_a = 1.3 
#     cp    = 1004.7088578330674

#     veg  = Dataset('veg.nc').variables['veg'][:,:]   # vegetation cover
#     psng = Dataset('psng.nc').variables['psng'][:,:] # snow over ground
#     psnv = Dataset('psnv.nc').variables['psnv'][:,:] # snow over vegetation
#     di   = Dataset('di.nc').variables['di'][:,:]     # surface ice fraction
#     # COMPUTE SATURATION VAPOR PRESSURE
#     QS_sat = e_sat(PTS,PS)
#     QA_sat = e_sat(PTA,PA)
#     # relative humidity at ground for portion of liquid and frozen parts
#     hu  = hu_hui(wg,wfc)
#     hui = hu_hui(wgf,wfc2)
#     # Halstead coefficient calculation
#     Ra = (PVMOD*CH)**-1
#     hv =(1 − δ)Ra /(Ra + Rs ) + δ
    
   
#     Egl = (1-veg)*(1-psng)*(1-di)* rho_a* CH * PVMOD *(hu * QS_sat - PQA )
#     #Ev = veg(1-psnv)*rho_a*CH*PVMOD* hv (qsat (Ts )-qa )
    
#     LE = L*Egl + L*Ev + Li* (Es + Egf )
    
#     return LE
#%% simple heat flux in model code (isba_fluxes.F90) ..  no WS, no CH
def heat_flux_model(PTS,PTA,PA,PS,PVMOD,PCH,P0=101315):
    # constants
    XMD       = 28.9644E-3
    XBOLTZ    = 1.380658E-23
    XAVOGADRO = 6.0221367E+23
    XRD       = XAVOGADRO * XBOLTZ / XMD
    XCP       = 1004.7088578330674
    PRHOA     = 1.3
    XRESA     = 100 
    
    def calc_EXN(P):
        return (P/P0)**(XRD/XCP)
        


    PEXNA = calc_EXN(PA)
    PEXNS = calc_EXN(PS)

    # as in code bu XRESE shit does not sit with me
    H = PRHOA * XCP * (PTS - PTA*PEXNS/PEXNA) / XRESA/ PEXNS
    
    #H = PRHOA * XCP * PVMOD * PCH *(PTS - PTA*PEXNS/PEXNA) / PEXNS
    return H
#%%  wind has lower limit for Ri computations
def wind_threshold(PVMOD,XVMODFAC=0.1,UREF=11): # factor and reference for forcing (atm level)
    thresh = XVMODFAC * np.min([10,UREF]) # 1 m/s for 10m when 0.1 is used
    PVNEW = PVMOD.copy()
    PVNEW[PVNEW<thresh] = thresh
    
    return PVNEW
#%%
def model_versions(emulate, MODFAC_F):
    RISHIFT = False # use XRISHIFT, only for MEPS
    XVMODFAC= 0.1   # used in wind_treshold during computation of Ri
    if emulate == 'REF':
        XRIMAX = None
        XVMODFAC = 0.0
    elif emulate == 'T2Mf': # T2Mfix
        XRIMAX = None
    elif emulate == 'AA':
        XRIMAX = 0.0
    elif emulate == 'MEPS':
        XRIMAX = 0.4
        RISHIFT = True
        XVMODFAC = 0.2
    elif emulate == 'RI02':
        XRIMAX = 0.2
    elif emulate == 'FORCE':
        XRIMAX = None
        XVMODFAC = MODFAC_F

    return XRIMAX, RISHIFT, XVMODFAC
#%% call all routines 
def call_all(PTA, PQA, PTS, PQS, PVMOD,PZ0, PZ0H, PS, emulate, PA=None, PH=2, 
             PHT=11, slope = 0, PRI_FORCE=None, MODFAC_F=0.1, rall=False):
    """
    This function calls all routines in order to calculate T2M as in HARMONIE-AROME.
    This includes calculating: Richardson number (PRI)
                               heat and drag coefficients (PCH, PCD)
                               Prandtl number (PRA)
                               heat fluxes (H)
                               2m-temperature (PTNM)
                               2m-specific humidity (PQNM)
    
    Parameters:
    PTA   : atm. Temperature, lowermost model level (K)
    PQA   : atm. sepcific humidity (kg/kg)
    PTS   : surf. Temperature (K)
    PQS   : surf. specific humidity (kg/kg)
    PVMOD : wind speed, lowermost model level (m/s)
    PZ0   : roughness length for momentum (m)
    PZ0H  : roughness length for heat (m)
    PS    : surface pressure (Pa)
    emulate : string, the model settings to be emulated
              REF : XRIMAX=None, XVMODFAC=0.2 (T2Mfix for T2M)
              AA  : XRIMAX=0, XVMODFAC=0.2 (AROME-Arctic, CARRA1)
              MEPS: XRIMAX=0.4, XRISHIFT = 0.1, XVMODFAC=0.2 (MEPS)
              RI02: XRIMAX=0.2, XRISHIFT=0.1 (test Patrick)
              FORCE : user defined RI and XVMODFAC
              for your own version modify "model_versions" function in HARMONIE.py
    optional parameters:
    PA        : pressure at model level, calculated if not given assuming 65 levels, default: None
    PH        : height to be interpolated towards, default: 2
    PHT       : height representative for atm. values, default: 11
    slope     : angle between surface orientation and atm., default: 0
    PRI_FORCE : sequence of Ri to be used in case emulate=='FORCE', default: None
    MODFAC_F  : XVMODFAC to be used in case 'FORCE', default: 0.1
    rall      : bool, decides what is returned, default: False
    
    Returns:
    rall = False : T2M, H, RI, Q2M, HU2M
    rall = True  : T2M, H, RI, CH, CD, Pra, Q2M, HU2M
    """
    resh = False
    if len(PTA.shape)==2: # 2D fields are given as input, reshape for calculation
        # for reshaping output later on
        resh = True
        D1  = PTA.shape[0]
        D2  = PTA.shape[1]
        PTA = PTA.flatten()
        PQA = PQA.flatten()
        PTS = PTS.flatten()
        PQS = PQS.flatten()
        PVMOD = PVMOD.flatten()
        PZ0 = PZ0.flatten()
        PZ0H= PZ0H.flatten()
        PS = PS.flatten()
    # some sanity checks
    if not len(PTA)==len(PTS)==len(PVMOD)==len(PS)==len(PQS)==len(PQA):
        print('PTA, PQA, PTS, PQS, PVMOD, and PS need to have the same length!')
        return
    if PH > PHT:
        print('PH needs to be smaller than PHT!')
        return
    if emulate not in ['REF', 'AA', 'MEPS', 'RI02', 'FORCE', 'T2Mf']:
        print('emulate not defined for ', emulate)
        print('these are your options:')
        print('REF  : no limit on XRI, XVMODFAC=0')
        print('AA   : XRIMAX = 0 (AROME-Arctic, CARRA1)')
        print('MEPS : XRIMAX=0.4, XRISHIFT=0.1 (MEPS)')
        print('RI02 : XRIMAX=0.2, (test Patrick)')
        print('T2Mf : no limit on XRI,')
        print('FORCE: Manually provide RI and activate RISHIFT')
        return
    # get PA from hybrid calc if not given
    if PA is None:
        PA = PS*0.99851963
    elif (len(PA) - len(PS)) != 0:
        print('PS and PA need to have the same length!')
        return
    # in case FORCE:
    if emulate == 'FORCE':
        try:
            if len(PRI_FORCE) != len(PTA):
                print('Provided series of RI needs to have the same length as PTA etc.')
                return
        except:
            print('When emulate==FORCE, PRI_FORCE must be given!')
    
    if not isinstance(rall, bool):
        print('rall needs to be True or False')
        return
    
    # settings accordign to emulated model version
    XRIMAX, RISHIFT,XVMODFAC = model_versions(emulate, MODFAC_F)
    
    # 
    
    # calculate Richardson
    if emulate != 'FORCE':
        PRI = calculate_Richardsons_number(PTA, PQA, PTS, PQS, PVMOD, PS ,PA,
                                     PZREF = PHT, PUREF = PHT, XRIMAX=XRIMAX, 
                                     PDIRCOSZW=np.cos(slope),XVMODFAC=XVMODFAC)
    else:
        PRI = PRI_FORCE
    
    # calc drag coeff for momentum, CD
    PCD = turbulence_coefficients(PRI,PZ0,PZ0H,PZREF=PHT,PUREF =PHT,
                                  RISHIFT=RISHIFT) 
    # calculate exchange coefficient for heat, CH. also provides Prandtl number
    PRA, PCH = surface_aero_cond(PRI, PZ0, PZ0H, PVMOD, PZREF=PHT, PUREF=PHT,
                                 RISHIFT=RISHIFT)
    # calculate T2M
    PTNM,PQNM, PHUNM  = CLS_TQ(PTA, PTS, PCD, PCH, PRI, PH, PHT, PZ0H, PS, PA, PQS, PQA)
    # calculate heat flux
    H        = heat_flux(PVMOD, PCH, PTS, PTA)
    
    # reshape to 2D if 2D was given
    if resh:
        PTNM = PTNM.reshape(D1,D2)
        H    = H.reshape(D1,D2)
        PRI  = PRI.reshape(D1,D2)
        PCH  = PCH.reshape(D1,D2)
        PCD  = PCD.reshape(D1,D2)
        PRA  = PRA.reshape(D1,D2)
        PQNM = PQNM.reshape(D1,D2)
        PHUNM= PHUNM.reshape(D1,D2)
        
    if rall:
        return PTNM, H, PRI, PCH, PCD, PRA, PQNM, PHUNM
    else:
        return PTNM, H, PRI, PQNM, PHUNM

#%% plottig routine for comparing two different settings

def compare_T2M_H(PTA,PTS,PTNM1,PTNM2,emulate1,emulate2,PRI1,PRI2,H1,H2,
               showRi=False,T2M_archive=[],H_archive=[]):
    # plot T2m and H
    plt.figure(figsize=(20, 8))
    plt.subplot(1,2,1)
    plt.plot(PTA, c='royalblue', lw=3, label='TA')
    plt.plot(PTNM1, lw=3, c='crimson', label='T2M, '+emulate1)
    if showRi:
        for i, txt in enumerate(PRI1):
            plt.text(i, PTNM1[i] + 0.3, f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)

    plt.plot(PTNM2, lw=3, c='crimson', label='T2M, '+emulate2, linestyle='--')
    if showRi:
        for i, txt in enumerate(PRI2):
            plt.text(i, PTNM2[i] + 0.3, f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)
    if len(T2M_archive)>0:        
        plt.plot(T2M_archive, lw=2, c='orange', label='T2M, archive')

    plt.plot(PTS, lw=3, c='k', label='TS')
    plt.ylabel('$T (K)$', fontsize=18)
    plt.xlabel('$fc (h)$', fontsize=18)
    #plt.suptitle('$z_0$ = 0.0003', fontsize=22)
    plt.title('Temperature development', fontsize=20)
    plt.legend(fontsize=18)
    plt.subplot(1,2,2)
    plt.plot(H1, lw=3, c='crimson', label='H, '+emulate1)
    if showRi:
        for i, txt in enumerate(PRI1):
            plt.text(i, H1[i], f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)

    plt.plot(H2, lw=3, c='crimson', label='H, '+emulate2, linestyle='--')
    if showRi:
        for i, txt in enumerate(PRI2):
            plt.text(i, H2[i], f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)
    if len(H_archive)>0:        
        plt.plot(H_archive, lw=2, c='orange', label='H, archive')
    plt.ylabel('$H (wm-2)$', fontsize=18)
    plt.xlabel('$fc (h)$', fontsize=18)
    plt.title('Heat flux development', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('T2M_and_H_'+emulate1+'vs'+emulate2+'.png',bbox_inches = 'tight',
                dpi=200)
#%% plot a single config
def plot_single_T2M_H(PTA,PTS,PTNM1,emulate1,PRI1,H1,
               showRi=False,T2M_archive=[],H_archive=[]):
    # plot T2m and H
    plt.figure(figsize=(20, 8))
    plt.subplot(1,2,1)
    plt.plot(PTA, c='royalblue', lw=3, label='TA')
    plt.plot(PTNM1, lw=3, c='crimson', label='T2M, '+emulate1)
    if showRi:
        for i, txt in enumerate(PRI1):
            plt.text(i, PTNM1[i] + 0.3, f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)
    if len(T2M_archive)>0:        
        plt.plot(T2M_archive, lw=2, c='orange', label='T2M, archive')

    plt.plot(PTS, lw=3, c='k', label='TS')
    plt.ylabel('$T (K)$', fontsize=18)
    plt.xlabel('$fc (h)$', fontsize=18)
    #plt.suptitle('$z_0$ = 0.0003', fontsize=22)
    plt.title('Temperature development', fontsize=20)
    plt.legend(fontsize=18)
    plt.subplot(1,2,2)
    plt.plot(H1, lw=3, c='crimson', label='H, '+emulate1)
    if showRi:
        for i, txt in enumerate(PRI1):
            plt.text(i, H1[i], f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)

    if len(H_archive)>0:        
        plt.plot(H_archive, lw=2, c='orange', label='H, archive')
    plt.ylabel('$H (wm-2)$', fontsize=18)
    plt.xlabel('$fc (h)$', fontsize=18)
    plt.title('Heat flux development', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('T2M_and_H_'+emulate1+'.png',bbox_inches = 'tight',
                dpi=200)
    
def plot_single_Q2M(PQA,PQS,PQNM1,emulate1,PRI1,
               showRi=False,Q2M_archive=[]):
    # plot T2m and H
    plt.figure(figsize=(20, 8))
    plt.plot(PQA*1000, c='royalblue', lw=3, label='QA')
    plt.plot(PQNM1*1000, lw=3, c='crimson', label='Q2M, '+emulate1)
    if showRi:
        for i, txt in enumerate(PRI1):
            plt.text(i, PQNM1[i]*1000 + 0.01, f'PRI={txt:.2f}', color='black', ha='center', va='bottom', fontsize=12)
    if len(Q2M_archive)>0:        
        plt.plot(Q2M_archive*1000, lw=2, c='orange', label='T2M, archive')

    plt.plot(PQS*1000, lw=3, c='k', label='QS')
    plt.ylabel('$Q (g/kg)$', fontsize=18)
    plt.xlabel('$fc (h)$', fontsize=18)
    #plt.suptitle('$z_0$ = 0.0003', fontsize=22)
    plt.title('specific humidity development', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig('Q2M_'+emulate1+'.png',bbox_inches = 'tight',
                dpi=200)
