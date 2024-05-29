#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:47:24 2024

@author: marvink
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from netCDF4 import Dataset
from bs4 import BeautifulSoup
import requests
sns.set(style='whitegrid')
#%% for extracting from operational archive of Met Norway

#%% first for meneuvering through OpenDap archive 

def fetch_initial_urls(url):
    """Fetch and list initial URLs from a given catalog URL."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '.html' in href:
            full_url = requests.compat.urljoin(url, href)
            links.append(full_url)
    return links

def get_opendap_url(url):
    """Extract the OPeNDAP URL and format it correctly by removing '.html'."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '.nc' in href:
            href = 'https://thredds.met.no/' + href
            return href.replace('.html', '')
    return None

def check_variables(dataset, var_names):
    """Check for the existence of variables in the dataset, adjust names if necessary, and identify missing variables."""
    available_vars = [var for var in var_names if var in dataset.variables]
    missing_vars = [var for var in var_names if var not in available_vars]

    # Try dropping 'SFX_' prefix and update var_names if successful
    adjusted_vars = {}
    for var in missing_vars:
        adjusted_var = var.replace('SFX_', '')
        if adjusted_var in dataset.variables:
            adjusted_vars[var] = adjusted_var
            available_vars.append(adjusted_var)

    # Filter out found variables from missing_vars
    missing_vars = [var for var in missing_vars if var not in adjusted_vars]
    return available_vars, missing_vars

def interactive_dataset_selection(catalog_url, var_names):
    urls = fetch_initial_urls(catalog_url)
    datasets = {}
    var_sets = {}

    filtered_urls = [url for url in urls if '.nc' in url]  # Filter URLs to include only those with '.nc'

    while var_names:  # Continue until all variables are found or options exhausted
        for index, url in enumerate(filtered_urls):
            print(f"{index + 1}: {url.split('/')[-1]}")
        choice = int(input("Select the number of the link to follow for the OPeNDAP file: ")) - 1
        selected_url = filtered_urls[choice]
        opendap_url = get_opendap_url(selected_url)
        dataset = Dataset(opendap_url)

        available_vars, missing_vars = check_variables(dataset, var_names)

        datasets[opendap_url] = dataset
        var_sets[opendap_url] = available_vars

        # Update var_names to only include still missing variables
        var_names = missing_vars

        if not missing_vars:
            print("All required variables found.")
            print("Extracting variables ...")
            return datasets, var_sets
        else:
            print(f"Missing variables from {opendap_url}: {missing_vars}")
            user_choice = input("Try another file? (yes/no): ")
            if user_choice.lower() != 'yes':
                raise Exception("Process aborted by the user.")

    return datasets, var_sets
#%% extract a single grid point time series closests to a specified location
def extract_timeseries(year,month,day,plon,plat,model):
    """
    This function gets all required fields from thredds (the operational
    archive of MET-Norway). The routine will extract a time series from the
    clostest grid point specified by the input.
    You will be able to choose which dataset to use for extraction, e.g which
    forecast file you want to use.
    When your chosen dataset does not contain all needed variables, the routine 
    will ask you to specify another dataset for the missing variables.
    
    
    NOTE1: Thredds stores grid box averages of Z0H and Z0. Differences will be 
    observed when comparing T2M and H from this routine with forecast data due 
    to the (current) inability to replicate SURFEX's weighted approach here.
    
    NOTE2: Thredds does not store surface humidity, so it cannot be extracted!
    Instead Q2M is extracted. For fully physical calculations it needs to come 
    from an external source (FA files for example) or needs to be set manually. 
    
    Parameters:
    year   : year as YYYY
    month  : month as MM
    day    : day as DD
    plon   : lon of point
    plat   : lat of point
    model  : model to extract from, either AA (AROME-Arctic) or MEPS
    
    Returns:
    dat    : netCDF data handler for the file
    TA     : atm. Temperature
    TS     : surf. Temperature
    QA     : atm. specific humidity
    QS     : surf. specific humidity IS ACTUALLY Q2M.. for real values need FA
    PZ0H   : grid box averaged roghness for heat
    PZ0    : grid box averaged rougness for momentum
    T2M    : 2m-temperature
    HS     : sensible heat flux
    PS     : surface pressure
    PA     : atm. pressure
    WS     : atm. wind speed
    """
    varnam1=['air_temperature_ml','air_temperature_0m','specific_humidity_ml',
             'SFX_Q2M','SFX_Z0H','SFX_Z0','SFX_T2M','SFX_H',
             'surface_air_pressure','hybrid','x_wind_ml','y_wind_ml']
    
    if model=='AA':
        url = f'https://thredds.met.no/thredds/catalog/aromearcticarchive/{year}/{month:02}/{day:02}/catalog.html'
        
    elif model=='MEPS':
        url = f'https://thredds.met.no/thredds/catalog/meps25epsarchive/{year}/{month:02}/{day:01}/catalog.html'
    else:
        print('model must either be "AA" or "MEPS"')
        return 
    # choose dataset of interest
    datasets, variables = interactive_dataset_selection(url, varnam1)
    # keys of dict output
    kk = [k for k in datasets.keys()]
    # get clostest x,y index to point
    lats = datasets[kk[0]].variables['latitude'][:,:]
    lons = datasets[kk[0]].variables['longitude'][:,:]
    abslat = np.abs(lats-plat)
    abslon= np.abs(lons-plon)
    c = np.maximum(abslon,abslat)
    x, y = np.where(c == np.min(c))
    x, y = x[0],y[0]
    
    # create dicts
    nn = []
    for k in kk:
        for vn in variables[k]:
            nn.append(vn)
    nn.append('WS')
    nn.append('PA')
    VV = {} # dict containing all the fields
    for v in nn:
        VV[v] = np.zeros(datasets[kk[0]].variables['time'].shape[0])
    # extract the data from thredds
    for k in kk:
        dat = datasets[k]
        for v in variables[k]:
            # model level variables
            if len(dat.variables[v].shape)==4:
                VV[v] = dat.variables[v][:,-1,x,y].data
            # surfex variables
            elif len(dat.variables[v].shape)==3:
                VV[v] = dat.variables[v][:,x,y].data
                # sanity check againsty roughness length of 0 from thredds
                if 'Z0' in v:
                    VV[v] = np.maximum(VV[v],3.123936e-06)
            # hybrid
            elif len(dat.variables[v].shape)==1:
                VV[v] = dat.variables[v][-1].data
    VV['WS'] = np.sqrt(VV['x_wind_ml']**2+VV['y_wind_ml']**2)
    VV['PA'] = VV['surface_air_pressure']*VV['hybrid']
    
    # this is a bit stupid, but I am too tired now :D, need to make sure that
    # the correct output is made, and depdening on chosen dataset the fields
    # have "SFX_" prefix or not
    if 'SFX_Z0' in [n for n in nn if 'SFX' in n]:
        return datasets, VV['air_temperature_ml'], VV['air_temperature_0m'],\
               VV['specific_humidity_ml'], VV['SFX_Q2M'],VV['SFX_Z0H'],\
               VV['SFX_Z0'],VV['SFX_T2M'], VV['SFX_H'],\
               VV['surface_air_pressure'],VV['PA'], VV['WS']
    else:
        return datasets, VV['air_temperature_ml'], VV['air_temperature_0m'],\
               VV['specific_humidity_ml'], VV['Q2M'],VV['Z0H'],\
               VV['Z0'],VV['T2M'], VV['H'],\
               VV['surface_air_pressure'],VV['PA'], VV['WS']

def extract_entire_domain(year,month,day,hh,model):
    """
    This function gets all required fields from thredds. The fields will be
    taken from the enitre model domain at the chosen forecast hour.
    You will be able to choose which dataset to use for extraction, e.g which
    forecast file you want to use.
    When your chosen dataset does not contain all needed variables, the routine 
    will ask you to specify another dataset for the missing variables.
    
    
    NOTE1: Thredds stores grid box averages of Z0H and Z0. Differences will be 
    observed when comparing T2M and H from this routine with forecast data due 
    to the (current) inability to replicate SURFEX's weighted approach here.
    
    NOTE2: Thredds does not store surface humidity, so it cannot be extracted!
    Instead Q2M is extracted. For fully physical calculations it needs to come 
    from an external source (FA files for example) or needs to be set manually.
    
    Parameters:
    year   : year as YYYY
    month  : month as MM
    day    : day as DD
    hh     : forecast hour to extract
    model  : model to extract from, either AA (AROME-Arctic) or MEPS
    
    Returns:
    dat    : netCDF data handler for the file
    TA     : atm. Temperature
    TS     : surf. Temperature
    QA     : atm. specific humidity
    QS     : surf. specific humidity IS ACTUALLY Q2M.. for real values need FA
    PZ0H   : grid box averaged roghness for heat
    PZ0    : grid box averaged rougness for momentum
    T2M    : 2m-temperature
    HS     : sensible heat flux
    PS     : surface pressure
    PA     : atm. pressure
    WS     : atm. wind speed
    """
    varnam1=['air_temperature_ml','air_temperature_0m','specific_humidity_ml',
             'SFX_Q2M','SFX_Z0H','SFX_Z0','SFX_T2M','SFX_H',
             'surface_air_pressure','hybrid','x_wind_ml','y_wind_ml']
    
    if model=='AA':
        url = f'https://thredds.met.no/thredds/catalog/aromearcticarchive/{year}/{month:02}/{day:02}/catalog.html'
        
    elif model=='MEPS':
        url = f'https://thredds.met.no/thredds/catalog/meps25epsarchive/{year}/{month:02}/{day:02}/catalog.html'
    else:
        print('model must either be "AA" or "MEPS"')
        return 
    # choose dataset of interest
    datasets, variables = interactive_dataset_selection(url, varnam1)
    # keys of dict output
    kk = [k for k in datasets.keys()]
    # get clostest x,y index to point
    nn = []
    for k in kk:
        for vn in variables[k]:
            nn.append(vn)
    nn.append('WS')
    nn.append('PA')
    VV = {} # dict containing all the fields
    for v in nn:
        VV[v] = np.zeros(datasets[kk[0]].variables['latitude'].shape)
    # extract the data from thredds
    for k in kk:
        dat = datasets[k]
        for v in variables[k]:
            # model level variables
            if len(dat.variables[v].shape)==4:
                VV[v] = dat.variables[v][hh,-1,:,:].data
            # surfex variables
            elif len(dat.variables[v].shape)==3:
                VV[v] = dat.variables[v][hh,:,:].data
                # sanity check againsty roughness length of 0 from thredds
                if 'Z0' in v:
                    VV[v] = np.maximum(VV[v],3.123936e-06)
            # hybrid
            elif len(dat.variables[v].shape)==1:
                VV[v] = dat.variables[v][-1].data
    VV['WS'] = np.sqrt(VV['x_wind_ml']**2+VV['y_wind_ml']**2)
    VV['PA'] = VV['surface_air_pressure']*VV['hybrid']

    # this is a bit stupid, but I am too tired now :D, need to make sure that
    # the correct output is made, and depdening on chosen dataset the fields
    # have "SFX_" prefix or not
    if 'SFX_Z0' in [n for n in nn if 'SFX' in n]:
        return datasets, VV['air_temperature_ml'], VV['air_temperature_0m'],\
               VV['specific_humidity_ml'], VV['SFX_Q2M'],VV['SFX_Z0H'],\
               VV['SFX_Z0'],VV['SFX_T2M'], VV['SFX_H'],\
               VV['surface_air_pressure'],VV['PA'], VV['WS']
    else:
        return datasets, VV['air_temperature_ml'], VV['air_temperature_0m'],\
               VV['specific_humidity_ml'], VV['Q2M'],VV['Z0H'],\
               VV['Z0'],VV['T2M'], VV['H'],\
               VV['surface_air_pressure'],VV['PA'], VV['WS']

#%%

#OLD ROUTINES NOT NEEDED ANYMORE
def old_t_retrieval(year,month,day,plon,plat,dn='det'):
    """
    This function gets all required fields from thredds. Only works for AA
    Always uses 00UTC run.
    
    NOTE: Thredds stores grid box averages of Z0H and Z0. Differences will be 
    observed when comparing T2M and H from this routine with forecast data due 
    to the (current) inability to replicate SURFEX's weighted approach here.
    
    Parameters:
    year   : year as YYYY
    month  : month as MM
    day    : day as DD
    plon   : lon of point
    plat   : lat of point
    model  : model to extract from, either AA (AROME-Arctic) or MEPS
    dn     : file identifier (changed between "det" and "full" on thredds)
    
    Returns:
    dat    : netCDF data handler for the file
    TA     : atm. Temperature
    TS     : surf. Temperature
    QA     : atm. specific humidity
    QS     : surf. specific humidity ()sometimes Q2M if surface is not avail)
    PZ0H   : grid box averaged roghness for heat
    PZ0    : grid box averaged rougness for momentum
    WS     : atm. wind speed
    T2M    : 2m-temperature
    HS     : sensible heat flux
    PS     : surface pressure
    PA     : atm. pressure
    """
    # hard coded search for 00 UTC start
    url = f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{year}/{month}/{day}/arome_arctic_{dn}_2_5km_{year}{month}{day}T00Z.nc'
    dat = Dataset(url)
    # get clostest x,y index to point
    lats = dat.variables['latitude'][:,:]
    lons = dat.variables['longitude'][:,:]
    abslat = np.abs(lats-plat)
    abslon= np.abs(lons-plon)
    c = np.maximum(abslon,abslat)
    x, y = np.where(c == np.min(c))
    x, y = x[0],y[0]
    if dn == 'det':
        TA   = dat.variables['air_temperature_ml'][:,-1,x,y].data
        TS   = dat.variables['air_temperature_0m'][:,0,x,y].data
        QA   = dat.variables['specific_humidity_ml'][:,-1,x,y].data
        QS   = dat.variables['SFX_Q2M'][:,x,y].data # not optimal since it is 2m... but 0m not stored, whatever
        PZ0H = dat.variables['SFX_Z0H'][:,x,y].data    
        PZ0  = dat.variables['SFX_Z0'][:,x,y].data
        UA   = dat.variables['x_wind_ml'][:,-1,x,y].data
        VA   = dat.variables['y_wind_ml'][:,-1,x,y].data
        WS   = np.sqrt(UA**2 + VA**2)
        T2M  = dat.variables['SFX_T2M'][:,x,y].data # for checking how far I am off (missing tile averaging etc)
        HS   = dat.variables['SFX_H'][:,x,y].data
        PS   = dat.variables['surface_air_pressure'][:,0,x,y].data
        PA   =  PS * dat.variables['hybrid'][-1].data
    elif dn == 'full':
        url = f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{year}/{month}/{day}/arome_arctic_sfx_2_5km_{year}{month}{day}T00Z.nc'
        sfx = Dataset(url)
        TA   = dat.variables['air_temperature_ml'][:,-1,x,y].data
        TS   = dat.variables['air_temperature_0m'][:,0,x,y].data
        QA   = dat.variables['specific_humidity_ml'][:,-1,x,y].data
        QS   = sfx.variables['Q2M'][:,x,y].data # not optimal since it is 2m... but 0m not stored, whatever
        PZ0H = sfx.variables['Z0H'][:,x,y].data    
        PZ0  = sfx.variables['Z0'][:,x,y].data
        UA   = dat.variables['x_wind_ml'][:,-1,x,y].data
        VA   = dat.variables['y_wind_ml'][:,-1,x,y].data
        WS   = np.sqrt(UA**2 + VA**2)
        T2M  = sfx.variables['T2M'][:,x,y].data # for checking how far I am off (missing tile averaging etc)
        HS   = sfx.variables['H'][:,x,y].data
        PS   = dat.variables['surface_air_pressure'][:,0,x,y].data
        PA   =  PS * dat.variables['hybrid'][-1].data
        
    return dat, TA, TS, QA, QS, PZ0H, PZ0, WS, T2M, HS, PS, PA
#%%
def entire_domain_old(year,month,day,h,dn='det'):
    """
    This function gets all required fields from thredds. Always uses 00UTC run.
    
    NOTE: Thredds stores grid box averages of Z0H and Z0. Differences will be 
    observed when comparing T2M and H from this routine with forecast data due 
    to the (current) inability to replicate SURFEX's weighted approach here.
    
    Parameters:
    year   : year as YYYY
    month  : month as MM
    day    : day as DD
    h      : forcast hour to extract, for now only one timestep is allowed
    dn     : file identifier (changed between "det" and "full" on thredds)
    
    Returns:
    dat    : netCDF data handler for the file
    TA     : atm. Temperature
    TS     : surf. Temperature
    QA     : atm. specific humidity
    QS     : surf. specific humidity ()sometimes Q2M if surface is not avail)
    PZ0H   : grid box averaged roghness for heat
    PZ0    : grid box averaged rougness for momentum
    WS     : atm. wind speed
    T2M    : 2m-temperature
    HS     : sensible heat flux
    PS     : surface pressure
    PA     : atm. pressure
    """
    # hard coded search for 00 UTC start
    url = f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{year}/{month}/{day}/arome_arctic_{dn}_2_5km_{year}{month}{day}T00Z.nc'
    dat = Dataset(url)
    # get clostest x,y index to point
    lats = dat.variables['latitude'][:,:]
    lons = dat.variables['longitude'][:,:]

    if dn == 'det':
        TA   = dat.variables['air_temperature_ml'][h,-1,:,:].data
        TS   = dat.variables['air_temperature_0m'][h,0,:,:].data
        QA   = dat.variables['specific_humidity_ml'][h,-1,:,:].data
        QS   = dat.variables['SFX_Q2M'][h,:,:].data # not optimal since it is 2m... but 0m not stored, whatever
        PZ0H = dat.variables['SFX_Z0H'][h,:,:].data    
        PZ0  = dat.variables['SFX_Z0'][h,:,:].data
        UA   = dat.variables['x_wind_ml'][h,-1,:,:].data
        VA   = dat.variables['y_wind_ml'][h,-1,:,:].data
        WS   = np.sqrt(UA**2 + VA**2)
        T2M  = dat.variables['SFX_T2M'][h,:,:].data # for checking how far I am off (missing tile averaging etc)
        HS   = dat.variables['SFX_H'][h,:,:].data
        PS   = dat.variables['surface_air_pressure'][h,0,:,:].data
        PA   =  PS * dat.variables['hybrid'][-1].data
    elif dn == 'full':
        url = f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{year}/{month}/{day}/arome_arctic_sfx_2_5km_{year}{month}{day}T00Z.nc'
        sfx = Dataset(url)
        TA   = dat.variables['air_temperature_ml'][h,-1,:,:].data
        TS   = dat.variables['air_temperature_0m'][h,0,:,:].data
        QA   = dat.variables['specific_humidity_ml'][h,-1,:,:].data
        QS   = sfx.variables['Q2M'][h,:,:].data # not optimal since it is 2m... but 0m not stored, whatever
        PZ0H = sfx.variables['Z0H'][h,:,:].data    
        PZ0  = sfx.variables['Z0'][h,:,:].data
        UA   = dat.variables['x_wind_ml'][h,-1,:,:].data
        VA   = dat.variables['y_wind_ml'][h,-1,:,:].data
        WS   = np.sqrt(UA**2 + VA**2)
        T2M  = sfx.variables['T2M'][h,:,:].data # for checking how far I am off (missing tile averaging etc)
        HS   = sfx.variables['H'][h,:,:].data
        PS   = dat.variables['surface_air_pressure'][h,0,:,:].data
        PA   =  PS * dat.variables['hybrid'][-1].data
    return dat, TA, TS, QA, QS, PZ0H, PZ0, WS, T2M, HS, PS, PA
