A recreation of all calculations required for T2M in HARMONIE-AROME. Including Richardsnon number, turbulent exchange coefficients, heat flux, and Prandtl number. 
Designed to get an intuitive idea about the model's behviour. Includes the option of extracting model data from thredds to "rerun" cases.

contains:
HARMONIE.py : all the routines translated from Fortran to python

calc_T2M.py : examples on how to use the functions and create some figures

extract_thredds.py : functions to extract either time series or entire model domain data from the operational archive of Met-Norway
