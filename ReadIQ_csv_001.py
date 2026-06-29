# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: polar [conda env:polar]
#     language: python
#     name: conda-env-polar-polar
# ---

# %% [markdown]
# # ReadIQ_csv 001
#
# Read IQ samples from time-data.csv, recoded in CRFS Mission Manager 

# %%
import numpy as np
import matplotlib.pyplot as plt
import myLTE as lte
import csv
import pandas as pd


# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.2*fsz[1]/2.0)

# %%
# Read single IQ data record in
# IQ_080525a_csv_N1/time-data.csv
#sigName = 'IQ_080525a_N1'
#folder_in = 'IQ_080525a_csv_N1'
#fname = 'time-data'
#-------------------------------------------------
# Read IQ data records in
# IQ_080525a_csv_N4/time-data.csv
sigName = 'IQ_080525a_N4'
folder_in = 'IQ_080525a_csv_N4'
fname = 'time-data'
#-------------------------------------------------

full_fname = f'{folder_in}/{fname}'
IQdf = pd.read_csv(full_fname + '.csv', header=0)   # IQ dataframe
IQdf.head()

# %%
# List column names
col_list = IQdf.columns.tolist()
print(col_list)

# %%
# Get sampling rate
Ts = IQdf.loc[0, 'sample_period']
Fs = int(np.round(1/Ts))   # sampling rate
print(Fs)


# %%
# Get I and Q data strings
yIt_str = IQdf.loc[0, 'i']
yQt_str = IQdf.loc[0, 'q']
print(yIt_str[:60])
print(yQt_str[:60])

# %%
