# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
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
# Select row in dataframe
ix = 0

# %%
# Get sampling rate and center frequency
Ts = IQdf.loc[ix, 'sample_period']
Fs = int(np.round(1/Ts))   # sampling rate
print(Fs)
fc = IQdf.loc[ix, 'center_frequency']
print(fc)


# %%
# Get I and Q data strings
yIt_str = IQdf.loc[ix, 'i']
yQt_str = IQdf.loc[ix, 'q']
print(yIt_str[:60])
print(yQt_str[:60])

# %%
# Convert IQ strings to complex-valued numpy array
yt = np.array(yIt_str[1:-1].split(','), dtype=float) + 1j*np.array(yQt_str[1:-1].split(','), dtype=float)
print(yt[:10])

# %%
# Display PSD of selected IQ data
thr = 0
NB = 256
ff, Syf, Syavgf = lte.computePSDw(yt, Fs, thr, NB)
psdcorr = 0.5   # correction baseband to bandpass power (baseband pwr=2*bandpass pwr)
Syf = 1000/50*psdcorr*Syf    # change for dBm and bandpass power
fig, axes = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
lte.psd4mplot((ff+fc)*1e-6, Syf, axes, ylbl='Pwr [dBm]', xlbl='$f$ [MHz]')
axes.set_title(f'PSD, {sigName}, ix={ix}')
plt.tight_layout()
plt.show()

# %%
