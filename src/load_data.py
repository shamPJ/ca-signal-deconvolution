"""

Ophys/Ephys Calibration Data.
https://portal.brain-map.org/explore/circuits/oephys

Dataset consist of simultaneously measured fluorescence and spiking activity 
of pyramidal neurons in layer 2/3 of primary visual cortex in transgenic mouse lines expressing 
genetically-encoded calcium indicators (GECIs) GCaMP6s or GCaMP6f.

We are using DATASET II:  JOINTLY RECORDED NEURONS WITH NOISE PROFILE MATCHED TO THE ALLEN BRAIN OBSERVATORY.

Using a novel, crowd-sourced, algorithmic approach to quality control, we further present a highly curated subset of 50 recordings 
from 35 cells, in which the high-zoom calcium imaging data was downsampled spatially and temporally to match the data quality observed 
in the Allen Brain Observatory (i.e., noise levels typically encountered in population-level two-photon imaging). 
The downsampled calibration dataset is described in Ledochowitsch et al., On the correspondence of electrical 
and optical physiology in in vivo population-scale two-photon calcium imaging.

Data loaded as .h5 files. Cux2-f (GCaMP6f) file wasn't available. Loaded Emx1-f (GCaMP6f) and Emx1-s (GCaMP6s) datasets.

h5 file contains data structure with
<KeysViewHDF5 ['dff', 'dte', 'dto', 'ephys_baseline_subtracted', 'ephys_raw', 'genotype', 'quiroga', 'sptimes']>

- dff
- dte
- dto
- ephys_baseline_subtracted
- ephys_raw
- genotype
- quiroga
- sptimes

"""
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os

wd = os.getcwd()
filename = "/data/102525.h5"
h5 = h5py.File(wd+filename,'r')

# Print objects in the h5 datastructure
print(h5.keys())

# Get data from h5 datastructure
dto = 1.0 * np.array(h5['dto'])
dte = 1.0 * np.array(h5['dte'])
quiroga = 1.0 * np.array(h5['quiroga'])
dff = np.array(h5['dff']).ravel()
sptimes = np.array(h5['sptimes']).ravel()
genotype = h5['genotype'][()].decode("utf-8") 
ephys = np.array(h5['ephys_baseline_subtracted']).ravel()

# Plot and save traces
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)

ax[0].plot(np.arange(0, len(dff)*dto, dto), dff, 'g')
ax[0].set_ylabel('DF/F')

ax[1].plot(np.arange(0, len(ephys)*dte, dte), 1000 * ephys, 'k')
ax[1].set_ylabel('mV')
ax[1].axhline(y=1000.0 * quiroga,  color = 'r', alpha = 0.5)

ax[2].eventplot(sptimes, orientation='horizontal', lineoffsets=-1, linelengths=0.5, linewidths=None, colors='r', linestyles='solid')
ax[2].set_ylabel('spikes')
ax[2].set_xlabel('Time [s]')
ax[2].tick_params(labelleft=False) 

for axis in ax:
    axis.spines[['right', 'top']].set_visible(False)

plt.savefig('traces.png')

print("dFF trace len", dff.shape)
print("ephys trace len", ephys.shape)

print("sampling period, ephys trace ", dte)
print("sampling period, calcium trace ", dto)
