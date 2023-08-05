"""

"""
import glob
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os

# https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb

rate = 5        # spike rate, Hz
bin_size = 0.001 # bin size, s
tmax = 10         # the total lenght of the spike train, s
time = np.arange(0,tmax,bin_size)

def homogeneous_poisson(rate, tmax, bin_size):
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = rate * bin_size
    spikes = (np.random.uniform(size=nbins) < prob_of_spike).astype(int)
    return spikes 

spikes = homogeneous_poisson(rate, tmax, bin_size)
decay = 3
cutoff = int(decay * 2/bin_size)
exp_kernel = np.exp(-time[:cutoff]/decay)

conv_signal = np.convolve(spikes, exp_kernel, mode='full')
conv_signal_cut = conv_signal[:-len(exp_kernel)+1]
print(spikes.shape, exp_kernel.shape, conv_signal.shape, conv_signal_cut.shape)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)

ax[0].plot(time, spikes, 'g')
ax[0].set_ylabel('spikes')

ax[1].plot(time[:cutoff], exp_kernel, 'k')
ax[1].set_ylabel('au')

ax[2].plot(time, conv_signal_cut, 'b')
ax[2].set_ylabel('au')
ax[2].set_xlabel('Time [s]')

for axis in ax:
    axis.spines[['right', 'top']].set_visible(False)

fig.tight_layout()
plt.savefig('conv.png')
plt.show()

