"""

"""
import glob
import h5py
from load_data import get_data
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d

# https://github.com/btel/python-in-neuroscience-tutorials/blob/master/poisson_process.ipynb

rate = 5          # spike rate, Hz
bin_size = 0.001  # bin size, s
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

# Kernel
exp_kernel = np.exp(-time[:cutoff]/decay)

# Convolution
conv_signal = np.convolve(spikes, exp_kernel, mode='full')
conv_signal_cut = conv_signal[:-len(exp_kernel)+1]
conv_signal_cut = conv_signal_cut + np.random.normal(0, 0.1, len(conv_signal_cut))
print(spikes.shape, exp_kernel.shape, conv_signal.shape, conv_signal_cut.shape)

# Deconvolution

# 1. Compute kernel DFT
kernel_fft = fft(exp_kernel)

# 2. Compute inverse DFT and corresponding kernel
inverse_fft = ifft(1/kernel_fft)

conv_signal_filt = gaussian_filter1d(conv_signal_cut, sigma=2)
deconv_signal = np.convolve(conv_signal_filt, inverse_fft, mode='full')
deconv_signal_cut = deconv_signal[:-len(exp_kernel)+1]
print(kernel_fft.shape, inverse_fft.shape, deconv_signal.shape)

fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

ax[0].plot(time, spikes, 'g')
ax[0].set_ylabel('spikes')

ax[1].plot(time[:cutoff], exp_kernel, 'k')
ax[1].plot(time[:cutoff], inverse_fft.real, 'r--')
#ax[1].plot(time[:cutoff], inverse_fft.imag, 'b--')
#ax[1].plot(time[:cutoff], ((1/kernel_fft)).real, 'b--')
ax[1].set_ylabel('au')
#ax[1].set_xscale('log')

ax[2].plot(time, conv_signal_cut, 'b')
ax[2].plot(time, conv_signal_filt, 'r')
ax[2].set_ylabel('au')
ax[2].set_xlabel('Time [s]')

ax[3].plot(time, deconv_signal_cut, 'b')
ax[3].set_ylabel('au')
ax[3].set_xlabel('Time [s]')

for axis in ax:
    axis.spines[['right', 'top']].set_visible(False)

fig.tight_layout()
plt.savefig('conv_deconv.png')
plt.show()

