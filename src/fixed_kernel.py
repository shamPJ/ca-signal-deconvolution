"""

"""
import glob
import h5py
from load_data import get_data
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

def get_matrices(data):
    n_samples = 1000
    ca_traces = np.zeros((len(data), n_samples))
    ephys_traces = np.zeros((len(data), n_samples))

    for i, key in enumerate(data):
        exp = data[key]
        # Get data from dict
        dto, dte, dff, ephys, sptimes, quiroga = exp["dto"], exp["dte"], exp["dff"], exp["ephys"], exp["sptimes"], exp["quiroga"]
        ca_traces[i] = dff[:n_samples]
        ephys_traces[i] = ephys[:n_samples]

    return dto, dte, ca_traces, ephys_traces

def low_pass_filt(trace):
    pass 

def smoothing(ca_traces):
    ca_traces_filt = gaussian_filter1d(ca_traces, sigma=1)
    return ca_traces_filt

def decay_const_search(traces):
    pass

def plot_func_filt(dto, ca_traces, ca_traces_filt):
    # Plot and save traces
    fig, ax = plt.subplots(len(ca_traces), 2, sharex=True, sharey=True)

    for i in range(len(ax)):
        n = len(ca_traces[i])
        ax[i, 0].plot(np.arange(0, n*dto, dto)[:n], ca_traces[i], 'g')
        ax[i, 0].set_ylabel('DF/F')

        ax[i, 1].plot(np.arange(0, n*dto, dto)[:n], ca_traces_filt[i], 'g')
        ax[i, 1].set_ylabel('DF/F filtered')

    for axis in ax:
        axis[0].spines[['right', 'top']].set_visible(False)
        axis[1].spines[['right', 'top']].set_visible(False)

    plt.savefig('traces_filt.png')

# Returns dict with keys - .h5 file name; values - another dict with data (keys "genotype", "dff", "ephys", "dto", "dte", "sptimes", "quiroga"})
data = get_data()
dto, dte, ca_traces, ephys_traces = get_matrices(data)
ca_traces_filt = smoothing(ca_traces)

plot_func_filt(dto, ca_traces, ca_traces_filt)


