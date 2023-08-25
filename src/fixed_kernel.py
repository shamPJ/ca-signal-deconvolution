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

def get_traces(data):

    """
    :out dto:       float, sampling period 2P imaging
    :out dte:       float, sampling period ephys recordings
    :out ca_traces: 
    :out ephys_traces: 
    """

    trace_len    = 100     # in seconds
    ca_traces    = []
    ephys_traces = []

    for i, key in enumerate(data):
        exp = data[key]
        # Get data from dict
        dto, dte, dff, ephys, s, q = exp["dto"], exp["dte"], exp["dff"], exp["ephys"], exp["sptimes"], exp["quiroga"]
        ca_traces.append(dff[:int(trace_len / dto)])
        ephys_traces.append(ephys[:int(trace_len / dte)])

    return dto, dte, ca_traces, ephys_traces

def low_pass_filt(trace):
    pass 

def smoothing(ca_traces):
    ca_traces_filt = gaussian_filter1d(ca_traces, sigma=1)
    return ca_traces_filt

def decay_const_search(traces):
    pass

# def plot_func_filt(dto, ca_traces, ca_traces_filt):
#     # Plot and save traces
#     fig, ax = plt.subplots(len(ca_traces), 2, sharex=True, sharey=True)

#     for i in range(len(ax)):
#         n = len(ca_traces[i])
#         ax[i, 0].plot(np.arange(0, n*dto, dto)[:n], ca_traces[i], 'g')
#         ax[i, 0].set_ylabel('DF/F')

#         ax[i, 1].plot(np.arange(0, n*dto, dto)[:n], ca_traces_filt[i], 'g')
#         ax[i, 1].set_ylabel('DF/F filtered')

#     for axis in ax:
#         axis[0].spines[['right', 'top']].set_visible(False)
#         axis[1].spines[['right', 'top']].set_visible(False)

#     plt.savefig('traces_filt.png')

def deconv(data_dict):

    # Get data from dict
    dto, dte, dff, ephys, sptimes, quiroga = data_dict["dto"], data_dict["dte"], data_dict["dff"], data_dict["ephys"], data_dict["sptimes"], data_dict["quiroga"]
    
    time_2p    = np.arange(len(dff))
    t_decay    = 3 # in seconds
    cutoff     = int(t_decay * 2/dto)

    # Kernel
    exp_kernel = np.exp(-time_2p[:cutoff]/t_decay)

    # Deconvolution
    
    kernel_fft     = fft(exp_kernel)       # 1. Compute kernel DFT
    inverse_kernel = ifft(1/kernel_fft)    # 2. Compute inverse DFT and corresponding kernel

    dff = gaussian_filter1d(dff, sigma=10)
    deconv_signal = np.convolve(dff, inverse_kernel, mode='full')
    deconv_signal_cut = deconv_signal[:-len(exp_kernel)+1]
    print(dff.shape, ephys.shape)
    print(kernel_fft.shape, inverse_kernel.shape, deconv_signal_cut.shape)

    # Plot and save traces
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

    ax[0].plot(np.arange(0, len(dff)*dto, dto), dff, 'g')
    ax[0].set_ylabel('DF/F')

    ax[1].plot(np.arange(0, len(ephys)*dte, dte), 1000 * ephys, 'k')
    ax[1].set_ylabel('mV')
    ax[1].axhline(y=1000.0 * quiroga,  color = 'r', alpha = 0.5)

    ax[2].eventplot(sptimes, orientation='horizontal', lineoffsets=-1, linelengths=0.5, linewidths=None, colors='r', linestyles='solid')
    ax[2].set_ylabel('spikes')
    ax[2].set_xlabel('Time [s]')
    ax[2].tick_params(labelleft=False) 

    ax[3].plot(np.arange(0, len(dff)*dto, dto), deconv_signal_cut, 'g')
    #ax[3].plot(np.arange(0, len(exp_kernel)*dto, dto), exp_kernel, 'k')
    ax[3].set_ylabel('deconv signal')

    for axis in ax:
        axis.spines[['right', 'top']].set_visible(False)
    plt.savefig('fixed_filt_deconv.png')

# Returns dict with keys - .h5 file name; values - another dict with data (keys "genotype", "dff", "ephys", "dto", "dte", "sptimes", "quiroga"})
data = get_data()
exp_id = next(iter(data))
deconv(data[exp_id])

#dto, dte, ca_traces, ephys_traces = get_traces(data)
# print(f"dto and dte values are {dto, dte}")
# print(f"2P and ephys traces shapes are {ca_traces[0].shape, ephys_traces[0].shape}")
# ca_traces_filt = smoothing(ca_traces)
# print(dto, dte, ca_traces.shape, ephys_traces.shape)
# plot_func_filt(dto, ca_traces, ca_traces_filt)


