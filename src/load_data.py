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

Data loaded as .h5 files, which contains data structure with
<KeysViewHDF5 ['dff', 'dte', 'dto', 'ephys_baseline_subtracted', 'ephys_raw', 'genotype', 'quiroga', 'sptimes']>

- dff
- dte - sampling periods for ephys 1/sampling rate, seconds
- dto - sampling periods for two-photon 1/sampling rate, seconds
- ephys_baseline_subtracted
- ephys_raw
- genotype
- quiroga
- sptimes

"""
import glob
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os


def get_data():
    # Get current working directory
    wd = os.getcwd()
    # List to store names of all .h5 files in the directory
    data = {}

    # Get all .h5 files' names
    path = os.path.join(wd, "../data", "*.h5")
    for file in glob.glob(path):
        file_name = os.path.split(file)[-1].split(".")[0]
        # Read h5 file
        h5 = h5py.File(file,'r')
        # Get data from h5 datastructure
        dto = 1.0 * np.array(h5['dto'])  # sampling periods 1/sampling rate, seconds
        dte = 1.0 * np.array(h5['dte'])
        quiroga = 1.0 * np.array(h5['quiroga'])
        dff = np.array(h5['dff']).ravel()
        sptimes = np.array(h5['sptimes']).ravel()
        genotype = h5['genotype'][()].decode("utf-8") 
        ephys = np.array(h5['ephys_baseline_subtracted']).ravel()

        # Store data in dict
        data[file_name] = {"genotype" : genotype, "dff": dff, "ephys": ephys, "dto": dto, "dte": dte, "sptimes": sptimes, "quiroga": quiroga}
    
    print("Number of .h5 files found in dir: ", len(data))

    return (data)

def plot_func(data_dict):
    # Get data from dict
    dto, dte, dff, ephys, sptimes, quiroga = data_dict["dto"], data_dict["dte"], data_dict["dff"], data_dict["ephys"], data_dict["sptimes"], data_dict["quiroga"]
    
    # binning at 1 second, sampling rate 4KHz
    stop = int( int(len(ephys) * dte) / dte )
    spikes = (ephys >= quiroga).astype(int) 
    frate = np.sum(spikes[:stop].reshape(-1, int(1/dte)), axis=-1) / 1
   
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

    ax[3].plot(np.arange(len(frate)), frate, 'b')
    ax[3].set_ylabel('firing rate')

    for axis in ax:
        axis.spines[['right', 'top']].set_visible(False)
    plt.savefig('traces.png')

def main():
    data = get_data()
    for key in iter(data):
        plot_func(data[key])
    plt.show()
if __name__ == "__main__":
    main()


