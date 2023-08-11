from FastLZeroSpikeInference import fast
from load_data import get_data
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plot_func(y, y_pred, sptimes, sptimes_est, speriod):
    # Plot and save traces
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

    ax[0].plot(np.arange(0, len(y)*speriod, speriod), y, 'g') 
    ax[0].set_ylabel('dff')

    ax[1].plot(np.arange(0, len(y_pred)*speriod, speriod), y_pred, 'g')
    ax[1].set_ylabel('est. dff')

    ax[2].eventplot(sptimes, orientation='horizontal', lineoffsets=-1, linelengths=0.5, linewidths=None, colors='r', linestyles='solid')
    ax[2].set_ylabel('spikes')

    ax[3].eventplot(sptimes_est*dto, orientation='horizontal', lineoffsets=-1, linelengths=0.5, linewidths=None, colors='r', linestyles='solid')
    ax[3].set_ylabel('est. spikes')

    for axis in ax:
        axis.spines[['right', 'top']].set_visible(False)
        
    plt.savefig('spike_inference.png')
    plt.show()

# -----------------------Toy example---------------------#
# from https://jewellsean.github.io/fast-spike-deconvolution/python_tutorial.html
gam = 0.952
x = np.concatenate([np.arange(100), np.arange(100)])
y = np.power(gam, x)

fit = fast.estimate_spikes(y, gam, 1, False, True)
y_pred = fit['estimated_calcium']
sptimes = fit['spikes']

#----------------------Allen calibration dataset----------------------#
gam = 0.986
data = get_data()

for key in iter(data):
        data_dict = data[key]
        dto, dte, dff, ephys, sptimes, quiroga = data_dict["dto"], data_dict["dte"], data_dict["dff"], data_dict["ephys"], data_dict["sptimes"], data_dict["quiroga"]
        
        fit = fast.estimate_spikes(dff, gam, 1, False, True)
        dff_pred = fit['estimated_calcium']
        sptimes_est = fit['spikes']
        plot_func(dff, dff_pred, sptimes, sptimes_est, dto)
    




