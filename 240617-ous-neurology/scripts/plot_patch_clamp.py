import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


data = 'scripts/data/SST_PFC_EPSCs.abf'
abf = pyabf.ABF(data)
print(abf)

time = abf.sweepX  # in seconds
current = abf.sweepY
fs = int(abf.dataPointsPerMs *1000)
# Optional: if you have several sweeps/channels, select the right one
# abf.setSweep(sweepNumber=1, channel=0)


# Define the signal variable: current or voltage
signal_raw = current

# Subtract mean or median of the full trace
signal_adjusted = signal_raw - np.median(signal_raw)

# Define the variables and sampling frequency
time = abf.sweepX  # in seconds
current = abf.sweepY
fs = int(abf.dataPointsPerMs *1000)

# Lowpass Bessel filter
b_lowpass, a_lowpass = signal.bessel(4,     # Order of the filter
                                     2000,  # Cutoff frequency
                                     'low', # Type of filter
                                     analog=False,  # Analog or digital filter
                                     norm='phase',  # Critical frequency normalization
                                     fs=fs)  # fs: sampling frequency

# Notch Besse filter (comment out if not needed)
b_notch, a_notch = signal.bessel(1,     # Order of the filter
                                 [59, 61],  # Cutoff frequency
                                 'bandstop', # Type of filter
                                 analog=False,  # Analog or digital filter
                                 fs=fs)  # fs: sampling frequency

# Combine both filter (comment out if not needed)
b_multiband = signal.convolve(b_lowpass, b_notch)
a_multiband = signal.convolve(a_lowpass, a_notch)

# For 2 filters (comment out if not needed)
signal_filtered = signal.filtfilt(b_multiband, a_multiband, signal_adjusted)

# For 1 filter
# signal_filtered = signal.filtfilt(b_lowpass, a_lowpass, signal_adjusted)

# Plot the raw trace
fig = plt.figure(figsize=(12, 6))
plt.plot(time[:1000], current[:1000], linewidth=0.5)

# Show graph and table
plt.axis('off')
plt.savefig('data/patch_clamp.png', transparent=True)
