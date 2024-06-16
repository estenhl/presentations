import matplotlib.pyplot as plt
import mne
import os


sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

df = raw.to_data_frame()
print(df)

fig = plt.figure()

for i in range(5):
    plt.plot(df['time'][:1000], df[f'EEG 00{i+1}'][:1000] - (150 * i))

plt.xlim([0, df['time'][999]])
plt.axis('off')

plt.savefig('data/eeg.png', transparent=True)

