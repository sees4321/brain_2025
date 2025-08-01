from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import mne
import os

path = 'D:\One_한양대학교\private object minsu\coding\data\EEG_fnirs_cognitive_open\datasetA'

MI_eeg = []
MA_eeg = []
MI_fnirs = []
MA_fnirs = []
MI_label = []
MA_label = []
for subj in range(1,30):
    data = loadmat(f'{path}/EEG/subject {subj:02}/with occular artifact/cnt.mat')
    event = loadmat(f'{path}/EEG/subject {subj:02}/with occular artifact/mrk.mat')
    datan = loadmat(f'{path}/NIRS/subject 01/cnt_processed.mat')
    eventn = loadmat(f'{path}/NIRS/subject 01/mrk.mat')

    MI = []
    MA = []
    MIn = []
    MAn = []
    label_MI = []
    label_MA = []
    for i in range(6):
        eeg = np.array(data['cnt'][0][i]['x'][0][0].T, float)
        eeg = mne.io.RawArray(eeg, mne.create_info([data['cnt'][0][i]['clab'][0][0][0][m][0] for m in range(32)], 200, ['eeg']*30+['eog']*2))
        events = event['mrk'][0][i]['time'][0][0][0]//5
        labels = event['mrk'][0][i]['y'][0][0][0]

        # Common average re-referencing
        eeg.set_eeg_reference()

        # filter
        eeg.notch_filter([12.5 * n for n in range(1,7)])
        eeg.filter(1,50, method='iir', iir_params=dict(order=3, ftype='butter'))

        # ica
        ica = mne.preprocessing.ICA(n_components=25, random_state=22)
        ica.fit(eeg)
        eog_indices, eog_scores = ica.find_bads_eog(eeg, ch_name=['VEOG', 'HEOG'])
        ica.exclude = eog_indices
        ica.apply(eeg)

        #nirs
        nirs = np.array(datan['out'][0][i]['x'][0][0].T, float)
        eventsn = eventn['mrk'][0][i]['time'][0][0][0]//100

        if i % 2 == 0:
            MI.append(np.stack([eeg._data[:30, events[m]:events[m]+200*10] for m in range(20)], 0))
            MIn.append(np.stack([nirs[:, eventsn[m]:eventsn[m]+10*10] for m in range(20)], 0))
            label_MI.append(labels)
        else:
            MA.append(np.stack([eeg._data[:30, events[m]:events[m]+200*10] for m in range(20)], 0))
            MAn.append(np.stack([nirs[:, eventsn[m]:eventsn[m]+10*10] for m in range(20)], 0))
            label_MA.append(labels)
    MI_eeg.append(np.concatenate(MI, 0))
    MA_eeg.append(np.concatenate(MA, 0))
    MI_fnirs.append(np.concatenate(MIn, 0))
    MA_fnirs.append(np.concatenate(MAn, 0))
    MI_label.append(np.concatenate(label_MI, 0))
    MA_label.append(np.concatenate(label_MA, 0))

MI_eeg = np.stack(MI_eeg, 0)
MA_eeg = np.stack(MA_eeg, 0)
MI_fnirs = np.stack(MI_fnirs, 0)
MA_fnirs = np.stack(MA_fnirs, 0)
MI_label = np.stack(MI_label, 0)
MA_label = np.stack(MA_label, 0)

np.savez_compressed(f'{path}/MI.npz', eeg=MI_eeg, fnirs=MI_fnirs, label=MI_label)
np.savez_compressed(f'{path}/MA.npz', eeg=MA_eeg, fnirs=MA_fnirs, label=MA_label)
print(MI_eeg.shape)
print(MA_eeg.shape)
print(MI_fnirs.shape)
print(MA_fnirs.shape)
print(MI_label.shape)
print(MA_label.shape)