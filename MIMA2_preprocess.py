from scipy.io import loadmat
import numpy as np
import mne

path = 'D:\One_한양대학교\private object minsu\coding\data\EEG_fnirs_cognitive_open\datasetB'

MI_eeg = []
MI_fnirs = []
MI_label = []
for subj in range(1,27):
    data = loadmat(f'{path}/EEG/VP0{subj:02}-EEG/cnt_wg.mat')
    event = loadmat(f'{path}/EEG/VP0{subj:02}-EEG/mrk_wg.mat')
    datan = loadmat(f'{path}/NIRS/VP0{subj:02}-NIRS/cnt_wg.mat')
    eventn = loadmat(f'{path}/NIRS/VP0{subj:02}-NIRS/mrk_wg.mat')

    eeg = np.array(data['cnt_wg'][0][0]['x'].T, float)
    eeg = mne.io.RawArray(eeg, mne.create_info([data['cnt_wg'][0][0]['clab'][0][m][0] for m in range(30)], 200, ['eeg']*28+['eog']*2))
    events = event['mrk_wg'][0][0]['time'][0]//5
    labels = event['mrk_wg'][0][0]['y'][0]

    # Common average re-referencing
    eeg.set_eeg_reference()

    # filter
    eeg.notch_filter(50)
    eeg.filter(1,50, method='iir', iir_params=dict(order=3, ftype='butter'))

    # ica
    ica = mne.preprocessing.ICA(n_components=25, random_state=22)
    ica.fit(eeg)
    eog_indices, eog_scores = ica.find_bads_eog(eeg, ch_name=['VEOG', 'HEOG'])
    ica.exclude = eog_indices
    ica.apply(eeg)

    #nirs
    nirs = np.concatenate([datan['cnt_wg']['oxy'][0][0]['x'][0][0].T, datan['cnt_wg']['deoxy'][0][0]['x'][0][0].T], dtype=float)*1e4
    eventsn = eventn['mrk_wg'][0][0]['time'][0]//100
    nirs = mne.filter.filter_data(nirs, 10, 0.01, 0.2, method='iir', iir_params=dict(order=3, ftype='butter'))

    MI_eeg.append(np.stack([eeg._data[:28, events[m]:events[m]+200*10] for m in range(60)], 0))
    MI_fnirs.append(np.stack([nirs[:, eventsn[m]:eventsn[m]+10*10] for m in range(60)], 0))
    MI_label.append(labels)

MI_eeg = np.stack(MI_eeg, 0)
MI_fnirs = np.stack(MI_fnirs, 0)
MI_label = np.stack(MI_label, 0)

np.savez_compressed(f'{path}/WG.npz', eeg=MI_eeg, fnirs=MI_fnirs, label=MI_label)
print(MI_eeg.shape)
print(MI_fnirs.shape)
print(MI_label.shape)