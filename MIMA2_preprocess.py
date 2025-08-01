from scipy.io import loadmat
import numpy as np
import mne

path = 'D:\One_한양대학교\private object minsu\coding\data\EEG_fnirs_cognitive_open\datasetB'

task = 'nback'
# task = 'wg'
# task = 'dsr'

MI_eeg = []
MI_fnirs = []
MI_label = []
for subj in range(1,27):
    data = loadmat(f'{path}/EEG/VP0{subj:02}-EEG/cnt_{task}.mat')
    event = loadmat(f'{path}/EEG/VP0{subj:02}-EEG/mrk_{task}.mat')
    datan = loadmat(f'{path}/NIRS/VP0{subj:02}-NIRS/cnt_{task}.mat')
    eventn = loadmat(f'{path}/NIRS/VP0{subj:02}-NIRS/mrk_{task}.mat')

    eeg = np.array(data[f'cnt_{task}'][0][0]['x'].T, float)
    eeg = mne.io.RawArray(eeg, mne.create_info([data[f'cnt_{task}'][0][0]['clab'][0][m][0] for m in range(30)], 200, ['eeg']*28+['eog']*2))
    events = event[f'mrk_{task}'][0][0]['time'][0]//5
    labels = event[f'mrk_{task}'][0][0]['y'][0]

    # Common average re-referencing
    eeg.set_eeg_reference()

    # filter
    eeg.notch_filter(50)
    eeg.filter(1,50, method='iir', iir_params=dict(order=3, ftype='butter'))

    # ica
    ica = mne.preprocessing.ICA(n_components=20, random_state=22)
    ica.fit(eeg)
    eog_indices, eog_scores = ica.find_bads_eog(eeg, ch_name=['VEOG', 'HEOG'])
    ica.exclude = eog_indices
    ica.apply(eeg)

    #nirs
    nirs = np.concatenate([datan[f'cnt_{task}']['oxy'][0][0]['x'][0][0].T, datan[f'cnt_{task}']['deoxy'][0][0]['x'][0][0].T], dtype=float)*1e4
    eventsn = eventn[f'mrk_{task}'][0][0]['time'][0]//100
    nirs = mne.filter.filter_data(nirs, 10, 0.01, 0.2, method='iir', iir_params=dict(order=3, ftype='butter'))

    if task == 'wg':
        MI_eeg.append(np.stack([eeg._data[:28, events[m]:events[m]+200*10] for m in range(60)], 0))
        MI_fnirs.append(np.stack([nirs[:, eventsn[m]:eventsn[m]+10*10] for m in range(60)], 0))
        MI_label.append(labels)
    elif task == 'nback':
        temp1 = []
        temp2 = []
        for i, j in [(112,7), (128,8), (144,9)]:
            for m in np.where(event['mrk_nback'][0][0]['event'][0][0][0] == i)[0]:
                temp1.append(eeg._data[:28, events[m]:events[m]+200*40])
            for m in np.where(eventn['mrk_nback'][0][0]['event'][0][0][0] == j)[0]:
                temp2.append(nirs[:, eventsn[m]:eventsn[m]+10*40])
        MI_eeg.append(np.stack(temp1,0))
        MI_fnirs.append(np.stack(temp2,0))
        MI_label.append([0]*9+[1]*9+[2]*9)
    elif task == 'dsr':
        temp1 = []
        temp2 = []
        for m in np.where(event[f'mrk_{task}'][0][0]['event'][0][0][0] == 48)[0]:
            temp1.append(eeg._data[:28, events[m]:events[m]+200*20])
            temp1.append(eeg._data[:28, events[m]+200*40:events[m]+200*60])
        for m in np.where(eventn[f'mrk_{task}'][0][0]['event'][0][0][0] == 3)[0]:
            temp2.append(nirs[:, eventsn[m]:eventsn[m]+10*20])
            temp2.append(nirs[:, eventsn[m]+10*40:eventsn[m]+10*60])
        if subj == 1:
            m = eventsn[-1]
            temp2.append(nirs[:, m:m+10*20])
            temp2.append(nirs[:, m+10*40:m+10*60])
        MI_eeg.append(np.stack(temp1,0))
        MI_fnirs.append(np.stack(temp2,0))
        MI_label.append([0,1]*18)
    else:
        print('error')
        break


MI_eeg = np.stack(MI_eeg, 0)
MI_fnirs = np.stack(MI_fnirs, 0)
MI_label = np.stack(MI_label, 0)

np.savez_compressed(f'{path}/{task}.npz', eeg=MI_eeg, fnirs=MI_fnirs, label=MI_label)
print(MI_eeg.shape)
print(MI_fnirs.shape)
print(MI_label.shape)