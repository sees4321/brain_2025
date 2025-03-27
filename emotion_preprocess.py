from scipy import signal
from scipy.io import loadmat

import mne
import numpy as np
import pandas as pd
import os


def preprocessing_ICA(edf, event, fs_origin=2048, fs=128):
    edf._data *= 1e6

    edf.resample(128)
    event[:,0] = event[:,0] // (fs_origin // fs)
    edf.notch_filter(60)
    edf.filter(1,50, method='iir', iir_params=dict(order=2, ftype='butter'))

    ica = mne.preprocessing.ICA(n_components=7, random_state=2222)
    ica.fit(edf)
    ica.exclude = [0]
    ica.apply(edf)

    signal_filtered = edf['data'][0]

    return signal_filtered, event

def preprocessing_scipy(edf, event, fs_origin=2048, fs=128):
    # down-sampling
    signal_raw = edf['data'][0] * 1e6
    print(f'raw data shape: {signal_raw.shape}')
    # signal_raw = signal_raw[:,:signal_raw.shape[1] // fs_origin * fs_origin]
    signal_downsampled = mne.filter.resample(signal_raw, down = fs_origin // fs)
    event[:,0] = event[:,0] // (fs_origin // fs)
    print(f'downsampled data shape: {signal_downsampled.shape}')

    # notch filter
    notch_b, notch_a = signal.iirnotch(60, 30, fs)
    signal_filtered = signal.filtfilt(notch_b, notch_a, signal_downsampled)

    # band-pass filtering
    butter_b, butter_a = signal.butter(2, [1, 50], 'bandpass', fs=fs)
    signal_filtered = signal.filtfilt(butter_b, butter_a, signal_filtered)

    return signal_filtered, event

def epoching(signal_filtered, event, fs=128):
    # epoching
    eeg_resting = []
    eeg_washoff = []
    eeg_ = []
    for i in range(9):
        e = event[i,0]
        if event[i,2] == 9:
            eeg_resting.append(signal_filtered[:,e:e+fs*60])
            eeg_resting.append(signal_filtered[:,e+65*fs:e+fs*125])
        else:
            eeg_.append(signal_filtered[:,e:e+fs*120])

            mid_point = (e + fs*120 + signal_filtered.shape[1])//2 if i == 8 else (e + fs*120 + event[i+1,0])//2 
            eeg_washoff.append(signal_filtered[:,mid_point-fs*15:mid_point+fs*15])
            # if i != 8:
            #     eeg_washoff.append(signal_filtered[:,e+fs*120:event[i+1,0]])
            # else:
            #     eeg_washoff.append(signal_filtered[:,e+fs*120:])

    eeg_resting = np.stack(eeg_resting, 0)
    eeg_ = np.stack(eeg_, 0)
    eeg_washoff = np.stack(eeg_washoff, 0)

    return eeg_resting, eeg_, eeg_washoff

def labeling(path, sub):
    dat_list = os.listdir(f'{path}/{sub}')
    label_raw = pd.read_csv(f'{path}/{sub}/' + [s for s in dat_list if 'Emotion' in s][0])
    label_np = np.array(label_raw[['resting','H1','H2','P1','P2','A1','A2','S1','S2']])
    label_np[2,0] = -1
    label_ = np.stack([label_np[2], [0, 1, 1, 2, 2, 3, 3, 4, 4]])
    label_ = label_[1, np.argsort(label_[0])]

    return label_[1:]

def preprocess_and_save(path, applyICA, path2):
    sub_list = os.listdir(f'{path}')
    eeg = []
    eeg_washoff = []
    eeg_resting = []
    label = []
    for sub in sub_list:
        edf = mne.io.read_raw_bdf(f'{path}/{sub}/emo.bdf', preload=True)
        event = mne.find_events(edf)
        if len(event) > 9:
            event = event[len(event)-9:]

        if applyICA:
            signal_filtered, event = preprocessing_ICA(edf, event)
        else:
            signal_filtered, event = preprocessing_scipy(edf, event)

        if sub == '1031_YDH':
            signal_filtered[[1,3]] = signal_filtered[[3,1]]
        
        eeg_resting_, eeg_, eeg_washoff_ = epoching(signal_filtered, event)
        label_ = labeling(path, sub)


        eeg.append(eeg_)
        eeg_resting.append(eeg_resting_)
        eeg_washoff.append(eeg_washoff_)
        label.append(label_)
    
    eeg = np.stack(eeg, 0)
    eeg_resting = np.stack(eeg_resting, 0)
    eeg_washoff = np.stack(eeg_washoff, 0)
    label = np.stack(label, 0)

    datf = loadmat(f'{path2}/fNIRS_epoch.mat')
    fnirs_resting = []
    fnirs_ = []
    for i in range(9):
        if i < 2:
            fnirs_resting.append(datf['epoch'][0][0][i][0][0].T)
        else:
            fnirs_.append(datf['epoch'][0][0][i][0][0].T)
    fnirs_resting = np.stack(fnirs_resting,0)
    fnirs_ = np.stack(fnirs_,0)

    name = 'emotion_data_ica.npz' if applyICA else 'emotion_data.npz'
    np.savez_compressed(name, eeg=eeg, eeg_resting=eeg_resting, eeg_washoff=eeg_washoff, fnirs=fnirs_, fnirs_resting= fnirs_resting, label=label)

if __name__ == '__main__':
    path = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025/day1)emotion+mist'
    path2 = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025'
    preprocess_and_save(path, True, path2)
    preprocess_and_save(path, False, path2)