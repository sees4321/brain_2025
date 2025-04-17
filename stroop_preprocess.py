import numpy as np
import os
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import mne

path = 'D:\One_한양대학교\private object minsu\coding\data\\fNIRS-EEG_Stroop'

eeg_ = []

for i in range(1,22):
    print(f'-------------------------{i}-----------------------')
    if i == 5: continue
    data = loadmat(f'{path}\Raw Data\Raw EEG Data\in mat\S{i}.mat')
    eeg = np.array(data['EEG']['data'][0][0], float)
    eeg = mne.io.RawArray(eeg, mne.create_info([data['EEG']['chanlocs'][0][0][0][m][0][0] for m in range(34)], 1000, ['eeg']*30+['eog']*4))
    events = np.array([data['EEG']['event'][0][0][0][m][1][0][0] for m in range(64)])//8

    # downsample
    # plt.figure(figsize=(22,4))
    # plt.subplot(211)
    # plt.plot(eeg['eeg'][0][0])
    # print(len(eeg))
    eeg.resample(128)
    # plt.subplot(212)
    # plt.plot(eeg['eeg'][0][0])
    # print(len(eeg))
    # plt.show()

    # Cz re-referencing
    eeg.set_eeg_reference(["CZ"])
    eeg.drop_channels(['CZ'])

    # filtering
    # plt.figure(figsize=(22,4))
    # plt.subplot(211)
    # plt.plot(eeg['eeg'][0][0])
    eeg.notch_filter(50)
    eeg.filter(0.5,45, method='iir', iir_params=dict(order=3, ftype='butter'))
    # plt.subplot(212)
    # plt.plot(eeg['eeg'][0][0])
    # plt.show()

    # ica
    ica = mne.preprocessing.ICA(n_components=25, random_state=2222)
    ica.fit(eeg)

    ica.plot_sources(eeg, show_scrollbars=True)
    # ica.plot_overlay(eeg, exclude=[0,1], start=128*30, stop=128*40)
    plt.show()
    nums = numbers = input("숫자들을 입력하세요 (쉼표로 구분): ").split(',')
    numbers = [int(num) for num in numbers]  # 각 값을 정수로 변환

    # plt.figure(figsize=(22,4))
    # plt.subplot(211)
    # plt.plot(eeg['eeg'][0][0][128*0:128*30])
    ica.exclude = numbers
    ica.apply(eeg)
    # plt.subplot(212)
    # plt.plot(eeg['eeg'][0][0][128*0:128*30])
    # plt.show()
    eeg_.append(np.stack([eeg['eeg'][0][:,events[m*4+15]-128*30:events[m*4+15]+128*2] for m in range(4)]))

eeg_ = np.stack(eeg_, 0)

import os 
fnirs = []
for file_name in os.listdir(f'{path}\Pre-processed Data\Pre-processed fNIRS Data\in mat')[1:]:
    temp = loadmat(f'{path}\Pre-processed Data\Pre-processed fNIRS Data\in mat\{file_name}')
    fnirs.append(temp[list(temp.keys())[-1]].swapaxes(1,2))
fnirs = np.stack([np.concatenate((fnirs[m], fnirs[m+4]), 1) for m in range(4)], 1)

fnirs = np.delete(fnirs, 5, 0)

label = np.zeros(fnirs.shape[:2])
label[:,:2] = 1

np.savez_compressed(f'{path}\stroop.npz', eeg=eeg_, fnirs=fnirs, label=label)

"""
-------------------------1-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1
-------------------------2-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,5
-------------------------3-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,2
-------------------------4-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,2
-------------------------5-----------------------
-------------------------6-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,2
-------------------------7-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,3
-------------------------8-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0
-------------------------9-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1
-------------------------10-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1
-------------------------11-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,5
-------------------------12-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,2
-------------------------13-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1
-------------------------14-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,18
-------------------------15-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0
-------------------------16-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1
-------------------------17-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0 
-------------------------18-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0
-------------------------19-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,2,7
-------------------------20-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,2
-------------------------21-----------------------
숫자들을 입력하세요 (쉼표로 구분): 0,1,2,8
"""