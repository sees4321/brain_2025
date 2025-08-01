import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from utils import *

class Emotion_DataModule():
    r'''
    Create emotion dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        data_mode (int): 0 - eeg + fnirs, 1 - only eeg, 2 - only fnirs
        label_type (int): 0 - arousal classification, 1 - valence classification
        ica (bool): load data_ica (default: True)
        start_point (int): start_point of the segmentation in seconds. (default: 60)
        window_len (int): window length in seconds for segmentation. (default: 60)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str, # D:/One_한양대학교/private object minsu/coding/data/brain_2025
                 data_mode:int = 0,
                 label_type:int = 0,
                 ica:bool = True,
                 start_point:int = 60,
                 window_len:int = 60,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform_eeg = None,
                 transform_fnirs = None,
                 ):
        super().__init__()

        self.data_mode = data_mode
        self.batch_size = batch_size
        self.num_val = num_val
        self.test_idx = 0
        
        # load data
        if ica:
            data = np.load(f'{path}/emotion_data_ica.npz')
        else:
            data = np.load(f'{path}/emotion_data.npz')
        self.fnirs = np.load(f'{path}/emo_fnirs.npy') # (36, 8, 26, 371) 
        # self.fnirs = data['fnirs'] # (36, 8, 26, 371) 
        # self.fnirs_resting = data['fnirs_resting'] # (36, 8, 26, 247) 
        self.eeg = data['eeg'] # (36, 8, 7, 15360)
        self.eeg_resting = data['eeg_resting'] # (36, 2, 7, 7680)
        self.eeg_washoff = data['eeg_washoff'] # (36, 8, 7, 3840)
        self.label = data['label'] # (36, 8)
        self.subjects = [i for i in range(self.eeg.shape[0])]

        # labeling
        if label_type == 0: # arousal
            self.label[self.label == 1.0] = 1
            self.label[self.label == 2.0] = 0
            self.label[self.label == 3.0] = 1
            self.label[self.label == 4.0] = 0
        else: # valence
            self.label[self.label == 1.0] = 1
            self.label[self.label == 2.0] = 1
            self.label[self.label == 3.0] = 0
            self.label[self.label == 4.0] = 0

        # segmentation
        self.eeg, n_windows = self.window_slicing(self.eeg, start_point, window_len, 120)
        self.fnirs, n_windows = self.window_slicing(self.fnirs, start_point-60, window_len, 60)
        self.label = np.repeat(self.label, n_windows, 1)
        self.data_shape_eeg = list(self.eeg.shape[-2:])
        self.data_shape_fnirs = list(self.fnirs.shape[-2:])

        if transform_eeg:
            self.eeg = transform_eeg(self.eeg)
        if transform_fnirs:
            self.fnirs = transform_fnirs(self.fnirs)
    
    def __len__(self):
        return self.subjects

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.subjects):
            eeg_torch = torch.from_numpy(self.eeg[self.subjects[self.test_idx]]).float()
            fnirs_torch = torch.from_numpy(self.fnirs[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            train_subjects, val_subjects = self.train_val_split()
            eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in train_subjects])).float()
            fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            self.test_idx += 1
            if len(val_subjects) > 0:
                eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in val_subjects])).float()
                fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in val_subjects])).float()
                label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
                val_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, True)

                return train_loader, val_loader, test_loader
            return train_loader, None, test_loader
        else:
            raise StopIteration
    
    def create_dataloader(self, eeg, fnirs, label, shuffle=False):
        if self.data_mode == 0:
            return DataLoader(BimodalDataSet(eeg, fnirs, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 1:
            return DataLoader(CustomDataSet(eeg, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 2:
            return DataLoader(CustomDataSet(fnirs, label), self.batch_size, shuffle=shuffle)
    
    def window_slicing(self, arr, start_point, window_len, total_sec):
        total_len = arr.shape[-1]
        
        start_idx = int(total_len * start_point / total_sec)
        window_length = int(total_len * window_len / total_sec)
        n_windows = (total_len - start_idx) // window_length
        
        out = np.stack([arr[:,:,:,start_idx + i * window_length : start_idx + (i + 1) * window_length] for i in range(n_windows)], 1)
        out = np.swapaxes(out, 0, 2)
        out = np.concatenate(out)
        out = np.swapaxes(out, 0, 1)
        return out, n_windows

    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]


class MIST_DataModule():
    r'''
    Create MIST dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        data_mode (int): 0 - eeg + fnirs, 1 - only eeg, 2 - only fnirs
        start_point (int): start_point of the segmentation in seconds. (default: 60)
        window_len (int): window length in seconds for segmentation. (default: 60)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str, # D:/One_한양대학교/private object minsu/coding/data/brain_2025
                 data_mode:int = 0,
                 start_point:int = 0,
                 window_len:int = 60,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform_eeg = None,
                 transform_fnirs = None,
                 ):
        super().__init__()

        self.data_mode = data_mode
        self.batch_size = batch_size
        self.num_val = num_val
        self.test_idx = 0
        
        # load data
        data = np.load(f'{path}/mist_data.npz')
        
        self.eeg = data['eeg'] # (36, 2, 7, 7680)
        self.fnirs = data['fnirs'] # (36, 2, 26, 367) 
        self.label = data['label'] # (36, 2)
        self.subjects = [i for i in range(self.eeg.shape[0])]
        # segmentation
        self.eeg, n_windows = self.window_slicing(self.eeg, start_point, window_len)
        self.fnirs, n_windows = self.window_slicing(self.fnirs, start_point, window_len)
        self.label = np.repeat(self.label, n_windows, 1)
        self.data_shape_eeg = list(self.eeg.shape[-2:])
        self.data_shape_fnirs = list(self.fnirs.shape[-2:])

        if transform_eeg:
            self.eeg = transform_eeg(self.eeg)
        if transform_fnirs:
            self.fnirs = transform_fnirs(self.fnirs)
    
    def __len__(self):
        return self.subjects

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.subjects):
            eeg_torch = torch.from_numpy(self.eeg[self.subjects[self.test_idx]]).float()
            fnirs_torch = torch.from_numpy(self.fnirs[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            train_subjects, val_subjects = self.train_val_split()
            eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in train_subjects])).float()
            fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            self.test_idx += 1
            if len(val_subjects) > 0:
                eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in val_subjects])).float()
                fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in val_subjects])).float()
                label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
                val_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, True)

                return train_loader, val_loader, test_loader
            return train_loader, None, test_loader
        else:
            raise StopIteration
    
    def create_dataloader(self, eeg, fnirs, label, shuffle=False):
        if self.data_mode == 0:
            return DataLoader(BimodalDataSet(eeg, fnirs, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 1:
            return DataLoader(CustomDataSet(eeg, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 2:
            return DataLoader(CustomDataSet(fnirs, label), self.batch_size, shuffle=shuffle)
    
    def window_slicing(self, arr, start_point, window_len):
        total_len = arr.shape[-1]
        
        start_idx = int(total_len * start_point / 60)
        window_length = int(total_len * window_len / 60)
        n_windows = (total_len - start_idx) // window_length
        
        out = np.stack([arr[:,:,:,start_idx + i * window_length : start_idx + (i + 1) * window_length] for i in range(n_windows)], 1)
        out = np.swapaxes(out, 0, 2)
        out = np.concatenate(out)
        out = np.swapaxes(out, 0, 1)
        return out, n_windows

    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]

class Stroop_DataModule():
    r'''
    Create stroop dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        data_mode (int): 0 - eeg + fnirs, 1 - only eeg, 2 - only fnirs
        label_type (int): 0 - arousal classification, 1 - valence classification
        ica (bool): load data_ica (default: True)
        start_point (int): start_point of the segmentation in seconds. (default: 60)
        window_len (int): window length in seconds for segmentation. (default: 60)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str, # 'D:\One_한양대학교\private object minsu\coding\data\\fNIRS-EEG_Stroop'
                 data_mode:int = 0,
                 start_point:int = 0,
                 window_len:int = 32,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform_eeg = None,
                 transform_fnirs = None,
                 ):
        super().__init__()

        self.data_mode = data_mode
        self.batch_size = batch_size
        self.num_val = num_val
        self.test_idx = 0
        
        # load data
        data = np.load(f'{path}/stroop.npz')
        self.fnirs = data['fnirs'][:,:,:,11:3211] # (20, 4, 40, 3711) 
        self.eeg = data['eeg'] # (20, 4, 29, 4096)
        self.label = data['label'] # (20, 4)
        self.subjects = [i for i in range(self.eeg.shape[0])]

        # segmentation
        self.eeg, n_windows = self.window_slicing(self.eeg, start_point, window_len, 32)
        self.fnirs, n_windows = self.window_slicing(self.fnirs, start_point, window_len, 32)
        self.label = np.repeat(self.label, n_windows, 1)
        self.data_shape_eeg = list(self.eeg.shape[-2:])
        self.data_shape_fnirs = list(self.fnirs.shape[-2:])

        if transform_eeg:
            self.eeg = transform_eeg(self.eeg)
        if transform_fnirs:
            self.fnirs = transform_fnirs(self.fnirs)
    
    def __len__(self):
        return self.subjects

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.subjects):
            eeg_torch = torch.from_numpy(self.eeg[self.subjects[self.test_idx]]).float()
            fnirs_torch = torch.from_numpy(self.fnirs[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            train_subjects, val_subjects = self.train_val_split()
            eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in train_subjects])).float()
            fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            self.test_idx += 1
            if len(val_subjects) > 0:
                eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in val_subjects])).float()
                fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in val_subjects])).float()
                label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
                val_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, True)

                return train_loader, val_loader, test_loader
            return train_loader, None, test_loader
        else:
            raise StopIteration
    
    def create_dataloader(self, eeg, fnirs, label, shuffle=False):
        if self.data_mode == 0:
            return DataLoader(BimodalDataSet(eeg, fnirs, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 1:
            return DataLoader(CustomDataSet(eeg, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 2:
            return DataLoader(CustomDataSet(fnirs, label), self.batch_size, shuffle=shuffle)
    
    def window_slicing(self, arr, start_point, window_len, total_sec):
        total_len = arr.shape[-1]
        
        start_idx = int(total_len * start_point / total_sec)
        window_length = int(total_len * window_len / total_sec)
        n_windows = (total_len - start_idx) // window_length
        
        out = np.stack([arr[:,:,:,start_idx + i * window_length : start_idx + (i + 1) * window_length] for i in range(n_windows)], 1)
        out = np.swapaxes(out, 0, 2)
        out = np.concatenate(out)
        out = np.swapaxes(out, 0, 1)
        return out, n_windows

    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]

class MIMA_DataModule():
    r'''
    Create MI,MA dataset for leave-one-subject-out cross-validation

    Args:
        path (str): path for the original data.
        data_mode (int): 0 - eeg + fnirs, 1 - only eeg, 2 - only fnirs
        label_type (int): 0 - MI classification, 1 - MA classification
        ica (bool): load data_ica (default: True)
        start_point (int): start_point of the segmentation in seconds. (default: 60)
        window_len (int): window length in seconds for segmentation. (default: 60)
        num_val (int): number of subjects for validation. (default: 3)
        batch_size (int): batch size of the dataloader. (default: 16)
        transform (function): transform function for the data (default: None)
    '''
    def __init__(self, 
                 path:str, # 'D:\One_한양대학교\private object minsu\coding\data\\fNIRS-EEG_Stroop'
                 data_mode:int = 0,
                 label_type:int = 0,
                 num_val:int = 3,
                 batch_size:int = 16,
                 transform_eeg = None,
                 transform_fnirs = None,
                 ):
        super().__init__()

        self.data_mode = data_mode
        self.batch_size = batch_size
        self.num_val = num_val
        self.test_idx = 0
        
        # load data
        tp = 'MI' if label_type == 0 else 'MA'
        data = np.load(f'{path}/{tp}.npz')
        self.fnirs = data['fnirs'] # (29, 60, 72, 100) 
        self.eeg = data['eeg'] # (29, 60, 30, 2000)
        self.label = data['label'] # (29, 60)
        self.subjects = [i for i in range(self.eeg.shape[0])]

        # segmentation
        self.data_shape_eeg = list(self.eeg.shape[-2:])
        self.data_shape_fnirs = list(self.fnirs.shape[-2:])

        if transform_eeg:
            self.eeg = transform_eeg(self.eeg)
        if transform_fnirs:
            self.fnirs = transform_fnirs(self.fnirs)
    
    def __len__(self):
        return self.subjects

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.test_idx < len(self.subjects):
            eeg_torch = torch.from_numpy(self.eeg[self.subjects[self.test_idx]]).float()
            fnirs_torch = torch.from_numpy(self.fnirs[self.subjects[self.test_idx]]).float()
            label_torch = torch.from_numpy(self.label[self.subjects[self.test_idx]]).long()
            test_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            train_subjects, val_subjects = self.train_val_split()
            eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in train_subjects])).float()
            fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in train_subjects])).float()
            label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in train_subjects])).long()
            train_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch)

            self.test_idx += 1
            if len(val_subjects) > 0:
                eeg_torch = torch.from_numpy(np.concatenate([self.eeg[i] for i in val_subjects])).float()
                fnirs_torch = torch.from_numpy(np.concatenate([self.fnirs[i] for i in val_subjects])).float()
                label_torch = torch.from_numpy(np.concatenate([self.label[i] for i in val_subjects])).long()
                val_loader = self.create_dataloader(eeg_torch, fnirs_torch, label_torch, True)

                return train_loader, val_loader, test_loader
            return train_loader, None, test_loader
        else:
            raise StopIteration
    
    def create_dataloader(self, eeg, fnirs, label, shuffle=False):
        if self.data_mode == 0:
            return DataLoader(BimodalDataSet(eeg, fnirs, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 1:
            return DataLoader(CustomDataSet(eeg, label), self.batch_size, shuffle=shuffle)
        elif self.data_mode == 2:
            return DataLoader(CustomDataSet(fnirs, label), self.batch_size, shuffle=shuffle)

    def train_val_split(self):
        subj = [i for i in self.subjects if i != self.subjects[self.test_idx]]
        random.shuffle(subj)
        return subj[self.num_val:], subj[:self.num_val]

if __name__ == '__main__':
    # emotion_dataset = Emotion_DataModule('D:/One_한양대학교/private object minsu/coding/data/brain_2025',
    #                                      label_type=0,
    #                                      ica=True,
    #                                      start_point=60,
    #                                      window_len=30,
    #                                      num_val=3,
    #                                      batch_size=16,
    #                                      transform_eeg=None,
    #                                      transform_fnirs=None)
    
    emotion_dataset = Stroop_DataModule('D:\One_한양대학교\private object minsu\coding\data\\fNIRS-EEG_Stroop',
                                        data_mode=0,
                                        start_point=0,
                                        window_len=32,
                                        num_val=0,
                                        batch_size=16,
                                        transform_eeg=None,
                                        transform_fnirs=None)
    eeg = emotion_dataset.eeg
    fnirs = emotion_dataset.fnirs
    label = emotion_dataset.label
    print(eeg.shape, fnirs.shape, label.shape)