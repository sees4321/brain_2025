import torch
from torch import nn
import torch.nn.functional as F

# Cai, Siqi, et al. "EEG-based auditory attention detection via frequency and channel neural attention." IEEE Transactions on Human-Machine Systems 52.2 (2021): 256-266.

class mySE(nn.Module):
    def __init__(self, se_weight_num, se_type, se_fcn_squeeze, conv_num):
        super(mySE, self).__init__()
        se_fcn_num_dict = {'avg': se_weight_num, 'max': se_weight_num, 'mix': se_weight_num * 2}
        se_fcn_num = se_fcn_num_dict.get(se_type)

        self.se_conv = nn.Sequential(
            nn.Conv3d(1, 1, (1, conv_num, 1), stride=(1, 1, 1)),
            nn.ELU(),
        )

        self.se_fcn = nn.Sequential(
            nn.Linear(se_fcn_num, se_fcn_squeeze),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(se_fcn_squeeze, se_weight_num),
            nn.Tanh(),
        )

    def forward(self, se_data, se_type):
        se_weight = se_data
        se_weight = self.se_conv(se_weight.unsqueeze(1))#.unsqueeze(0))
        se_weight = se_weight.squeeze(0)

        avg_data = torch.mean(se_weight, axis=-1)
        max_data = torch.max(se_weight, axis=-1)[0]

        mix_data = torch.cat((avg_data, max_data), dim=1)
        data_dict = {'avg': avg_data, 'max': max_data, 'mix': mix_data}
        se_weight = data_dict.get(se_type)
        se_weight = torch.mean(se_weight, axis=0).squeeze(0).transpose(0, 1)

        se_weight = self.se_fcn(se_weight)

        # mask
        se_weight = (se_weight - torch.min(se_weight)) / (torch.max(se_weight) - torch.min(se_weight))

        # weighted
        output = ((se_data.transpose(1, 3)) * se_weight).transpose(1, 3)

        return output


# the main model
class AttnCNN(nn.Module):
    def __init__(self, input_size, eeg_band=8, is_se_band=True, is_se_channel=True, se_band_type='max', se_channel_type = 'avg', window_time=2):
        super(AttnCNN, self).__init__()

        cnn_ken_num = 32
        fcn_input_num = cnn_ken_num
        self.is_se_band = is_se_band
        self.is_se_channel = is_se_channel
        self.eeg_band = eeg_band
        self.eeg_channel_new = input_size[0]
        self.window_len = input_size[1]
        self.se_band_type = se_band_type
        self.se_channel_type = se_channel_type
        self.window_time = window_time

        self.se_band = mySE(self.eeg_band, self.se_band_type, 5, self.eeg_channel_new)
        self.se_channel = mySE(self.eeg_channel_new, self.se_channel_type, 8, self.eeg_band)
        self.cnn_conv_eeg = nn.Sequential(
            nn.Conv2d(self.eeg_band, cnn_ken_num, (self.eeg_channel_new, 9), stride=(self.eeg_channel_new, 1)),
            nn.ELU(),
            nn.AdaptiveMaxPool2d((1, 1 * self.window_time)),
        )

        self.cnn_fcn = nn.Sequential(
            nn.Linear(fcn_input_num * self.window_time, fcn_input_num),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(fcn_input_num, 1),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        # split the wav and eeg data
        eeg = x

        # frequency attention
        if self.is_se_band:
            # eeg = eeg.view(self.eeg_band, self.eeg_channel_new, self.window_len)
            eeg = self.se_band(eeg, self.se_band_type)

        # channel attention
        if self.is_se_channel:
            # eeg = eeg.view(self.eeg_band, self.eeg_channel_new, self.window_len).transpose(0, 1)
            eeg = eeg.transpose(1,2)
            eeg = self.se_channel(eeg, self.se_channel_type).transpose(1, 2)

        # normalization
        # eeg = eeg.view(1, self.eeg_band, self.eeg_channel_new, self.window_len)

        # convolution
        y =  eeg
        y = self.cnn_conv_eeg(y)
        y = torch.flatten(y,1)

        # classification
        output = self.cnn_fcn(y)

        return output
    