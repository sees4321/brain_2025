import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EF_net(nn.Module):

    def __init__(self, n_class, dropout=0.5):
        super().__init__()
        self.eegblock1 = nn.Sequential(
            nn.Conv2d(1,32, (1,9)),
            nn.ReLU(),
            nn.Conv2d(32,32, (1,9)),
            nn.ReLU(),
            nn.Conv2d(32,32, (1,9)),
            nn.ReLU(),
            nn.MaxPool2d((1,9)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32)
        )
        self.eegblock2 = nn.Sequential(
            nn.Conv2d(32,64, (3,5)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,5)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,5)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1,512)),
            # nn.MaxPool2d((1,7)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64)
        )
        self.eegblock3 = nn.Sequential(
            nn.Linear(512*64,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),
        )

        self.fnirsblock1 = nn.Sequential(
            nn.Conv2d(1,32, (1,5)),
            nn.ReLU(),
            nn.Conv2d(32,32, (1,5)),
            nn.ReLU(),
            nn.MaxPool2d((1,5)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32)
        )
        self.fnirsblock2 = nn.Sequential(
            nn.Conv2d(32,64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,3)),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d((1,512)),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64)
        )
        self.fnirsblock3 = nn.Sequential(
            nn.AdaptiveMaxPool1d(256),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.fusionblock1 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fusionblock2 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,n_class),
            nn.Sigmoid() if n_class == 1 else nn.Softmax()
        )

    def forward(self, eeg, fnirs):
        # eeg: (channel, time)
        # fnirs: (channel, time)
        eeg = eeg.unsqueeze(1)
        fnirs = fnirs.unsqueeze(1)
        eeg = self.eegblock1(eeg)
        eeg = self.eegblock2(eeg)
        eeg = torch.flatten(eeg, 1)
        eeg = self.eegblock3(eeg)

        fnirs = self.fnirsblock1(fnirs)
        fnirs = self.fnirsblock2(fnirs)
        fnirs = torch.flatten(fnirs, 1)
        fnirs = self.fnirsblock3(fnirs)

        fusion = torch.concat((eeg,fnirs),1)
        fusion = self.fusionblock1(fusion)
        fusion = fusion / (torch.norm(fusion, p=2, dim=1, keepdim=True) + 1e-12)
        fusion = self.fusionblock2(fusion)
        return fusion

if __name__ == "__main__":
    x = torch.randn((8,7,7680))
    y = torch.randn((8,26,371))
    model = EF_net(1)
    out = model(x,y)
    print(x.shape, y.shape, out)