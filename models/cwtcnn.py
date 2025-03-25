import torch
import torch.nn as nn

class CWTCNN(nn.Module):
    def __init__(self, input_window=1280,sampling_rate=128, num_class=1) -> None:
        super().__init__()
        self.in_win = input_window
        self.block1_kernel_length = sampling_rate // 2  # half of the sampling rate
        self.block1_pooling_length = sampling_rate // 32  # 32 Hz로 downsampling
        self.block2_kernel_length = 16  # half of the sampling rate (32 Hz)
        self.block2_pooling_length = 8  # 4 Hz로 downsampling -> dimension reduction

        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,(1,self.block1_kernel_length),padding=(0,self.block1_kernel_length//2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,32,(64,1),(64,1),groups=8),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,64,(11,1),groups=8),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d((1,self.block1_pooling_length),(1,self.block1_pooling_length)),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,64,(1,self.block2_kernel_length),padding=(0,self.block2_kernel_length//2)),
            nn.Conv2d(64,128,1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d((1,self.block2_pooling_length),(1,self.block2_pooling_length)),
            nn.Dropout2d(0.25),
        )
        
        self.last_inchan = 128* int(self.in_win//self.block1_pooling_length//self.block2_pooling_length)
        self.classifier = nn.Sequential(
            nn.Linear(self.last_inchan,256),
            nn.GELU(),
            nn.Linear(256,num_class),
        )
    
    def forward(self, x, return_feat=False): # B 704 1280
        x = self.block1(x)
        x = self.block2(x)
        feat = torch.flatten(x,1)
        x = self.classifier(feat)
        # x = torch.squeeze(x)
        return (feat, x) if return_feat else x