import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep4Net(nn.Module):
    def __init__(self, input_size = [3, 7680], n_classes = 1, pool_mode = "mean", cls = True):
        super(Deep4Net, self).__init__()
        self.cls = cls
        conv_len = 13
        num_filters = 25
        pool_len = 2
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        # Conv Pool Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, num_filters, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters),
            nn.ELU(),
            nn.Conv2d(num_filters, num_filters, (input_size[0], 1), groups=num_filters, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len))
        )
        # Conv Pool Block 1
        self.block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters*2),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters*4),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*8, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters*8),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len))
        )
        self.dim_feat = num_filters*8*(input_size[1]//pool_len//pool_len//pool_len//pool_len)
        
        # Classification Layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.dim_feat, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()

        )
        
    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        
        x = torch.flatten(x,1)

        if not self.cls:
            return x
        x = self.fc1(x)
        return x

# Example usage:
# model = EEGNet()
# print(model)
