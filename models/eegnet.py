import torch
from torch import nn
import torch.nn.functional as F
    
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    def __init__(self, input_size, sampling_rate, n_classes, F1=8, D=2, F2=16, final_conv_length="auto", pool_mode="mean", drop_prob=0.25):
        super(EEGNet,self).__init__()
        self.in_chans = int(input_size[0])
        self.input_window_samples = int(input_size[1])
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.block1_kernel_length = sampling_rate // 2  # half of the sampling rate
        self.block1_pooling_length = sampling_rate // 32  # 32 Hz로 downsampling
        self.block2_kernel_length = 16  # half of the sampling rate (32 Hz)
        self.block2_pooling_length = 8  # 4 Hz로 downsampling -> dimension reduction
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.drop_prob = drop_prob
        
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.block1 = nn.Sequential(
            # Conv2D
            nn.Conv2d(1, self.F1, kernel_size=(1, self.block1_kernel_length),
                      stride=1, bias=False, padding=(0, self.block1_kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # DepthwiseConv2D
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block1_pooling_length), stride=(1, self.block1_pooling_length)),
            nn.Dropout(p=self.drop_prob),
        )

        self.block2 = nn.Sequential(
            # SeparableConv2D
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size=(1, self.block2_kernel_length),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, self.block2_kernel_length // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, 1), stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block2_pooling_length), stride=(1, self.block2_pooling_length)),
            nn.Dropout(p=self.drop_prob)
        )

        if self.final_conv_length == "auto":
            n_out_time = int(self.input_window_samples // self.block1_pooling_length // self.block2_pooling_length)
            self.final_conv_length = n_out_time
        self.last_inchan = self.F2 * self.final_conv_length
        
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Linear(self.last_inchan, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax()
        )

        # weight initialization
        # for module in [self.block1, self.block2, self.classifier]:
        #     module.apply(_glorot_weight_zero_bias)
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, return_feat=False):
        x = x.unsqueeze(1)
        out = self.block1(x)
        out = self.block2(out)
        feat = torch.flatten(out,1)
        y = self.classifer(feat)
        return (feat, y) if return_feat else y