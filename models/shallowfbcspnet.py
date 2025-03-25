import torch
from torch import nn
import torch.nn.functional as F

class ShallowFBCSPNet(nn.Module):
    def __init__(self, input_size, sampling_rate, n_classes=1, pool_mode="mean", actv_mode="elu",
                 batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5,
                 with_hil=True, batch_norm2=True):
        super(ShallowFBCSPNet, self).__init__()

        self.in_chans = int(input_size[0])
        self.input_time_length = int(input_size[1])
        self.filter_time_length = round(25 * sampling_rate / 250)
        self.pool_time_length = round(75 * sampling_rate / 250)
        self.pool_time_stride = round(15 * sampling_rate / 250)
        self.with_hil = with_hil

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        actv_func = dict(relu=nn.ReLU(), elu=nn.ELU(), prelu=nn.PReLU(), lrelu=nn.LeakyReLU())[actv_mode]
        self.n_filters_conv = 40

        self.TSconv1 = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, self.filter_time_length),
                        stride=1, bias=not batch_norm, padding=(0, self.filter_time_length//2)),
        )
        self.TSconv11 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(self.in_chans, 1), bias=not batch_norm),
            nn.BatchNorm2d(40, momentum=batch_norm_alpha, affine=True),
        )
        if batch_norm2:
            self.TSconv1.add_module(
                "bnorm",nn.BatchNorm2d(40, momentum=batch_norm_alpha, affine=True),
            )
            self.TSconv1.add_module(
                "elu",actv_func,
            )
            self.TSconv11.add_module(
                "elu",actv_func,
            )
        self.pooling1 = pool_class(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride))
        self.drop1 = nn.Dropout(p=drop_prob)
        self.final_conv_length = (self.input_time_length - self.pool_time_length + 2*0) // self.pool_time_stride + 1
        self.classfier = nn.Sequential(
            # nn.Conv2d(self.n_filters_conv, n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            nn.Linear(self.n_filters_conv  * self.final_conv_length * n_classes, n_classes, bias=True),
            # nn.LogSoftmax(dim=1),
            nn.Sigmoid()
        )
        
        for module in [self.TSconv1, self.TSconv11, self.pooling1, self.drop1, self.classfier]:
            module = module.cuda()
    
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.TSconv1(x) #
        y = self.TSconv11(y) 
        y = self.pooling1(y)
        y = self.drop1(y)
        y = torch.flatten(y,1)
        y = self.classfier(y)
        y = torch.squeeze(y)
        return y
    
