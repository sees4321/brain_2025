import torch
from torch import nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

from scipy.signal import hilbert

def make_input(data:np.ndarray, only_eeg:bool = False): # 32 * 8 * 8 * 7500 => 32 * 8 * 8 * 30 * 250
    ### data shape: 32 * 7680 (32 electrode channels * 60 s samples of 128 Hz)
    ### 26 * 371
    a,b,c,d = data.shape
    dat1 = data.reshape(a,b,c,30,d//30)
    if only_eeg: 
        return dat1
    dat2 = np.imag(hilbert(data))
    dat2 = dat2.reshape(a,b,c,30,d//30)
    return np.concatenate((dat1, dat2),axis=3)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # Query, Key, Value 변환
        query = self.query_layer(query)  # (Batch, Time, Embed)
        key = self.key_layer(key)        # (Batch, Time, Embed)
        value = self.value_layer(value)  # (Batch, Time, Embed)

        # Attention 점수 계산
        attn_scores = torch.matmul(query, key.transpose(-2, -1))  # (Batch, Time, Time)
        attn_weights = self.softmax(attn_scores)  # Softmax로 Attention Weight 계산

        # Weighted sum
        output = torch.matmul(attn_weights, value)  # (Batch, Time, Embed)
        return output, attn_weights

class EEGNet2(nn.Module):
    def __init__(self, input_size=[7,7680], sampling_rate=128, F1=8, D=2, F2=16, final_conv_length="auto", pool_mode="mean", drop_prob=0.25, emb_dim=64):
        super(EEGNet2,self).__init__()
        self.in_chans = int(input_size[0])
        self.input_window_samples = int(input_size[1])
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

        if self.final_conv_length == "auto":
            n_out_time = int(self.input_window_samples // self.block1_pooling_length // self.block2_pooling_length)
            self.final_conv_length = n_out_time
        self.last_inchan = self.F2 * self.final_conv_length
        self.dim_feat = self.last_inchan

        self.block2 = nn.Sequential(
            # SeparableConv2D
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size=(1, self.block2_kernel_length),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, self.block2_kernel_length // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, 1), stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block2_pooling_length), stride=(1, self.block2_pooling_length)),
            nn.Dropout(p=self.drop_prob),
        )
        self.fc = nn.Linear(self.dim_feat, emb_dim)
        self.dim_feat = emb_dim

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        feat = torch.flatten(out,1)
        return self.fc(feat)

class EEGNet_fNIRS(nn.Module):
    def __init__(self, input_shape=[26,371], pool_mode="mean", drop_prob=0.25, cls = False):
        super(EEGNet_fNIRS,self).__init__()

        self.pool_mode = pool_mode
        self.drop_prob = drop_prob

        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.block1_kernel_length = 3
        self.block2_kernel_length = 3
        self.block2_pooling_length = 2
        self.in_chans = 26
        self.n_classes = 1
        self.cls = cls
        
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.block = nn.Sequential(
            # Conv2D
            nn.Conv2d(1, self.F1, kernel_size=(1, self.block1_kernel_length),
                      stride=1, bias=False, padding=(0, self.block1_kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # DepthwiseConv2D
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            # pool_class(kernel_size=(1, self.block1_pooling_length), stride=(1, self.block1_pooling_length)),
            nn.Dropout(p=self.drop_prob),

            # SeparableConv2D
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, kernel_size=(1, self.block2_kernel_length),
                      stride=1, bias=False, groups=self.F1 * self.D, padding=(0, self.block2_kernel_length // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, 1), stride=1, bias=False, padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block2_pooling_length), stride=(1, self.block2_pooling_length)),
            nn.Dropout(p=self.drop_prob)
        )
        self.last_inchan = self.F2 * (input_shape[1]//2)
        self.dim_feat = self.last_inchan
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Linear(self.dim_feat, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax()
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block(x)
        x = torch.flatten(x, 1)
        if self.cls:
            x = self.classifer(x)
        return x

class EEGNet_fNIRS2(nn.Module):
    def __init__(self, pool_mode="mean", drop_prob=0.25, cls = False):
        super(EEGNet_fNIRS2,self).__init__()

        self.pool_mode = pool_mode
        self.drop_prob = drop_prob

        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.block1_kernel_length = 3
        self.block2_pooling_length = 2
        self.in_chans = 26
        self.n_classes = 1
        self.cls = cls
        
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.block = nn.Sequential(
            # Conv2D
            nn.Conv2d(1, self.F1, kernel_size=(1, self.block1_kernel_length),
                      stride=1, bias=False, padding=(0, self.block1_kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),

            # SeparableConv2D
            nn.Conv2d(self.F1, self.F1 * self.D, kernel_size=(self.in_chans,1),
                      stride=1, bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block2_pooling_length), stride=(1, self.block2_pooling_length)),
            nn.Dropout(p=self.drop_prob)
        )
        self.last_inchan = self.F2 * 185
        self.dim_feat = self.last_inchan
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Linear(self.dim_feat, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax()
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block(x)
        x = torch.flatten(x, 1)
        if self.cls:
            x = self.classifer(x)
        return x

class resBlock(nn.Module):
    def __init__(self, inn, hid, k):
        super(resBlock,self).__init__()
        # k2 = k//2+1
        k2 = k
        self.conv = nn.Sequential(
            # nn.Conv2d(inn,inn,1),
            nn.Conv2d(inn,inn,(1,k2),padding=(0,k2//2)),
            nn.BatchNorm2d(inn),
            nn.ELU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(inn,hid,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(hid),
            nn.ELU(),
            nn.Conv2d(hid,inn,1),
        )
        self.relu = nn.ELU()

    def forward(self,x):
        out = self.conv(x)
        out = self.layer(out) + out
        out = self.relu(out)
        return out
    
class HiRENet2(nn.Module):
    def __init__(self, num_chan=7, conv_chan=16, num_classes=1, drop_prob=0.5, withhil = True, cls = False):
        super(HiRENet2, self).__init__()

        self.withhil = withhil
        self.num_chan = num_chan
        self.conv_chan = conv_chan
        self.n_classes = num_classes
        self.conv_len = 13
        self.cls = cls

        self.layerx = resBlock(self.num_chan, self.conv_chan, self.conv_len)
        if self.withhil:
            self.layery = resBlock(self.num_chan, self.conv_chan, self.conv_len)
            self.num_chan = self.num_chan*2

        self.layer4 = nn.Sequential( 
                nn.Conv2d(self.num_chan, self.conv_chan, kernel_size=(1, self.conv_len), padding=(0, self.conv_len//2)),
                nn.BatchNorm2d(self.conv_chan, momentum=.1, affine=True),
                nn.ELU(),
                nn.Conv2d(self.conv_chan, self.conv_chan*2, kernel_size=(30, 1)),
                # nn.Conv2d(self.num_chan, self.conv_chan*2, kernel_size=(30, 1)),
                nn.BatchNorm2d(self.conv_chan*2, momentum=.1, affine=True),
                nn.ELU(),
            )    
        self.avgdrp = nn.Sequential(
            nn.AvgPool2d((1,self.conv_len),(1,2)),
            # nn.MaxPool2d((1,self.conv_len),(1,2)),
            nn.Dropout2d(drop_prob)
        )
        final_dim = 128 - self.conv_len//2
        self.dim_feat = self.conv_chan * 2 * final_dim
        self.classifer = nn.Sequential(
            nn.Conv2d(self.conv_chan * 2, self.n_classes, kernel_size=(1, final_dim), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            # nn.Linear(self.dim_feat, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.layerx(x[:,:,:30,:])
        if self.withhil:
            outy = self.layery(x[:,:,30:,:])
            out = torch.cat((out,outy),dim=1)
        out = self.layer4(out)
        out = self.avgdrp(out)
        if self.cls:
            out = self.classifer(out)
        else:
            out = torch.flatten(out, 1)
        return out

class EEGNet_fNIRS3(nn.Module):
    def __init__(self, pool_mode="mean", drop_prob=0.25, cls = False, emb_dim = 64):
        super(EEGNet_fNIRS3,self).__init__()

        self.pool_mode = pool_mode
        self.drop_prob = drop_prob

        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.block1_kernel_length = 3
        self.block2_pooling_length = 2
        self.in_chans = 26
        self.n_classes = 1
        self.cls = cls
        
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.block = nn.Sequential(
            # Conv2D
            nn.Conv2d(1, self.F1, kernel_size=(1, self.block1_kernel_length),
                      stride=1, bias=False, padding=(0, self.block1_kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),

            # SeparableConv2D
            nn.Conv2d(self.F1, self.F1 * self.D, kernel_size=(self.in_chans,1),
                      stride=1, bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, self.block2_pooling_length), stride=(1, self.block2_pooling_length)),
            nn.Dropout(p=self.drop_prob)
        )
        self.last_inchan = self.F2 * 185
        self.dim_feat = self.last_inchan
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Linear(self.dim_feat, self.n_classes),
            # nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax()
        )
        self.emb = nn.Linear(self.dim_feat, emb_dim)
        self.dim_feat = emb_dim

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='elu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block(x)
        x = torch.flatten(x, 1)
        if self.cls:
            x = self.classifer(x)
        else:
            x = self.emb(x)
        return x

class resBlock3(nn.Module):
    def __init__(self, inn, hid, k):
        super(resBlock3,self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(inn,inn,1),
            nn.Conv2d(inn,inn,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(inn),
            nn.ELU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(inn,hid,(1,k),padding=(0,k//2)),
            nn.BatchNorm2d(hid),
            nn.ELU(),
            nn.Conv2d(hid,inn,1),
        )
        self.relu = nn.ELU()

    def forward(self,x):
        x = self.conv(x)
        x = self.layer(x) + x
        x = self.relu(x)
        return x
    
class HiRENet3(nn.Module):
    def __init__(self, num_chan=7, conv_chan=32, num_classes=1, withhil = True, cls = False, emb_dim = 64):
        super(HiRENet3, self).__init__()

        self.withhil = withhil
        self.num_chan = num_chan
        self.conv_chan = conv_chan
        self.n_classes = num_classes
        self.conv_len = 13
        self.conv_len2 = 13
        self.cls = cls

        self.layerx = resBlock3(self.num_chan, self.conv_chan, self.conv_len)
        if self.withhil:
            self.layery = resBlock3(self.num_chan, self.conv_chan, self.conv_len)
            self.num_chan = self.num_chan*2

        self.layer4 = nn.Sequential( 
                nn.Conv2d(self.num_chan, self.conv_chan, kernel_size=(1, self.conv_len2), padding=(0, self.conv_len2//2)),
                nn.BatchNorm2d(self.conv_chan, momentum=.1, affine=True),
                nn.ELU(),
                nn.Conv2d(self.conv_chan, self.conv_chan*2, kernel_size=(30, 1)),
                # nn.Conv2d(self.num_chan, self.conv_chan*2, kernel_size=(30, 1)),
                nn.BatchNorm2d(self.conv_chan*2, momentum=.1, affine=True),
                nn.ELU(),
            )    
        self.avgdrp = nn.Sequential(
            nn.AvgPool2d((1,self.conv_len2),(1,2)),
            # nn.Dropout(0.25),
            nn.Dropout2d(0.5)
        )
        final_dim = 128 - self.conv_len2//2
        self.dim_feat = self.conv_chan * 2 * final_dim
        self.emb = nn.Sequential(
            # nn.Conv2d(self.conv_chan * 2, self.n_classes, kernel_size=(1, final_dim), bias=True),
            nn.Linear(self.dim_feat, emb_dim),
        )
        self.dim_feat = emb_dim
        self.classifer = nn.Sequential(
            nn.Conv2d(self.conv_chan * 2, self.n_classes, kernel_size=(1, final_dim), bias=True),
            # nn.Linear(self.dim_feat, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.layerx(x[:,:,:30,:])
        if self.withhil:
            outy = self.layery(x[:,:,30:,:])
            out = torch.cat((out,outy),dim=1)
        out = self.layer4(out)
        out = self.avgdrp(out)
        out = self.classifer(out) if self.cls else self.emb(torch.flatten(out, 1))
        return out

class FNIRSSubNet(nn.Module):
    def __init__(self, input_channels=26, temporal_filters=32, temporal_length=7, spatial_filters=32, emb_dim=64, dropout=0.1, num_samples=371, cls=False):
        super(FNIRSSubNet, self).__init__()
        self.conv1 = nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_length))
        self.conv2 = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(input_channels, 1))
        self.batch_norm = nn.BatchNorm2d(spatial_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(spatial_filters * (num_samples - temporal_length + 1), emb_dim)  # Adjust based on input dimensions
        self.dim_feat = emb_dim
        self.sig = nn.Sigmoid()
        self.cls = cls

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return self.sig(x) if self.cls else x
    
class Bimodal_model(nn.Module):
    def __init__(self, eeg_model, fnirs_model, n_classes, attn=False):
        super(Bimodal_model,self).__init__()

        self.eeg_subnet = eeg_model
        self.fnirs_subnet = fnirs_model
        self.n_classes = n_classes
        self.attn = attn
        self.hid = self.eeg_subnet.dim_feat + self.fnirs_subnet.dim_feat
        
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Linear(self.hid, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax(dim=1)
        )
        self.apply_initialization()

    def forward(self, eeg, fnirs):
        x = self.eeg_subnet(eeg)
        y = self.fnirs_subnet(fnirs)
        out = torch.concat([x, y], 1)
        out = self.classifer(out)
        return out
    
    def apply_initialization(self):
    # Apply Xavier initialization to EEG and fNIRS subnets
        for module in self.eeg_subnet.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        for module in self.fnirs_subnet.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        # Apply He initialization to GRU and fully connected layers
        for module in self.modules():
            if module not in (self.eeg_subnet, self.fnirs_subnet):  # Exclude EEG and fNIRS subnets
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        init.zeros_(module.bias)
    
class Bimodal_attn_model(nn.Module):
    def __init__(self, eeg_model, fnirs_model, n_classes):
        super(Bimodal_attn_model,self).__init__()

        self.eeg_subnet = eeg_model
        self.fnirs_subnet = fnirs_model
        self.n_classes = n_classes
        self.hid = self.eeg_subnet.dim_feat + self.fnirs_subnet.dim_feat
        self.att_eeg_to_fnirs = CrossAttention(self.eeg_subnet.dim_feat)
        self.att_fnirs_to_eeg = CrossAttention(self.fnirs_subnet.dim_feat)
        
        self.classifer = nn.Sequential(
            # nn.Conv2d(self.F2, self.n_classes, kernel_size=(1, self.final_conv_length), bias=True),
            # DepthwiseConv2D에서 채널 전체에 대해서 진행했으므로 n_out_spatial = 1
            # nn.LogSoftmax(dim=1),
            nn.Dropout(0.1),
            nn.Linear(self.hid, self.n_classes),
            nn.Sigmoid() if self.n_classes == 1 else nn.LogSoftmax(dim=1)
        )
        self.apply_initialization()

    def forward(self, eeg, fnirs):
        eeg_features = self.eeg_subnet(eeg)
        fnirs_features = self.fnirs_subnet(fnirs)
                # Cross Attention: EEG → fNIRS
        eeg_to_fnirs, _ = self.att_eeg_to_fnirs(
            query=fnirs_features.unsqueeze(1),  # Shape: [Batch, 1, Feature]
            key=eeg_features.unsqueeze(1),     # Shape: [Batch, 1, Feature]
            value=eeg_features.unsqueeze(1)    # Shape: [Batch, 1, Feature]
        )
        eeg_to_fnirs = eeg_to_fnirs.squeeze(1)  # Shape: [Batch, Feature]

        # Cross Attention: fNIRS → EEG
        fnirs_to_eeg, _ = self.att_fnirs_to_eeg(
            query=eeg_features.unsqueeze(1),   # Shape: [Batch, 1, Feature]
            key=fnirs_features.unsqueeze(1),  # Shape: [Batch, 1, Feature]
            value=fnirs_features.unsqueeze(1) # Shape: [Batch, 1, Feature]
        )
        fnirs_to_eeg = fnirs_to_eeg.squeeze(1)  # Shape: [Batch, Feature]
        combined_features = torch.cat((eeg_to_fnirs, fnirs_to_eeg), dim=1)  # Shape: [Batch, Feature * 2]
        return self.classifer(combined_features)
    
    def apply_initialization(self):
    # Apply Xavier initialization to EEG and fNIRS subnets
        for module in self.eeg_subnet.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        for module in self.fnirs_subnet.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        # Apply He initialization to GRU and fully connected layers
        for module in self.modules():
            if module not in (self.eeg_subnet, self.fnirs_subnet):  # Exclude EEG and fNIRS subnets
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        init.zeros_(module.bias)