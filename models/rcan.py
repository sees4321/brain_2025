import torch
from torch import nn
import torch.nn.functional as F

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
    
## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size, padding=kernel_size//2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN) - DE Features (5 band * 5 channel * 5 temporal)
class RCAN(nn.Module):
    def __init__(self, n_feats=16, kernel_size=3):
        super(RCAN, self).__init__()
        n_resgroups = 3
        n_resblocks = 3
        reduction = 4
        act = nn.ReLU(True)
        conv = nn.Conv2d
        
        def head():
            return nn.Sequential(
                nn.Conv2d(5,n_feats,1),
                nn.BatchNorm2d(n_feats),
                nn.ELU(),
            )
        self.head1 = head()
        self.head2 = head()
        self.head3 = head()

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size,padding=kernel_size//2))
        
        self.body1 = nn.Sequential(*modules_body)
        self.body2 = nn.Sequential(*modules_body)
        self.body3 = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.Linear(n_feats*5*5*3,64),
            nn.ELU(),
            nn.Linear(64,1)
        )

    def forward(self, x): 
        # B 5 5 5
        y1 = self.head1(x)
        y2 = self.head2(x.transpose(1,2))
        y3 = self.head3(x.transpose(1,3))
        # B 16 5 5
        y1 = self.body1(y1) + y1
        y2 = self.body2(y2) + y2
        y3 = self.body3(y3) + y3
        # B 16 5 5
        y = torch.concat((y1,y2,y3),1)
        y = torch.flatten(y,1)
        # B 16*5*5*5
        y = self.tail(y)
        return y
   