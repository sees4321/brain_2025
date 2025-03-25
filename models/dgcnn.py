import torch
from torch import nn
import torch.nn.functional as F

# Song, Tengfei, et al. "EEG emotion recognition using dynamical graph convolutional neural networks." IEEE Transactions on Affective Computing 11.3 (2018): 532-541.

def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

class GraphConvolution(nn.Module):
    def __init__(self, num_in, num_out, bias=False):
        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear_wInit(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_wInit, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
    
class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim, num_out))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, 
                 in_features: int = 5,
                 num_electrodes: int = 11, 
                 num_layers: int = 8,
                 hid: int = 32, 
                 n_classes=1):
        #in_features: number of input features
        #num_electrodes: number of electrodes
        #num_layers: batch size
        #hid: number of hid

        super(DGCNN, self).__init__()
        self.layer1 = Chebynet(in_features, num_layers, hid)
        self.BN1 = nn.BatchNorm1d(in_features)
        self.fc1 = Linear_wInit(num_electrodes * hid, 64)
        self.fc2 = Linear_wInit(64, n_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.xavier_normal_(self.A)

    def forward(self, x, return_feat=False):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        feat = self.layer1(x, L)
        feat = feat.reshape(x.shape[0], -1)
        y = F.relu(self.fc1(feat))
        y = self.fc2(y)

        return (feat, y) if return_feat else y
