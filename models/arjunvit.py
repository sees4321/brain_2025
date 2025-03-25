from typing import Tuple
import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, in_channels: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hid_channels, in_channels),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 hid_channels: int,
                 heads: int = 8,
                 head_channels: int = 64,
                 dropout: float = 0.):
        super(Attention, self).__init__()
        inner_channels = head_channels * heads
        project_out = not (heads == 1 and head_channels == hid_channels)

        self.heads = heads
        self.scale = head_channels**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(hid_channels, inner_channels * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_channels, hid_channels),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 hid_channels: int,
                 depth: int,
                 heads: int,
                 head_channels: int,
                 mlp_channels: int,
                 dropout: float = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        hid_channels,
                        Attention(hid_channels,
                                  heads=heads,
                                  head_channels=head_channels,
                                  dropout=dropout)),
                    PreNorm(
                        hid_channels,
                        FeedForward(hid_channels, mlp_channels,
                                    dropout=dropout))
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ArjunViT(nn.Module):
    r'''
    Arjun et al. employ a variation of the Transformer, the Vision Transformer to process EEG signals for emotion recognition. For more details, please refer to the following information. 
    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.
    - Paper: Arjun A, Rajpoot A S, Panicker M R. Introducing attention mechanism for eeg signals: Emotion recognition with vision transformers[C]//2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021: 5723-5726.
    - URL: https://ieeexplore.ieee.org/abstract/document/9629837
    Below is a recommended suite for use in emotion recognition tasks:
    .. code-block:: python
        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.MeanStdNormalize(),
                        transforms.To2d()
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = ArjunViT(chunk_size=128,
                         t_patch_size=50,
                         num_electrodes=32,
                         num_classes=2)
    Args:
       num_electrodes (int): The number of electrodes. (defualt: :obj:`32`)
        chunk_size (int): Number of data points included in each EEG chunk. (defualt: :obj:`128`)
        t_patch_size (int): The size of each input patch at the temporal (chunk size) dimension. (defualt: :obj:`32`)
        patch_size (tuple): The size (resolution) of each input patch. (defualt: :obj:`(3, 3)`)
        hid_channels (int): The feature dimension of embeded patch. (defualt: :obj:`32`)
        depth (int): The number of attention layers for each transformer block. (defualt: :obj:`3`)
        heads (int): The number of attention heads for each attention layer. (defualt: :obj:`4`)
        head_channels (int): The dimension of each attention head for each attention layer. (defualt: :obj:`8`)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (defualt: :obj:`64`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
        embed_dropout (float): Probability of an element to be zeroed in the dropout layers of the embedding layers. (defualt: :obj:`0.0`)
        dropout (float): Probability of an element to be zeroed in the dropout layers of the transformer layers. (defualt: :obj:`0.0`)
        pool_func (str): The pool function before the classifier, optionally including :obj:`cls` and :obj:`mean`, where :obj:`cls` represents selecting classification-related token and :obj:`mean` represents the average pooling. (defualt: :obj:`cls`)
    '''
    def __init__(self,
                 num_electrodes: int = 32,
                 chunk_size: int = 128,
                 t_patch_size: int = 32,
                 hid_channels: int = 32,
                 depth: int = 3,
                 heads: int = 4,
                 head_channels: int = 64,
                 mlp_channels: int = 64,
                 num_classes: int = 2,
                 embed_dropout: float = 0.,
                 dropout: float = 0.,
                 pool_func: str = 'cls'):
        super(ArjunViT, self).__init__()
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.t_patch_size = t_patch_size
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.head_channels = head_channels
        self.mlp_channels = mlp_channels
        self.num_classes = num_classes
        self.embed_dropout = embed_dropout
        self.dropout = dropout
        self.pool_func = pool_func

        assert chunk_size % t_patch_size == 0, f'EEG chunk size {chunk_size} must be divisible by the temporal patch size {t_patch_size}.'

        num_patches = chunk_size // t_patch_size
        patch_channels = num_electrodes * t_patch_size

        assert pool_func in {
            'cls', 'mean'
        }, 'pool_func must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p) -> b w (c p)', p=t_patch_size),
            nn.Linear(patch_channels, hid_channels),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hid_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_channels))
        self.dropout = nn.Dropout(embed_dropout)

        self.transformer = Transformer(hid_channels, depth, heads,
                                       head_channels, mlp_channels, dropout)

        self.pool_func = pool_func

        self.mlp_head = nn.Sequential(nn.LayerNorm(hid_channels),
                                      nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor, return_feat=False) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`chunk_size` corresponds to :obj:`chunk_size`.
        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.to_patch_embedding(x)
        x = rearrange(x, 'b ... d -> b (...) d')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        feat = x.mean(dim=1) if self.pool_func == 'mean' else x[:, 0]
        y =  self.mlp_head(feat)
        return (feat, y) if return_feat else y

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionEmbedding3D(nn.Module):
    def __init__(self, in_channels, temporature: float = 10000.0):
        super(PositionEmbedding3D, self).__init__()
        self.in_channels = in_channels
        self.temporature = temporature

        in_channels = int(np.ceil(in_channels / 6) * 2)
        if in_channels % 2:
            in_channels += 1
        self.in_channels = in_channels
        inv_freq = 1.0 / (temporature**(
            torch.arange(0, in_channels, 2).float() / in_channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 5:
            raise RuntimeError(
                "The input must be five-dimensional to perform thres-dimensional position embedding!"
            )

        if self.cached_penc is not None and self.cached_penc.shape == x.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, a, b, c, orig_ch = x.shape
        pos_a = torch.arange(a, device=x.device).type(self.inv_freq.type())
        pos_b = torch.arange(b, device=x.device).type(self.inv_freq.type())
        pos_c = torch.arange(c, device=x.device).type(self.inv_freq.type())
        sin_inp_a = torch.einsum("i,j->ij", pos_a, self.inv_freq)
        sin_inp_b = torch.einsum("i,j->ij", pos_b, self.inv_freq)
        sin_inp_c = torch.einsum("i,j->ij", pos_c, self.inv_freq)
        emb_a = get_emb(sin_inp_a).unsqueeze(1).unsqueeze(1)
        emb_b = get_emb(sin_inp_b).unsqueeze(1)
        emb_c = get_emb(sin_inp_c)
        emb = torch.zeros((a, b, c, self.in_channels * 3),
                          device=x.device).type(x.type())
        emb[:, :, :, :self.in_channels] = emb_a
        emb[:, :, :, self.in_channels:2 * self.in_channels] = emb_b
        emb[:, :, :, 2 * self.in_channels:] = emb_c

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(
            batch_size, 1, 1, 1, 1)
        return self.cached_penc
    
class SimpleViT(nn.Module):
    r'''
    A Simple and Effective Vision Transformer (SimpleViT). The authors of Vision Transformer (ViT) present a few minor modifications and dramatically improve the performance of plain ViT models. For more details, please refer to the following information. 
    It is worth noting that this model is not designed for EEG analysis, but shows good performance and can serve as a good research start.
    - Paper: Beyer L, Zhai X, Kolesnikov A. Better plain ViT baselines for ImageNet-1k[J]. arXiv preprint arXiv:2205.01580, 2022.
    - URL: https://arxiv.org/abs/2205.01580
    - Related Project: https://github.com/lucidrains/vit-pytorch
    Below is a recommended suite for use in emotion recognition tasks:
    .. code-block:: python
        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.MinMaxNormalize(axis=-1),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = SimpleViT(chunk_size=128,
                          grid_size=(9, 9),
                          t_patch_size=32,
                          num_classes=2)
    It can also be used for the analysis of features such as DE, PSD, etc:
    .. code-block:: python
        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BandDifferentialEntropy({
                            "delta": [1, 4],
                            "theta": [4, 8],
                            "alpha": [8, 14],
                            "beta": [14, 31],
                            "gamma": [31, 49]
                        }),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = SimpleViT(chunk_size=5,
                          grid_size=(9, 9),
                          t_patch_size=1,
                          num_classes=2)
    Args:
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`128`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        patch_size (tuple): The size (resolution) of each input patch. (defualt: :obj:`(3, 3)`)
        t_patch_size (int): The size of each input patch at the temporal (chunk size) dimension. (defualt: :obj:`32`)
        s_patch_size (tuple): The size (resolution) of each input patch at the spatial (grid size) dimension. (defualt: :obj:`(3, 3)`)
        hid_channels (int): The feature dimension of embeded patch. (defualt: :obj:`32`)
        depth (int): The number of attention layers for each transformer block. (defualt: :obj:`3`)
        heads (int): The number of attention heads for each attention layer. (defualt: :obj:`4`)
        head_channels (int): The dimension of each attention head for each attention layer. (defualt: :obj:`8`)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (defualt: :obj:`64`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
    '''
    def __init__(self,
                 chunk_size: int = 128,
                 grid_size: Tuple[int, int] = (9, 9),
                 t_patch_size: int = 32,
                 s_patch_size: Tuple[int, int] = (3, 3),
                 hid_channels: int = 32,
                 depth: int = 3,
                 heads: int = 4,
                 head_channels: int = 8,
                 mlp_channels: int = 64,
                 num_classes: int = 2):
        super(SimpleViT, self).__init__()
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.t_patch_size = t_patch_size
        self.s_patch_size = s_patch_size
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.head_channels = head_channels
        self.mlp_channels = mlp_channels
        self.num_classes = num_classes

        grid_height, grid_width = pair(grid_size)
        patch_height, patch_width = pair(s_patch_size)

        assert grid_height % patch_height == 0 and grid_width % patch_width == 0, f'EEG grid size {grid_size} must be divisible by the spatial patch size {s_patch_size}.'
        assert chunk_size % t_patch_size == 0, f'EEG chunk size {chunk_size} must be divisible by the temporal patch size {t_patch_size}.'

        patch_channels = t_patch_size * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b c h w (p1 p2 p0)',
                      p0=t_patch_size,
                      p1=patch_height,
                      p2=patch_width),
            nn.Linear(patch_channels, hid_channels),
        )
        self.position_embedding = PositionEmbedding3D(hid_channels)

        self.transformer = Transformer(hid_channels, depth, heads,
                                       head_channels, mlp_channels)

        self.linear_head = nn.Sequential(nn.LayerNorm(hid_channels),
                                         nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 128, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`128` corresponds to :obj:`chunk_size`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.
        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        b, *_ = x.shape
        x = self.to_patch_embedding(x)
        pe = self.position_embedding(x)
        x = rearrange(x + pe, 'b ... d -> b (...) d')

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.linear_head(x)