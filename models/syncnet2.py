import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def segment_data(data:torch.Tensor, num_seg = 12):
    # data shape: (B, channels, time_samples), 여기서 time_samples = 7680
    end = data.size(-1)
    segment_length = ceil(end/num_seg)
    if end % segment_length != 0:
        end = (end // segment_length + 1) * segment_length
    data = F.pad(data, (0,(end-data.size(-1))), mode='replicate')
    segments = []
    for i in range(0, end, segment_length):
        segment = data[:, :, i:i + segment_length]  # (batch, channels, segment_length)
        segments.append(segment)
    return torch.stack(segments, dim=1)  # (batch, num_segments, channels, segment_length)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.attn = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        # x shape: (batch, out_dim, channels, time)
        attn_weights = torch.softmax(self.attn(x.mean(dim=-1)), dim=-1)  # (batch, out_dim, channels)
        x = x * attn_weights.unsqueeze(-1)  # (batch, out_dim, channels, time)
        return x
        # return x.sum(dim=1, keepdim=True)  # (batch, 1, channels, time)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)  # (max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, dim)
    
    def forward(self, x):
        # x shape: (batch, tokens, embed_dim)
        return x + self.pe[:, :x.size(1), :].to(x.device)  # (batch, tokens, embed_dim)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_heads, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim*num_segments, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax()
        )
    
    def forward(self, x):
        # x shape: (batch, tokens, embed_dim)
        x = self.transformer(x)  # (batch, tokens, embed_dim)
        x = torch.flatten(x,1)
        # x = x.mean(dim=1)  # (batch, embed_dim)
        return self.fc(x)  # (batch, num_classes)

class EEG_Temporal_Encoder(nn.Module):
    def __init__(self, in_channels, in_size, kernel_size, hid_dim, out_dim, emb_dim, act=nn.GELU, pool=nn.AvgPool3d, groups=4):
        super(EEG_Temporal_Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, hid_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            # nn.BatchNorm3d(hid_dim),
            nn.GroupNorm(groups, hid_dim),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(hid_dim, hid_dim, (1, in_channels, 1)),
            act(),
            # nn.BatchNorm3d(hid_dim),
            nn.GroupNorm(groups, hid_dim),
            pool(kernel_size=(1, 1, 2), stride=(1, 1, 2)),

            nn.Conv3d(hid_dim, out_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            # nn.BatchNorm3d(out_dim),
            nn.GroupNorm(groups, out_dim),
            pool(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
        )
        self.chan_attn = ChannelAttention(in_channels)
        self.embedding = nn.Linear(out_dim*in_size//4, emb_dim)

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1) # (batch, 1, num_segments, channels, segment_length) 
        x = self.conv_block(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        # x = self.chan_attn(x)
        x = self.conv_block2(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        x = x.squeeze(3).permute(0,2,1,3).flatten(2)
        return self.embedding(x)

class fNIRS_Temporal_Encoder(nn.Module):
    def __init__(self, in_channels, in_size, kernel_size, hid_dim, out_dim, emb_dim, act=nn.GELU, groups=4):
        super(fNIRS_Temporal_Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, hid_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            # nn.BatchNorm3d(hid_dim),
            nn.GroupNorm(groups, hid_dim),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(hid_dim, out_dim, (1, in_channels, 1)),
            act(),
            # nn.BatchNorm3d(out_dim),
            nn.GroupNorm(groups, out_dim),
        )
        self.chan_attn = ChannelAttention(in_channels)
        self.embedding = nn.Linear(out_dim*in_size, emb_dim)

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1) # (batch, 1, num_segments, channels, segment_length) 
        x = self.conv_block(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        # x = self.chan_attn(x)
        x = self.conv_block2(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        x = x.squeeze(3).permute(0,2,1,3).flatten(2)
        return self.embedding(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(input_dim, input_dim*2, 2, batch_first=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim*2*num_segments, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax()
        )
    
    def forward(self, x):
        # x shape: (batch, tokens, embed_dim)
        x, (h_n, c_n) = self.lstm(x)  # (batch, tokens, embed_dim)
        x = torch.flatten(x,1)
        return self.fc(x)  # (batch, num_classes)

class SyncNet2(nn.Module):
    def __init__(self, 
                 eeg_shape, 
                 fnirs_shape, 
                 num_segments=12,
                 embed_dim=128, 
                 num_heads=4, 
                 num_layers=2, 
                 use_lstm = False,
                 num_groups = 4,
                 actv_mode = "elu", 
                 pool_mode = "mean", 
                 num_classes=1):
        super(SyncNet2, self).__init__()

        self.num_segments = num_segments
        actv = dict(elu=nn.ELU, gelu=nn.GELU, relu=nn.ReLU)[actv_mode]
        pool = dict(max=nn.MaxPool3d, mean=nn.AvgPool3d)[pool_mode]

        self.eeg_emb = EEG_Temporal_Encoder(eeg_shape[0], round(eeg_shape[-1]/num_segments), 13, 16, 32, embed_dim, actv, pool, num_groups)
        self.fnirs_emb = fNIRS_Temporal_Encoder(fnirs_shape[0], ceil(fnirs_shape[-1]/num_segments), 5, 16, 32, embed_dim, actv, num_groups)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.fusion_conv = nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
        if use_lstm:
            self.classifier = LSTMClassifier(embed_dim, num_segments, num_classes)
        else:
            self.classifier = TransformerClassifier(embed_dim, num_segments, num_heads, num_layers, num_classes)
    
    def forward(self, eeg, fnirs):
        # segmentation
        eeg = segment_data(eeg, self.num_segments) # (batch, num_segments, channels, segment_length) 
        fnirs = segment_data(fnirs, self.num_segments) # (batch, num_segments, channels, segment_length)
        # print(eeg.shape)

        eeg = self.eeg_emb(eeg) # (batch, num_segments, embed_dim)
        fnirs = self.fnirs_emb(fnirs) # (batch, num_segments, embed_dim)
        # print(eeg.shape)

        fused_tokens = torch.cat([eeg, fnirs], dim=2)  # (batch, total_tokens, embed_dim*2)
        fused_tokens = self.fusion_conv(fused_tokens.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, total_tokens, embed_dim)
        fused_tokens = self.pos_encoder(fused_tokens)  # (batch, total_tokens, embed_dim)
        # print(eeg.shape)
        return self.classifier(fused_tokens)  # (batch, num_classes)
    

class SyncNet3(nn.Module):
    def __init__(self, 
                 eeg_shape, 
                 data_mode=1,
                 num_segments=12,
                 embed_dim=128, 
                 num_heads=4, 
                 num_layers=2, 
                 use_lstm = False,
                 num_groups = 4,
                 actv_mode = "elu", 
                 pool_mode = "mean", 
                 num_classes=1):
        super(SyncNet3, self).__init__()

        self.num_segments = num_segments
        actv = dict(elu=nn.ELU, gelu=nn.GELU, relu=nn.ReLU)[actv_mode]
        pool = dict(max=nn.MaxPool3d, mean=nn.AvgPool3d)[pool_mode]
        
        if data_mode==1:
            self.eeg_emb = EEG_Temporal_Encoder(eeg_shape[0], round(eeg_shape[-1]/num_segments), 13, 16, 32, embed_dim, actv, pool, num_groups)
        elif data_mode == 2:
            self.eeg_emb = fNIRS_Temporal_Encoder(eeg_shape[0], round(eeg_shape[-1]/num_segments), 5, 16, 32, embed_dim, actv, num_groups)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        if use_lstm:
            self.classifier = LSTMClassifier(embed_dim, num_segments, num_classes)
        else:
            self.classifier = TransformerClassifier(embed_dim, num_segments, num_heads, num_layers, num_classes)
    
    def forward(self, eeg):
        # segmentation
        eeg = segment_data(eeg, self.num_segments) # (batch, num_segments, channels, segment_length) 
        # print(eeg.shape)

        eeg = self.eeg_emb(eeg) # (batch, num_segments, embed_dim)
        # print(eeg.shape)

        eeg = self.pos_encoder(eeg)  # (batch, total_tokens, embed_dim)
        # print(eeg.shape)
        return self.classifier(eeg)  # (batch, num_classes)