import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from eegnet import EEGNet


class Config:
    def __init__(self, **kwargs):

        self.eeg_channels = 64
        self.eeg_num_samples = 200
        self.eeg_temporal_filters = 40
        self.eeg_temporal_length = 5
        self.eeg_spatial_filters = 40
        self.eeg_dropout = 0.1

        self.fnirs_channels = 16
        self.fnirs_num_samples = 200
        self.fnirs_temporal_filters = 40
        self.fnirs_temporal_length = 5
        self.fnirs_spatial_filters = 40
        self.fnirs_dropout = 0.1

        self.num_classes = 2
        self.gru_input = 500
        self.gru_hidden = 250
        self.gru_dropout = 0.2

        self.__dict__.update(kwargs)

    def get_eeg_params(self):
        return {
            "input_channels": self.eeg_channels,
            "num_samples": self.eeg_num_samples,
            "temporal_filters": self.eeg_temporal_filters,
            "temporal_length" : self.eeg_temporal_length,
            "spatial_filters": self.eeg_spatial_filters,
            "gru_input": self.gru_input,
            "dropout": self.eeg_dropout
        }

    def get_fnirs_params(self):
        return {
            "input_channels": self.fnirs_channels,
            "num_samples": self.fnirs_num_samples,
            "temporal_filters": self.fnirs_temporal_filters,
            "temporal_length" : self.fnirs_temporal_length,
            "spatial_filters": self.fnirs_spatial_filters,
            "gru_input": self.gru_input,
            "dropout": self.fnirs_dropout
        }


# EEG Subnet
class EEGSubNet(nn.Module):
    def __init__(self, input_channels, temporal_filters, temporal_length, spatial_filters, gru_input, dropout, num_samples):
        super(EEGSubNet, self).__init__()
        self.conv1 = nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_length))
        self.conv2 = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(input_channels, 1))
        self.batch_norm = nn.BatchNorm2d(spatial_filters)
        self.mean_pool = nn.AvgPool2d(kernel_size = (1, 5), stride=(1, 5))
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear((spatial_filters * (num_samples - temporal_length + 1)), gru_input)  # Adjust based on input dimensions
        self.fc = nn.Linear(60920, gru_input)
        #self.fc = nn.Linear(61160, gru_input)
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch_norm(x)
        x = self.mean_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# EEG Subnet
class EEGSubNet_Att(nn.Module):
    def __init__(self, input_channels, temporal_filters, temporal_length, spatial_filters, gru_input, dropout, num_samples):
        super(EEGSubNet_Att, self).__init__()
        self.conv1 = nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_length), stride=(1, 4))
        self.conv2 = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(input_channels, 1))
        self.batch_norm = nn.BatchNorm2d(spatial_filters)
        self.mean_pool = nn.AvgPool2d(kernel_size = (1, 50), stride=(1, 50))
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear((spatial_filters * (num_samples - temporal_length + 1)), gru_input)  # Adjust based on input dimensions
        self.fc = nn.Linear(1520, gru_input)
        #self.fc = nn.Linear(61160, gru_input)
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch_norm(x)
        x = self.mean_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# fNIRS Subnet
class FNIRSSubNet(nn.Module):
    def __init__(self, input_channels, temporal_filters, temporal_length, spatial_filters, gru_input, dropout, num_samples):
        super(FNIRSSubNet, self).__init__()
        self.conv1 = nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_length))
        self.conv2 = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(input_channels, 1))
        self.batch_norm = nn.BatchNorm2d(spatial_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(spatial_filters * (num_samples - temporal_length + 1), gru_input)  # Adjust based on input dimensions

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


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


class BimodalNet(nn.Module):
    def __init__(self, config):
        super(BimodalNet, self).__init__()
        self.eeg_subnet = EEGSubNet(**config.get_eeg_params())
        self.fnirs_subnet = FNIRSSubNet(**config.get_fnirs_params())
        self.gru = nn.GRU(config.gru_input, config.gru_hidden, batch_first=True)  # 500 + 500 from subnets
        self.dropout = nn.Dropout(config.gru_dropout)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)
        self.apply_initialization()

    def forward(self, eeg_data, fnirs_data):
        #eeg_features = self.eeg_subnet(eeg_data)
        fnirs_features = self.fnirs_subnet(fnirs_data)
        #combined_features = torch.cat((eeg_features, fnirs_features), dim=1).unsqueeze(1)  # Add time dim for GRU
        combined_features = fnirs_features
        gru_out, _ = self.gru(combined_features)
        #gru_out = gru_out[:, -1, :]  # Take last GRU output
        x = self.dropout(gru_out)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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
                elif isinstance(module, nn.GRU):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            init.kaiming_uniform_(param, nonlinearity='relu')
                        elif 'bias' in name:
                            init.zeros_(param)


class BimodalAttNet(nn.Module):
    def __init__(self, config):
        super(BimodalAttNet, self).__init__()
        #self.eeg_subnet = EEGSubNet_Att(**config.get_eeg_params())
        self.eeg_subnet = EEGNet()
        self.fnirs_subnet = FNIRSSubNet(**config.get_fnirs_params())
        self.att_eeg_to_fnirs = CrossAttention(config.gru_input)
        self.att_fnirs_to_eeg = CrossAttention(config.gru_input)
        self.dropout = nn.Dropout(config.gru_dropout)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)
        self.apply_initialization()

    def forward(self, eeg_data, fnirs_data):
        eeg_features = self.eeg_subnet(eeg_data)
        fnirs_features = self.fnirs_subnet(fnirs_data)
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
        x = self.dropout(combined_features)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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
                elif isinstance(module, nn.GRU):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            init.kaiming_uniform_(param, nonlinearity='relu')
                        elif 'bias' in name:
                            init.zeros_(param)


class BimodalEEGNet(nn.Module):
    def __init__(self, config):
        super(BimodalEEGNet, self).__init__()
        self.eeg_subnet = EEGNet()
        self.fnirs_subnet = FNIRSSubNet(**config.get_fnirs_params())
        self.gru = nn.GRU(config.gru_input*2, config.gru_hidden, batch_first=True)  # 500 + 500 from subnets
        self.dropout = nn.Dropout(config.gru_dropout)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)
        self.apply_initialization()

    def forward(self, eeg_data, fnirs_data):
        eeg_features = self.eeg_subnet(eeg_data)
        fnirs_features = self.fnirs_subnet(fnirs_data)
        combined_features = torch.cat((eeg_features, fnirs_features), dim=1).unsqueeze(1)  # Add time dim for GRU
        gru_out, _ = self.gru(combined_features)
        gru_out = gru_out[:, -1, :]  # Take last GRU output
        x = self.dropout(gru_out)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

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
                elif isinstance(module, nn.GRU):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            init.kaiming_uniform_(param, nonlinearity='relu')
                        elif 'bias' in name:
                            init.zeros_(param)


if __name__ == "__main__":
    # 모델 초기화

    config = Config()

    model = BimodalNet(config)

    # 입력 데이터 샘플
    eeg_data = torch.rand(32, 1, (getattr(config, 'eeg_channels')), (getattr(config, 'eeg_num_samples')))  # batch_size x 1 x channels x samples
    fnirs_data = torch.rand(32, 1, (getattr(config, 'fnirs_channels')), (getattr(config, 'fnirs_num_samples')))

    # 출력
    output = model(eeg_data, fnirs_data)
    print(output.shape)  # Expected output: [32, num_classes]