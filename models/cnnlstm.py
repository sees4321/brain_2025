import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, input_size = [3, 7680], n_classes = 1, pool_mode = "mean", cls=False):
        super(CNNLSTM, self).__init__()
        self.cls = cls

        conv_len = 13
        num_filters = 16
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
        self.block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters*2),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len)),
            nn.Conv2d(num_filters*2, num_filters*4, (1, conv_len), padding=(0, conv_len//2)),
            nn.BatchNorm2d(num_filters*4),
            nn.ELU(),
            pool_class(kernel_size=(1, pool_len), stride=(1, pool_len))
        )


        self.block3 = nn.Sequential(
            nn.LSTM(num_filters*4, num_filters*8, 2, batch_first=True)
        )
        
        self.dim_feat = num_filters*8*(input_size[1]//pool_len//pool_len//pool_len)

        # Classification Layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.dim_feat, 128*10),
            nn.Linear(128*10, n_classes),
            nn.Sigmoid() if n_classes == 1 else nn.LogSoftmax()

        )
        
    def forward(self, x):
        x = self.block1(x) # (B, f, 1, 3840)
        # print(x.shape)
        x = self.block2(x) # (B, f*4, 1, 960)
        # print(x.shape)
        x = torch.squeeze(x).transpose(1,2)
        x, (h_n, c_n) = self.block3(x) # (B, 960, f*8)
        # print(x.shape)
        
        x = torch.flatten(x,1)
        if not self.cls:
            return x
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    inp = torch.randn((16,1,7,7680))
    model = CNNLSTM([inp.shape[-2],inp.shape[-1]],cls=True)
    out = model(inp)
    print(out.shape)

r'''
conv1 = layers.Conv2D(16, (49,1), padding='same', activation='relu')(input_high)  #(BATCH,2048,6,16)
averagepool = layers.AveragePooling2D((32,1))(conv1)    #(BATCH,64,6,16)
reshape = layers.Reshape((64,input_shape[2]*16),input_shape=(64,input_shape[2],16))(averagepool)    #(BATCH,64,96) 
conv2 = layers.Conv1D(64, 3, padding='same', activation='relu')(reshape)     #(BATCH,64,64)
maxpool1 = layers.MaxPooling1D(2)(conv2)    #(Batch,32,64)
conv3 = layers.Conv1D(64, 3, padding='same', activation='relu')(maxpool1)     #(BATCH,32,64)
maxpool2 = layers.MaxPooling1D(2)(conv3)    #(Batch,16,64)

conso_sub1 = layers.Subtract()([averagepool[:,:,4,:], averagepool[:,:,5,:]])    #(BATCH,64,16)
conso_conv1 = layers.Conv1D(32, 3, padding='same', activation='relu')(conso_sub1)
conso_maxpool1 = layers.MaxPooling1D(4)(conso_conv1)    # (BATCH,16,16)

concat = layers.concatenate([maxpool2, conso_maxpool1])

lstm1 = layers.LSTM(128,return_sequences=True)(concat)  # (BATCH,16,128)

att1_conv = layers.Conv1D(128, 1, activation='relu')(lstm1)

lstm2 = layers.LSTM(128,return_sequences=True)(att1_conv)  # (BATCH,16,128)
lstm3 = layers.LSTM(100)(lstm2)  # (BATCH,100)

output = layers.Softmax()(lstm3)

model = keras.Model(inputs=[input_high], outputs = [output])
'''