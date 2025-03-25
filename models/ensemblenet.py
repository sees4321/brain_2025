import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnsembleNet(nn.Module):
    # def __init__(self, *models:nn.Module, sm_model:nn.Module):
    def __init__(self, sm_model:nn.Module):
        super().__init__()
        
        # self.models = [*models]
        # self.num_model = len(models)
        self.sm_model = sm_model
        self.num_model = 3
        # for m in self.models:
        #     for p in m.parameters():
        #         p.requires_grad = False


        emb_size = 128
        self.model_emb = nn.Parameter(torch.ones((self.num_model,1024)), True)
        nn.init.xavier_uniform_(self.model_emb)
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.emb_layer = nn.Sequential(
            nn.Linear(1024, emb_size),
            nn.GELU()
        )
        self.sm_emb_layer = nn.Sequential(
            nn.Linear(256, emb_size),
            nn.GELU()
        )
        self.weight_layer = nn.Sequential(
            nn.Linear(self.num_model, self.num_model),
            nn.GELU()
        )
    
    def forward(self, x):
        # yhat = []
        # for i in range(self.num_model):
        #     yhat.append(self.sigmoid(self.models[i](x)))
        # yhat = torch.concat(yhat, dim=1) # [B, num_models]
        # yhat = self.softmax((yhat*2-1)**2)
        model_emb = self.emb_layer(self.model_emb) # [num_model, model_emb]
        sm_emb = self.sm_emb_layer(self.sm_model(x)) # [B, model_emb]
        w = F.softplus(self.weight_layer(torch.matmul(sm_emb, model_emb.t()))) # [B, num_models]
        return w 
        # return (w, loss) if self.training else w