import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune

from trainer import *
from models.syncnet import SyncNet
from models.syncnet2 import SyncNet2
from models.eegnet import EEGNet
from models.shallowfbcspnet import ShallowFBCSPNet
from models.deep4net import Deep4Net
from models.cnnlstm import CNNLSTM
from models.bimodalnet_old1 import BimodalNet, Config
# from models.fnirs_model import *
# from models.hirenet import HiRENet, make_input
from models.MTCA_CapsNet import MTCA_CapsNet
from modules import Emotion_DataModule
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix

ManualSeed(2222)

def leave_one_out_cross_validation(label_type:int=0):
    learning_rate = 5e-4
    num_batch = 32
    num_epochs = 51
    min_epoch = 50
    time = datetime.datetime.now().strftime('%m%d_%H%M')

    emotion_dataset = Emotion_DataModule('D:/KMS/data/brain_2025',
                                        label_type=label_type,
                                        ica=False,
                                        start_point=60,
                                        window_len=60,
                                        num_val=3,
                                        batch_size=num_batch,
                                        transform_eeg=None,
                                        transform_fnirs=None)
    config = Config(
        eeg_channels=emotion_dataset.eeg.shape[2],
        eeg_num_samples=emotion_dataset.eeg.shape[-1],
        fnirs_channels=emotion_dataset.fnirs.shape[2],
        fnirs_num_samples=emotion_dataset.fnirs.shape[-1],
        eeg_temporal_length=64,
        num_classes=1,
    )

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    # preds = np.zeros((num_subj,8)) # model predictions
    # targets = np.zeros((num_subj,8)) # labels
    for subj, data_loaders in enumerate(emotion_dataset):
        train_loader, val_loader, test_loader = data_loaders

        # model = ShallowFBCSPNet([3,fs*60], fs).to(DEVICE)
        # model = EEGNet([7, fs*60], fs, 1).to(DEVICE)
        # model = HiRENet2(cls = True).to(DEVICE)
        # model = MTCA_CapsNet(2, 7500).to(DEVICE)
        # model = BimodalNet(config).to(DEVICE)
        # model = EEGNet_fNIRS(cls = True).to(DEVICE)
        # model = FNIRSSubNet(emb_dim=1, cls = True).to(DEVICE)
        dim = 64
        # model = Bimodal_model(HiRENet2(drop_prob=0.25), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(Deep4Net([emotion_dataset.eeg.shape[-2], emotion_dataset.eeg.shape[-1]],cls=False), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(CNNLSTM(emotion_dataset.data_shape_eeg,cls=False), 
        #                       EEGNet_fNIRS(emotion_dataset.data_shape_fnirs, pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(EEGNet2(), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_attn_model(HiRENet3(emb_dim=dim), EEGNet_fNIRS3(emb_dim=dim), 1).to(DEVICE)

        model = SyncNet2(emotion_dataset.data_shape_eeg, 
                        emotion_dataset.data_shape_fnirs, 
                        num_segments=12,
                        embed_dim=256,
                        num_heads=4,
                        num_layers=2,
                        use_lstm=False,
                        num_groups=4,
                        actv_mode="elu",
                        pool_mode="mean", 
                        num_classes=1).to(DEVICE)

        es = EarlyStopping(model, patience=10, mode='min')
        train_acc, train_loss, val_acc, val_loss = train_bin_cls2(model, 
                                                                train_loader=train_loader, 
                                                                val_loader=val_loader,
                                                                num_epoch=num_epochs, 
                                                                optimizer_name='Adam',
                                                                learning_rate=str(learning_rate),
                                                                early_stop=es,
                                                                min_epoch=min_epoch,
                                                                exlr_on=False)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)
        vl_acc.append(val_acc)
        vl_loss.append(val_loss)

        model.load_state_dict(torch.load('best_model.pth'))
        # prune.l1_unstructured(model.classifer[0], name='weight', amount=0.3)
        test_acc, preds, targets = test_bin_cls2(model, tst_loader=test_loader)
        ts_acc.append(test_acc)
        bcm = BinaryConfusionMatrix()
        cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
        # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
        ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
        ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
        # ts_acc.append(val_acc[-1])
        # ts_acc[subj], preds[subj], targets[subj] = DoTest_bin(model, tst_loader=test_loader)
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')

#type chan n_chan
#a0 v1 / 0full, 123 / 8 3 3 2
leave_one_out_cross_validation(0)
leave_one_out_cross_validation(1)


# for typ in range(2):
#     for chan in range(3):
#         n_chan = 2 if chan == 2 else 3
#         main(typ, chan+1, n_chan)
# SaveResults_mat(f'eegnet_{time}',ts_acc,preds,targets,tr_acc,tr_loss,num_batch,num_epochs,learning_rate)