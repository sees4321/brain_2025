import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune

from trainer import *
from models.syncnet import SyncNet
from models.syncnet2 import SyncNet2, SyncNet3, SyncNet4, SyncNet_ablation
from models.eegnet import EEGNet
from models.shallowfbcspnet import ShallowFBCSPNet
from models.deep4net import Deep4Net
from models.cnnlstm import CNNLSTM
from models.bimodalnet_old1 import BimodalNet, Config
# from models.fnirs_model import *
from models.fnirsnet import fNIRSNet
from models.hirenet import HiRENet, make_input
from models.fnirs_transformer import fNIRS_PreT, divide_ab
from models.efnet import EF_net
from models.MTCA_CapsNet import MTCA_CapsNet
from modules import Emotion_DataModule, MIST_DataModule
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix


def leave_one_out_cross_validation(data_mode:int=0):
    ManualSeed(0) 
    learning_rate = 1e-3
    num_batch = 16
    num_epochs = 50
    min_epoch = 50
    time = datetime.datetime.now().strftime('%m%d_%H%M')
    path = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025'
    # path = 'D:/KMS/data/brain_2025'

    # emotion_dataset = Emotion_DataModule(path,
    #                                      data_mode=data_mode,
    #                                      label_type=label_type,
    #                                      ica=True,
    #                                      start_point=60,
    #                                      window_len=60,
    #                                      num_val=0,
    #                                      batch_size=num_batch,
    #                                      transform_eeg=None,
    #                                      transform_fnirs=None)
    
    emotion_dataset = MIST_DataModule(path,
                                        data_mode=data_mode,
                                        start_point=0,
                                        window_len=60,
                                        num_val=0,
                                        batch_size=num_batch,
                                        transform_eeg=None,
                                        transform_fnirs=None) #divide_ab if data_mode == 2 else None)
    # config = Config(
    #     eeg_channels=emotion_dataset.eeg.shape[2],
    #     eeg_num_samples=emotion_dataset.eeg.shape[-1],
    #     fnirs_channels=emotion_dataset.fnirs.shape[2],
    #     fnirs_num_samples=emotion_dataset.fnirs.shape[-1],
    #     eeg_temporal_length=64,
    #     num_classes=1,
    # )

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    # preds = np.zeros((num_subj,8)) # model predictions
    # targets = np.zeros((num_subj,8)) # labels
    cf_out = np.zeros((2,2),int)
    for subj, data_loaders in enumerate(emotion_dataset):
        train_loader, val_loader, test_loader = data_loaders

        # model = ShallowFBCSPNet([3,fs*60], fs).to(DEVICE)
        # model = EEGNet([7, fs*60], fs, 1).to(DEVICE)
        # model = HiRENet2(cls = True).to(DEVICE)
        # model = MTCA_CapsNet(2, 7500).to(DEVICE)
        # model = BimodalNet(config).to(DEVICE)
        # model = EEGNet_fNIRS(cls = True).to(DEVICE)
        # model = FNIRSSubNet(emb_dim=1, cls = True).to(DEVICE)
        # dim = 64
        # model = Bimodal_model(HiRENet2(drop_prob=0.25), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(Deep4Net([emotion_dataset.eeg.shape[-2], emotion_dataset.eeg.shape[-1]],cls=False), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(CNNLSTM(emotion_dataset.data_shape_eeg,cls=False), 
        #                       EEGNet_fNIRS(emotion_dataset.data_shape_fnirs, pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_model(EEGNet2(), EEGNet_fNIRS(pool_mode="mean"), 1).to(DEVICE)
        # model = Bimodal_attn_model(HiRENet3(emb_dim=dim), EEGNet_fNIRS3(emb_dim=dim), 1).to(DEVICE)

        if data_mode == 0:
            model = SyncNet2(emotion_dataset.data_shape_eeg, 
                            emotion_dataset.data_shape_fnirs, 
                            num_segments=seg,
                            embed_dim=256,
                            num_heads=4,
                            num_layers=2,
                            use_lstm=False,
                            num_groups=2,#2 if seg==30 else 4,
                            actv_mode="elu",
                            pool_mode="max", 
                            num_classes=1).to(DEVICE)
            # model = SyncNet_ablation(emotion_dataset.data_shape_eeg, 
            #                          emotion_dataset.data_shape_fnirs).to(DEVICE)
            # model = BimodalNet(config).to(DEVICE)
            # model = EF_net(1).to(DEVICE)
            trainer = train_bin_cls2
            tester = test_bin_cls2
        else:
            # model = SyncNet3(emotion_dataset.data_shape_eeg if data_mode==1 else emotion_dataset.data_shape_fnirs, 
            #                 data_mode=data_mode,
            #                 num_segments=12,
            #                 embed_dim=256,
            #                 num_heads=4,
            #                 num_layers=2,
            #                 use_lstm=False,
            #                 num_groups=4,
            #                 actv_mode="elu",
            #                 pool_mode="mean", 
            #                 num_classes=1).to(DEVICE)
            # model = HiRENet(7,16,num_seg=30).to(DEVICE)
            if data_mode == 1:
                model = EEGNet(emotion_dataset.eeg.shape[-2:], 200, 1).to(DEVICE)
            elif data_mode == 2:
                model = fNIRS_PreT(1,emotion_dataset.fnirs.shape[-1],32,2,4,32).to(DEVICE)
            # model = fNIRSNet(1,367).to(DEVICE)
            trainer = train_bin_cls
            tester = test_bin_cls

        # es = EarlyStopping(model, patience=10, mode='min')
        train_acc, train_loss, val_acc, val_loss = trainer(model, 
                                                            train_loader=train_loader, 
                                                            val_loader=val_loader,
                                                            num_epoch=num_epochs, 
                                                            optimizer_name='Adam',
                                                            learning_rate=str(learning_rate),
                                                            early_stop=None,
                                                            min_epoch=min_epoch,
                                                            exlr_on=False)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)
        vl_acc.append(val_acc)
        vl_loss.append(val_loss)

        # model.load_state_dict(torch.load('best_model.pth'))
        # prune.l1_unstructured(model.classifer[0], name='weight', amount=0.3)
        test_acc, preds, targets = tester(model, tst_loader=test_loader)
        ts_acc.append(test_acc)

        bcm = BinaryConfusionMatrix()
        cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
        # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
        ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
        ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
        cf_out += cf.numpy()

        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')
    # plot_confusion_matrix(cf_out,['High','Low'])

seg = 30
if __name__ == "__main__":
    # for seg in [8, 12, 20, 24, 30]:
    #     if seg not in [24, 30]: continue
    # or seg in [64,128,256,512]:
    print('-'*32 + str(seg))
    leave_one_out_cross_validation(0)
    # leave_one_out_cross_validation(1)
    # leave_one_out_cross_validation(2)
