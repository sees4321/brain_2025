import datetime
import time
import numpy as np
import torch.nn.utils.prune as prune

from trainer import *
from models.syncnet import SyncNet
from models.syncnet2 import SyncNet2, SyncNet3, SyncNet4

from models.fnirsnet import fNIRSNet
from models.fnirs_transformer import fNIRS_PreT, divide_ab
from models.eegnet import EEGNet
from models.hirenet import *
from models.bimodalnet_old1 import BimodalNet, Config
from models.efnet import EF_net

from modules import MIMA_DataModule
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassF1Score


def leave_one_out_cross_validation(data_mode:int=0, label_type:int=0):
    ManualSeed(0)
    learning_rate = 1e-3
    num_batch = 64
    num_epochs = 50
    min_epoch = 50
    start_time = datetime.datetime.now().strftime('%m%d_%H%M')
    # path = 'D:\One_한양대학교\private object minsu\coding\data\EEG_fnirs_cognitive_open\datasetA'
    path = 'D:/KMS/data/brain_2025'
    
    
    dataset = MIMA_DataModule(path,
                                data_mode=data_mode,
                                label_type=label_type,
                                num_val=0,
                                batch_size=num_batch,
                                # transform_eeg=make_input if data_mode == 1 else None,
                                transform_eeg=make_input if aaa == 2 else None,
                                transform_fnirs=divide_ab if data_mode == 2 else None)

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    # preds = np.zeros((num_subj,8)) # model predictions
    # targets = np.zeros((num_subj,8)) # labels
    for subj, data_loaders in enumerate(dataset):
        tm = time.time()
        train_loader, val_loader, test_loader = data_loaders


        if data_mode == 0:
            if label_type == 4:
                trainer = train_bin_cls3
                tester = test_bin_cls3
                num_classes = 3
            else:
                trainer = train_bin_cls2
                tester = test_bin_cls2
                num_classes = 1
            # model = SyncNet2(dataset.data_shape_eeg, 
            #                 dataset.data_shape_fnirs, 
            #                 num_segments=4,
            #                 embed_dim=128,
            #                 num_heads=2,
            #                 num_layers=1,
            #                 num_groups=1,
            #                 actv_mode="elu",
            #                 pool_mode="mean", 
            #                 k_size=[15, 3],
            #                 hid_dim=[128, 32],
            #                 num_classes=num_classes).to(DEVICE)
            # config = Config(
            #     eeg_channels=dataset.eeg.shape[-2],
            #     eeg_num_samples=dataset.eeg.shape[-1],
            #     fnirs_channels=dataset.fnirs.shape[-2],
            #     fnirs_num_samples=dataset.fnirs.shape[-1],
            #     eeg_temporal_length=64,
            #     num_classes=num_classes,
            # )
            # model = BimodalNet(config).to(DEVICE)
            model = EF_net(num_classes).to(DEVICE)
            # model = None
        else:
            if label_type == 4:
                trainer = train_bin_cls4
                tester = test_bin_cls4
                num_classes = 3
            else:
                trainer = train_bin_cls
                tester = test_bin_cls
                num_classes = 1
            # model = SyncNet3(dataset.data_shape_eeg if data_mode==1 else dataset.data_shape_fnirs, 
            #                 data_mode=data_mode,
            #                 num_segments=4,
            #                 embed_dim=128,
            #                 num_heads=4,
            #                 num_layers=2,
            #                 use_lstm=False,
            #                 num_groups=4,
            #                 actv_mode="gelu",
            #                 pool_mode="mean", 
            #                 num_classes=num_classes).to(DEVICE)
            if data_mode == 1:
                if aaa == 1:
                    learning_rate = 5e-5
                    model = EEGNet(dataset.eeg.shape[-2:], 200, num_classes).to(DEVICE)
                else:
                    learning_rate = 5e-5
                    model = HiRENet(dataset.eeg.shape[-3],dataset.eeg.shape[-1],num_seg=dataset.eeg.shape[-2]//2,num_classes=num_classes).to(DEVICE)
            else:
                model = fNIRS_PreT(num_classes,dataset.fnirs.shape[-1],32,2,4,32).to(DEVICE)
                # model = fNIRSNet(num_classes,dataset.fnirs.shape[-1],dataset.fnirs.shape[-2]).to(DEVICE)

        # es = EarlyStopping(model, patience=10, mode='min')
        es = None
        train_acc, train_loss, val_acc, val_loss = trainer(model, 
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

        if es:
            model.load_state_dict(torch.load('best_model.pth'))
        # prune.l1_unstructured(model.classifier.fc[0], name='weight', amount=0.3)
        test_acc, preds, targets = tester(model, tst_loader=test_loader)
        # ts_acc.append(test_acc)

        if num_classes == 1:
            bcm = BinaryConfusionMatrix()
            cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
            # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
            ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
            ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
            ts_acc.append((cf[0,0]+cf[1,1])/(cf[0,0]+cf[0,1]+cf[1,0]+cf[1,1]) * 100)
        else:
            f1_ = MulticlassF1Score(num_classes=3, average='none')
            ts_sen.append(list(f1_(torch.from_numpy(preds), torch.from_numpy(targets))))

            f1_ = MulticlassF1Score(num_classes=3, average='micro')
            ts_acc.append(f1_(torch.from_numpy(preds), torch.from_numpy(targets)) * 100)

        print(f'[{subj:0>2}] acc: {ts_acc[-1]:.2f} %,  training acc: {train_acc[-1]:.2f} %,  training loss: {train_loss[-1]:.4f},  avg Acc: {np.mean(ts_acc):.2f} %,  time: {time.time() - tm:.1f}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')
    
    
    if label_type == 4:
        ts_sen = np.array(ts_sen)
        print(f'[{data_mode} {label_type}]  avg Acc: {np.mean(ts_acc):.2f} %,  std: {np.std(ts_acc):.2f},  f1: {np.mean(ts_sen[:,0])*100:.2f}  {np.mean(ts_sen[:,1])*100:.2f}  {np.mean(ts_sen[:,2])*100:.2f}')
    else:
        print(f'[{data_mode} {label_type}]  avg Acc: {np.mean(ts_acc):.2f} %,  std: {np.std(ts_acc):.2f},  sen: {np.mean(ts_sen)*100:.2f},  spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')

aaa = 1
if __name__ == "__main__":
    # leave_one_out_cross_validation(1,4)
    # for dat_type in [0,1,2]:
    #     print('-'*32 + str(dat_type))
    #     leave_one_out_cross_validation(dat_type,2)

    # for dat_type in [1,2]:
    #     for label_type in [2,3,4]:
    #         print('-'*32 + str(dat_type) + str(label_type))
    #         leave_one_out_cross_validation(dat_type,label_type)
    dat_type = [0,1,1]
    for i in range(1):
        aaa = i
        for label_type in [2,3]:
            print('-'*32 + str(i) + str(label_type))
            leave_one_out_cross_validation(dat_type[i], label_type)

    # for i, v in enumerate(['wg','dsr','nback']):
    #     for dat_type in [0,1,2]:
    #         leave_one_out_cross_validation(dat_type,i+2)
    # for i, v in enumerate(['wg','dsr','nback']):
    #     with open(f"{v}.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(0,i,config)
    # for i in range(3):
    #     # for j in range(3):
    #     leave_one_out_cross_validation(2,i)