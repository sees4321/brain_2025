import datetime
import time
import numpy as np
import torch.nn.utils.prune as prune
import yaml

from box import Box
from trainer import *
from models.syncnet import SyncNet
from models.syncnet2 import SyncNet2, SyncNet3, SyncNet4, SyncNet_ablation
from modules import MIMA_DataModule
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassF1Score, MulticlassConfusionMatrix


def leave_one_out_cross_validation(config, data_mode:int=0, label_type:int=0):
    ManualSeed(0)
    learning_rate = config.training.learning_rate
    num_batch = config.training.num_batch
    num_epochs = config.training.num_epochs
    min_epoch = 50
    start_time = datetime.datetime.now().strftime('%m%d_%H%M')
    # path = 'D:\One_한양대학교\private object minsu\coding\data\EEG_fnirs_cognitive_open'
    path = 'D:/KMS/data/brain_2025'
    
    dataset = MIMA_DataModule(path,
                                data_mode=data_mode,
                                label_type=label_type,
                                num_val=0,
                                batch_size=num_batch,
                                transform_eeg=None,
                                transform_fnirs=None)

    tr_acc = []
    tr_loss = []
    vl_acc = []
    vl_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    # preds = np.zeros((num_subj,8)) # model predictions
    # targets = np.zeros((num_subj,8)) # labels
    cf_size = 2 if config.model.num_classes < 3 else 3
    cf_out = np.zeros((cf_size,cf_size),int)
    tm = time.time()
    for subj, data_loaders in enumerate(dataset):
        times = []
        train_loader, val_loader, test_loader = data_loaders


        if data_mode == 0:
            model = SyncNet2(dataset.data_shape_eeg, 
                            dataset.data_shape_fnirs, 
                            num_segments=seg,#config.model.num_segments,
                            embed_dim=config.model.embed_dim,
                            num_heads=config.model.num_heads,
                            num_layers=config.model.num_layers,
                            use_lstm=True,
                            num_groups=config.model.num_groups,
                            actv_mode=config.model.actv_mode,
                            pool_mode=config.model.pool_mode, 
                            k_size=config.model.k_size,
                            hid_dim=config.model.hid_dim,
                            num_classes=config.model.num_classes).to(DEVICE)
            # model = SyncNet_ablation(dataset.data_shape_eeg, 
            #                          dataset.data_shape_fnirs,
            #                          num_classes=config.model.num_classes).to(DEVICE)
            if config.model.num_classes > 1:
                trainer = train_bin_cls3
                tester = test_bin_cls3
            else:
                trainer = train_bin_cls2
                tester = test_bin_cls2
        else:
            model = SyncNet3(dataset.data_shape_eeg if data_mode==1 else dataset.data_shape_fnirs, 
                            data_mode=data_mode,
                            num_segments=config.model.num_segments,
                            embed_dim=config.model.embed_dim,
                            num_heads=config.model.num_heads,
                            num_layers=config.model.num_layers,
                            num_groups=config.model.num_groups,
                            actv_mode=config.model.actv_mode,
                            pool_mode=config.model.pool_mode, 
                            k_size=config.model.k_size,
                            hid_dim=config.model.hid_dim,
                            num_classes=config.model.num_classes).to(DEVICE)
            if config.model.num_classes > 1:
                trainer = train_bin_cls4
                tester = test_bin_cls4
            else:
                trainer = train_bin_cls
                tester = test_bin_cls
        times.append(time.time() - tm)
        # es = EarlyStopping(model, patience=10, mode='min')
        es = None
        train_acc, train_loss, val_acc, val_loss = trainer(model, 
                                                            train_loader=train_loader, 
                                                            val_loader=val_loader,
                                                            num_epoch=num_epochs, 
                                                            optimizer_name=config.training.optimizer,
                                                            learning_rate=str(learning_rate),
                                                            early_stop=es,
                                                            min_epoch=min_epoch,
                                                            exlr_on=config.training.exlr_on)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)
        vl_acc.append(val_acc)
        vl_loss.append(val_loss)
        times.append(time.time() - times[-1] - tm)

        if es:
            model.load_state_dict(torch.load('best_model.pth'))
        # prune.l1_unstructured(model.classifier.fc[0], name='weight', amount=0.3)
        test_acc, preds, targets = tester(model, tst_loader=test_loader)
        times.append(time.time() - times[-1] - tm)
        
        if config.model.num_classes > 1:
            f1_ = MulticlassF1Score(num_classes=3, average='none')
            ts_sen.append(list(f1_(torch.from_numpy(preds), torch.from_numpy(targets))))

            f1_ = MulticlassF1Score(num_classes=3, average='micro')
            ts_acc.append(f1_(torch.from_numpy(preds), torch.from_numpy(targets)) * 100)
            bcm = MulticlassConfusionMatrix(3)
            cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
            cf_out += cf.numpy()
        else:
            bcm = BinaryConfusionMatrix()
            cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
            # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
            ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
            ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))
            ts_acc.append((cf[0,0]+cf[1,1])/(cf.sum()) * 100)
            cf_out += cf.numpy()
        times.append(time.time() - times[-1] - tm)
        times.append(sum(times))
        # print(f'[{subj:0>2}] acc: {ts_acc[-1]:.2f} %,  training acc: {train_acc[-1]:.2f} %,  training loss: {train_loss[-1]:.4f},  avg Acc: {np.mean(ts_acc):.2f} %,  time: {[round(t,1) for t in times]}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    if config.model.num_classes > 1:
        ts_sen = np.array(ts_sen)
        print(f'[{data_mode} {label_type}]  avg Acc: {np.mean(ts_acc):.2f} %,  std: {np.std(ts_acc):.2f},  f1: {np.mean(ts_sen[:,0])*100:.2f}  {np.mean(ts_sen[:,1])*100:.2f}  {np.mean(ts_sen[:,2])*100:.2f}, time: {time.time() - tm:.2f}')
    
        # print(f'[{data_mode} {label_type}]  avg Acc: {np.mean(ts_acc):.2f} %,  std: {np.std(ts_acc):.2f},  sen0: {np.mean(ts_sen[:,0])*100:.2f},{np.mean(ts_spc[:,0])*100:.2f},' \
        #       + f'sen1: {np.mean(ts_sen[:,1])*100:.2f},{np.mean(ts_spc[:,1])*100:.2f}, sen1: {np.mean(ts_sen[:,2])*100:.2f},{np.mean(ts_spc[:,2])*100:.2f}')
    else:
        print(f'[{data_mode} {label_type}]  avg Acc: {np.mean(ts_acc):.2f} %,  std: {np.std(ts_acc):.2f},  sen: {np.mean(ts_sen)*100:.2f},  spc: {np.mean(ts_spc)*100:.2f}, time: {time.time() - tm:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')
    if label_type == 2:
        lab = ['WG','Resting']
    elif label_type == 3:
        lab = ['DSR','Resting']
    elif label_type == 4:
        lab = ['0-back','2-back','3-back']
    # plot_confusion_matrix(cf_out, lab)


seg = 20
hid1, hid2, gr = 0, 0, 0
if __name__ == "__main__":
    with open(f"yamls/nback_best.yaml", 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
        leave_one_out_cross_validation(config,1,4)
    # plot_confusion_matrix(np.array([[172,20,42],[36,124,74],[32,56,144]],int), ['0-back','2-back','3-back'])
    # plot_confusion_matrix(np.array([[436,32],[18,450]],int), ['DSR','Resting'])
    # for seg in [1,2]:#,1]:
    #     with open(f"yamls/nback{9}.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,seg,4)
    #     with open(f"yamls/wg_best.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,seg,2)
    #     with open(f"yamls/dsr_best.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,seg,3)
    # for i in range(8):
    #     # if i in [0,1]: continue
    #     with open(f"yamls/nback{i+1}.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,0,4)
    # with open(f"yamls/dsr_best.yaml", 'r') as f:
    #     config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #     config = Box(config_yaml)
    #     leave_one_out_cross_validation(config,0,3)
    # with open(f"yamls/wg_best.yaml", 'r') as f:
    #     config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #     config = Box(config_yaml)
    #     leave_one_out_cross_validation(config,0,2)
    # for i, v in enumerate(['wg','dsr','nback']):
    #     with open(f"yamls/{v}.yaml", 'r') as f:
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,0,i+2)
    # for i in range(3):
    #     # for j in range(3):
    #     leave_one_out_cross_validation(2,i)

    # with open(f"yamls/wg_best.yaml", 'r') as f:
    #     ff = [5, 4, 2]
    #     for seg in ff:
    #     # for seg in [64,128,256,512]:
    #         print('-'*32 + str(seg))
    #         config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #         config = Box(config_yaml)
    #         leave_one_out_cross_validation(config,0,2)
    # for i, v in enumerate(['wg','dsr']):
        # ff = [(5,125,30,5), (4,128,32,4), (2,128,32,2)] if i == 0 else [(10,125,30,5), (8,128,32,4), (4,128,32,4)]
        # ff = [(5,125,30,5), (2,128,32,2)] if i == 0 else [(10,125,30,5), (4,128,32,4)]
    # ff = [(16,125,30,5), (8,128,32,2)] 
    
    # for seg in [16,8]:
    # for seg in [64,128,256,512]:
    # for seg,hid1,hid2,gr in ff:
        # with open(f"yamls/{v}_best.yaml", 'r') as f:    
    # with open(f"yamls/nback_best.yaml", 'r') as f:
    #     print('-'*32 + str(seg))
    #     config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    #     config = Box(config_yaml)
    #     # leave_one_out_cross_validation(config,0,i+2)
    #     leave_one_out_cross_validation(config,0,4)
    with open(f"yamls/nback_best.yaml", 'r') as f:
        print('-'*32 + str(seg))
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
        # leave_one_out_cross_validation(config,0,i+2)
        leave_one_out_cross_validation(config,1,4)
            # for dat_mode in [1,2]:
            #     print(v,dat_mode)
            #     leave_one_out_cross_validation(config,dat_mode,i+2)