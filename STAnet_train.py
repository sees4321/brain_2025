import numpy as np

from trainer import *
from models.stanet import STANet, make_3d_input_for_stanet
from modules import Emotion_DataModule, MIST_DataModule, MIMA_DataModule
from utils import *
from torchmetrics.classification import BinaryConfusionMatrix

def train_bin_cls_(model:nn.Module, 
                train_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                exlr_on:bool = False,
                **kwargs):
    criterion = nn.BCELoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [], []
    tr_correct, tr_total = 0, 0
    early_stopped = False
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, z, y = data
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            
            pred, _, ls = model(x,z)
            pred = torch.squeeze(pred)
            loss = criterion(pred, torch.squeeze(y.float())) + ls
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int()
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        if exlr_on: exlr.step()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))
        
    # if not early_stopped:
    #     torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss

def test_bin_cls_(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, z, y in tst_loader:
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            pred, _, _ = model(x,z)
            pred = torch.squeeze(pred)
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def emotion_classification(emotion_dataset, learning_rate, num_epochs, dat_type=0):
    ManualSeed(0)
    num_classes = 3 if dat_type > 3 else 1
    tr_acc = []
    tr_loss = []
    ts_acc = []
    ts_sen = []
    ts_spc = []
    # preds = np.zeros((num_subj,8)) # model predictions
    # targets = np.zeros((num_subj,8)) # labels
    emotion_dataset.test_idx = 0
    for subj, data_loaders in enumerate(emotion_dataset):
        train_loader, val_loader, test_loader = data_loaders

        model = STANet().to(DEVICE)

        trainer = train_bin_cls_
        tester = test_bin_cls_

        # es = EarlyStopping(model, patience=10, mode='min')
        train_acc, train_loss = trainer(model, 
                                        train_loader=train_loader, 
                                        num_epoch=num_epochs, 
                                        optimizer_name='Adam',
                                        learning_rate=str(learning_rate),
                                        exlr_on=False)
        tr_acc.append(train_acc)
        tr_loss.append(train_loss)

        # model.load_state_dict(torch.load('best_model.pth'))
        # prune.l1_unstructured(model.classifer[0], name='weight', amount=0.3)
        test_acc, preds, targets = tester(model, tst_loader=test_loader)
        ts_acc.append(test_acc)

        bcm = BinaryConfusionMatrix()
        cf = bcm(torch.from_numpy(preds), torch.from_numpy(targets))
        # cf = bcm(torch.from_numpy(np.argmax(preds,1)), torch.from_numpy(targets))
        ts_sen.append(cf[1,1]/(cf[1,1]+cf[1,0]))
        ts_spc.append(cf[0,0]/(cf[0,0]+cf[0,1]))

        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[-1]:.2f} %, training loss: {train_loss[-1]:.3f}')
        # print(f'[{subj:0>2}] acc: {test_acc} %, training acc: {train_acc[es.epoch]:.2f} %, training loss: {train_loss[es.epoch]:.3f}, val acc: {val_acc[es.epoch]:.2f} %, val loss: {val_loss[es.epoch]:.3f}, es: {es.epoch}')

    tr_acc, tr_loss = np.array(tr_acc), np.array(tr_loss)
    print(f'trn Acc: {np.mean(tr_acc[:,-1]):.2f} %, trn loss: {np.std(tr_loss[:,-1]):.3f}')
    print(f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}, sen: {np.mean(ts_sen)*100:.2f}, spc: {np.mean(ts_spc)*100:.2f}')
    # np.save('ts_acc.npy',ts_acc)
    # print('end')

def train_emotion():
    # path = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025'
    path = 'D:/KMS/data/brain_2025'
    emotion_dataset = Emotion_DataModule(path,
                                         data_mode=0,
                                         label_type=0,
                                         ica=True,
                                         start_point=60,
                                         window_len=60,
                                         num_val=0,
                                         batch_size=16,
                                         transform_eeg=make_3d_input_for_stanet,
                                         transform_fnirs=make_3d_input_for_stanet)
    for label_type in [0,1]:
        emotion_dataset.change_label(label_type)
        # for set_ in [(5e-4,100,16),(5e-4,50,32),(5e-4,25,16),(5e-4,50,8),(1e-4,50,16),(1e-4,100,16),(1e-3,50,16),(1e-3,25,16)]:
        for set_ in [(5e-4,5,64),(5e-4,50,64),(1e-3,50,32),(1e-3,50,64),(5e-3,50,32),(1e-2,50,32),]:
            print(label_type, set_)
            learning_rate, num_epochs, batch_size = set_
            emotion_dataset.change_batch_size(batch_size)
            emotion_classification(emotion_dataset, learning_rate, num_epochs)
    # 0 (0.001, 50, 16) avg Acc: 60.76 %, std: 11.47, sen: 55.56, spc: 65.97
    # 1 (0.0005, 50, 32) avg Acc: 60.42 %, std: 14.58, sen: 45.14, spc: 75.69
    # print()

def train_stress():
    # path = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025'
    path = 'D:/KMS/data/brain_2025'
    emotion_dataset = MIST_DataModule(path,
                                        data_mode=0,
                                        start_point=0,
                                        window_len=60,
                                        num_val=0,
                                        batch_size=16,
                                        transform_eeg=make_3d_input_for_stanet,
                                        transform_fnirs=make_3d_input_for_stanet)
    # for set_ in [(5e-4,100,16),(5e-4,50,32),(5e-4,25,16),(5e-4,50,8),(1e-4,50,16),(1e-4,100,16),(1e-3,50,16),(1e-3,25,16)]:
    for set_ in [(5e-4,50,32),(5e-4,50,64),(1e-3,50,32),(1e-3,50,64),(5e-3,50,32),(1e-2,50,32)]:
        print('-'*32, set_)
        learning_rate, num_epochs, batch_size = set_
        emotion_dataset.change_batch_size(batch_size)
        emotion_classification(emotion_dataset, learning_rate, num_epochs)

def train_MIMA(label_type):
    # path = 'D:/One_한양대학교/private object minsu/coding/data/brain_2025'
    path = 'D:/KMS/data/brain_2025'
    dataset = MIMA_DataModule(path,
                            data_mode=0,
                            label_type=label_type,
                            num_val=0,
                            batch_size=16,
                            transform_eeg=1,
                            transform_fnirs=1)
    
    for set_ in [(5e-4,100,16),(5e-4,50,32),(5e-4,50,64),(5e-4,50,16),(1e-4,50,32),(1e-4,100,32),(1e-3,50,32),(1e-3,100,32)]:
    # for set_ in [(5e-4,50,32),(5e-4,50,64),(1e-3,50,32),(1e-3,50,64),(5e-3,50,32),(1e-2,50,32)]:
        print('-'*32, set_)
        learning_rate, num_epochs, batch_size = set_
        dataset.change_batch_size(batch_size)
        emotion_classification(dataset, learning_rate, num_epochs)

if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("error", category=RuntimeWarning) # 모든 RuntimeWarning을 예외로 처리
    # train_stress()
    train_MIMA(2)
    train_MIMA(3)
    train_MIMA(4)
    # for i in [2,3,4]:
    #     print('-'*50, i)
    #     train_MIMA(i)