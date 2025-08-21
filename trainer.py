import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DEVICE, EarlyStopping

OPT_DICT = {'Adam':opt.Adam,
            'AdamW':opt.AdamW,
            'SGD':opt.SGD}

def train_bin_cls(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                exlr_on:bool = False,
                **kwargs):
    criterion = nn.BCELoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    vl_correct, vl_total = 0, 0
    early_stopped = False
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = torch.squeeze(model(x))
            loss = criterion(pred, torch.squeeze(y.float()))
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int()
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        if exlr_on: exlr.step()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))

        if early_stop:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                # if epoch % 10 == 0:
                for i, data in enumerate(val_loader, 0):
                    x, y = data
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    pred = torch.squeeze(model(x))
                    predicted = (pred > 0.5).int()
                    vl_total += y.size(0)
                    vl_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y.float())
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc =round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_bin_cls(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, y in tst_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            pred = torch.squeeze(pred)
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_cls(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                **kwargs):
    criterion = nn.CrossEntropyLoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    vl_correct, vl_total = 0, 0
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
            pred = torch.squeeze(model(x, z))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(pred, 1).int()
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))

        if early_stop:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                # if epoch % 10 == 0:
                for i, data in enumerate(val_loader, 0):
                    x, z, y = data
                    x = x.to(DEVICE)
                    z = z.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    pred = torch.squeeze(model(x, z))
                    predicted = torch.argmax(pred, 1).int()
                    vl_total += y.size(0)
                    vl_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y)
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc = round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
        
    if not early_stopped:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_cls(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.empty((0, 2))
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, z, y in tst_loader:
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x, z)
            pred = torch.squeeze(pred)
            predicted = torch.argmax(pred, 1).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.concatenate([preds, pred.to('cpu').numpy()],0)
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_bin_cls2(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                exlr_on:bool = False,
                verbose_time:bool = False,
                **kwargs):
    criterion = nn.BCELoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    vl_correct, vl_total = 0, 0
    early_stopped = False
    time_total = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        start.record()
        model.train()
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, z, y = data
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = torch.squeeze(model(x, z))
            loss = criterion(pred, torch.squeeze(y.float()))
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int()
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        if exlr_on: exlr.step()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))
        end.record()
        torch.cuda.synchronize()
        time_total += [start.elapsed_time(end)]

        if early_stop:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                # if epoch % 10 == 0:
                for i, data in enumerate(val_loader, 0):
                    x, z, y = data
                    x = x.to(DEVICE)
                    z = z.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    pred = torch.squeeze(model(x, z))
                    predicted = (pred > 0.5).int()
                    vl_total += y.size(0)
                    vl_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y.float())
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc = round(100 * vl_correct / vl_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
    if verbose_time: print(f'inference time = {np.mean(time_total):.2f}')
    if not early_stopped and early_stop:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_bin_cls2(model:nn.Module, tst_loader:DataLoader):
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
            pred = model(x, z)
            pred = torch.squeeze(pred)
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_bin_cls3(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                exlr_on:bool = False,
                **kwargs):
    criterion = nn.CrossEntropyLoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [0], [0]
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    vl_correct, vl_total = 0, 0
    early_stopped = False
    # for epoch in tqdm(range(num_epoch), ncols=150):
    # import time 
    for epoch in range(num_epoch):
        model.train()
        # tr_correct, tr_total = 0, 0
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # print('start batch')
            # tm = time.time()
            x, z, y = data
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            # print(f'td: {time.time()-tm}')
            optimizer.zero_grad()
            pred = torch.squeeze(model(x, z))
            # pred = torch.softmax(pred, dim=1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # if epoch == num_epoch - 1:
            predicted = torch.argmax(pred, 1)
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        if exlr_on: exlr.step()
        # if epoch == num_epoch - 1:
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))

    #     if early_stop:
    #         with torch.no_grad():
    #             model.eval()
    #             val_loss = 0.0
    #             # if epoch % 10 == 0:
    #             for i, data in enumerate(val_loader, 0):
    #                 x, z, y = data
    #                 x = x.to(DEVICE)
    #                 z = z.to(DEVICE)
    #                 y = y.to(DEVICE)
                    
    #                 pred = torch.squeeze(model(x, z))
    #                 predicted = (pred > 0.5).int()
    #                 vl_total += y.size(0)
    #                 vl_correct += (predicted == y).sum().item()
    #                 loss = criterion(pred, y.float())
    #                 val_loss += loss.item()

    #             val_loss = round(val_loss/len(val_loader), 4)
    #             val_acc = round(100 * vl_correct / vl_total, 4)
    #             vl_loss.append(val_loss)
    #             vl_acc.append(val_acc)

    #             if epoch > min_epoch: 
    #                 if early_stop.mode == 'min':
    #                     early_stop(val_loss, epoch)
    #                 else:
    #                     early_stop(val_acc, epoch)
    #             if early_stop.early_stop:
    #                 early_stopped = True
    #                 break  
        
    # if not early_stopped and early_stop:
    #     torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def test_bin_cls3(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.array([],int)
    targets = np.array([],int)
    with torch.no_grad():
        model.eval()
        for x, z, y in tst_loader:
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x, z)
            pred = torch.squeeze(pred)
            predicted = torch.argmax(pred,1)
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,predicted.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def train_bin_cls4(model:nn.Module, 
                train_loader:DataLoader, 
                val_loader:DataLoader, 
                num_epoch:int, 
                optimizer_name:str, 
                learning_rate:str, 
                early_stop:EarlyStopping = None,
                min_epoch:int = 0,
                exlr_on:bool = False,
                **kwargs):
    criterion = nn.CrossEntropyLoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    tr_correct, tr_total = 0, 0
    # for epoch in tqdm(range(num_epoch), ncols=150):
    for epoch in range(num_epoch):
        model.train()
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = torch.squeeze(model(x))
            # pred = torch.softmax(pred, dim=1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(pred,1)
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()
            trn_loss += loss.item()
        if exlr_on: exlr.step()
        tr_loss.append(round(trn_loss/len(train_loader), 4))
        tr_acc.append(round(100 * tr_correct / tr_total, 4))

    return tr_acc, tr_loss, vl_acc, vl_loss

def test_bin_cls4(model:nn.Module, tst_loader:DataLoader):
    total = 0
    correct = 0
    preds = np.array([],int)
    targets = np.array([],int)
    with torch.no_grad():
        model.eval()
        for x, y in tst_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            pred = torch.squeeze(pred)
            predicted = torch.argmax(pred,1)
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,predicted.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets