import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report
import pandas as pd

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)
    f.close()


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
    f.close()
    return info_list

def rgb_loader(path: str):
    return Image.open(path).convert('RGB')

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数目
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.6f}, acc: {:.6f}.".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print("WARNING: no-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数目
    accu_num = torch.zeros(1).to(device)

    sample_num = 0

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[validation epoch {}] loss: {:.6f}, acc: {:.6f}.".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def test_model(model, data_loader, device, epoch, class_labels:list):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数目
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    test_result = {'labels': np.array([], dtype='u1'), 'preds': np.array([], dtype='u1')}
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        test_result['labels'] = np.concatenate((test_result['labels'], np.array(labels, dtype='u1')))
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        test_result['preds'] = np.concatenate((test_result['preds'], np.array(pred_classes.cpu(), dtype='u1')))

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.6f}, acc: {:.6f}.".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)
        result = classification_report(test_result['labels'], test_result['preds'], target_names=class_labels, zero_division=0, output_dict=True, digits=6)
        # print(classification_report(test_result['labels'], test_result['preds'], target_names=class_labels, zero_division=0, digits=6))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, pd.DataFrame(result).transpose()