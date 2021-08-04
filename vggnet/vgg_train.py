import warnings
import os
import math
import pandas as pd
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

warnings.filterwarnings('ignore')

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter



from utils import rgb_loader, train_one_epoch, evaluate, test_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def dataset_split(dataset, split_rate:float = 0.2):
    dataset_len = len(dataset)
    lengths = [dataset_len - int(dataset_len * split_rate), int(dataset_len * split_rate)]
    # 设置seed，确保随机结果的可复现性
    left_dataset, split_dataset = random_split(dataset=dataset, lengths=lengths, generator=torch.Generator().manual_seed(10))
    return left_dataset, split_dataset

def dataset_split_by_path(data_path, transform, loader=rgb_loader, split_rate:float = 0.2):
    dataset = ImageFolder(root=data_path, transform=transform, loader=loader)
    return dataset_split(dataset, split_rate)


def generate_class_json(root_path:str):
    # 遍历文件夹，一个文件夹对应一个类别
    number_class = [
        cla for cla in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, cla))
    ]
    # 排序，保证顺序一致性
    number_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(number_class))
    json_str = json.dumps(dict(
        (val, key) for key, val in class_indices.items()),indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)
    json_file.close()


def model_modify(model, num_classes:int):
    model_name = model._get_name().lower()
    if model_name == 'resnet':
        features = model.fc.in_features
        model.fc = nn.Linear(features, num_classes, bias=True)
    elif model_name == 'vgg':
        features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(features, num_classes, bias=True)
    elif model_name == 'googlenet':
        features = model.fc.in_features
        model.fc = nn.Linear(features, num_classes, bias=True)
    else:
        raise Exception("Model `{}` modify code does not exits!, Must Change Code of Function `model_modify`!".format(model_name))
    return model


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    
    if os.path.exists('../weight') is False:
        os.makedirs("../weight")
    
    tb_writer = SummaryWriter(args.log_path)
    num_workers = args.nw if args.nw > 0 else int(os.cpu_count()/2)
    batch_size = args.batch_size
    labels = os.listdir(args.data_path)
    test_dataframe = pd.DataFrame()

    # 准备数据集
    train_dataset, test_dataset = dataset_split_by_path(args.data_path, transform=data_transform, loader=rgb_loader, split_rate=0.2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    # 准备好模型
    model = torch.load(args.weight)
    model = model_modify(model, args.num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x : ((1+math.cos(x*math.pi/args.epoch))/2)*(1-args.lrf) + args.lrf # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epoch):
        epoch_train_dataset, epoch_val_dataset = dataset_split(train_dataset, 0.2)
        train_dataloader = DataLoader(dataset = epoch_train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        val_dataloader = DataLoader(dataset=epoch_val_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            data_loader = val_dataloader,
            device = device,
            epoch = epoch
        )

        # test
        test_loss, test_acc, epoch_test_dataframe = test_model(
            model=model,
            data_loader = test_dataloader,
            device = device,
            epoch = epoch,
            class_labels = labels
        )

        test_dataframe = test_dataframe.append(pd.DataFrame(epoch_test_dataframe))
        test_dataframe.to_csv('./' + model._get_name().lower() + "_test_result.csv", index=True)

        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], test_loss, epoch)
        tb_writer.add_scalar(tags[5], test_acc, epoch)
        tb_writer.add_scalar(tags[6], optimizer.param_groups[0]['lr'], epoch)

        # 保存每轮训练完之后的模型权重
        torch.save(model, os.path.join(args.save_path, model._get_name().lower() + '_epoch_' + str(epoch) + '_.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在目录
    parser.add_argument('--data_path', type=str, default='F:/360Downloads/PavementData2107/')

    # 预训练权重
    parser.add_argument('--weight', type=str, default='./vggnet/vgg.pth', help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./weight/')
    parser.add_argument('--log_path', type=str, default='./log/vgg')

    parser.add_argument('--device', default='cuda', help='device id(i.e. 0 or 0, 1 or cpu)')

    opt = parser.parse_args()
    main(opt)