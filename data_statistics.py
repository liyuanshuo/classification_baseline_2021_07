import functools
import os
import time
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image


def cost_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_start = time.time()
        res = func(*args, **kw)
        time_end = time.time()
        print(f'{func.__name__} use {(time_end - time_start):.4f} seconds')
        return res

    return wrapper

# 首先是统计各个类别的图像数目
@cost_time
def label_statistics(path):
    labels = os.listdir(path)
    for label in labels:
        num = len(os.listdir(os.path.join(path, label)))
        print("| {} | {} |".format(label, num))

def rgb_loader(path: str):
    '''
    部分图像文件有问题
    添加异常处理，把有问题的图像路径打印出来，并将结果置为None
    '''
    try:
        res = Image.open(path).convert('RGB')
    except:
        print("error file path : {}.".format(path))
        res = None
    return res

@cost_time
def get_mean_std_stat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=int(os.cpu_count()/2),
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        # rgb_loader中将异常图像置为了None，这里也需要加上异常处理，遇到异常就跳过
        try:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        except:
            continue
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    path = 'F:/360Downloads/PavementData2107/'
    label_statistics(path)

    dataset = ImageFolder(root=path, transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), loader=rgb_loader)
    mean_std = get_mean_std_stat(dataset)
    print(mean_std)



