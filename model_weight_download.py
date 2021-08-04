import torch
import os
import torchvision.models as models


weight_save_path = './weight/'

def path_exist(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def vgg_weight():
    vgg16 = models.vgg16(pretrained=True)
    torch.save(vgg16, os.path.join(weight_save_path, vgg16._get_name().lower()+'.pth'))

def googlenet_weight():
    googlenet = models.googlenet(pretrained=True)
    torch.save(googlenet, os.path.join(weight_save_path, googlenet._get_name().lower()+'.pth'))

def resnet18_weight():
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18, os.path.join(weight_save_path, resnet18._get_name().lower()+'.pth'))

if __name__ == '__main__':
    path_exist(weight_save_path)
    vgg_weight()
    googlenet_weight()
    resnet18_weight()
