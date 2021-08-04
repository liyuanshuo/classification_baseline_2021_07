import os
import shutil
from tqdm import tqdm

import time
import functools
from PIL import Image

def cost_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_start = time.time()
        res = func(*args, **kw)
        time_end = time.time()
        print(f'{func.__name__} use {(time_end - time_start):.4f} secends')
        return res

    return wrapper

origin_path = 'E:/Work/Data2107/'

target_path = 'F:/360Downloads/PavementData2107/'

image_type = ['.jpg', '.JPG', '.png', '.PNG']

def path_exist(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

@cost_time
def single_dir_move(origin_dir_path, target_dir_path):
    path_exist(target_dir_path)
    for root, dirs , files in tqdm(list(os.walk(origin_dir_path))):
        for file in files:
            if os.path.splitext(file)[-1] in image_type:
                try:
                    Image.open(os.path.join(root, file)).load()
                except:
                    continue
                shutil.copy(os.path.join(root, file), os.path.join(target_dir_path, file))
    print("Move '{}' Data To '{}' Finished By Function".format(origin_dir_path, target_dir_path), end=' ')

@cost_time
def move_data(origin_path:str, target_path:str):
    path_exist(target_path)
    labels = os.listdir(origin_path)
    for label in tqdm(labels):
        label_origin_path = os.path.join(origin_path, label)
        lebel_target_path = os.path.join(target_path, label)
        single_dir_move(label_origin_path, lebel_target_path)

if __name__ == '__main__':
    move_data(origin_path, target_path)