import os
import shutil
from tqdm import tqdm

import time
import functools
from PIL import Image

import random

def cost_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_start = time.time()
        res = func(*args, **kw)
        time_end = time.time()
        print(f'{func.__name__} use {(time_end - time_start):.4f} secends')
        return res

    return wrapper

origin_path = 'F:/360Downloads/PavementData2107/normal_old/'

target_path = 'F:/360Downloads/PavementData2107/normal/'


def path_exist(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


@cost_time
def clear_data(origin_path:str, target_path:str, select_num : int = 10000):
    # 设置随机种子，保证随机结果的可复现性
    random.seed(0)
    path_exist(target_path)
    files = os.listdir(origin_path)
    sample_files = random.sample(files, k=select_num)
    for file in tqdm(sample_files):
        file_origin_path = os.path.join(origin_path, file)
        file_target_path = os.path.join(target_path, file)
        try:
            Image.open(file_origin_path).load()
        except:
            continue
        shutil.copy(file_origin_path, file_target_path)


if __name__ == '__main__':
    clear_data(origin_path, target_path, 6000)