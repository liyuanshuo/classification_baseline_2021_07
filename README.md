# 可行性验证(Baseline)，基于2021年07月提供的路面病害数据集


## 1、整理分类的数据集（暂且作为单标签的多分类）

将数据集进行规范化，方便直接利用pytorch的ImageFolder直接构造训练集、测试集

代码文件`move_data.py`中将原始的数据集文件移动到一个新的目录下，并将同一个类别标签的图像放在同一个文件夹中

代码文件`data_statistics.py`来统计数据分布信息

| 类别名称 | 图像数目 |
| :----: | :----: |
| 1横向裂缝 | 1077 |
| 2纵向裂缝 | 3890 |
| 3修补 | 4882 |
| 4块状裂缝 | 7 |
| 5松散 | 5 |
| 6波浪拥包 | 1 |
| 7坑槽 | 1 |
| 无病害 | 43976 |

对于数据集的归一化需要计算灰度均值和方差，公开数据集的ImageNet为mean=(0.485, 0.456, 0.406)，std=(0.229, 0.224, 0.225)，但是这个ImageNet只能作为参考。

([0.22442342, 0.22442342, 0.22442342], [0.079417765, 0.079417765, 0.079417765])

**对于4、5、6、7四类病害的数据太少了，这里分类Baseline暂且不考虑这四类**

**同时对于无病害类别的数目太多了，为了降低训练成本，这里从无病害类别中随机选择1W张图像参与训练与测试**

**对于训练集，验证集，测试集的划分，由于选择的总数不到2W，因此选择 6:2:2 的比例进行划分**

代码文件`clear_normal.py`随机从无病害类别的4W张图像中，随机抽样出6K张保存到`normal`文件夹中

之后对文件夹重新命名一下，并去除4、5、6、7四类病害，最后得到的数据分析信息为：

| 类别名称 | 图像数目 |
| :----: | :----: |
| longitudinal_crack | 3890 |
| mend | 4882 |
| normal | 6000 |
| transverse_crack | 1077 |


----------------

GitHub出现的一些小问题解决办法;

error: Failed to connect to github.com port 443: Timed out


尝试两条命令：

`git config --global --unset http.proxy`

`git config --global --unset https.proxy`


## 数据上传云端方式

1. 压缩后上传到微软的学生OneDrive云盘中，生成一个共享链接，比如`https://stnuceducn-my.sharepoint.com/:u:/g/personal/1405024239_st_nuc_edu_cn/EbwL4VAWuoFDrz_156GCmwkBGd5vUcCu7BXOgivT5mU9Jw?e=RmyxKF`
2. 在第一步获取到的共享链接后面加上`download=1`就可以拼接成一个完整的下载链接
3. 在notebook中直接利用wget命令`!wget "https://stnuceducn-my.sharepoint.com/:u:/g/personal/1405024239_st_nuc_edu_cn/EbwL4VAWuoFDrz_156GCmwkBGd5vUcCu7BXOgivT5mU9Jw?e=RmyxKF&download=1" -O dataset.zip`
4. 获取下载后，解压即可，`!unzip baseline.zip > /dev/null 2>&1`
5. 记得删掉压缩包，节约线上的存储空间`rm -rf dataset.zip`