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


## 2、模型fine-tune的实验结果

### 2.1 googlenet

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack|0.595408895|0.506715507|0.547493404|819|
|mend|0.764011799|0.805181347|0.784056509|965|
|normal|0.801822323|0.904109589|0.849899396|1168|
|transverse_crack|0.623188406|0.396313364|0.484507042|217|
|accuracy|0.73650994|

### 2.2 resnet

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack|	0.61815562|	0.523809524|	0.567085261|	819|
|mend	|0.759813084|	0.842487047|	0.799017199|	965|
|normal|	0.786786787|	0.897260274	|0.8384	|1168|
|transverse_crack|	0.657534247|	0.221198157	|0.331034483|	217|
|accuracy	|0.737772168	|


### 2.3 VGG16

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack	|0.816993464|	0.610500611|	0.69881202|	819|
|mend	|0.796107507|	0.89015544	|0.840508806|	965|
|normal	|0.902479339|	0.934931507	|0.918418839|	1168|
|transverse_crack|	0.597014925|	0.737327189|	0.659793814|	217|
|accuracy|	0.823919217	|


### 2.4 VGG19

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|transverse_crack|	0.682471264|	0.648021828|	0.66480056|	733|
|normal	|0.823076923|	0.842519685|	0.832684825|	1016|
|mend	|0.905737705|	0.917774086|	0.911716172|	1204|
|longitudinal_crack|	0.647887324|	0.638888889|	0.643356643|	216|
|accuracy	|0.81224361|



### 2.5 efficientnet_v2_b3


|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack	|0.712793734|	0.666666667|	0.688958991|	819|
|mend	|0.817073171	|0.833160622|	0.825038481|	965|
|normal	|0.866452132	|0.922089041|	0.893405226|	1168|
|transverse_crack	|0.681818182	|0.552995392|	0.610687023|	217|
|accuracy	|0.803723572	|




### 2.6 alexnet

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
| transverse_crack| 	0.593150685| 	0.555840822| 	0.573889993| 	779|
| normal| 	0.796203796| 	0.828482328| 	0.812022415|	962|
| mend| 	0.779648609	| 0.8875|	0.830085737|	1200|
| longitudinal_crack| 	0.555555556|	0.175438596|	0.266666667|	228|
| accuracy	| 0.736825497	|


### 2.7 vit_small_patch16_224

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|transverse_crack|	0.627737226|	0.42002442|	0.50329188|	819|
|normal	|0.745318352|	0.824870466	|0.783079193	|965|
|mend|	0.741331484	|0.915239726|	0.819157088	|1168|
|longitudinal_crack	|0.414414414|	0.211981567	|0.280487805|	217|
|accuracy|	0.71158094|




### 2.8 cspdarknet53

|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack	|0.69571231	|0.662714097|	0.678812416	|759|
|mend	|0.829847909	|0.874749499	|0.851707317|	998|
|normal	|0.900854701	|0.875415282	|0.887952822|	1204|
|transverse_crack	|0.65625|	0.706730769|	0.680555556|	208|
|accuracy	|0.813190281|



|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack|	0.772823779	|0.479578393|	0.591869919|	759|
|mend	|0.753016895	|0.937875752|	0.835341365	|998|
|normal	|0.855495772	|0.924418605|	0.888622754	|1204|
|transverse_crack|	0.701298701	|0.519230769	|0.596685083	|208|
|accuracy	|0.795519091	|




### 2.9 repvgg


|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack	|0.635951662	|0.591292135	|0.612809316	|712|
|mend	|0.799418605	|0.818452381|	0.808823529	|1008|
|normal	|0.883003953	|0.921617162	|0.901897457|	1212|
|transverse_crack	|0.547619048|	0.485232068|	0.514541387|	237|
|accuracy	|0.781950142|



### 2.10 efficientnet_v2_b1


|类别|precision|recall|f1-score|support|
| :----: | :----: |:----: |:----: |:----: |
|longitudinal_crack|	0.746143058|	0.64957265|	0.694516971	|819|
|mend|	0.812121212	|0.833160622	|0.822506394	|965|
|normal|	0.871544715	|0.917808219|	0.894078399	|1168|
|transverse_crack	|0.580508475	|0.631336406	|0.604856512|	217|
|accuracy|	0.803092458	|


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