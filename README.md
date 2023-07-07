# RMVPE
## 1. 训练
数据集是我处理过的 mir1k 和 ptdb 混合数据集, 统一精度到 10ms 一帧

训练集放到`Hybrid/train`目录下

测试集放到`Hybrid/test`目录下

其中`wav`文件和`pv`文件同目录

执行训练
```bash
python train.py 
```
训练参数在文件里自己改
## 2. 可视化
```bash
tensorboard --logdir=runs
```
## 3. 测试
```bash
# 查看使用方法
python main.py -h 
```
