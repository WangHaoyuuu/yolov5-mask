## 相关依赖包的安装
``` pip install -r requirements.txt ```

数据集为收集的不同的数据集统一整理后得到的

源码包含了yolov5的部分使用到的源码

## GUI
visual.py为自创GUI文件，直接运行即可


## 是否启用GPU
使用之前可以改动是否启用GPU进行预测，

将self.device = 'cpu'中'cpu'改成'0'
即可使用GPU进行预测。

## 权重位置
训练好的权重文件位于
runs/train/exp3/weights/best.pt

如果不想使用GUI可以直接运行detect.py文件。