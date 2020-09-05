# YOLO_football

### 此项目是基于darknet YOLO开发足球视频分析模型，其中包含4000多张标注过的球员与足球的训练数据集，以及预训练模型。（原自己的商业项目第一版）
### This project is based on darknet YOLO to develop a football video analysis model, which contains more than 4000 annotated training data sets of players and footballs, and pre-training models。

### B站视频地址
- https://www.bilibili.com/video/BV17a4y1j72p
- https://www.bilibili.com/video/BV1gK411K7uw

## 环境(Requirements)
```参考darknet caffe配置 https://pjreddie.com/darknet/yolo/```

## 例子🌰(Demo)
```./darknet deterctor test cfg/voc.data yolo.cfg backup/MODELNAME.werghts VIDEOPATH```

## 训练(train)
```./darknet deterctor train cfg/voc.data yolo.cfg backup/MODELNAME.werghts -gpu 0```

## 数据集(dataset)

- 链接: https://pan.baidu.com/s/15JGCpAlMLRSY9dsfurSTWQ 提取码: bjex 
![dataset](https://github.com/tommyMessi/YOLO_football/blob/master/image/data_img.png)
- 网盘为原数据，需要自行转caffe需要的格式。

## 可视化实例
### 例子🌰1
![img1](https://github.com/tommyMessi/YOLO_football/blob/master/image/predictions_20200906_025432.jpg)
![img2](https://github.com/tommyMessi/YOLO_football/blob/master/image/predictions.jpg)

## 其他
如果需要整理好的caffe格式集与模型 或者有合作需要 关注微信公众账号 hulugeAI 留言：foolball
