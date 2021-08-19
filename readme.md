Based on [Air Quality Predictions with a Semi-Supervised Bidirectional LSTM Neural Network](https://www.doi.org/10.1016/j.apr.2020.09.003) by Zhang et al.

require：\
tensorflow 1.13.1 or later\
keras 2.2.4 or later\
pyemd 0.5.1

./data/xxx存放原始和经过预处理的PM2.5序列数据;\
./result/xxx存放模型相关结果数据

1.Dataprocess.py中decompose_PM25对原始数据进行EMD分解

2.Config.py中修改参数进行不同数据实验

version控制不同版本测试，imf_num对应不同分解量，

Train.py对不同imf数据进行训练，Predict.py对相应imf进行预测

3.Dataprocess.py中mergeCSV合并成最终的PM2.5预测结果.