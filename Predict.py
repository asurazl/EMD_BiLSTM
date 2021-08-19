import keras
import math
import Config
import MyModel
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import DataProcess
# Public Parameter
DATA_PATH = Config.DATA_PATH
LEARNING_RATE = Config.LEARNING_RATE
DELETE_TRAIN = Config.DELETE_TRAIN
DELETE_TEST = Config.DELETE_TEST

# test different parameter
VERSION = Config.VERSION
# load param
WINDOW_SIZE, FEATURE_SIZE, PRE_STEP, NUM_TRAIN, NUM_VAL, BATCH_SIZE, EPOCHS = Config.getConfig(VERSION)
# load data
data = pd.read_csv(DATA_PATH, index_col=False, encoding='UTF-8')

imfNum = Config.IMF_NUM
ckpt_path = Config.PATH_CKPT + 'imf_' + str(imfNum)+'/weights.350-0.00.h5'
imf = np.array(data.iloc[1:, imfNum:imfNum+1])
x_train, y_train, x_val, y_val, y_test, x_test, pm25_min, pm25_max, NUM_TEST = DataProcess.getIMFdata(imf)
# print samples
print('Number train samples: ' + str(NUM_TRAIN))
print('Number validation samples: ' + str(NUM_VAL))
print('Number test samples: ' + str(NUM_TEST))
# input shape
input_shape = (WINDOW_SIZE, FEATURE_SIZE)
# create model
model = MyModel.getModel(VERSION, input_shape)
# compile model with adam optimizer
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.mean_squared_error)
# load weights
model.load_weights(ckpt_path)
# evaluate
MSE_loss = model.evaluate(x_test, y_test, verbose=1, batch_size=BATCH_SIZE)
print('MSE loss: ' + str(MSE_loss), ckpt_path[-18:])
# predict
test_predict = model.predict(x_test, verbose=1, batch_size=BATCH_SIZE)
# reshape
test_predict = np.reshape(test_predict, (NUM_TEST, -1))
pm25_truth = np.reshape(y_test, (NUM_TEST, -1))
# become original temperature
predict = np.array(test_predict) * (pm25_max - pm25_min) + pm25_min
truth = np.array(pm25_truth) * (pm25_max - pm25_min) + pm25_min

# saving compare data to csv file
# dataframe = pd.DataFrame({'pm25imf_truth': truth[:, 0],
#                           'pm25imf_predict': predict[:, 0]})
# resultPath = Config.PATH_PREDICT + 'imf'+str(imfNum) + '.csv'
# dataframe.to_csv(resultPath, index=False, sep=',')

# EvaluationParam RMSE  MAE  MAPE
print('Evaluation Parameters in predict imf ' + str(imfNum-1) + ':')
rmse = math.sqrt(mean_squared_error(predict, truth))
mae = np.sum(np.absolute(predict - truth)) / len(truth)
mape = np.sum(np.absolute((predict - truth) / truth) * 100 / len(truth))
print('RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape)
DataProcess.computeCorrelation(predict, truth)




