import os
import shutil
import keras
import numpy as np
import pandas as pd
import DataProcess
import Config
import MyModel

# public param
DATA_PATH = Config.DATA_PATH
LEARNING_RATE = Config.LEARNING_RATE
DELETE_TRAIN = Config.DELETE_TRAIN
DELETE_TEST = Config.DELETE_TEST
Config.creatFolder()
# for version param control
VERSION = Config.VERSION
# get param
WINDOW_SIZE, FEATURE_SIZE, PRE_STEP, NUM_TRAIN, NUM_VAL, BATCH_SIZE, EPOCHS = DataProcess.getConfig(VERSION)
# get dataset
data = pd.read_csv(DATA_PATH, index_col=False, encoding='UTF-8')

imfNum = Config.IMF_NUM
# creat folder for imf
if DELETE_TRAIN:
    path_logs = Config.PATH_LOGS + 'imf_' + str(imfNum) + '/'
    path_ckpt = Config.PATH_CKPT + 'imf_' + str(imfNum) + '/'
    if os.path.exists(path_logs):
        shutil.rmtree(path_logs)
    os.makedirs(path_logs)
    if os.path.exists(path_ckpt):
        shutil.rmtree(path_ckpt)
    os.makedirs(path_ckpt)
imf = np.array(data.iloc[1:, imfNum:imfNum + 1])
x_train, y_train, x_val, y_val, y_test, x_test, pm25_min, pm25_max, NUM_TEST = DataProcess.getIMFdata(imf)
# print dataset num
print('Number train samples: ' + str(NUM_TRAIN))
print('Number validation samples: ' + str(NUM_VAL))
print('Number test samples: ' + str(NUM_TEST))
# model param
input_shape = (WINDOW_SIZE, FEATURE_SIZE)
# creat model
model = MyModel.getModel(VERSION, input_shape)
# compile model
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.mean_squared_error)
# call_back data
call_back_tensorboard = keras.callbacks.TensorBoard(log_dir=path_logs, histogram_freq=20, batch_size=BATCH_SIZE,
                                                    write_graph=True, write_grads=True,
                                                    write_images=False, embeddings_freq=0,
                                                    embeddings_layer_names=None, embeddings_metadata=None)
call_back__ckpt = keras.callbacks.ModelCheckpoint(path_ckpt + 'weights.{epoch:02d}-{val_loss:.2f}.h5',
                                                  monitor='val_loss', verbose=1, save_best_only=True,
                                                  save_weights_only=False, mode='auto', period=5)
# train
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
          shuffle=True, validation_data=(x_val, y_val),
          callbacks=[call_back_tensorboard, call_back__ckpt])
