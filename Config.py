import os

# public parameter
DATA_PATH = r'./data/Beijing_PM25_1702-1803_hourly_imfs.csv'
IMF_NUM = 0
VERSION = 'hourly'
LEARNING_RATE = 1e-4
DELETE_TRAIN = True
DELETE_TEST = True

PATH_LOGS = './result/' + VERSION + '/logs/'
PATH_CKPT = './result/' + VERSION + '/trained/'
PATH_PREDICT = './result/' + VERSION + '/predicted/'

# creat folder for created version
def creatFolder():
    if DELETE_TRAIN:
        path_logs = PATH_LOGS
        path_ckpt = PATH_CKPT
        path_predict = PATH_PREDICT
        if os.path.exists(path_logs) is False:
            os.makedirs(path_logs)
        if os.path.exists(path_ckpt) is False:
            os.makedirs(path_ckpt)
        if os.path.exists(path_predict) is False:
            os.makedirs(path_predict)

# NUM_TRAIN and NUM_VAL given for experiment convenience
# Modify them under your needs
# imf_0: WINDOW_SIZE=4 ,others: WINDOW_SIZE=8
def getConfig(version):
    if version == 'hourly':
        if IMF_NUM == 0:
            WINDOW_SIZE = 4
        else:
            WINDOW_SIZE = 8
        FEATURE_SIZE = 1
        PRE_STEP = 1
        BATCH_SIZE = 64
        EPOCHS = 500
        NUM_TRAIN = 6912
        NUM_VAL = 1080

    if version == 'daily':
        if IMF_NUM == 0:
            WINDOW_SIZE = 4
        else:
            WINDOW_SIZE = 8
        FEATURE_SIZE = 1
        PRE_STEP = 1
        BATCH_SIZE = 64
        EPOCHS = 500
        NUM_TRAIN = 1200
        NUM_VAL = 300

    return (WINDOW_SIZE, FEATURE_SIZE, PRE_STEP, NUM_TRAIN, NUM_VAL, BATCH_SIZE, EPOCHS)
