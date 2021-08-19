import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from PyEMD import EMD, Visualisation
import Config
import glob


# decompose PM2.5 data by EMD
filePath = r'./data/Beijing_PM25_1702-1803_hourly.csv'
def decompose_PM25(filePath):
    data = pd.read_csv(filePath, index_col=False, encoding='UTF-8')
    pm25 = data.values[:, 1]
    pm25 = pm25.astype('float32')
    emd = EMD()
    imfs = emd(pm25)
    imfsarr, res = emd.get_imfs_and_residue()
    vis = Visualisation(emd)
    vis.plot_imfs()
    vis.show()
    dataframe = pd.DataFrame(imfs.T)
    dataframe.to_csv('./data/Beijing_PM25_1702-1803_hourly_imfs.csv', index=False, sep=',')
# decompose_PM25()


# meger all imf predicted data for final predicted result
def megerCSV():
    csvx_list = glob.glob(Config.PATH_PREDICT+'*.csv')
    print('Find %s CSV files' % len(csvx_list))
    data = pd.read_csv(r'./data/imfs.csv', index_col=False, encoding='UTF-8')
    observed = np.sum(data.values[-1071:-1, :], axis=1)
    dataframe = pd.DataFrame({'observed': observed[:1070]})
    for i in csvx_list:
        data = pd.read_csv(i, index_col=False, encoding='UTF-8')
        headerName = i[25:i.index('.csv')]
        dataframe[headerName] = np.array(data.iloc[-1070:, 1:2])
    dataframe['predicted'] = np.sum(dataframe.iloc[:, 1:], axis=1)
    dataframe.to_csv(Config.PATH_PREDICT+'observed.csv', index=False, sep=',')
    predict = dataframe.iloc[:, -1]
    truth = dataframe.iloc[:, 0]
    rmse = math.sqrt(mean_squared_error(predict, truth))
    mae = np.sum(np.absolute(predict - truth)) / len(truth)
    mape = np.sum(np.absolute((predict - truth) / truth) * 100 / len(truth))
    print('RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape)
# megerCSV()


# construct dataset of each imf
def getIMFdata(imf):
    WINDOW_SIZE, FEATURE_SIZE, PRE_STEP, NUM_TRAIN, NUM_VAL, BATCH_SIZE,EPOCHS = Config.getConfig(Config.VERSION)
    imf = np.array(imf)
    pm25_min = np.min(imf)
    pm25_max = np.max(imf)
    scaledImf = (imf-pm25_min)/(pm25_max-pm25_min)
    features = []
    predict = []
    for i in range(len(scaledImf) - WINDOW_SIZE - PRE_STEP):
        end_ix = i + WINDOW_SIZE
        out_end_ix = end_ix + PRE_STEP
        x = scaledImf[i:end_ix, :]
        y = scaledImf[end_ix:out_end_ix, :]
        features.append(x)
        predict.append(y)
    features = np.array(features)
    predict = np.array(predict)

    # train samples
    x_train = np.reshape(features[0:NUM_TRAIN, :], (NUM_TRAIN, WINDOW_SIZE, FEATURE_SIZE))
    y_train = np.reshape(predict[0:NUM_TRAIN], (NUM_TRAIN, -1))

    # validation samples
    x_val = np.reshape(features[NUM_TRAIN: NUM_TRAIN + NUM_VAL, :], (NUM_VAL, WINDOW_SIZE, FEATURE_SIZE))
    y_val = np.reshape(predict[NUM_TRAIN: NUM_TRAIN + NUM_VAL], (NUM_VAL, -1))

    # test samples
    NUM_TEST = len(predict) - NUM_TRAIN - NUM_VAL
    x_test = np.reshape(features[NUM_TRAIN + NUM_VAL: len(predict), :], (NUM_TEST, WINDOW_SIZE, FEATURE_SIZE))
    y_test = np.reshape(predict[NUM_TRAIN + NUM_VAL: len(predict)], (NUM_TEST, -1))

    return (x_train, y_train, x_val, y_val, y_test, x_test, pm25_min, pm25_max, NUM_TEST)


def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("R：", SSR / SST, "R-squared：", (SSR / SST) ** 2)
    return SSR / SST,(SSR / SST) ** 2


def EvaluationParam():
    data = pd.read_csv(r'./result/hourly/predicted/imf0.csv', index_col=False, encoding='UTF-8')
    truth = data.iloc[:, 0]
    predict = data.iloc[:, 1]
    rmse = math.sqrt(mean_squared_error(predict, truth))
    mae = np.sum(np.absolute(predict - truth)) / len(truth)
    mape = np.sum(np.absolute((predict - truth) / truth)) / len(truth) * 100
    print('RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape)
    computeCorrelation(predict, truth)
# EvaluationParam()