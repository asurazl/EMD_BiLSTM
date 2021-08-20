import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math


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


def EvaluationParam(filePath):
    data = pd.read_csv(filePath, index_col=False, encoding='UTF-8')
    truth = data.iloc[:, 0]
    predict = data.iloc[:, 1]
    rmse = math.sqrt(mean_squared_error(predict, truth))
    mae = np.sum(np.absolute(predict - truth)) / len(truth)
    mape = np.sum(np.absolute((predict - truth) / truth)) / len(truth) * 100
    print('RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape)
    computeCorrelation(predict, truth)


if __name__ == '__main__':
    # observed and predicted PM2.5 data file
    filePath = r'./result/hourly/pm_predicted.csv'

    EvaluationParam(filePath)