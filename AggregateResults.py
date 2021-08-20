import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import Config
import glob


# meger all imfs predicted data for final predicted result
def megerCSV(filePath, imf_savePath, pm_savePath):
    csvx_list = glob.glob(filePath + '*.csv')
    print('Find %s CSV files' % len(csvx_list))
    data = pd.read_csv(r'./data/Beijing_PM25_1702-1803_hourly_imfs.csv', index_col=False, encoding='UTF-8')
    observed = np.sum(data.values[-1071:-1, :], axis=1)
    dataframe = pd.DataFrame({'observed': observed[:1070]})
    for i in csvx_list:
        data = pd.read_csv(i, index_col=False, encoding='UTF-8')
        headerName = i[25:i.index('.csv')]
        dataframe[headerName] = np.array(data.iloc[-1070:, 1:2])
    dataframe['predicted'] = np.sum(dataframe.iloc[:, 1:], axis=1)
    dataframe.to_csv(imf_savePath, index=False, sep=',')
    predict = np.abs(dataframe.iloc[:, -1])
    truth = dataframe.iloc[:, 0]
    dataframe_pm = pd.DataFrame({'observed': truth, 'predicted': predict})
    dataframe_pm.to_csv(pm_savePath, index=False, sep=',')
    rmse = math.sqrt(mean_squared_error(predict, truth))
    mae = np.sum(np.absolute(predict - truth)) / len(truth)
    mape = np.sum(np.absolute((predict - truth) / truth) * 100 / len(truth))
    print('RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape)


if __name__ == '__main__':

    # imf predicted results: Config.PATH_PREDICT
    filePath = Config.PATH_PREDICT
    # aggregate all imf results
    imf_savePath = r'./result/' + Config.VERSION + '/' + 'imf_predicted.csv'
    # finall pm2.5 predicted result
    pm_savePath = r'./result/' + Config.VERSION + '/' + 'pm_predicted.csv'

    megerCSV(filePath, imf_savePath, pm_savePath)
