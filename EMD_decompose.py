import pandas as pd
from PyEMD import EMD, Visualisation


# decompose PM2.5 data by EMD
def decompose_PM25(filePath, savePath):
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
    dataframe.to_csv(savePath, index=False, sep=',')


if __name__ == '__main__':
    # origianl PM2.5 data path
    filePath = r'./data/Beijing_PM25_1702-1803_hourly.csv'
    # decomposed PM2.5 data path
    savePath = r'./data/Beijing_PM25_1702-1803_hourly_imfs_v1.csv'

    decompose_PM25(filePath, savePath)

