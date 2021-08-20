import numpy as np
import math
import Config


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
