from sklearn.externals import joblib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import gdal
from sklearn.model_selection import train_test_split

def read_info(file):
    dem = gdal.Open(file)
    col = dem.RasterXSize
    row = dem.RasterYSize
    band = dem.RasterCount
    geotransform = dem.GetGeoTransform()
    proj = dem.GetProjection()
    data = np.zeros([row, col, band])
    for i in range(band):
        sing_band = dem.GetRasterBand(i+1)
        data[:,:,i] = sing_band.ReadAsArray()
    return col, row, geotransform, proj, data

def save_tif(data, ouput_file, geotransform, proj):
    row, col = data.shape
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(ouput_file, col, row, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform(geotransform)
    outRaster.SetProjection(proj)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outband.SetNoDataValue(-1)
    outRaster.FlushCache()
    del outRaster

def get_modelling_data(landslide_index, seed):
    np.random.seed(seed)
    landslide_train_index, landslide_test_index = train_test_split(landslide_index, test_size=0.29999, shuffle=True)
    return  landslide_train_index, landslide_test_index

def get_data(data_index, all_data):
    data = []
    for i in range(len(data_index)):
        single_data = all_data[data_index[i][0], data_index[i][1]]
        data.append(round(single_data, 10))
    return np.array(data)

def get_auc(landslide_index, nonlandslide_index, prob):
    landslide_prob = get_data(landslide_index, prob)
    nonlandslide_prob = get_data(nonlandslide_index, prob)
    data_prob = np.append(landslide_prob, nonlandslide_prob)
    # print(data_prob)
    data_predict = []
    for i in data_prob:
        if i >= 0.5:
            data_predict.append(1)
        else:
            data_predict.append(0)
    data_true = np.append(np.ones(len(landslide_prob), dtype=int), np.zeros(len(nonlandslide_prob), dtype=int))
    auc = metrics.roc_auc_score(data_true,data_prob)
    acc = metrics.accuracy_score(data_true, data_predict)
    print("acc",acc)
    return auc

def get_cum_area_neg(landslide_index, prob):
    landslide_prob = get_data(landslide_index, prob)
    for i in range(len(landslide_prob)):
        if landslide_prob[i] == -1:
            landslide_prob[i] = 0.9

    #将概率值排序，并去除None值
    prob_1D = np.sort(prob, axis=None)
    prob_1D = prob_1D[::-1]
    prob_1D = filter(lambda a: a != -1.0, prob_1D[::-1])
    prob_1D = np.array([i for i in prob_1D])
    interval_value = []

    #按10%为间隔得到基于整个研究区易发性值得间隔
    for i in range(5053, len(prob_1D), 5053):
        interval_value.append(round(prob_1D[i], 10))
    # interval_value.append(0)
    print(len(interval_value), interval_value)


    curve_axis = np.zeros([99, 2])
    #ROC曲线的横坐标就是易发性值的间隔
    x_value = [i/100.0 for i in range(1,100)]
    # print(x_value)
    curve_axis[:, 0] = x_value
    # curve_axis[:, 0] = 1 - np.array(interval_value)
    # print(len(landslide_prob))


    for i in range(len(interval_value)):
        count = 0
        for prob in landslide_prob:
            if prob <= interval_value[i]:
                count += 1
        curve_axis[i, 1] = count / len(landslide_prob)
    curve_axis = np.concatenate((np.array([[0,0]]), curve_axis), axis = 0)
    curve_axis = np.concatenate((curve_axis, np.array([[1,1]])), axis = 0)
    auc = 0.0
    pre_x = 0.0
    pre_y = 0.0
    # print(curve_axis)
    for x,y in curve_axis:
        auc += (x-pre_x)*y
        pre_x = x
        # print(auc)
    # np.savetxt("FR_mrf.txt", curve_axis)
    plt.scatter(curve_axis[:,0],curve_axis[:,1])
    plt.show()
    return auc

def get_cum_area(landslide_index, prob):
    landslide_prob = get_data(landslide_index, prob)
    for i in range(len(landslide_prob)):
        if landslide_prob[i] == -1:
            landslide_prob[i] = 0.9

    #将概率值排序，并去除None值
    prob_1D = np.sort(prob, axis=None)
    prob_1D = filter(lambda a: a != -1.0, prob_1D[::-1])
    prob_1D = np.array([i for i in prob_1D])
    interval_value = []
    # print(len(prob_1D))
    #按10%为间隔得到基于整个研究区易发性值得间隔
    for i in range(5053, len(prob_1D), 5053):
        interval_value.append(round(prob_1D[i], 10))
    # interval_value.append(0)
    # print(len(interval_value), interval_value)


    curve_axis = np.zeros([99, 2])
    #ROC曲线的横坐标就是易发性值的间隔
    x_value = [i/100.0 for i in range(1,100)]
    # print(x_value)
    curve_axis[:, 0] = x_value
    # curve_axis[:, 0] = 1 - np.array(interval_value)
    # print(len(landslide_prob))

    # #按10%为间隔得到基于整个研究区易发性值得间隔
    # for i in range(50530, len(prob_1D), 50530):
    #     interval_value.append(round(prob_1D[i], 10))
    # # interval_value.append(0)
    # print(len(interval_value), interval_value)
    #
    #
    # curve_axis = np.zeros([9, 2])
    # #ROC曲线的横坐标就是易发性值的间隔
    # x_value = [i/10.0 for i in range(1,10)]
    # # print(x_value)
    # curve_axis[:, 0] = x_value
    # # curve_axis[:, 0] = 1 - np.array(interval_value)
    # # print(len(landslide_prob))

    for i in range(len(interval_value)):
        count = 0
        for prob in landslide_prob:
            if prob >= interval_value[i]:
                count += 1
        curve_axis[i, 1] = count / len(landslide_prob)
    curve_axis = np.concatenate((np.array([[0,0]]), curve_axis), axis = 0)
    curve_axis = np.concatenate((curve_axis, np.array([[1,1]])), axis = 0)
    auc = 0.0
    pre_x = 0.0
    pre_y = 0.0
    # print(curve_axis)
    for x,y in curve_axis:
        auc += (x-pre_x)*y
        pre_x = x
        # print(auc)
    # np.savetxt("FR_mrf.txt", curve_axis)
    plt.plot(curve_axis[:,0],curve_axis[:,1])
    plt.scatter(curve_axis[:, 0], curve_axis[:, 1])
    plt.show()
    # return auc
    return auc, curve_axis

if __name__ == '__main__':
    landslide_test_index = np.load("landslide_test_index.npy")

    # col, row, geotransform, proj, data = read_info('SVM1.tif')
    col, row, geotransform, proj, data = read_info('AdaPU.tif')
    data = data.reshape((row, col))
    # train_auc = get_auc(landslide_train_index,nonlandslide_train_index, data)
    # test_auc = get_auc(landslide_test_index,nonlandslide_test_index, data)
    test_auc = get_cum_area(landslide_test_index, data)
    print(test_auc)