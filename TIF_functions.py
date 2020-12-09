import gdal
import numpy as np
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy as np
from common_func import evaluate_method

def read_tif(file):
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

def get_sus_map(imgs,model):
    # lenth=len(imgs)
    row=imgs.shape[0]
    col=imgs.shape[1]
    imgs=np.reshape(imgs,[ row*col, imgs.shape[2]])

    proba=model.predict_proba(imgs)
    proba=proba[:,1]
    proba=np.reshape(proba,[row,col])
    return proba

def save_tif(data, ouput_file, geotransform, proj):
    row,col = data.shape
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(ouput_file, col, row, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform(geotransform)
    outRaster.SetProjection(proj)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outband.SetNoDataValue(-1)
    outRaster.FlushCache()
    del outRaster

def get_suscep_DL_1D(factor_file, ouput_file, model):
    col, row, geotransform, proj, data = read_tif(factor_file)
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(ouput_file, col, row, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform(geotransform)
    susceptibility = np.zeros([row, col])

    #get each grid cell susceptibility
    for i in range(row):
        for j in range(col):
            single_data = data[i, j, :]
            if single_data[0] == 0:
                susceptibility[i, j] = 0
            else:
                single_data = single_data.reshape(1, len(single_data))
                single_data = np.expand_dims(single_data, axis=2)
                y_single_probility = model.predict(single_data)
                y_single_probility = round(y_single_probility[0][1], 4)
                # print(y_single_probility)
                susceptibility[i, j] = y_single_probility

    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(susceptibility)
    outRaster.SetProjection(proj)
    outRaster.FlushCache()
    del outRaster

# train_data_x = np.load('train_x.npy')
# test_data_x = np.load('test_x.npy')
#
# train_data_y = np.append(np.ones(76,dtype=int),np.zeros(76,dtype=int))
# test_data_y = np.append(np.ones(32,dtype=int),np.zeros(32,dtype=int))
# # print(train_data_x[0].shape)
# #
# # SVM = svm.SVC(probability=True)
# # SVM.fit(train_data_x,train_data_y)
# #
# model = load_model('my_model_1Dshangyou.h5')
# get_suscep_DL_1D("shangyou.tif", "SVM_suscept1.tif",model)


# y_pred = SVM.predict(test_data_x)                            #得到输出标签值
# accuracy = SVM.score(test_data_x,test_data_y)                           #得到分类正确率
# y_probability = SVM.predict_proba(test_data_x)               #得到分类概率值
# y_probability_first = [x[1] for x in y_probability]
# test_auc = metrics.roc_auc_score(test_data_y,y_probability_first)    #得到AUC值
# fpr, tpr, thresholds = metrics.roc_curve(test_data_y, y_probability_first)
# kappa = metrics.cohen_kappa_score(test_data_y, y_pred)
# aic = evaluate_method.AIC(test_data_y, y_probability_first, 13, 152)
# ks = evaluate_method.ks_calc_auc(test_data_y, y_probability_first)
#
# print ('accuracy = %f' %accuracy)
# print ('AUC = %f'%test_auc)
# print('Kappa = %f'%kappa)
# print(ks)
# #输出混淆矩阵
# print (confusion_matrix(test_data_y,y_pred))

