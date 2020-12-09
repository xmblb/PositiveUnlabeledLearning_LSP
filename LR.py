import numpy as np
from sklearn.model_selection import train_test_split
from common_func import TIF_functions
import gdal
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from deepLearning.data_sanxia_PUlearning2 import get_AUC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from deepLearning.data_MM1 import common_func


def get_data(data_index, all_data):
    data = []
    for i in range(len(data_index)):
        single_data = all_data[data_index[i][0], data_index[i][1],:]
        data.append(single_data)
    return np.array(data)


# def get_modelling_data(landslide_index, nonlandslide_total_index,total_data, seed):
#     np.random.seed(seed)
#     # get nonlandslide index from the total nonlandslide area
#     nonlandslide_value = np.random.choice(len(nonlandslide_total_index), size=len(landslide_index),replace=False)
#     nonlandslide_index = np.array([nonlandslide_total_index[a] for a in nonlandslide_value])
#
#     landslide_train_index, landslide_test_index = train_test_split(landslide_index, test_size=0.29999, shuffle=True)
#     nonlandslide_train_index, nonlandslide_test_index = train_test_split(nonlandslide_index, test_size=0.29999, shuffle=True)
#
#
#     train_index = np.concatenate((landslide_train_index, nonlandslide_train_index), axis = 0)
#     test_index = np.concatenate((landslide_test_index, nonlandslide_test_index), axis = 0)
#     # print(len(train_index))
#     train_data = get_data(train_index, total_data)
#     test_data = get_data(test_index, total_data)
#     return  train_data, test_data, landslide_test_index, nonlandslide_test_index

def get_modelling_data(landslide_data, nonlandslide_all_index, total_data, seed):
    np.random.seed(seed)
    landslide_number = [i for i in range(197)]
    landslide_train_number, landslide_test_number = train_test_split(landslide_number, test_size=0.297, shuffle=True)
    np.savetxt('test_number.txt', landslide_test_number)
    landslide_train_index = np.array([])
    for i in range(len(landslide_train_number)):
        index = np.argwhere(landslide_data == landslide_train_number[i])
        if i == 0:
            landslide_train_index = index
        else:
            landslide_train_index = np.append(landslide_train_index, index, axis=0)
        # print(index.shape, landslide_train_index.shape)
    # landslide_train_index = np.array(landslide_train_index)

    landslide_test_index = np.array([])
    for i in range(len(landslide_test_number)):
        index = np.argwhere(landslide_data == landslide_test_number[i])
        if i == 0:
            landslide_test_index = index
        else:
            landslide_test_index = np.append(landslide_test_index, index, axis=0)

    np.save("landslide_train_index.npy", landslide_train_index)
    np.save("landslide_test_index.npy", landslide_test_index)
    # print(landslide_train_index.shape, landslide_train_index)
    landslide_train_data = get_data(landslide_train_index, total_data)
    # print(landslide_train_data.shape)
    landslide_test_data = get_data(landslide_test_index, total_data)


    nonlandslide_value = np.random.choice(range(len(nonlandslide_all_index)),
                                          len(landslide_train_index)+len(landslide_test_index), replace=False)
    nonlandslide_index = np.array([nonlandslide_all_index[c] for c in nonlandslide_value])

    nonlandslide_train_index, nonlandslide_test_index = train_test_split(nonlandslide_index,
                                                                       test_size = len(landslide_test_index), shuffle=True)

    nonlandslide_train_data = get_data(nonlandslide_train_index, total_data)
    nonlandslide_test_data = get_data(nonlandslide_test_index, total_data)

    train_data = np.concatenate((landslide_train_data, nonlandslide_train_data), axis = 0)
    test_data =  np.concatenate((landslide_test_data, nonlandslide_test_data), axis = 0)
    return train_data, test_data, landslide_train_index, landslide_test_index, nonlandslide_test_index

def remove_factors(importance, train_data, test_data, total_data):

    save_number = []
    for i in range(len(importance)):
        if importance[i]>=0.01:
            save_number.append(i)
    new_train_data = np.ones((len(train_data), len(save_number)))
    new_test_data = np.ones((len(test_data), len(save_number)))
    new_total_data = np.ones((total_data.shape[0], total_data.shape[1], len(save_number)))

    for j in range(len(save_number)):
        new_train_data[:,j] = train_data[:,save_number[j]]
        new_test_data[:, j] = test_data[:,save_number[j]]
        new_total_data[:,:,j] = total_data[:,:,save_number[j]]
    return new_train_data, new_test_data, new_total_data



if __name__ == '__main__':
    total_data = np.load("../total_data_standard.npy")
    col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
    col, row, geotransform, proj, landslide_area = TIF_functions.read_tif('../landslide_number.tif')
    nonlandslide_total_index = np.load("../nonlandslide_index.npy")

    landslide_area = landslide_area.reshape((row, col))

    # plt.figure()
    # plt.imshow(landslide_area[:,:])
    # plt.show()
    test_resutls = []
    for seed in range(2,3):
        train_data_x, test_data_x, landslide_train_index, \
        landslide_test_index, nonlandslide_test_index = get_modelling_data(landslide_area,
                                                                           nonlandslide_total_index,
                                                                           total_data,
                                                                            seed)
        train_data_y = np.append(np.ones(len(landslide_train_index), dtype=int), np.zeros(len(landslide_train_index), dtype=int))
        test_data_y = np.append(np.ones(len(landslide_test_index), dtype=int), np.zeros(len(landslide_test_index), dtype=int))
        # model = DecisionTreeClassifier()
        model = LogisticRegression(C=0.1)
        # model = RandomForestClassifier(n_estimators=10, random_state=1)
        # model = svm.SVC(probability=True)
        model.fit(train_data_x,train_data_y)
        # importance = model.feature_importances_
        #
        # new_train_data, new_test_data, new_total_data = remove_factors(importance, train_data_x, test_data_x, total_data)
        # model = RandomForestClassifier(random_state=6)
        # # model = svm.SVC(probability=True)
        # model.fit(new_train_data,train_data_y)

        # model_name = "SVM" + str(seed) + ".model"
        total_prob = TIF_functions.get_sus_map(total_data, model)
        total_prob[altitude[:, :, 0] == -9999] = -1

        # TIF_functions.save_tif(total_prob, "LR.tif", geotransform, proj)
        train_auc,train_axis = get_AUC.get_cum_area(landslide_train_index, total_prob)
        test_auc,test_axis = get_AUC.get_cum_area(landslide_test_index, total_prob)

        np.savetxt('svm_train_curve.txt', train_axis)
        np.savetxt('svm_test_curve.txt', test_axis)

        test_resutls.append(test_auc)
        print(seed,train_auc, test_auc)
        # print(seed, train_auc, test_auc)
    print("average:", np.average(np.array(test_resutls)))
    print("std:", np.std(np.array(test_resutls)))

# if __name__ == '__main__':
#     col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
#     landslide_index = np.load("../landslide_index.npy")
#     nonlandslide_total_index = np.load("../nonlandslide_index.npy")
#
#     total_data = np.load("../total_data_standard.npy")
#     row, col, num_factors = total_data.shape
#     test_resutls = []
#
#     for seed in range(1, 10):
#         train_x, test_x, landslide_test_index, nonlandslide_test_index = \
#             get_modelling_data(landslide_index, nonlandslide_total_index,total_data, seed)
#
#         landslide_train_index = len(landslide_index)-len(landslide_test_index)
#
#         # generate the data labels
#         train_data_y = np.append(np.ones(landslide_train_index, dtype=int), np.zeros(landslide_train_index, dtype=int))
#         test_data_y = np.append(np.ones(len(landslide_test_index), dtype=int), np.zeros(len(landslide_test_index), dtype=int))
#
#         # model = RandomForestClassifier(random_state=6)
#         model = LogisticRegression()
#         # model = svm.SVC(probability=True)
#         model.fit(train_x, train_data_y)
#
#         total_prob = TIF_functions.get_sus_map(total_data, model)
#         total_prob[altitude[:, :, 0] == -9999] = -1
#
#         # TIF_functions.save_tif(total_prob, "RF.tif", geotransform, proj)
#         test_auc = get_AUC.get_cum_area(landslide_test_index, total_prob)
#
#         test_resutls.append(test_auc)
#         print(seed, test_auc)
#         # print(seed, train_auc, test_auc)
#     print("average:", np.average(np.array(test_resutls)))
#     print("std:", np.std(np.array(test_resutls)))


