import numpy as np
from sklearn.model_selection import train_test_split
from common_func import TIF_functions
import gdal
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_score,KFold
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


def get_modelling_data(landslide_data, nonlandslide_all_index, total_data, seed):
    np.random.seed(seed)
    landslide_number = [i for i in range(197)]
    landslide_train_number, landslide_test_number = train_test_split(landslide_number, test_size=0.297, shuffle=True)
    # np.save("train_number.npy", landslide_number)
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

    train_index = np.concatenate((landslide_train_index, nonlandslide_train_index), axis = 0)
    nonlandslide_train_data = get_data(nonlandslide_train_index, total_data)
    nonlandslide_test_data = get_data(nonlandslide_test_index, total_data)

    train_data = np.concatenate((landslide_train_data, nonlandslide_train_data), axis = 0)
    test_data =  np.concatenate((landslide_test_data, nonlandslide_test_data), axis = 0)
    return train_data, test_data, landslide_train_index, landslide_test_index, nonlandslide_train_index

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

def get_val_index(landslide_number, landslide_data):
    landslide_index = np.array([])
    for i in range(len(landslide_number)):
        index = np.argwhere(landslide_data == landslide_number[i])
        if i == 0:
            landslide_index = index
        else:
            landslide_index = np.append(landslide_index, index, axis=0)
    return landslide_index



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
        landslide_test_index, nonlandslide_train_index = get_modelling_data(landslide_area,
                                                                           nonlandslide_total_index,
                                                                           total_data,
                                                                            seed)

        train_data_y = np.append(np.ones(len(landslide_train_index), dtype=int), np.zeros(len(landslide_train_index), dtype=int))
        test_data_y = np.append(np.ones(len(landslide_test_index), dtype=int), np.zeros(len(landslide_test_index), dtype=int))
        # model = DecisionTreeClassifier()

        train_number = np.load('train_number.npy')
        train_y = np.ones(len(train_number), dtype=int)
        #cross validation
        penl = ["l1", "l2"]
        C_value = [0.001, 0.01, 0.1, 1, 10, 100]
        kFold = StratifiedKFold(y=train_y, n_folds=3, random_state=1,shuffle=True)
        for p in penl:
            for c in C_value:
                model = LogisticRegression()
                scores = []
                for k, (train, val) in enumerate(kFold):
                    new_landslide_train_index = get_val_index(train_number[train], landslide_area)
                    val_index = get_val_index(train_number[val], landslide_area)

                    new_nonlandslide_train_index, a = train_test_split(nonlandslide_train_index,
                                                                           test_size = len(val_index), shuffle=True)
                    new_train_index = np.concatenate((new_landslide_train_index,new_nonlandslide_train_index), axis=0)

                    new_train_data = get_data(new_train_index, total_data)
                    new_train_y = np.append(np.ones(len(new_landslide_train_index), dtype=int), np.zeros(len(new_nonlandslide_train_index), dtype=int))

                    model.fit(new_train_data, new_train_y)
                    total_prob = TIF_functions.get_sus_map(total_data, model)
                    total_prob[altitude[:, :, 0] == -9999] = -1

                    val_auc = get_AUC.get_cum_area(val_index, total_prob)
                    scores.append(val_auc)
                print(p, c, ':',np.mean(scores))




