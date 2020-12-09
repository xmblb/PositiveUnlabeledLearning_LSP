import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from deepLearning.data_sanxia_PUlearning2 import get_AUC
from common_func import TIF_functions

def get_data(data_index, all_data):
    data = []
    for i in range(len(data_index)):
        single_data = all_data[data_index[i][0], data_index[i][1],:]
        data.append(single_data)
    return np.array(data)

def get_sus_map(imgs,model):
    # lenth=len(imgs)
    row=imgs.shape[0]
    col=imgs.shape[1]
    imgs=np.reshape(imgs,[ row*col, imgs.shape[2]])

    proba=model.predict_proba(imgs)
    proba=proba[:,1]
    return proba


if __name__ == '__main__':
    total_data = np.load("../total_data_standard.npy")
    col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
    # altitude = altitude.reshape((row, col))
    col, row, geotransform, proj, landslide_area = TIF_functions.read_tif('../landslide_number.tif')
    nonlandslide_total_index = np.load("../nonlandslide_index.npy")
    landslide_train_index = np.load('landslide_train_index.npy')
    landslide_test_index = np.load('landslide_test_index.npy')

    landslide_area = landslide_area.reshape((row, col))

    # plt.figure()
    # plt.imshow(landslide_area[:,:])
    # plt.show()
    test_resutls = []
    for seed in range(2,3):

        NU = nonlandslide_total_index.shape[0]
        T = 100
        K = len(landslide_train_index)

        total_prob = np.ones((row*col, T))
        for i in range(T):
            # Bootstrap resample
            bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)
            # Positive set + bootstrapped unlabeled set
            data_bootstrap_index = np.concatenate((landslide_train_index, nonlandslide_total_index[bootstrap_sample, :]), axis=0)
            data_bootstrap = get_data(data_bootstrap_index, total_data)
            train_label =np.append(np.ones(len(landslide_train_index), dtype=int), np.zeros(len(landslide_train_index), dtype=int))
            # Train model
            model = DecisionTreeClassifier(max_depth=None, max_features=None,
                                           criterion='gini', class_weight='balanced')
            model.fit(data_bootstrap, train_label)

            # model_name = "SVM" + str(seed) + ".model"
            single_prob = get_sus_map(total_data, model)

            total_prob[:,i] = single_prob

        total_prob = np.mean(total_prob,axis=1)
        # print(total_prob.shape)
        total_prob=np.reshape(total_prob,(row,col))

        total_prob[altitude[:, :, 0] == -9999] = -1

        TIF_functions.save_tif(total_prob, "PUbagging.tif", geotransform, proj)
        train_auc, train_axis = get_AUC.get_cum_area(landslide_train_index, total_prob)
        test_auc, test_axis = get_AUC.get_cum_area(landslide_test_index, total_prob)

        np.savetxt('Pubag_train_curve.txt', train_axis)
        np.savetxt('Pubag_test_curve.txt', test_axis)

        test_resutls.append(test_auc)
        print(seed,train_auc, test_auc)

        # print(seed, train_auc, test_auc)
    print("average:", np.average(np.array(test_resutls)))
    print("std:", np.std(np.array(test_resutls)))