import numpy as np
import jenkspy
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split
from common_func import TIF_functions
from sklearn.linear_model import LogisticRegression
from common_func import evaluate_method
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from deepLearning.data_sanxia_PUlearning2 import get_AUC
np.random.seed(1)

def get_data(data_index, all_data):
    data = []
    for i in range(len(data_index)):
        single_data = all_data[data_index[i][0], data_index[i][1],:]
        data.append(single_data)
    return np.array(data)

def evaluate_model(evalModel, landslide_test_index):
    col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
    total_data = np.load("../total_data_standard.npy")
    row, col, num_factors = total_data.shape

    total_prob = TIF_functions.get_sus_map(total_data, evalModel)
    total_prob[altitude[:, :, 0] == -9999] = -1

    test_auc = get_AUC.get_cum_area(landslide_test_index, total_prob)
    # print(test_auc)
    # TIF_functions.save_tif(total_prob, "RF_pu111.tif",geotransform, proj)
    return test_auc

def naturalBreak(inputdata, numclass):
    prob = np.array([i for i in inputdata if i >= 0])
    sample_data = []
    for i in range(len(prob)):
        if i % 5 == 0:
            sample_data.append(prob[i])
    breaks = jenkspy.jenks_breaks(sample_data, numclass)
    return breaks[4]

def singleIter (ps_data, unlab_data, classifier, ratio, unlab_weights = None):

    neg_data_value = np.random.choice(len(unlab_data), size= int(ratio*len(ps_data)),replace=True, p=unlab_weights)
    # print(neg_data_value[0:11])
    neg_data = np.array([unlab_data[a] for a in neg_data_value])
    train_data_x = np.concatenate((ps_data, neg_data), axis=0)
    # print(len(train_data_x))
    train_data_y = np.append(np.ones(len(ps_data), dtype=int), np.zeros(len(neg_data), dtype=int))

    model = classifier.fit(train_data_x, train_data_y)
    # evaluate_model( model,landslide_test_index)
    unlab_prob = model.predict_proba(unlab_data)

    pos_prob = np.array([b[1] for b in unlab_prob])
    unlab_prob = np.array([a[0] for a in unlab_prob])
    breaks = pos_prob.max()*0.9
    # breaks = naturalBreak(pos_prob, 5)
    # print(breaks)
    #delete
    new_pos_index = np.argwhere(pos_prob >= breaks)
    unlabel_index = np.argwhere(unlab_prob > 1-breaks)
    new_unlab_prob = np.array([unlab_prob[a[0]] for a in unlabel_index])

    add_pos_data = np.array([unlab_data[a[0]] for a in new_pos_index])
    new_unlabel_data = np.array([unlab_data[a[0]] for a in unlabel_index])

    new_ps_data = np.concatenate((ps_data, add_pos_data), axis=0)

    # print(new_pos_index)


    # print(unlab_prob)

    # real_neg_value = np.argwhere(unlab_prob > 0.5)
    # new_unlabel_data = np.array([unlab_data[a] for a in real_neg_value])
    # real_neg_weights = np.array([unlab_prob[b] for b in real_neg_value])
    # new_unlabel_data = new_unlabel_data.reshape((len(new_unlabel_data), 12))
    # print(new_unlabel_data.shape)
    # new_weights = real_neg_weights/np.sum(real_neg_weights)
    # new_weights = new_weights.reshape(len(new_weights),)
    # print(new_weights)

    # new_weights = unlab_prob / np.sum(unlab_prob)
    new_weights = new_unlab_prob/np.sum(new_unlab_prob)
    return new_weights, train_data_x, train_data_y, new_ps_data, new_unlabel_data

def adaSample(ps_data, unlab_data, classifier,iteration, landslide_test_index, weights = None):
    ratio = 1
    iteration_auc = []
    for i in range(iteration):
        new_weights, train_x, train_y, ps_data, unlab_data = singleIter(ps_data, unlab_data, classifier, ratio, unlab_weights = weights)
        weights = new_weights
        ratio += 0
        final_model = classifier.fit(train_x, train_y)
        auc = evaluate_model(final_model, landslide_test_index)
        iteration_auc.append(auc)
    print(iteration_auc)


    return final_model, iteration_auc

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
    col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
    # landslide_index = np.load("../landslide_index.npy")
    nonlandslide_total_index = np.load("../nonlandslide_index.npy")
    landslide_train_index = np.load("landslide_train_index.npy")
    landslide_test_index = np.load("landslide_test_index.npy")
    total_data = np.load("../total_data_standard.npy")
    col, row, geotransform, proj, landslide_area = TIF_functions.read_tif('../landslide_number.tif')
    landslide_area = landslide_area.reshape((row, col))
    row, col, num_factors = total_data.shape
    test_resutls = []

    for seed in range(2, 3):

        landslide_train_data = get_data(landslide_train_index, total_data)
        train_data_y = np.ones(len(landslide_train_data), dtype=int)
        unlab_data = get_data(nonlandslide_total_index, total_data)
        # base_model = DecisionTreeClassifier()

        train_number = np.load('train_number.npy')
        train_y = np.ones(len(train_number), dtype=int)
        # n_estim = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        n_estim = [ 20,  40,  60, 80,  100]
        depth_value = [5, 10, 15, 20, 25, 30]
        kFold = StratifiedKFold(y=train_y, n_folds=3, random_state=1,shuffle=True)
        results = []
        for est in n_estim:
            print("特征数：", est)
            base_model = RandomForestClassifier( n_estimators=5,random_state=1)
            scores = np.ones((3, 20))
            for k, (train, val) in enumerate(kFold):
                new_landslide_train_index = get_val_index(train_number[train], landslide_area)
                val_index = get_val_index(train_number[val], landslide_area)
                new_landslide_train_data = get_data(new_landslide_train_index, total_data)

                PUmodel, auc = adaSample(ps_data=new_landslide_train_data, unlab_data=unlab_data,
                            classifier = base_model, iteration=5, landslide_test_index = val_index)

                scores[k,:] = auc

        results.append(np.mean(scores, axis=0))
        np.savetxt("results.txt", np.array(results))
                # val_auc = evaluate_model(PUmodel, landslide_train_index[val])
                # scores.append(val_auc)
            # print(est, ':',np.mean(scores))

# joblib.dump(PUmodel, 'LR.pkl')
# evaluate_model(test_data_x, test_data_y, PUmodel)

