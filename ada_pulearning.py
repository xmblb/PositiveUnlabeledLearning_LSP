import numpy as np
import jenkspy
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
    total_data = np.load("../total_data_standard_new.npy")
    row, col, num_factors = total_data.shape

    total_prob = TIF_functions.get_sus_map(total_data, evalModel)
    total_prob[altitude[:, :, 0] == -9999] = -1

    test_auc, axis = get_AUC.get_cum_area(landslide_test_index, total_prob)
    print(test_auc)
    # TIF_functions.save_tif(total_prob, "AdaPU.tif",geotransform, proj)
    return test_auc, axis

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
    # print(pos_prob.max())

    new_pos_index = np.argwhere(pos_prob >= breaks)
    unlabel_index = np.argwhere(unlab_prob > 1-breaks)
    new_unlab_prob = np.array([unlab_prob[a[0]] for a in unlabel_index])

    add_pos_data = np.array([unlab_data[a[0]] for a in new_pos_index])
    new_unlabel_data = np.array([unlab_data[a[0]] for a in unlabel_index])

    new_ps_data = np.concatenate((ps_data, add_pos_data), axis=0)

    # print(new_pos_index)

    # new_unlab_prob = unlab_prob
    # new_unlabel_data = unlab_data
    # new_ps_data = ps_data

    # print(unlab_prob)

    # new_weights = unlab_prob / np.sum(unlab_prob)
    new_weights = new_unlab_prob/np.sum(new_unlab_prob)
    return new_weights, train_data_x, train_data_y, new_ps_data, new_unlabel_data

def adaSample(ps_data, unlab_data, classifier,iteration, landslide_test_index, weights = None):
    ratio = 1
    for i in range(iteration):
        new_weights, train_x, train_y, ps_data, unlab_data = singleIter(ps_data, unlab_data, classifier, ratio, unlab_weights = weights)
        weights = new_weights
        ratio += 0
        final_model = classifier.fit(train_x, train_y)
        evaluate_model(final_model, landslide_test_index)


    return final_model


if __name__ == '__main__':
    col, row, geotransform, proj, altitude = TIF_functions.read_tif('../altitude.tif')
    # landslide_index = np.load("../landslide_index.npy")
    nonlandslide_total_index = np.load("../nonlandslide_index.npy")
    landslide_train_index = np.load("landslide_train_index.npy")
    landslide_test_index = np.load("landslide_test_index.npy")
    total_data = np.load("../total_data_standard.npy")
    row, col, num_factors = total_data.shape
    test_resutls = []

    for seed in range(2, 3):

        landslide_train_data = get_data(landslide_train_index, total_data)

        unlab_data = get_data(nonlandslide_total_index, total_data)
        # base_model = DecisionTreeClassifier()
        base_model = RandomForestClassifier(n_estimators=10, random_state=1)
        # base_model = LogisticRegression(C=10)
        PUmodel = adaSample(ps_data=landslide_train_data, unlab_data=unlab_data,
                            classifier = base_model, iteration=13, landslide_test_index = landslide_test_index)
        train_auc, train_axis = evaluate_model(PUmodel, landslide_train_index)
        test_auc, test_axis = evaluate_model(PUmodel, landslide_test_index)

        # np.savetxt('ada_train_curve1.txt', train_axis)
        # np.savetxt('ada_test_curve1.txt', test_axis)

        test_resutls.append(test_auc)
        print(seed,train_auc, test_auc)
    print("average:", np.average(np.array(test_resutls)))
    print("std:", np.std(np.array(test_resutls)))

# joblib.dump(PUmodel, 'LR.pkl')
# evaluate_model(test_data_x, test_data_y, PUmodel)

