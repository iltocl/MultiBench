import sys
import os
sys.path.append(os.path.abspath('../../code'))
from load_AffectDatasets import load_dataset_pkl

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

def evaluating_with_MLPCLassifier(_X_train, _y_train, _X_test, _plot):
    clf = MLPClassifier(activation="relu", 
                        solver="adam", 
                        random_state=1, verbose=False, 
                        max_iter=100000).fit(_X_train, _y_train)
    if _plot == True:
        plt.plot(clf.loss_curve_)
        plt.show()
        print("CLF:", clf.get_params)
    else:
        print("CLF:", clf.n_iter_, clf.get_params)
    y_pred = clf.predict(_X_test)
    
    #print(classification_report(_y_test, y_pred, digits=4))
    return y_pred

def count_class1_votes(_pred_labels, _n):
    all_embeddings = [_pred_labels[i:i + _n] for i in range(0, len(_pred_labels), _n)]

    lst_votes_class1 = []
    i = 0
    for video_embeddings in all_embeddings:
        votes = np.bincount(video_embeddings.astype(int))#.argmax()
        votes_for_class0 = votes[0]
        if votes_for_class0 == 50:
            votes_for_class1 = 0
        else:
            votes_for_class1 = votes[1]
        
        lst_votes_class1.append(votes_for_class1)
        i = i + 1

    return lst_votes_class1

def evaluate_labels(_threshold, _y_test, _lst_votes, _n):
    i = 0
    for i in range(0, len(_y_test)):
        if _y_test[i] == _n:
            _y_test[i] = 1
        i = i + 1
    #print(len(_lst_votes), _lst_votes)
    #print(len(_y_test), _y_test)

    arr_preds = []
    for i in _lst_votes:
        if i >= _threshold:
            arr_preds.append(1)
        else:
            arr_preds.append(0)
    #print(len(arr_preds), arr_preds)
    return _y_test, arr_preds


def extract_results_from_report(_name, _int_threshold, _str_modality, _report, _file_txt):
    if not os.path.exists(_file_txt):
        print("Creating file...")
        with open(_file_txt, "a") as f:
                f.write(f"votes, modality, acc, macro_f1, f1_class0, f1_class1, representations\n")
    
    acc = _report["accuracy"]
    macro_f1 = _report["macro avg"]["f1-score"]
    f1_0 = _report["0"]["f1-score"]
    f1_1 = _report["1"]["f1-score"]
    # Build the line (comma-separated as you asked)
    line = f"{_int_threshold}, {_str_modality}, {acc}, {macro_f1}, {f1_0}, {f1_1}, {_name}\n"
    print(f"votes: {_int_threshold}, modality: {_str_modality}, acc: {acc}, macro-f1: {macro_f1}, f1_0: {f1_0}, f1_1: {f1_1}, variation: {_name}\n")
    #print(line)
    # Append to a text file
    with open(_file_txt, "a") as f:
        f.write(line)
    print("Results saved on:", _file_txt)

# ------------------------------------------- BASELINES
def load_modality(_str_M, _file_path, _str_dataset, _str_video_representation):
    if _str_M == "T":
        test_ids, X_train_T, X_test_T, X_valid_T, y_train_T, y_test_T, y_valid_T= load_dataset_pkl(_file_path_pkl=_file_path, 
                    _str_dataset=_str_dataset, 
                    _str_video_representation=_str_video_representation, 
                    _M="T")
        X_train, X_test, X_valid = X_train_T, X_test_T, X_valid_T
        y_train, y_test, y_valid = y_train_T, y_test_T, y_valid_T
    elif _str_M == "V":
        test_ids, X_train_V, X_test_V, X_valid_V, y_train_V, y_test_V, y_valid_V= load_dataset_pkl(_file_path_pkl=_file_path, 
                    _str_dataset=_str_dataset, 
                    _str_video_representation=_str_video_representation, 
                    _M="V")
        X_train, X_test, X_valid = X_train_V, X_test_V, X_valid_V
        y_train, y_test, y_valid = y_train_V, y_test_V, y_valid_V
    elif _str_M == "A":
        test_ids, X_train_A, X_test_A, X_valid_A, y_train_A, y_test_A, y_valid_A= load_dataset_pkl(_file_path_pkl=_file_path, 
                    _str_dataset=_str_dataset, 
                    _str_video_representation=_str_video_representation, 
                    _M="A")
        X_train, X_test, X_valid = X_train_A, X_test_A, X_valid_A
        y_train, y_test, y_valid = y_train_A, y_test_A, y_valid_A
    else:
        print("Error: Available modalities: T, V or A for MUSTARD dataset")
    return X_train, y_train, X_test, y_test, test_ids

def get_predictions(_file_path, _str_dataset, _str_video_representation, _str_M, _plot):
    if os.path.exists(_file_path):
        print(f"File:'{_file_path}' exists.")
    else:
        print(f"Error: File '{_file_path}' does NOT exist. Check the file path.")
    # Load Dataset
    X_train, y_train, X_test, y_test, test_ids = load_modality(_str_M, _file_path, _str_dataset, _str_video_representation)
    
    print(f"Dataset {_str_dataset} loaded successfully. Modality: {_str_M}.")
    
    # Evaluate with MLP
    lst_preds = evaluating_with_MLPCLassifier(X_train, y_train, X_test, _plot)
    
    if _str_dataset == "mustard":
        _n = 50 # value by default
    
    # Obtain numver of votes for class 1 (sarcasm)
    lst_votes_for_class1 = count_class1_votes(lst_preds, _n)
    ds_test_labels = count_class1_votes(y_test, _n)

    return ds_test_labels, lst_votes_for_class1

def get_predictions_2modalities(_file_path1, _file_path2, _str_dataset, _str_video_representation, _m1, _m2):
    # Load Dataset
    X_train1, y_train1, X_test1, y_test1, test_ids1 = load_modality(_m1, _file_path1, _str_dataset, _str_video_representation)
    X_train2, y_train2, X_test2, y_test2, test_ids2 = load_modality(_m2, _file_path2, _str_dataset, _str_video_representation)
    test_ids = test_ids1

    # concat_modalities = np.concatenate((a, b), axis=1)
    X_train = np.concatenate((X_train1, X_train2), axis=1)
    X_test = np.concatenate((X_test1, X_test2), axis=1)
    #print(X_train.shape, X_test.shape)
    y_train = y_train1
    y_test = y_test1

    # Evaluate with MLP
    lst_preds = evaluating_with_MLPCLassifier(X_train, y_train, X_test, False)
    
    if _str_dataset == "mustard":
        _n = 50 # by default
    
    # Obtain votes for class 1
    lst_votes_for_class1 = count_class1_votes(lst_preds, _n)
    ds_test_labels = count_class1_votes(y_test, _n)

    return ds_test_labels, lst_votes_for_class1 #, test_ids, X_test

def get_predictions_3modalities(_file_path1, _file_path2, _file_path3, _str_dataset, _str_video_representation, _m1, _m2, _m3):
    # Load Dataset
    X_train1, y_train1, X_test1, y_test1, test_ids1 = load_modality(_m1, _file_path1, _str_dataset, _str_video_representation)
    X_train2, y_train2, X_test2, y_test2, test_ids2 = load_modality(_m2, _file_path2, _str_dataset, _str_video_representation)
    X_train3, y_train3, X_test3, y_test3, test_ids3 = load_modality(_m3, _file_path3, _str_dataset, _str_video_representation)
    
    test_ids = test_ids1

    # concat_modalities = np.concatenate((a, b), axis=1)
    X_train = np.concatenate((X_train1, X_train2, X_train3), axis=1)
    X_test = np.concatenate((X_test1, X_test2, X_test3), axis=1)
    #print(X_train.shape, X_test.shape)
    y_train = y_train1
    y_test = y_test1

    # Evaluate with MLP
    lst_preds = evaluating_with_MLPCLassifier(X_train, y_train, X_test, False)
    
    if _str_dataset == "mustard":
        _n = 50 # by default
    
    # Obtain votes for class 1
    lst_votes_for_class1 = count_class1_votes(lst_preds, _n)
    ds_test_labels = count_class1_votes(y_test, _n)

    return ds_test_labels, lst_votes_for_class1 #, test_ids, X_test
