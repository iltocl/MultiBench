"""
Affect Datasets correspond to:
'mustard': data/datasets/sarcasm.pkl
'hsdvmx': data/datasets/hsdvmx.pkl
"""
import sys
import os
sys.path.append(os.path.abspath('./'))

import pickle
# for scikit-learn
import numpy as np

# ----------------------------------------------------
# about-labels
# ----------------------------------------------------
def assign_modalities_text_vision_audio(_ds_partition, _M):
    """Receives a dict ds_partition and returns each partition modalities
    """
    try: 
        ds_partition_t = _ds_partition['text']
    except:
       ds_partition_t = None
    
    try:
        ds_partition_v = _ds_partition['vision']
    except:
       ds_partition_v = None
    
    try:
        ds_partition_a = _ds_partition['audio']
    except:
       ds_partition_a = None

    return ds_partition_t, ds_partition_v, ds_partition_a

def labels_posneg_to_binary(_ds_train, _ds_test, _ds_valid):
    """Given {-1, 1} values it returns the corresponding {0, 1} values
    """
    ds_train_labels = np.array(_ds_train['labels'])
    ds_test_labels = np.array(_ds_test['labels'])
    ds_valid_labels = np.array(_ds_valid['labels'])
    
    ds_train_labels[ds_train_labels == -1] = 0
    ds_test_labels[ds_test_labels == -1] = 0
    ds_valid_labels[ds_valid_labels == -1] = 0
    
    return ds_train_labels, ds_test_labels, ds_valid_labels

 
def checkForInfinitiesorNaN(_ds_partition):
    #print(np.any(np.isinf(_ds_partition)))  # Check for infinities
    #print(np.any(np.isnan(_ds_partition)))
    _ds_partition = _ds_partition.astype(np.float32)
    # To deal with infinites
    # Clip values to a specific range (e.g., between -1e10 and 1e10)
    _ds_partition = np.clip(_ds_partition, -1e10, 1e10)
    # To deal with NaN
    # Replace it with 0
    _ds_partition = np.nan_to_num(_ds_partition, nan=0.0, posinf=1e10, neginf=-1e10)

    # For example, check the range of values after a transformation
    #print("Min:", np.min(_ds_partition))
    #print("Max:", np.max(_ds_partition))
    return _ds_partition

def labels_class_or_subclass(_ds_train, _ds_test, _ds_valid, _labels):
    print(f"labels_class_or_subclass, _labels: {_labels}")
    
    if _labels == 'labels':
        print("labels -> 3 classes")
        ds_train_labels = np.array(_ds_train['labels'])
        ds_test_labels = np.array(_ds_test['labels'])
        ds_valid_labels = np.array(_ds_valid['labels'])    
    elif _labels == 'labels_hs':
        print("labels_hs -> sub-classes")
        ds_train_labels = np.array(_ds_train['labels_hs'])
        ds_test_labels = np.array(_ds_test['labels_hs'])
        ds_valid_labels = np.array(_ds_valid['labels_hs'])
    elif _labels == 'labels_bin':
        print("labels_bin -> 3 classes to binary")
        ds_train_labels = np.array(_ds_train['labels_bin'])
        ds_test_labels = np.array(_ds_test['labels_bin'])
        ds_valid_labels = np.array(_ds_valid['labels_bin'])
        #ds_train_labels[ds_train_labels == 2] = 1 # hate-speech to negative
        #ds_test_labels[ds_test_labels == 2] = 1
        #ds_valid_labels[ds_valid_labels == 2] = 1
    else:
        print("Error: labels_class_or_subclass, check _labels")

    return ds_train_labels, ds_test_labels, ds_valid_labels

# -----------------------------------------------------------------------------
# representations: promSumVectors, bagVectors
# -----------------------------------------------------------------------------

def embedRepresentationsForEachSample(_ds_partition_modality):
  """ (promSumVect) transform each modality of shape(n_samples, m, D) to shape(n_samples,D) 
  """
  X_modality = np.zeros((_ds_partition_modality.shape[0], _ds_partition_modality.shape[2]), dtype=float)
  #print(X_modality.shape)
  for i in range(_ds_partition_modality.shape[0]):
    reduced_sample = np.mean(_ds_partition_modality[i], axis=0)
    X_modality[i] = reduced_sample
    #print(reduced_sample.shape)
  return X_modality

def bag_embedRepresentationsForEachSample(_ds_partition_modality, _ds_partition_labels):
  """ Each modality instances as bag of embeddings of shape(n_samples, m, D) 
  """
  num_instances = _ds_partition_modality.shape[0]
  n = _ds_partition_modality.shape[1]
  m = _ds_partition_modality.shape[2]
  num_all = num_instances * n

  X_data = np.zeros((num_all, m), dtype=float)
  X_labels = np.zeros((num_all, ), dtype=float)
  #print(X_data.shape, X_labels.shape)
  
  #print(_ds_partition_modality.shape[0])
  #print(_ds_partition_modality.shape[1])
  #print(_ds_partition_modality.shape[2])

  counter = 0
  for instance in range(num_instances):
    for i in range(n):
      #print("video", instance, "n",i)
      sub_instance = _ds_partition_modality[instance][i]
      sub_label = _ds_partition_labels[instance]
      #print("label", sub_label, "sub_instance", sub_instance[:5])
      X_data[counter] = sub_instance
      X_labels[counter] = sub_label
      counter +=1

  return X_data, X_labels

# -----------------------------------------------------------------------------
# load_dataset_pkl
# -----------------------------------------------------------------------------

def load_dataset_pkl(_file_path_pkl, _str_dataset, _str_video_representation, _M, _labels=None):
    """ Receives a pkl that contains the dataset information in a dictionary format and returns the loaded data partitions and labels
    
    Parameters
    ------
    _file_path_pkl: string, path where the pkl file exists
    _str_dataset: string, dataset to work with (mustard)
    _str_video_representation: string, representations will be generated between: 
        'avgSumVectors': given an input of n x m it applies an average sum of the vectors to work and generate a 1 x n representation
        'bagVectors': given an input of n x m, it generates a representation for each n vector
    _M: modality to work with: {T, V, A} where T=text, V=video, A=audio

    Returns
    -------
    ds_train, ds_test, ds_valid: dict
        A dictionary for each parititon
    ds_train_labels, ds_test_labels, ds_valid_labels: list
        A list of labels for each partition
    """
    with open(_file_path_pkl, 'rb') as file:
        ds_data = pickle.load(file)
    # verifying partitions and assing it to a particular group
    if len(ds_data.keys()) == 3:
        #print("DS partitions:", ds_data.keys())
        ds_train = ds_data['train']
        ds_test = ds_data['test']
        ds_valid = ds_data['valid']
        # video ids for test
        ds_ids = ds_data['test']['id']
    else:
       #print("DS partitions:", ds_data.keys())
       print("def load_dataset_pkl: Verify your dataset partitions. Required format ['train', 'test', 'valid']")
    
    # Available datasets to work with...
    if _str_dataset == "mustard":
        print("Working on:", _str_dataset)
        print("Video representation:", _str_video_representation)
        # assigning modalities data to each partition
        ds_train_t, ds_train_v, ds_train_a = assign_modalities_text_vision_audio(ds_train, _M)
        ds_test_t, ds_test_v, ds_test_a = assign_modalities_text_vision_audio(ds_test, _M)
        ds_valid_t, ds_valid_v, ds_valid_a = assign_modalities_text_vision_audio(ds_valid, _M)
        # assigning labels to each partition
        ds_train_labels, ds_test_labels, ds_valid_labels = labels_posneg_to_binary(ds_train, ds_test, ds_valid)
        # work with video representations as: promSumVectors or load each "point" of the videos
        if _str_video_representation == "promSumVectors":
            y_train, y_test, y_valid = ds_train_labels, ds_test_labels, ds_valid_labels
            print("Labels", set(y_train))
            # representations for each modality
            if _M == "T":
                X_train_T = embedRepresentationsForEachSample(ds_train_t)
                X_test_T = embedRepresentationsForEachSample(ds_test_t)
                X_valid_T = embedRepresentationsForEachSample(ds_valid_t)
                X_train, X_test, X_valid = X_train_T, X_test_T, X_valid_T
            elif _M == "V":
                X_train_V = embedRepresentationsForEachSample(ds_train_v)
                X_test_V = embedRepresentationsForEachSample(ds_test_v)
                X_valid_V = embedRepresentationsForEachSample(ds_valid_v)
                X_train, X_test, X_valid = X_train_V, X_test_V, X_valid_V
            elif _M == "A":    
                X_train_A = embedRepresentationsForEachSample(ds_train_a)
                X_test_A = embedRepresentationsForEachSample(ds_test_a)
                X_valid_A = embedRepresentationsForEachSample(ds_valid_a)
                X_train, X_test, X_valid = X_train_A, X_test_A, X_valid_A
            else:
               print("Error: no modality", _M)
            return ds_ids, X_train, X_test, X_valid, y_train, y_test, y_valid
        elif _str_video_representation == "bagVectors":
            if _M == "T":
                X_train_T, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_t, ds_train_labels)
                X_test_T, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_t, ds_test_labels)
                X_valid_T, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_t, ds_valid_labels)
                X_train, X_test, X_valid = X_train_T, X_test_T, X_valid_T
            elif _M == "V":
                X_train_V, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_v, ds_train_labels)
                X_test_V, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_v, ds_test_labels)
                X_valid_V, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_v, ds_valid_labels)
                X_train, X_test, X_valid = X_train_V, X_test_V, X_valid_V
            elif _M == "A":
                X_train_A, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_a, ds_train_labels)
                X_test_A, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_a, ds_test_labels)
                X_valid_A, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_a, ds_valid_labels)
                X_train, X_test, X_valid = X_train_A, X_test_A, X_valid_A
            else:
               print("Error: no modality", _M)
            # labels at "point level"
            y_train, y_test, y_valid = X_train_labels, X_test_labels, X_valid_labels
            return ds_ids, X_train, X_test, X_valid, y_train, y_test, y_valid  
    # end if mustard dataset
    elif _str_dataset == "hsdvmx":
        print("Working on:", _str_dataset)
        print("Video representation:", _str_video_representation)
        # assigning modalities data to each partition
        ds_train_t, ds_train_v, ds_train_a = assign_modalities_text_vision_audio(ds_train, _M)
        ds_test_t, ds_test_v, ds_test_a = assign_modalities_text_vision_audio(ds_test, _M)
        ds_valid_t, ds_valid_v, ds_valid_a = assign_modalities_text_vision_audio(ds_valid, _M)
        # assigning labels to each partition
        # labels = {0,1,2} OR labels_hs = {0,1,2,3,4}
        ds_train_labels, ds_test_labels, ds_valid_labels = labels_class_or_subclass(ds_train, ds_test, ds_valid, _labels)
        # work with video representations as: promSumVectors or load each "point" of the videos
        if _str_video_representation == "promSumVectors":
            y_train, y_test, y_valid = ds_train_labels, ds_test_labels, ds_valid_labels
            print("Labels", set(y_train))
            # representations for each modality
            if _M == "T":
                X_train_T = embedRepresentationsForEachSample(ds_train_t)
                X_test_T = embedRepresentationsForEachSample(ds_test_t)
                X_valid_T = embedRepresentationsForEachSample(ds_valid_t)
                X_train, X_test, X_valid = X_train_T, X_test_T, X_valid_T
            elif _M == "V":
                X_train_V = embedRepresentationsForEachSample(ds_train_v)
                X_test_V = embedRepresentationsForEachSample(ds_test_v)
                X_valid_V = embedRepresentationsForEachSample(ds_valid_v)
                X_train, X_test, X_valid = X_train_V, X_test_V, X_valid_V
            elif _M == "A":    
                X_train_A = embedRepresentationsForEachSample(ds_train_a)
                X_test_A = embedRepresentationsForEachSample(ds_test_a)
                X_valid_A = embedRepresentationsForEachSample(ds_valid_a)
                X_train, X_test, X_valid = X_train_A, X_test_A, X_valid_A
            else:
               print("Error: no modality", _M)
            return ds_ids, X_train, X_test, X_valid, y_train, y_test, y_valid
        elif _str_video_representation == "bagVectors":
            if _M == "T":
                X_train_T, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_t, ds_train_labels)
                X_test_T, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_t, ds_test_labels)
                X_valid_T, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_t, ds_valid_labels)
                X_train, X_test, X_valid = X_train_T, X_test_T, X_valid_T
            elif _M == "V":
                X_train_V, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_v, ds_train_labels)
                X_test_V, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_v, ds_test_labels)
                X_valid_V, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_v, ds_valid_labels)
                X_train, X_test, X_valid = X_train_V, X_test_V, X_valid_V
            elif _M == "A":
                X_train_A, X_train_labels = bag_embedRepresentationsForEachSample(ds_train_a, ds_train_labels)
                X_test_A, X_test_labels = bag_embedRepresentationsForEachSample(ds_test_a, ds_test_labels)
                X_valid_A, X_valid_labels = bag_embedRepresentationsForEachSample(ds_valid_a, ds_valid_labels)
                X_train, X_test, X_valid = X_train_A, X_test_A, X_valid_A
            else:
               print("Error: no modality", _M)
            # labels at "point level"
            y_train, y_test, y_valid = X_train_labels, X_test_labels, X_valid_labels
            return ds_ids, X_train, X_test, X_valid, y_train, y_test, y_valid  
    else:
       print("def load_dataset_pkl: dataset", _str_dataset, " isn't available.")
