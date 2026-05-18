import os
import torch
import numpy as np
import random

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import Sequential, Transformer, Identity, MLP
#from training_structures.unimodal import train, test  # noqa
from training_structures.unimodal import train as train_unimodal, test as test_unimodal
from torch import nn

from fusions.common_fusions import ConcatEarly# noqa
from training_structures.Supervised_Learning import train as train_mmdl
from training_structures.Supervised_Learning import test as test_mmdl
from training_structures.Supervised_Learning import MMDL


def set_seed(_seed=42):
    random.seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    np.random.seed(_seed)
    
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed) # multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_with_Transformer_M1enriched_seed(_data_type, _dir_file_pkl, _total_epochs, _modality_num, _features_n, _seed, _early_stop, _load_precomputed=False, _encoder_pt=None, _head_pt=None):
    """
    For MUSTARD, _modality_num: 0 = vision, 1 = audio, 2 = text
    """
    name = _dir_file_pkl.split('/')[-1].split('.')[0]
    # verify if the file exists
    if os.path.exists(_dir_file_pkl):
        print(f"\n------\nOK: {_dir_file_pkl}.")
    else:
        print(f"Error: check {_dir_file_pkl} file.")
    
    if _data_type == "sarcasm":
        max_seq_len_mustard = 50 # by default
    
    # Create the training, validation, and test-set dataloaders
    traindata_all, validdata_all, testdata_all = get_dataloader(_dir_file_pkl, robust_test=False, max_pad=True, 
                                                            data_type=_data_type, max_seq_len=max_seq_len_mustard,
                                                            z_norm=True, task='classification')
        
    set_seed(_seed)
    print("set_seed", _seed)

    if _data_type == "sarcasm":
        # MUSTARD dataset
        encoder = Identity().cuda()
        features_n = _features_n # it can vary according to the given representations in the pkl
        head = Sequential(Transformer(features_n, features_n, 5).cuda(), MLP(indim=features_n, hiddim=25, outdim=2)).cuda() # configuration can be adjusted
    else:
        print("_data_type N/A")    
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    mdl = nn.Sequential(encoder, head)
    
    if _load_precomputed and os.path.exists(_encoder_pt) and os.path.exists(_head_pt):
        print(f"Loading pretrained encoder: {_encoder_pt} ")
        #encoder.load_state_dict(torch.load(_encoder_pt))
        encoder = torch.load(_encoder_pt)
        print(f"Loading pretrained encoder: {_head_pt} ")
        #head.load_state_dict(torch.load(_head_pt))
        head = torch.load(_head_pt)
        mdl = nn.Sequential(encoder, head)    
        mdl.eval() 
    else:
        print("Begin training...")
        print("Epochs:", _total_epochs)
        save_encoder_file = "./encoder.pt"
        save_head_file = "./head.pt"
        mdl = train_unimodal(encoder, head, 
            traindata_all, validdata_all, 
            _total_epochs, 
            task="classification", 
            optimtype=torch.optim.AdamW, 
            lr=1e-5,
            weight_decay=0.01, 
            criterion=torch.nn.CrossEntropyLoss(),
            save_encoder=save_encoder_file, 
            save_head=save_head_file,
            modalnum=_modality_num,
            early_stop=_early_stop,
            track_complexity=False)

    print("Testing...")
    test_unimodal(encoder, head, testdata_all, 'affect', 
            criterion=torch.nn.CrossEntropyLoss(),
            task="classification", 
            modalnum=_modality_num, 
            no_robust=True)
    

    
def eval_with_Transformer_seed(_dir_file_pkl, _data_type, _features_n, _model_architecture, _total_epochs, _seed, _early_stop, _load_precomputed=True, _path_pt=None):
    name = _dir_file_pkl.split('/')[-1].split('.')[0]
    
    if os.path.exists(_dir_file_pkl):
        print(f"\n------\nOK: {_dir_file_pkl}.")
    else:
        print(f"Error: check {_dir_file_pkl} file.")
    
    if _data_type == "sarcasm":
        max_seq_len_mustard = 50 # by default
    
    # Create the training, validation, and test-set dataloaders.
    traindata_all, validdata_all, testdata_all = get_dataloader(_dir_file_pkl, robust_test=False, max_pad=True, 
                                                                data_type=_data_type, max_seq_len=max_seq_len_mustard,
                                                                z_norm=True, task='classification')
    set_seed(_seed)
    print("set_seed", _seed)
    
    # model architecture
    if _model_architecture == 1:
        # to process sarcasm.pkl where dimensions of T+V+A=752
        encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
        features_n = _features_n
        head = Sequential(Transformer(features_n, 752, 8).cuda(), MLP(752, 360, 2)).cuda()
        fusion = ConcatEarly().cuda()
    elif _model_architecture == 2:
        # to process CM_ and MM_ variations where d=50 per each modality (T,V,A)
        encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
        features_n = _features_n
        head = Sequential(Transformer(features_n, 150, 5).cuda(), MLP(indim=150, hiddim=80, outdim=2)).cuda() # 5 heads
        fusion = ConcatEarly().cuda()
        #fusion = TensorFusion().cuda()
    else:
        print("Error: _model_architecture should be 1 or 2")
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    mdl = MMDL(encoders, fusion, head, has_padding=False).to(device)
    
    if _load_precomputed and os.path.exists(_path_pt):
        print(f"Loading pretrained model: {_path_pt} ")
        checkpoint = torch.load(_path_pt)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            mdl.load_state_dict(checkpoint['model_state_dict'])
        else:
            mdl = checkpoint 
            
        mdl.eval() 
    else:
        print("Begin training...")
        print("Epochs:", _total_epochs)
        save_model_as = "trained_e" + str(_total_epochs)+ "_" + name + '.pt'
        mdl = train_mmdl(encoders, fusion, head, 
                traindata_all, validdata_all, 
                _total_epochs, 
                task="classification", 
                optimtype=torch.optim.AdamW,
                lr=1e-5, 
                early_stop=_early_stop,
                save=save_model_as, 
                weight_decay=0.01, 
                objective=torch.nn.CrossEntropyLoss(),
                track_complexity=False,
                is_packed=False)

    print("Testing...")
    test_mmdl(mdl, testdata_all, 'affect', 
                      is_packed=False,
                      criterion=torch.nn.CrossEntropyLoss(), 
                      task="classification", 
                      no_robust=True)
    