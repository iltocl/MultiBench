"""Implements training pipeline for unimodal comparison."""
#from sklearn.metrics import accuracy_score, f1_score
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect

import torch
from torch import nn
from utils.AUPRC import AUPRC
from eval_scripts.performance import eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
softmax = nn.Softmax()

from sklearn.metrics import classification_report
import numpy as np

import matplotlib.pyplot as plt

from training_structures.Supervised_Learning import deal_with_objective # added

def train(encoder, head, train_dataloader, valid_dataloader, total_epochs, early_stop=False, optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0, criterion=nn.CrossEntropyLoss(), auprc=False, save_encoder='encoder.pt', save_head='head.pt', modalnum=0, task='classification', track_complexity=True, 
          additional_optimizing_modules=[], objective_args_dict=None, input_to_float=True):
    """Train unimodal module.

    Args:
        encoder (nn.Module): Unimodal encodder for the modality
        head (nn.Module): Takes in the unimodal encoder output and produces the final prediction.
        train_dataloader (torch.utils.data.DataLoader): Training data dataloader
        valid_dataloader (torch.utils.data.DataLoader): Validation set dataloader
        total_epochs (int): Total number of epochs
        early_stop (bool, optional): Whether to apply early-stopping or not. Defaults to False.
        optimtype (torch.optim.Optimizer, optional): Type of optimizer to use. Defaults to torch.optim.RMSprop.
        lr (float, optional): Learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay of optimizer. Defaults to 0.0.
        criterion (nn.Module, optional): Loss module. Defaults to nn.CrossEntropyLoss().
        auprc (bool, optional): Whether to compute AUPRC score or not. Defaults to False.
        save_encoder (str, optional): Path of file to save model with best validation performance. Defaults to 'encoder.pt'.
        save_head (str, optional): Path fo file to save head with best validation performance. Defaults to 'head.pt'.
        modalnum (int, optional): Which modality to apply encoder to. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        track_complexity (bool, optional): Whether to track the model's complexity or not. Defaults to True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(encoder, head).to(device)
    # Esto te avisará apenas ocurra el primer NaN en los gradientes
    torch.autograd.set_detect_anomaly(True)
        
    batch = next(iter(train_dataloader)) # to verify the labels
    labels = batch[-1]
    unique = torch.unique(labels) 
    print(f"Labels: {labels.shape}, {labels.dtype}, {unique}")
    
    trainloss_values =  []
    vallloss_values = []
    acc_values = []

    def _trainprocess():
        #additional_params = []
        #for m in additional_optimizing_modules:
        #    additional_params.extend([p for p in m.parameters() if p.requires_grad])

        op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
        #op = optimtype([p for p in model.parameters() if p.requires_grad] + additional_params, lr=lr, weight_decay=weight_decay)
        
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            inp = inp.float()
            if torch.isnan(inp).any():
                inp = torch.nan_to_num(inp, nan=0.0)
                print("NaN", inp) 
            return inp
        
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train() # added
            
            for batch in train_dataloader:
                op.zero_grad()
                
                #out = model(j[modalnum].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                out = model(_processinput(batch[modalnum]).float().to(device))

                if torch.isnan(out).any():
                    print(f"Epoch {epoch}: NaN detected.")
                    continue
                
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss = criterion(out, batch[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                else:
                    #loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = batch[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(criterion, out, batch[-1], objective_args_dict) #added
                # --- train loss ---
                totalloss += loss * len(batch[-1])
                totals += len(batch[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                op.step()

            trainloss = totalloss/totals
            trainloss_values.append(trainloss)
            print("Epoch "+str(epoch)+" train loss: "+str(trainloss))
            
            # ---- validating the model ----
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    out = model(j[modalnum].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                    
                    if torch.isnan(out).any():
                        print(f"Validation. NaN detected.")
                        continue

                    if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                        loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                    else:
                        #loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                        if not (objective_args_dict is None):
                            objective_args_dict['reps'] = model.reps
                            objective_args_dict['fused'] = model.fuseout
                            objective_args_dict['inputs'] = j[:-1]
                            objective_args_dict['training'] = False
                        loss = deal_with_objective(criterion, out, j[-1], objective_args_dict) #added
                    
                    totalloss += loss*len(j[-1])

                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                #pred = torch.cat(pred, 0).cpu().numpy()
                pred = torch.cat(pred, 0)
            #true = torch.cat(true, 0).cpu().numpy()
            true = torch.cat(true,0)
            
            totals = true.shape[0]
            valloss = totalloss/totals
            vallloss_values.append(valloss)

            if task == "classification":
                #acc = accuracy_score(true, pred)
                acc = accuracy(true,pred)
                acc_values.append(acc)
                #print("Epoch "+str(epoch)+" valid loss: "+str(valloss) + " acc: "+str(acc))
                print(f"Epoch {epoch}, valid loss: {valloss} ,acc: {acc}")
                print(f"val acc: {acc:.4f} - best acc: {bestacc:.4f}")
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1

            # ------- early stopping --------
            if early_stop and patience > 12:
                print(f"Stopped by early_stop and patience. Epoch {epoch}")
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))

            #trainloss = totalloss/ totals
            #trainloss_values.append(trainloss.item())
            # end epoch
        return model
    if track_complexity:
        all_in_one_train(_trainprocess, [encoder, head])
    else:
        _trainprocess()
    
    # ---- plotting loss -------------------------------------------
    plt.figure()
    plt.plot(torch.tensor(trainloss_values).cpu().detach().numpy(), label="Train Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(torch.tensor(acc_values).cpu().detach().numpy(), label="Acc")
    plt.title("Acc Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()

    # end train()


def single_test(encoder, head, test_dataloader, auprc=False, modalnum=0, task='classification', criterion=None):
    """Test unimodal model on one dataloader.

    Args:
        encoder (nn.Module): Unimodal encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloader (torch.utils.data.DataLoader): Data Loader for test set.
        auprc (bool, optional): Whether to output AUPRC or not. Defaults to False.
        modalnum (int, optional): Index of modality to consider for the test with the given encoder. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.

    Returns:
        dict: Dictionary of (metric, value) relations.
    """
    print("def single_test")
    model = nn.Sequential(encoder, head)
    with torch.no_grad():
        pred = []
        true = []
        totalloss = 0
        pts = []
        for j in test_dataloader:
            model.eval()##
            out = model(j[modalnum].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            
            if criterion is not None:
                #loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                if type(criterion) == nn.CrossEntropyLoss:
                    #print("criterion: CrossEntropy")
                    if len(j[-1].size()) == len(out.size()):
                        truth1 = j[-1].squeeze(len(out.size())-1)
                    else:
                        truth1 = j[-1]
                    loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                else:
                    loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                totalloss += loss*len(j[-1])
                #print("totalloss", totalloss)
                

            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if criterion is not None:
            print("loss: " + str(totalloss / totals))
        if task == "classification":
            lst_pred = pred
            print("lst_pred", lst_pred)
            trues = []
            for e in true:
                label = int(e.item())
                trues.append(label)
            lst_true = np.array(trues)
            lst_true[lst_true == -1] = 0
            print("lst_true", lst_true)
            report = classification_report(y_true=lst_true, y_pred=lst_pred, digits=4)
            print(report)
            return lst_pred
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'F1 score (micro)': f1_score(true, pred, average="micro"), 'F1 score (macro)': f1_score(true, pred, average="macro")}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}
        else:
            return {'MSE': (totalloss / totals).item()}


def test(encoder, head, test_dataloaders_all, dataset='default', method_name='My method', auprc=False, modalnum=0, task='classification', criterion=None, no_robust=False):
    """Test unimodal model on all provided dataloaders.

    Args:
        encoder (nn.Module): Encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloaders_all (dict): Dictionary of noisetype, dataloader to test.
        dataset (str, optional): Dataset to test on. Defaults to 'default'.
        method_name (str, optional): Method name. Defaults to 'My method'.
        auprc (bool, optional): Whether to output AUPRC scores or not. Defaults to False.
        modalnum (int, optional): Index of modality to test on. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.
        no_robust (bool, optional): Whether to not apply robustness methods or not. Defaults to False.
    """
    if no_robust:
        def _testprocess():
            single_test(encoder, head, test_dataloaders_all,
                        auprc, modalnum, task, criterion)
        all_in_one_test(_testprocess, [encoder, head])
        return

    def _testprocess():
        single_test(encoder, head, test_dataloaders_all[list(
            test_dataloaders_all.keys())[0]][0], auprc, modalnum, task, criterion)
    all_in_one_test(_testprocess, [encoder, head])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                encoder, head, test_dataloader, auprc, modalnum, task, criterion)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)
