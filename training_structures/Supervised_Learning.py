"""Implements supervised learning training procedures."""
import torch
from torch import nn
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
#import pdb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

softmax = nn.Softmax()


class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #print("\n func deal_with_objective:")
    #print("criterion:", type(objective))
    #print("PRED", pred.size(), pred[:5])#pred.size(), type(pred))
    #print("TRUTH", truth.size(), truth)#truth.size(), type(truth))

    if type(objective) == nn.CrossEntropyLoss:
        # check if values for TRUTH
        #print(type(truth.size()), type(pred.size()))
        # len(truth.size()) == len(pred.size()):
        #    truth = truth.squeeze(len(pred.size())-1)
        #else:
        #    truth = truth
        
        if (truth == -1).any():
            truth[truth == -1] = 0
            truth1 = truth 
        else:
            truth1 = truth
        #print("TRUTH", truth1)
        # check values for PRED
        #sigmoid_predictions = torch.sigmoid(pred)
        #print("sigmoid preds", sigmoid_predictions)
        #return objective(pred, truth1.to(device))
        return nn.functional.binary_cross_entropy_with_logits(input=pred, target=truth1.to(device))        
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)




def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, 
        clip_val=1.0,
        track_complexity=False):
    """
    Handle running a simple supervised training loop.
    
    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    valloss_values = [] # for plotting
    trainloss_values = []

    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] + additional_params, lr=lr, weight_decay=weight_decay)
        #print("Using optimizer", op.load_state_dict)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if torch.isnan(inp).any():
                print(f"NaN detected in model input.")
                #continue  # Salta este batch si los resultados son NaN

            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for batch in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device) for i in batch[0]], batch[1]])
                else:
                    model.train()
                    out = model([_processinput(i).to(device) for i in batch[:-1]]) #preds
                    #print("out", out[:10])
                    if torch.isnan(out).any():
                        print(f"NaN detected in model output.")
                        continue
            
                
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = batch[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                
                # j[-1] are truth labels
                loss = deal_with_objective(objective=objective, pred=out, truth=batch[-1], args=objective_args_dict)
                print("loss", loss)

                totalloss += loss * len(batch[:-1])
                totals += len(batch[-1])
                
                loss.backward()
                #with torch.autograd.detect_anomaly():
                #    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()

            # Calculate the training loss and training accuracy
            trainloss = totalloss/ totals
            trainloss_values.append(trainloss.item())
            print("EPOCH "+str(epoch)+" TRAIN LOSS: "+ str(trainloss.item()) )
            
            ####
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    #print("lote de datos j", j)
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i).to(device)
                                    for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
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
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            
            if task == "posneg-classification":
                # added by me
                #print("task:", task)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                #print("Epoch: "+str(epoch))
                valloss_values.append(valloss.item())
                #
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save) # saves the model
                    #torch.save(model.state_dict(), save) # saves only the weights
                else:
                    patience += 1
            
            elif task == "classification":
                print("TASK:", task)
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) + " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save) #saves all the model
                    #torch.save(model.state_dict(), save) #saves only the weights
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                #print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                #      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                    #torch.save(model.state_dict(), save) #saves only the weights
                else:
                    patience += 1
            elif task == "regression":
                #print("task:", task)
                #print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                valloss_values.append(valloss.item())
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save) # saves the model
                    #torch.save(model.state_dict(), save) # saves only the weights
                else:
                    patience += 1
            
            print("PATIENCE=", patience)
            if early_stop and patience > 10:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
    if track_complexity:
        all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        print("else _trainprocess()")
        _trainprocess()

    plt.plot(trainloss_values, label="Train Loss")
    plt.plot(valloss_values, label="Validation Loss")

    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model


def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """
    #
    # prints to review modality dimensionality
    #print((test_dataloader['train']['audio'].shape))
    #print((alldata['train']['vision'].shape))
    #print((alldata['train']['text'].shape))

    #
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
        
    #model.eval()
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        preds=[]
        true = []
        pts = []
        for batch in test_dataloader:
            model.eval()
            if is_packed:
                print("is_packed=True")
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in batch[0]], batch[1]])
            else:
                print("is_packed=False")
                out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                            for i in batch[:-1]])
                #print("out from model", out)
            """
            criterions
            """
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, batch[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                #print("criterion: CrossEntropy")
                #if len(j[-1].size()) == len(out.size()):
                #    truth1 = j[-1].squeeze(len(out.size())-1)
                #else:
                #    truth1 = j[-1]
                #loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
                loss = deal_with_objective(criterion, out, batch[-1], None)
                print("loss", loss)
            else:
                loss = criterion(out, batch[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            
            totalloss += loss*len(batch[-1])
            print("totalloss", totalloss)
            """
            tasks
            """
            if task == "classification":
                #pred.append(torch.argmax(out, 1))
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    #print("i",i)
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(0)
                    else:
                        prede.append(0)
                
                pred.append(torch.LongTensor(prede))
                #print("pred", len(pred), pred)

            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    print("i",i)
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
                print("pred", type(pred), len(pred), pred[:10])

            true.append(batch[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), batch[-1][i].item())
                        for i in range(batch[-1].size(0))]
        
        if pred:
            pred = torch.cat(pred, 0)
        #print("pred", pred)
        true = torch.cat(true, 0)
        #print("true", true)
        totals = true.shape[0]
        testloss = totalloss/totals
        
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            #print("acc: "+str(accuracy(true, pred)))
            #return {'Accuracy': accuracy(true, pred)}
            lst_pred = pred.numpy()
            print("lst_pred", lst_pred)
            trues = []
            for e in true:
                label = int(e.item())
                trues.append(label)
            lst_true = np.array(trues)
            print("lst_true", lst_true)
            
            report = classification_report(y_true=lst_true, y_pred=lst_pred, digits=4)
            print(report)
            return lst_pred #report

        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs, macro_f1_score, f1_score, p, r = eval_affect(trueposneg, pred)
            #acc2, macro_f1_score2, f1_score2, p2, r2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("accs: "+str(accs))
            print("macro_f1_score: "+ str(macro_f1_score))
            print("f1_score: "+ str(f1_score))
            print("p: "+ str(p))
            print("r: "+ str(r))
            
            #print("acc2: "+str(acc2))
            #print("macro_f1_score2: "+ str(macro_f1_score2))
            #print("f1_score2: "+ str(f1_score2))
            #print("p2: "+ str(p2))
            #print("r2: "+ str(r2))
            
            # added more metrics
            return {'Accuracy': accs}



def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False):
    """
    Handle getting test results for a simple supervised training loop.
    
    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there   
    """
    if no_robust:
        print("if no_robust True")
        def _testprocess():
            single_test(model, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(_testprocess, [model])
        return

    print("if no_robust False")
    def _testprocess():
        #single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[
        #            0]][0], is_packed, criterion, task, auprc, input_to_float)
        single_test(model, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(_testprocess, [model])
    

        for noisy_modality, test_dataloaders in test_dataloaders_all.items():
            print("Testing on noisy data ({})...".format(noisy_modality))
            robustness_curve = dict()
            for test_dataloader in tqdm(test_dataloaders):
                single_test_result = single_test(
                    model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
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


