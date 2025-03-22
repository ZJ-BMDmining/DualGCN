import os
import pickle
# import time
# # os.environ ['CUDA_VISIBLE_DEVICES']='0'
# import torch as t
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import torch
from torch import nn

import numpy as np


import shap
import time
shap.initjs()

import torch as t 

import torch.nn.functional as F

from sklearn.metrics import precision_score,f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

from dataset import (collate_func,DataLoader,ExprDataset)
from scheduler import CosineAnnealingWarmRestarts
from model import (DualGCN,edge_transform_func)

pathjoin = os.path.join







def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    # parser.add_argument('-in','--infile',type=str,)
    parser.add_argument('-in', '--infile', type=str,
                        default='./PPMI-8种疾病状态亚型分类/dataset.npz')
    parser.add_argument('-out','--outdir',type=str,default='./')
    parser.add_argument('-cuda','--cuda',type=bool,default=True)
    parser.add_argument('-bs','--batch_size',type=int,default=32)
    return parser





def te(model,loader,predicts=False,feature=False):
    model.eval()

    correct = 0
    features_array=[]
    y_pred =[]
    y_true=[]
    y_output=[]
    for data in loader:
        data = data.to(device)
        # print(data.y.shape)
        output,features = model(data)    #模型多了一个features的输出
        pred = output.max(dim=1)[1].cpu().data.numpy()
        y = data.y.cpu().data.numpy()
        y_pred.extend(pred)
        y_true.extend(y)
        y_output.extend(output.cpu().data.numpy())
        if feature:
            features_array.append(features.cpu().data.numpy())




    if feature:
        with open("feature.pkl", "wb") as f:
            pickle.dump([features_array, y_pred,y_true], f)

    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average='macro')
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average='macro')
    if predicts:
         return acc,f1,recall,precision,y_true,np.array(y_pred),y_output
    else:
        return acc,f1,recall,precision



def train2(model,optimizer,train_loader,epoch,device,loss_fn =None,scheduler =None,verbose=False  ):
    model.train()

    loss_all = 0
    iters = len(train_loader)
    for idx,data in enumerate( train_loader):
        # print('epoch,idx',epoch,idx)
        # data.x = data.x + t.rand_like(data.x)*1e-5
        data = data.to(device)
        if verbose:
            print(data.y.shape,data.edge_index.shape)
        optimizer.zero_grad()
        output ,feature= model(data)
        if loss_fn is None:
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None,)
        else:
            loss = loss_fn(output, data.y.reshape(-1))

        if model.edge_weight is not None:
            l2_loss = 0 
            if isinstance(model.edge_weight,nn.Module):  # nn.ParamterList
                for edge_weight in model.edge_weight :
                    l2_loss += 0.1* t.mean((edge_weight)**2)
            elif isinstance(model.edge_weight,t.Tensor):
                l2_loss =0.1* t.mean((model.edge_weight)**2)
            # print(loss.cpu().detach().numpy(),l2_loss.cpu().detach().numpy())
            loss+=l2_loss


        
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        if not (scheduler is  None):
            scheduler.step( (epoch -1) + idx/iters) # let "epoch" begin from 0 

    return loss_all / iters # len(train_dataset)




class Logging(object):
    def __init__(self, name='log.txt',root = './'):
        self.name = os.path.join(root,name)
        self.write_format = "{time_str} " \
                            "[Epoch] {e} " \
                            "[train_loss] {train_loss:.4f} " \
                            "[ACC] {ACC:.4f} " \
                            "[Recall] {Recall:.4f} " \
                            "[F1-score] {F1_score:.4f} " \
                            "[Precision score] {Precision_score:.4f}\n"

        # with open(self.name, 'w') as f:
        #     f.write(f"[Seed] {seed} [Lr] {lr} [Batch] {batch_size}\n")

    def write_(self, *args):
        e, train_loss,  ACC, Recall, F1_score, Precision_score = args
        data = self.write_format.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),e=e, train_loss=train_loss, ACC=ACC, Recall=Recall, F1_score=F1_score,Precision_score=Precision_score)
        with open(self.name, "a") as f:
            f.write(data)

if __name__ == '__main__':


    parser = get_parser()
    args = parser.parse_args()
    print('args:',args)

    cuda_flag = args.cuda
    npz_file = args.infile
    save_folder = args.outdir
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag  else 'cpu')
    #     device = torch.device('cpu')

    prob_file =  pathjoin(save_folder,'predicted_probabilities.txt')
    y_score_train_file=pathjoin(save_folder,'predicted_probabilities_train.txt')
    y_score_valid_file=pathjoin(save_folder,'predicted_probabilities_valid.txt')
    pred_file_train = pathjoin(save_folder,'predicted_label_train.txt')
    true_file_train = pathjoin(save_folder,'true_label_train.txt')
    pred_file_valid = pathjoin(save_folder, 'predicted_label_valid.txt')
    true_file_valid = pathjoin(save_folder, 'true_label_valid.txt')
    os.makedirs(pathjoin(save_folder,'models'),exist_ok=True)

    data= np.load(npz_file,allow_pickle=True)
    print(data)
    logExpr = data['logExpr'].T

    str_labels = data['str_labels']
    used_edge = edge_transform_func(data['edge_index'],)

    num_samples = logExpr.shape[0]


    init_lr =0.01
    min_lr = 0.0001
    max_epoch= 500
    weight_decay  = 1e-4
    weight_decay  = 1e-4
    dropout_ratio = 0.1

    print('use wegithed cross entropy.... ')
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([ np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha,1,20)
    alpha = alpha/ np.sum(alpha)
    loss_fn = t.nn.CrossEntropyLoss(weight = t.tensor(alpha).float())
    loss_fn = loss_fn.to(device)


    dataset = ExprDataset(Expr=logExpr,edge=used_edge,y=label,device=device)
    gene_num = dataset.gene_num
    class_num = len(np.unique(label))


    kf = StratifiedKFold(n_splits=3,shuffle=True)
    for tr, ts in kf.split(X=label,y=label):
        train_index = tr
        test_index = ts

    train_dataset = dataset.split(t.tensor(train_index).long())
    test_dataset = dataset.split(t.tensor(test_index).long())
    # add more samples for those small celltypes
    train_dataset.duplicate_minor_types(dup_odds=5)
    test_dataset.duplicate_minor_types(dup_odds=5)

    num_workers = 0
    assert num_workers == 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              collate_fn=collate_func, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_func)


    model = DualGCN(in_channel = dataset.num_expr_feaure , num_nodes=gene_num,
                out_channel=class_num,edge_num=dataset.edge_num,
                dropout_ratio = dropout_ratio,
                ).to(device)

    print(model)


    optimizer = torch.optim.Adam(model.parameters(),lr=init_lr ,weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer,2, 2, eta_min=min_lr, lr_max_decay=0.5)
    max_metric = float(0)
    max_metric_count = 0
    weights_list = []

    Log = Logging(name='log.txt')




    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_pre = 0
    for epoch in range(1, max_epoch):
        train_loss = train2(model,optimizer,train_loader,epoch,device,loss_fn,scheduler =scheduler )
        train_acc,train_f1,train_recall,train_precision= te(model,train_loader,predicts=False)
        lr = optimizer.param_groups[0]['lr']

        print('epoch\t%03d,lr : %.06f,loss: %.06f,T-acc: %.04f,T-f1: %.04f,T-recall: %.04f,T-pre:%.04f'%(
                    epoch,lr,train_loss,train_acc,train_f1,train_recall,train_precision))

        Log.write_(epoch + 1, train_loss, train_acc, train_f1, train_recall, train_precision)

        test_acc1, test_f11, test_recall1, test_precision1, test_true, test_pred, y_score_test = te(model,
                                                                                                       test_loader,
                                                                                                       predicts=True,
                                                                                                       feature=True)
        print('Test-acc: %.04f,Test-f1: %.04f,Test-recall: %.04f,Test-pre:%.04f' % (
            test_acc1, test_f11, test_recall1, test_precision1))

        if best_acc < test_acc1:
            best_acc = test_acc1
        if best_f1 < test_f11:
            best_f1 = test_f11
        if best_recall < test_recall1:
            best_recall = test_recall1
        if best_pre < test_precision1:
            best_pre = test_precision1
    print('best-acc: %.04f,best-f1: %.04f,best-recall: %.04f,best-pre:%.04f' % (
        best_acc, best_f1, best_recall, best_pre))




    #stage two
    extend_epoch = 200
    print('stage 2 training...')
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for final_, valid_ in sss.split(dataset.y[train_index],dataset.y[train_index]):
        train_index2,valid_index2 =train_index[final_],train_index[valid_]
    valid_dataset = dataset.split(t.tensor(valid_index2).long())
    train_dataset = dataset.split(t.tensor(train_index2).long())
    if True:
        train_dataset.duplicate_minor_types(dup_odds=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True,collate_fn = collate_func,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True,collate_fn = collate_func)
    lr = optimizer.param_groups[0]['lr']
    print('stage2 initilize lr:',lr)

    max_metric = float(0)
    max_metric_count = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=lr ,weight_decay=weight_decay,)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.1, patience=2, verbose=True,min_lr=0.00001)
    old_lr = lr
    for epoch_idx,epoch in enumerate(range(max_epoch,(max_epoch+extend_epoch))):
        if old_lr != lr:
            max_metric_count = 0
            print('reset max_metric_count to 0 due to updating lr from %f to %f'%(old_lr,lr))
            old_lr = lr

        train_loss = train2(model,optimizer,train_loader,epoch,device,loss_fn,verbose=False  )


        valid_acc,valid_f1,valid_recall,valid_precision,valid_true,valid_pred,y_score_valid= te(model,valid_loader,predicts=True)

        lr = optimizer.param_groups[0]['lr']

        print(
            'epoch\t%03d,Test_lr : %.06f,loss: %.06f,valid-acc: %.04f,valid-f1: %.04f,valid-recall:%04f,valid-precision:%04f' % (
            epoch,
            lr, train_loss,  valid_acc, valid_f1, valid_recall,
            valid_precision))
        scheduler.step(valid_f1)
        lr = optimizer.param_groups[0]['lr']

        if valid_f1 >max_metric:
            max_metric=valid_f1
            tmp_file = pathjoin(save_folder,'models','model.pth')
            weights_list.append(tmp_file)
            t.save(model,tmp_file)
            max_metric_count=0
            max_metric=valid_f1
        else:
            if epoch_idx >=2: #ignore first two epochs
                max_metric_count+=1
            if max_metric_count >3:
                print('break at epoch',epoch)
                break

        if lr <= 0.00001:
            break


    test_acc, test_f1, test_recall, test_precision, test_true, test_pred, y_score_test = te(model,
                                                                                               test_loader,
                                                                                               predicts=True)
