import torch
import torch.nn as nn
import random
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve
import torch.nn.functional as F
import os
from model import SsTGN_allfeature_newloss as SsTGN 
from data_process import cross_data_compute
import yaml
import argparse

def edge_loss(x, edge_index, edge_weight=None):
    if x.dim() == 1:
        x = x.view(-1, 1)
    row, col = edge_index
    
    x_i = x[row]
    x_j = x[col]
    
    squared_diff = torch.sum((x_i - x_j)**2, dim=1)
    
    if edge_weight is not None:
        loss = (squared_diff * edge_weight).sum()
    else:
        loss = squared_diff.sum()
    num_edges = edge_index.shape[1]
        
    return loss /(num_edges + 1e-8)

def test(model,valid_feature,alpha):
    set_seed(0)
    model.eval()
    loss_test=0
    acc_test=0
    auc_test=0
    pre_test=0
    spe_test=0
    sen_test=0
    f1_test=0
    mcc_test=0
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0),reduction='mean')
    with torch.no_grad():
        for i,data in enumerate(valid_feature[1]):
            data = data.to('cuda')
            out,edge,_ ,edge_weight= model(valid_feature[0][i][0],data.x,data.edge_index, data.batch,valid_feature[2][i][0],valid_feature[3][i][0])

        pred=F.sigmoid(out.squeeze(1)).cpu()
        loss_edge = edge_loss(data.y.float(), edge,edge_weight)
        loss = criterion(out.squeeze(1), data.y.squeeze(1).float())
        loss_test+=loss.item()+alpha*loss_edge.item()
        auc_batch=roc_auc_score(data.y.squeeze(1).cpu(),(pred).cpu())
        pred=(pred>0.5).float()
        print(confusion_matrix(data.y.squeeze(1).cpu(),(pred).cpu()))
        acc_batch=accuracy_score(data.y.squeeze(1).cpu(),(pred).cpu())
        pre_batch=precision_score(data.y.squeeze(1).cpu(),(pred).cpu())
        sen_batch=recall_score(data.y.squeeze(1).cpu(),(pred).cpu(),pos_label=1)
        spe_batch=recall_score(data.y.squeeze(1).cpu(),(pred).cpu(),pos_label=0)
        f1_batch=f1_score(data.y.squeeze(1).cpu(),(pred).cpu())
        mcc_batch=matthews_corrcoef(data.y.squeeze(1).cpu(),(pred).cpu())
        loss_test+=loss.item()
        ls_test=loss.item()*len(data.y.squeeze(1))
        acc_test+=acc_batch
        auc_test+=auc_batch
        pre_test+=pre_batch
        sen_test+=sen_batch
        spe_test+=spe_batch
        f1_test+=f1_batch
        mcc_test+=mcc_batch
    loss_test=loss_test/len(valid_feature[1])
    acc_test=acc_test/len(valid_feature[1])
    auc_test=auc_test/len(valid_feature[1])
    pre_test=pre_test/len(valid_feature[1])
    spe_test=spe_test/len(valid_feature[1])
    sen_test=sen_test/len(valid_feature[1])
    f1_test=f1_test/len(valid_feature[1])
    mcc_test=mcc_test/len(valid_feature[1])
    print(len(valid_feature[1]))

    return loss_test, acc_test,auc_test,pre_test,sen_test,spe_test,f1_test,mcc_test

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def get_files(path):
    files=[f for f in os.listdir(path)]
    return files
def names_deal_ck(root,alpha):
    files_ls=get_files(root)
    for f in files_ls:
        path='{}/{}'.format(root,f)
        
        
        value = cfg['settings']['sim_threshold1']
        value2 = cfg['settings']['sim_threshold2']
        value_o=cfg['settings']['drop_edge_num1']
        value_o2=cfg['settings']['drop_edge_num2']
        lr=cfg['lr']
        name='{}_{}_{}_{}'.format(value,value_o,value2,value_o2)
        model=SsTGN(cfg['settings']['finger_dim'],
                        cfg['settings']['graph_dim'],
                        cfg['settings']['smi_dim'],
                        cfg['settings']['des_dim'],
                        cfg['settings']['des_dim'],
                        cfg['settings']['out_dim'],value,value_o,value2,value_o2).cuda()
        model.load_state_dict(torch.load(path))

        test_loss, test_acc,test_auc,test_pre,spe_test,sen_test,test_f1,test_mcc=test(model,test_feature,alpha=alpha)
        print('Test set results:')
        print(f'Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, Pre: {test_pre:.4f}, Sen: {sen_test:.4f}, Spe: {spe_test:.4f}, F1: {test_f1:.4f}, MCC: {test_mcc:.4f}')
        result[f]=[test_loss, test_acc,test_auc,test_pre,spe_test,sen_test,test_f1,test_mcc]
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='path to the config file')
args = parser.parse_args()
cfg = load_config(args.config)
cross_data=cross_data_compute(cfg['paths']['graph_path'],
                              cfg['settings']['seed'],
                              cfg['data']['selected_features'],
                              cfg['batch_size'])
set_seed(cfg['settings']['seed'])
n=0
test_feature=cross_data[n][4]
test_batch_num=cross_data[n][5]
alpha=cfg['settings']['alpha']
ck_path=cfg['paths']['save_path']
#ck_path=cfg['paths']['model_path']
for ck_n in range(1):
    data_n=0
    ck_n=ck_n
    result=pd.DataFrame()
    result.index=['loss','acc','auc','pre','spe','sen','f1','mcc']
    #names_deal_ck('{}/ck_{}'.format(ck_path,ck_n),alpha)
    names_deal_ck('{}'.format(ck_path),alpha)
    result.to_csv('{}/{}_test_result_meta.csv'.format(ck_path,ck_n))
    result=pd.read_csv('{}/{}_test_result_meta.csv'.format(ck_path,ck_n))
    data_new=result.T
    data_new.columns=data_new.iloc[0]
    data_new=data_new[1:]

    data_new=data_new.sort_values(by='auc',ascending=False)
    data_new.to_csv('{}/{}_test_result_meta_new.csv'.format(ck_path,ck_n))