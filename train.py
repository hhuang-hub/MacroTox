import yaml
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import random
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
import torch.nn.functional as F
from model import SsTGN_allfeature_newloss as SsTGN 
from data_process import cross_data_compute

def test(model,valid_feature,valid_batch_num,alpha):
    #set_seed(0)
    model.eval()
    loss_test=0
    acc_test=0
    auc_test=0
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
            auc_test+=auc_batch
            pred=(pred>0.5).float()
            acc_batch=accuracy_score(data.y.squeeze(1).cpu(),(pred).cpu())
            acc_test+=acc_batch

    return loss_test/valid_batch_num, acc_test/valid_batch_num,auc_test/valid_batch_num

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


def train(model,train_feature,train_batch_num,valid_feature,valid_batch_num,test_feature,test_batch_num,lr,name,data_num,save_path,alpha):
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0),reduction='mean')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=20,T_mult=1,eta_min=1e-8)
    loss_list=[]
    acc_list=[]
    auc_list=[]
    loss_list_valid=[]
    loss_edge_list=[]
    best_auc=0
    best_loss=0
    best_acc=0
    final_auc=0
    final_loss=0
    final_acc=0
    n=0
    for epoch in range(5000):
        loss_edge_origin=0
        n+=1
        model.train()
        loss=0
        acc=0
        for i,data in enumerate(train_feature[1]):
            data = data.to('cuda')
            out,edge,_,edge_weight = model(train_feature[0][i][0],data.x,data.edge_index, data.batch,train_feature[2][i][0],train_feature[3][i][0])
            assert len(data.y.squeeze(1).unique())==2
            loss_label = criterion(out.squeeze(1), data.y.squeeze(1).float())
            
            pred=F.sigmoid(out.squeeze(1)).cpu()
            
            pred=(pred>0.5).float()
            loss_edge = edge_loss(data.y.float(), edge,edge_weight)
            
            loss_batch = loss_label + alpha*loss_edge
            
            loss_batch.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            loss_edge_origin=loss_edge.item()+loss_edge_origin
            loss+=loss_batch.item()*len(out.squeeze(1))
            acc_batch=accuracy_score(data.y.squeeze(1).cpu(),(pred).cpu())
            acc+=acc_batch*len(out.squeeze(1))
        loss_list.append(loss/train_batch_num)
        loss_edge_list.append(loss_edge_origin/train_batch_num)
        acc_list.append(acc/train_batch_num)
        print(f"Epoch {epoch}, Loss: {loss_list[-1]}, Accuracy: {acc_list[-1]}")
        valid_loss,valid_acc,valid_auc=test(model,valid_feature,valid_batch_num,alpha)
        loss_list_valid.append(valid_loss)
        scheduler.step()
        print(f"Valid Loss: {valid_loss}, Valid Accuracy: {valid_acc}, Valid AUC:{valid_auc}")
        

        if valid_auc>best_auc:
            n=0
            best_auc=valid_auc
            best_loss=valid_loss
            best_acc=valid_acc
            torch.save(model.state_dict(),'{}/ck_{}/best_{}.pth'.format(save_path,data_num,name))
        #if epoch%10==0:
        #    torch.save(model.state_dict(),'checkpoint_graph_new/cartox_graph_epoch{}.pth'.format(epoch))
        if n>cfg['num_epochs']:
            final_auc=valid_auc
            final_loss=valid_loss
            final_acc=valid_acc
            torch.save(model.state_dict(),'{}/ck_{}/final_{}.pth'.format(save_path,data_num,name))
            break
    return best_loss,best_acc,best_auc,final_loss,final_acc,final_auc,loss_list,loss_list_valid,auc_list,loss_edge_list

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
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
result_final_df=pd.DataFrame()
save_Path=cfg['paths']['save_path']
os.makedirs(save_Path,exist_ok=True)
data_n=0
alpha=cfg['settings']['alpha']
for lr in [cfg['lr']]:
    for data_num,data in enumerate(cross_data):
        ls_ls=[]
        auc_ls=[]
        param=[]
        data_num_final=data_num+data_n
        os.makedirs('{}/ck_{}'.format(save_Path,data_num_final),exist_ok=True)
        os.makedirs('{}/result_{}'.format(save_Path,data_num_final),exist_ok=True)
        def objective(trial):
            set_seed(cfg['settings']['seed'])
            value = cfg['settings']['sim_threshold1']
            value2 = cfg['settings']['sim_threshold2']
            value_o=cfg['settings']['drop_edge_num1']
            value_o2=cfg['settings']['drop_edge_num2']
            a=trial.suggest_float('name', 0, 1)
            #lr=0.0005
            model=SsTGN(cfg['settings']['finger_dim'],
                        cfg['settings']['graph_dim'],
                        cfg['settings']['smi_dim'],
                        cfg['settings']['des_dim'],
                        cfg['settings']['des_dim'],
                        cfg['settings']['out_dim'],value,value_o,value2,value_o2).cuda()
            result=pd.DataFrame()
            auc_max_final=0
            global data_num,save_path,lr
            #name='{}_{}_{}_{}'.format(value,value2,value_o,value_o2)
            name='{}'.format(a)
            best_test_loss,best_test_acc,best_test_auc,final_test_loss,final_test_acc,final_test_auc,loss_list,loss_list_valid,auc_list_valid,edge_ls=train(model,data[0],data[1],data[2],data[3],data[4],data[5],lr,name,data_num_final,save_path=save_Path,alpha=alpha)
            ls_ls.append([loss_list,loss_list_valid])
            auc_ls.append(auc_list_valid)

            
            plt.figure()
            plt.plot(loss_list,label='train')
            plt.plot(loss_list_valid,label='valid')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig('{}/result_{}/{}_loss_meta.png'.format(save_Path,data_num_final,name),dpi=300)
            plt.clf()
            plt.figure()
            plt.plot(auc_list_valid,label='valid AUC')
            plt.xlabel('epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.savefig('{}/result_{}/{}_auc_meta.png'.format(save_Path,data_num_final,name),dpi=300)
            plt.clf()
            plt.figure()
            plt.plot(edge_ls,label='edge loss')
            plt.xlabel('epoch')
            plt.ylabel('edge loss')
            plt.legend()
            plt.savefig('{}/result_{}/{}_edge_loss_meta.png'.format(save_Path,data_num_final,name),dpi=300)
            plt.clf()

            return best_test_auc
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
            param.append(value)