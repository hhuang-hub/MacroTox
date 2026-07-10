import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool,GATConv,SAGEConv,GINConv
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve
import torch.nn.functional as F
import os
from model import SsTGN_allfeature_newloss as SsTGN 
from graph_process import Graph_custom
from data_process import mordred_compute_all,reset,set_ls,maccs_compute,molformer_cal
from torch_geometric.loader import DataLoader as g_DataLoader
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def smis_to_allfeatures(smis,graph_root):
    df=pd.DataFrame({'smiles':smis,'lables':[0]*len(smis)})
    graph=Graph_custom(df,graph_root,smiles_col='smiles',label_col='lables')
    graph=torch.load(os.path.join(graph_root,"processed/data.pt"),weights_only=False)
    #graph=graph[0].cuda()
    graph=g_DataLoader(graph,batch_size=len(graph),shuffle=False)
    graph=set_ls(graph)[0].cuda()
    mean_path = os.path.join(BASE_DIR, "data", "des_mean.csv")
    std_path = os.path.join(BASE_DIR, "data", "des_std.csv")
    mean=pd.read_csv(mean_path)
    std=pd.read_csv(std_path)
    mean_values = pd.Series(mean.iloc[:,1].values, index=mean.iloc[:,0])
    std_values  = pd.Series(std.iloc[:,1].values, index=std.iloc[:,0])

    mean_name = mean.iloc[:,0].to_list()
    des=mordred_compute_all(smis,mean_name,mean_values,std_values)
    finger=maccs_compute(smis)
    smis_molformer=molformer_cal(smis)
    return graph,des,finger,smis_molformer

def predict(smis,graph_root):
    graph,des,finger,smis_molformer=smis_to_allfeatures(smis,graph_root)
    model_path=os.path.join(BASE_DIR, "model", "best_bone.pth")
    value = 0.6
    value2 = 0.6
    value_o=15
    value_o2=5
    model = SsTGN(167,12,768,65,65,1,value,value_o,value2,value_o2).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output,_,_,_ = model(finger,graph.x,graph.edge_index,graph.batch,smis_molformer,des)
        output=F.sigmoid(output)
        return output

def read_csv(file,smi):
    df=pd.read_csv(file)
    smis=df[smi].tolist()
    return smis

input=sys.argv[1]
output_path=sys.argv[2]
graph_name=sys.argv[3]
graph_path=os.path.join(BASE_DIR, "graph_new", graph_name)
if input.endswith(".csv"):
    smis=read_csv(input,"smiles")[:100]
else:
    smis=[input]
print('start predict')
toxic_pred=predict(smis,graph_path)
df=pd.DataFrame({'smiles':smis,'toxicity':toxic_pred.cpu().numpy().reshape(-1)})
df.to_csv(output_path,index=False,float_format='%.4f')

#CUDA_VISIBLE_DEVICES=2 python /dataStor/home/hehuang/cartox_predict/SL-MTGNN/predict.py "C1=CC=C(C=C1)C(=O)O" "output.csv" "graph_root"
