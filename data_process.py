import torch
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as g_DataLoader
from torch.utils.data import DataLoader as d_DataLoader
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import random
import os
from torch.utils.data import Subset
import math
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from graph_process import load_config


def mordred_compute(smis,labels,des_name,cluster=False):
    calc=Calculator(descriptors, ignore_3D=True)
    smis_copy=smis.copy()
    mols=[]
    for smi in smis:
        mol=Chem.MolFromSmiles(smi)
        if mol is None:
            smis_copy.remove(smi)
            del labels[smis.index(smi)]
        else:
            mols.append(mol)
    df_des=calc.pandas(mols)
    df1_new=df_des[des_name]
    name=[]
    idx_=[]
    #去除无效数据，即删去行
    for i in df1_new.columns[:]:
        idx1=[]
        idx2=df1_new[i].index
        for n,des in enumerate(df1_new[i]):
            if isinstance(des,float) or isinstance(des,int):
                pass
            else:
                idx1.append(idx2[n])
        if len(idx1)>20:
            name.append(i)
        else:
            idx_.extend(idx1)
    idx_=list(set(idx_))
    labels_new=[]
    smis_new=[]
    for i in range(len(labels)):
        if i in idx_:
            pass
        else:
            labels_new.append(labels[i])
            smis_new.append(smis_copy[i])
    df1_new=df1_new.drop(idx_,axis=0)
    df1_new=df1_new.drop(name,axis=1)
    name_final=list(df1_new.columns)
    df1_new=df1_new.dropna()
    if cluster==True:
        feature_des=torch.tensor(df1_new.values.astype(float),dtype=torch.float)
    else:
        feature_des=df1_new
        #labels_new=torch.tensor(labels_new,dtype=torch.float).cuda()
    return feature_des,name_final,smis_new,labels_new

def des_name_deal(des_name_train,des_name_valid):
    des_name=[]
    des_name.append(des_name_train)
    des_name.append(des_name_valid)
    df={}
    des_name_final=[]
    idx_des_final=[]
    for i,names in enumerate(des_name):
        for name in names:
            df[name]=df.get(name,0)+1
    for k,v in df.items():
        if v==2:
            des_name_final.append(k)
            idx_des_final.append(list(des_name_train).index(k))
    return des_name_final

def val_split(a):
    y_ls=[i.y.item() for i in a]
    assert len(set(y_ls))==2,'data should have two classes'

def graph_deal(graph,smis):
    graph_new=[]
    for i in graph:
        smi=i.smiles
        if smi in smis:
            graph_new.append(i)
    return graph_new


def mordred_compute_all(smis,des_name_final,mean,std):
    calc=Calculator(descriptors, ignore_3D=True)
    smis_copy=smis.copy()
    mols=[]
    for smi in smis:
        mol=Chem.MolFromSmiles(smi)
        mols.append(mol)
    df_mordred=calc.pandas(mols)
    df_mordred_final=df_mordred[des_name_final]
    df_mordred_final=(df_mordred_final-mean)/std
    des=torch.tensor(df_mordred_final.values.astype(float),dtype=torch.float).cuda()
    return des
    
def cross_split_ablation_allfeature(a,des_name_original,batch_size,seed):
    train_graph,test_graph=train_test_split(a,test_size=0.1,random_state=seed)
    test_des_df,test_des_name,test_smis,test_labels=mordred_compute([i.smiles for i in test_graph],[i.y.item() for i in test_graph],des_name_original)
    test_graph_new=graph_deal(test_graph,test_smis)

    val_split(test_graph_new)
    test_fingerprint=maccs_compute(test_smis)
    test_smis=molformer_cal(test_smis)

    train_finger_df,train_finger_name,train_smis,train_labels=mordred_compute([i.smiles for i in train_graph],[i.y.item() for i in train_graph],des_name_original)
    des_name_final=des_name_deal(test_des_name,train_finger_name)
    print(des_name_final)
    test_des=test_des_df[des_name_final]
    train_des=train_finger_df[des_name_final]
    mean,std=train_des.mean(),train_des.std()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mean_path = os.path.join(BASE_DIR, "data", "des_mean.csv")
    std_path = os.path.join(BASE_DIR, "data", "des_std.csv")
    mean.to_csv(mean_path)
    std.to_csv(std_path)
    test_des=(test_des-mean)/std
    test_des=torch.tensor(test_des.values.astype(float),dtype=torch.float).cuda()

    test_des_final=d_DataLoader(TensorDataset(test_des),batch_size=len(test_des),shuffle=False)
    test_graph_final=g_DataLoader(test_graph_new,batch_size=len(test_graph_new),shuffle=False)
    test_finger_final=d_DataLoader(TensorDataset(test_fingerprint),batch_size=len(test_fingerprint),shuffle=False)
    test_smis_final=d_DataLoader(TensorDataset(test_smis),batch_size=len(test_smis),shuffle=False)
    
    test_des_final=set_ls(test_des_final)
    test_finger_final=set_ls(test_finger_final)
    test_smis_final=set_ls(test_smis_final)
    test_feature=[test_finger_final,test_graph_final,test_smis_final,test_des_final]
    train_graph_new=graph_deal(train_graph,train_smis)
    test_batch_num=1
    kf=KFold(n_splits=5,shuffle=True,random_state=seed)
    
    sp=kf.split(train_graph_new)
    data=[]
    num=0
    for train_index, valid_index in sp:
        num+=1
        print(len(train_index),len(valid_index))
        train_graph = [train_graph_new[i] for i in train_index]
        valid_graph = [train_graph_new[i] for i in valid_index]
        val_split(valid_graph)
        val_split(train_graph)
        
        train_des=mordred_compute_all([i.smiles for i in train_graph],des_name_final,mean,std)
        valid_des=mordred_compute_all([i.smiles for i in valid_graph],des_name_final,mean,std)

        train_fingerprint=maccs_compute([i.smiles for i in train_graph])
        val_fingerprint=maccs_compute([i.smiles for i in valid_graph])
        train_batch_num=len(train_graph)
        valid_batch_num=1

        train_smis=molformer_cal([i.smiles for i in train_graph])
        valid_smis=molformer_cal([i.smiles for i in valid_graph])
        
        bs=batch_size
        train_graph_final=g_DataLoader(train_graph,batch_size=bs,shuffle=False)
        valid_graph_final=g_DataLoader(valid_graph,batch_size=len(val_fingerprint),shuffle=False)
        
        train_finger_final=d_DataLoader(TensorDataset(train_fingerprint),batch_size=bs,shuffle=False)
        valid_finger_final=d_DataLoader(TensorDataset(val_fingerprint),batch_size=len(val_fingerprint),shuffle=False)

        train_smis_final=d_DataLoader(TensorDataset(train_smis),batch_size=bs,shuffle=False)
        valid_smis_final=d_DataLoader(TensorDataset(valid_smis),batch_size=len(valid_smis),shuffle=False)

        train_des_final=d_DataLoader(TensorDataset(train_des),batch_size=bs,shuffle=False)
        valid_des_final=d_DataLoader(TensorDataset(valid_des),batch_size=len(valid_des),shuffle=False)
        
        train_finger_final=set_ls(train_finger_final)
        valid_finger_final=set_ls(valid_finger_final)

        train_smis_final=set_ls(train_smis_final)
        valid_smis_final=set_ls(valid_smis_final)

        train_des_final=set_ls(train_des_final)
        valid_des_final=set_ls(valid_des_final)
        
        train_feature=[train_finger_final,train_graph_final,train_smis_final,train_des_final]
        valid_feature=[valid_finger_final,valid_graph_final,valid_smis_final,valid_des_final]
        
        
        
        data.append([train_feature,train_batch_num,valid_feature,valid_batch_num,test_feature,test_batch_num])
    return data
    
class LocalModelLoader:
    def __init__(self, model_path):
        """
        从本地路径加载模型
        
        Args:
            model_path: 本地模型目录路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查必需文件是否存在
        required_files = [
            'pytorch_model.bin', 
            'config.json', 
            'tokenizer_config.json',
            'vocab.json'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必需文件缺失: {file}")
        
        # 从本地加载
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path,trust_remote_code=True).to(self.device)
        self.model.eval()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "molformer")
loader = LocalModelLoader(model_dir)
def molformer_cal(smiles):
    inputs = loader.tokenizer(smiles, return_tensors="pt", padding=True).to(loader.device)

    with torch.no_grad():
        outputs = loader.model(**inputs)
    return outputs[1]

def reset(train_graph):
    train_num_samples = len(train_graph)
    indices = np.arange(train_num_samples)
    train_graph=Subset(train_graph, indices)
    return train_graph

def remove_nan(a):
    a_cy=a.copy()
    del_idx=[]
    for  num,i in enumerate(a_cy):
        for k in i.x:
            l=[]
            for v in k:
                
                if math.isnan(v.item()) or pd.isna(v.item()) or np.isnan(v.item()):
                    l.append(v)
                    break
            if len(l)!=0:
                a.remove(i)
                del_idx.append(num)
                break
    print(len(a))
    return a

def graph_compute(root):
    graph=torch.load(os.path.join(root,'processed/data.pt'),weights_only=False)
    graph=remove_nan(graph)
    graph=reset(graph)
    return graph

def feature_compute(root,cross_split,batch_size,seed,des_name=None):
    graph=graph_compute(root)
    cross_data=cross_split(graph,des_name,batch_size,seed)
    return cross_data

def maccs_compute(smiles):
    smis_copy=smiles.copy()
    mols=[]
    for idx,smi in enumerate(smiles):
        
        mol=Chem.MolFromSmiles(smi)
        mols.append(mol)

    mol_hs = [Chem.AddHs(mol) for mol in mols]
    maccs_fingerprint = [list(MACCSkeys.GenMACCSKeys(mol_h)) for mol_h in mol_hs]
    maccs_fingerprint=torch.tensor(maccs_fingerprint,dtype=torch.float).cuda()

    return maccs_fingerprint
def set_ls(des):
    l=[]
    for i,d in enumerate(des):
        l.append(d)
    return l
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def cross_data_compute(root,seed,des_name,batch_size):
    set_seed(seed)
    cross_data=feature_compute(root,cross_split_ablation_allfeature,batch_size,seed,des_name=des_name)
    return cross_data