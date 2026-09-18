import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GATConv,SAGEConv,GINConv
import random


class SsTGN_allfeature_newloss(nn.Module):
    def __init__(self, finger_dim,graph_dim,smi_dim,des_dim,des_graph_dim,output_dim,value,value_o,value2,value_o2):
        super(SsTGN_allfeature_newloss, self).__init__()
        self.fc_finger=nn.Sequential(nn.Linear(finger_dim,128),nn.BatchNorm1d(128),nn.Dropout(0.5),nn.ReLU())
        self.finger_fc1=nn.Sequential(nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU())
        
        self.fc_des=nn.Sequential(nn.Linear(des_dim,256),nn.BatchNorm1d(256),nn.Dropout(0.8),nn.ReLU())
        self.des_fc1=nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU())
        
        self.fc_smi=nn.Sequential(nn.Linear(smi_dim,512),nn.BatchNorm1d(512),nn.Dropout(0.95),nn.ReLU())
        self.smi_fc1=nn.Sequential(nn.Linear(512,128),nn.BatchNorm1d(128),nn.Dropout(0.95),nn.ReLU())
        
        self.gnn_graph=GINConv(nn.Sequential(nn.Linear(graph_dim,64),nn.ReLU()))
        self.graph_gin_bn1=nn.BatchNorm1d(64)
        self.gnn_graph2=GINConv(nn.Sequential(nn.Linear(64,32),nn.ReLU()))
        self.graph_gin_bn2=nn.BatchNorm1d(32)

        self.all_gat1=GATConv(64+128+32+128 ,64,heads=4,concat=True)
        self.all_gat1_bn=nn.BatchNorm1d(256)
        self.all_gat2=GATConv(256 ,32,heads=4,concat=True)
        self.all_gat2_bn=nn.BatchNorm1d(128)
        self.all_drop=nn.Dropout(0.6)

        self.all_fc1=nn.Linear(128,output_dim)

        self.edge_param=nn.Parameter(torch.randn(128,128))
        nn.init.xavier_uniform_(self.edge_param)
        


        
        self.input_dim=des_dim
        self.cat_ratio1=nn.Parameter(torch.tensor(0.2))
        self.cat_ratio2=nn.Parameter(torch.tensor(0.5))
        self.value=value
        self.value_o=value_o
        self.value2=value2
        self.value_o2=value_o2

    def mask_exact_pairs(self, edge_index, edge_weights, n_pairs):
        
        if edge_index.shape[1] == 0:
            return edge_index, edge_weights

        num_edges = edge_index.shape[1]   

        num_drop = n_pairs
        if num_drop >= num_edges:
            num_drop = num_edges - 1 
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)

        if num_drop > 0:
            drop_indices = random.sample(range(num_edges), num_drop)
            keep_mask[drop_indices] = False
        
        new_edge_index = edge_index[:, keep_mask]
        new_edge_weights = edge_weights[keep_mask]

        return new_edge_index, new_edge_weights


    
    def edge_compute1(self,x,value,value_o):
        cos=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1).cuda()
        cos = torch.tril(cos, diagonal=-1).cuda()
        mask = cos >= value
        indices = torch.nonzero(mask).t()
        weights = cos[indices[0], indices[1]] 
    
        if self.training:
            edge_final, weight_final = self.mask_exact_pairs(indices, weights, value_o)
        else:
            edge_final, weight_final = indices, weights
        return edge_final.cuda(), weight_final.cuda()
    
    def edge_compute2(self,x,value2,value_o2):
        cos=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1).cuda()
        cos = torch.tril(cos, diagonal=-1).cuda()
        mask = cos >= value2
        indices = torch.nonzero(mask).t()
        weights = cos[indices[0], indices[1]] 
    
        if self.training:
            edge_final, weight_final = self.mask_exact_pairs(indices, weights, value_o2)
        else:
            edge_final, weight_final = indices, weights

        return edge_final, weight_final.cuda()
    
    def forward(self, x_finger,x_graph,edge_idx,batch,x_smis,x_des):
        x_finger=self.fc_finger(x_finger)
        x_finger1 = self.finger_fc1(x_finger)

        x_des=self.fc_des(x_des)
        x_des1 = self.des_fc1(x_des)
        
        
        x_smis=self.fc_smi(x_smis)
        x_smis1 = self.smi_fc1(x_smis)
        
        x_graph1 = self.gnn_graph(x_graph,edge_idx)
        x_graph1 = self.graph_gin_bn1(x_graph1)
        x_graph2 = self.gnn_graph2(x_graph1,edge_idx)
        x_graph2 = self.graph_gin_bn2(x_graph2)
        x_graph3=torch_geometric.nn.global_mean_pool(x_graph2,batch)

        x_all1=torch.cat((x_finger1,x_smis1,x_graph3,x_des1),dim=1)
        edge1,edge1_weight=self.edge_compute1(x_all1,self.value,self.value_o)
        x_all2=self.all_gat1(x_all1,edge1)
        
        x_all2=self.all_gat1_bn(x_all2)
        
        x_all2=F.relu(x_all2)
        edge2,edge2_weight=self.edge_compute2(x_all2,self.value2,self.value_o2)
        x_all3=self.all_gat2(x_all2,edge2)
        x_all3=self.all_gat2_bn(x_all3)
        x_all3=F.relu(x_all3)
        edge3,edge3_weight=self.edge_compute2(torch.matmul(x_all3,self.edge_param),0.95,0)
        x_all4=self.all_fc1(x_all3)

        return x_all4,edge3,x_all3,edge3_weight
    
