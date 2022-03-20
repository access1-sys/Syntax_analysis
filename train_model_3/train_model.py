import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from save_speech import word2squence
from GCN import GCN
from math import log2


class Overall_modedl(nn.Module):
    def __init__(self):
        super(Overall_modedl,self).__init__()
        #定义超参数
        self.hidden_size=128    #LSTM神经元个数
        self.embedding_dim=300  #词向量表示
        self.num_layer=1        #LSTM网络层数
        self.bidriectional=True #LSTM网络是否双向
        self.bi_num=2 if self.bidriectional else 1
        self.dropout=0.5        #随机更新
        self.gcn_hiddend=256    #GCN隐层单元数
        self.gcn_output=64      #GCN输出尺寸

        self.embedding=nn.Embedding(36,self.embedding_dim)
        self.lstm1=nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layer,bidirectional=self.bidriectional,
                          dropout=self.dropout)
        self.lstm2=nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layer,bidirectional=self.bidriectional,
                          dropout=self.dropout)
        # self.lstm3=nn.LSTM(self.embedding_dim*2,self.hidden_size,self.num_layer,bidirectional=self.bidriectional,
        #                    dropout=self.dropout)
        #self.gcn=GCN(self.hidden_size*self.bi_num*2,self.gcn_hiddend,self.gcn_output,self.dropout)
        self.conv1 = nn.Sequential(
            nn.Conv1d(60, 30, 3),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(2),
            nn.Conv1d(30, 60, 3),
            nn.ELU(inplace=True),
            nn.MaxPool1d(2)
        )

    def init_hidden_state(self,batch_size):
        h_0=torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        c_0=torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        return h_0,c_0

    def forward(self,text_glove,pos,adj):
        text_glove=text_glove.view(text_glove.shape[0],1,-1)
        pos=self.embedding(pos).view(len(pos),1,-1)
        #feature=torch.cat([text_glove,pos],dim=-1)      #特征融合

        h_0,c_0=self.init_hidden_state(text_glove.size(1))

        text_glove,(h_n,c_n)=self.lstm1(text_glove,(h_0,c_0))
        pos,(h_n,c_n)=self.lstm2(pos,(h_0,c_0))
        output=torch.cat([text_glove,pos],dim=-1)

        output=output.permute(1,0,2)
        output=self.conv1(output)

        #output,(h_n,c_n)=self.lstm3(output,(h_0,c_0))
        output=output.view(output.size(1),-1)           #LSTM层输出
        #print(output.shape)

        #output=self.gcn(output,adj)                     #GCN层输出

        return output


# train_model=Overall_modedl().to(config.device)
#
#
# dataloader=get_dataloader(train=True)
# for idx,(text,text_glove,pos,adj,bio_label) in enumerate(dataloader):
#     print(text)
#     text_glove=torch.tensor(text_glove,dtype=torch.float32).to(config.device)
#     pos=torch.tensor(pos).to(config.device)
#     adj=torch.tensor(adj,dtype=torch.float32).to(config.device)
#     output=train_model(text_glove,pos,adj)
#     print(output.shape)
#     break
















