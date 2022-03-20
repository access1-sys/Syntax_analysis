from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset,DataLoader
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
import numpy as np
import pickle
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from train_model_1.save_speech import word2squence
import spacy
import csv
import dill
from math import log2


class Paper50Dataset(Dataset):
    def __init__(self,train):
        super(Paper50Dataset,self).__init__()
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        self.mode_data=[]

        self.ws_dict=pickle.load(open(config.speech_dict_path,'rb'))
        self.glove_dict=pickle.load(open(config.glove_dict_path,'rb'))
        self.tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
        self.nlp=spacy.load('en_core_web_sm')
        self.PMI_dict=dill.load(open('PMI_dict.pkl','rb'))

        with open("data_label.csv", 'r', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        data = np.array(data)

        if train:
            self.mode_data=np.concatenate((data[:int(data.shape[0]*0.0)],data[int(data.shape[0]*0.04):]),axis=0)
        else:
            self.mode_data=data[int(data.shape[0]*0.0):int(data.shape[0]*0.1)]

    def adjacency(self,sentences,max_len):
        doc=self.nlp(sentences)                 #将文本标记化以生成Doc对象
        adj=np.zeros([max_len,max_len])
        for tok in doc:
            #print(tok,'---->',tok.head)
            if not str(tok).isspace():          #检测字符串是否只由空格组成
                if tok.i+1<max_len and tok.head.i+1<max_len:
                    adj[tok.i+1][tok.head.i+1]=1            #邻接矩阵，当前词和他的头节点置1
                    adj[tok.head.i+1][tok.i+1]=1            #构建无向图，如果构建有向图，需注释该行
        adj=adj+np.eye(max_len)

        return adj

    def PMI_adj(self,text,max_len):
        adj = np.zeros([max_len, max_len])
        for i in range(max_len):
            for j in range(i + 1, max_len):
                adj[i][j] = self.PMI_dict.query(text[i], text[j])
                adj[j][i] = self.PMI_dict.query(text[i], text[j])
        adj = adj + np.eye(max_len)

        return adj

    def __getitem__(self, item):
        file_name,mwe,text,bio_label,pos=self.mode_data[item]
        text_glove=self.glove_dict.transform(text.split())
        pos=self.ws_dict.transform(pos.split())
        adj_sd=self.adjacency(str(text),max_len=len(text.split()))
        adj_pmi=self.PMI_adj(text.split(),max_len=len(text.split()))
        bio_label=bio_label.split()


        return text,text_glove,pos,adj_sd,adj_pmi,bio_label

    def __len__(self):
        return len(self.mode_data)

def get_dataloader(train=False):
    dataset=Paper50Dataset(train=train)
    return dataset


if __name__=='__main__':
    dataloader=get_dataloader(train=True)
    print(len(dataloader))
    for idx,(text,text_glove,pos,adj_sd,adj_pmi,bio_label) in enumerate(dataloader):
        print(idx)
        print(text)
        # print(text_glove.shape)
        # print(pos)
        # print(pos.shape)
        # print(adj)
        print(adj_sd.shape)
        print(adj_pmi.shape)
        print(bio_label)
        break












