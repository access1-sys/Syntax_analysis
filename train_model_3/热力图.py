import torch
import torch.autograd as autograd
import torch.nn as nn
import time
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from train_model_CRF_mul import Overall_model_CRF
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from train_model_1.save_speech import word2squence
from dataloader_mul import get_dataloader
from tqdm import tqdm
import numpy as np
import pandas as pd
from math import log2
import matplotlib.pyplot as plt
import seaborn as sns
import dill

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
PMI_dict=dill.load(open('PMI_dict.pkl','rb'))

def PMI_adj(text, max_len):
    adj = np.zeros([max_len, max_len])
    for i in range(max_len):
        for j in range(i + 1, max_len):
            adj[i][j] = PMI_dict.query(text[i], text[j])
            adj[j][i] = PMI_dict.query(text[i], text[j])
    adj = adj + np.eye(max_len)

    return adj


text='in these extensions agents hold beliefs about multiple issues at the same time rather than about a single parameter'.split()
mwe="hold beliefs about,at the same time,rather than"
print(text)
adj_pmi=PMI_adj(text,len(text))
for i in range(len(adj_pmi)):
    adj_pmi[i][i]=0.001
adj_pmi=pd.DataFrame(adj_pmi)
adj_pmi.columns=['in', 'these', 'extensions', 'agents', 'hold', 'beliefs', 'about', 'multiple', 'issues', 'at', 'the', 'same', 'time', 'rather', 'than', 'about', 'a', 'single', 'parameter']
adj_pmi.index=['in', 'these', 'extensions', 'agents', 'hold', 'beliefs', 'about', 'multiple', 'issues', 'at', 'the', 'same', 'time', 'rather', 'than', 'about', 'a', 'single', 'parameter']

sns.set(font_scale=1.5)
sns.set_context({"figure.figsize":(16,16)})
sns.heatmap(data=adj_pmi,square=True,cmap="RdBu_r")

plt.show()




