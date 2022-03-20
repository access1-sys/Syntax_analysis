import torch
import torch.autograd as autograd
import torch.nn as nn
import time
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from train_model_CRF import Overall_model_CRF
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from train_model_1.save_speech import word2squence
from dataloader import get_dataloader
from tqdm import tqdm
import numpy as np
import pandas as pd
from math import log2

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

def Accuracy(output,target):
    for i,j in zip(output,target):
        if int(i)!=int(j):
            return False

    return True

def output_mwe(text,target,output):
    be=['be','was','were','are','is','am','being','been']
    text=text.split()
    tar_text=[]
    output_text=[]

    for idx,(t,tar,out) in enumerate(zip(text,target,output)):
        if int(out)==0:
            if text[idx-1] in be:
                output_text.append(text[idx-1])
            elif text[idx-2] in be:
                output_text.append(text[idx-2])
                output_text.append(text[idx-1])
            output_text.append(t)
        elif int(out)==1:
            output_text.append(t)
        if int(tar)==1 or int(tar)==0:
            tar_text.append(t)


    return [' '.join(text),' '.join(tar_text),' '.join(output_text)]


#MWE_model=Overall_model_CRF(tag_to_ix).to(config.device)
MWE_model=torch.load(os.path.join(config.model3_path,'model/Overall_model_CRF_7.pt'))


def test(dataloader):
    MWE_model.eval()
    correct=0
    right_not=[]
    res=[]

    with torch.no_grad():
        for text,text_glove,pos,adj,bio_label in tqdm(dataloader):
            text_glove = torch.tensor(text_glove, dtype=torch.float32).to(config.device)
            pos = torch.tensor(pos).to(config.device)
            adj = torch.tensor(adj, dtype=torch.float32).to(config.device)
            targets = torch.tensor([tag_to_ix[t] for t in bio_label], dtype=torch.long).to(config.device)

            score,output=MWE_model(text_glove,pos,adj)

            res.append(output_mwe(text,targets.numpy(),output))
            # if Accuracy(output, targets.numpy()):
            #     correct += 1
            #     right_not.append('True')
            # else:
            #     right_not.append('False')


    for data in res:
        print(data)
        if data[1]==data[2]:
            correct+=1
            right_not.append('True')
        else:
            right_not.append('False')

    res=pd.DataFrame(res)
    right_not=pd.Series(right_not)
    res=pd.concat((right_not,res),axis=1)
    res.columns=['right_not','text','mwe_text','output_text']
    print('测试集准确率为：{}/{} ({:.2f}%)\n'.format(correct,len(dataloader),
                                                100.*correct/len(dataloader)))

    return correct,res


if __name__=='__main__':
    dataloader=get_dataloader(train=False)
    #MWE_model.load_state_dict(torch.load(os.path.join(config.model3_path,'Overall_model_CRF.pth')))
    correct,res=test(dataloader)

    # print('测试集准确率为：{}/{} ({:.2f}%)\n'.format(correct,len(dataloader),
    #                                             100.*correct/len(dataloader)))
    #res.to_csv(os.path.join(config.model3_path,"output_be.csv"),header=True,index=False)


