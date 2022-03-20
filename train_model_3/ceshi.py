import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from train_model_1.save_speech import word2squence
import csv
import numpy as np
from nltk.tag import StanfordPOSTagger

# mwe_dict=set()
#
# with open('data_label.csv','r',encoding='utf-8') as f:
#     reader=csv.reader(f)
#     data=[row for row in reader]
# data=np.array(data)
# for mwe in data.T[1]:
#     for m in mwe.split(','):
#         mwe_dict.add(m)
#
# print(mwe_dict)
# print(len(mwe_dict))




def output_mwe(text,target,output):
    tar_text=[]
    output_text=[]

    l_mwe=''
    p_mwe=''
    for i,(t,l,p) in enumerate(zip(text,target,output)):
        l_i,p_i=i,i
        if int(l)==0:
            l_mwe+=t+' '
            l_i+=1
            while l_i<len(target)-1 and int(target[l_i])==1:
                l_mwe+=text[l_i]+' '
                l_i+=1
            tar_text.append(l_mwe.strip())
            l_mwe=''

        if int(p)==0:
            p_mwe+=t+' '
            p_i+=1
            while p_i<len(output)-1 and int(output[p_i])==1:
                p_mwe+=text[p_i]+' '
                p_i+=1
            output_text.append(p_mwe.strip())
            p_mwe=''
    #
    # for t,tar,out in zip(text.split(),target,output):
    #     if int(tar)==1 or int(tar)==0:
    #         tar_text.append(t)
    #     if int(out)==1 or int(out)==0:
    #         output_text.append(t)

    return [' '.join(text),','.join(tar_text),','.join(output_text)]


def Evaluation(res):
    label_list = []
    pred_list = []

    for row in res:
        for label in row[1].split(','):
            label_list.append(label)
        for pred in row[2].split(','):
            pred_list.append(pred)

    print(label_list)
    print(pred_list)


# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
#
# text='this will be used to study the effectiveness of possible defense mechanisms against disinformation efforts'
# mwe='this will be used to,the effectiveness of'
# target='B I I I I O B I I O O O O O O'
# target=[tag_to_ix[t] for t in target.split()]
# # print(target)
# # print(mwe)
# res=[]
# res.append(output_mwe(text.split(),target,target))
# print(res)
# Evaluation(res)


adj=torch.rand(34,34)
zero=torch.zeros(adj.shape[0],60-adj.shape[0])
adj=torch.cat([adj,zero],dim=-1)
adj=adj.view(adj.shape[0],1,-1)

print(adj.shape)



# if __name__=='__main__':
#     dataloader = get_dataloader(train=False)
#     for idx,data in enumerate(dataloader):
#         print(idx)
#         print(data[0])
#         break

# text='at the search stage we dramatically extend the set of candidate patches that are compared to the limited set of patches that point to the same index'.split()
#
# eng_tagger = StanfordPOSTagger(
#     model_filename=r'D:\stanford-postagger-2015-04-20\models\english-bidirectional-distsim.tagger',
#     path_to_jar=r'D:\stanford-postagger-2015-04-20\stanford-postagger.jar')
#
# pos_list = eng_tagger.tag(text)
# pos = ' '.join([i[1] for i in pos_list])
#
# print(pos)



