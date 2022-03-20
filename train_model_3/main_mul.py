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

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

def Evaluation_d(res,keep=4):
    label_list=set()
    pred_list=set()
    N=0

    for row in res:
        for label in row[1].split(','):
            label_list.add(label)
        for pred in row[2].split(','):
            pred_list.add(pred)

    for t in pred_list:
        if t in label_list:
            N+=1

    precision=N/len(pred_list)
    recall=N/len(label_list)
    f1=2*precision*recall/(precision+recall)

    return round(precision,keep),round(recall,keep),round(f1,keep)

def Evaluation(res,keep=4):
    label_list=[]
    pred_list=[]
    N=0

    for row in res:
        for label in row[1].split(','):
            label_list.append(label)
        for pred in row[2].split(','):
            pred_list.append(pred)

    for t in pred_list:
        if t in label_list:
            N+=1

    precision=N/len(pred_list)
    recall=N/len(label_list)
    f1=2*precision*recall/(precision+recall)

    return round(precision,keep),round(recall,keep),round(f1,keep)

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


MWE_model=Overall_model_CRF(tag_to_ix).to(config.device)
optimizer=optim.Adam(MWE_model.parameters(),weight_decay=1e-4)

def train(epoch,dataloader):
    mode=True
    MWE_model.train(mode)

    average_loss=0
    res=[]

    print("正在进行第：",epoch,"轮训练")
    with tqdm(dataloader) as t:
        for text,text_glove,pos,adj_sd,adj_pmi,bio_label in t:
            text_glove=torch.tensor(text_glove,dtype=torch.float32).to(config.device)
            pos=torch.tensor(pos).to(config.device)
            adj_sd = torch.tensor(adj_sd, dtype=torch.float32).to(config.device)
            adj_pmi = torch.tensor(adj_pmi, dtype=torch.float32).to(config.device)
            targets=torch.tensor([tag_to_ix[t] for t in bio_label],dtype=torch.long).to(config.device)


            score,output=MWE_model(text_glove,pos,adj_sd,adj_pmi)

            res.append(output_mwe(text.split(), targets.numpy(), output))

            MWE_model.zero_grad()
            loss=MWE_model.neg_log_likelihood(text_glove,pos,adj_sd,adj_pmi,targets)
            average_loss+=np.float(loss)
            loss.backward()
            optimizer.step()

    precision, recall, f1 = Evaluation(res)

    print('已完成：{}轮训练\t训练集准确率为：{}，召回率为：{}，f1值为：{}\t平均损失为：{:.2f}\n'.format(
        epoch,precision,recall,f1,1.*average_loss/len(dataloader)
    ))

    return precision,recall,f1,1.*average_loss/len(dataloader)



def test(dataloader):
    #MWE_model.eval()
    res=[]
    average_loss = 0

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for text, text_glove, pos, adj_sd,adj_pmi, bio_label in t:
                text_glove = torch.tensor(text_glove, dtype=torch.float32).to(config.device)
                pos = torch.tensor(pos).to(config.device)
                adj_sd = torch.tensor(adj_sd, dtype=torch.float32).to(config.device)
                adj_pmi = torch.tensor(adj_pmi, dtype=torch.float32).to(config.device)
                targets = torch.tensor([tag_to_ix[t] for t in bio_label], dtype=torch.long).to(config.device)

                score,output=MWE_model(text_glove,pos,adj_sd,adj_pmi)

                loss = MWE_model.neg_log_likelihood(text_glove,pos,adj_sd,adj_pmi,targets)
                average_loss += np.float(loss)

                res.append(output_mwe(text.split(),targets.numpy(),output))

    precision, recall, f1 = Evaluation(res)
    res=pd.DataFrame(res)
    res.columns=['text','mwe_text','output_text']

    print('测试集准确率为：{}，召回率为：{}，f1值为：{}\t平均损失为：{:.2f}\n'.format(
        precision,recall,f1,1.*average_loss/len(dataloader)))

    # print('测试集准确率为：{}/{} ({:.2f}%)\n'.format(correct,len(dataloader),
    #                                             100.*correct/len(dataloader)))

    return precision,recall,f1,1.*average_loss/len(dataloader),res


def create_train_logs(file_path):
    f=open(file_path,'a',encoding='utf-8')
    f.write('epoch'+'\t'+'train_precison'+'\t'+'train_recall'+'\t'+'train_f1'+'\t'+'train_loss'+'\t'
            +'test_precision'+'\t'+'test_recall'+'\t'+'test_f1'+'\t'+'test_loss'+'\n')
    f.close()



if __name__=='__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader=get_dataloader(train=False)


    model_name='5_LSTM_SD_PMI_GCN_10'       #保存的模型名字

    MWE_model.load_state_dict(torch.load(os.path.join(config.model3_path, 'model/' + model_name + '.pth')))

    train_max,test_max=0,0
    train_logs_path=os.path.join(config.model3_path,'train_logs/'+model_name+'.txt')
    #create_train_logs(train_logs_path)

    for i in range(25,50):
        train_precision,train_recall,train_f1,train_loss=train(i,train_dataloader)

        test_precision,test_recall,test_f1,test_loss,res=test(test_dataloader)

        f = open(train_logs_path, 'a', encoding='utf-8')
        f.write(str(i)+'\t'+str(train_precision)+'\t'+str(train_recall)+'\t'+str(train_f1)+'\t'+str(train_loss)+'\t'
                +str(test_precision)+'\t'+str(test_recall)+'\t'+str(test_f1)+'\t'+str(test_loss)+'\n')
        f.close()

        torch.save(MWE_model.state_dict(), os.path.join(config.model3_path, 'model/'+model_name+'.pth'))
        if train_max<=train_precision:
            train_max=train_precision

        if test_max<=test_f1:
            test_max=test_f1
            torch.save(MWE_model.state_dict(), os.path.join(config.model3_path, 'model/' + model_name + '_max.pth'))
            res.to_csv(os.path.join(config.model3_path, 'output/'+model_name+'.csv'), header=True, index=False)

        print('训练集最高准确率为：{}，测试集最高F1为：{}\n'.format(train_max,test_max))


