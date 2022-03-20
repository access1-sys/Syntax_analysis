import csv
import numpy as np
import re
import pandas as pd
from nltk.tag import StanfordPOSTagger

eng_tagger = StanfordPOSTagger(
    model_filename=r'D:\stanford-postagger-2015-04-20\models\english-bidirectional-distsim.tagger',
    path_to_jar=r'D:\stanford-postagger-2015-04-20\stanford-postagger.jar')

def tokenize(text):
    '''
    去除句子中的标点符号
    :param text:
    :return:
    '''
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@','\''
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]

    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip().lower() for i in text.split()]


def biolabel(file_nem, text, mwe):
    '''
    生成BIO标签的函数，长度为max_len
    :param file_nem: 文件名
    :param text: 输入的完整文本
    :param mwe: 输入的多次表达式
    :return: 输出  文件名，多词术语，完整文本，BIO标签，pos
    '''
    text = tokenize(text)
    max_len=len(text)
    label = ['O'] * max_len
    mwe = tokenize(mwe)
    pos_list = eng_tagger.tag(text)
    pos = ' '.join([i[1] for i in pos_list])
    b = 0
    m_i = 0
    t_i = 0
    while t_i < len(text):
        if text[t_i] == mwe[m_i] and b == 0:
            label[t_i] = 'B'
            m_i += 1
            b += 1
            t_i += 1
            while m_i < len(mwe) and text[t_i] == mwe[m_i]:
                label[t_i] = 'I'
                m_i += 1
                t_i += 1
        if m_i == len(mwe):
            break
        else:
            b = 0
            m_i = 0
            label = ['O'] * max_len
        t_i += 1

    return [str(file_nem), ' '.join(mwe), ' '.join(text), ' '.join(label), pos]

def biolabel_O(file_nem,text,mwe):
    '''
    生成BIO标签的函数，长度为max_len
    :param file_nem: 文件名
    :param text: 输入的完整文本
    :param mwe: 输入的多次表达式
    :return: 输出  文件名，多词术语，完整文本，BIO标签，pos
    '''
    text = tokenize(text)
    max_len=len(text)
    label = ['O'] * max_len
    mwe=tokenize(mwe)
    pos_list=eng_tagger.tag(text)
    pos=' '.join([i[1] for i in pos_list])
    b=0
    m_i=0
    t_i=0
    while t_i<len(text):
        if text[t_i]==mwe[m_i]:
            if b==0:
                label[t_i] ='B'
                b+=1
            else:
                label[t_i]='I'
            m_i+=1
        if m_i==len(mwe):
            break
        t_i+=1

    return [str(file_nem),' '.join(mwe),' '.join(text),' '.join(label),pos]

#读取文件中所有数据
with open("paper50_all2.csv",'r',encoding='ISO-8859-1') as f:
    reader=csv.reader(f)
    data=[row for row in reader]
data=np.array(data)

res=[]
all_O=[]

for i,row in enumerate(data[:3500]):
    if row[2]=='null':
        continue
    test=biolabel(row[0],row[1],row[2])
    if test[3].split()==['O'] * len(test[3].split()):
        test=biolabel_O(row[0],row[1],row[2])
        all_O.append(test[0])
    print(test)
    res.append(test)

print(all_O)

res=pd.DataFrame(res)
res.to_csv('data_label.csv',header=False,index=False)






