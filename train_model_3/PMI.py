from collections import defaultdict
from math import log2
import csv
import numpy as np
import pickle
import dill
import re

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

class PMI():
    # 使用互信息计算两者之间的相似性
    def __init__(self,dataSet):
        self.wordCount = defaultdict(lambda: 0)  # 统计单词频率
        self.pairsCount = defaultdict(lambda: defaultdict(lambda: 0))  # 统计两两单词共同出现的频率
        self.count = 0.0  # 统计单词无序对数目
        self.num = 0.0  # 统计总的单词数
        self.dataSet = dataSet
        self._update()


    def _update(self):
        """遍历整个数据集，更新各个变量"""
        for sample in self.dataSet:
            n = len(sample)
            self.count += n * (n - 1) / 2
            self.num += n
            for word in sample:
                self.wordCount[word] += 1
            for i in range(n):
                for j in range(n):
                    self.pairsCount[sample[i]][sample[j]] += 1

    def query(self, x, y):
        """计算x和y的互信息, I(x,y)=p(x,y)log[p(x,y) / (p(x)p(y))]"""
        p_x = self.wordCount[x] / self.num
        p_y = self.wordCount[y] / self.num
        p_xy = self.pairsCount[x][y] / self.count
        if p_x == 0 or p_y == 0:  # 表示数据集里面没有x或者y，无法得到二者关系
            return 0
        return p_xy * log2(p_xy / p_x / p_y)

def save_PMI():
    with open("data_label.csv", 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    text = []
    for row in data:
        text.append(row[2].split())

    PMI_dict = PMI(text)
    #pickle.dump(PMI_dict, open('PMI_dict.pkl', 'wb'))
    dill.dump(PMI_dict, open('PMI_dict.pkl', 'wb'))

def PMI_adj(text,max_len):
    adj=np.zeros([max_len,max_len])
    for i in range(max_len):
        for j in range(i+1,max_len):
            adj[i][j]=PMI_dict.query(text[i],text[j])
            adj[j][i] = PMI_dict.query(text[i], text[j])
    adj=adj+np.eye(max_len)

    return adj



if __name__=='__main__':
    save_PMI()
    PMI_dict=dill.load(open('PMI_dict.pkl','rb'))
    # print(PMI_dict.query('have', 'to'))
    # print(PMI_dict.query('we','have'))
    text='the elements of the total transfer matrix can then be used to compute the transfer function for the through port'.split()
    adj=PMI_adj(text,len(text))
    print(adj)

