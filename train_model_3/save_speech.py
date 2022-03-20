import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
import numpy as np
import pickle


def tokenize(text):
    '''
    去除句子中的标点符号
    :param text:
    :return:
    '''
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]

    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]

class word2squence:
    UNK_RAG="UNK"       #特殊字符
    PAD_TAG="PAD"       #填充字符

    UNK=1
    PAD=0

    def __init__(self):
        #定义字典
        self.dict={
            self.UNK_RAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}    #获取所有单词词频字典

    def __len__(self):
        return (self.dict)

    def fit(self,sentence):
        '''
        把单个句子保存到dict中
        :param sentence: [word,word,word,,,]
        :return:
        '''
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1

    def build_vocab(self,min_count=0,max_count=None,max_feature=None):
        '''
        生成字典
        :param min_count:最小词频数
        :param max_count:最大词频数
        :param max_feature:一共保留多少词语
        :return:
        '''
        del self.count["''"]
        #删除小于，大于的单词
        if min_count is not None:
            self.count={k:v for k,v in self.count.items() if v>min_count}
        if max_count is not None:
            self.count={k:v for k,v in self.count.items() if v<max_count}

        #限制字典最大单词数
        if isinstance(max_feature,int):
            self.count=sorted(list(self.count.items()),key=lambda x:x[1])
            if max_feature is not None and len(self.count)>max_feature:
                self.count=self.count[-max_feature:]
            for w,_ in self.count:
                self.dict[w]=len(self.dict)
        else:
            for w in sorted(self.count.keys()):
                self.dict[w]=len(self.dict)
        self.fited=True
        #准备一个index——>word的字典
        self.inversed_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def to_index(self,word):
        '''
        通过字典把word转换为数字
        :param word:
        :return:
        '''
        assert self.fited,"必须先进行fit操作"
        return self.dict.get(word,self.UNK)

    def to_word(self,index):
        '''
        通过字典把数字转换为word
        :return:
        '''
        assert self.fited,"必须先进行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_RAG

    def transform(self,sentence,max_len=None):
        '''
        把句子转换为数组
        :param sentence:句子：[word,word,word,,,,,]
        :param max_len: 句子的最大长度
        :return: [1,54,12,421,,,,,]
        '''
        assert self.fited,"必须先进行fit操作"
        if max_len is not None:
            r=[self.PAD]*max_len
        else:
            r=[self.PAD]*len(sentence)
        if max_len is not None and len(sentence)>max_len:
            sentence=sentence[:max_len]
        for index,word in enumerate(sentence):
            r[index]=self.to_index(word)
        return np.array(r,dtype=np.int64)

    def inverse_transform(self,indices):
        '''
        实现从数组转换为句子
        :param indices: [1,2,3,4,,,,]
        :return: [word1,word2,word3,,,]
        '''
        sentence=[]
        for i in indices:
            word=self.to_word(i)
            sentence.append(word)
        return sentence

def save_speech():
    '''
    保存词性分析后的    {字符：唯一编号}字典
    :return:
    '''
    ws=word2squence()
    all_not = {config.AP_all_speech: 1, config.AP_not_speech: 0}
    all_file_list=[]
    for file_list in all_not:
        all_file_list.extend([os.path.join(file_list,j) for j in os.listdir(file_list)])
    for file_path in all_file_list:
        ws.fit(tokenize(open(file_path).read()))

    ws.build_vocab()            #ws.dict:(36)
    print(len(all_file_list))

    pickle.dump(ws,open('speech_dict.pkl','wb'))



if __name__=='__main__':
    #save_speech()      #生成并保存字典
    ws=pickle.load(open('speech_dict.pkl','rb'))
    print(ws.dict)
    print(len(ws.dict))
    print(ws.transform('IN CD NN CC NN VBD CD NNS NNS VBN IN DT NN'.split(),10))







