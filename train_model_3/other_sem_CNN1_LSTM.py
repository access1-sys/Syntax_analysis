import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataloader_other import get_dataloader
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

        #self.embedding=nn.Embedding(36,self.embedding_dim)
        self.lstm3=nn.LSTM(73,self.hidden_size,self.num_layer,bidirectional=self.bidriectional,
                           dropout=self.dropout)

        self.conv1=nn.Sequential(
            nn.Conv1d(60,30,3),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(2),
            nn.Conv1d(30,60,3),
            nn.ELU(inplace=True),
            nn.MaxPool1d(2)
        )


    def init_hidden_state(self,batch_size):
        h_0=torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        c_0=torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        return h_0,c_0

    def forward(self,text_glove,pos,adj):
        text_glove=text_glove.view(1,text_glove.shape[0],-1)
        output=self.conv1(text_glove)

        #print(output.shape)
        output=output.permute(1,0,2)

        h_0,c_0=self.init_hidden_state(output.size(1))

        output,(h_n,c_n)=self.lstm3(output,(h_0,c_0))
        output=output.view(output.size(0),-1)           #LSTM层输出

        #output=self.gcn(output,adj)                     #GCN层输出

        return output


START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    # 将 argmax 作为 python int 返回
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
# 以数值稳定的方式为前向算法计算对数总和 exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#创建模型
class Overall_model_CRF(nn.Module):
    def __init__(self, tag_to_ix):
        super(Overall_model_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix                  #标签转索引的字典
        self.tagset_size = len(tag_to_ix)           #标签转索引的字典类别数


        # Maps the output of the LSTM into tag space.
        # 将 LSTM 的输出映射到标签空间。
        self.overall_model=Overall_modedl()
        self.hidden2tag = nn.Linear(256, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 过渡参数矩阵。 条目 i,j 是转换 *to* i *from* j 的分数。
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))        #nn.Parameter会自动被认为是module的可训练参数

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 这两个语句强制约束我们永远不会转移到开始标签，我们永远不会从停止标签转移
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 做前向算法来计算分区函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)        #生成尺寸[1,标签种类]的矩阵，值全为-10000
        # START_TAG has all of the score.
        # START_TAG 拥有所有分数。
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 包裹在一个变量中，以便我们获得自动反向传播
        forward_var = init_alphas

        # Iterate through the sentence
        # 遍历句子
        for feat in feats:
            alphas_t = []  # 这个时间步的前向张量
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # 广播发射分数：与之前的标签无关
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # trans_score 的第 i 个条目是从 i 过渡到 next_tag 的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # next_tag_var 的第 i 个条目是我们执行 log-sum-exp 之前的边 (i -> next_tag) 的值
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                # 此标签的前向变量是所有分数的 log-sum-exp。
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self,text_glove,pos,adj):
        '''
        # 包含 词嵌入层+LSTM+线形层，
        :param sentence: 句子
        :return: 特征，[len(sentence),self.tagset_size]
        '''
        output=self.overall_model(text_glove,pos,adj)
        lstm_feats = self.hidden2tag(output)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 给出提供的标签序列的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化日志空间中的 viterbi 变量
        init_vvars = torch.full((1, self.tagset_size), -100000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # 第 i 步的 forward_var 保存第 i-1 步的维特比变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 持有此步骤的反向指针
            viterbivars_t = []  # 保存这一步的维特比变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] 保存上一步标签 i 的 viterbi 变量，加上从标签 i 转换到 next_tag 的分数。
                # 我们在这里不包括排放分数，因为最大值不依赖于它们（我们在下面添加它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # 现在添加排放分数，并将 forward_var 分配给我们刚刚计算的一组 viterbi 变量
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转换到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        # 按照后向指针解码最佳路径。
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        # 弹出开始标签（我们不想将其返回给调用者）
        # if best_path[-1]!=3:
        #     best_path[-1]=3
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 完整性检查
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self,text_glove,pos,adj,tags):
        feats = self._get_lstm_features(text_glove,pos,adj)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self,text_glove,pos,adj):  # 不要将此与上述所有内容混淆。
        # 从 BiLSTM 获取排放分数
        lstm_feats = self._get_lstm_features(text_glove,pos,adj)

        # 给定特征，找到最佳路径。
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq