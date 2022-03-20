import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
from train_model_1.Glove_dict import Glove        #不可以删除，下面程序需要读取glove_dict，调用Glove类里的transform函数
from train_model_1.save_speech import word2squence


torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def argmax(vec):
    # return the argmax as a python int
    # 将 argmax 作为 python int 返回
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    #准备序列:  句子转索引列表
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
# 以数值稳定的方式为前向算法计算对数总和 exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


#创建模型
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim          #词嵌入embedding数
        self.hidden_dim = hidden_dim                #LSTM的隐层节点数
        self.vocab_size = vocab_size                #单词转索引的字典单词数
        self.tag_to_ix = tag_to_ix                  #标签转索引的字典
        self.tagset_size = len(tag_to_ix)           #标签转索引的字典类别数

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # 将 LSTM 的输出映射到标签空间。
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

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

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 创建初始h_0，c_0
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

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

    def _get_lstm_features(self, sentence):
        '''
        # 包含 词嵌入层+LSTM+线形层，
        :param sentence: 句子
        :return: 特征，[len(sentence),self.tagset_size]
        '''
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
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
        init_vvars = torch.full((1, self.tagset_size), -10000.)
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
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 完整性检查
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # 不要将此与上述所有内容混淆。
        # 从 BiLSTM 获取排放分数
        lstm_feats = self._get_lstm_features(sentence)

        # 给定特征，找到最佳路径。
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# 补一些训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(36, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 在训练前检查预测
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

dataloader=get_dataloader(train=False)
i=0
# 确保加载了 LSTM 部分前面的 prepare_sequence
for epoch in range(300):  # 再说一次，通常你不会做 300 个时代，这是玩具数据
    for text,sentence, tags in dataloader:
        sentence=sentence.reshape(-1)
        tags=tags[0].split()
        # Step 1. 记住 Pytorch 累积梯度。我们需要在每个实例之前清除它们
        model.zero_grad()
        # 步骤 2. 为网络准备好我们的输入，即将它们转换为单词索引的张量。
        sentence_in = torch.tensor(sentence,dtype=torch.long)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # 步骤 3. 运行我们的前传。
        loss = model.neg_log_likelihood(sentence_in, targets)

        # 步骤 4. 通过调用 optimizer.step() 计算损失、梯度和更新参数
        loss.backward()
        optimizer.step()
        i+=1
        if i==5:
            break
    #if epoch%30==0:
    print("已训练",epoch,"轮")

# 训练后检查预测
with torch.no_grad():
    # precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    # print(model(precheck_sent))
    for idx,(text,pos,bio_label) in enumerate(dataloader):
        print(text)
        pos=torch.tensor(pos.reshape(-1),dtype=torch.long)
        bio_label = bio_label[0].split()
        targets = torch.tensor([tag_to_ix[t] for t in bio_label], dtype=torch.long)
        print(bio_label)
        print(model(pos))
        if idx==4:
            break
        #break

# We got it!















