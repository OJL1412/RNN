import torch
import torch.nn as nn
import torch.nn.functional as F


class M_RNN(nn.Module):
    """
    ① vocab_size: 处理好的词的数量
    ② emb_dim: 给定的每个词的词向量纬度
    ③ hidden_state: 上一个词的隐状态
    """
    def __init__(self, vocab_size, emb_dim=32, hidden_state=None, drop_out=0.1, bind_emb=True):
        super(M_RNN, self).__init__()

        self.v_size = vocab_size

        hidden_state = emb_dim if hidden_state is None else hidden_state  # 给定一个默认的输出值

        # Embedding进行词嵌入，随机初始化映射为一个向量矩阵，参数1是嵌入字典的词的数量，参数2是每个嵌入向量的大小，此处为词向量维度32
        self.w_emb = nn.Embedding(self.v_size, emb_dim)

        # 激活函数前加nn.LayerNorm: 先算o_size个数的均值和方差，再计算（o_size-均值）/方差，用以排除极大和极小的数，稳定数据
        # 激活函数后接nn.Dropout: 用来随机丢掉部分数据特征（弄成0），使得剩下的数据特征包含丢掉的数据的特征，使数据的特征间产生关联
        self.net = nn.Sequential(
            nn.Linear(emb_dim + hidden_state, hidden_state, bias=False),
            nn.LayerNorm(hidden_state),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_state, emb_dim, bias=False),
            nn.Linear(emb_dim, self.v_size)  # 32 --> vocab_size
        )

        self.hidden_state_init = nn.Parameter(torch.zeros(1, hidden_state))  # 隐状态初始化

        if bind_emb:
            self.classifier[-1].weight = self.w_emb.weight  # 绑定词向量与分类器的权重

    """
    ① 传入input_words: 输入的语料，表示有多少行，每行有多少个词（batch_size, seql）
    ② hidden_state: 上一个词的隐状态，也是需要进行加和的参数
    """
    def forward(self, input_words):
        hidden_state = self.hidden_state_init.expand(input_words.size(0), -1)
        input = self.w_emb(input_words)

        _state = []
        for i in input.unbind(1):
            state = self.net(torch.cat((i, hidden_state), dim=-1))  # 将现在的词向量与上一个词的隐状态进行拼接，放入net中进行处理输出
            _state.append(state)    # 将处理后的结果放入设置的列表进行存储

        output = self.classifier(torch.stack(_state, dim=1))    # s使用stack进行堆叠处理，结果过classifier

        return output

    def decode(self, input_words):
        hidden_state = self.hidden_state_init.expand((input_words.size(0), -1))
        input = self.w_emb(input_words)

        state = self.net(torch.cat((input, hidden_state), dim=1))

        output = torch.argmax(self.classifier(state), dim=1)

        return output
