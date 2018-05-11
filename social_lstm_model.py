# coding:utf-8


'''
date: 2018/5/10
content:
搭建social lstm模型。
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class social_lstm(nn.Module):
    def __init__(self,
                 batch_size,
                 seq_length,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 input_dim,
                 neighbor_size):
        super(social_lstm, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.neighbor_size = neighbor_size

        # Embedding
        self.coordinat_embedds = nn.Embedding(input_dim, hidden_dim)
        self.social_embedds = nn.Embedding(neighbor_size * neighbor_size * hidden_dim, hidden_dim)

        # LSTMCell,这里注意cell的输入是2倍的embedding_dim，详情见论文
        self.cell = nn.LSTMCell(embedding_dim * 2, hidden_dim)

        self.hidden_2_params = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, pedID, input_seq, pedsInDataset, posInFrame, gridMatrix, hidden_states, cell_states):
        '''

        ------------------------------------------
        :param pedID: 当前人的ID
        :param input_seq: 此人的轨迹帧
        :param pedsInDataset: 数据集中的所有人ID
        :param posInFrame: 每一帧中每个人的位置
        :param gridMatrix: 数据集的掩码矩阵
        :param hidden_states: 所有人的隐藏层状态（这个所有人是所有数据中的最大人数）
        :param cell_states: 同上
        :return:
        '''





        pass

    def _get_social_tensor(self, ):
        pass

    def init_hidden(self):
        ret = Variable(torch.zeros(self.batch_size, self.hidden_dim)), \
              Variable(torch.zeros(self.batch_size, self.hidden_dim))

        return ret.to(device)

