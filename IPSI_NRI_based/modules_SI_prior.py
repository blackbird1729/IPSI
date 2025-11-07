import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLPEncoderpre(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoderpre, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.transpose(1,0), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # x = self.mlp1(x)  # 2-layer ELU net per node
        x=inputs
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)

class RNNEncoder(nn.Module):
    """RNN 编码器（支持 LSTM、GRU），带有批归一化、Dropout以及参数初始化"""

    def __init__(self, input_dim, hidden_dim, rnn_type="LSTM", bidirectional=False, dropout=0.5):
        """
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏状态维度
        :param rnn_type: RNN 类型，支持 "LSTM" 或 "GRU"
        :param bidirectional: 是否双向 RNN
        :param dropout: dropout 概率
        """
        super(RNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)  # 计算最终维度

        # 选择 RNN 类型，若层数>1时，可直接在 rnn 内部使用 dropout 参数
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,dropout=dropout)

        # 修正 BatchNorm 维度：针对 [batch * node, feature]
        self.batchnorm = nn.BatchNorm1d(self.output_dim)

        # 添加 dropout 层（应用在隐藏状态上）
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化 RNN 权重"""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 [batchsize, node_num, time_steps, features]
        :return: 输出张量，形状为 [batchsize, node_num, output_dim]
        """
        batchsize, node_num, time_steps, _ = x.shape
        x = x.view(batchsize * node_num, time_steps, -1)  # 合并 batch 和 node 维度

        # RNN 计算，h_n 表示最终的隐藏状态
        _, h_n = self.rnn(x)

        # 如果是 LSTM，则 h_n 是一个元组 (h_n, c_n)
        if isinstance(h_n, tuple):
            h_n = h_n[0]

        # 处理双向情况
        if self.bidirectional:
            # 对于双向，取最后两层的隐藏状态进行拼接
            h_n = h_n[-2:].transpose(0, 1).reshape(batchsize * node_num, -1)
        else:
            h_n = h_n[-1]  # 只取最后一层的隐藏状态

        # BatchNorm 对隐藏状态进行归一化
        h_n = self.batchnorm(h_n)
        # 应用 Dropout，防止过拟合
        h_n = self.dropout(h_n)
        # 恢复形状 (batchsize, node_num, hidden_dim 或 hidden_dim*2)
        h_n = h_n.view(batchsize, node_num, -1)

        return h_n

class MLP_emb(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP_emb, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)





