from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class adap_gconv(nn.Module):

    def __init__(self):
        super(adap_gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum("bhw, bwc->bhc", (A.double(), x.double()))
        return x.contiguous()


class static_gconv(nn.Module):

    def __init__(self):
        super(static_gconv, self).__init__()

    def forward(self, A, x):

        x = torch.einsum("hw, bwc->bhc", (A.double(), x.double()))
        return x.contiguous()


class linear(nn.Module):

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.fc = nn.Conv1d(c_in, c_out, kernel_size=1).double()

    def forward(self, x):
        x = x.double()
        out = F.relu(self.fc(x.permute(0, 2, 1)).permute(0, 2, 1),
                     inplace=True)
        return out


class GCN_Fus(nn.Module):

    def __init__(self,
                 config,
                 c_in,
                 c_out,
                 gdep,
                 dropout_prob,
                 graph_num,
                 type=None):
        super().__init__()
        self.adap_gconv = []
        self.act = config.get("act_gcn", "relu")
        c_in, c_out

        self.hidden_channels = hidden_channels = c_out

        gdep = config["gcn_depth"]

        if graph_num >= config["num_static_graph"] + 1:
            self.adap_gconv1 = adap_gconv()
            self.adap_gconv.append(self.adap_gconv1)
        if graph_num >= config["num_static_graph"] + 2:
            self.adap_gconv2 = adap_gconv()
            self.adap_gconv.append(self.adap_gconv2)
        if graph_num == config["num_static_graph"] + 3:
            self.adap_gconv3 = adap_gconv()
            self.adap_gconv.append(self.adap_gconv3)
        self.num_static_graph = config["num_static_graph"]

        self.static_gconv = nn.ModuleList(
            [static_gconv() for _ in range(self.num_static_graph)])

        self.fc1 = linear((gdep + 1) * c_in, c_out)
        self.weight1 = nn.Parameter(torch.FloatTensor(graph_num + 1),
                                    requires_grad=True)
        self.weight1.data.fill_(1.0 / (graph_num + 1))

        self.fc3 = linear(c_out, c_out)
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.graph_num = graph_num
        self.gdep = gdep
        self.type = type
        self.selu1 = nn.SELU()
        self.layerNorm2 = nn.LayerNorm([self.hidden_channels]).double()
        self.end_attn = True
        self.query_end = nn.Linear(hidden_channels, hidden_channels).double()
        self.key_end = nn.Linear(hidden_channels, hidden_channels).double()
        self.value_end = nn.Linear(hidden_channels, hidden_channels).double()

    def self_attention(self,
                       features,
                       query,
                       key,
                       value,
                       hidden_dim,
                       masks=None,
                       pos_encoder=None):
        Q = query(features)
        K = key(features)
        V = value(features)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim**
                                                                   0.5)
        if masks is not None:
            attention_scores = attention_scores.masked_fill(
                masks == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

    def forward(self, x, static_norm_adjs, adap_norm_adjs=None):

        h1 = x

        out1 = [h1]

        weight1 = F.softmax(self.weight1, dim=0)

        for gdep_no in range(self.gdep):

            h_next1 = weight1[0] * x
            for i in range(0, len(static_norm_adjs)):
                h_next1 += weight1[i + 1] * self.static_gconv[i](
                    static_norm_adjs[i], h1)
                weight1

            if adap_norm_adjs is not None:
                for dd in range(0, len(adap_norm_adjs)):

                    adap_gconv = self.adap_gconv[dd]

                    h_next1 += weight1[len(static_norm_adjs) +
                                       dd] * adap_gconv(
                                           adap_norm_adjs[dd], h1)
            h1 = h_next1

            out1.append(h1)

        ho_1 = torch.cat(out1, dim=-1)
        name1 = self.act
        fn = self._get_activation(name1)

        ho_1 = self.fc1(ho_1)

        ho_1 = fn(ho_1)

        if self.end_attn:
            attention_output = self.self_attention(ho_1, self.query_end,
                                                   self.key_end,
                                                   self.value_end,
                                                   self.hidden_channels, None,
                                                   None)
            attention_output = fn(attention_output)
            attention_output = F.dropout(attention_output,
                                         p=self.dropout_prob,
                                         training=self.training)
            combined_output = ho_1 + attention_output
            combined_output = self.layerNorm2(combined_output)

            ho_1 = fn(combined_output)
        ho_3 = self.fc3(ho_1)

        ho_3 = fn(ho_3)

        return ho_3.double()

    def _get_activation(self, name):
        if name == "tanh":
            f = torch.tanh
        elif name == "relu":
            f = F.relu
        elif name == "selu":
            f = F.selu
        elif name == "elu":
            f = F.elu
        elif name == "silu":
            f = nn.SiLU()
        elif name == "sigmoid":
            f = torch.sigmoid
        elif name == "identity":
            f = lambda x: x
        else:
            raise NotImplementedError
        return f
