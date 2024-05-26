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


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum("hw, bwtc->bhtc", (A.double(), x.double()))
        return x.contiguous()


class linear(nn.Module):

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.fc = nn.Conv1d(c_in, c_out, kernel_size=1).double()

    def forward(self, x):
        x = x.double()
        out = F.relu(self.fc(x.permute(0, 2, 1)).permute(0, 2, 1), inplace=True)
        return out


class GCN_Fus(nn.Module):

    def __init__(self, c_in, c_out, gdep, dropout_prob, graph_num, type=None):
        super().__init__()
        graph_num
        gdep = 1

        self.adap_gconv1 = adap_gconv()
        self.adap_gconv2 = adap_gconv()
        self.adap_gconv3 = adap_gconv()
        self.adap_gconv = [self.adap_gconv1, self.adap_gconv2, self.adap_gconv3]
        self.static_gconv1 = static_gconv()
        self.static_gconv2 = static_gconv()
        self.static_gconv =[self.static_gconv1, self.static_gconv2]

        self.fc1 = linear((gdep + 1) * c_in, c_out)
        self.weight1 = nn.Parameter(torch.FloatTensor(graph_num + 1),
                                    requires_grad=True)
        self.weight1.data.fill_(1.0 / (graph_num + 1))

        self.fc3 = linear(c_out, c_out)

        self.dropout = nn.Dropout(dropout_prob)
        self.graph_num = graph_num
        self.gdep = gdep
        self.type = type

    def forward(self, x, norm_adj, adap_norm_adj=None):
        # print("adap_norm_adj", len(adap_norm_adj))
        h1 = x

        out1 = [h1]

        weight1 = F.softmax(self.weight1, dim=0)

        for _ in range(self.gdep):
            h_next1 = weight1[0] * x
            for i in range(0, len(norm_adj)):
                h_next1 += weight1[i + 1] * self.static_gconv[i](norm_adj[i], h1)
            if adap_norm_adj is not None:
                for dd in range(0, len(adap_norm_adj)):
                    # print("dd", dd)
                    adap_gconv = self.adap_gconv[dd]
                    # print("adap_gconv", adap_gconv)
                    h_next1 += weight1[ len(norm_adj)+ dd] * adap_gconv(
                        adap_norm_adj[dd], h1)
            h1 = h_next1
            out1.append(h1)

        ho_1 = torch.cat(out1, dim=-1)
        name1 = "relu"
        fn = self._get_activation(name1)

        ho_1 = self.fc1(ho_1)

        ho_1 = fn(ho_1)

        ho_3 = self.fc3(ho_1)
        name1 = "relu"
        fn = self._get_activation(name1)
        ho_3 = fn(ho_3)
        return ho_3.float()

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
