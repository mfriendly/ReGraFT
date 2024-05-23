from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class dyn_gconv(nn.Module):

    def __init__(self):
        super(dyn_gconv, self).__init__()

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
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1).double()

    def forward(self, x):
        x = x.double()
        out = F.elu(self.mlp(x.permute(0, 2, 1)).permute(0, 2, 1),
                    inplace=True)
        return out


class GCN_Fus(nn.Module):

    def __init__(self, c_in, c_out, gdep, dropout_prob, graph_num, type=None):
        super().__init__()
        if True:
            self.dyn_gconv = dyn_gconv()
            self.static_gconv = static_gconv()
            self.mlp1 = linear((gdep + 1) * c_in, c_out)
            self.weight1 = nn.Parameter(torch.FloatTensor(graph_num + 1 + 1),
                                        requires_grad=True)
            self.weight1.data.fill_(1.0 / (graph_num + 1 + 1))
            self.gconv = gconv()
            self.mlp2 = linear((gdep + 1) * c_in, c_out)
            self.mlp3 = linear(c_out * 2, c_out)
            self.weight2 = nn.Parameter(torch.FloatTensor(graph_num + 1),
                                        requires_grad=True)
            self.weight2.data.fill_(1 / (graph_num + 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.graph_num = graph_num
        self.gdep = gdep
        self.type = type

    def forward(self, x, norm_adj, dyn_norm_adj=None):
        h1 = x
        h2 = x
        out1 = [h1]
        out2 = [h2]
        weight1 = F.softmax(self.weight1, dim=0)
        weight2 = F.softmax(self.weight2, dim=0)
        for _ in range(self.gdep):
            h_next1 = weight1[0] * x
            for i in range(0, len(norm_adj)):
                h_next1 += weight1[i + 1] * self.static_gconv(norm_adj[i], h1)
            if dyn_norm_adj is not None:
                for dd in range(0, len(dyn_norm_adj)):
                    h_next1 += weight1[dd + 1] * self.dyn_gconv(
                        dyn_norm_adj[dd], h1)
            h1 = h_next1
            out1.append(h1)
            h2 = weight2[0] * x
            for jj in range(1, len(norm_adj)):
                h2 += weight2[jj] * self.static_gconv(norm_adj[jj], h2)
            out2.append(h2)
        ho_1 = torch.cat(out1, dim=-1)
        name1 = "relu"
        fn = self._get_activation(name1)
        ho_2 = torch.cat(out2, dim=-1)
        ho_1 = self.mlp1(ho_1)
        ho_2 = self.mlp2(ho_2)
        ho_1 = fn(ho_1)
        ho_2 = fn(ho_2)
        ho3 = torch.cat([ho_1, ho_2], dim=-1)
        ho_3 = self.mlp3(ho3)
        name1 = "relu"  # keep relu
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
