import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ReGraFT_model.mgru_cell import MGRUCell as MGRUCell


class Encoder(nn.Module):

    def __init__(
        self,
        config,
        in_channels,
        hidden_channels,
        gcn_depth,
        alpha,
        past_steps,
        dropout_prob,
        dropout_type,
        node_num,
        static_norm_adjs,
        device,
    ):
        super(Encoder, self).__init__()
        self.config = config
        self.act = config["act_encoder"]
        self.dropout = nn.Dropout(p=dropout_prob)
        self.seq_length = past_steps
        self.static_norm_adjs = static_norm_adjs
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.RNN_layer = 1
        self.device = device

        self.RNNCell = nn.ModuleList([
            MGRUCell(
                config,
                hidden_channels,
                hidden_channels,
                dropout_type=dropout_type,
                gcn_depth=gcn_depth,
                alpha=alpha,
                dropout_prob=dropout_prob,
                node_num=config["num_nodes"],
                static_norm_adjs=static_norm_adjs,
            )
        ])

    def forward(self, x, seq_length, adjs):

        batch_size, node_num, time_len, dim = x.shape
        Hidden_State = [
            self.initHidden(batch_size, node_num, self.hidden_channels)
            for _ in range(self.RNN_layer)
        ]
        outputs = []
        hiddens = []
        adjs_output = []
        for i in range(seq_length):
            input_cur = x[:, :, i * 1:i * 1 + 1, :]
            adap_norm_adjs = []
            for j, rnn_cell in enumerate(self.RNNCell):
                input_time = x
                cur_h = Hidden_State[j]
                cur_out, cur_h, adap_norm_adj, similarity = rnn_cell(
                    input_cur, input_time, cur_h, adjs)
                Hidden_State[j] = cur_h
                name1 = self.act
                fn = self._get_activation(name1)
                input_cur = fn(cur_out)
                input_time = None
                adap_norm_adjs.extend(adap_norm_adj)
            outputs.append(cur_out.unsqueeze(dim=2))

            try:
                stacked_adap_norm_adjs = torch.stack(adap_norm_adjs)
                average_adap_norm_adj = torch.mean(stacked_adap_norm_adjs,
                                                    dim=0)
                adjs_output.append(average_adap_norm_adj)

            except Exception as e:
                pass

            hidden = torch.stack(Hidden_State, dim=1).unsqueeze(dim=3)
            hiddens.append(hidden)
        outputs = torch.cat(outputs, dim=2)
        hiddens = torch.cat(hiddens, dim=3)
        if self.config["skip_connection"]:
            outputs_2 = outputs + x
        else:
            outputs_2 = outputs
        return outputs_2, hiddens, adjs_output

    def initHidden(self, batch_size, num_nodes, hidden_dim):
        pass

        Hidden_State = Variable(
            torch.zeros((batch_size, num_nodes, hidden_dim)))
        return Hidden_State

    def init_weights(self):
        INITRANGE = 0.04
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

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
