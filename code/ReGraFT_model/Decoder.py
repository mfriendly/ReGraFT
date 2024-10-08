import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(
        self,
        config,
        in_channels,
        hidden_channels,
        output_channels,
        gcn_depth,
        alpha,
        past_steps,
        dropout_prob,
        dropout_type,
        node_num,
        static_norm_adjs,
        use_curriculum,
        cl_decay_steps,
    ):
        super(Decoder, self).__init__()
        self.act = config["act_decoder"]
        self.dropout = nn.Dropout(p=dropout_prob)
        self.static_norm_adjs = static_norm_adjs
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.RNN_layer = 1
        self.use_curriculum = use_curriculum
        self.cl_decay_steps = cl_decay_steps
        from ReGraFT_model.mgru_cell import MGRUCell as MGRUCell

        self.RNNCell = nn.ModuleList([
            MGRUCell(
                config,
                in_channels,
                hidden_channels,
                dropout_type=dropout_type,
                gcn_depth=gcn_depth,
                alpha=alpha,
                dropout_prob=dropout_prob,
                node_num=node_num,
                static_norm_adjs=static_norm_adjs,
            )
        ])
        self.fc_final = nn.Conv1d(self.hidden_channels,
                                  self.output_channels,
                                  kernel_size=1)

    def forward(
        self,
        decoder_input,
        target_time,
        target_cl,
        Hidden_State,
        adjs,
        task_level=2,
        global_step=None,
    ):
        batch_size, node_num, time_len, dim = decoder_input.shape
        Hidden_State = [
            Hidden_State[:, l, :, :] for l in range(self.RNN_layer)
        ]
        outputs_final = []
        sims = []
        for i in range(task_level):
            decoder_input = cur_time = target_time[:, :, i:i + 1, :]
            for j, rnn_cell in enumerate(self.RNNCell):
                cur_h = Hidden_State[j]
                decoder_input, cur_time, cur_h
                cur_out, cur_h, _, similarity = rnn_cell(
                    decoder_input, cur_time, cur_h, adjs)
                sims.append(similarity)
                Hidden_State[j] = cur_h
                name1 = self.act
                fn = self._get_activation(name1)
                decoder_input = fn(cur_out)
                cur_time = None
            decoder_output = self.fc_final(cur_out.permute(0, 2, 1)).permute(
                0, 2, 1)
            decoder_input = decoder_output.view(batch_size, node_num, 1,
                                                self.output_channels)
            outputs_final.append(decoder_output)
            if self.training and self.use_curriculum:
                c = np.random.uniform(0, 1)
                prob = self._compute_sampling_threshold(global_step)
                if global_step < self.cl_decay_steps:
                    prob = 0.5
                elif global_step == self.cl_decay_steps:
                    pass
                if c < prob:
                    decoder_input = cur_time = target_cl[:, :, i:i + 1, :]
        outputs_final = torch.stack(outputs_final, dim=2)
        outputs_final = outputs_final.view(batch_size, node_num, task_level,
                                           self.output_channels)
        Hidden_State_concatenated = torch.cat(Hidden_State, dim=1)
        return outputs_final, Hidden_State_concatenated, sims

    def _compute_sampling_threshold(self, global_step):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(global_step / self.cl_decay_steps))

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
