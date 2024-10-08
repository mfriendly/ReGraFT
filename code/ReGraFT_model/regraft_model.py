import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import ReGraFT_model.mha as mha
from ReGraFT_model.Decoder import Decoder as GraphDecoder
from ReGraFT_model.Encoder import Encoder as GraphEncoder


class TimeDistributed(nn.Module):

    def __init__(self, module, args=None, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x0):
        if isinstance(self.module, nn.Embedding):
            x0 = x0.to(torch.int64).to(self.device)
        if len(x0.size()) <= 2:
            return self.module(x0)
        x_reshape = x0.contiguous().view(-1, x0.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x0.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x0.size(1), y.size(-1))
        return y


class GLU(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class SelfNormalizingPrimingNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_state_size,
                 output_size,
                 dropout_rate,
                 hidden_context_size=None,
                 batch_first=True):
        super(SelfNormalizingPrimingNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(
                self.input_size, self.output_size),
                                              batch_first=batch_first)
        self.fc1 = TimeDistributed(nn.Linear(self.input_size,
                                             self.hidden_state_size),
                                   batch_first=batch_first)
        self.selu = nn.SELU()
        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size,
                                                     self.hidden_state_size),
                                           batch_first=batch_first)
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,
                                             self.output_size),
                                   batch_first=batch_first)
        self.dropout = nn.Dropout(dropout_rate)
        self.glu = TimeDistributed(GLU(self.output_size),
                                   batch_first=batch_first)

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.selu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        x = x + residual
        return x


class PositionalEncoder(torch.nn.Module):

    def __init__(self, d_model, max_seq_len=800):
        super().__init__()
        self.d_model_o = d_model
        if d_model % 2 == 1:
            d_model += 1
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for di in range(0, d_model, 2):
                pe[pos, di] = math.sin(pos / (10000**((2 * di) / d_model)))
                pe[pos, di + 1] = math.cos(pos /
                                           (10000**((2 * (di + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe[:, :, :self.d_model_o]
            return x


class VariablePrimingNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 num_inputs,
                 hidden_size,
                 dropout,
                 context=None):
        super(VariablePrimingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context
        if self.context is not None:
            self.flattened_snp = SelfNormalizingPrimingNetwork(
                self.num_inputs * self.input_size, self.hidden_size,
                self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_snp = SelfNormalizingPrimingNetwork(
                self.num_inputs * self.input_size, self.hidden_size,
                self.num_inputs, self.dropout)
        self.single_variable_snps = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_snps.append(
                SelfNormalizingPrimingNetwork(self.input_size,
                                              self.hidden_size,
                                              self.hidden_size, self.dropout))
        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_snp(embedding, context)
        else:
            sparse_weights = self.flattened_snp(embedding)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        var_outputs = []
        for nnn in range(self.num_inputs):
            var_outputs.append(self.single_variable_snps[nnn](
                embedding[:, :, (nnn * self.input_size):(nnn + 1) *
                          self.input_size]))
        var_outputs = torch.stack(var_outputs, axis=-1)
        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(axis=-1)
        return outputs, sparse_weights


class ReGraFT(nn.Module):

    def __init__(self, config, num_nodes, adjs, device):
        super().__init__()
        self.config = config
        self.adjs = adjs
        self.cl = config["curriculum"]
        self.num_nodes = num_nodes
        self.past_steps = config["past_steps"]
        self.future_steps = config["future_steps"]
        self.in_channels = config["input_size"]
        self.hidden_size = self.in_channels
        self.hidden_size_wide = self.in_channels * config["embedding_dim"]
        print("self.hidden_size_wide", self.hidden_size_wide)
        self.device = device
        self.batch_size = config["BATCHSIZE"]
        self.static_variables = config["static_variables_len"]
        self.time_varying_categorical_variables = config[
            "time_varying_categorical_variables_len"]
        self.time_varying_real_variables_encoder = config[
            "time_varying_real_variables_encoder_len"]
        self.time_varying_real_variables_decoder = config[
            "time_varying_real_variables_decoder_len"]
        self.num_input_series_to_mask = config["num_masked_series"]
        if "rnn_layers" not in config:
            config["rnn_layers"] = 1
        self.rnn_layers = config["rnn_layers"]
        self.dropout = config["dropout"]
        self.embedding_dim = config["embedding_dim"]

        self.attn_heads = config["attn_heads"]
        if self.attn_heads == 0:
            self.use_transform = False
        self.total_steps = self.past_steps + self.future_steps
        config["static_variables_len"] = 0
        self.pos_fc_en = TimeDistributed(nn.Linear(self.hidden_size,
                                                   self.hidden_size_wide),
                                         batch_first=True)
        self.pos_fc_de = TimeDistributed(nn.Linear(self.hidden_size,
                                                   self.hidden_size_wide),
                                         batch_first=True)
        self.post_rnn_glu = TimeDistributed(GLU(self.hidden_size_wide))
        self.post_rnn_norm = TimeDistributed(
            nn.BatchNorm1d(self.hidden_size_wide))

        self.attn_heads = config["attn_heads"]
        if self.hidden_size_wide % self.attn_heads != 0:
            print(
                f"Model Embedding dimension ({self.hidden_size_wide}) is not divisible by attention heads ({self.attn_heads}). "
            )

        self.multihead_attn = mha.MultiheadAttention(config,
                                                     self.hidden_size_wide,
                                                     self.attn_heads)
        self.encoder_variable_priming = VariablePrimingNetwork(
            config["embedding_dim"],
            (config["time_varying_real_variables_encoder_len"] +
             config["time_varying_categorical_variables_len"]),
            self.hidden_size_wide,
            self.dropout,
            config["embedding_dim"] * config["static_variables_len"],
        )
        self.decoder_variable_priming = VariablePrimingNetwork(
            config["embedding_dim"],
            (config["time_varying_real_variables_decoder_len"] +
             config["time_varying_categorical_variables_len"] + 1),
            self.hidden_size_wide,
            self.dropout,
            config["embedding_dim"] * config["static_variables_len"],
        )
        self.gcn_depth = config["gcn_depth"]
        self.dropout_type = config["dropout_type"]
        self.dropout_prob = config["dropout"]
        self.encoder_graph_rnn = GraphEncoder(
            config=config,
            in_channels=self.in_channels,
            hidden_channels=self.hidden_size_wide,
            gcn_depth=self.gcn_depth,
            alpha=config["gcn_alpha"],
            past_steps=self.past_steps,
            dropout_prob=self.dropout_prob,
            dropout_type=self.dropout_type,
            node_num=self.num_nodes,
            static_norm_adjs=self.adjs,
            device=self.device,
        )
        self.decoder_graph_rnn = GraphDecoder(
            config=config,
            in_channels=self.in_channels,
            hidden_channels=self.hidden_size_wide,
            output_channels=self.hidden_size_wide,
            gcn_depth=self.gcn_depth,
            alpha=config["gcn_alpha"],
            past_steps=self.past_steps,
            dropout_prob=self.dropout_prob,
            dropout_type=self.dropout_type,
            node_num=self.num_nodes,
            static_norm_adjs=self.adjs,
            use_curriculum=self.cl,
            cl_decay_steps=self.config["curriculum_step"],
        )
        self.fc1 = nn.Conv1d(self.in_channels, self.hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(self.in_channels, self.hidden_size, kernel_size=1)
        self.mha_hid = self.in_channels
        self.post_attn_glu = TimeDistributed(GLU(self.mha_hid))
        self.post_attn_norm = TimeDistributed(
            nn.BatchNorm1d(self.mha_hid, self.mha_hid))
        self.out_channels = self.in_channels
        self.pos_wise_ff = SelfNormalizingPrimingNetwork(
            self.mha_hid, self.mha_hid, self.mha_hid, self.dropout)
        self.pre_output_norm = TimeDistributed(
            nn.BatchNorm1d(self.mha_hid, self.mha_hid))
        self.pre_output_glu = TimeDistributed(GLU(self.mha_hid))
        self.output_layer = TimeDistributed(nn.Linear(self.mha_hid,
                                                      self.out_channels),
                                            batch_first=True)

    def forward(self, x, adjs, global_step, test=False):

        print("■ Using Padded MHA")

        batch_size, _, _, self.in_channels = x["inputs"].size()
        self.batch_size = batch_size
        adjs = [a.to(self.device) for a in self.adjs]
        x_encoder = x["inputs"][:, :, :self.past_steps, :]

        x_decoder = x["inputs"][:, :, self.past_steps:, :]
        decoder_cl_label = x["inputs"][:, :, self.past_steps:, :].unsqueeze(-1)
        x_encoder_tmp = (x_encoder.permute(0, 1, 2, 3).contiguous().view(
            -1, self.batch_size * self.num_nodes, self.in_channels))
        x_decoder_tmp = (x_decoder.permute(0, 1, 2, 3).contiguous().view(
            -1, self.batch_size * self.num_nodes, self.in_channels))

        x_encoder_tmp = self.pos_fc_en(x_encoder_tmp)

        x_decoder_tmp = self.pos_fc_de(x_decoder_tmp)

        x_encoder, encoder_sparse_weights = self.encoder_variable_priming(
            x_encoder_tmp)
        x_decoder, decoder_sparse_weights = self.decoder_variable_priming(
            x_decoder_tmp)
        x_encoder = (x_encoder.permute(0, 1, 2).contiguous().view(
            self.batch_size, self.num_nodes, -1, self.hidden_size_wide))
        x_decoder = (x_decoder.permute(0, 1, 2).contiguous().view(
            self.batch_size, self.num_nodes, -1, self.hidden_size_wide))
        (encoder_outputs, encoder_hiddens,
         encoder_adjs_output_list) = self.encoder_graph_rnn(
             x_encoder, self.past_steps, adjs)
        x_decoder_seen = x_decoder[:, :, :, 1:self.hidden_size_wide]
        x_decoder_masked = torch.zeros(
            (batch_size, self.num_nodes, self.future_steps * 1, 1),
            device=self.device)
        x_decoder = torch.cat(
            [x_decoder_masked,
             x_decoder_seen.to(x_decoder_masked.device)],
            dim=-1)
        encoder_hiddens_ = encoder_hiddens[:, :, :, -1, :]
        decoder_output, _, decoder_adjs_output_list = self.decoder_graph_rnn(
            x_decoder,
            x_decoder,
            decoder_cl_label,
            encoder_hiddens_,
            adjs,
            task_level=self.future_steps,
            global_step=global_step,
        )
        decoder_adjs_output_list

        encoder_output = encoder_hiddens[:, -1, :, :, :]
        x_encoder_permute = (x_encoder.permute(0, 1, 2, 3).contiguous().view(
            -1, self.batch_size * self.num_nodes, self.hidden_size_wide))
        x_decoder_permute = (x_decoder.permute(0, 1, 2, 3).contiguous().view(
            -1, self.batch_size * self.num_nodes, self.hidden_size_wide))
        encoder_output_permute = (encoder_output.permute(
            0, 1, 2, 3).contiguous().view(-1, self.batch_size * self.num_nodes,
                                          self.hidden_size_wide))
        decoder_output_permute = (decoder_output.permute(
            0, 1, 2, 3).contiguous().view(-1, self.batch_size * self.num_nodes,
                                          self.hidden_size_wide))
        rnn_input = torch.cat([x_encoder_permute, x_decoder_permute], dim=0)
        rnn_output = torch.cat(
            [encoder_output_permute, decoder_output_permute], dim=0)
        rnn_output = self.post_rnn_glu(rnn_output + rnn_input)
        attn_input = self.post_rnn_norm(rnn_output)

        attn_output, attn_output_weights = self.multihead_attn(
            attn_input[self.past_steps:, :, :],
            attn_input[:self.past_steps, :, :],
            attn_input[:self.past_steps, :, :],
        )
        attn_output = self.post_attn_glu(attn_output) + attn_input[
            self.past_steps:, :, :]
        attn_output = self.post_attn_norm(attn_output)
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_glu(output) + rnn_output[
            self.past_steps:, :, :]
        output = self.pre_output_norm(output).view(self.batch_size,
                                                   self.num_nodes, -1,
                                                   self.hidden_size_wide)
        print(f"==>> final before output.shape: {output.shape}")

        output = self.output_layer(output).view(self.batch_size,
                                                self.num_nodes, -1,
                                                self.out_channels)
        output = output[:, :, :, 0].unsqueeze(-1)
        outputs_final = output

        return (
            outputs_final,
            attn_output_weights,
            encoder_sparse_weights,
            decoder_sparse_weights,
            encoder_adjs_output_list,
            decoder_adjs_output_list,
        )

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
