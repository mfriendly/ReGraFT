import torch.nn.functional as F
import torch
import torch.nn as nn
from ReGraFT_model.GCN_fusion import GCN_Fus as GCN
import math


class MGGRUCell(nn.Module):

    def __init__(
        self,
        config,
        in_channels,
        hidden_channels,
        dropout_type="zoneout",
        gcn_depth=2,
        dropout_prob=0.3,
        node_num=51,
        static_norm_adjs=None,
        alpha=1,

    ):
        super().__init__()
        self.config = config
        self.end_attn = config['end_attn']
        self.start_attn = config['start_attn']
        self.gcn_depth = gcn_depth
        self.hidden_channels = hidden_channels
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.xh_fc = nn.Linear(28, hidden_channels)
        self.t_fc = nn.Linear(28, hidden_channels)
        self.query_A = nn.Linear(hidden_channels, hidden_channels)
        self.key_A = nn.Linear(hidden_channels, hidden_channels)
        self.value_A = nn.Linear(hidden_channels, hidden_channels)
        self.query_B = nn.Linear(hidden_channels, hidden_channels)
        self.key_B = nn.Linear(hidden_channels, hidden_channels)
        self.value_B = nn.Linear(hidden_channels, hidden_channels)
        self.attention = nn.Parameter(torch.Tensor(3, 1))
        self.bn_A = nn.BatchNorm1d(hidden_channels)
        self.bn_B = nn.BatchNorm1d(hidden_channels)
        self.input_FC = nn.Linear((hidden_channels + hidden_channels),
                                  hidden_channels)
        self.gate_FC1 = nn.Linear(hidden_channels, hidden_channels)
        dropout_prob = config["dropout"]
        if config['adaptive_graph'] == 'attn':
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_attn as AdaptiveSimilarityGenerator
        elif config['adaptive_graph'] == 'fc':
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_fc as AdaptiveSimilarityGenerator
        elif config['adaptive_graph'] == 'pool':
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_pool as AdaptiveSimilarityGenerator
        elif config['adaptive_graph'] == 'fusionfap' or config[
                'adaptive_graph'] == 'fusionfp' or config[
                    'adaptive_graph'] == 'fusionfa':
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_attn as AdaptiveSimilarityGenerator_attn
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_fc as AdaptiveSimilarityGenerator_fc
            from ReGraFT_model.AdaptiveSimilarity import AdaptiveSimilarityGenerator_pool as AdaptiveSimilarityGenerator_pool
        self.adapGraph_attn = AdaptiveSimilarityGenerator_attn(
            hidden_channels,
            hidden_channels,
            dropout_prob,
            node_num=node_num,
            reduction=16,
            alpha=alpha)
        self.adapGraph_fc = AdaptiveSimilarityGenerator_fc(hidden_channels,
                                                           hidden_channels,
                                                           dropout_prob,
                                                           node_num=node_num,
                                                           reduction=16,
                                                           alpha=alpha)
        self.adapGraph_pool = AdaptiveSimilarityGenerator_pool(
            hidden_channels,
            hidden_channels,
            dropout_prob,
            node_num=node_num,
            reduction=16,
            alpha=alpha)
        self.static_norm_adjs = static_norm_adjs
        self.update_GCN1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth,
                               dropout_prob,
                               len(static_norm_adjs) + 1)
        self.update_GCN2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth,
                               dropout_prob,
                               len(static_norm_adjs) + 1)
        self.reset_GCN1 = GCN(hidden_channels * 2, hidden_channels, gcn_depth,
                              dropout_prob,
                              len(static_norm_adjs) + 1)
        self.reset_GCN2 = GCN(hidden_channels * 2, hidden_channels, gcn_depth,
                              dropout_prob,
                              len(static_norm_adjs) + 1)
        self.layerNorm = nn.LayerNorm([self.hidden_channels])
        self.state_candidate_GCN1 = GCN(hidden_channels * 2, hidden_channels,
                                        gcn_depth, dropout_prob,
                                        len(static_norm_adjs) + 1)
        self.state_candidate_GCN2 = GCN(hidden_channels * 2, hidden_channels,
                                        gcn_depth, dropout_prob,
                                        len(static_norm_adjs) + 1)
        self.selu1 = nn.SELU()

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
                masks == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

    def forward(self,
                x,
                input_time,
                Hidden_State,
                adjs,
                encoder_hidden=None,
                adaptive=True):
        if self.start_attn:
            attention_output = self.self_attention(x, self.query_A, self.key_A,
                                                   self.value_A,
                                                   self.hidden_channels, None,
                                                   None)
            attention_output = self.selu1(attention_output)
            combined_output = x + attention_output
            combined_output = self.selu1(combined_output)
            combined_output = F.dropout(combined_output,
                                        p=self.dropout_prob,
                                        training=self.training)
            x = self.layerNorm(combined_output)
        if x.dim() == 4:
            x = x.cuda().squeeze(2)
        batch_size, node_num, hidden_channels = x.shape
        self.static_norm_adjs = adjs
        Hidden_State = Hidden_State.cuda()
        Hidden_State = Hidden_State.view(batch_size, node_num,
                                         self.hidden_channels)
        if encoder_hidden is not None:
            Hidden_State = Hidden_State + encoder_hidden
        combined = torch.cat((x, Hidden_State), -1)
        if adaptive:
            adap_norm_adj = []
            adap_norm_adjT = []
            if self.config['adaptive_graph'] == 'fusionfap' or self.config[
                    'adaptive_graph'] == 'fusionfa' or self.config[
                        'adaptive_graph'] == 'fusionap' or self.config[
                            'adaptive_graph'] == 'attn':
                adap_norm_adj_attn, adap_adj_attn, correlation_attn = self.adapGraph_attn(
                    x.float(), Hidden_State.float())
                del adap_adj_attn
                adap_norm_adjT_attn = adap_norm_adj_attn.transpose(1, 2)
                adap_norm_adj.append(adap_norm_adj_attn)
                adap_norm_adjT.append(adap_norm_adjT_attn)
            if self.config['adaptive_graph'] == 'fusionfap' or self.config[
                    'adaptive_graph'] == 'fusionfp' or self.config[
                        'adaptive_graph'] == 'fusionfa' or self.config[
                            'adaptive_graph'] == 'fc':
                adap_norm_adj_fc, adap_adj_fc, correlation_fc = self.adapGraph_fc(
                    x.float(), Hidden_State.float())
                del adap_adj_fc
                adap_norm_adjT_fc = adap_norm_adj_fc.transpose(1, 2)
                adap_norm_adj.append(adap_norm_adj_fc)
                adap_norm_adjT.append(adap_norm_adjT_fc)
            if self.config['adaptive_graph'] == 'fusionfap' or self.config[
                    'adaptive_graph'] == 'fusionfp' or self.config[
                        'adaptive_graph'] == 'fusionfp' or self.config[
                            'adaptive_graph'] == 'pool':
                adap_norm_adj_pool, adap_adj_pool, correlation_pool = self.adapGraph_pool(
                    x.float(), Hidden_State.float())
                del adap_adj_pool
                adap_norm_adjT_pool = adap_norm_adj_pool.transpose(1, 2)
                adap_norm_adj.append(adap_norm_adj_pool)
                adap_norm_adjT.append(adap_norm_adjT_pool)
                adap_norm_adj.append(adap_norm_adj_pool)
                adap_norm_adjT.append(adap_norm_adjT_pool)
        else:
            adap_norm_adj = adap_norm_adjT = []
        norm_adjs = [adj for adj in self.static_norm_adjs]
        norm_adjTs = [adj.T for adj in self.static_norm_adjs]
        update_gate = torch.sigmoid(
            self.update_GCN1(combined, norm_adjs, adap_norm_adj) +
            self.update_GCN2(combined, norm_adjTs, adap_norm_adjT))
        update_gate = F.dropout(update_gate,
                                p=self.dropout_prob,
                                training=self.training)
        reset_gate = torch.sigmoid(
            self.reset_GCN1(combined, norm_adjs, adap_norm_adj) +
            self.reset_GCN2(combined, norm_adjTs, adap_norm_adjT))
        reset_gate = F.dropout(reset_gate,
                               p=self.dropout_prob,
                               training=self.training)
        name = "selu"
        fn = self._get_activation(name)
        Hidden_State = fn(Hidden_State)
        temp = torch.cat((x, torch.mul(reset_gate, Hidden_State)), -1)
        state_candidate_State = torch.tanh(
            self.state_candidate_GCN1(temp, norm_adjs, adap_norm_adj) +
            self.state_candidate_GCN2(temp, norm_adjTs, adap_norm_adjT))
        state_candidate_State = F.dropout(state_candidate_State,
                                          p=self.dropout_prob,
                                          training=self.training)
        next_Hidden_State = torch.mul(update_gate, Hidden_State) + torch.mul(
            1.0 - update_gate, state_candidate_State)
        next_Hidden_State = F.dropout(next_Hidden_State,
                                      p=self.dropout_prob,
                                      training=self.training)
        if self.end_attn:
            attention_output = self.self_attention(next_Hidden_State,
                                                   self.query_B, self.key_B,
                                                   self.value_B,
                                                   self.hidden_channels, None,
                                                   None)
            attention_output = self.selu1(attention_output)
            attention_output = F.dropout(attention_output,
                                         p=self.dropout_prob,
                                         training=self.training)
            combined_output = x + attention_output
            combined_output = self.selu1(combined_output)
            next_Hidden_State = combined_output
        next_hidden = self.layerNorm(next_Hidden_State)
        output = next_hidden
        if self.dropout_type == "zoneout":
            next_hidden = self.zoneout(prev_h=Hidden_State,
                                       next_h=next_hidden,
                                       rate=self.dropout_prob,
                                       training=self.training)
        output = output + x
        output = self.selu1(output)
        adap_norm_adj = [a.float() for a in adap_norm_adj]
        return (output.float(), next_hidden.float(), adap_norm_adj,
                correlation_fc.float())

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

    def zoneout(self, prev_h, next_h, rate, training=True):
        if training:
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * prev_h + (1 - d) * next_h
        else:
            next_h = rate * prev_h + (1 - rate) * next_h
        return next_h
