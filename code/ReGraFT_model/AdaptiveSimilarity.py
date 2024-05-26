import torch.nn.functional as F
import torch.nn as nn
import torch, os, math


class AdaptiveSimilarityGenerator_pool(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout_prob,
        node_num,
        reduction=1,
        alpha=1.,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.start_fc = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        reduction = 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc2 = nn.Sequential(
            nn.Conv1d(node_num, hidden_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(hidden_channels // reduction, node_num, kernel_size=1),
            nn.Sigmoid())
        self.dropout = nn.Dropout(dropout_prob)
        self.alpha = alpha

    def forward(self, x, hidden):
        batch_size, node_num, hidden_dim = x.shape
        node_feature = x + hidden
        node_feature = self.start_fc(node_feature.permute((0, 2, 1))).permute(
            (0, 2, 1))
        node_feature_pool = node_feature
        node_feature_pool1 = self.avg_pool(node_feature_pool).permute(
            (0, 2, 1))
        node_feature_pool2 = self.max_pool(node_feature_pool).permute(
            (0, 2, 1))
        node_feature_pool = (node_feature_pool1 + node_feature_pool2) / 2
        node_feature_pool = self.fc2(node_feature_pool.permute(
            (0, 2, 1))).permute((0, 2, 1))
        node_feature = torch.mul(node_feature_pool.expand_as(node_feature),
                                 node_feature)
        similarity = torch.matmul(node_feature, node_feature.transpose(
            2, 1)) / math.sqrt(hidden_dim)
        adj = F.relu(torch.tanh(self.alpha * similarity))
        norm_adj = adj / torch.unsqueeze(adj.sum(dim=-1), dim=-1)
        return norm_adj, adj, similarity


class AdaptiveSimilarityGenerator_attn(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout_prob,
        node_num,
        reduction=1,
        alpha=1.,
    ):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels + hidden_channels,
                             hidden_channels,
                             kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_channels,
                                               num_heads=2,
                                               dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.alpha = alpha

    def forward(self, x, hidden):
        batch_size, node_num, hidden_dim = x.shape
        node_feature = torch.cat([x, hidden], dim=-1)
        node_feature = node_feature.permute(0, 2, 1)
        node_feature = self.fc1(node_feature)
        node_feature = F.relu(node_feature)
        node_feature = self.dropout(node_feature)
        node_feature = self.fc2(node_feature)
        node_feature = node_feature.permute(2, 0, 1)
        node_feature = self.dropout(node_feature)
        attn_output, _ = self.attention(node_feature, node_feature,
                                        node_feature)
        attn_output = attn_output.permute(1, 0, 2)
        similarity = torch.matmul(attn_output, attn_output.transpose(
            2, 1)) / math.sqrt(hidden_dim)
        adj = F.relu(torch.tanh(self.alpha * similarity))
        norm_adj = adj / torch.unsqueeze(adj.sum(dim=-1), dim=-1)
        return norm_adj, adj, similarity


class AdaptiveSimilarityGenerator_fc(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 dropout_prob,
                 node_num,
                 reduction=1,
                 alpha=1.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.alpha = alpha

    def forward(self, x, hidden):
        batch_size, node_num, hidden_dim = x.shape
        node_feature = x + hidden
        node_feature = node_feature.permute(0, 2, 1)
        node_feature = self.fc1(node_feature)
        node_feature = F.relu(node_feature)
        node_feature = self.dropout(node_feature)
        node_feature = self.fc2(node_feature)
        node_feature = node_feature.permute(0, 2, 1)
        similarity = torch.matmul(node_feature, node_feature.transpose(
            2, 1)) / math.sqrt(hidden_dim)
        adj = F.relu(torch.tanh(self.alpha * similarity))
        norm_adj = adj / torch.unsqueeze(adj.sum(dim=-1), dim=-1)
        return norm_adj, adj, similarity
