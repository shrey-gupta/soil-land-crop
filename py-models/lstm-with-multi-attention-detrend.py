import pandas as pd
import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import pearsonr

import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame
from geopy.distance import geodesic
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

import torch.nn.functional as F
import wrangling as wr

'''
Input: (batch_size, seq_len, input_dim)
   │
   ▼
LSTM Layer (stacked if layer_dim > 1)
Output shape: (batch_size, seq_len, hidden_dim)
   │
   ▼
Multi-Head Attention Block:
   ├── Linear Projections: Q, K, V → (batch_size, seq_len, hidden_dim)
   ├── MultiHeadAttention(num_heads=4)
   ├── Residual Connection: Attention Output + LSTM Output
   └── LayerNorm(hidden_dim)
Output shape: (batch_size, seq_len, hidden_dim)
   │
   ▼
Temporal Pooling (mean across time steps)
Output shape: (batch_size, hidden_dim)
   │
   ▼
Fully Connected Layers:
   ├── Linear(hidden_dim → fc_dims[0]) → ReLU
   ├── Linear(fc_dims[0] → fc_dims[1]) → ReLU
   └── Linear(fc_dims[1] → output_dim)
Output shape: (batch_size, output_dim)
'''


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply multi-head attention
        attn_output, _ = self.attn(q, k, v)

        # Residual + LayerNorm
        out = self.layer_norm(attn_output + x)
        return out


class LSTMModel_with_multihead_att(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dims=[256, 64], output_dim=1,
                 dropout=0.1, n_heads=4, use_cuda=False):
        super(LSTMModel_with_multihead_att, self).__init__()
        self.name = 'LSTMModel_with_multihead_att'

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.use_cuda = use_cuda

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)

        # Multi-head Attention Block
        self.attention_block = MultiHeadAttentionBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden and cell state
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()

        if self.use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
            x = x.cuda()

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Apply Multi-head Attention with Residual & Norm
        attn_out = self.attention_block(lstm_out)

        # Mean pooling across time (seq_len)
        context = torch.mean(attn_out, dim=1)

        # Fully connected layers
        out = F.relu(self.fc1(context))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
