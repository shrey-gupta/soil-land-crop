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
Input: Sequence of shape (batch_size, seq_len, input_dim)

│
├── LSTM Layer (input_dim → hidden_dim)
│   • Processes input sequences through multiple LSTM layers
│   • Output shape: (batch_size, seq_len, hidden_dim)
│
├── Attention Layer
│   ├── Linear(hidden_dim → 1) → compute attention score for each timestep
│   ├── Softmax along seq_len → normalize scores to get attention weights
│   └── Weighted sum of LSTM outputs → context vector
│       • Output shape: (batch_size, hidden_dim)
│
├── Fully Connected Layer 1: Linear(hidden_dim → 256)
│   ├── LayerNorm
│   └── Activation: ReLU
│
├── Fully Connected Layer 2: Linear(256 → 64)
│   ├── LayerNorm
│   └── Activation: ReLU
│
├── Fully Connected Layer 3: Linear(64 → output_dim)
│
└── Output: Final prediction (regression or classification depending on output_dim)
'''

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # Linear layer to compute attention scores
    
    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        attn_scores = self.attn(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize scores
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # Weighted sum
        return context

class LSTMModel_with_att(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dims=[256, 64], output_dim=1, dropout=0.0, use_cuda=False):
        super(LSTMModel_with_att, self).__init__()
        self.name = 'LSTMModel_with_att'
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], output_dim)

        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(fc_dims[0])
        # self.bn3 = nn.BatchNorm1d(fc_dims[1])

        # Attention Layer
        self.attention = Attention(hidden_dim)

        # Layer Normalization for each layer
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.ln_fc1 = nn.LayerNorm(fc_dims[0])
        self.ln_fc2 = nn.LayerNorm(fc_dims[1])

    def forward(self, x):
        batch_size = x.size(0)  # Get current batch size

        # Initialize hidden state dynamically
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0.detach()))  # Pass through LSTM 
        context = self.attention(out)  

        # # Fully connected layers
        # last_state = F.relu(context)
        # last_state = F.relu(self.fc1(last_state))
        
        # out = F.relu(self.fc2(last_state))
        # out = self.fc3(out)

        # FC layers w/ layer normalization
        last_state = self.fc1(context)
        last_state = self.ln_fc1(last_state)
        last_state = F.relu(last_state)
    
        last_state = self.fc2(last_state)
        last_state = self.ln_fc2(last_state)
        last_state = F.relu(last_state)
    
        out = self.fc3(last_state)

        return out

def performance_report(y_true, y_pred):
    r_square = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r = pearsonr(y_true.flatten(), y_pred.flatten())[0]

    print('---------------')
    print(f'r_square: {r_square:.4f}')
    print(f'rmse: {rmse:.4f}')
    print(f'Pearson r: {r:.4f}')
    print(f'Pearson r^2: {r**2:.4f}')
    print(f'y_true mean: {y_true.mean():.4f}, y_pred mean: {y_pred.mean():.4f}')

def train_test_split_by_year_sequence(X, y_df, test_year, seed=42):
    train_index = y_df['year'] < test_year
    test_index = y_df['year'] == test_year

    np.random.seed(seed)
    valid_train_years = y_df['year'].unique()
    valid_train_years = valid_train_years[valid_train_years < (test_year - 1)]
    random_year = np.random.choice(valid_train_years)

    val_index = y_df['year'] == random_year

    X_train, y_train = X[train_index], y_df[train_index]
    X_test, y_test = X[test_index], y_df[test_index]
    X_val, y_val = X[val_index], y_df[val_index]

    return X_train, y_train, X_test, y_test, X_val, y_val



def main():
    y_df = pd.read_csv('/Users/guptsh/Downloads/Soil_Land_Crop/datasets/gridMET_data/modified/label.csv')
    y_df['pred'] = np.nan

    contents = ['tmmn', 'tmmx', 'pr', 'srad', 'sph', 'rmin', 'rmax', 'vs', 'th',  # 9 main
                'vpd', 'pet', 'etr', 'erc', 'bi', 'fm100', 'fm1000']  # except pdsi so far
    X = wr.read_feature_data(contents)

    ##### When using norm = "std"
    X, wind_zero_idx = wr.feature_engineering(X, contents, norm = "std", intersect = False)
    N, feature_num = X.shape[0], X.shape[1]

    ### daily
    X_dcpy = X.copy()
    X_day = X_dcpy.reshape((N, -1))

    ### weekly average
    X_wcpy = X.copy()
    X_wcpy = X_wcpy.reshape((-1, feature_num, 7, 30))
    X_wcpy = X_wcpy.mean(axis = 2)
    X_weekly = X_wcpy.reshape((N, -1))

    ### seasonal average
    X_scpy = X.copy()
    X_ave_season = X_scpy.reshape((-1, feature_num, 210, 1))
    X_ave_season = X_ave_season.mean(axis=2)
    X_ave_season = X_ave_season.reshape((-1, feature_num))

    print("Daily data shape:", X_day.shape)
    print("Weekly data shape:", X_weekly.shape)
    print("Seasonal data shape:", X_ave_season.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64 ##256
    all_years = np.arange(2000, 2006)  
    y_df = wr.county_detrend_func_loess(y_df, None, all_years)  


    for year in all_years:
        print(f"\nTest Year: {year}")
        
        X_train, y_train, X_test, y_test, X_val, y_val = train_test_split_by_year_sequence(X_day, y_df, year)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype = torch.float32).to(device)
        X_train_tensor = X_train_tensor.reshape(-1, 19, 210)
        X_train_tensor = X_train_tensor.permute(0, 2, 1).float()

        y_train_tensor = torch.tensor(y_train['detrend'].values, dtype=torch.float32).unsqueeze(1).to(device)  
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        X_test_tensor = X_test_tensor.reshape(-1, 19, 210)
        X_test_tensor = X_test_tensor.permute(0, 2, 1).float()
        
        y_test_tensor = torch.tensor(y_test['detrend'].values, dtype=torch.float32).unsqueeze(1).to(device)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_val_tensor = X_val_tensor.reshape(-1, 19, 210)
        X_val_tensor = X_val_tensor.permute(0, 2, 1).float()
        
        y_val_tensor = torch.tensor(y_val['detrend'].values, dtype=torch.float32).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

        model = LSTMModel_with_att(input_dim = 19, hidden_dim = 64, layer_dim = 2, use_cuda = torch.cuda.is_available()).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        num_epochs = 300

        # Training Loop
        for epoch in range(num_epochs):
            model.train()  
            train_loss = 0.0  # Track loss

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  
                
                preds = model(X_batch)  
                loss = loss_fn(preds, y_batch)  
                loss.backward()  
                
                optimizer.step()  
                train_loss += loss.item()
            
            # Compute training loss per batch
            train_loss /= len(train_loader)
    
            # --- Validation Step ---
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = loss_fn(val_preds, y_val_tensor).item()
    
            # Print both training and validation loss
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        model.eval()  
        with torch.no_grad():
            y_pred = model(X_test_tensor)

        y_true_np = y_test_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        y_df.loc[y_df['year'] == year, 'pred'] = y_pred_np.flatten()

        ## adding trend back to the prediction
        y_df[['year'] == year, 'orig_pred'] = y_df[['year'] == year, 'pred'] + y_df[['year'] == year, 'trend']

        print(f"\nPerformance Report for Year {year}:")
        # performance_report(y_true_np, y_pred_np)
        performance_report(y_df[['year'] == year, 'orig_pred'].values, y_df[['year'] == year, 'yield'].values)

    print("\nTraining completed for all years.")

    test_years = list(range(2000, 2006)) 
    final_df = y_df[y_df['year'].isin(test_years) & y_df['pred'].notna()]

    # y_true_all = final_df['detrend'].values
    # y_pred_all = final_df['pred'].values

    y_true_all = final_df['yield'].values
    y_pred_all = final_df['orig_pred'].values

    print("\n=== Overall Performance Report for 2000 to 2019 ===")
    performance_report(y_true_all, y_pred_all)

    final_df.to_csv("plots_data/lstm_with_attn_2000_2005_detrend_loess.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x = y_true_all, y = y_pred_all, alpha=0.7, edgecolor=None)
    plt.xlabel("True Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Scatter Plot: True vs Predicted Yield")
    plt.xlim(0)
    plt.ylim(0)
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig("plots_data/lstm_with_attn_2000_2005_detrend_loess.png")  


if __name__ == "__main__":
    main()