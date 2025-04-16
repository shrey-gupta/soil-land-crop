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


class BiLSTMModel_no_att(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dims = [256, 64], output_dim=1, dropout = 0.0, use_cuda=False): 
        super(BiLSTMModel_no_att, self).__init__()
        self.name = 'LSTMModel_no_att'
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout, bidirectional=True)

        self.fc1 = nn.Linear(2 * hidden_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], output_dim)

        self.bn1 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(fc_dims[0])
        self.bn3 = nn.BatchNorm1d(fc_dims[1])

    def forward(self, x):
        batch_size = x.size(0)  # Get current batch size

        # Initialize hidden state dynamically
        h0 = torch.zeros(2 * self.layer_dim, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(2 * self.layer_dim, batch_size, self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))  # Pass through LSTM
        state = torch.mean(out, dim=1)  # Reduce sequence dimension

        # Fully connected layers
        last_state = F.relu(state)
        last_state = F.relu(self.fc1(last_state))
        
        out = F.relu(self.fc2(last_state))
        out = self.fc3(out)

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

def train_test_split_by_year_sequence(X, y_df, test_year):
    train_index = y_df['year'] < test_year
    test_index = y_df['year'] == test_year

    X_train, y_train = X[train_index], y_df[train_index]
    X_test, y_test = X[test_index], y_df[test_index]

    return X_train, y_train, X_test, y_test


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
    
    batch_size = 256
    all_years = np.arange(2000, 2019)  
    y_df = wr.county_detrend_func(y_df, None, all_years)  


    model = BiLSTMModel_no_att(input_dim = 19, hidden_dim = 64, layer_dim = 2, use_cuda = torch.cuda.is_available()).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    for year in all_years:
        print(f"\nTest Year: {year}")
        
        X_train, y_train, X_test, y_test = train_test_split_by_year_sequence(X_day, y_df, year)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype = torch.float32).to(device)
        X_train_tensor = X_train_tensor.reshape(-1, 19, 210)
        X_train_tensor = X_train_tensor.permute(0, 2, 1).float()

        y_train_tensor = torch.tensor(y_train['yield'].values, dtype=torch.float32).unsqueeze(1).to(device)  
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        X_test_tensor = X_test_tensor.reshape(-1, 19, 210)
        X_test_tensor = X_test_tensor.permute(0, 2, 1).float()
        
        y_test_tensor = torch.tensor(y_test['yield'].values, dtype=torch.float32).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

        # Training Loop
        for epoch in range(num_epochs):
            model.train()  
            epoch_loss = 0.0  # Track loss

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  
                
                preds = model(X_batch)  
                loss = loss_fn(preds, y_batch)  
                loss.backward()  
                
                optimizer.step()  
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

        model.eval()  
        with torch.no_grad():
            y_pred = model(X_test_tensor)

        y_true_np = y_test_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        y_df.loc[y_df['year'] == year, 'pred'] = y_pred_np.flatten()

        print(f"\nPerformance Report for Year {year}:")
        performance_report(y_true_np, y_pred_np)

    print("\nTraining completed for all years.")


if __name__ == "__main__":
    main()