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

def array_std_norm_2d(X):
    std = X.std(axis=0)
    X = X[:, std!=0]
    X = (X - X.mean(axis=0))/X.std(axis=0)
    return X

def array_0_1_norm_2d(X):
    X = (X - X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))
    return X

def array_std_norm_3d(X):
    mean = X.mean(axis = (1,2)).reshape((-1,1,1))
    std = X.std(axis = (1,2)).reshape((-1,1,1))
    X = (X-mean)/std
    return X

def array_0_1_norm_3d(X):
    max = X.max(axis = (1,2)).reshape((-1,1,1))
    min = X.min(axis = (1,2)).reshape((-1,1,1))
    X = (X-min)/(max-min)
    return X

def feature_engineering(X, contents, norm = None, intersect = False, select_by_imp = False, wind = "all", wind_zero = "wd"):
    # wind is none, all or vs
    if norm == 'std':
        func_2d = array_std_norm_2d
        func_3d = array_std_norm_3d
    elif norm == '0_1':
        func_2d = array_0_1_norm_2d
        func_3d = array_0_1_norm_3d
    else:
        func_2d = lambda x: x
        func_3d = lambda x: x

    shape = X.shape

    tmmx_idx = contents.index('tmmx')
    tmmn_idx = contents.index('tmmn')
    tmp_diff = X[tmmx_idx,:,:] - X[tmmn_idx,:,:]
    tmp_diff = func_2d(tmp_diff)
    tmp_diff = tmp_diff.reshape([1] + list(shape)[1:])

    rmax_idx = contents.index('rmax')
    rmin_idx = contents.index('rmin')
    humidity_diff = X[rmax_idx, :, :] - X[rmin_idx, :, :]
    humidity_diff = func_2d(humidity_diff)
    humidity_diff = humidity_diff.reshape([1] + list(shape)[1:])

    wind_direction_idx = contents.index('th')
    angle = X[wind_direction_idx, :, :]/360 * 2 * np.pi
    direction_sin = np.sin(angle).reshape([1] + list(shape)[1:])
    direction_cos = np.cos(angle).reshape([1] + list(shape)[1:])

    vs_idx = contents.index('vs')
    print("vs idx is {}".format(vs_idx))
    vs = X[vs_idx, :, :].reshape([1] + list(shape)[1:])

    X = np.delete(X, [vs_idx, wind_direction_idx], axis = 0)

    X = func_3d(X)
    if wind == "all":
        X = np.concatenate((X,tmp_diff,humidity_diff,vs,direction_sin,direction_cos), axis = 0)
    elif wind == "none":
        X = np.concatenate((X, tmp_diff, humidity_diff), axis=0)
    elif wind == "vs":
        X = np.concatenate((X, tmp_diff, humidity_diff, vs), axis=0)

    # feature selection: use season averaged feature to get feature importance through RF model.
    feature_importance = [0.02193165, 0.06284084, 0.04975114, 0.10442722, 0.03611288,
                                   0.0040242, 0.01542977, 0.0218085, 0.15880153, 0.00250056,
                                   0.00381332, 0.00489928, 0.00491362, 0.00160292, 0.01153419,
                                   0.43149476, 0.01074497, 0.04018165, 0.01318702]
    feature_importance = feature_importance[:vs_idx] + feature_importance[vs_idx+1:-2] + [feature_importance[vs_idx]] + feature_importance[-2:]

    if wind == "all":
        feature_importance = feature_importance
    elif wind == "none":
        feature_importance = feature_importance[:-3]
    elif wind == "vs":
        feature_importance = feature_importance[:-2]

    if select_by_imp:
        important_idx = [i for i, x in enumerate(feature_importance) if x>0.04]
        keep_idx = [i for i, x in enumerate(feature_importance) if x>0.01]
    else:
        keep_idx = list(range(len(feature_importance)))
    # county_feature_imp = important_feature_by_modeling(X, y_df['yield'], threshold='mean')

    if intersect:
        X_inter = None
        for i in range(len(important_idx)):
            for j in range(i, len(important_idx)):
                X_add = np.multiply(X[i,:,:],X[j,:,:]).reshape([1] + list(shape)[1:])
                if X_inter is None:
                    X_inter = X_add
                else:
                    X_inter = np.concatenate((X_inter, X_add), axis = 0)
        X = np.concatenate((X[keep_idx,:,:], X_inter), axis = 0)
        # inter = intersect_array(county_feature_imp, county_feature_imp)
        # X = np.concatenate((county_feature_imp, inter), axis=1)
    else:
        X = X[keep_idx,:,:]
        # X = county_feature_imp

    # remain wind idx is used to test the model performance by set wind features as 0.
    if wind == "all":
        wind_zero_idx = list(range(len(feature_importance)))[-2:]
    elif wind == "none":
        wind_zero_idx = None
    elif wind == "vs":
        wind_zero_idx = None

    X = X.swapaxes(0,1)
    print(X.shape)

    return X, wind_zero_idx

def read_feature_data(contents):
    X = None
    for content in contents:
        X_content = np.load('/Users/guptsh/Downloads/Soil_Land_Crop/datasets/gridMET_data/modified/X_{}'.format(content), allow_pickle=True)
        print("pre-shape", X_content.shape)
        shape = [1] + list(X_content.shape)
        X_content = X_content.reshape(shape)
        print("post-shape", X_content.shape)
        
        if X is None:
            X = X_content
        else:
            X = np.concatenate((X, X_content),axis=0)
    
    return X

def county_detrend_func(y_df, test_year, all_years, record_count_threshold = 3):
    y_df['trend'] = np.nan
    y_df['detrend'] = np.nan
    y_df['county_detrend_flag'] = np.nan
    unique_county_ids = y_df['id2'].unique()
    for id in unique_county_ids:
        county_df = y_df[y_df['id2'] == id]

        X = np.array(county_df.year).reshape((-1, 1))
        y = np.array(county_df['yield'])

        # print(X.shape)
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_df.loc[y_df['id2'] == id, 'trend'] = y_pred

        if county_df.shape[0] <record_count_threshold:
            y_df.loc[y_df['id2'] == id, 'county_detrend_flag'] = False
        else:
            y_df.loc[y_df['id2'] == id, 'county_detrend_flag'] = True
    y_df['detrend'] = y_df['yield'] - y_df['trend']
    
    return y_df

def normalize_feature_county(X, y_df, replace = True):
    unique_county_ids = y_df['id2'].unique()
    if replace:
        sub_X = X.copy()
        for id in unique_county_ids:
            county_X = X[y_df['id2'] == id]
            feature_mean = county_X.mean(axis = 0)
            sub_X[y_df['id2'] == id] = (county_X - feature_mean)
        return sub_X
    else:
        sub_X = X.copy()
        for id in unique_county_ids:
            county_X = X[y_df['id2'] == id]
            feature_mean = county_X.mean(axis=0, keepdims = True)
            sub_X[y_df['id2'] == id] = feature_mean
        new_X = np.concatenate((X, sub_X), axis = 1)
        return new_X