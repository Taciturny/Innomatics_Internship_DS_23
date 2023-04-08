#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_colwidth', None)
import re
import csv
from skimpy import skim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib



data = pd.read_csv('laptop_details.csv')
df = data.copy()


def extract_features(row):
    product = row['Product']
    feature = row['Feature']
    
    ram_size = re.findall(r'(\d+) ?(?:GB|TB)', feature)
    ram_size = ram_size[0] + ' GB' if ram_size else None
    
    ram_type = re.findall(r'(?:LP)?DDR\d\S*|Unified\sMemory', feature)
    ram_type = ram_type[0] if ram_type else None
    
    processor = re.findall(r'(?:AMD|Intel|M1|Qualcomm|Apple)[\s\w]+Processor', feature)
    processor = processor[0] if processor else None

    storage = re.findall(r'[\d]+\s(?:GB|TB)\s(?:HDD|SSD|EMMC)', product)
    storage = storage[0] if storage else None
    
    os = re.findall(r'(Windows (?:10|11)|Mac OS|Linux|DOS|Chrome)[\s\w]*Operating System', feature)
    os = os[0] if os else None

    display = re.findall(r'\d+(?:\.\d+)?\s*(?:cm|inch)\s*(?:\(|:)?\s*\d+(?:\.\d+)?\s*(?:cm|inch)?', feature)
    display = display[0] if display else None
    
    brand = re.findall(r'^\w+', product)
    brand = brand[0] if brand else None
    
    return pd.Series([ram_size, ram_type, processor, storage, os, display, brand], 
                     index=['Ram Size', 'Ram Type', 'Processor', 'Storage', 'OS', 'Display', 'Brand'])

df[['Ram Size', 'Ram Type', 'Processor', 'Storage', 'OS', 'Display', 'Brand']] = df.apply(extract_features, axis=1)


## Data Cleaning
df.MRP = df.MRP.apply(lambda x : x.replace('₹', '').replace(',', '')).astype(float)
df.drop(df.columns[[0, 1, 3, 10]], axis=1, inplace=True)

df['Ram Type'] = df['Ram Type'].str.replace('DDR4,','DDR4')

## Rename the column names
new_column_names = {'MRP': 'price',
                    'Ram Size': 'ram_size',
                    'Ram Type': 'ram_type',
                    'Processor': 'processor',
                    'Storage': 'storage',
                    'OS': 'os',
                    'Display': 'display',
                    'Brand': 'brand'}

# this extracts the last word in each value, as the storage type
df['storage_type'] = df.storage.apply(lambda x: x.split()[-1])

# using the split function, join the first 2 element
df['storage'] = df.storage.apply(lambda x: " ".join(x.split()[:2]))

df['display'] = df['display'].apply(lambda x: float(re.findall("\d+\.\d+|\d+", x)[0]))


dict_ram_type = {'LPDDR3':0, 'Unified Memory':1, 'LPDDR4':2,'DDR4':3, 'LPDDR4X':4, 'LPDDR5':5, 'DDR5':6}
dict_processor = {'AMD Athlon Dual Core Processor':0, 
                  'AMD Dual Core Processor':1, 
                  'Intel Celeron Dual Core Processor':2, 
                  'Intel Celeron Quad Core Processor':3, 
                  'Intel Pentium Quad Core Processor':4, 
                  'Intel Pentium Silver Processor':5, 
                  'AMD Ryzen 3 Dual Core Processor':6, 
                  'AMD Ryzen 3 Quad Core Processor':7, 
                  'AMD Ryzen 3 Hexa Core Processor':8, 
                  'AMD Ryzen 5 Dual Core Processor':9, 
                  'AMD Ryzen 5 Quad Core Processor':10, 
                  'AMD Ryzen 5 Hexa Core Processor':11, 
                  'AMD Ryzen 7 Quad Core Processor':12, 
                  'AMD Ryzen 7 Octa Core Processor':13, 
                  'AMD Ryzen 9 Octa Core Processor':14, 
                  'Apple M1 Processor':15, 
                  'Apple M1 Pro Processor':16, 
                  'Apple M1 Max Processor':17, 
                  'Apple M2 Processor':18, 
                  'Intel Core i3 Processor':19, 
                  'Intel OptaneIntel Core i3 Processor':20, 
                  'Intel Core i5 Processor':21, 
                  'Intel Evo Core i5 Processor':22, 
                  'Intel Core i7 Processor':23, 
                  'Intel Core i9 Processor':24, 
                  'Qualcomm Snapdragon 7c Gen 2 Processor':25}
dict_storage = {'32 GB':0, '64 GB':1, '128 GB':2, '256 GB':3, '512 GB':4, '1 TB':5, '2 TB':6}
dict_storage_type = {'EMMC':0, 'HDD':1, 'SSD':2}
dicts_cols = {'ram_type':dict_ram_type, 'processor':dict_processor, 'storage':dict_storage, 'storage_type': dict_storage_type}

le = LabelEncoder()

for col, col_dict in dicts_cols.items():
    df[col] = le.fit_transform(df[col].astype(str))

le = LabelEncoder()
df['os'] = le.fit_transform(df['os'])
os_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

#drop the target variable from the rest

X = df.drop('price', axis=1)
y = np.log(df['price'])

# split the data into 70% train set and 30% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ### We will use three models, Linear regression, Grandiant Boosting and Xgboost

models = {
    'Linear Regression': LinearRegression(),
    'Gbr': GradientBoostingRegressor(),
    'XGB': XGBRegressor()
}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluates given machine learning models.
    models : a dictionary of different Scikit-Learn or XGBoost machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : testing labels
    '''
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Make predictions with the trained model
        y_pred = model.predict(X_test)
        # Calculate model evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # Add the model evaluation metrics to the model_scores dictionary
        model_scores[name] = {'R2_score': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}
    return model_scores


model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)


# Model Scores Comparison

model_scores = {'Linear Regression': 0.5242482148894944,
                'Gbr': 0.9388765296011099,
                'XGB': 0.930967238775686}

# scores_series = pd.Series(model_scores)
# scores_series.plot.bar(rot=0);


# ## Hyperparameter Tuning
# # GridSearchCV for GradientBoostingRegressor
# Gbr_reg_grid = {'learning_rate': [0.05, 0.1, 0.15],
#                 'max_depth': [3, 4, 5],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'max_features': [None, 'sqrt', 'log2'],
#                 'n_estimators': [100, 200, 300]}



# # Setup random seed
# np.random.seed(42)

# # Setup random hyperparameter search for GradientBoostingRegressor
# gbr_grid = GridSearchCV(GradientBoostingRegressor(), 
#                         param_grid = Gbr_reg_grid,
#                         cv=5,
#                         n_jobs=-1,
#                         verbose=True)

# # Fit random hyperparameter search model for GradientBoostingRegressor()
# gbr_grid.fit(X_train, y_train)


# print(" Results from GridSearchCV " )
# print("\n The best parameters:\n", gbr_grid.best_params_)
# print("\n The best score:\n", gbr_grid.best_score_)

# GridModel = GradientBoostingRegressor(learning_rate=0.05, max_features='sqrt', max_depth=5, n_estimators=300, min_samples_split=10, min_samples_leaf = 2)
# GridModel.fit(X_train,y_train)
# y_pred_grid =GridModel.predict(X_test)

# evaluate the model
# mse = mean_squared_error(y_test, y_pred_grid)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred_grid)
# mae = mean_absolute_error(y_test, y_pred_grid)

# print("MSE: ", mse)
# print("RMSE: ", rmse)
# print("R-squared: ", r2)


# # RandomizedSearchCV for GradientBoostingRegressor
# Gbr_reg_rs = {'learning_rate': [0.05, 0.1, 0.15],
#               'max_depth': [3, 4, 5],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'max_features': [None, 'sqrt', 'log2'],
#               'n_estimators': [100, 200, 300]}



# # Setup random seed
# np.random.seed(42)

# # Setup random hyperparameter search for GradientBoostingRegressor
# gbr_rs = RandomizedSearchCV(GradientBoostingRegressor(), 
#                               param_distributions = Gbr_reg_rs,
#                               cv=5,
#                               n_jobs=-1,
#                               n_iter=20,
#                               verbose=True)

# # Fit random hyperparameter search model for GradientBoostingRegressor()
# gbr_rs.fit(X_train, y_train)

# print(" Results from RandomSearchCV " )
# print("\n The best parameters:\n", gbr_rs.best_params_)
# print("\n The best score:\n", gbr_rs.best_score_)

rsModel = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=300,max_features=None, min_samples_split=5, min_samples_leaf=4)
rsModel.fit(X_train,y_train)

y_pred_rs = rsModel.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred_rs)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rs)
mae = mean_absolute_error(y_test, y_pred_rs)

# print("MSE: ", mse)
# print("RMSE: ", rmse)
# print("R-squared: ", r2)
