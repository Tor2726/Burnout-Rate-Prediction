# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:18:03 2022

@author: Kenny
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# Category Encoder (categorical variable)
import category_encoders as ce
# Min max normalization (numerical variable)
from sklearn.preprocessing import MinMaxScaler
# Model (can also view the scikit learn to choose the model)
from xgboost import XGBRegressor


#%%
os.chdir('D:\清大\課業\一上\AI\HW2')

df = pd.read_csv(r'train.csv')
df_test = pd.read_csv(r'test.csv')

#%% 處理缺失值
df = df.drop(['Employee ID','Date of Joining'],axis=1)
df_test = df_test.drop(['Employee ID','Date of Joining'],axis=1)

print(df.isna().sum())
print(df_test.isna().sum())

df = df.fillna(0)
df_test = df_test.fillna(0)

#%%
X = df[df.columns[0:-1]]
y = df[df.columns[-1]]

X_test = df_test

#%% 類別型資料
X_category = X[['Gender','Company Type','WFH Setup Available']]
X_test_category = X_test[['Gender','Company Type','WFH Setup Available']]

CE = ce.TargetEncoder().fit(X_category,y)
X[['Gender','Company Type','WFH Setup Available']] = CE.transform(X_category)
X_test[['Gender','Company Type','WFH Setup Available']] = CE.transform(X_test_category)









#%%
# 建立 XGBClassifier 模型
model = XGBRegressor()
# 使用訓練資料訓練模型
model.fit(X, y)

#%%
from sklearn.model_selection import cross_val_score, cross_validate
# directly use cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("R2: %0.2f (std: %0.2f)" % (scores.mean(), scores.std()))
print(scores)
# need other scoring
scores = cross_validate(model, X, y, cv=5,
                       scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error'))
print("====================================")
print("R2: {} (std: {})".format(scores['test_r2'].mean(), scores['test_r2'].std()))
print("MSE: {} (std: {})".format(-scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print("MAPE: {} (std: {})".format(-scores['test_neg_mean_absolute_percentage_error'].mean(), scores['test_neg_mean_absolute_percentage_error'].std()))

#%%
y_pred = model.predict(X_test)

#%%
df_ans = pd.DataFrame(y_pred.astype(float), columns=['Burn Rate'])
df_ans.to_csv('mySubmission.csv', index_label='Employee ID')












