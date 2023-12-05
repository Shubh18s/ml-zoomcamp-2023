import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pickle

# Parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

# data preparation
df = pd.read_csv("data.csv")

columns = ['Make', 'Model', 'Year', 'Engine HP',
       'Engine Cylinders', 'Transmission Type', 'Vehicle Style',
       'highway MPG', 'city mpg', 'MSRP']

df = df[columns].copy()

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df = df.fillna(0)
df["above_average"] = np.where(df["msrp"] > df["msrp"].mean(), 1, 0)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']

categorical = ['make', 'model', 
       'transmission_type', 'vehicle_style']

# training

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X=dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation

Kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold=0
for train_idx, test_idx in Kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[test_idx]

    y_train = df_train.above_average.values
    y_val = df_val.above_average.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print('training the final model...')

dv, model = train(df_full_train, df_full_train.above_average.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.above_average.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

#### Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')