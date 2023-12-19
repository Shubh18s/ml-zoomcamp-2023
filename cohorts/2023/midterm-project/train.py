import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pickle


# parameters
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'
train_data = "adult.data"
test_data = "adult.test"


# data preparation
columns=('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income')

df_full_train = pd.read_csv(train_data, names=columns)
df_test = pd.read_csv(test_data, names=columns, skiprows=1)

df_full_train.columns = df_full_train.columns.str.lower().str.replace('-', '_')
df_test.columns = df_test.columns.str.lower().str.replace('-', '_')

categorical_columns = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
for col in categorical_columns:
    df_full_train[col] = df_full_train[col].str.lower().str.strip().str.replace('-', '_')
    df_test[col] = df_test[col].str.lower().str.strip().str.replace('-', '_')

df_full_train = df_full_train.replace('?', np.nan)
df_full_train = df_full_train.dropna()

df_full_train['income>50k'] = (df_full_train['income'] == '>50k').astype(int)
df_test['income>50k'] = (df_test['income'] == '>50k.').astype(int)

categorical = ['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'native_country']
numerical = ['age', 'capital_gain', 'capital_loss','hours_per_week']


# training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

# prediction
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X=dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation
Kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0
for train_idx, test_idx in Kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[test_idx]

    y_train = df_train['income>50k'].values
    y_val = df_val['income>50k'].values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')   
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


print('training the final model...')
dv, model = train(df_full_train, df_full_train['income>50k'].values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test['income>50k'].values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')


# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
