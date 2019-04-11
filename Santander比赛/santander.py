import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import time
import datetime
from tqdm import tqdm
import lightgbm as lgb
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
import warnings
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
from sklearn import manifold
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

train = pd.read_csv('G:/collections/work/kaggle/santander/train.csv/train.csv')
test = pd.read_csv('G:/collections/work/kaggle/santander/test.csv/test.csv')

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


test_x = test.drop(['ID_code'], axis = 1)
train_x = train.drop(['ID_code', 'target'], axis = 1)
train_y = train['target']

n_fold = 2
folds = KFold(n_splits = n_fold, shuffle = True)
print(train_x.values)
train_x.head()

def train_model(X=train_x.values ,y=train_y.values,featurename=train_x.columns.tolist(), X_test=test_x, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        X_train, y_train = augment(X_train, y_train)
        X_train = pd.DataFrame(X_train)
        if model_type == 'lgb':
            train_data = lgb.Dataset(data=X_train, label=y_train)
            valid_data = lgb.Dataset(data=X_valid, label=y_valid)
            model = lgb.train(params,train_data,num_boost_round=20000,
                    valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3000)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_valid, pos_label=1)
        scores.append(metrics.auc(fpr, tpr))

        prediction += y_pred

    prediction = prediction / n_fold

    return oof, prediction



params = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    # 'top_rate':0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}

oof_lgb, prediction_lgb = train_model(params=params, model_type='lgb',plot_feature_importance=True)

submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = prediction_lgb
submission.to_csv("submission2.csv", index=False)