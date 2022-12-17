# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
import pandas as pd
from model import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

train_X = pd.read_csv('train_X.csv')
train_Y = pd.read_csv('train_Y.csv')
# print(train_X)
# print(train_Y)
featuresCC = ['Client', 'netCA_flow', 'TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'ActBal_CA', 'ActBal_SA', 'Age', 'Tenure']

clinets = train_X['Client']
X = train_X[featuresCC].fillna(0).drop(columns='Client').to_numpy()
Y_cc = train_Y['Sale_CC'].to_numpy()
Y_mf = train_Y['Sale_MF'].to_numpy()
Y_cl = train_Y['Sale_CL'].to_numpy()
# print(X)
# print(Y_cc)

X_train, X_test, Y_cc_train, Y_cctest = train_test_split(X, Y_cc, test_size=0.2, random_state = 20, stratify=Y_cc)

neg, pos = pd.value_counts(Y_cc_train).to_numpy()
total = neg + pos
print('Clients:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'
      .format(total, pos, 100 * pos / total))
print(pd.value_counts(Y_cc_train))

over = SMOTE(sampling_strategy=0.6, k_neighbors=2)
X_over, y_over = over.fit_resample(X_train, Y_cc_train)

print(pd.value_counts(y_over))

params = {'criterion': ['gini', 'entropy'],
          'min_samples_leaf': [1, 2, 3, 4, 5, 6],
          'max_depth': [3, 6, 8, 10, 13, 15, 18]}
model = DecisionTreeClassifier(class_weight={0: 0.6, 1: 2.3})
# Create gridsearch instance
grid = GridSearchCV(estimator=model, param_grid=params, cv=10,
                    n_jobs=1, scoring='average_precision',
                    verbose=1)
# Fit the model
grid.fit(X_train, Y_cc_train)
# Assess the score
print(grid.best_score_, grid.best_params_)
# print('Mean ROC AUC: %.3f' % mean(scores))
#
predDT = DecisionTreeClassifier(max_depth=grid.best_params_['max_depth'],
                                criterion=grid.best_params_['criterion'],
                                min_samples_leaf=grid.best_params_['min_samples_leaf']).fit(X_over, y_over).predict(X_test)
print(confusion_matrix(Y_cctest, predDT))
print(plot_cm(Y_cctest, predDT))
plt.show()