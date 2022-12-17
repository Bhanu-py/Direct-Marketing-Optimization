# fit a svm on an imbalanced classification dataset
import pandas as pd
from numpy import mean
from model import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

train_X = pd.read_csv('data/train_X.csv')
train_Y = pd.read_csv('data/train_Y.csv')
# print(train_X)
# print(train_Y)
# modeling without Over Sampling
# clinets = train_X['Client']
Y_cc = train_Y['Sale_CC']
Y_mf = train_Y['Sale_MF']
Y_cl = train_Y['Sale_CL']
Y_no = train_Y['No_sale']

# Features obtained from the feature selection done in RFfeature.py or in Jupyter Notebook
featuresMF = ['TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder',
              'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'netCA_flow',
              'Age', 'Tenure', 'Count_MF', 'ActBal_CA', 'ActBal_SA']
featuresCC = ['TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder',
              'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'netCA_flow',
              'Age', 'Tenure', 'ActBal_CA', 'ActBal_SA']
featuresCL = ['TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder',
              'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'netCA_flow',
              'Age', 'Tenure', 'ActBal_CA', 'ActBal_SA']
featuresNO = ['TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder',
              'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'netCA_flow',
              'Age', 'Tenure', 'Count_MF', 'ActBal_CA', 'ActBal_SA']

X_train, X_test, Y_cc_train, Y_cctest = train_test_split(X, Y_cc, test_size=0.2, random_state = 20, stratify=Y_cc)


X = X_train
y = Y_cc_train

neg, pos = pd.value_counts(Y_cc_train).to_numpy()
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
# total / (2 * np.bincount(Y_cc_train))

class_weight = {0: weight_for_0, 1: weight_for_1/1.5}
model = SVC(kernel='sigmoid', class_weight=class_weight, C=1, random_state=0)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
# summarize performance

print('Mean ROC AUC: %.3f' % mean(scores))

modelfit = model.fit(X, y)
predsvc = modelfit.predict(X_test)

# print(predsvc)
print('predsvc: ', roc_auc_score(Y_cctest, predsvc))
print(pd.value_counts(Y_cctest))
print(confusion_matrix(Y_cctest, predsvc))
print(plot_cm(Y_cctest, predsvc))
plt.show()
