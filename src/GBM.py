# grid search positive class weights with xgboost for imbalance classification
from numpy import mean
import pandas as pd
from model import *
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# generate dataset
train_X = pd.read_csv('data/train_X.csv')
train_Y = pd.read_csv('data/train_Y.csv')
# print(train_X)
# print(train_Y)

Y_cc = train_Y['Sale_CC'].to_numpy()
Y_mf = train_Y['Sale_MF'].to_numpy()
Y_cl = train_Y['Sale_CL'].to_numpy()
Y_no = train_Y['No_sale']
# print(X)
# print(Y_cc)

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

X_traincc, X_testcc, Y_traincc, Y_testcc = train_test_split(train_X[featuresCC].fillna(0),
                                                        Y_cc, test_size=0.2, random_state=20, stratify=Y_cc)
X_trainmf, X_testmf, Y_trainmf, Y_testmf = train_test_split(train_X[featuresMF].fillna(0),
                                                            Y_mf, test_size=0.2, random_state=20, stratify=Y_mf)
X_traincl, X_testcl, Y_traincl, Y_testcl = train_test_split(train_X[featuresCL].fillna(0),
                                                            Y_cl, test_size=0.2, random_state=20, stratify=Y_cl)
X_trainno, X_testno, Y_trainno, Y_testno = train_test_split(train_X[featuresNO].fillna(0),
                                                            Y_no, test_size=0.2, random_state=20, stratify=Y_no)

# # print(X_trainno.shape)
# # score = {'AUC': 'roc_auc', 'fbeta':'f1_weighted', 'jacc': 'jaccard', 'f1_mic': 'f1_micro', 'F1': 'f1'}
# # refit = 'F1'
features_to_normalize = ['netCA_flow', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'TransactionsDebCash_Card',
                         'VolumeDeb_PaymentOrder', 'ActBal_CA', 'ActBal_SA', 'TransactionsCred_CA', 'TransactionsDeb_PaymentOrder',
                         'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'Age', 'Tenure', 'TransactionsDeb_CA', 'Count_MF']
features_to_normalize = [i for i in features_to_normalize if i in featuresCC]
features_to_normalize = [i for i in features_to_normalize if i in featuresMF]
features_to_normalize = [i for i in features_to_normalize if i in featuresCL]

X_traincc[features_to_normalize] = StandardScaler().fit_transform(X_traincc[features_to_normalize])
X_testcc[features_to_normalize] = StandardScaler().fit_transform(X_testcc[features_to_normalize])


X_trainmf[features_to_normalize] = StandardScaler().fit_transform(X_trainmf[features_to_normalize])
X_testmf[features_to_normalize] = StandardScaler().fit_transform(X_testmf[features_to_normalize])

X_traincl[features_to_normalize] = StandardScaler().fit_transform(X_traincl[features_to_normalize])
X_testcl[features_to_normalize] = StandardScaler().fit_transform(X_testcl[features_to_normalize])


# Over-Sampling
over = SMOTEENN(sampling_strategy='auto')
X_over, y_over = over.fit_resample(X_trainmf, Y_trainmf)

X = X_trainmf
y = Y_trainmf
cat = ['Sex', 'Count_CA', 'Count_SA', 'Count_OVD', 'Count_CC', 'Count_CL']
cat = ['c' if i in cat else 'q' for i in X_traincc.columns]

# print(X_train.columns)
# print(X_train)
print(pd.value_counts(y))

# tuning
# define model
modelmf = XGBClassifier()
# define grid
weights = [1, 5, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights,
                  max_depth=[6,8,10,12,16]
                  # max_delta_step=[i for i in range(10)]
                  )
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
# scoring = {"AUC": "roc_auc", "f1_macro": "f1_macro", "f1_weighted":"f1_weighted"}
gridmf = GridSearchCV(estimator=modelmf, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1', verbose=1)
# execute the grid search
grid_resultmf = gridmf.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_resultmf.best_score_, grid_resultmf.best_params_))
# print(grid_result.cv_results_)
# report all configurations
means = grid_resultmf.cv_results_['mean_test_score']
stds = grid_resultmf.cv_results_['std_test_score']
params = grid_resultmf.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print('Best Parameters: ', gridmf.best_params_)

# model.fit(X, y, sample_weight=[99]).summary()
predmf = gridmf.predict(X_testmf)
print(roc_auc_score(Y_testmf, predmf))
print(pd.value_counts(Y_testmf))
print(confusion_matrix(Y_testmf, predmf))
# print(plot_cm(Y_testcc, pred))
# plt.show()

modelmf = XGBClassifier(max_depth=gridmf.best_params_['max_depth'],
                       scale_pos_weight=gridmf.best_params_['scale_pos_weight'])
modelmf.fit(X, y)
predmf = modelmf.predict(X_testmf.to_numpy())
print('pred1: ', roc_auc_score(Y_testmf, predmf))
print(pd.value_counts(Y_testmf))
print(confusion_matrix(Y_testmf, predmf))
print(plot_cm(Y_testmf, predmf, p=0.45))
plt.show()

print('F1-Score MF: {}\n'.format(f1_score(Y_testmf, predmf)))
print('Fbeta-Score MF: {}\n'.format(fbeta_score(Y_testmf, predmf, beta=2)))

#############################################################################################
#######################################################################################################
#############################################################################################################

# Over-Sampling
over = SMOTEENN(sampling_strategy='auto')
X_over, y_over = over.fit_resample(X_traincc, Y_traincc)

X = X_traincc
y = Y_traincc
cat = ['Sex', 'Count_CA', 'Count_SA', 'Count_OVD', 'Count_CC', 'Count_CL']
cat = ['c' if i in cat else 'q' for i in X_traincc.columns]

# print(X_train.columns)
# print(X_train)
print(pd.value_counts(y))

# tuning
# define model
modelcc = XGBClassifier()
# define grid
weights = [1, 5, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights,
                  max_depth=[6,8,10,12,16]
                  # max_delta_step=[i for i in range(10)]
                  )
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
# scoring = {"AUC": "roc_auc", "f1_macro": "f1_macro", "f1_weighted":"f1_weighted"}
gridcc = GridSearchCV(estimator=modelcc, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
# execute the grid search
grid_result = gridcc.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# print(grid_result.cv_results_)
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print('Best Parameters: ', gridcc.best_params_)

# model.fit(X, y, sample_weight=[99]).summary()
pred = gridcc.predict(X_testcc)
print(roc_auc_score(Y_testcc, pred))
print(pd.value_counts(Y_testcc))
print(confusion_matrix(Y_testcc, pred))
# print(plot_cm(Y_testcc, pred))
# plt.show()
modelcc = XGBClassifier(max_depth=gridcc.best_params_['max_depth'],
                       scale_pos_weight=gridcc.best_params_['scale_pos_weight'])
modelcc.fit(X, y)
predcc = modelcc.predict(X_testcc.to_numpy())
print('predcc: ', roc_auc_score(Y_testcc, predcc))
print(pd.value_counts(Y_testcc))
print(confusion_matrix(Y_testcc, predcc))
print(plot_cm(Y_testcc, predcc, p=0.45))
plt.show()

print('F1-Score CC: {}\n'.format(f1_score(Y_testcc, predcc)))
print('Fbeta-Score CC: {}\n'.format(fbeta_score(Y_testcc, predcc, beta=2)))

#############################################################################################
#######################################################################################################
#############################################################################################################

# Over-Sampling
over = SMOTEENN(sampling_strategy='auto')
X_over, y_over = over.fit_resample(X_traincl, Y_traincl)

X = X_traincl
y = Y_traincl
cat = ['Sex', 'Count_CA', 'Count_SA', 'Count_OVD', 'Count_CC', 'Count_CL']
cat = ['c' if i in cat else 'q' for i in X_traincl.columns]

# print(X_train.columns)
# print(X_train)
print(pd.value_counts(y))

# tuning
# define model
modelcl = XGBClassifier()
# define grid
weights = [1, 5, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights,
                  max_depth=[6,8,10,12,16]
                  # max_delta_step=[i for i in range(10)]
                  )
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
# scoring = {"AUC": "roc_auc", "f1_macro": "f1_macro", "f1_weighted":"f1_weighted"}
gridcl = GridSearchCV(estimator=modelcl, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
# execute the grid search
grid_result = gridcl.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# print(grid_result.cv_results_)
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print('Best Parameters: ', gridcl.best_params_)

# model.fit(X, y, sample_weight=[99]).summary()
predcl = gridcl.predict(X_testcl)
print(roc_auc_score(Y_testcl, predcl))
print(pd.value_counts(Y_testcl))
print(confusion_matrix(Y_testcl, predcl))
print(plot_cm(Y_testcc, pred))
plt.show()
modelcl =XGBClassifier(max_depth=gridcl.best_params_['max_depth'],
                       scale_pos_weight=gridcl.best_params_['scale_pos_weight'])
modelcl.fit(X, y)
predcl = modelcl.predict(X_testcl.to_numpy())
print('predcc: ', roc_auc_score(Y_testcl, predcl))
print(pd.value_counts(Y_testcl))
print(confusion_matrix(Y_testcl, predcl))
print(plot_cm(Y_testcl, predcl, p=0.45))
plt.show()

print('F1-Score CL: {}\n'.format(f1_score(Y_testcl, predcl)))
print('Fbeta-Score CL: {}\n'.format(fbeta_score(Y_testcl, predcl, beta=2)))

#############################################################################################
#######################################################################################################
#############################################################################################################


#set up plotting area
plt.figure(0).clf()

y_predmf = modelmf.predict_proba(X_testmf)[:, 1]
fpr, tpr, _ = roc_curve(Y_testmf, y_predmf)
auc = round(roc_auc_score(Y_testmf, y_predmf), 4)
plt.plot(fpr,tpr,label="Mutual Funds, AUC="+str(auc))

y_predcc = modelcc.predict_proba(X_testcc)[:, 1]
fpr, tpr, _ = roc_curve(Y_testcc, y_predcc)
auc = round(roc_auc_score(Y_testcc, y_predcc), 4)
plt.plot(fpr,tpr,label="Credit Card, AUC="+str(auc))

y_predcl = modelcl.predict_proba(X_testcl)[:, 1]
fpr, tpr, _ = roc_curve(Y_testcl, y_predcl)
auc = round(roc_auc_score(Y_testcl, y_predcl), 4)
plt.plot(fpr,tpr,label="Consumer Loan, AUC="+str(auc))

# y_predno = lr.predict_proba(X_testno)[:, 1]
# fpr, tpr, _ = roc_curve(Y_testno, y_predno)
# auc = round(roc_auc_score(Y_testno, y_predno), 4)
# plt.plot(fpr,tpr,label="Consumer Loan, AUC="+str(auc))

plt.plot([0, 1], [0, 1])

#add legend
plt.legend()
plt.show()

