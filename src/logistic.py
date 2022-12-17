from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, fbeta_score, make_scorer, accuracy_score, recall_score, roc_curve, roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from sklearn.preprocessing import StandardScaler


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

X_train, X_test, Y_traincc, Y_testcc = train_test_split(train_X[featuresCC].fillna(0),
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

X_train[features_to_normalize] = StandardScaler().fit_transform(X_train[features_to_normalize])
X_test[features_to_normalize] = StandardScaler().fit_transform(X_test[features_to_normalize])


X_trainmf[features_to_normalize] = StandardScaler().fit_transform(X_trainmf[features_to_normalize])
X_testmf[features_to_normalize] = StandardScaler().fit_transform(X_testmf[features_to_normalize])

X_traincl[features_to_normalize] = StandardScaler().fit_transform(X_traincl[features_to_normalize])
X_testcl[features_to_normalize] = StandardScaler().fit_transform(X_testcl[features_to_normalize])

# over = SMOTEENN(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy='auto')
X_over, y_over = under.fit_resample(X_train, Y_traincc)
# X_train = X_over
# Y_traincc = y_over
print(pd.value_counts(y_over))
w = {0: 0.6, 1: 2.5}

#Setting the range for class weights
weights = np.linspace(0.1,0.99,200)

params = {
          'class_weight': [{0: x, 1: 1.0-x} for x in weights],
          }
score = {'AUC': 'roc_auc', 'fbeta':'f1_weighted', 'jacc': 'jaccard', 'f1_mic': 'f1_micro', 'F1': 'f1'}
refit = 'F1'
s = 'f1'

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0)
}
classifier = LogisticRegression(penalty='l2', random_state=3, class_weight=weights)
grid = GridSearchCV(estimator=classifier, param_grid=params, cv=StratifiedKFold(10),
                    n_jobs=1, scoring='roc_auc', verbose=1,  return_train_score=True)
grid.fit(X_train, Y_traincc)
print(grid.best_score_, grid.best_params_)
lr = LogisticRegression(random_state=3, penalty='l2',
                        class_weight=grid.best_params_['class_weight']).fit(X_train, Y_traincc)

problog = lr.predict_proba(X_test)
predlog = np.array([1 if i[1] >= 0.5 else 0 for i in problog])
# print(predlog)
print(confusion_matrix(Y_testcc, predlog))
print(plot_cm(Y_testcc, predlog, p=0.5))
plt.show()
# print(confusion_matrix(Y_testcc, predlog))
print(plot_cm(Y_testcc, predlog, p=0.5))
plt.show()
print('F1-Score: {}\n'.format(f1_score(Y_testcc, predlog)))
print('Fbeta-Score: {}\n'.format(fbeta_score(Y_testcc, predlog, beta=2)))

#############################################################################################
#######################################################################################################
#############################################################################################################

over = SMOTEENN(sampling_strategy=0.5)
# under = RandomUnderSampler(sampling_strategy=0.6)
X_overmf, y_overmf = over.fit_resample(X_trainmf, Y_trainmf)
# X_trainmf = X_overmf
# Y_trainmf = y_overmf
print(pd.value_counts(y_overmf))
weights = np.linspace(0.0,0.99,200)

params = {
          'class_weight': [{0:x, 1:1.0-x} for x in weights]
          }

classifier = LogisticRegression(penalty='l2', random_state=3, solver='newton-cg', class_weight=weights)
grid = GridSearchCV(estimator=classifier, param_grid=params, cv=StratifiedKFold(10),
                    n_jobs=1, scoring=s, verbose=1)
grid.fit(X_trainmf, Y_trainmf)
print(grid.best_score_, grid.best_params_)
lrmf = LogisticRegression(random_state=3, solver='newton-cg', penalty='l2',
                          class_weight=grid.best_params_['class_weight']).fit(X_trainmf, Y_trainmf)

predlogmf = lrmf.predict(X_testmf)
# print(predlog)
print(confusion_matrix(Y_testmf, predlogmf))
print(plot_cm(Y_testmf, predlogmf, p=0.5))
plt.show()
print('F1-Score MF: {}\n'.format(f1_score(Y_testmf, predlogmf)))
print('Fbeta-Score MF: {}\n'.format(fbeta_score(Y_testmf, predlogmf, beta=2)))


#############################################################################################
#######################################################################################################
#############################################################################################################

over = SMOTEENN(sampling_strategy=0.5)
# under = RandomUnderSampler(sampling_strategy=0.6)
X_overcl, y_overcl = over.fit_resample(X_traincl, Y_traincl)
# X_traincl = X_overcl
# Y_traincl = y_overcl
print(pd.value_counts(y_overcl))
weights = np.linspace(0.0,0.99,200)

params = {
          'class_weight': [{0:x, 1:1.0-x} for x in weights]
          }

classifier = LogisticRegression(penalty='l2', random_state=3, solver='newton-cg', class_weight=weights)
grid = GridSearchCV(estimator=classifier, param_grid=params, cv=StratifiedKFold(10),
                    n_jobs=1, scoring=s, verbose=1)
grid.fit(X_traincl, Y_traincl)
print(grid.best_score_, grid.best_params_)
lrcl = LogisticRegression(random_state=3, solver='newton-cg', penalty='l2',
                          class_weight=grid.best_params_['class_weight']).fit(X_traincl, Y_traincl)

predlogcl = lrcl.predict(X_testcl)
# print(predlog)
print(confusion_matrix(Y_testcl, predlogcl))
print(plot_cm(Y_testcl, predlogcl, p=0.5))
plt.show()
print('F1-Score CL: {}\n'.format(f1_score(Y_testcl, predlogcl)))
print('Fbeta-Score CL: {}\n'.format(fbeta_score(Y_testcl, predlogcl, beta=1.5)))


#############################################################################################
#######################################################################################################
############################################################################################################no
features_to_normalize = ['netCA_flow', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'TransactionsDebCash_Card',
                         'VolumeDeb_PaymentOrder', 'ActBal_CA', 'ActBal_SA', 'TransactionsCred_CA', 'TransactionsDeb_PaymentOrder',
                         'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL', 'Age', 'Tenure', 'TransactionsDeb_CA', 'Count_MF']
features_to_normalize = [i for i in features_to_normalize if i in featuresNO]
X_trainno[features_to_normalize] = StandardScaler().fit_transform(X_trainno[features_to_normalize])
X_testno[features_to_normalize] = StandardScaler().fit_transform(X_testno[features_to_normalize])
# print(X_trainno)

# over = SMOTEENN(sampling_strategy=0.6)
# under = RandomUnderSampler(sampling_strategy=0.6)
# X_overno, y_overno = over.fit_resample(X_trainno, Y_trainno)
# X_trainno = X_overno
# Y_trainno = y_overno
print(pd.value_counts(Y_trainno))
weights = np.linspace(0.0,0.99,200)

params = {
          'class_weight': [{0:x, 1:1.0-x} for x in weights]
          }

classifier = LogisticRegression(penalty='l2', random_state=3, solver='newton-cg', class_weight=weights)
grid = GridSearchCV(estimator=classifier, param_grid=params, cv=StratifiedKFold(10),
                    n_jobs=1, scoring=s, verbose=1)
grid.fit(X_trainno, Y_trainno)
print(grid.best_score_, grid.best_params_)
lrno = LogisticRegression(random_state=3, solver='newton-cg', penalty='l2',
                          class_weight=grid.best_params_['class_weight']).fit(X_trainno, Y_trainno)

predlogno = lrno.predict(X_testno)
# print(predlog)
print(confusion_matrix(Y_testno, predlogno))
print(plot_cm(Y_testcl, predlogno, p=0.5))
plt.show()
print('F1-Score No: {}\n'.format(f1_score(Y_testno, predlogno)))
print('Fbeta-Score No: {}\n'.format(fbeta_score(Y_testno, predlogno, beta=2)))

#############################################################################################
#######################################################################################################
#############################################################################################################

#set up plotting area
plt.figure(0).clf()

y_predmf = lrmf.predict_proba(X_testmf)[:, 1]
fpr, tpr, _ = roc_curve(Y_testmf, y_predmf)
auc = round(roc_auc_score(Y_testmf, y_predmf), 4)
plt.plot(fpr,tpr,label="Mutual Funds, AUC="+str(auc))

y_predcc = lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_testcc, y_predcc)
auc = round(roc_auc_score(Y_testcc, y_predcc), 4)
plt.plot(fpr,tpr,label="Credit Card, AUC="+str(auc))

y_predcl = lr.predict_proba(X_testcl)[:, 1]
fpr, tpr, _ = roc_curve(Y_testcl, y_predcl)
auc = round(roc_auc_score(Y_testcl, y_predcl), 4)
plt.plot(fpr,tpr,label="Consumer Loan, AUC="+str(auc))

y_predno = lrno.predict_proba(X_testno)[:, 1]
fpr, tpr, _ = roc_curve(Y_testno, y_predno)
auc = round(roc_auc_score(Y_testno, y_predno), 4)
plt.plot(fpr,tpr,label="No Product, AUC="+str(auc))

plt.plot([0, 1], [0, 1])

#add legend
plt.legend()
plt.show()