import pandas as pd
from model import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

train_X = pd.read_csv('data/train_X.csv')
train_Y = pd.read_csv('data/train_Y.csv')
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

##No Imbalance Handling
#Define model
model_ori=AdaBoostClassifier()
#Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv_ori=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores_ori = cross_validate(model_ori, X_train, Y_cc_train, scoring=scoring, cv=cv_ori, n_jobs=-1)

# summarize performance
print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
predada = AdaBoostClassifier().fit(X_train, Y_cc_train).predict(X_test)
print(confusion_matrix(Y_cctest, predada))
print(plot_cm(Y_cctest, predada))
plt.show()

##Using SMOTE-ENN to balance the data
#Define model
model=AdaBoostClassifier(algorithm="SAMME", n_estimators=200)
#Define SMOTE-ENN
resample=SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'))
#Define pipeline
pipeline = Pipeline(steps=[('r', resample), ('m', model)])
#Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#Evaluate model
scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1']
scores = cross_validate(pipeline, X_train, Y_cc_train, scoring=scoring, cv=cv, n_jobs=-1)

# summarize performance
print('Mean Accuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores['test_recall_macro']))
print('Mean F1: %.4f' % np.mean(scores['test_f1']))
fitpipeline = pipeline.fit(X_train, Y_cc_train)

# print(fitpipeline.feature_importances_())
predadabal = fitpipeline.predict(X_test)
print(confusion_matrix(Y_cctest, predadabal))
print(plot_cm(Y_cctest, predadabal))
plt.show()