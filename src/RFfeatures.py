import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


train_X = pd.read_csv('data/train_X.csv').fillna(0)
train_Y = pd.read_csv('data/train_Y.csv').fillna(0)
# print(train_X)
# print(train_Y)
# modeling without Over Sampling
# clinets = train_X['Client']
Y_cc = train_Y['Sale_CC']
Y_mf = train_Y['Sale_MF']
Y_cl = train_Y['Sale_CL']
# print(train_X)

X_traincc, X_testcc, Y_traincc, Y_testcc = train_test_split(train_X,
                                                            Y_cc, test_size=0.2, random_state=20, stratify=Y_cc)
X_trainmf, X_testmf, Y_trainmf, Y_testmf = train_test_split(train_X,
                                                            Y_mf, test_size=0.2, random_state=20, stratify=Y_mf)
X_traincl, X_testcl, Y_traincl, Y_testcl = train_test_split(train_X,
                                                            Y_cl, test_size=0.2, random_state=20, stratify=Y_cl)

selCC = SelectFromModel(RandomForestClassifier(n_estimators=500),threshold=0.02)
selCC.fit(X_traincc, Y_traincc)
selMF = SelectFromModel(RandomForestClassifier(n_estimators=500), threshold=0.02)
selMF.fit(X_trainmf, Y_trainmf)
selCL = SelectFromModel(RandomForestClassifier(n_estimators=500), threshold=0.02)
selCL.fit(X_traincl, Y_traincl)

selected_featCC = X_traincc.columns[(selCC.get_support())]
selected_featMF = X_trainmf.columns[(selMF.get_support())]
selected_featCL = X_traincl.columns[(selCL.get_support())]

print(selected_featCC)
print(selCC.estimator_.feature_importances_.ravel())


plt.title('Feature Importances MF')
f_i = list(zip(selected_featMF, selMF.estimator_.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i], color='midnightblue', align='center')
plt.show()

plt.title('Feature Importances CC')
f_i = list(zip(selected_featCC, selCC.estimator_.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i], color='midnightblue', align='edge')
plt.show()

plt.title('Feature Importances CL')
f_i = list(zip(selected_featCL, selCL.estimator_.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i], color='midnightblue', align='center')
plt.show()

featuresMF = list(selected_featMF)
featuresCC = list(selected_featCC)
featuresCL = list(selected_featCL)



# print(selCC..estimator_.feature_importances_)
# print(pd.Series(selCC.estimator_,selCC.feature_importances_,ravel()).hist())