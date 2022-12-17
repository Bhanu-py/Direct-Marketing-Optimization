import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import *
from sklearn.model_selection import train_test_split
from keras.models import Model
import matplotlib.pyplot as plt
from numpy.random import seed
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

seed(1)
from tensorflow import random
random.set_seed(2)

train_X = pd.read_csv('train_X.csv')
train_Y = pd.read_csv('train_Y.csv')
# print(train_X)
# print(train_Y)
featuresCC = ['Client', 'netCA_flow', 'TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'ActBal_CA', 'ActBal_SA', 'Age', 'Tenure']
featuresMF = ['NetCA_flow', 'TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'Count_MF', 'ActBal_CA', 'ActBal_SA', 'Age', 'Tenure']
featuresCL = ['NetCA_flow', 'TransactionsCred_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDeb_PaymentOrder', 'Count_MF', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'Age', 'Tenure']
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
print('Examples:\n    Total: {}\n    Negative: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, neg, pos, 100 * pos / total))

over = SMOTE(sampling_strategy=0.6)
under = RandomUnderSampler(sampling_strategy=0.7)
X_over, y_over = over.fit_resample(X_train, Y_cc_train)
print('After Over-sampling: \n', pd.value_counts(y_over))
# X_over, y_over = under.fit_resample(X_over, y_over)
# print('After Under-sampling: \n', pd.value_counts(y_over))


checkpoint = keras.callbacks.ModelCheckpoint('model/weighted_model_dummy7.h5', verbose=1, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(
                                    monitor='val_f1_loss',
                                    verbose=1,
                                    patience=100,
                                    # mode='max',
                                    # restore_best_weights=True
)

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1/1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

weighted_model = make_model(X_train.shape[1])

weighted_history = weighted_model.fit(
                            X_over,
                            y_over,
                            batch_size=25,
                            epochs=1000,
                            callbacks=[early_stopping, checkpoint],
                            validation_split=0.25,
                            # The class weights go here
                            class_weight=class_weight)


print(plot_metrics(weighted_history))
plt.show()

train_predictions_weighted = weighted_model.predict(X_train)
test_predictions_weighted = weighted_model.predict(X_test)

weighted_results = weighted_model.evaluate(X_test, Y_cctest, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()

plot_cm(Y_cctest, test_predictions_weighted, p=0.5)
plt.show()

# Plot the ROC
# plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
# plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", Y_cc_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", Y_cctest, test_predictions_weighted, color=colors[1], linestyle='--')

# Plot the AUPRC
plt.legend(loc='lower right')
plt.show()

plot_prc("Train Weighted", Y_cc_train, train_predictions_weighted, color=colors[1])
plot_prc("Test Weighted", Y_cctest, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')
plt.show()

