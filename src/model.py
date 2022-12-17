import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential  # importing Sequential model
from keras.layers import Dense  # importing Dense layers
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras import metrics
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from numpy.random import seed
from keras import regularizers
from tensorflow import keras

seed(1)
from tensorflow import random
from sklearn.metrics import fbeta_score

random.set_seed(2)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def eager_binary_fbeta(ytrue, ypred, beta=2.0, threshold=0.5):
    ypred = np.array(1 if x[0] >= threshold else 0 for x in list(ypred))
    return fbeta_score(ytrue, ypred, average='binary', beta=beta)


def f1_weighted(true, pred):  # shapes (batch, 4)

    # for metrics include these two lines, for loss, don't include them
    # these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 4)

    ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
    # all with shape (4,)

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1  # for metrics, return only 'weighted_f1'


Metrics = [
    f1_loss,
    f1_weighted,
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    tfa.losses.WeightedKappaLoss(num_classes=1, name='cohen_kappa'),
    tfa.losses.SigmoidFocalCrossEntropy(name='sfc')
]


def prop_model(input_shape):
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def multi_label(input_shape, metrics=None):
    # create model
    if metrics is None:
        metrics = Metrics
    model = Sequential()
    model.add(Dense(12, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=Metrics)

    return model


seed(1)  # for reproducibility


def make_model(input_shape, metrics=Metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential(
        [keras.layers.Dense(12, activation='relu', input_shape=(input_shape,), kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(0.0001)),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(15, activation='relu'),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(8, activation='relu'),
         keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
         ])
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=10000,
        decay_rate=0.9)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100.5])
    plt.ylim([35, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
