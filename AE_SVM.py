from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from CNN_tabular import calculating_class_weights
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from Comparison_Detection import RANDOM_SEED
import os
from tensorflow.keras.models import load_model
from sklearn import svm
import numpy as np
import torch

def AE_fit(x_train, x_val, data_path, dataset_name):
    model = keras.Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu', name="layer4"))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(x_train.shape[1], activation='linear'))
    # Compile model
    model.compile(optimizer='adam', loss='mean_absolute_error')

    # class_weights = calculating_class_weights(y_train)
    # class_weights = {0: class_weights[0], 1: class_weights[1]}
    early_stopping_monitor = EarlyStopping(monitor='loss', restore_best_weights=True, patience=7)

    if (os.path.isfile(data_path + "/models/AE_" + dataset_name + str(RANDOM_SEED) + ".h5") == False):
        history_callback = model.fit(x_train, x_train,
                  verbose=1,
                  epochs=60,
                  batch_size=10,
                  validation_data=(x_val, x_val),
                  callbacks=[early_stopping_monitor])

        model.save(data_path + "/models/AE_" + dataset_name + str(RANDOM_SEED) + ".h5")  # creates a HDF5 file
        loss_history = history_callback.history["loss"]
    else:
        model = load_model(data_path + "/models/AE_" + dataset_name + str(RANDOM_SEED) + ".h5")



    return model


def AE_SVM(x_train, x_val, data_path, dataset_name):
    model = AE_fit(x_train, x_val, data_path, dataset_name)

    x_pred = model.predict(x_train)

    all_recon_errors = []
    for i in range(len(x_pred)):
        error = np.mean((x_train[i] - x_pred[i]) * (x_train[i] - x_pred[i]))
        all_recon_errors.append(error)
    all_recon_errors = np.array(all_recon_errors).reshape(-1, 1)

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(all_recon_errors)
    y_pred_train = clf.predict(all_recon_errors)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_right_train = y_pred_train[y_pred_train == 1].size
    acc = n_right_train/(n_error_train + n_right_train)
    print("SVM Error Score on Train: ", n_error_train, "seed: ", RANDOM_SEED)
    print("SVM Accuracy Score on Train: ", acc, "seed: ", RANDOM_SEED)
    print("---------------------------------------------------")
    return model, clf



def AE_SVM_test(x_test, model, clf, type):


    x_pred = model.predict(x_test)

    all_recon_errors = []
    for i in range(len(x_pred)):
        error = np.mean((x_test[i] - x_pred[i]) * (x_test[i] - x_pred[i]))
        all_recon_errors.append(error)
    all_recon_errors = np.array(all_recon_errors).reshape(-1, 1)

    y_pred_train = clf.predict(all_recon_errors)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_right_train = y_pred_train[y_pred_train == 1].size
    acc = n_right_train/(n_error_train + n_right_train)
    outliers_rate = n_error_train/(n_error_train + n_right_train)
    print("SVM sum outliers on ", type, ": ", n_error_train, "seed: ", RANDOM_SEED)
    print("SVM ", type, " Rate as legitimate samples: ", acc, "seed: ", RANDOM_SEED)
    print("SVM ", type, " Rate as adversary samples", outliers_rate, "seed: ", RANDOM_SEED)
    print("---------------------------------------------------")
