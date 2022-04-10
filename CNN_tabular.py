from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D, MaxPool2D, UpSampling2D,Lambda
from tab2img.converter import Tab2Img
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import math
import random
import pickle
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from Comparison_Detection import RANDOM_SEED
# from Preprocess_twitter import RANDOM_SEED
# from Preprocess_German import RANDOM_SEED




def allUnique(x):
     seen = set()
     return not any(i in seen or seen.add(i) for i in x)

def transform_adv_imgs(x_attack, xg_model):
    pred_adv = xg_model.predict(x_attack)
    # x_attack = numpy.array([np.array(xi) for xi in x_attack])
    # transform adversarial samples to images:
    model = Tab2Img()
    # X = (x_attack.iloc[:len(x_attack),:]).to_numpy().astype(float)
    adv_images = model.fit_transform(x_attack, pred_adv)
    return adv_images


def load_picke(path):
  with open(path, 'rb') as handle:
    b = pickle.load(handle)
  return b

def save_picke(path, to_save):
  with open(path, 'wb') as handle:
    pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def transform_data_to_img(X,y):
    model = Tab2Img()
    print("x type: ", type(X))
    X = (X.iloc[:len(X),:]).to_numpy().astype(float)
    y = y[:len(y)].values
    print("x type: ", type(X))
    print("y type: ", type(y))
    images = model.fit_transform(X, y)
    return images


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def find_idx(image, x, data):
    if data == 'twitter' or data == 'twitter_HATE':
        modulu = 10
    elif data == 'twitter_embd' or data == 'twitter_embd_HATE':
        modulu = 4
    elif data == 'kdd' or data == 'kdd_HATE':
        modulu = 6
    elif data == 'Beth' or data == 'Beth_HATE':
        modulu = 3
    elif data == 'RISK' or data == 'RISK_HATE':
        modulu = 4
    elif data == 'fraud' or data == 'fraud_HATE':
        modulu = 8
    elif data == 'iot' or data == 'iot_HATE':
        modulu = 4
    elif data == 'kdd_test' or data == 'kdd_test_HATE':
        modulu = 5
    elif data == 'lending':
        modulu = 9
    elif data == 'sdn' or data == 'sdn_HATE':
        modulu = 8
    elif data == 'Ddos' or data == 'Ddos_HATE':
        modulu = 6
    elif data == 'lending_NO_SCALE':
        modulu = 9
    elif data == 'car_ins' or data == 'vehicle':
        modulu = 3

    else:
        "ERROR: declare a modulu !"
    idx = []
    # reg_idx = []
    for i in range(x.shape[1]):
        j =0
        for k in range(x.shape[1]):
            if (k % modulu) == 0:
                j+=1
            # if x.iloc[0,i].astype('float32') == image[0][j-1][k%10].astype('float32'):
            #     reg_idx.append(k)
            # if isclose(x.iloc[0,i].astype('float32'), image[0][j-1][k%modulu].astype('float32')):
            if x.iloc[0,i].astype('float32') == image[0][j-1][k%modulu]:
                k_idx = str(j-1) + str(k%modulu)
                k_idx = int(k_idx)
                idx.append(k)
                break
    return idx

def find_idx_row(x_1, x):
    idx = []
    # reg_idx = []
    for i in range(len(x)):
        j =0
        for k in range(len(x)):
            # if isclose(x.iloc[i].astype('float32'), x_1.iloc[k].astype('float32')):
            if x.iloc[i] == x_1.iloc[k]:
                idx.append(k)
                break
    return idx

def plot_image(images,idx, title):
    print("images shape: ", images.shape)
    plt.figure(figsize=(8, 8))
    c = plt.imshow(images[idx], interpolation ='nearest', origin ='lower', vmin = 0, vmax = 0.01)
    plt.colorbar(c)
    plt.title(title)
    plt.show()


def make_unique_row(x, len_features):
    len_unique = 0
    idx = 0
    x_ = make_unique_row_once(x)
    while (len_features != len_unique and idx != 10):
        x_ = make_unique_row_once(x_)
        idxx = find_idx_row(x_, x_)
        len_unique = len(set(idxx))
        idx += 1
    if idx == 10 :
        print("NOT UNIQUE !, len: ", len_unique)
    else:
        print("UNIQUE ROW, len: ", len_unique)
    return x_

def make_unique_row_once(x):
    idx = []
    # x = x.astype('float32')
    for i in range(len(x)):
        if i in idx:
            continue

        add_val = x[i]
        z = random.uniform(0.00001, 0.0001)
        multiply_val = 1
        for k in range(i+1,len(x)):
            if k == i:
                continue
            # if isclose(x[i].astype('float32'), x[k].astype('float32')):
            if x[i] == x[k]:
                x[k] = add_val + z*multiply_val
                idx.append(k)
                multiply_val+=1


    return x



def find_idx_german(image, x):
    idx = []
    for i in range(x.shape[1]):
        j =0
        for k in range(x.shape[1]):
            if (k % 5) == 0:
                j+=1
            # if x.iloc[0,i].astype('float32') == image[0][j-1][k%10]:
            if isclose(x.iloc[0,i].astype('float64'), image[0][j-1][k%5]):
                idx.append(k)
                break
    return idx

def flatten_shap_images(shaps):
    flatten_shap_images = []
    for i in range(len(shaps[0])):
        flat_image = (np.array(shaps[0][i])).reshape(1, -1)
        flatten_shap_images.append(flat_image)
    return flatten_shap_images

def arrange_idx(flatten_shap_images, tab_img_idx, data):
    args_shaps = []
    for i in range(len(flatten_shap_images)):
        arg_shap = []
        if data == 'german':
            for j in range(flatten_shap_images[0].shape[1]-1):
                arg_shap.append(flatten_shap_images[i][0, tab_img_idx[j]])
            args_shaps.append(arg_shap)
        else:
            for j in range(len(tab_img_idx)):
                arg_shap.append(flatten_shap_images[i][0,tab_img_idx[j]])
            args_shaps.append(arg_shap)
    return args_shaps

def arrange_idx_test(flatten_shap_images, tab_img_idx, data):
    args_shaps = []
    for i in range(len(flatten_shap_images)):
        arg_shap = []
        if data == 'german':
            for j in range(flatten_shap_images[0].shape[1]-1):
                arg_shap.append(flatten_shap_images[i][0, tab_img_idx[j]])
            args_shaps.append(arg_shap)
        else:
            for j in range(flatten_shap_images[0].shape[1]):
                arg_shap.append(flatten_shap_images[0,tab_img_idx[j]])
            args_shaps.append(arg_shap)
    return args_shaps

def find_uniq_idx(images, x_train, data):
    if data == 'german':
        check_unique = 23
    elif data == 'kdd':
        check_unique = 34
    elif data == 'Beth' or data == 'Beth_HATE':
        check_unique = 8
    elif data == 'RISK' or data == 'RISK_HATE':
        check_unique = 15
    elif data == 'iot' or data == 'iot_HATE':
        check_unique = 15
    elif data == 'kdd_test' or data == 'kdd_test_HATE':
        check_unique = 24
    elif data == 'twitter_embd' or data == 'twitter_embd_HATE':
        check_unique = 15
    elif data == 'fraud' or data == 'fraud_HATE':
        check_unique = 63
    elif data == 'sdn' or data == 'sdn_HATE':
        check_unique = 63
    elif data == 'Ddos' or data == 'Ddos_HATE':
        check_unique = 35
    elif data == 'lending':
        check_unique = 79
    elif data == 'lending_NO_SCALE':
        check_unique = 79
    elif data == 'car_ins' or data == 'vehicle':
        check_unique = 8
    elif data == 'twitter' or data == 'twitter_HATE':
        check_unique = 99
    idx = []
    for i in range(x_train.shape[0]):
        idx = find_idx(images[i:i + 1], x_train[i:i + 1], data)
        # idx = find_idx_german(images[i:i + 1], x_train[i:i + 1])
        len_unique = len(set(idx))
        if len_unique > check_unique:
            print("LEN UNIQUE: ", len_unique)
            return idx
        print('NOT UNIQUE IDX')
        if allUnique(idx):
            return idx
        if i == x_train.shape[0] - 2:
            break
    return idx



def transform_data_to_naive_img(X):
    rows, cols = 10, 10
    images = []
    for x in X.iterrows():
        image = []
        for i in range(rows):
            col = []
            for j in range(cols):
                col.append(x[1][j + i*10])
            image.append(col)
        image = np.array(image)
        images.append(image)
    images = np.array(images)
    return images


def transform_new_img_to_exist_idx(X_new, idx_tab, data):
    if data == 'german':
        sqrt_image_size = 5
    elif data == 'kdd':
        sqrt_image_size = 6
    elif data == 'sdn' or data == 'sdn_HATE':
        sqrt_image_size = 8
    elif data == 'Beth' or data == 'Beth_HATE':
        sqrt_image_size = 3
    elif data == 'RISK' or data == 'RISK_HATE':
        sqrt_image_size = 4
    elif data == 'twitter_embd' or data == 'twitter_embd_HATE':
        sqrt_image_size = 4
    elif data == 'fraud':
        sqrt_image_size = 8
    elif data == 'kdd_test' or data == 'kdd_test_HATE':
        sqrt_image_size = 5
    elif data == 'iot' or data == 'iot_HATE':
        sqrt_image_size = 4
    elif data == 'Ddos' or data == 'Ddos_HATE':
        sqrt_image_size = 6
    elif data == 'lending':
        sqrt_image_size = 9
    elif data == 'lending_NO_SCALE':
        sqrt_image_size = 9
    elif data == 'car_ins' or data == 'vehicle':
        sqrt_image_size = 3
    elif data == 'twitter' or data == 'twitter_HATE':
        sqrt_image_size = 10
    images = []
    for x in X_new.iterrows():
        w, h = sqrt_image_size, sqrt_image_size
        image = [[0 for z in range(w)] for y in range(h)]
        for idx in range(len(idx_tab)):
            row_idx = int(idx_tab[idx] / sqrt_image_size)
            col_idx = int(idx_tab[idx] % sqrt_image_size)
            image[row_idx][col_idx] = x[1][idx]
        image = np.array(image)
        image = image.astype('float64')
        images.append(image)
    images = np.array(images)
    return images


def extend_dimension_to_RGB(images):
    np_images = np.array(images)
    print("shape images: ",np_images.shape)
    X = np.repeat(np_images[..., np.newaxis], 3, -1)
    return X


# helper function to calculate the prevalence of labels (in order to tackle the imbalance problem of trunk-flex)
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    # number_dim = np.shape(y_true)[1]
    # weights = np.empty([number_dim, 2])
    weights = compute_class_weight(class_weight='balanced', classes=[0.,1.], y=y_true)
    # weights =  dict(zip(np.unique(train_classes), class_weights))
    return weights

# #
# def getPreTrainedModel(data_set):
#     """
#     :param upsample_size: size for umsampling in order to fit the pretrained image sizes
#     :return: Pre Trained model (ResNet)
#     """
#     if data_set == 'german':
#         row_image_size = 5
#     elif data_set == 'kdd':
#         row_image_size = 6
#     elif data_set == 'lending':
#         row_image_size = 9
#     elif data_set == 'sdn':
#         row_image_size = 6
#     elif data_set == 'lending_NO_SCALE':
#         row_image_size = 9
#     elif data_set == 'car_ins' or data_set == 'vehicle':
#         row_image_size = 4
#     else:
#         row_image_size = 10
#     model = keras.Sequential()
#     model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape = (row_image_size, row_image_size, 1)))
#     model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(.25))
#     model.add(BatchNormalization())
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model

#
# def getPreTrainedModel(data_set):
#     """
#     :param upsample_size: size for umsampling in order to fit the pretrained image sizes
#     :return: Pre Trained model (ResNet)
#     """
#     if data_set == 'german':
#         row_image_size = 5
#     elif data_set == 'kdd':
#         row_image_size = 6
#     elif data_set == 'kdd_test':
#         row_image_size = 6
#     elif data_set == 'lending':
#         row_image_size = 9
#     elif data_set == 'RISK':
#         row_image_size = 6
#     elif data_set == 'iot':
#         row_image_size = 4
#     elif data_set == 'sdn':
#         row_image_size = 6
#     elif data_set == 'lending_NO_SCALE':
#         row_image_size = 9
#     elif data_set == 'car_ins':
#         row_image_size = 3
#     else:
#         row_image_size = 10
#     model = keras.Sequential()
#     model.add(Conv2D(32, (2,2), padding='same', activation="relu", input_shape = (row_image_size, row_image_size, 1)))
#     # model.add(MaxPooling2D((2, 2), strides=2))
#
#     model.add(Conv2D(64, (2,2), padding='same', activation="relu"))
#     model.add(Conv2D(128, (2,2), padding='same', activation="relu"))
#     model.add(Conv2D(64, (2,2), padding='same', activation="relu"))
#     model.add(MaxPooling2D((3, 3), strides=2)),
#
#     model.add(Flatten())
#     # model.add(Dense(100, activation="relu"))
#     model.add(Dense(256, activation='relu'))
#     # model.add(Dropout(.5))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(.25))
#     model.add(BatchNormalization())
#
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model


# def getPreTrainedModel():
#     """
#     :param upsample_size: size for umsampling in order to fit the pretrained image sizes
#     :return: Pre Trained model (ResNet)
#     """
#
#     model = keras.Sequential()
#     model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape = (10, 10, 1)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(.25))
#     model.add(BatchNormalization())
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model






def getPreTrainedModel(data_set):
    """
    :param upsample_size: size for umsampling in order to fit the pretrained image sizes
    :return: Pre Trained model (ResNet)
    works for KDD & TWITTER
    """
    if data_set == 'german':
        row_image_size = 5
    elif data_set == 'kdd' or data_set == 'kdd_HATE':
        row_image_size = 6
    elif data_set == 'Beth' or data_set == 'Beth_HATE':
        row_image_size = 3
    elif data_set == 'RISK' or data_set == 'RISK_HATE':
        row_image_size = 4
    elif data_set == 'kdd_test' or data_set == 'kdd_test_HATE':
        row_image_size = 5
    elif data_set == 'iot' or data_set == 'iot_HATE':
        row_image_size = 4
    elif data_set == 'Ddos' or data_set == 'Ddos_HATE':
        row_image_size = 6
    elif data_set == 'twitter_embd' or data_set == 'twitter_embd_HATE':
        row_image_size = 4
    elif data_set == 'sdn' or data_set == 'sdn_HATE':
        row_image_size = 8
    elif data_set == 'fraud':
        row_image_size = 8
    elif data_set == 'lending':
        row_image_size = 9
    elif data_set == 'lending_NO_SCALE':
        row_image_size = 9
    elif data_set == 'car_ins' or data_set == 'vehicle':
        row_image_size = 3
    elif data_set == 'twitter' or data_set == 'twitter_HATE':
        row_image_size = 10
    model = keras.Sequential()
    model.add(Conv2D(64, kernel_size=(2,2), activation='relu', input_shape = (row_image_size, row_image_size, 1)))
    if data_set != 'RISK' and data_set != 'twitter_embd' and data_set != 'iot'  and data_set != 'Beth' and data_set != 'kdd_test' and data_set != 'RISK_HATE' and data_set != 'twitter_embd_HATE' and data_set != 'iot_HATE' and data_set != 'Beth_HATE' and data_set != 'kdd_test_HATE' :
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    # model.add(Dropout(.5))
    # if data_set != 'Beth':
    #     model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    # model.add(Dropout(.5))
    # model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    if data_set == 'twitter' or data_set == 'kdd_test' or data_set == 'sdn'  or data_set == 'twitter_HATE'  or data_set == 'kdd_test_HATE'  or data_set == 'sdn_HATE' :
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.25))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def getDenseModel(data_set):
    # create model
    model = keras.Sequential()
    model.add(Dense(60, activation='relu'))
    model.add(Dense(128, input_dim=60, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, input_dim=128, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_model(model, X_train, y_train, X_test, y_test, X_val, y_val, model_dataset_names, data_path):
    class_weights = calculating_class_weights(y_train)
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    # X_train, y_train, X_test, y_test, X_val, y_val = shuffle_data(X_train, y_train, X_test, y_test, X_val, y_val)
    print("X_train shape: ", X_train.shape)


    early_stopping_monitor = EarlyStopping(monitor='loss', restore_best_weights=True,patience=15)
    if (os.path.isfile(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".h5") == False):
        # fit model
        history = model.fit(X_train, y_train,
                            verbose=1,
                            epochs=80,
                            batch_size=10,
                            validation_data=(X_val, y_val),
                            class_weight=class_weights,
                            callbacks=[early_stopping_monitor])

        model.save(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".h5")  # creates a HDF5 file
    else:
        model = load_model(data_path + "/models/" + model_dataset_names + str(RANDOM_SEED) + ".h5")

    # y_pred = model.predict(X_test)
    # classes = np.argmax(y_pred, axis=1)
    y_pred = model.predict_classes(X_test)
    y_pred_prob = model.predict_proba(X_test)
    # y_pred = indices_to_one_hot(y_pred,3)
    print("y_pred: ",y_pred.reshape(1,-1))
    # print("y_test: ", y_test)
    # print("y_test ",y_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score on Test: ", acc, "seed: ", RANDOM_SEED)
    score_test_roc = roc_auc_score(y_test, y_pred_prob)
    print("ROC-AUC Score on Test: ", score_test_roc, "seed: ", RANDOM_SEED)
    # roc_plot(y_pred_prob, y_test)
    return model


