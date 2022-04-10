import pandas as pd
import numpy as np
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
from Comparison_Detection import RANDOM_SEED
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from CNN_tabular import calculating_class_weights
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from scipy.stats import zscore




TARGET = 'pred'

n_adv = 5
THRESHOLD = 0.5

data_path = 'data_twitter_embd'
dataset_name = 'twitter_embd'


def read_and_preprocess_twitter():
    df = pd.read_csv("data_twitter/twitter_100_5000.csv")
    df = df.drop("Unnamed: 0", axis=1)

    print(df.shape)

    enc = LabelEncoder()
    enc.fit(df["normal_neigh"])
    df["normal_neigh"] = enc.transform(df["normal_neigh"])
    df["normal_neigh"] = df["normal_neigh"].astype('float')
    enc.fit(df["hate_neigh"])
    df["hate_neigh"] = enc.transform(df["hate_neigh"])
    df["hate_neigh"] = df["hate_neigh"].astype('float')
    df = df.astype('float32')
    # print(df.info(verbose=True))
    print_class_freq(df)

    hate_train_df = df[df.pred == 1]

    # y = df["pred"]
    # del df["pred"]
    # df = df.apply(zscore)
    # df["pred"] = y



    return df

def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df.pred == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df.pred == 0].shape[0])

def balanced_train_data(df_train):
    """
    sampeling to generate balanced train dataset
    """
    hate_train_df = df_train[df_train.pred == 1]
    print("hate_train_df shape:", hate_train_df.shape)
    df2 = df_train[df_train[TARGET] == 0].sample(len(hate_train_df)*5, axis=0, random_state=RANDOM_SEED)
    df_train = pd.concat([hate_train_df, df2],  axis=0, sort=False)
    print("df shape: ", df_train.shape)
    return df_train

def train_test_attack_split(df, num_of_adv):

    df_train, df_val= train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test= train_test_split(df_val, test_size=0.5, random_state=RANDOM_SEED)

    print("df_train.shape: ",df_train.shape)
    print("df_test.shape: ",df_test.shape)


    df_attack = df_train.sample(n=num_of_adv, random_state=RANDOM_SEED)
    df_train = df_train.drop(df_attack.index)
    df_attack = pd.concat([df_attack, df_test])

    return df_train, df_val, df_test, df_attack



def embedding(x_train, x_val):


    model = keras.Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, input_dim=64, activation='relu'))
    model.add(Dense(16, input_dim=128, activation='relu', name="layer4"))
    model.add(Dense(128, input_dim=64, activation='relu'))
    model.add(Dense(x_train.shape[1], activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='mean_absolute_error')

    # class_weights = calculating_class_weights(y_train)
    # class_weights = {0: class_weights[0], 1: class_weights[1]}
    early_stopping_monitor = EarlyStopping(monitor='loss', restore_best_weights=True, patience=7)

    if (os.path.isfile(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5") == False):
        model.fit(x_train, x_train,
                    verbose=1,
                    epochs=80,
                    batch_size=10,
                    validation_data=(x_val, x_val),
                    callbacks=[early_stopping_monitor])

        model.save(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")  # creates a HDF5 file
    else:
        model = load_model(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")

    extract = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    return extract

def load_twitter_data():

    df = read_and_preprocess_twitter()
    # ################## Validation Strategy - 80/20 ##################
    balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(balanced_df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test
    #
    # if os.path.isfile(data_path + "/embd_x_train_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv") == False:
    #     extract = embedding(x_train, x_val)
    #     embd_x_train = extract.predict(x_train)
    #     embd_x_val = extract.predict(x_val)
    #     embd_x_test = extract.predict(x_test)
    #     embd_x_attack = extract.predict(x_attack)
    #     embd_x_train = pd.DataFrame(data=embd_x_train)
    #     embd_x_val = pd.DataFrame(data=embd_x_val)
    #     embd_x_test = pd.DataFrame(data=embd_x_test)
    #     embd_x_attack = pd.DataFrame(data=embd_x_attack)
    #     embd_x_train.to_csv(data_path + "/embd_x_train_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_val.to_csv(data_path + "/embd_x_val_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_test.to_csv(data_path + "/embd_x_test_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_attack.to_csv(data_path + "/embd_x_attack_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     model = load_model(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")
    # else:
    #     embd_x_train = pd.read_csv(data_path + "/embd_x_train_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_train = embd_x_train.drop("Unnamed: 0", axis=1)
    #     embd_x_val = pd.read_csv(data_path + "/embd_x_val_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_val = embd_x_val.drop("Unnamed: 0", axis=1)
    #     embd_x_test = pd.read_csv(data_path + "/embd_x_test_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_test = embd_x_test.drop("Unnamed: 0", axis=1)
    #     embd_x_attack = pd.read_csv(data_path + "/embd_x_attack_" + dataset_name + "_" + str(RANDOM_SEED) + ".csv")
    #     embd_x_attack = embd_x_attack.drop("Unnamed: 0", axis=1)
    #     model = load_model(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")
    #
    #
    # return embd_x_train, embd_x_val, embd_x_attack, embd_x_test, y_train, y_attack, y_val, y_test, model