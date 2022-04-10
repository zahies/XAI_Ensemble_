import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
from Comparison_Detection import RANDOM_SEED
import os
from sklearn import preprocessing
from scipy.stats import zscore


TARGET = 'sus'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_beth():

    if (os.path.isfile("data_beth/Beth_after_preprocess.csv") == False):
        to_fit = pd.read_csv("data_beth/labelled_training_data.csv")
        # df = df.drop("Unnamed: 0", axis=1)
        print(to_fit.shape)


        # print(df.info())

        to_fit["processId"] = to_fit["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        to_fit["parentProcessId"] = to_fit["parentProcessId"].map(
            lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        to_fit["userId"] = to_fit["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
        to_fit["mountNamespace"] = to_fit["mountNamespace"].map(
            lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
        to_fit["eventId"] = to_fit["eventId"]  # Keep eventId values (requires knowing max value)
        to_fit["returnValue"] = to_fit["returnValue"].map(
            lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error



        # to_fit = train_df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
        to_fit['sus'] = to_fit['sus']

        print_class_freq(to_fit)

        to_fit['args'] = to_fit['args'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(to_fit['args'])
        to_fit['args'] = le.transform(to_fit['args'])

        to_fit['eventName'] = to_fit['eventName'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(to_fit['eventName'])
        to_fit['eventName'] = le.transform(to_fit['eventName'])

        to_fit['processName'] = to_fit['processName'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(to_fit['processName'])
        to_fit['processName'] = le.transform(to_fit['processName'])

        to_fit['hostName'] = to_fit['hostName'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(to_fit['hostName'])
        to_fit['hostName'] = le.transform(to_fit['hostName'])

        # del to_fit['mountNamespace']
        # del to_fit['parentProcessId']
        # del to_fit['eventName']
        # del to_fit['threadId']






        del to_fit['timestamp']
        del to_fit['stackAddresses']
        del to_fit['processId']
        del to_fit['processName']
        del to_fit['args']
        del to_fit['evil']
        # del to_fit['returnValue']
        # del to_fit['argsNum']


        print(to_fit.info())

        to_fit = to_fit.astype('float32')
        to_fit.to_csv("data_beth/Beth_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_beth/Beth_after_preprocess.csv")
        to_fit = to_fit.drop("Unnamed: 0", axis=1)
        to_fit = to_fit.astype('float32')
    return to_fit









def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df[TARGET] == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df[TARGET] == 0].shape[0])


def balanced_train_data(df_train):
    """
    sampeling to generate balanced train dataset
    """
    fixed_rnd = 321
    attack_train_df = df_train[df_train[TARGET] == 1]
    df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df)*4, axis=0, random_state=fixed_rnd)
    df_train = pd.concat([attack_train_df, df2],  axis=0, sort=False)
    df_train = df_train.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)
    return df_train

def train_test_attack_split(df, num_of_adv):

    df_train, df_val= train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test= train_test_split(df_val, test_size=0.5, random_state=RANDOM_SEED)



    df_attack = df_train.sample(n=num_of_adv, random_state=RANDOM_SEED)
    df_train = df_train.drop(df_attack.index)
    df_attack = pd.concat([df_attack, df_test])
    # print("df_train.shape: ",df_train.shape)
    # print("df_test.shape: ",df_test.shape)

    return df_train, df_val, df_test, df_attack


def load_beth_data():

    df = read_and_preprocess_beth()
    # ################## Validation Strategy - 80/20 ##################
    balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(balanced_df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    print("x_train.shape: ",x_train.shape)
    print("x_test.shape: ",x_test.shape)
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test
