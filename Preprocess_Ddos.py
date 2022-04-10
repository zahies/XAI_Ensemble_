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


TARGET = 'attack'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_ddos():

    if (os.path.isfile("data_Ddos/DDos_after_preprocess.csv") == False):
        df = pd.read_csv("data_Ddos/DDoSdata.csv", low_memory=False)
        df = df.drop("Unnamed: 0", axis=1)
        print(df.shape)

        # print(df.info())


        le = preprocessing.LabelEncoder()
        le.fit(df['flgs'])
        df['flgs'] = le.transform(df['flgs'])

        le = preprocessing.LabelEncoder()
        le.fit(df['proto'])
        df['proto'] = le.transform(df['proto'])

        le = preprocessing.LabelEncoder()
        le.fit(df['state'])
        df['state'] = le.transform(df['state'])

        le = preprocessing.LabelEncoder()
        le.fit(df['state'])
        df['state'] = le.transform(df['state'])

        df['dport'] = df['dport'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(df['dport'])
        df['dport'] = le.transform(df['dport'])

        df['daddr'] = df['daddr'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(df['daddr'])
        df['daddr'] = le.transform(df['daddr'])

        df['saddr'] = df['saddr'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(df['saddr'])
        df['saddr'] = le.transform(df['saddr'])

        df['sport'] = df['sport'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(df['sport'])
        df['sport'] = le.transform(df['sport'])


        del df['category']
        del df['subcategory']

        print(df.shape)


        print_class_freq(df)

        # print(df.info())
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # print(df.info())

        # print(df.isnull().sum(axis=0))
        df = df.dropna()
        to_fit = df

        # test = to_fit.loc[~(to_fit == 0).astype(int).sum(axis=1) > 5]
        # df_car = to_fit[~(to_fit == 0).astype(int).sum(axis=1) >= 5]

        del to_fit["ltime"]
        del to_fit["seq"]
        del to_fit["dur"]
        del to_fit["mean"]
        del to_fit["stddev"]
        del to_fit["sum"]
        del to_fit["Pkts_P_State_P_Protocol_P_SrcIP"]

        # mm = preprocessing.MinMaxScaler()
        # to_fit[['pkSeqID']] = mm.fit_transform(to_fit[['pkSeqID']])


        # scaler = preprocessing.MinMaxScaler()
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)

        # pkSeqID = to_fit['pkSeqID']
        # y = to_fit["attack"]
        # del to_fit["attack"]
        # scaler = preprocessing.StandardScaler()
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)
        # to_fit["attack"] = y
        # to_fit['pkSeqID'] = pkSeqID

        # y = to_fit["attack"]
        # del to_fit["attack"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["attack"] = y

        # mm = preprocessing.StandardScaler()
        # to_fit[['pkSeqID']] = mm.fit_transform(to_fit[['pkSeqID']])



        to_fit = to_fit.astype('float32')
        # print(to_fit.info())
        to_fit.to_csv("data_Ddos/DDos_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_Ddos/DDos_after_preprocess.csv")
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
    attack_train_df = df_train[df_train[TARGET] == 0]
    df2 = df_train[df_train[TARGET] == 1].sample(len(attack_train_df)*3, axis=0, random_state=fixed_rnd)
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


def load_ddos_data():

    df = read_and_preprocess_ddos()
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
