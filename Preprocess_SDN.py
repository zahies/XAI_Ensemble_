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


TARGET = 'Class'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_sdn():

    if (os.path.isfile("data_SDN/SDN_after_preprocess.csv") == False):
        df = pd.read_csv("data_SDN/SDN_Intrusion.csv")
        df = df.drop("Unnamed: 0", axis=1)
        df.columns = df.columns.str.replace(' ', '_')
        print(df.shape)
        is_attack = df.Class.map(lambda a: 0 if a == 'BENIGN' else 1)

        df['Class'] = is_attack

        print_class_freq(df)

        # print(df.info())
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # print(df.info())

        # print(df.isnull().sum(axis=0))
        df = df.dropna()
        to_fit = df
        # df.groupby(np.isinf(df['data'])).count()
        # print(df.describe())




        del to_fit["Total_Length_of_Fwd_Packets"]
        # del to_fit["_Idle_Min"]
        del to_fit["Bwd_Packet_Length_Max"]
        del to_fit["_ECE_Flag_Count"]
        del to_fit["_CWE_Flag_Count"]
        del to_fit["_URG_Flag_Count"]
        del to_fit["_ACK_Flag_Count"]
        del to_fit["_PSH_Flag_Count"]
        del to_fit["_RST_Flag_Count"]
        del to_fit["_SYN_Flag_Count"]
        del to_fit["FIN_Flag_Count"]
        del to_fit["_Packet_Length_Variance"]
        del to_fit["_Packet_Length_Mean"]
        del to_fit["_Bwd_Packet_Length_Min"]
        del to_fit["_Min_Packet_Length"]

        # del to_fit["_Bwd_Avg_Packets/Bulk"]
        # del to_fit["_Bwd_Avg_Bytes/Bulk"]
        # del to_fit["_Fwd_Avg_Bulk_Rate"]
        # del to_fit["Fwd_Avg_Bytes/Bulk"]
        # del to_fit["_Fwd_Header_Length.1"]
        # del to_fit["_Avg_Bwd_Segment_Size"]
        # del to_fit["_Avg_Fwd_Segment_Size"]
        # del to_fit["_Fwd_Packet_Length_Mean"]
        # del to_fit["_Max_Packet_Length"]
        # del to_fit["_Bwd_Packet_Length_Mean"]
        # del to_fit["_Bwd_URG_Flags"]
        # del to_fit["_Fwd_URG_Flags"]
        # del to_fit["_Bwd_PSH_Flags"]
        # del to_fit["Fwd_PSH_Flags"]
        # del to_fit["_Idle_Min"]
        #
        # del to_fit["_Active_Min"]
        # del to_fit["_Active_Max"]
        # del to_fit["_Active_Std"]
        # del to_fit["_Total_Length_of_Bwd_Packets"]
        # del to_fit["_Bwd_IAT_Mean"]
        # del to_fit["_Fwd_Packet_Length_Min"]
        # del to_fit["_Subflow_Bwd_Bytes"]
        # del to_fit["_Subflow_Bwd_Packets"]
        # del to_fit["Subflow_Fwd_Packets"]
        # del to_fit["Bwd_Avg_Bulk_Rate"]
        # del to_fit["_Fwd_Avg_Packets/Bulk"]
        #
        #
        # del to_fit["_Subflow_Fwd_Bytes"]
        # del to_fit["_Bwd_Packet_Length_Std"]
        # del to_fit["_Fwd_Packet_Length_Max"]


        # del to_fit["_Idle_Max"]
        #
        #
        # del to_fit["_Fwd_IAT_Max"]
        # del to_fit["Fwd_IAT_Total"]
        # # del to_fit["_Idle_Max"]
        # del to_fit["_Total_Fwd_Packets"]
        # del to_fit["_Total_Backward_Packets"]
        # del to_fit["_Fwd_Packet_Length_Std"]
        # del to_fit["_Flow_IAT_Mean"]
        # del to_fit["_Flow_IAT_Std"]
        # del to_fit["_Flow_IAT_Max"]
        # del to_fit["_Flow_IAT_Min"]
        # del to_fit["_Fwd_IAT_Mean"]











        # del to_fit["_Idle_Max"]
        # del to_fit["_Flow_Packets/s"]
        # del to_fit["_Fwd_IAT_Std"]
        # del to_fit["_Fwd_IAT_Min"]
        # del to_fit["_Bwd_IAT_Max"]
        # del to_fit["_Bwd_IAT_Min"]
        # del to_fit["_Fwd_Header_Length"]
        # del to_fit["_Flow_Duration"]
        # del to_fit["Fwd_Packets/s"]
        # del to_fit["Bwd_IAT_Total"]
        # del to_fit["_Bwd_IAT_Std"]
        # del to_fit["_Flow_Packets/s"]
        # del to_fit["Fwd_Packets/s"]
        # del to_fit["Idle_Mean"]
        # del to_fit["_Idle_Std"]




        # to_fit = to_fit[~(to_fit["_Subflow_Fwd_Bytes"] >= 20000)]
        to_fit = to_fit[~(to_fit["_Bwd_Header_Length"] >= 150000)]
        to_fit = to_fit[~(to_fit["Flow_Bytes/s"] >= 135000000)]
        # to_fit = to_fit[~(to_fit["_Bwd_Header_Length"] >= 150000)]
        # to_fit = to_fit[~(to_fit["_Bwd_Header_Length"] >= 150000)]

        # to_fit["Flow_Bytes/s"] = zscore(to_fit["Flow_Bytes/s"])
        # to_fit["Bwd_IAT_Total"] = zscore(to_fit["Bwd_IAT_Total"])
        # to_fit["_Bwd_IAT_Std"] = zscore(to_fit["_Bwd_IAT_Std"])
        # to_fit["Active_Mean"] = zscore(to_fit["Active_Mean"])
        # to_fit["Idle_Mean"] = zscore(to_fit["Idle_Mean"])
        # to_fit["_Idle_Std"] = zscore(to_fit["_Idle_Std"])


        # to_fit["_Flow_Duration"] = zscore(to_fit["_Flow_Duration"])
        # to_fit["Fwd_IAT_Total"] = zscore(to_fit["Fwd_IAT_Total"])
        # to_fit["_Bwd_IAT_Max"] = zscore(to_fit["_Bwd_IAT_Max"])
        # to_fit["_Fwd_IAT_Min"] = zscore(to_fit["_Fwd_IAT_Min"])
        # to_fit["_Bwd_IAT_Min"] = zscore(to_fit["_Bwd_IAT_Min"])



        scaler = preprocessing.RobustScaler()
        to_fit[['Flow_Bytes/s','Bwd_IAT_Total','_Bwd_IAT_Std','Active_Mean','Idle_Mean','_Idle_Std']] = scaler\
            .fit_transform(to_fit[['Flow_Bytes/s','Bwd_IAT_Total','_Bwd_IAT_Std','Active_Mean','Idle_Mean','_Idle_Std']])

        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']] = scaler\
        #     .fit_transform(to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']])
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)


        # y = to_fit["Class"]
        # del to_fit["Class"]
        # to_fit = to_fit.replace(0,-1)
        # to_fit["Class"] = y

        # y = to_fit["Class"]
        # del to_fit["Class"]
        # scaler = preprocessing.MinMaxScaler()
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)
        # to_fit["Class"] = y

        # zeros = (to_fit == 0).astype(int).sum(axis=1)
        # drop_idx = []
        # for idx, value in zeros.items():
        #     if value > 3:
        #         drop_idx.append(idx)
        #
        # to_fit = to_fit.drop(drop_idx)

        # y = to_fit["Class"]
        # del to_fit["Class"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["Class"] = y













        # test = to_fit.loc[~(to_fit == 0).astype(int).sum(axis=1) > 5]
        # df_car = to_fit[~(to_fit == 0).astype(int).sum(axis=1) >= 5]








        to_fit = to_fit.astype('float32')
        print(to_fit.info())
        to_fit.to_csv("data_SDN/SDN_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_SDN/SDN_after_preprocess.csv")
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
    attack_train_df = df_train[df_train[TARGET] == 1].sample(2500, axis=0, random_state=fixed_rnd)
    df2 = df_train[df_train[TARGET] == 0].sample(len(attack_train_df), axis=0, random_state=fixed_rnd)
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


def load_sdn_data():

    df = read_and_preprocess_sdn()
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


