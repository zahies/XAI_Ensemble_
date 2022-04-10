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



TARGET = 'LOAN'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_car():
    df_car = pd.read_csv("data_Car_Ins/Car_Insurance_Claim.csv")
    print(df_car.shape)

    df_car["AGE"].replace({"16-25": "Young", "26-39": "Middle Age", "40-64": "Old", "65+": "Very Old"}, inplace=True)
    df_car["DRIVING_EXPERIENCE"].replace({"0-9y": "Newbie", "10-19y": "Amateur", "20-29y": "Advanced", "30y+": "Expert"},
                                       inplace=True)
    df_car.isna().sum()
    feature_cols = ["CREDIT_SCORE", "ANNUAL_MILEAGE"]
    fillna(df_car, feature_cols)
    df_car.rename(columns={'OUTCOME': "LOAN"}, inplace=True)
    df_car = df_car.drop(["ID"], axis=1)
    df_car = df_car.drop_duplicates()

    # remove outliers

    df_car = df_car[~(df_car["ANNUAL_MILEAGE"] >= 18000)]
    df_car = df_car[~(df_car["ANNUAL_MILEAGE"] <= 5000)]
    df_car = df_car[~(df_car["SPEEDING_VIOLATIONS"] >= 15)]
    df_car = df_car[~(df_car["DUIS"] >= 5)]
    df_car = df_car[~(df_car["PAST_ACCIDENTS"] >= 10)]

    data2 = df_car
    subset = ["CREDIT_SCORE", "VEHICLE_OWNERSHIP", "MARRIED", "CHILDREN", "PAST_ACCIDENTS", "SPEEDING_VIOLATIONS",
              "DUIS", "LOAN", "POSTAL_CODE", "ANNUAL_MILEAGE"]
    data2 = data2.drop(subset, axis=1)
    data2 = data2.apply(preprocessing.LabelEncoder().fit_transform)
    df_car = df_car.drop(
        ["AGE", "RACE", "GENDER", "EDUCATION", "DRIVING_EXPERIENCE", "INCOME", "VEHICLE_YEAR", "VEHICLE_TYPE"], axis=1)
    df_car = pd.concat([df_car, data2], axis=1)









    del df_car["RACE"]
    del df_car["VEHICLE_TYPE"]
    del df_car["DUIS"]
    del df_car["EDUCATION"]
    del df_car["CHILDREN"]


    del df_car["VEHICLE_YEAR"]
    del df_car["VEHICLE_OWNERSHIP"]
    del df_car["POSTAL_CODE"]

    # del df_car["RACE"]
    # del df_car["VEHICLE_TYPE"]
    # del df_car["VEHICLE_OWNERSHIP"]


    print(df_car.info())
    print_class_freq(df_car)
    df_car = df_car.astype('float32')
    df_car.to_csv("data_Car_Ins/Car_Insurance_After_Preprocess.csv")
    return df_car

def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df[TARGET] == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df[TARGET] == 0].shape[0])


def fillna(dataframe, feature_cols):
    total_cols = 0
    for y in feature_cols:
        total_cols += 1
        if dataframe[y].isna().sum() > 1:
            try:
                dataframe[y] = dataframe[y].fillna(int(np.mean(dataframe[y])))
            except ValueError:
                pass
        else:
            continue
    print(f"There are {total_cols} columns")

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



    df_attack = df_train.sample(n=num_of_adv, random_state=RANDOM_SEED)
    df_train = df_train.drop(df_attack.index)
    df_attack = pd.concat([df_attack, df_test])
    # print("df_train.shape: ",df_train.shape)
    # print("df_test.shape: ",df_test.shape)

    return df_train, df_val, df_test, df_attack


def load_car_data():

    df = read_and_preprocess_car()
    # ################## Validation Strategy - 80/20 ##################
    # balanced_df = balanced_train_data(df)
    df_train, df_val, df_test, df_attack = train_test_attack_split(df, n_adv)
    y_train = df_train[TARGET]
    y_attack = df_attack[TARGET]
    y_val = df_val[TARGET]
    y_test = df_test[TARGET]
    # remove y and unnecessary columns
    x_train, x_val, x_attack, x_test = [x.drop([TARGET], axis=1) for x in [df_train, df_val, df_attack, df_test]]
    print("x_train.shape: ",x_train.shape)
    print("x_test.shape: ",x_test.shape)
    return x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test


