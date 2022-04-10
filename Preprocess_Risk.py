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


TARGET = 'grade'
# RANDOM_SEED = 654321
n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_risk():

    if (os.path.isfile("data_credit/risk_after_preprocess.csv") == False):
        to_fit = pd.read_csv("data_credit/credit_risk.csv", low_memory=False)
        # df = df.drop("Unnamed: 0", axis=1)
        # df.columns = df.columns.str.replace(' ', '_')
        print(to_fit.shape)
        is_risk = to_fit.grade.map(lambda a: 0 if a == 'A' or a == "B" or a == "C" or a == "D" else 1)

        to_fit['grade'] = is_risk
        del to_fit["sub_grade"]

        print(to_fit.isnull().sum(axis=0))
        # print(to_fit.info())

        # 49
        del to_fit["inq_last_12m"]
        del to_fit["total_cu_tl"]
        del to_fit["inq_fi"]
        del to_fit["all_util"]
        del to_fit["max_bal_bc"]
        del to_fit["open_rv_24m"]
        del to_fit["open_rv_12m"]
        del to_fit["il_util"]
        del to_fit["total_bal_il"]
        del to_fit["mths_since_rcnt_il"]
        del to_fit["open_il_24m"]
        del to_fit["open_il_12m"]
        del to_fit["open_il_6m"]
        del to_fit["open_acc_6m"]
        del to_fit["verification_status_joint"]
        del to_fit["dti_joint"]
        del to_fit["annual_inc_joint"]
        del to_fit["mths_since_last_major_derog"]
        del to_fit["mths_since_last_record"]
        del to_fit["mths_since_last_delinq"]
        del to_fit["desc"]
        del to_fit["funded_amnt"]



        # print(to_fit.info())

        to_fit.replace([np.inf, -np.inf], np.nan, inplace=True)


        to_fit = to_fit.dropna()
        # print(to_fit.info())



        # to_fit['args'] = to_fit['args'].astype(str)
        le = preprocessing.LabelEncoder()
        le.fit(to_fit['application_type'])
        to_fit['application_type'] = le.transform(to_fit['application_type'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['last_credit_pull_d'])
        to_fit['last_credit_pull_d'] = le.transform(to_fit['last_credit_pull_d'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['next_pymnt_d'])
        to_fit['next_pymnt_d'] = le.transform(to_fit['next_pymnt_d'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['last_pymnt_d'])
        to_fit['last_pymnt_d'] = le.transform(to_fit['last_pymnt_d'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['initial_list_status'])
        to_fit['initial_list_status'] = le.transform(to_fit['initial_list_status'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['earliest_cr_line'])
        to_fit['earliest_cr_line'] = le.transform(to_fit['earliest_cr_line'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['addr_state'])
        to_fit['addr_state'] = le.transform(to_fit['addr_state'])

        le = preprocessing.LabelEncoder()
        le.fit(to_fit['zip_code'])
        to_fit['zip_code'] = le.transform(to_fit['zip_code'])


        to_fit[['title','purpose','pymnt_plan','issue_d','verification_status']] = \
            to_fit[['title','purpose','pymnt_plan','issue_d','verification_status']]\
            .apply(LabelEncoder().fit_transform)

        to_fit[['emp_title','emp_length','home_ownership','term']] = \
            to_fit[['emp_title','emp_length','home_ownership','term']]\
            .apply(LabelEncoder().fit_transform)


        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['member_id','tot_cur_bal']] = scaler.fit_transform(to_fit[['member_id','tot_cur_bal']])
        #
        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['id','annual_inc','out_prncp_inv','total_pymnt','total_rec_prncp','total_rec_int']] = scaler.fit_transform\
        #     (to_fit[['id','annual_inc','out_prncp_inv','total_pymnt','total_rec_prncp','total_rec_int']])

        # to_fit["int_rate"] = zscore(to_fit["int_rate"])
        # to_fit["tot_coll_amt"] = zscore(to_fit["tot_coll_amt"])


        to_fit["member_id"] = zscore(to_fit["member_id"])
        to_fit["id"] = zscore(to_fit["id"])
        to_fit["annual_inc"] = zscore(to_fit["annual_inc"])
        to_fit["out_prncp_inv"] = zscore(to_fit["out_prncp_inv"])
        to_fit["total_pymnt"] = zscore(to_fit["total_pymnt"])
        to_fit["total_rec_prncp"] = zscore(to_fit["total_rec_prncp"])
        to_fit["total_rec_int"] = zscore(to_fit["total_rec_int"])
        to_fit["tot_cur_bal"] = zscore(to_fit["tot_cur_bal"])
        to_fit["zip_code"] = zscore(to_fit["zip_code"])
        to_fit["earliest_cr_line"] = zscore(to_fit["earliest_cr_line"])




        # scaler = preprocessing.RobustScaler()
        # to_fit[['Flow_Bytes/s','Bwd_IAT_Total','_Bwd_IAT_Std','Active_Mean','Idle_Mean','_Idle_Std']] = scaler\
        #     .fit_transform(to_fit[['Flow_Bytes/s','Bwd_IAT_Total','_Bwd_IAT_Std','Active_Mean','Idle_Mean','_Idle_Std']])

        # scaler = preprocessing.MinMaxScaler()
        # to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']] = scaler\
        #     .fit_transform(to_fit[['_Flow_Duration', 'Fwd_IAT_Total', '_Bwd_IAT_Max', '_Fwd_IAT_Min', '_Bwd_IAT_Min', '_Flow_IAT_Max']])
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)


        # y = to_fit["Class"]
        # del to_fit["Class"]
        # to_fit = to_fit.replace(0,-1)
        # to_fit["Class"] = y




        # 36
        del to_fit["funded_amnt_inv"]

        # del to_fit["inq_last_6mths"]
        del to_fit["policy_code"]
        del to_fit["collection_recovery_fee"]
        # del to_fit["open_acc"]
        del to_fit["pub_rec"]
        # del to_fit["total_pymnt_inv"]
        del to_fit["recoveries"]
        del to_fit["term"]
        del to_fit["default_ind"]
        # del to_fit["emp_length"]
        del to_fit["out_prncp"]
        # del to_fit["home_ownership"]
        del to_fit["verification_status"]
        del to_fit["pymnt_plan"]
        del to_fit["purpose"]
        del to_fit["title"]

        # 25
        del to_fit["acc_now_delinq"]
        del to_fit["inq_last_6mths"]
        del to_fit["total_rec_late_fee"]
        del to_fit["emp_length"]
        del to_fit["initial_list_status"]
        del to_fit["last_pymnt_d"]
        del to_fit["next_pymnt_d"]
        del to_fit["last_credit_pull_d"]
        del to_fit["collections_12_mths_ex_med"]
        del to_fit["application_type"]
        del to_fit["delinq_2yrs"]
        del to_fit["tot_coll_amt"]

        # 16
        del to_fit["installment"]
        del to_fit["total_pymnt_inv"]
        del to_fit["total_acc"]
        del to_fit["open_acc"]
        del to_fit["total_rev_hi_lim"]
        del to_fit["loan_amnt"]
        del to_fit["revol_bal"]
        del to_fit["emp_title"]
        del to_fit["last_pymnt_amnt"]


        zeros = (to_fit == 0).astype(int).sum(axis=1)
        drop_idx = []
        for idx, value in zeros.items():
            if value > 11:
                drop_idx.append(idx)

        to_fit = to_fit.drop(drop_idx)
        print(to_fit.shape)

        # int_rate = to_fit["int_rate"]
        # y = to_fit["grade"]
        # del to_fit["grade"]
        # scaler = preprocessing.MinMaxScaler(feature_range=(1,100))
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)
        # to_fit["grade"] = y
        # to_fit["int_rate"] = int_rate


        # int_rate = to_fit["int_rate"]
        # y = to_fit["grade"]
        # del to_fit["grade"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["grade"] = y
        # to_fit["int_rate"] = int_rate

        print(to_fit.info())
        print_class_freq(to_fit)
        to_fit = to_fit.dropna()


        to_fit = to_fit.astype('float32')
        print(to_fit.info())
        to_fit.to_csv("data_credit/risk_after_preprocess.csv")
    else:
        to_fit = pd.read_csv("data_credit/risk_after_preprocess.csv")
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
    fixed_rnd = 123
    # test = df_train[df_train[TARGET] == 1]
    attack_train_df = df_train[df_train[TARGET] == 1].sample(3000, axis=0, random_state=fixed_rnd)
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


def load_risk_data():

    df = read_and_preprocess_risk()
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


