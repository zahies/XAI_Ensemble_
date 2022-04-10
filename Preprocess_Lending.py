import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import xgboost
from sklearn.model_selection import train_test_split
import os
from sklearn import preprocessing
from scipy.stats import zscore
from Comparison_Detection import RANDOM_SEED



TARGET = 'loan_status_bin'

n_adv = 5
THRESHOLD = 0.5


def read_and_preprocess_lending():
    file_path_lending = 'data_Lending/accepted_2007_to_2018Q4.csv'

    # df = pd.read_csv("data_twitter/twitter_100_5000.csv")
    # df = df.drop("Unnamed: 0", axis=1)
    # df = df.apply(zscore)

    if (os.path.isfile("data_Lending/Lending_after_preprocess_NO_SCALE.csv") == False):
        df = pd.read_csv(file_path_lending, low_memory=False)
        print(df.shape)
        # df.dropna(inplace=True)
        print(df.shape)
        print(df.info(verbose=True))
        df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]
        print(df.shape)
        print(df.info(verbose=True))

        cols_for_output = ["installment", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
                           "recoveries", "collection_recovery_fee"]

        cols_to_drop = ["id", "member_id", "funded_amnt", "funded_amnt_inv", "int_rate", "grade", "sub_grade",
                        "emp_title", "pymnt_plan", "url", "desc", "title", "zip_code", "addr_state",
                        "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
                        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
                        "last_fico_range_low", "policy_code", "hardship_flag", "hardship_type", "hardship_reason",
                        "hardship_status", "deferral_term", "hardship_amount", "hardship_start_date",
                        "hardship_end_date", "payment_plan_start_date", "hardship_length", "hardship_dpd",
                        "hardship_loan_status", "orig_projected_additional_accrued_interest",
                        "hardship_payoff_balance_amount", "hardship_last_payment_amount", "disbursement_method",
                        "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status", "settlement_date",
                        "settlement_amount", "settlement_percentage", "settlement_term"]

        df = df.drop(columns=cols_to_drop)
        df = df.drop(columns=cols_for_output)

        df['loan_status_bin'] = df['loan_status'].map({'Charged Off': 1, 'Fully Paid': 0})

        df['emp_length_num'] = df['emp_length'].apply(emp_to_num)
        df['long_emp'] = df['emp_length'].apply(lambda x: 1 * (x == '10+ years'))
        df['short_emp'] = df['emp_length'].apply(lambda x: 1 * (x == '1 year' or x == '< 1 year'))

        df["term"] = df["term"].map(lambda term_str: term_str.strip())

        extract_num = lambda term_str: float(term_str[:2])
        df["term_num"] = df["term"].map(extract_num)
        del df['term']
        df["issue_d"] = df["issue_d"].astype("datetime64[ns]")

        # categorical_cols = ["home_ownership", "verification_status", "purpose",
        #                     "verification_status_joint"]

        onehot_cols = ["home_ownership", "purpose", "application_type"]
        for i, col_name in enumerate(onehot_cols):
            print(
                df.groupby(col_name)[col_name].count(),
                "\n" if i < len(onehot_cols) - 1 else "",
            )

        for col in onehot_cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])




        cr_hist_age_months = get_credit_history_age(df, "earliest_cr_line")
        df["earliest_cr_line"] = cr_hist_age_months
        df["sec_app_earliest_cr_line"] = get_credit_history_age(df, "sec_app_earliest_cr_line").astype("Int64")
        del df['emp_length']
        # sanity check
        # df.head()

        # Fill joint columns in individual applications
        for joint_col, indiv_col in zip(
                ["annual_inc_joint", "dti_joint", "verification_status_joint"],
                ["annual_inc", "dti", "verification_status"],
        ):
            df[joint_col] = [
                joint_val if app_type == "Joint App" else indiv_val
                for app_type, joint_val, indiv_val in zip(
                    df["application_type"], df[joint_col], df[indiv_col]
                )
            ]

        def replace_list_value(l, old_value, new_value):
            i = l.index(old_value)
            l.pop(i)
            l.insert(i, new_value)

        new_metric_cols = ["open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
                           "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
                           "inq_fi", "total_cu_tl", "inq_last_12m"]

        joint_cols = ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
                      "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
                      "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
                      "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
                      "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog"]

        joint_new_metric_cols = ["revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
                                 "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
                                 "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
                                 "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
                                 "sec_app_collections_12_mths_ex_med", "sec_app_inv_mths_since_last_major_derog"]

        mths_since_last_cols = [
            col_name
            for col_name in df.columns
            if "mths_since" in col_name or "mo_sin_rcnt" in col_name
        ]
        mths_since_old_cols = [
            col_name for col_name in df.columns if "mo_sin_old" in col_name
        ]

        for col_name in mths_since_last_cols:
            df[col_name] = [
                0.0 if pd.isna(months) else 1 / 1 if months == 0 else 1 / months
                for months in df[col_name]
            ]
        df.loc[:, mths_since_old_cols].fillna(0, inplace=True)

        # Rename inverse columns
        rename_mapper = {}
        for col_name in mths_since_last_cols:
            rename_mapper[col_name] = col_name.replace("mths_since", "inv_mths_since").replace(
                "mo_sin_rcnt", "inv_mo_sin_rcnt"
            )
        df.rename(columns=rename_mapper, inplace=True)

        replace_list_value(new_metric_cols, "mths_since_rcnt_il", "inv_mths_since_rcnt_il")
        replace_list_value(
            joint_cols,
            "sec_app_mths_since_last_major_derog",
            "sec_app_inv_mths_since_last_major_derog",
        )

        cols_to_search = [
            col for col in df.columns if col not in new_metric_cols + joint_new_metric_cols
        ]
        df.dropna(axis="index", subset=cols_to_search, inplace=True)

        df.drop(
            columns=[
                "verification_status",
                "verification_status_joint",
                "loan_status",
                "issue_d",
            ],
            inplace=True,
        )

        loans_1 = df.drop(columns=new_metric_cols + joint_new_metric_cols)
        loans_2 = df.drop(columns=joint_new_metric_cols)
        loans_2.info(verbose=True, null_counts=True)

        df["il_util_imputed"] = [
            True if pd.isna(util) & pd.notna(bal) & pd.notna(limit) else False
            for util, bal, limit in zip(
                df["il_util"], df["total_bal_il"], df["total_il_high_credit_limit"]
            )
        ]
        new_metric_onehot_cols = ["il_util_imputed"]
        df["il_util"] = [
            0.0
            if pd.isna(util) & pd.notna(bal) & (limit == 0)
            else float(round(bal / limit * 100))
            if pd.isna(util) & pd.notna(bal) & pd.notna(limit)
            else util
            for util, bal, limit in zip(
                df["il_util"], df["total_bal_il"], df["total_il_high_credit_limit"]
            )
        ]

        loans_2 = df.drop(columns=joint_new_metric_cols)
        loans_2.info(verbose=True, null_counts=True)


        print(df.info(verbose=True))

        loans_2.dropna(axis="index", inplace=True)

        loans_3 = df.dropna(axis="index")





        print_class_freq(loans_3)
        fixed_rnd = 123
        charge_off = loans_3[loans_3[TARGET] == 1]
        df2 = loans_3[loans_3[TARGET] == 0].sample(len(charge_off), axis=0, random_state=fixed_rnd)
        to_fit = pd.concat([charge_off, df2], axis=0, sort=False)
        to_fit = to_fit.sample(frac=1, random_state=fixed_rnd).reset_index(drop=True)
        print(to_fit.shape)


        """ del less important features """


        del to_fit['sec_app_fico_range_high']
        del to_fit['dti_joint']
        del to_fit['pub_rec']
        del to_fit['collections_12_mths_ex_med']
        del to_fit['chargeoff_within_12_mths']
        del to_fit["fico_range_high"]
        del to_fit["acc_now_delinq"]
        del to_fit["delinq_amnt"]
        del to_fit["application_type"]
        del to_fit["long_emp"]
        del to_fit["num_tl_30dpd"]
        del to_fit["num_tl_90g_dpd_24m"]
        del to_fit["num_tl_120dpd_2m"]
        del to_fit["annual_inc_joint"]

        to_fit = to_fit.astype('float32')


        print('-----------to_fit---------')
        to_fit.info(verbose=True)

        # scaler = preprocessing.MinMaxScaler()
        # scaled_df = scaler.fit_transform(to_fit)
        # to_fit = pd.DataFrame(scaled_df, columns=to_fit.columns)

        # y = to_fit["loan_status_bin"]
        # del to_fit["loan_status_bin"]
        # to_fit = to_fit.apply(zscore)
        # to_fit["loan_status_bin"] = y

        to_fit.to_csv("data_Lending/Lending_after_preprocess_NO_SCALE.csv")
    else:
        to_fit = pd.read_csv("data_Lending/Lending_after_preprocess_NO_SCALE.csv")
        to_fit = to_fit.drop("Unnamed: 0", axis=1)
        to_fit = to_fit.astype('float32')
    return to_fit

def print_class_freq(df):
    print("Class 1 frequency: ")
    print(df[df[TARGET] == 1].shape[0])
    print("Class 0 frequency: ")
    print(df[df[TARGET] == 0].shape[0])


def get_credit_history_age(df, col_name):
    earliest_cr_line_date = df[col_name].astype("datetime64[ns]")
    cr_hist_age_delta = df["issue_d"] - earliest_cr_line_date
    MINUTES_PER_MONTH = int(365.25 / 12 * 24 * 60)
    cr_hist_age_months = cr_hist_age_delta / np.timedelta64(MINUTES_PER_MONTH, "m")
    return cr_hist_age_months.map(
        lambda value: np.nan if pd.isna(value) else round(value)
    )


def emp_to_num(term):
    if pd.isna(term):
        return None
    elif term[2] == '+':
        return 10
    elif term[0] == '<':
        return 0
    else:
        return int(term[0])

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


    return df_train, df_val, df_test, df_attack


def load_lending_data():

    df = read_and_preprocess_lending()
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

