import numpy as np
import pandas as pd
import pickle 


if __name__ == "__main__":

    #
    # train用データ(2015, 2016年のデータ)読み込み
    #
    fname = "Intermediate/2015_add9999_all.csv"
    df2015 = pd.read_csv(fname)
    fname = "Intermediate/2016_add9999_all.csv"
    df2016 = pd.read_csv(fname)

    # merge
    df_train = pd.concat([df2015, df2016], axis=0)

    # write out
    df_train.to_csv("CIG_RJFK_train_add9999.csv", index=False)
    with open("CIG_RJFK_train_add9999.pkl", "wb") as f:
        pickle.dump(df_train, f)
    
    
    #
    # test用データ読み込み
    #
    fname = "Intermediate/2017_add9999_all.csv"
    df2017 = pd.read_csv(fname)

    # write out
    df2017.to_csv("CIG_RJFK_test_add9999.csv", index=False)
    with open("CIG_RJFK_test_add9999.pkl", "wb") as f:
        pickle.dump(df2017, f)
