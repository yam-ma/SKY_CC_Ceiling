import numpy as np
import pandas as pd

## confusion matrix ##
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":

    #
    # テスト用データを読む
    #
    filename = "../CIG_RJFK_0506_add9999_test.csv"
    df_test = pd.read_csv(filename)

    label_test = list(df_test['CIG_category'])

    #
    # 予測データを読む
    #
    filename = "../conflevel_SMOTE_add9999_0506.csv"
    df_pp = pd.read_csv(filename)
    df_pp.columns = ['predict', '2', '3', '4', '5', '6', '9999', '99999']
    
    label_pp = df_pp['predict']

    #
    # Confusion matrix
    #
    cfm = confusion_matrix(label_test, label_pp)
    print("Confusion matrix")
    print(cfm)
