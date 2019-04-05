#########################################################################
# make_balance_test.py: カテゴリごとのデータ数を揃える実験。
#                      1) 手動で突出したカテゴリを削る、
#                      2) SMOTEを使ったdata augmentation
#                      を試す予定。
#
#                      v01: 手動で突出したカテゴリ(6)を1~5の平均に合わせる
#                      v02: v01を関数化
#                      v03: SMOTEで計算
#########################################################################

import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import random

## SMOTE
from imblearn.over_sampling import SMOTE

## Oversampling
from imblearn.over_sampling import RandomOverSampler

## Undersampling
from imblearn.under_sampling import RandomUnderSampler

#========================================================
# 手動で突出カテゴリを切り詰めたデータフレームを返す
#========================================================
def balance_by_hand(df):

    #
    # カテゴリ別に出現回数を調べる
    #
    category = df['CIG_category'].tolist()
    c = collections.Counter(category)    
    
    # 出現回数の多い順2番目から格納する
    c2 = c.most_common()[1:]
    # 最大値を除いて、残りの平均をとる
    hist_mean = int(np.mean([c2[i][1] for i in range(len(c2))]))
    max_hist_value = c.most_common()[0][0]  # 最大値の出現数

    #
    # 最大値をとるカテゴリーを平均値まで削る
    #
    # print(df[df['CIG_category'] == max_hist_value].sample(hist_mean))
    dat6 = df[df['CIG_category'] == max_hist_value].sample(hist_mean)
    dato = df[df['CIG_category'] != max_hist_value]
    df_balanced = pd.concat([dat6, dato], axis=0)

    
    return df_balanced

#==========
# Main 
#==========

if __name__ == "__main__":

    flist = ["Input/CIG_RJFK_0102_add9999_train.csv"
             , "Input/CIG_RJFK_0304_add9999_train.csv"
             , "Input/CIG_RJFK_0506_add9999_train.csv"
             , "Input/CIG_RJFK_0708_add9999_train.csv"
             , "Input/CIG_RJFK_0910_add9999_train.csv"
             , "Input/CIG_RJFK_1112_add9999_train.csv"]

    #----------------
    # file loop
    #----------------
    for file in flist:
        #
        # データ読み込み
        #
        filename = file
        dat = pd.read_csv(filename)

        #==========================
        # データ数を手で減らす
        #==========================
        dat0 = balance_by_hand(dat)

        print(len(dat), len(dat0))
        c = collections.Counter(dat0['CIG_category'])
        print(c)

        #
        # データ書き出し
        #
        filename = filename.split("/")[1]
        oname = filename[0:22]+"mean"+filename[21:]
        print(oname)
        dat0.to_csv(oname, index=False)

        #==========================
        # SMOTEでバランスする
        #==========================
        #smote = SMOTE(random_state = 71)

        Y = dat["CIG_category"]        
        X = dat.drop(["CIG_category","station_code", "cavok"], axis=1)
       
        columnsX = X.columns
        
        # NaN除去
        X = X.fillna(0)
        Y = Y.fillna(0)

        # 数が少ない場合、ランダムノイズを足したデータを足す
        c = collections.Counter(Y)
        print(dict(c))
        ratio_ros = dict(c)
        for item, cc in c.items():
            print(item, cc)
            # 少ない場合には要素を10個に増やす
            if 0 < cc < 11:
                print("too few")
                print(ratio_ros[item])
                ratio_ros[item] = 10
                print(ratio_ros)

        ros = RandomOverSampler(ratio = ratio_ros)
        X, Y = ros.fit_sample(X, Y)
        #c = collections.Counter(Y)
        #print("over-sampled Counter")
        #print(c)

        
        # SMOTEで水増し
        smote = SMOTE(kind='svm')
        X_resampled, Y_resampled = smote.fit_sample(X, Y)

        c = collections.Counter(Y_resampled)
        print("re-sampled Counter")
        print(c)

        # Undersamplingで全て同じ数に切り揃える
        # ratio = {2:100, 3:100, 4:100, 5:100, 6:100, 9999:100, 99999:100}
        # ratio = {2:200, 3:200, 4:200, 5:200, 6:200, 9999:200, 99999:200}
        ratio = {2:300, 3:300, 4:300, 5:300, 6:300, 9999:300, 99999:300}
        rus = RandomUnderSampler(ratio = ratio)
        X_resampled, Y_resampled = rus.fit_sample(X_resampled, Y_resampled)

        c = collections.Counter(Y_resampled)
        print("under-sampled Counter")
        print(c)


        #
        # データフレームにする
        #
        df_X = pd.DataFrame(X_resampled)
        df_X.columns = columnsX
        df_Y = pd.DataFrame(Y_resampled)
        df_Y.columns = ['CIG_category']
        df = pd.concat([df_X, df_Y], axis=1)
        print(df)

        
        # 書き出し
        oname = filename[0:22]+"SMOTE300"+filename[21:]
        print(oname)
        df.to_csv(oname, index=False)
        
        print(dat.columns)
        print(df.columns)
