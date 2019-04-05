#=========================================================================
# comp_metar_human: 観測データとAreForecastのCeilingを比較する。
#                   欠測値や天候の分布などを見るために、元データも入れる。
#
#                   Mar. 12. 2019, M. Yamada
#=========================================================================

import numpy as np
import pandas as pd

## classification report
from sklearn.metrics import classification_report 

## confusion matrix ##
from sklearn.metrics import confusion_matrix


#-----------------------------
# (metar・ML用)CIGカテゴリ分け
#-----------------------------
def cig_categorize(cig_list):

    cc_list = []
    for c in cig_list:
        if c < -10000:  # BKNもOVCも値がない場合
            cat = 99999
        elif c < 0:     # BKNはあるが、計測不能(-9999が入っていた場合)
            cat = 9999 
        elif 0 <= c < 200:
            cat = 1
        elif 200 <= c < 500:
            cat = 2
        elif 500 <= c < 1000:
            cat = 3
        elif 1000 <= c < 2000:
            cat = 4
        elif 2000 <= c < 4000:
            cat = 5
        else:    # 4000ft以上
            cat = 6 

        cc_list.append(cat)
    
    return cc_list


#-----------------------------
# AFTER用CIGカテゴリ分け
#-----------------------------
def cig_categorize_after(cig_list):

    cc_list = []
    for c in cig_list:
        if c >= 30479.695:  # CIG=30479.695の時
            cat = 99999
        elif c < 0:     # BKNはあるが、計測不能(-9999が入っていた場合)
            cat = 9999 
        elif 0 <= c < 200:
            cat = 1
        elif 200 <= c < 500:
            cat = 2
        elif 500 <= c < 1000:
            cat = 3
        elif 1000 <= c < 2000:
            cat = 4
        elif 2000 <= c < 4000:
            cat = 5
        else:    # 4000ft以上
            cat = 6 

        cc_list.append(cat)
    
    return cc_list

#-----------------------------------
# AFTERデータに適切なヘッダをつける
#-----------------------------------
def after_add_header(df):

    with open("../AFTER_0600/header.txt") as f:
        s = f.read()
        print(s.split(" "))

    s = s.replace(",", "")
    df.columns = s.split(" ")
    
    return df


if __name__ == "__main__":

    #
    # read METAR data
    #
    filename = "../df_2017.csv"
    df_metar = pd.read_csv(filename)

    cig_list = list(df_metar['ceiling'])
    
    df_metar['CIG_metar'] = cig_categorize(cig_list)

    df_metar.columns = ['date', 'str_cloud', 'ceiling_metar', 'CIG_metar']
    print(df_metar)

    #
    # read AFTER data
    #
    filename = "../AFTER_0600/RJFK.csv"
    df_after = pd.read_csv(filename)

    # ヘッダをつける
    df_after = after_add_header(df_after)

    df_after = df_after.loc[:, ['forecast_time', 'CLING']]

    cig_list = list(df_after['CLING'])

    # 2パターン用意する
    df_after2 = df_after.copy()  

    # ML用分類と同じ
    df_after['CIG_after'] = cig_categorize(cig_list)
    # CIG=30479.695は晴れ扱いにする
    df_after2['CIG_after'] = cig_categorize_after(cig_list)

    # 列名書き換え
    df_after = df_after.rename(columns = {'forecast_time':'date'})
    df_after2 = df_after2.rename(columns = {'forecast_time':'date'})

    # date列を100倍する
    df_after['date'] = df_after['date'].map(lambda x: int(x)*100)
    df_after2['date'] = df_after2['date'].map(lambda x: int(x)*100)
    
    print(df_after)

    #
    # merge METAR df and AFTER df
    #
    df = pd.merge(df_metar, df_after, on = "date", how = "inner")
    print(df)

    df2 = pd.merge(df_metar, df_after2, on = "date", how = "inner")
    # print(df)

    # write out data
    df.to_csv('metar_after.csv', index=False)
    df2.to_csv('metar_after_v2.csv', index=False)

    #
    # Confusion matrix & Classification report
    #
    print("Confusion matrix for v1")
    conf_matrix = confusion_matrix(df['CIG_metar'], df['CIG_after'])
    print(conf_matrix)

    print("Classification report for v1")
    print(classification_report(df['CIG_metar'], df['CIG_after']))

    print("Confusion matrix for v2")
    conf_matrix = confusion_matrix(df2['CIG_metar'], df2['CIG_after'])
    print(conf_matrix)

    print("Classification report for v2")
    print(classification_report(df2['CIG_metar'], df2['CIG_after']))
