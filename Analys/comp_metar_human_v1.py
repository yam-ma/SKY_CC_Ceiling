#=========================================================================
# comp_metar_human: 観測データとAreForecastのCeilingを比較する。
#                   欠測値や天候の分布などを見るために、元データも入れる。
#
#                   Mar. 12. 2019, M. Yamada
#=========================================================================

import numpy as np
import pandas as pd

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

    df_after['CIG_after'] = cig_categorize(cig_list)

    # 列名書き換え
    df_after = df_after.rename(columns = {'forecast_time':'date'})

    # date列を100倍する
    df_after['date'] = df_after['date'].map(lambda x: int(x)*100)
    
    print(df_after)

    #
    # merge METAR df and AFTER df
    #
    df = pd.merge(df_metar, df_after, on = "date", how = "inner")
    print(df)

    df.to_csv('metar_after.csv', index=False)
