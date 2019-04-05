#=========================================================================
# comp_metar_human_2month: 観測データとAreForecastのCeilingを比較する。
#                   欠測値や天候の分布などを見るために、元データも入れる。
#                   MLと比較できるように、2ヶ月ごとに集計する。
#
#                   Mar. 19. 2019, M. Yamada
#=========================================================================

import numpy as np
import pandas as pd

## classification report
from sklearn.metrics import classification_report 

## confusion matrix ##
from sklearn.metrics import confusion_matrix

#=========================
# 2ヶ月ごとに切り分ける
#=========================
def cut_2month(df):

    # date列を文字列にする
    df['date'] = df['date'].map(lambda x: str(x))

    
    df_0102 = df[
        (df['date'].str[4:6] == "01") |
        (df['date'].str[4:6] == "02")
    ]

    df_0304 = df[
        (df['date'].str[4:6] == "03") |
        (df['date'].str[4:6] == "04")
    ]

    df_0506 = df[
        (df['date'].str[4:6] == "05") |
        (df['date'].str[4:6] == "06")
    ]

    df_0708 = df[
        (df['date'].str[4:6] == "07") | 
        (df['date'].str[4:6] == "08")
    ]
 
    df_0910 = df[
        (df['date'].str[4:6] == "09") | 
        (df['date'].str[4:6] == "10")
    ]

    df_1112 = df[
        (df['date'].str[4:6] == "11") | 
        (df['date'].str[4:6] == "12")
    ]

    return df_0102, df_0304, df_0506, df_0708, df_0910, df_1112

#=========
# main
#=========
if __name__ == "__main__":

    filelist = ["metar_after.csv", "metar_after_v2.csv"
                , "metar_before.csv", "metar_before_v2.csv"]

    #-------------
    # file loop
    #-------------
    for file in filelist:
    
        #------------
        # read data
        #------------
        df = pd.read_csv(file)

        # 切り分け
        df_0102, df_0304, df_0506, df_0708, df_0910, df_1112 = cut_2month(df)
        
        #---------------------------------------------
        # Confusion matrix & Classification report
        #---------------------------------------------

        print("Processing:") 
        print(file)

        # 0102
        cig_metar = df_0102['CIG_metar']
        cig_human = df_0102.iloc[:, 5]

        print("Confusion matrix: 0102 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 0102 ", file)
        print(classification_report(cig_metar, cig_human))
        
        # 0304
        cig_metar = df_0304['CIG_metar']
        cig_human = df_0304.iloc[:, 5]

        print("Confusion matrix: 0304 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 0304 ", file)
        print(classification_report(cig_metar, cig_human))
              
        # 0506
        cig_metar = df_0506['CIG_metar']
        cig_human = df_0506.iloc[:, 5]

        print("Confusion matrix: 0506 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 0506 ", file)
        print(classification_report(cig_metar, cig_human))

        # 0708
        cig_metar = df_0708['CIG_metar']
        cig_human = df_0708.iloc[:, 5]

        print("Confusion matrix: 0708 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 0708 ", file)
        print(classification_report(cig_metar, cig_human))
              
        # 0910
        cig_metar = df_0910['CIG_metar']
        cig_human = df_0910.iloc[:, 5]

        print("Confusion matrix: 0910 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 0910 ", file)
        print(classification_report(cig_metar, cig_human))

              
        # 1112
        cig_metar = df_1112['CIG_metar']
        cig_human = df_1112.iloc[:, 5]

        print("Confusion matrix: 1112 ", file)
        conf_matrix = confusion_matrix(cig_metar, cig_human)
        print(conf_matrix)

        print("Classification report: 1112 ", file)
        print(classification_report(cig_metar, cig_human))
