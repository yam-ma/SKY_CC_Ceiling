##
## comp_rewrite.py: humanとMLでedit数を比較する
##                  
##                  edit数：humanはBEFORE->AFTERの差
##                          その中でMLが正しく当てられた数を計算する
##
##                  Apr. 1. 2019, M. Yamada
##

import numpy as np
import pandas as pd
import glob

from sklearn.metrics import confusion_matrix

#==================================================================
# クラスごとに確率を出して、Confidence factorを多数決で取り出す
#==================================================================
def get_conf_factor(dat):

    nm = 5  # モデルの数
    conf_factors = []
    
    for i in range(len(dat)):
        dat_line = dat.iloc[i, 0:5].tolist()
        # print("dat_line")
        # print(dat_line)
        n_1 = dat_line.count(1.0)/nm
        n_2 = dat_line.count(2.0)/nm        
        n_3 = dat_line.count(3.0)/nm
        n_4 = dat_line.count(4.0)/nm        
        n_5 = dat_line.count(5.0)/nm
        n_6 = dat_line.count(6.0)/nm
        n_9999 = dat_line.count(9999.0)/nm
        n_99999 = dat_line.count(99999.0)/nm

        c_factor = np.max([n_1, n_2, n_3, n_4, n_5, n_6, n_9999, n_99999])

        conf_factors.append(c_factor)

    return conf_factors


#====================
# ML data読み込み
#====================
def ml_data_read():
    flist = glob.glob("../Results/CIG_RJFK_*_add9999_SMOTE100_predict.csv")
    
    print(flist)

    # 初期の空データフレーム
    df_ml = pd.DataFrame(index=[], columns=[])

    for file in flist:
        #
        # 予測値を読み込む
        #
        df = pd.read_csv(file)

        #
        # 対応する日付を読み込む
        #
        file_date = file.replace("Results", "Input")
        file_date = file_date.replace("SMOTE100_predict", "test")
        print("file_date:", file_date)

        df2 = pd.read_csv(file_date)

        #
        # 必要な列を切り取る
        #
        conf_factors = get_conf_factor(df)
        df = pd.DataFrame({'voting': df['voting'], 'conf_factor': conf_factors})
        df2 = df2['date']

        df = pd.concat([df2, df], axis=1)
        # print(df)
        
        # print("df_ml")
        # print(df_ml)
        df_ml = pd.concat([df_ml, df], axis=0)

        # print("df_ml after concat")
        # print(df_ml.reset_index())


    #
    # 日付列でソートする
    #

    df_ml = df_ml.sort_values('date')
    df_ml = df_ml.reset_index(drop = True)
    
    return df_ml

#========
# main
#========

if __name__ == "__main__":

    #
    # human data読み込み
    #
    filename = "metar_all_v2.csv"
    df_human = pd.read_csv(filename)

    #
    # ML data読み込み
    #
    df_ml = ml_data_read()

    print(df_ml)

    #
    # 比較用にデータフレームをmergeする
    #
    df = pd.merge(df_human, df_ml, on = "date", how = "inner")
    print(df)
    print(df.columns)

    #----------------------
    # 99999に着目する
    #----------------------

    #
    # threshold loop
    #
    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:

        print("++++ Confidence factor threshold =", threshold)
    
        #
        # 観測が99999で、beforeでも99999予測をしたもの
        #
        df_99999 = df[(df['CIG_before'] == 99999)
                      & (df['CIG_metar'] == 99999)]
    
        #
        # 観測とbeforeが99999のうち、afterが99999でない予想をしたもの
        #
        print("観測(metar)とbeforeが99999のうち、afterが99999でない予想をしたもの")

        df0 = df_99999[df_99999['CIG_after'] != 99999]
        print(len(df0))

        #
        # そのうち、MLがconfidence factor >= thresholdで
        # 99999と予測をしたもの(editを減らせる数)
        #
        print("そのうち、MLがcf", threshold, "で99999と予測をしたもの")
        a = len(df0[(df0['voting'] == 99999)
                    & (df0['conf_factor'] >= threshold)])
        print(a)
        print("MLがcf",threshold,"で正解で、afterが外した割合(edit削減割合)")
        print(a/len(df0)*100)

        #
        # MLを採用できる数
        #
        print("全予測数(99999)",len(df))
        b = len(df[ df['conf_factor'] >= threshold])
        print("検討しなくていい数(cfがthresholdを上回ったデータの数)", b)
        print("検討しなくていい割合", b/len(df)*100)

        #
        # 見逃しの割合(ML)
        #
        c = len(df[ (df['voting'] == 99999) & (df['conf_factor'] >= threshold) ])
        print("MLが99999と予測した数", c)
        d = len(df[ (df['CIG_metar'] != 99999) &
                    (df['voting'] == 99999) & (df['conf_factor'] >= threshold) ])
        print("MLが99999と予測したうち、実際の観測が99999ではなかった場合の数", d)
        print("比率", d/c*100)

        #
        # 見逃しの割合(human)
        #
        e = len(df[ (df['voting'] == 99999) & 
                    (df['CIG_after'] == 99999) & (df['conf_factor'] >= threshold) ])
        print("MLが99999と予測した数のうち、afterでも99999と予測された数", e)
        f = len(df[ (df['CIG_metar'] != 99999) & (df['voting'] == 99999) & 
                    (df['CIG_after'] == 99999) & (df['conf_factor'] >= threshold) ])
        print("MLとafterで99999と予測された中で、実際の観測が99999ではなかった場合の数", f)
        print("比率", f/e*100)

        #---------------------------------------
        # カテゴリ1&2(0<cig<500)に着目する
        #---------------------------------------

        #
        # MLがカテゴリ1&2と予測した数
        #
        g = len(df[ (df['voting'] == 2) & (df['conf_factor'] >= threshold) ])
        print("MLがカテゴリ1&2と予測した数", g)

        #
        # そのうち実際にmetarがカテゴリ1&2と予測した数
        #
        h = len(df[ (df['voting'] == 2) & (df['conf_factor'] >= threshold)
                    & (df['CIG_metar'] == 2) ])
        print("そのうち実際にmetarがカテゴリ1&2と予測した数", h)

        #
        # afterがカテゴリ1&2と予測した数
        #
        j = len( df[ (df['voting'] == 2) & (df['conf_factor'] >= threshold)
                     & (df['CIG_after'] == 2) ])
        print("afterがカテゴリ1&2と予測した数", j)

        #
        # そのうち実際にmetarがカテゴリ1&2と予測した数
        #
        k = len( df[ (df['voting'] == 2) & (df['conf_factor'] >= threshold)
                     & (df['CIG_after'] == 2) & (df['CIG_metar'] ==2) ])
        print("そのうち実際にmetarがカテゴリ1&2と予測した数", k)
        print(" ")

    #---------------------------------------------------------
    # confidence factorごとに見逃しの数を比較する
    # MLが99999と予測したものを、metarがどう観測したのか
    #---------------------------------------------------------
    df_overview = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 9999, 99999], columns = [])
    
    for threshold in thresholds:

        c1 = len(df[ (df['voting'] == 99999)
                  & (df['conf_factor'] >= threshold)
                  & (df['CIG_metar'] == 1)])

        c2 = len(df[ (df['voting'] == 99999)
                & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 2)])

        c3 = len(df[ (df['voting'] == 99999)
                & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 3)])

        c4 = len(df[ (df['voting'] == 99999)
                & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 4)])

        c5 = len(df[ (df['voting'] == 99999)
                     & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 5)])

        c6 = len(df[ (df['voting'] == 99999)
                     & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 6)])
       
        c9999 = len(df[ (df['voting'] == 99999)
                     & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 9999)])

        c99999 = len(df[ (df['voting'] == 99999)
                     & (df['conf_factor'] >= threshold)
                & (df['CIG_metar'] == 99999)])
        

        c = pd.DataFrame({threshold:[c1, c2, c3, c4, c5, c6, c9999, c99999]}, index=[1, 2, 3, 4, 5, 6, 9999, 99999])
        
        df_overview = pd.concat([df_overview, c], axis=1)

    print("見逃し事例の比較")
    print(df_overview)

    
    #---------------------------
    # 1年間分まとめた混同行列
    #---------------------------
    print("Confusion Matrix")
    for threshold in thresholds:

        df1 = df[ df['conf_factor'] >= threshold ]
        print("Threshold=", threshold)
        print(confusion_matrix(df1['CIG_metar'], df1['voting']))

        print(confusion_matrix(df1['CIG_metar'], df1['CIG_after']))
