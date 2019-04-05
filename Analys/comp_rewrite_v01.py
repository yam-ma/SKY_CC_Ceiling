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
    # 観測が99999で、beforeでも99999予測をしたもの
    #
    df_99999 = df[(df['CIG_before'] == 99999) & (df['CIG_metar'] == 99999)]
    
    #
    # 観測とbeforeが99999のうち、afterが99999でない予想をしたもの
    #
    print(df_99999[df_99999['CIG_after'] != 99999][['CIG_after','voting']])

    print("観測(metar)とbeforeが99999のうち、afterが99999でない予想をしたもの")

    df0 = df_99999[df_99999['CIG_after'] != 99999]
    print(len(df0))

    #
    # そのうち、MLが99999と予測をしたもの(editを減らせる数)
    #
    print("そのうち、MLが99999と予測をしたもの")
    print(len(df0[df0['voting'] == 99999]))
    print("MLが正解で、afterが外した割合(edit削減割合)")
    print(len(df0[df0['voting'] == 99999])/len(df0)*100)

    #
    # 
    #
