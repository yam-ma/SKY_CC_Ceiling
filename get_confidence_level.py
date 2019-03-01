##
## get_confidence_level: 牧野氏定義のConfidence factorを計算する
##

import numpy as np
import pandas as pd


if __name__ == "__main__":

    #
    # データ読み込み
    #
    filename = "CIG_RJFK_0102_predict.csv"
    dat = pd.read_csv(filename)

    print(dat.head())

    # votingの列は落とす
    dat = dat.drop('voting', axis=1)
    
    #
    # クラスごとに確率を出す
    #
    for i in range(len(dat)):
        dat_line = dat.iloc[i, 0:5].tolist()
        # print("dat_line")
        # print(dat_line)
        n_1 = dat_line.count(1.0)/5.0
        n_2 = dat_line.count(2.0)/5.0        
        n_3 = dat_line.count(3.0)/5.0
        n_4 = dat_line.count(4.0)/5.0        
        n_5 = dat_line.count(5.0)/5.0
        n_6 = dat_line.count(6.0)/5.0

        c_level = np.max([n_1, n_2, n_3, n_4, n_5, n_6])
        print("i, c_level", i, c_level)
        # print(n_1, n_2, n_3, n_4, n_5, n_6)
