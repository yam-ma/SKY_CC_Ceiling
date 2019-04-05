##
## get_confidence_level: 牧野氏定義のConfidence factorを計算する
##                       出力はConfusion matrixとclassification report
##
##                       Mar. 20. 2019, M. Yamada

import numpy as np
import pandas as pd

## classification report
from sklearn.metrics import classification_report 

## confusion matrix ##
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

#=========
# main
#=========
if __name__ == "__main__":

    #
    # テストデータ読み込み(正解を得る)
    #
    filename = "Input/CIG_RJFK_0102_add9999_test.csv"
    print("filename = ", filename)
    df_test = pd.read_csv(filename)

    # 必要列を切り出し
    df_test = df_test[['CIG_category']]
    
    #
    # 予測データ読み込み
    #
    filename = "Results/CIG_RJFK_0102_add9999_SMOTE200_predict.csv"
    dat = pd.read_csv(filename)

    # votingの列は落として保存
    dat_voting = dat['voting'].tolist()
    dat = dat.drop('voting', axis=1)

    #
    # データ行ごとにConfusion factorを出す
    #
    conf_factors = get_conf_factor(dat)

    #
    # data frameを作る
    #
    
    df_cl = pd.DataFrame({'predict':dat_voting
                          , 'Confidence Factor':conf_factors})

    df = pd.concat([df_test, df_cl], axis=1)

    #
    # Confidence factorの閾値を定めて、閾値以上のデータだけで
    # Confusion matrix/ Classification Reportを出す
    #

    threds = [0.5, 0.7, 0.9]

    for threshold in threds:
        # 閾値以上を取り出す
        print("Threshold = ", threshold)
        df = df.loc[ df['Confidence Factor'] >= threshold, :]

        # confusion matrix
        conf_matrix = confusion_matrix(df['CIG_category'], df['predict'])
        print("Confusion matrix:")
        print(conf_matrix)

        # classification report
        print("Classification report:")
        print(classification_report(df['CIG_category'], df['predict']))
