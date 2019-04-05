##
## stacking_test.py: Ceiling分類をstackingを使って分類してみる試行
##
##                   Mar. 28. 2019, M. Yamada
##

import numpy as np
import pandas as pd

## Scaler
from sklearn.preprocessing import StandardScaler

## classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

import xgboost as xgb

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import StackingClassifier

from sklearn.ensemble import ExtraTreesClassifier

## cross_val_score
from sklearn.model_selection import cross_val_score

## accurarcy score
from sklearn.metrics import accuracy_score

#=========================
# 前処理残りとデータ準備
#=========================
def process_and_set_data(dat_train, dat_test):
    
    #
    # ラベルと変数に分類、変数は標準化
    #
    label_train = dat_train["CIG_category"]
    label_test = dat_test["CIG_category"]
   
    x_train = dat_train.drop("CIG_category", axis=1)
    x_test = dat_test.drop("CIG_category", axis=1)

    # 余計な列が残っている場合取り除く
    if ('bulletin' in x_train.columns):
        x_train = x_train.drop('bulletin', axis=1)
    if ('station_code' in x_train.columns):
        x_train = x_train.drop('station_code', axis=1)
    if ('cavok' in x_train.columns):
        x_train = x_train.drop('cavok', axis=1)
    if ('bulletin' in x_test.columns):
        x_test = x_test.drop('bulletin', axis=1)
    if ('station_code' in x_test.columns):
        x_test = x_test.drop('station_code', axis=1)
    if ('cavok' in x_test.columns):
        x_test = x_test.drop('cavok', axis=1)

    #
    # NaN除去
    #
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)    

    
    # 標準化
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, label_train, x_test_std, label_test
    

#========
# main
#========
if __name__ == "__main__":

    #--------------------------------
    # データ読み込み(2ヶ月分x2年)
    #--------------------------------
    filename0 = "Input/CIG_RJFK_1112_add9999_SMOTE100_train.csv"
    # filename0 = "Input/CIG_RJFK_1112_add9999_train.csv"
    dat_train = pd.read_csv(filename0)

   
    filename = "Input/CIG_RJFK_1112_add9999_test.csv"
    dat_test = pd.read_csv(filename)

    #
    # アウトプット用の名前を用意する
    #
    filename0 = filename0.split("/")[1]
    pickle_name = filename0[0:30]+'_model.pkl' # SMOTE
    # pickle_name = filename0[0:21]+'_model.pkl'
    print("pickle name:", pickle_name)
    predict_name = filename0[0:30]+'_predict.csv'  # SMOTE
    # predict_name = filename0[0:21]+'_predict.csv'
    
    #-------------------------
    # 前処理残りとデータ準備
    #-------------------------
    x_train_std, label_train, x_test_std, label_test  \
        = process_and_set_data(dat_train, dat_test)

    #----------------------------------
    # 基本分類子とメタ分類子を定義する
    #----------------------------------

    # 基本分類子
    clf_base1 = [
        ExtraTreesClassifier(),
        LogisticRegression(),
        KNeighborsClassifier(),
        xgb.XGBClassifier(),
        MLPClassifier(max_iter = 10000),
        OneVsRestClassifier(SVC(probability=True))
        ]

    clf_base2 = [
        RandomForestClassifier() for _ in range(10)
        ]

    clf_base3 = [
        GaussianNB() for _ in range(10)
        ]

    clf_base4 = [
        xgb.XGBClassifier() for _ in range(10)
        ]

    clf_base5 = [
        ExtraTreesClassifier() for _ in range(10)
        ]

    clf_base6 = [
        ExtraTreesClassifier()
        ] + [
        xgb.XGBClassifier()    
        ] + [
        RandomForestClassifier()
        ]
    
    # メタ分類子
    rf = RandomForestClassifier()
    lg = LogisticRegression()

    #-------------------------------------------
    # スタッキングモデルをトレーニング・テスト
    #-------------------------------------------
    print("clf_base6")
    sclf = StackingClassifier(classifiers = clf_base6
                              , meta_classifier = rf)

    classifiersList = clf_base1+[sclf]
    labelList = ['Extra Trees', 'Logistic', 'KNN'
                 , 'XGboost', 'NN', 'SVM','StackingClassifier']

    for clf, label in zip(classifiersList, labelList):
        scores = cross_val_score(clf, x_train_std, label_train
                                 , cv = 10, scoring = 'accuracy')
        print("Accuracy: {} [{}]".format(scores.mean(), label))

        clf.fit(x_train_std, label_train)
        y_pred = clf.predict(x_test_std)

        acc = accuracy_score(label_test, y_pred)
        print("Test accuracy: {} [{}]".format(acc, label))
        
    #-------------------------
    # ハイパーパラメータ探査
    #-------------------------
    
