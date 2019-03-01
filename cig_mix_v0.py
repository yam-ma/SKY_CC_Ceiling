import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

## for SVM ##
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

## for NN ##
from sklearn.neural_network import MLPClassifier

## for Random Forest ##
from sklearn.ensemble import RandomForestClassifier

## for Naive Bayes ##
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

## for xgBoost ##
import xgboost as xgb

if __name__ == "__main__":

    #
    # データ読み込み
    #
    filename = "CIG_RJFK_train.csv"
    dat_train = pd.read_csv(filename)

    print(dat_train.head())
    
    filename = "CIG_RJFK_test.csv"
    dat_test = pd.read_csv(filename)

    print(dat_test.head())

    #================#
    # 前処理残り
    #================#
    
    #
    # ラベルと変数に分類、変数は標準化
    #
    label_train = dat_train["CIG_category"]
    label_test = dat_test["CIG_category"]

    x_train = dat_train.drop("CIG_category", axis=1)
    x_test = dat_test.drop("CIG_category", axis=1)

    print(x_train.columns)
    print(x_test.columns)

    # print(x_test['station_code'])
    # 余計な列が残っている場合取り除く
    if ('bulletin' in x_train.columns):
        x_train = x_train.drop('bulletin', axis=1)
    if ('station_code' in x_train.columns):
        x_train = x_train.drop('station_code', axis=1)
    if ('bulletin' in x_test.columns):
        x_test = x_test.drop('bulletin', axis=1)
    if ('station_code' in x_test.columns):
        x_test = x_test.drop('station_code', axis=1)

    #
    # NaN除去
    #
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)    

    
    # 標準化
    sc = StandardScaler()
    # sc.fit(x_train)
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.fit_transform(x_test)

    
    #================#
    # SVM
    #================#
    # C = 1
    # kernel = 'rbf'
    # gamma = 0.01

    # hyper parameters
    params = [{
        'estimator__kernel': ['rbf', 'linear'],
        'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma':[0.01, 0.001]
        }]
    
    # estimator = SVC(C = C, kernel=kernel, gamma=gamma)
    classifier =  OneVsRestClassifier(SVC())

    clf = GridSearchCV(
        estimator = classifier, param_grid=params, cv=5
    )
    # classifier.fit(x_train_std, label_train)
    print('Here: multi-SVM')
    ## clf.fit(x_train_std, label_train)

    # ベストフィットで予測
    ## label_pred = clf.predict(x_test_std)
    ## label_score = cls.score(x_test_std, label_test)
    
    ## print(label_pred)
    ## print(label_score)
    
    # plt.hist(label_pred)
    # plt.show()

    #================#
    # NN
    #================#
    print('Here: NN')

    params = {
        'solver': ['adam', 'sgd'],
        'hidden_layer_sizes': [10, 50, 100, 200], 
        'alpha': [0.01, 0.001, 0.0001]
    }

    classifier = MLPClassifier(max_iter = 10000)

    #clf = GridSearchCV(
    #    estimator = classifier, param_grid=params, cv=5
    #)

    
    clf = MLPClassifier(solver="sgd", max_iter=10000)
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)

    #================#
    # Random Forest
    #================#
    print("Here: Random Forest")

    params = {
        'n_estimators':[10, 100],   # 決定木の数
        'max_features':[1, 'auto', None],  # 各決定木で分類に使用する説明変数の数
        'max_depth': [1, 5, 10, None],  # 各決定木の深さ
        'min_samples_leaf':[1, 2, 4,]  # 決定木の葉に分類されるサンプル数を決めるパラメータ

    }

    classifier =  RandomForestClassifier(min_samples_leaf=3, random_state=None)

    #clf = GridSearchCV(
    #    estimator = classifier, param_grid=params, cv=5
    #)

    
    clf = RandomForestClassifier(min_samples_leaf=3, random_state=None)
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)
    
    #================#
    # Naive Bayes
    #================#
    print("Here: Naive Bayes: Gaussian")

    clf = GaussianNB()
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)
    
    print("Here: Naive Bayes: Bernoulli")

    params = {
        'alpha': [0.0001, 0.1, 1.0]
    }

    classifier =  BernoulliNB()

    clf = GridSearchCV(
        estimator = classifier, param_grid=params, cv=5
    )
    
    # clf = BernoulliNB()
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)

    # 入力が非0でないといけないとエラーが出るので使わない
    # print("Here: Naive Bayes: Multinomial") 

    # clf = MultinomialNB()
    # clf.fit(x_train_std, label_train)

    # label_pred = clf.predict(x_test_std)
    # label_score = clf.score(x_test_std, label_test)    

    # print(label_pred)
    # print(label_score)
    
    #================#
    # xGBoost
    #================#
    print("Here: xGBoost")

    classifier = xgb.XGBClassifier()

    params = {
        'max_depth':[3, 6, 9],
        'subsample':[0.5, 0.95, 1.0],
        'colsample_bytree':[0.5, 1]
    }

    clf = GridSearchCV(
        estimator = classifier, param_grid=params, cv=5
    )

    # clf = xgb.XGBClassifier()
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)
    
    #================#
    # データコンパイル
    #================#
