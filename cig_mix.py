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

## for Voting ##
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV

## classification report
from sklearn.metrics import classification_report 

## confusion matrix ##
from sklearn.metrics import confusion_matrix


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
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    
    #================#
    # SVM
    #================#
    print('Here: multi-SVM')

    svm = OneVsRestClassifier(SVC(probability=True))
    
    svm_params = {
        'estimator__kernel': ['rbf', 'linear'],
        'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma':[0.01, 0.001]
        }
    
    # estimator = SVC(C = C, kernel=kernel, gamma=gamma)
    classifier =  OneVsRestClassifier(SVC())

    clf = GridSearchCV(
        estimator = svm, param_grid=svm_params, cv=5
    )
    # classifier.fit(x_train_std, label_train)

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

    nn = MLPClassifier(max_iter = 10000)    
    
    nn_params = {
        'solver': ['adam', 'sgd'],
        'hidden_layer_sizes': [10, 50, 100, 200], 
        'alpha': [0.01, 0.001, 0.0001]
    }

    #classifier = MLPClassifier(max_iter = 10000)

    #clf = GridSearchCV(
    #    estimator = nn, param_grid=nn_params, cv=5
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

    rf =  RandomForestClassifier(min_samples_leaf=3, random_state=None)
    
    rf_params = {
        'n_estimators':[10, 100],   # 決定木の数
        'max_features':[1, 'auto', None],  # 各決定木で分類に使用する説明変数の数
        'max_depth': [1, 5, 10, None],  # 各決定木の深さ
        'min_samples_leaf':[1, 2, 4,]  # 決定木の葉に分類されるサンプル数を決めるパラメータ

    }

    #classifier =  RandomForestClassifier(min_samples_leaf=3, random_state=None)

    #clf = GridSearchCV(
    #    estimator = rf, param_grid=rf_params, cv=5
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

    gnb = GaussianNB()
    gnb.fit(x_train_std, label_train)

    label_pred = gnb.predict(x_test_std)
    label_score = gnb.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)
    
    print("Here: Naive Bayes: Bernoulli")

    bnb =  BernoulliNB()
    
    bnb_params = {
        'alpha': [0.0001, 0.1, 1.0]
    }

    # classifier =  BernoulliNB()

    clf = GridSearchCV(
        estimator = bnb, param_grid=bnb_params, cv=5
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

    # classifier = xgb.XGBClassifier()
    xgbc = xgb.XGBClassifier()

    xgbc_params = {
        'max_depth':[3, 6, 9],
        'subsample':[0.5, 0.95, 1.0],
        'colsample_bytree':[0.5, 1]
    }

    # clf = GridSearchCV(
    #     estimator = xgbc, param_grid=xgbc_params, cv=5
    # )

    clf = xgb.XGBClassifier()
    clf.fit(x_train_std, label_train)

    label_pred = clf.predict(x_test_std)
    label_score = clf.score(x_test_std, label_test)    

    print(label_pred)
    print(label_score)
    
    #================#
    # モデルコンパイル
    #================#

    # パラメータセット辞書を作る
    params = {}
    params.update({"svm__"+k: v for k, v in svm_params.items()})
    params.update({"nn__"+k: v for k, v in nn_params.items()})
    params.update({"rf__"+k: v for k, v in rf_params.items()})
    params.update({"bnb__"+k: v for k, v in bnb_params.items()})
    params.update({"xgbc__"+k: v for k, v in xgbc_params.items()})

    eclf = VotingClassifier(estimators=[("svm", svm), 
                                        ("nn", nn),
                                        ("rf", rf),
                                        ("bnb", bnb),
                                        ("xgbc", xgbc)],
                            voting = "soft")

    # 混成estimatorsからベストフィットモデルを探査する
    clf = RandomizedSearchCV(eclf, param_distributions=params, cv=5,
                             n_iter=100, n_jobs=-1, verbose=3)


    clf.fit(x_train_std, label_train)

    #
    # predict using test data
    #
    predict = clf.predict(x_test_std)

    conf_matrix = confusion_matrix(label_test, predict
                                   , labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print("Confusion matrix")
    print(conf_matrix)

    print("Classification report:")
    print(classification_report(label_test, predict))

    # write-out best fit paramneters
    print("best-fit parameters:")
    print(clf.best_params_)

    #
    # write out confidence level
    #
    pp = clf.predict_proba(x_test_std) # ndarray
    print("pp")
    print(pp)
    pp_df = pd.DataFrame(pp)
    predict_df = pd.DataFrame(predict)
    pp_df0 = pd.concat([predict_df, pp_df], axis=1)

    pp_df0.to_csv("conflevel_all.csv", index=False)
    
