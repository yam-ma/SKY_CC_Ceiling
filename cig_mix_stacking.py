##
## cig_mix_stacking.py: Ceiling分類をstackingを使って分類してみる
##                      分類器は、votingの結果と比較するために、
##                      SVM、NN、RF, NaiveBayes, XGBoostとしてみる。
##                      探査させるハイパーパラメータもvotingと同様にする。
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

## grid search ##
from sklearn.model_selection import GridSearchCV

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
    filename0 = "Input/CIG_RJFK_0102_add9999_SMOTE100_train.csv"
    # filename0 = "Input/CIG_RJFK_0102_add9999_train.csv"
    dat_train = pd.read_csv(filename0)

   
    filename = "Input/CIG_RJFK_0102_add9999_test.csv"
    dat_test = pd.read_csv(filename)

    #
    # アウトプット用の名前を用意する
    #
    filename0 = filename0.split("/")[1]
    pickle_name = filename0[0:30]+'_stacking_model.pkl' # SMOTE
    # pickle_name = filename0[0:21]+'_stacking_model.pkl'
    print("pickle name:", pickle_name)
    predict_name = filename0[0:30]+'_stacking_predict.csv'  # SMOTE
    # predict_name = filename0[0:21]+'_stacking_predict.csv'
    
    #-------------------------
    # 前処理残りとデータ準備
    #-------------------------
    x_train_std, label_train, x_test_std, label_test  \
        = process_and_set_data(dat_train, dat_test)

    #----------------------------------
    # 基本分類子とメタ分類子を定義する
    #----------------------------------

    # 基本分類子
    svm = OneVsRestClassifier(SVC(probability=True))
    nn = MLPClassifier(max_iter = 10000)
    extra = ExtraTreesClassifier()
    rf0 = RandomForestClassifier()
    xgbc = xgb.XGBClassifier()    
    bnb = BernoulliNB()
    
    clf_base = [
        svm,
        nn,
        bnb, 
        extra
        ] + [
        xgbc
        ] + [
        rf0
        ] 
    
    # メタ分類子
    rf = RandomForestClassifier()
    lr = LogisticRegression()

    #-------------------------------------------
    # スタッキングモデルを定義
    #-------------------------------------------
    print("clf_base")
    sclf = StackingClassifier(classifiers = clf_base
                              , meta_classifier = rf)
       
    #-------------------------
    # ハイパーパラメータ定義
    #-------------------------
    params = {
        #
        # Multi-class SVM
        #
        'onevsrestclassifier__estimator__kernel':['rbf', 'linear'],
        'onevsrestclassifier__estimator__C':[1, 10, 100],
        'onevsrestclassifier__estimator__gamma':[0.01, 0.001],
        #
        # NN
        #
        'mlpclassifier__solver':['adam', 'sgd'],
        'mlpclassifier__hidden_layer_sizes':[10, 50, 100, 200],
        'mlpclassifier__alpha':[0.01, 0.001, 0.0001],        
        #
        # Random Forest
        #
        'randomforestclassifier__n_estimators':[10, 100],
        'randomforestclassifier__max_features':[1, 'auto', None],
        'randomforestclassifier__max_depth':[1, 5, 10, None],
        'randomforestclassifier__min_samples_leaf':[1, 2, 4,],        
        #
        # Naive Bayes
        #
        'bernoullinb__alpha':[0.0001, 0.1, 1.0],
        # 
        # XGBoost
        # 
        'xgbclassifier__max_depth':[3, 6, 9],
        'xgbclassifier__subsample':[0.5, 0.95, 1.0],
        'xgbclassifier__colsample_bytree':[0.5, 1]
    }

    
    #-------------------------
    # ハイパーパラメータ探査
    #-------------------------
    grid = GridSearchCV(estimator = sclf, param_grid = params
                        , cv = 5, verbose = 3,  n_jobs = -1)
    grid.fit(x_train_std, label_train)

    print("Best fit params")
    print(grid.best_params_)

    #--------------------
    # モデルを保存する
    #--------------------
    with open(pickle_name, mode = 'wb') as f:
        pickle.dump(grid, f)


    #
    # predict using test data
    #
    predict = clf.predict(x_test_std)

    conf_matrix = confusion_matrix(label_test, predict
                                   , labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9999, 99999])

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

    pp_df0.to_csv("conflevel_SMOTE100_add9999_stacking_0102.csv", index=False)

    #-------------------------------------
    # それぞれの学習器から予測を出す
    #-------------------------------------

    svm.fit(x_train_std, label_train)
    predict_svm = svm.predict(x_test_std)
    predict_df_svm = pd.DataFrame(predict_svm)
    print("type of predict_df_svm", type(predict_df_svm))
    
    nn.fit(x_train_std, label_train)   
    predict_nn = nn.predict(x_test_std)
    predict_df_nn = pd.DataFrame(predict_nn)
    print("type of predict_df_nn", type(predict_df_nn))
    
    rf.fit(x_train_std, label_train)    
    predict_rf = rf.predict(x_test_std)
    predict_df_rf = pd.DataFrame(predict_rf)
    print("type of predict_df_rf", type(predict_df_rf))
    
    bnb.fit(x_train_std, label_train)    
    predict_bnb = bnb.predict(x_test_std)
    predict_df_bnb = pd.DataFrame(predict_bnb)
    print("type of predict_df_bnb", type(predict_df_bnb))
    
    xgbc.fit(x_train_std, label_train)    
    predict_xgbc = xgbc.predict(x_test_std)
    predict_df_xgbc = pd.DataFrame(predict_xgbc)
    print("type of predict_df_xgbc", type(predict_df_xgbc))
    
    # concat
    predict_df = pd.concat([predict_df, predict_df_svm, predict_df_nn,
                            predict_df_rf, predict_df_bnb
                            , predict_df_xgbc], axis=1)
    predict_df.columns = ["voting", "svm", "nn", "rf", "bnb", "xgbc"]

    #
    # write out 
    #
    # predict_name = filename[0:27]+'_SMOTE_predict.csv'
    predict_df.to_csv(predict_name, index=False)

    
