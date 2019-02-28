# SKY_CC_Ceiling

## Codes

cut_cig.py-> metarデータとMSM pointデータから、Ceilingの値を切り出す。
　　　　　　　　アウトプットは
             ＊df_(year).csv: metarデータからのCeilingフィールドの切り出し。確認用
             ＊(year)_all.pickle, (year)_all.csv: 計算に使うデータを作るコード(train_test_make.py)に入れる
             
train_test_make.py-> 2015&2016年のデータを訓練用に、2017年のデータをテスト用にpklファイルにする。
　　　　　　　　　　　　　アウトプット：
                      ＊CIG_RJFK_train.csv, CIG_RJFK_test.csv: 計算に使うデータ
                      ＊CIG_RJFK_train.pkl, CIG_RJFK_test.pkl: 上のフォーマット違い版。特に使わない。
                      
cig_mix.py-> ベースライン計算本体。
　　　　　　　　インプット：CIG_RJFK_train.csv, CIG_RJFK_test.csv（これらを変えても計算できる）
          　　基本的には、全てのデータを使って計算することを念頭に置いている。
              アウトプット：
            

make_2month.py-> 2ヶ月ごとのデータセットを作る。訓練用データもテストデータも2ヶ月ごとに分割する。
　　
