##
## cut_cig.py: v6: 晴れの日を晴れとしてカウントする新しい分類にする。
##                 0 < c < 200はデータがないが、一つのカテゴリとする。
##
##                 BKN/OVCに加えて、FEWとSCTも別カテゴリとして分類する。
##                 SCTとFEWが混在するときは、SCTを優先する。
##                 それに伴って、カテゴリも新しく2つ増やす。
##
##                 Mar. 27. 2019, M. Yamada
##


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import pickle

if __name__ == "__main__":

    airport = "RJFK"
    
    #
    # METARデータ
    #
    
    dir_root = "/home/yam-ma/Ceiling/"
    # years = ["2013", "2014", "2015", "2016", "2017"]
    years = ["2015", "2016", "2017"]
    # years = ["2015"] # for debug
    # filename = "RJFK.txt"
    filename = airport+".txt"
    
    for year in years:

        fname = dir_root+year+"/"+filename
        dat = pd.read_csv(fname)

        date = dat['date']

        # 欠損値の穴埋めをする(v5: 欠損：晴れ)
        dat2 = dat.fillna({"str_cloud":"-99999"})
        cloud = dat2['str_cloud']

        # ceiling格納
        cs = []

        for i in range(len(dat)):
            c = cloud[i]

            #
            # BKNかOVCがある場合
            #
            if "BKN" in c or "OVC" in c:
                lin = c.split(" ")
                # print(lin)
                # ind = lin.index("BKN")
                # BKNが複数ある場合があるので、全てのインデックスを取り出す
                indices = [j for j, x in enumerate(lin) if x == "BKN" or x == "OVC"]
                bkn = []
                for k in indices:
                    bkn.append(int(lin[k+1])*100) # 100ft単位
                cl = max(min(bkn), -9999)  
                # print(indices, bkn, cl)
                # cl = lin[ind+1]
                # print(ind, lin[ind], lin[ind+1], cl)
            #
            # BKNもOVCもなく、SCTがある場合(FEWがあっても良い)
            #
            elif "SCT" in c:
                cl = -77777
            #
            # BKNもOVCもSCTもなくFEWだけがある場合
            #
            elif "FEW" in c:
                cl = -88888
            #
            # 雲のsignatureがない場合
            #
            else:
                cl = -99999  # BKNがない場合

            cs.append(cl)

        # チェック用
        # print(len(date), len(cloud), len(cs))
        df = pd.DataFrame({"date":date, "str":cloud, "ceiling":cs})
        oname = "df_"+year+".csv"
        df.to_csv(oname, index=False)

        #
        # カテゴリ作成
        #
        cs_category = []
        for c in cs:
            if c ==  -99999:  # BKNもOVCもSCTもFEWも値がない場合
                cat = 99999
            elif c == -88888:  # BKNとOVCはなくSCTがある場合
                cat = 88888
            elif c == -77777:  # FEWだけある場合
                cat = 77777
            elif c < 0:     # BKNはあるが、計測不能(-9999が入っていた場合)
                cat = 9999     
            elif 0 <= c < 200:
                cat = 1
            elif 200 <= c < 500:
                cat = 2
            elif 500 <= c < 1000:
                cat = 3
            elif 1000 <= c < 2000:
                cat = 4
            elif 2000 <= c < 4000:
                cat = 5
            else:    # 4000ft以上
                cat = 6 

            cs_category.append(cat)
            # print(c, cat)

        df2 = pd.DataFrame({"date":date, "CIG":cs, "CIG_category":cs_category})
        oname = "cat_"+year+".csv"
        df2.to_csv(oname, index=False)
            
        #
        # MSM_pointデータ
        #
        root_dir = "/home/ai-corner/part1/SKY-DATA/MSM_point/"
        fname = root_dir+year+"_"+airport+".csv"

        File = open(fname, "rt")
        lines = File.readlines()[5:]
    
        lines2 = [[line] for line in lines[2:]]
        # print(lines2)
        print([lines[0]][-1])
        columns = [lines[0]][-1]

        df_msm = pd.DataFrame(lines2)
        df_msm.columns=["csv"]

        tmp = df_msm['csv'].str.split(",", expand=True)

        tmp.columns = columns.split(",")
        tmp = tmp.drop("\n", axis=1) # 最後の列は改行なので削る
        print(tmp)


        #
        # metarとMSM_pointを結合する
        #

        # metarデータにカテゴリを付随させる
        df_tmp = df2["CIG_category"]
        df_metar = pd.concat([dat2, df_tmp], axis=1)

        # file_nameは明らかに関係ないので落とす
        df_metar = df_metar.drop("file_name", axis=1)
        print(df_metar)

        # MSM_pointデータの日付フォーマットをmetarデータのそれに合わせる
        tmp['ValidityDate/Time'] = tmp['ValidityDate/Time']+"00"
        tmp = tmp.rename(columns={'ValidityDate/Time':"date"})
        tmp['date'] = tmp['date'].map(lambda x: int(x))
        print("tmp")
        print(tmp)

        # マージ
        df_all = pd.merge(tmp, df_metar, on="date")
        print("df_all")
        print(df_all)

        # NaNしかない列は落とす
        df_all = df_all.dropna(how='all', axis=1)

        #================#
        # 欠測値の処理
        #================#
        print("upper")

        # 補間の参照用にいったん欠損値を落としたデータフレームにする
        df_all = df_all.replace({'nan':pd.np.nan})
        df_nona = df_all.dropna(subset=['Geop1000'])

        print(df_nona[['date', 'Temp800']])
        
        # plt.plot(df_nona['date'], df_nona['Geop1000'], 'o')
        # plt.show()

        # 欠測値の入った日時を切り出す
        print("date_nan")
        print(df_all[df_all['Geop1000'].isnull()].date)
        date_nan = df_all[df_all['Geop1000'].isnull()].date

        
        print("floor")
        for d in date_nan:
            # 該当月を抜き出す
            month = math.floor(d/100000000)
            cond = (month<=df_nona['date']/100000000) \
                               & (df_nona['date']/100000000 < month+1)
            x_data = df_nona.loc[cond, 'date'].tolist()
            # print("ind",month, x_data)

            #
            # 列ごとにループをまわす
            #
            columns = df_all.columns[12:84]
            ## columns = df_all.columns[12:13]  # for debug
            for col in columns:
                            
                # 該当月のデータからupperのデータを抜き出す
                # y_data = df_nona.loc[cond, 'Geop1000'].tolist()
                y_data = df_nona.loc[cond, col].tolist()
                y_data = [float(y) for y in y_data]
            
                # print("xy", x_data, y_data)
            
                # 線形補間
                p = np.polyfit(x_data, y_data, 1)
                yy = np.polyval(p, d)
                ## p = interpolate.interp1d(x_data, y_data)
                ## yy = p(d)
                print(col, "yy=", yy)

                # df_all.loc[df_all['date'] == d, 'Geop1000'] = str(yy)
                df_all.loc[df_all['date'] == d, col] = str(yy)
                # ind = df_all.loc[df_all['date'] == d, 'Geop1000']
                # print(ind)


        # プロットして確認
        print("df_all")
        print(df_all['Geop1000'])
        # plt.plot(df_all['date'], df_all['Geop1000'], 'o')
        # plt.plot(df_nona['date'], df_nona['Geop1000'], 'ro')
        # plt.show()

        #
        # MSM残すデータの確認用
        #
        columns = df_all.columns
        print("columns")
        print(columns)
        print(columns[12:84])

        #
        # str_cloud列の削除
        #
        df_all = df_all.drop('str_cloud', axis=1)

        #
        # bulletin列の削除
        #
        if ('bulletin' in df_all.columns): 
            df_all = df_all.drop('bulletin', axis=1)

        #
        # dateで重複した行を落とす
        #
        df_all = df_all.drop_duplicates(subset='date')

        #
        # すべて欠測値の列を削除する
        #
        df_all = df_all.dropna(how='all')
        
        #
        # CIGカテゴリの欠測値行の削除(v5: 削除しない)
        #
        # df_all = df_all.replace({99999:pd.np.nan})
        # df_all = df_all.dropna(subset=['CIG_category'])
        # print(df_all)

        #
        # データ書き出し
        #
        pick_name = year+"_add9999_all.pickle"
        with open(pick_name, mode = 'wb') as f:
            pickle.dump(df_all, f)

        csv_name = year+"_add9999_all.csv"
        df_all.to_csv(csv_name, index=False)
