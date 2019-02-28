import pandas as pd
import numpy as np


if __name__ == "__main__":

    dir_root = "/home/yam-ma/Ceiling/"
    years = ["2013", "2014", "2015", "2016", "2017"]
    filename = "RJFK.txt"
    
    for year in years:

        fname = dir_root+year+"/"+filename
        print(fname)

        dat = pd.read_csv(fname)

        date = dat['date']

        # 欠損値の穴埋めをする
        dat2 = dat.fillna({"str_cloud":"-99999"})
        cloud = dat2['str_cloud']

        # ceiling格納
        cs = []

        for i in range(len(dat)):
            c = cloud[i]

            if "BKN" in c or "OVC" in c:
                lin = c.split(" ")
                print(lin)
                # ind = lin.index("BKN")
                # BKNが複数ある場合があるので、全てのインデックスを取り出す
                indices = [j for j, x in enumerate(lin) if x == "BKN" or x == "OVC"]
                bkn = []
                for k in indices:
                    bkn.append(int(lin[k+1]))
                cl = max(min(bkn), -9999)
                print(indices, bkn, cl)
                # cl = lin[ind+1]
                # print(ind, lin[ind], lin[ind+1], cl)
            else:
                cl = "-99999"  # BKNがない場合

            cs.append(cl)

        # チェック用
        print(len(date), len(cloud), len(cs))
        df = pd.DataFrame({"date":date, "str":cloud, "ceiling":cs})
        oname = "df_"+year+".csv"
        df.to_csv(oname, index=False)
