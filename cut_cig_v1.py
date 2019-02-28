import pandas as pd
import numpy as np


if __name__ == "__main__":

    #
    # METARデータ
    #
    
    dir_root = "/home/yam-ma/Ceiling/"
    years = ["2013", "2014", "2015", "2016", "2017"]
    filename = "RJFK.txt"
    
    for year in years:

        fname = dir_root+year+"/"+filename
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
            else:
                cl = -99999  # BKNがない場合

            cs.append(cl)

        # チェック用
        # print(len(date), len(cloud), len(cs))
        df = pd.DataFrame({"date":date, "str":cloud, "ceiling":cs})
        oname = "df_"+year+".csv"
        df.to_csv(oname, index=False)

        # カテゴリ作成
        cs_category = []
        for c in cs:
            if c < 0:
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
            else:
                cat = 9999

            cs_category.append(cat)
            # print(c, cat)

        df2 = pd.DataFrame({"date":date, "CIG":cs, "CIG_category":cs_category})
        oname = "cat_"+year+".csv"
        df2.to_csv(oname, index=False)
            
    #
    # MSM_pointデータ
    #
    year = "2015"
    root_dir = "/home/ai-corner/part1/SKY-DATA/MSM_point/"
    fname = root_dir+year+"_RJFK.csv"

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
    # 
    #
    
