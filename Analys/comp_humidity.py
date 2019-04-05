##
## Area ForecastのCeiling計算の傾向：湿度の依存性を見る
##                        湿度が高い→露点温度が高い(仕様では)CIGを低く見積もる
##                        比較はとりあえず2017年だけでする
##           
##                        Mar. 19. 2019, M. Yamada
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#===============================
# 水の飽和水蒸気圧を計算する
#===============================
def sat_water_vapor(Tkin):
    #
    # Tkin[deg], T[K]
    #
    T = Tkin + 273.15

    ew = -6096.9385/T + 21.2409642  \
         -2.711193/100*T  \
         +1.673952/100000*T**2  \
         +2.433502*math.log(T)            # ln(ew)

    return math.exp(ew)

#==========================================
# 水蒸気圧から露点温度を計算する
#==========================================
def T_dew(ew):

    y = math.log(ew/611.213)

    if y >= 0:
        Td = 13.715*y + 8.4262/10*y**2 \
             +1.9048/100*y**3 \
             +7.8258/1000*y**4
    else:
        Td = 13.7204*y + 7.36631/10*y**2 \
             +3.32136/100*y**3 \
             +7.78591/10000*y**4
        
    return Td


#================================
# テテンの式(Tkin -> Pressure)
#================================
def tetens(Tkin):

    a = 7.5
    b = 237.3

    c = a*Tkin/(b+Tkin)
    
    es = 6.11*10.0**c
    
    return es

#================================
# テテンの式逆算(Pressure -> Tdew)
#================================
def tetens_inv(es):

    a = 7.5
    b = 237.3

    c = np.log10(6.11) + a - np.log10(es) 
    
    Tdew = a*b/c - b
    
    return Tdew


#=========
# Main
#=========

if __name__ == "__main__":

    #------------------------
    # read MSM_point data
    #------------------------
    filename = "/home/ai-corner/part1/SKY-DATA/MSM_point/2017_RJFK.csv"
    
    File = open(filename, "rt")
    lines = File.readlines()[5:]

    lines2 = [[line] for line in lines[2:]]
    columns = [lines[0]][-1]

    df_msm = pd.DataFrame(lines2)
    df_msm.columns = ["csv"]

    tmp = df_msm['csv'].str.split(",", expand=True)

    tmp.columns = columns.split(",")
    df_msm = tmp.drop("\n", axis = 1) # 最後の列は改行なので削る

    print(df_msm['Relative humidity'])

    # 列名をValidityDate/Timeからdateに変更する
    df_msm = df_msm.rename(columns = {'ValidityDate/Time':'date'})

    #
    # MSMデータでTkinとRHからTdewを計算する
    #
    df_msm['Temperature']  \
        = df_msm['Temperature'].map(lambda x: float(x)-273.15) # K->deg
    es = tetens(df_msm['Temperature'])

    print("es=")
    print(es)

    # RH = e/es*100
    df_msm['Relative humidity']  \
        = df_msm['Relative humidity'].map(lambda x: float(x))
    e = es*df_msm['Relative humidity']/100.0

    Tdew = tetens_inv(e)
    print("Tdew=")
    print(Tdew)

    df_msm['Tdew_MSM'] = Tdew

    # 列名をTemperatureからTkin_MSMに変更する
    df_msm = df_msm.rename(columns = {'Temperature':'Tkin_MSM'})


    #------------------------
    # read metar data
    #------------------------
    filename = "../2017/RJFK.txt"
    df_metar = pd.read_csv(filename)


    # Tkinを計算する
    Tkin = []
    for i in range(len(df_metar)):
        if df_metar['temp_sign'][i] < 0:
            Tk = df_metar['temp'][i]*(-1)
        else:
            Tk = df_metar['temp'][i]

        Tkin.append(Tk)

    # Tdewを計算する
    Tdew = []
    for i in range(len(df_metar)):
        if df_metar['dew_point_sign'][i] < 0:
            Td = df_metar['dew_point'][i]*(-1)
        else:
            Td = df_metar['dew_point'][i]

        Tdew.append(Td)


    print("Tdew=")
    print(Tdew)

    # add columns
    df_metar['Tkin_metar'] = Tkin
    df_metar['Tdew_metar'] = Tdew

    print(df_metar)
            

    #----------------------
    # metarとMSMの比較
    #----------------------

    #
    # merge dataframes
    #

    # 日付をintに直す
    df_msm['date'] = df_msm['date'].map(lambda x: int(x)*100)
    
    df = pd.merge(df_metar, df_msm, on = 'date', how = 'inner')
    
    # 必要な列をピックアップする
    df = df[['date', 'Tkin_metar', 'Tdew_metar', 'Tkin_MSM', 'Tdew_MSM']]

    df['diff_Tkin'] = df['Tkin_metar'] - df['Tkin_MSM']
    df['diff_Tdew'] = df['Tdew_metar'] - df['Tdew_MSM']
    df['diff'] = df['diff_Tkin'] - df['diff_Tdew']
    print(df)

    #
    # plot histogram
    #
    plt.hist(df['diff'], bins=20)
    plt.xlabel('(Tkin-Tdew)(metar)-(Tkin-Tdew)(MSM)')
    plt.ylabel('Frequency')
    plt.show()
