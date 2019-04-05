##
## AreaForecastのCIGの計算式中係数Aを可視化する
##
##           Mar. 20. 2019, M. Yamada
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

#========================
# read MSM_point data
#========================
def read_MSM_point(filename):
    
    File = open(filename, "rt")
    lines = File.readlines()[5:]

    lines2 = [[line] for line in lines[2:]]
    columns = [lines[0]][-1]

    df_msm = pd.DataFrame(lines2)
    df_msm.columns = ["csv"]

    tmp = df_msm['csv'].str.split(",", expand=True)

    tmp.columns = columns.split(",")
    df_msm = tmp.drop("\n", axis = 1) # 最後の列は改行なので削る

    return df_msm

#=====================
# 比例係数Aの計算
#=====================
def MSM_calc_A(df_msm):
    T950 = df_msm['Temp950'].map(lambda x: float(x))
    T850 = df_msm['Temp850'].map(lambda x: float(x))

    Tprof = -2.0*(T950 - T850 - 3.5)

    A = 0.5 + 1.2/(1.0+np.exp(Tprof))
    A = 125.5 * A

    return A


#===============================
# Tkin-Tdew項の計算
#===============================
def MSM_calc_Tdew(df_msm):
    
    # Tkin -> es
    df_msm['Temperature'] \
        = df_msm['Temperature'].map(lambda x: float(x)-273.15) # K->deg
    es = tetens(df_msm['Temperature'])

    # RH -> e
    df_msm['Relative humidity']  \
        = df_msm['Relative humidity'].map(lambda x: float(x))
    e = es*df_msm['Relative humidity']/100.0

    # e -> Tdew
    Tdew = tetens_inv(e)

    return Tdew 

#==========
# CIG計算
#==========
def MSM_calc_CIG(df_msm):

    df_msm['Low cloud cover']=df_msm['Low cloud cover'].map(lambda x:float(x))

    cig = []
    for i in range(len(df_msm)):
        # print(i, df_msm['A'].tolist()[i])
        if df_msm['Low cloud cover'].tolist()[i] < 0.6:
            c = 20000
        else:
            c = df_msm['A'].tolist()[i] *  \
                (df_msm['Temperature'].tolist()[i]-df_msm['Tdew_MSM'].tolist()[i])
            
        cig.append(c*3.281)   # m->ft

    return cig


#==================
# main
#==================
if __name__ == "__main__":

    #------------------------
    # read MSM_point data
    #------------------------
    filename = "/home/ai-corner/part1/SKY-DATA/MSM_point/2017_RJFK.csv"
    
    df_msm = read_MSM_point(filename)
    
    # 列名をValidityDate/Timeからdateに変更する
    df_msm = df_msm.rename(columns = {'ValidityDate/Time':'date'})

    #
    # 比例係数Aの計算
    #
    df_msm['A'] = MSM_calc_A(df_msm)
   
    #
    # Tkin-Tdew項の計算
    #
    df_msm['Tdew_MSM'] = MSM_calc_Tdew(df_msm)
    
    #
    # CIG計算
    #
    # A のnan除去
    df_msm = df_msm.dropna(subset=['A'])

    df_msm['CIG'] = MSM_calc_CIG(df_msm)

    #
    #
    #

    
    #-------------
    # plot
    #-------------

    #
    # 日付に直す
    #
    print(df_msm['date'].tolist())
    df_msm['date2'] = pd.to_datetime(df_msm['date'], format='%Y%m%d%H%M')

    df_msm = df_msm.set_index('date2')

    plt.plot(df_msm.index, df_msm['CIG'])
    plt.show()
