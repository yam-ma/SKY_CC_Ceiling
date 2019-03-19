#
# metarデータの季節変動を調べる
#
#                      Mar. 18. 2019, M. Yamada
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == "__main__":

    #
    # metarデータ読み込み
    # 

    filename = "../CIG_RJFK_train_add9999.csv"
    df = pd.read_csv(filename)


    #
    # 日付を文字列に
    #
    df['date'] = df['date'].map(lambda x: str(x))
    
    # print(df['date'].str[4:6])

    dat0102 = df[
        (df['date'].str[4:6] == "01" ) |
        (df['date'].str[4:6] == "02" )
    ]

    dat0304 = df[
        (df['date'].str[4:6] == "03" ) |
        (df['date'].str[4:6] == "04" )
    ]

    dat0506 = df[
        (df['date'].str[4:6] == "05" ) |
        (df['date'].str[4:6] == "06" )
    ]

    dat0708 = df[
        (df['date'].str[4:6] == "07" ) |
        (df['date'].str[4:6] == "08" )
    ]

    dat0910 = df[
        (df['date'].str[4:6] == "09" ) |
        (df['date'].str[4:6] == "10" )
    ]

    dat1112 = df[
        (df['date'].str[4:6] == "11" ) |
        (df['date'].str[4:6] == "12" )
    ]
    

    #
    # 晴れの日(99999)の割合の変化を調べる
    #
    hare = []
    
    print(dat0102['CIG_category'].value_counts(normalize=True))
    a = dat0102['CIG_category'].value_counts(normalize=True)
    print(type(a))
    print(a[99999])
    hare.append(a[99999])
    
    a = dat0304['CIG_category'].value_counts(normalize=True)
    hare.append(a[99999])
    a = dat0506['CIG_category'].value_counts(normalize=True)
    hare.append(a[99999])
    a = dat0708['CIG_category'].value_counts(normalize=True)
    hare.append(a[99999])
    a = dat0910['CIG_category'].value_counts(normalize=True)
    hare.append(a[99999])
    a = dat1112['CIG_category'].value_counts(normalize=True)
    hare.append(a[99999])

    indices = ["0102", "0304", "0506", "0708", "0910", "1112"]

    # plot
    # plt.bar(indices, hare)
    # plt.title("Fraction of Category 99999")
    # plt.xlabel("Month")
    # plt.ylabel("%")
    # plt.show()


    #
    # Relative humidity の変化を調べる
    #
    humidity = []
    a = dat0102['Relative humidity'].mean()
    humidity.append(a)
    a = dat0304['Relative humidity'].mean()
    humidity.append(a)
    a = dat0506['Relative humidity'].mean()
    humidity.append(a)
    a = dat0708['Relative humidity'].mean()
    humidity.append(a)
    a = dat0910['Relative humidity'].mean()
    humidity.append(a)
    a = dat1112['Relative humidity'].mean()
    humidity.append(a)

    print(a)

    # plot

    plt.plot(indices, humidity)
    plt.title("Relative humidity")
    plt.xlabel("Month")
    plt.ylim(0, 100)
    plt.show()
    
    
