##
## metar_after_histplot.py: 新旧の分類で分けたmetar/AFTERデータのクラスごと
##                          の頻度分布を描く
##
##                          Mar.13.2019, M. Yamada


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":

    #-----------------------------------
    # v1: metarとAFTER共通の分類
    #-----------------------------------

    filename = "metar_after.csv"
    dat = pd.read_csv(filename)


    #
    # count
    #
    from pandas import Series
    s = Series(dat['CIG_metar'])
    vc = s.value_counts()
    vc = vc.sort_index()
    print(vc)

    s2 = Series(dat['CIG_after'])
    vc2 = s2.value_counts()
    vc2 = vc2.sort_index()
    print(vc2)

    vc.plot(kind='bar')
    plt.savefig('CIG_metar_hist_v1.png')
    vc2.plot(kind='bar')
    plt.savefig('CIG_after_hist_v1.png')

    #-----------------------------------
    # v2: metarとAFTERそれぞれの分類
    #-----------------------------------

    filename = "metar_after_v2.csv"
    dat = pd.read_csv(filename)


    #
    # count
    #
    from pandas import Series
    s = Series(dat['CIG_metar'])
    vc = s.value_counts()
    vc = vc.sort_index()
    print(vc)

    s2 = Series(dat['CIG_after'])
    vc2 = s2.value_counts()
    vc2 = vc2.sort_index()
    print(vc2)

    vc.plot(kind='bar')
    plt.savefig('CIG_metar_hist_v2.png')
    vc2.plot(kind='bar')
    plt.savefig('CIG_after_hist_v2.png')
    
