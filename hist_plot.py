import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    filename = "Input/CIG_RJFK_train_rm9999.csv"
    dat = pd.read_csv(filename)

    plt.hist(dat['CIG_category'])
    plt.show()


    filename = "Input/CIG_RJFK_1112_mean_rm9999_train.csv"
    dat = pd.read_csv(filename)

    plt.hist(dat['CIG_category'])
    plt.show()
    
    filename = "Intermediate/cat_2017.csv"
    dat = pd.read_csv(filename)
    print(dat)
