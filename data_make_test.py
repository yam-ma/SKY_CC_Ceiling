import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":

    with open('test_RJAA.pkl', 'rb') as fread:
        test_dat = pickle.load(fread)
        print(test_dat)


    print(type(test_dat))

    print(test_dat.columns)

    # 
