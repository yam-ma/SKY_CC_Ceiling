import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    rand = np.random.rand(1000000, 6)

    print(rand[1])
    print(rand.shape)
    
    #
    # 割合で表して、上位二位までの差を取る
    #
    frac = [rand[i]/np.sum(rand[i]) for i in range(rand.shape[0])]
    # frac = rand
    print(frac[1])

    
    diff2 = []
    for i in range(rand.shape[0]-1):
        fr = frac[i]
        print(i, fr)
        fr_sorted = sorted(fr, reverse=True)
        diff_2 = fr_sorted[0]-fr_sorted[1]
        diff2.append(diff_2)

    #
    # ヒストグラム
    #
    plt.hist(diff2, bins=10)
    plt.show()
