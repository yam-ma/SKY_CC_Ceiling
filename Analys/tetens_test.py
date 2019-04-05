##
## test & plot tetens eq. (相対湿度と露点温度の換算)
##
##               参考サイト:
##               http://www.cuc.ac.jp/~nagaoka/2017/shori/03/Tetens/index.html
##
##               Mar. 19. 2019, M. Yamada
##

import numpy as np
import math

#
# テテンの式(Tkin -> Pressure)
#
def tetens(Tkin):

    a = 7.5
    b = 237.3

    c = a*Tkin/(b+Tkin)
    
    es = 6.11*10.0**c
    
    return es

#
# テテンの式逆算(Pressure -> Tdew)
#
def tetens_inv(es):

    a = 7.5
    b = 237.3

    c = np.log10(6.11) + a - np.log10(es) 
    
    Tdew = a*b/c - b
    
    return Tdew

if __name__ == "__main__":

    #
    # 温度(deg)→飽和水蒸気量(Pa)
    #
    Tkin = np.arange(40)
    print(Tkin)

    es  = tetens(Tkin)

    print(es)

    #
    # 飽和水蒸気量(Pa)->温度(deg)
    #
    Tdew = tetens_inv(es)

    print(Tdew)
