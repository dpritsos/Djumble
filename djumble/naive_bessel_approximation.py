
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def Idx(d, x):

    t = 0.01
    conv = False

    R = 1.0
    t1 = np.power((x*np.exp(1.0))/(2.0*d), d)
    t2 = 1 + (1.0/(12.0*d)) + (1/(288*np.power(d, 2.0))) - (139.0/(51840.0*np.power(d, 3.0)))
    t1 = t1*np.sqrt((d/(2.0*np.pi))/t2)
    M = 1.0/d
    k = 1.0

    while not conv:

        R = R*((0.25*np.power(x, 2.0))/(k*(d+k)))
        M = M + R

        if R/M < t:
            conv = True

        k += 1

    return t1*M

if __name__=='__main__':

    # print Idx(1000, 500, 0.1)
    d = 1000.0
    k = 1000.0
    kd = np.power(k, ((d/2.0)-1.0))
    pid = np.power(2.0*np.pi, d/2.0)
    #
    Idk = sp.jv((d/2.0)-1.0, k)
    # Idkx = Idx((d/2.0)-1.0, k)
    cdk = kd / (pid*Idk)

    print 'kd', kd
    print 'pid', pid
    print 'kd/pid', kd/pid
    print 'Idk', Idk
    # print 'Idkx', Idkx
    # print 'Cdk', cdk
    # print 'log(Cdk)*1500', np.log(1/cdk)*1500
    print 'log(Idk)*1500', np.log(np.abs(1/Idk))*1500
    # print 'log(Cdk)', np.log(1/cdk)
    print 'log(Idk)', np.log(Idk)
    # print 'log(Idkx)', np.log(Idkx)*1500
    # a = sp.jv(3, range(0, 1000))
    # plt.plot(range(0, 1000000), sp.jv(100, range(0, 1000000)))
    # plt.plot(range(0, 1000000), sp.jv(1000, range(0, 1000000)))
    # plt.plot(range(0, 1000000), sp.jv(10000, range(0, 1000000)))
    # plt.plot(range(0, 1000000), sp.jv(100000, range(0, 1000000)))
    # plt.show()
    # print a[-1]
