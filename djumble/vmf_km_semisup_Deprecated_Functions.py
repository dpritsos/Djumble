def Jd(self, d, x):
    """ Naive Bessel function approximation of the first kind.

        TESTING purpose only!

    """

    t = 0.9
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
