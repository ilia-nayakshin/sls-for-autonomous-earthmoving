import numpy as np



def test_cylinder(x, y, z):
    '''cylinder with equation x^2 + y^2 = R^2'''
    r2 = 200 **2
    yy, zz = np.meshgrid(y, z)
    xx = np.sqrt(r2 - 100*yy**2)
    return xx, yy, zz


def test_corrugations(x, y, z):
    '''corrugated sin-wave surface situated at the z=d plane'''
    d = 200
    xx, yy = np.meshgrid(x, y)
    zz = 30*np.sin(xx/4) + 10*np.sin(yy/4) + d
    return xx, yy, zz


def test_plane(d, n):
    '''returns a plane function with equation x.n = d.'''
    n = n / np.linalg.norm(n)
    def plane(x, y, z):
        xx, yy = np.meshgrid(x, y)
        zz = (d - n[0] * xx - n[1] * yy) / n[2]
        return xx, yy, zz
    return plane


def test_embankment_x(d, ddiff, slope):
    '''returns a function that models an embankment with given slope in x direction'''
    xdiff = ddiff / np.tan(slope)
    xmin = -xdiff / 2
    xmax = xdiff / 2
    dmin = d - ddiff/2
    dmax = d + ddiff/2
    def embankment(x, y, z):
        xx, yy = np.meshgrid(x, y)
        xismin = np.less_equal(x, xmin).astype(int)
        xismax = np.greater_equal(x, xmax).astype(int)
        zz = xismin * dmin + xismax * dmax + np.logical_not(xismin) * np.logical_not(xismax) * np.add((xx * np.tan(slope)), d)
        # if x <= xmin:
        #     zz = [[dmin]]
        # elif x >= xmax:
        #     zz = [[dmax]]
        # else:
        #     zz = [[d]] + xx * np.tan(slope)
        return xx, yy, zz
    return embankment


def test_embankment_y(d, ddiff, slope):
    '''returns a function that models an embankment with given slope in y direction'''
    ydiff = ddiff / np.tan(slope)
    ymin = -ydiff / 2
    ymax = ydiff / 2
    dmin = d - ddiff/2
    dmax = d + ddiff/2
    def embankment(x, y, z):
        xx, yy = np.meshgrid(x, y)
        if y <= ymin:
            zz = [[dmin]]
        elif y >= ymax:
            zz = [[dmax]]
        else:
            zz = [[d]] + xx * np.tan(slope)
        return xx, yy, zz
    return embankment