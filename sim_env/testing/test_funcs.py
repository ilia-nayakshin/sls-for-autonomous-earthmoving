import numpy as np



def test_cylinder(x, y, z):
    '''cylinder with equation x^2 + y^2 = R^2'''
    r2 = 200 **2
    yy, zz = np.meshgrid(y, z)
    xx = np.sqrt(r2 - 100*yy**2)
    return xx, yy, zz


def test_corrugations(d, t_x, t_y, s_x, s_y):
    '''corrugated sin-wave surface situated at the z=d plane'''
    def surface(x, y, z):
        xx, yy = np.meshgrid(x, y)
        zz = s_x*np.sin(xx/t_x) + s_y*np.sin(yy/t_y) + d
        return xx, yy, zz   
    return surface


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
        if type(x) != np.float64:
            xismin = np.less_equal(x, xmin).astype(int)
            xismax = np.greater_equal(x, xmax).astype(int)
            zz = xismin * dmin + xismax * dmax + np.logical_not(xismin) * np.logical_not(xismax) * np.add((xx * np.tan(slope)), d)
        else:
            if x <= xmin:
                zz = [[dmin]]
            elif x >= xmax:
                zz = [[dmax]]
            else:
                zz = [[d]] + xx * np.tan(slope)
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


def test_trench_x(d, ddiff, width, slope):
    '''returns a function that models a trench with given slope in the x direction'''
    xdiff = ddiff / np.tan(slope)
    xmin = width / 2
    xmax = xmin + xdiff
    dmin = d - ddiff/2
    dmax = d + ddiff/2
    def embankment(x, y, z):
        xx, yy = np.meshgrid(x, y)
        if type(x) != np.float64:
            xismin = np.less_equal(np.abs(x), xmin).astype(int)
            xismax = np.greater_equal(np.abs(x), xmax).astype(int)
            zz = xismin * dmin + xismax * dmax + np.logical_not(xismin) * np.logical_not(xismax) * np.add(-np.add(np.abs(xx), - xmin) * np.tan(slope), dmax)
        else:
            if np.abs(x) <= xmin:
                zz = [[dmax]]
            elif np.abs(x) >= xmax:
                zz = [[dmin]]
            else:
                zz = [[dmax]] - ( np.abs(xx) - np.array([[xmin]]) ) * np.tan(slope)
        return xx, yy, zz
    return embankment

def test_2d_gaussian(d, scale, sigma_x, sigma_y=None, rho=0):
    '''2d gaussian function. If sigma_y is not set will use sigma_y = sigma_x.'''
    if sigma_y is None:
        sigma_y = sigma_x
    def gaussian(x, y, z):
        xx, yy = np.meshgrid(x, y)
        zz = d - scale * np.exp(-0.5*((xx / sigma_x)**2 + (yy / sigma_y)**2 - 2*rho*(xx / sigma_x)*(yy / sigma_y)))
        return xx, yy, zz
    return gaussian
