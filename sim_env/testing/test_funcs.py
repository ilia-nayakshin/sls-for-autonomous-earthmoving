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