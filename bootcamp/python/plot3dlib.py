from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def plot3d_points(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot3d(xs, ys, f):
    zs = np.zeros((len(ys),len(xs)))
    for i in range(len(xs)):
        for j in range(len(ys)):
            zs[j,i] = f(xs[i], ys[j])
    plot3d_points(xs, ys, zs)

def demo():
    xs = np.arange(-2,2,0.1)
    ys = np.arange(-2,2,0.05)
    plot3d(xs, ys, lambda x,y: np.sin(x*y))
