import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(points, dists, plane_coeffs):
    """显示3D点云与平整度热图"""
    X, Y, Z = points[:,0], points[:,1], points[:,2]
    a,b,c = plane_coeffs

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=dists, cmap='turbo', s=10)
    xx = np.linspace(np.min(X), np.max(X), 30)
    yy = np.linspace(np.min(Y), np.max(Y), 30)
    xxm, yym = np.meshgrid(xx, yy)
    zzm = a*xxm + b*yym + c
    ax.plot_surface(xxm, yym, zzm, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='距离平面 (m)')
    plt.title('3D点云与拟合平面')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, c=dists, cmap='turbo', s=8)
    plt.colorbar(label='距离平面 (m)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.title('平整度热图')
    plt.show()
