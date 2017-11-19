import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Model.FitnessFunction.exp_function as fitness
import Model.FitnessFunction.F101 as F101
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# 2D plot
x = np.linspace(-5.0, 5.0, 1000)
y = [fitness.func([e]) for e in x]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('fitness')

# 3D plot
fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
#
# x = y = np.linspace(-5.0, 5.0, 400)
# X, Y = np.meshgrid(x, y)
#
# cmap = matplotlib.cm.get_cmap("hot")
#
# zs = np.array([fitness.func([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)
# surf = ax1.plot_surface(X, Y, Z, cmap=cmap)

###
ax2 = fig.add_subplot(111, projection='3d')
x = y = np.linspace(-510.0, 512.0, 100)
X, Y = np.meshgrid(x, y)

cmap = matplotlib.cm.get_cmap("jet")

zs = np.array([F101.func([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
surf = ax2.plot_surface(X, Y, Z, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=False)
ax2.view_init(azim=0, elev=90)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(zs)
plt.colorbar(m)

plt.show()
plt.close()
