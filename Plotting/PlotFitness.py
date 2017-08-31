import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Model.FitnessFunction.ExpFitnessFunction as fitness
from mpl_toolkits.mplot3d import Axes3D

# 2D plot
x = np.linspace(-5.0, 5.0, 500)
y = [fitness.func([e]) for e in x]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('fitness')

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = y = np.linspace(-5.0, 5.0, 400)
X, Y = np.meshgrid(x, y)

cmap = matplotlib.cm.get_cmap("hot")

zs = np.array([fitness.func([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
surf = ax.plot_surface(X, Y, Z, cmap=cmap)
plt.show()
plt.close()
