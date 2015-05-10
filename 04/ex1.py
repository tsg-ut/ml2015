import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

theta = np.linspace(-np.pi, np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
z = 2*x+3*y

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x,y,z)

ax.scatter(2/np.sqrt(13),3/np.sqrt(13),np.sqrt(13))
ax.scatter(-2/np.sqrt(13),-3/np.sqrt(13),-np.sqrt(13))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
