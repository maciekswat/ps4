import numpy as np
import matplotlib

matplotlib.use('Qt4Agg')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



# plt.plot(gaussian(np.linspace(-3, 3, 120), mu, sig))
#
#
#
# plt.get_current_fig_manager().window.raise_()
#
#
# plt.show()



mu, sig = (-1, 1)

mu_noise, sig_noise = (0, 1)

fig = plt.figure()
plt.title('Recall Rate Change')
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
R = gaussian(X, mu, sig) * gaussian(Y, mu, sig)*5

noise = np.random.normal(mu_noise, sig_noise, X.shape[0] * Y.shape[0])

noise = noise.reshape((X.shape[0], Y.shape[0]))

# Z = np.sin(R)
Z = R + noise
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

# labels = [item.get_text() for item in ax.xaxis.get_ticklabels()]
labels =  ax.get_xticklabels()

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlabel('Recall Rate Change')

ax.set_xlabel('Param 1')
ax.set_ylabel('Param 2')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.get_current_fig_manager().window.raise_()

plt.show()




# mu, sig = (-5, 2)
#
# mu_noise , sig_noise = (0,1)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-20, 20, 1.0)
# Y = np.arange(-20, 20, 1.0)
# X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X**2 + Y**2)
# R = gaussian(X, mu, sig) * gaussian(Y, mu, sig)*15
#
# noise = np.random.normal(mu, sig, X.shape[0] * Y.shape[0])
#
# noise = noise.reshape((X.shape[0], Y.shape[0]))
#
# # Z = np.sin(R)
# Z = R + noise
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # ax.set_zlim(-1.01, 1.01)
#
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.get_current_fig_manager().window.raise_()
#
# plt.show()
#
#
#
