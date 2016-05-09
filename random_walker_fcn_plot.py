import cPickle as pickle
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
from random import randint, random
import math
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd



# out_dict = pickle.load(open('random_walker_data_4_0_10_top.pkl','rb'))
# out_dict = pickle.load(open('random_walker_data_miss.pkl','rb'))
out_dict = pickle.load(open('random_walker_data_ok.pkl','rb'))



mu, sig = (out_dict['mu'], out_dict['sig'])

mu_noise, sig_noise = (out_dict['mu_noise'], out_dict['sig_noise'])

fig = plt.figure()
plt.title('Recall Rate Change')
ax = fig.gca(projection='3d')
X = out_dict['X']
Y = out_dict['Y']
Z = out_dict['Z']

starting_loc = out_dict['starting_loc']

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

scatter = ax.scatter(starting_loc[0]*0.25-5, starting_loc[1]*0.25-5, 2, zdir='z', s=40, c='r', depthshade=True)


starting_loc = out_dict['starting_loc']
# ax.set_zlim(-1.01, 1.01)

# labels = [item.get_text() for item in ax.xaxis.get_ticklabels()]
labels =  ax.get_xticklabels()

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlabel('Recall Rate Change')

ax.set_xlabel('Param 1')
ax.set_ylabel('Param 2')

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.get_current_fig_manager().window.raise_()

plt.show()

