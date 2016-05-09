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


s_new_sorted = out_dict['s_new_sorted']
print s_new_sorted

loc_array = np.array(s_new_sorted['loc'].values,dtype=np.dtype('int,int'))

X = out_dict['X']
Y = out_dict['Y']
Z = out_dict['Z']

X_visit = loc_array['f0']*0.25-5.0
Y_visit = loc_array['f1']*0.25-5.0

Z_visit = s_new_sorted['count'].values

print

fig = plt.figure()
plt.title('Random Walker Visits')
ax = fig.gca(projection='3d')




starting_loc = out_dict['starting_loc']

print starting_loc


dx = 0.25 * np.ones_like(Z_visit)
dy = 0.25 * np.ones_like(Z_visit)
dz = Z_visit
x = X_visit
y = Y_visit
z =np.zeros_like(Z_visit)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.2)

print x
print y


bar3d = ax.bar3d(x=x, y=y, z=z, dx=dx, dy=dy,dz=dz,color='r')



# scatter = ax.scatter(x, y, Z_visit, zdir='z', s=40, c='r', depthshade=True)

scatter = ax.scatter(starting_loc[0]*0.25-5, starting_loc[1]*0.25-5, 90, zdir='z', s=40, c='r', depthshade=True)


starting_loc = out_dict['starting_loc']
# ax.set_zlim(-1.01, 1.01)

# labels = [item.get_text() for item in ax.xaxis.get_ticklabels()]
labels =  ax.get_xticklabels()

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlabel('Random Walker Visits')

ax.set_xlabel('Param 1')
ax.set_ylabel('Param 2')

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.get_current_fig_manager().window.raise_()

plt.show()
#
