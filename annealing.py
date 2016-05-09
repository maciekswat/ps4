import numpy as np
import matplotlib

matplotlib.use('Qt4Agg')
import numpy as np
from random import randint, random
import math
from collections import defaultdict

import pandas as pd

import cPickle as pickle


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Lattice(object):
    def __init__(self, dims=np.array([40, 40], dtype=np.int)):
        self.offsets = self.generate_neighbor_list(neighbor_order=2)
        self.dims = dims

    def set_neighbor_order(self, neighbor_order):
        self.offsets = self.generate_neighbor_list(neighbor_order=2)

    def get_dims(self):
        return self.dims

    def generate_neighbor_list(self, neighbor_order=2):

        order_1 = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        order_2 = order_1 + [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        order_3 = order_2 + [[0, 2], [0, -2], [2, 0], [-2, 0]]

        if neighbor_order == 1:
            return np.array(order_1, dtype=np.int)

        elif neighbor_order == 2:
            return np.array(order_2, dtype=np.int)
        elif neighbor_order == 3:
            return np.array(order_3, dtype=np.int)

        else:
            raise NotImplemented('Neighbor order > 2 not implemented')

    def pick_neighbor(self, x):
        while True:
            idx = randint(0, self.offsets.shape[0] - 1)

            offset = self.offsets[idx]

            neighbor = x + offset

            if neighbor[0] >= 0 and neighbor[0] <= self.dims[0] - 1 and neighbor[1] >= 0 and neighbor[1] <= self.dims[
                1] - 1:
                return neighbor


class Metropolis(object):
    def __init__(self, lattice=None, fcn_vals=None, T=10.0):
        self.T = T
        self.set_fcn_vals(fcn_vals)
        self.set_lattice(lattice)

        self.freq_measure_dict = defaultdict(lambda: 0)

        self.starting_loc = None

        self.check_starting_loc()

    def check_starting_loc(self):
        if self.lattice and self.starting_loc is None:
            self.starting_loc = np.squeeze(self.lattice.get_dims() / 2)

    def set_starting_loc(self, starting_loc):
        self.starting_loc = starting_loc

    def set_lattice(self, lattice):
        self.lattice = lattice

    def set_fcn_vals(self, fcn_vals):
        self.fcn_vals = fcn_vals

    def acceptance_function(self, delta, T):
        if delta >= 0:
            return 1.0
        else:
            return math.exp(delta / (1.0 * T))

    def run(self, steps=1000):

        self.check_starting_loc()
        if self.starting_loc is None:
            raise ValueError('self.starting-loc cannot be None')

        assert self.fcn_vals is not None, "self.fcn_vals - an array representing a function to be maximized cannot be None"

        x = self.starting_loc
        for i in xrange(steps):
            neighbor = lattice.pick_neighbor(x)

            self.freq_measure_dict[tuple(x)] += 1
            self.freq_measure_dict[tuple(neighbor)] += 1

            val = self.fcn_vals[tuple(x)]
            val_neighbor = self.fcn_vals[tuple(neighbor)]

            delta = val_neighbor - val

            prob_thresh = random()
            accept_prob = self.acceptance_function(delta, self.T)
            if prob_thresh < accept_prob:
                x = neighbor
                # print neighbor,self.fcn_vals[tuple(neighbor)]

        print self.freq_measure_dict


mu, sig = -2.0, 1.0

mu_noise, sig_noise = 0.0, 1.0

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
R = gaussian(X, mu, sig) * gaussian(Y, mu, sig) * 5

noise = np.random.normal(mu, sig, X.shape[0] * Y.shape[0])

noise = noise.reshape((X.shape[0], Y.shape[0]))

# Z = np.sin(R)
Z = R + noise

lattice = Lattice(dims=np.array([X.shape[0], Y.shape[0]], dtype=np.int))
lattice.generate_neighbor_list(neighbor_order=3)

s_cumulative = None

for i in range(1):
    metro = Metropolis(lattice=lattice, T=1.0)
    metro.set_fcn_vals(Z)

    # pick random staring point
    starting_loc = np.array([randint(0, X.shape[0] - 1), randint(0, Y.shape[0] - 1)], dtype=np.int)
    starting_loc = np.array([38,16])
    starting_loc = np.array([4,0])

    metro.set_starting_loc(starting_loc=starting_loc)


    metro.run(steps=500)

    fmd = metro.freq_measure_dict
    s = pd.Series(fmd, index=fmd.keys())

    s_sorted = s.sort_values(inplace=False, ascending=False)

    s_top = s_sorted.ix[range(10)]

    print s_top

    if s_cumulative is None:
        s_cumulative = s_top
    else:
        s_cumulative = pd.concat([s_cumulative, s_top])


        # for loc,count in s_top.iteritems():
        #     print 'loc=',loc,' val=',Z[tuple(loc)]

print s_cumulative

for loc, count in s_cumulative.iteritems():
    print 'loc=', loc, ' val=', Z[tuple(loc)]


class ValueExtractor(object):
    def __init__(self, fcn_vals):
        self.fcn_vals = fcn_vals

    def __call__(self, *args, **kwargs):
        loc = args[0]

        return self.fcn_vals[tuple(loc)]


v_extractor = ValueExtractor(fcn_vals=Z)

s_new = pd.DataFrame({'loc': s_cumulative.index, 'count': s_cumulative.values})

values = s_new['loc'].apply(v_extractor)

s_new['values'] = pd.Series(values, index=s_new.index)

print s_new

s_new_sorted = s_new.sort_values(by='values', ascending=False)

print '---------------AFTER SORTING---------------'
print 'STARTING LOC:', starting_loc
print s_new_sorted

out_dict = {
    'X': X,
    'Y': Y,
    'Z': Z,
    'mu': mu,
    'sig': sig,
    'mu_noise': mu_noise,
    'sig_noise': sig_noise,
    's_new_sorted': s_new_sorted,
    'starting_loc':starting_loc


}

s_new_sorted.to_csv('random_walker_frequency.txt')

pickle.dump(out_dict, open("random_walker_data.pkl", "wb"))

# print s_new

# x = np.array([10, 10], dtype=np.int)
# for i in range(100):
#     neighbor = lattice.pick_neighbor(x)
#     print neighbor
#     x = neighbor





# mu, sig = (-1, 1)
# # plt.plot(gaussian(np.linspace(-3, 3, 120), mu, sig))
# #
# #
# #
# # plt.get_current_fig_manager().window.raise_()
# #
# #
# # plt.show()
#
#
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X**2 + Y**2)
# R = gaussian(X, mu, sig) * gaussian(Y, mu, sig)*5
#
# noise = np.random.normal(mu, sig, X.shape[0] * Y.shape[0])
#
# noise = noise.reshape((X.shape[0], Y.shape[0]))
#
# # Z = np.sin(R)
# Z = R + noise
